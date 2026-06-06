from __future__ import annotations

import argparse
import csv
import json
import random
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from torch import nn

from muon_project.data import build_loaders
from muon_project.models import build_model
from muon_project.optim import Muon, OptimizerBundle, parse_coeffs, split_muon_parameters


METRIC_FIELDS = [
    "run_id",
    "dataset",
    "model",
    "optimizer",
    "lr",
    "weight_decay",
    "seed",
    "epoch",
    "train_loss",
    "val_loss",
    "val_acc",
    "epoch_time",
    "peak_memory",
    "ns_steps",
    "ns_coeffs",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_optimizer(args, model: nn.Module):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay), []

    coeffs = parse_coeffs(args.ns_coeffs)
    muon_params, aux_params, muon_names = split_muon_parameters(model)
    if not muon_params:
        raise ValueError("No eligible matrix parameters found for Muon.")

    muon = Muon(
        muon_params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        ns_steps=args.ns_steps,
        ns_coeffs=coeffs,
    )
    aux = None
    if aux_params:
        aux = torch.optim.AdamW(
            aux_params,
            lr=args.aux_lr,
            weight_decay=args.weight_decay,
        )
    return OptimizerBundle(muon, aux), muon_names


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)
    return total_loss / max(1, total_count)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_count += x.size(0)
    return total_loss / max(1, total_count), total_correct / max(1, total_count)


def append_metrics(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_seed(args, seed: int) -> None:
    set_seed(seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, val_loader, info = build_loaders(
        args.dataset,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        seed=seed,
        subset_train=args.subset_train,
        subset_val=args.subset_val,
        num_workers=args.num_workers,
    )
    model = build_model(args.model, info.input_shape, info.num_classes, use_bias=args.use_bias).to(device)
    optimizer, muon_names = make_optimizer(args, model)
    criterion = nn.CrossEntropyLoss()
    run_id = args.run_id or uuid.uuid4().hex[:10]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / f"config_{run_id}_seed{seed}.json"
    config = vars(args).copy()
    config.update({"seed": seed, "device_resolved": str(device), "muon_parameter_names": muon_names})
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    rows = []
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        epoch_time = time.perf_counter() - start
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
        row = {
            "run_id": run_id,
            "dataset": args.dataset,
            "model": args.model,
            "optimizer": args.optimizer,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time": epoch_time,
            "peak_memory": peak_memory,
            "ns_steps": args.ns_steps if args.optimizer == "muon" else "",
            "ns_coeffs": args.ns_coeffs if args.optimizer == "muon" else "",
        }
        rows.append(row)
        print(
            f"seed={seed} epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={epoch_time:.2f}s"
        )

    append_metrics(output_dir / "metrics.csv", rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train AdamW or Muon on small image benchmarks.")
    parser.add_argument("--dataset", choices=["fake", "mnist", "fashion_mnist"], default="mnist")
    parser.add_argument("--model", choices=["mlp", "small_cnn"], default="mlp")
    parser.add_argument("--optimizer", choices=["adamw", "muon"], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--aux-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--ns-steps", type=int, default=3)
    parser.add_argument("--ns-coeffs", default="muon", help="'muon', 'standard', or 'a,b,c'.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--subset-train", type=int, default=None)
    parser.add_argument("--subset-val", type=int, default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-bias", action="store_true")
    parser.add_argument("--run-id", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for seed in args.seeds:
        run_seed(args, seed)


if __name__ == "__main__":
    main()
