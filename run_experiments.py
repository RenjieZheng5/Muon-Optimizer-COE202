from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ADAMW_LRS = [1e-4, 3e-4, 1e-3, 3e-3]
MUON_LRS = [1e-3, 3e-3, 1e-2, 3e-2]
WEIGHT_DECAYS = [0.0, 1e-4, 1e-2]


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def base_cmd(args, optimizer: str, lr: float, weight_decay: float, seeds: list[int]) -> list[str]:
    return [
        sys.executable,
        "train.py",
        "--dataset",
        args.dataset,
        "--model",
        args.model,
        "--optimizer",
        optimizer,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(lr),
        "--weight-decay",
        str(weight_decay),
        "--seeds",
        *[str(seed) for seed in seeds],
        "--output-dir",
        str(args.output_dir),
        "--data-dir",
        args.data_dir,
    ]


def add_subset_flags(cmd: list[str], args) -> list[str]:
    if args.subset_train:
        cmd += ["--subset-train", str(args.subset_train)]
    if args.subset_val:
        cmd += ["--subset-val", str(args.subset_val)]
    return cmd


def run_minimal(args) -> None:
    args.dataset = "mnist"
    args.model = "mlp"
    seeds = [0, 1, 2]
    for optimizer, lr in [("adamw", 1e-3), ("muon", 1e-2)]:
        cmd = base_cmd(args, optimizer, lr, 1e-4, seeds)
        if optimizer == "muon":
            cmd += ["--ns-steps", "3", "--ns-coeffs", "muon"]
        run(add_subset_flags(cmd, args))


def run_main(args) -> None:
    args.dataset = "fashion_mnist"
    args.model = args.model or "mlp"
    seeds = [0, 1, 2, 3, 4]
    for optimizer, lr in [("adamw", 1e-3), ("muon", 1e-2)]:
        cmd = base_cmd(args, optimizer, lr, 1e-4, seeds)
        if optimizer == "muon":
            cmd += ["--ns-steps", "3", "--ns-coeffs", "muon"]
        run(add_subset_flags(cmd, args))


def run_ablation(args) -> None:
    args.dataset = "fashion_mnist"
    args.model = args.model or "mlp"
    seeds = [0, 1, 2, 3, 4]
    for coeffs in ["muon", "standard"]:
        for q in [1, 2, 3, 5]:
            cmd = base_cmd(args, "muon", args.muon_lr, args.weight_decay, seeds)
            cmd += ["--ns-steps", str(q), "--ns-coeffs", coeffs]
            run(add_subset_flags(cmd, args))


def run_search(args) -> None:
    seeds = [0]
    for lr in ADAMW_LRS:
        for wd in WEIGHT_DECAYS:
            run(add_subset_flags(base_cmd(args, "adamw", lr, wd, seeds), args))
    for lr in MUON_LRS:
        for wd in WEIGHT_DECAYS:
            cmd = base_cmd(args, "muon", lr, wd, seeds)
            cmd += ["--ns-steps", "3", "--ns-coeffs", "muon"]
            run(add_subset_flags(cmd, args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run predefined experiment suites.")
    parser.add_argument("--mode", choices=["minimal", "main", "ablation", "search"], required=True)
    parser.add_argument("--dataset", choices=["fake", "mnist", "fashion_mnist"], default="fashion_mnist")
    parser.add_argument("--model", choices=["mlp", "small_cnn"], default="mlp")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--subset-train", type=int, default=None)
    parser.add_argument("--subset-val", type=int, default=None)
    parser.add_argument("--muon-lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "minimal":
        run_minimal(args)
    elif args.mode == "main":
        run_main(args)
    elif args.mode == "ablation":
        run_ablation(args)
    elif args.mode == "search":
        run_search(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
