from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def label_columns(df: pd.DataFrame) -> pd.Series:
    labels = df["optimizer"].astype(str)
    muon_mask = df["optimizer"] == "muon"
    labels.loc[muon_mask] = (
        "muon-q"
        + df.loc[muon_mask, "ns_steps"].astype(str)
        + "-"
        + df.loc[muon_mask, "ns_coeffs"].astype(str)
    )
    return labels


def summarize(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    last = df.sort_values("epoch").groupby("run_id", as_index=False).tail(1)
    group_cols = ["dataset", "model", "optimizer", "lr", "weight_decay", "ns_steps", "ns_coeffs"]
    summary = (
        last.groupby(group_cols, dropna=False)
        .agg(
            final_val_acc_mean=("val_acc", "mean"),
            final_val_acc_std=("val_acc", "std"),
            final_val_loss_mean=("val_loss", "mean"),
            final_train_loss_mean=("train_loss", "mean"),
            epoch_time_mean=("epoch_time", "mean"),
            peak_memory_mean=("peak_memory", "mean"),
            seeds=("seed", "nunique"),
        )
        .reset_index()
        .sort_values(["dataset", "model", "optimizer", "final_val_acc_mean"], ascending=[True, True, True, False])
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    return summary


def plot_curve(df: pd.DataFrame, y: str, filename: str, ylabel: str, output_dir: Path) -> None:
    df = df.copy()
    df["label"] = label_columns(df)
    curve = (
        df.groupby(["dataset", "model", "label", "epoch"], as_index=False)[y]
        .mean()
        .sort_values("epoch")
    )
    plt.figure(figsize=(9, 5))
    for label, group in curve.groupby("label"):
        plt.plot(group["epoch"], group[y], marker="o", linewidth=1.8, label=label)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=180)
    plt.close()


def time_to_threshold(df: pd.DataFrame, output_dir: Path, threshold: float | None) -> None:
    if threshold is None:
        threshold = float(df["train_loss"].quantile(0.25))
    rows = []
    for run_id, group in df.sort_values("epoch").groupby("run_id"):
        group = group.copy()
        group["elapsed_time"] = group["epoch_time"].cumsum()
        reached = group[group["train_loss"] <= threshold]
        row = group.iloc[-1].to_dict()
        row["threshold"] = threshold
        row["time_to_threshold"] = reached["elapsed_time"].iloc[0] if not reached.empty else None
        row["epoch_to_threshold"] = int(reached["epoch"].iloc[0]) if not reached.empty else None
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "time_to_threshold.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Muon project metrics.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.results)
    summarize(df, output_dir)
    plot_curve(df, "train_loss", "loss_curve.png", "Training loss", output_dir)
    plot_curve(df, "val_acc", "accuracy_curve.png", "Validation accuracy", output_dir)
    time_to_threshold(df, output_dir, args.threshold)
    print(f"Wrote analysis artifacts to {output_dir}")


if __name__ == "__main__":
    main()
