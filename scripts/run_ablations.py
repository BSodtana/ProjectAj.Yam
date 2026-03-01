#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run 3-seed ablations and aggregate mean±std metrics.")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--seeds", type=str, default="42,43,44")
    p.add_argument("--limit", type=int, default=50000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--split_strategy", type=str, default="cold_drug", choices=["cold_drug", "tdc"])
    p.add_argument("--split_seed", type=int, default=42)
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _read_eval_metrics(path: Path, prefer_calibrated: bool) -> dict:
    data = json.loads(path.read_text())
    if prefer_calibrated and "calibrated" in data:
        return data["calibrated"]
    return data["uncalibrated"]


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    ablations = [
        {"name": "GAT", "encoder_type": "gat", "use_class_weights": False, "calibrate": False},
        {"name": "GCN", "encoder_type": "gcn", "use_class_weights": False, "calibrate": False},
        {"name": "GIN", "encoder_type": "gin", "use_class_weights": False, "calibrate": False},
        {"name": "GAT+ClassWeights", "encoder_type": "gat", "use_class_weights": True, "calibrate": False},
        {"name": "GAT+ClassWeights+TempScaling", "encoder_type": "gat", "use_class_weights": True, "calibrate": True},
    ]

    all_rows: list[dict] = []
    py = sys.executable
    root = Path(__file__).resolve().parents[1]
    out_root = Path(args.output_dir) / "ablations"
    out_root.mkdir(parents=True, exist_ok=True)

    for ab in ablations:
        for seed in seeds:
            run_name = f"{ab['name']}_seed{seed}".replace("+", "_")
            run_out = out_root / run_name
            run_out.mkdir(parents=True, exist_ok=True)

            train_cmd = [
                py,
                str(root / "scripts" / "train.py"),
                "--data_dir",
                args.data_dir,
                "--output_dir",
                str(run_out),
                "--seed",
                str(seed),
                "--batch_size",
                str(args.batch_size),
                "--epochs",
                str(args.epochs),
                "--lr",
                str(args.lr),
                "--weight_decay",
                str(args.weight_decay),
                "--device",
                args.device,
                "--encoder_type",
                ab["encoder_type"],
                "--split_strategy",
                args.split_strategy,
                "--split_seed",
                str(args.split_seed),
            ]
            if args.limit is not None:
                train_cmd.extend(["--limit", str(args.limit)])
            if ab["use_class_weights"]:
                train_cmd.append("--use_class_weights")
            _run(train_cmd)

            eval_cmd = [
                py,
                str(root / "scripts" / "evaluate.py"),
                "--data_dir",
                args.data_dir,
                "--output_dir",
                str(run_out),
                "--checkpoint",
                str(run_out / "checkpoints" / "best.pt"),
                "--batch_size",
                str(args.batch_size),
                "--device",
                args.device,
                "--split_strategy",
                args.split_strategy,
                "--split_seed",
                str(args.split_seed),
            ]
            if ab["calibrate"]:
                eval_cmd.append("--calibrate_temperature")
            _run(eval_cmd)

            metrics_path = run_out / "evaluation_metrics.json"
            metrics = _read_eval_metrics(metrics_path, prefer_calibrated=ab["calibrate"])
            row = {"ablation": ab["name"], "seed": seed}
            row.update(metrics)
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    raw_csv = out_root / "ablation_results_raw.csv"
    df.to_csv(raw_csv, index=False)

    metric_cols = [
        "accuracy",
        "macro_f1",
        "micro_f1",
        "cohen_kappa",
        "macro_roc_auc_ovr",
        "macro_pr_auc_ovr",
        "ece",
        "brier_score",
        "loss",
    ]
    agg = df.groupby("ablation")[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).strip("_") for c in agg.columns.to_flat_index()]
    agg_csv = out_root / "ablation_results_mean_std.csv"
    agg.to_csv(agg_csv, index=False)

    md_path = out_root / "ablation_table.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| Ablation | Accuracy | Macro-F1 | Micro-F1 | Kappa | Macro ROC-AUC | Macro PR-AUC | ECE | Brier |\\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\\n")
        for _, r in agg.iterrows():
            def fmt(m):
                return f"{r[f'{m}_mean']:.4f} ± {r[f'{m}_std']:.4f}"

            f.write(
                f"| {r['ablation']} | {fmt('accuracy')} | {fmt('macro_f1')} | {fmt('micro_f1')} | "
                f"{fmt('cohen_kappa')} | {fmt('macro_roc_auc_ovr')} | {fmt('macro_pr_auc_ovr')} | "
                f"{fmt('ece')} | {fmt('brier_score')} |\\n"
            )

    print(f"raw_results={raw_csv}")
    print(f"mean_std_results={agg_csv}")
    print(f"table_markdown={md_path}")


if __name__ == "__main__":
    main()
