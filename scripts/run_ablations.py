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
    p.add_argument("--class_weight_beta", type=float, default=0.9999)
    p.add_argument("--class_weight_normalize", type=str, default="sample_mean", choices=["sample_mean", "mean_seen", "none"])
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--ablation_suite", type=str, default="default", choices=["default", "feature"])
    p.add_argument("--baseline_use_class_weights", action="store_true")
    p.add_argument("--baseline_class_weight_method", type=str, default="inv_sqrt", choices=["inv_sqrt", "effective_num"])
    p.add_argument("--baseline_label_smoothing", type=float, default=0.0)
    p.add_argument("--calibrate_temperature", action="store_true")
    p.add_argument("--use_ecfp_features", action="store_true")
    p.add_argument("--use_physchem_features", action="store_true")
    p.add_argument("--use_maccs_features", action="store_true")
    p.add_argument("--ecfp_bits", type=int, default=2048)
    p.add_argument("--ecfp_radius", type=int, default=2)
    p.add_argument("--physchem_dim", type=int, default=0)
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


def _default_ablations() -> list[dict]:
    return [
        {"name": "GAT", "encoder_type": "gat", "use_class_weights": False, "calibrate": False, "label_smoothing": 0.0},
        {"name": "GCN", "encoder_type": "gcn", "use_class_weights": False, "calibrate": False, "label_smoothing": 0.0},
        {"name": "GIN", "encoder_type": "gin", "use_class_weights": False, "calibrate": False, "label_smoothing": 0.0},
        {
            "name": "GAT+ClassWeights(inv_sqrt)",
            "encoder_type": "gat",
            "use_class_weights": True,
            "class_weight_method": "inv_sqrt",
            "calibrate": False,
            "label_smoothing": 0.0,
        },
        {
            "name": "GAT+ClassWeights(effective_num)",
            "encoder_type": "gat",
            "use_class_weights": True,
            "class_weight_method": "effective_num",
            "calibrate": False,
            "label_smoothing": 0.0,
        },
        {
            "name": "GAT+ClassWeights(inv_sqrt)+LS0.05",
            "encoder_type": "gat",
            "use_class_weights": True,
            "class_weight_method": "inv_sqrt",
            "calibrate": False,
            "label_smoothing": 0.05,
        },
        {
            "name": "GAT+ClassWeights(inv_sqrt)+TempScaling",
            "encoder_type": "gat",
            "use_class_weights": True,
            "class_weight_method": "inv_sqrt",
            "calibrate": True,
            "label_smoothing": 0.0,
        },
    ]


def _feature_isolation_ablations(args: argparse.Namespace) -> list[dict]:
    common = {
        "encoder_type": "gat",
        "use_class_weights": bool(args.baseline_use_class_weights),
        "class_weight_method": str(args.baseline_class_weight_method),
        "calibrate": bool(args.calibrate_temperature),
        "label_smoothing": float(args.baseline_label_smoothing),
    }
    return [
        {**common, "name": "GAT-baseline", "use_ecfp_features": False, "use_physchem_features": False, "use_maccs_features": False},
        {**common, "name": "GAT+ECFP", "use_ecfp_features": True, "use_physchem_features": False, "use_maccs_features": False},
        {**common, "name": "GAT+Physchem", "use_ecfp_features": False, "use_physchem_features": True, "use_maccs_features": False},
        {**common, "name": "GAT+MACCS", "use_ecfp_features": False, "use_physchem_features": False, "use_maccs_features": True},
        {**common, "name": "GAT+ECFP+Physchem", "use_ecfp_features": True, "use_physchem_features": True, "use_maccs_features": False},
    ]


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    if args.ablation_suite == "feature":
        ablations = _feature_isolation_ablations(args)
    else:
        ablations = _default_ablations()

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
            ckpt_path = run_out / "checkpoints" / "best.pt"
            if ckpt_path.exists():
                raise RuntimeError(
                    f"Refusing to reuse existing checkpoint for ablation run '{run_name}': {ckpt_path}"
                )

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
                str(ab["encoder_type"]),
                "--split_strategy",
                args.split_strategy,
                "--split_seed",
                str(args.split_seed),
            ]
            if args.limit is not None:
                train_cmd.extend(["--limit", str(args.limit)])
            if bool(ab["use_class_weights"]):
                train_cmd.extend(
                    [
                        "--use_class_weights",
                        "--class_weight_method",
                        str(ab.get("class_weight_method", "inv_sqrt")),
                        "--class_weight_normalize",
                        str(args.class_weight_normalize),
                        "--class_weight_beta",
                        str(args.class_weight_beta),
                    ]
                )
            train_cmd.extend(["--label_smoothing", str(float(ab.get("label_smoothing", 0.0)))])

            use_ecfp = bool(ab.get("use_ecfp_features", args.use_ecfp_features))
            use_physchem = bool(ab.get("use_physchem_features", args.use_physchem_features))
            use_maccs = bool(ab.get("use_maccs_features", args.use_maccs_features))
            if use_ecfp:
                train_cmd.append("--use_ecfp_features")
            if use_physchem:
                train_cmd.append("--use_physchem_features")
            if use_maccs:
                train_cmd.append("--use_maccs_features")
            train_cmd.extend(
                [
                    "--ecfp_bits",
                    str(int(args.ecfp_bits)),
                    "--ecfp_radius",
                    str(int(args.ecfp_radius)),
                    "--physchem_dim",
                    str(int(args.physchem_dim)),
                ]
            )
            _run(train_cmd)

            eval_cmd = [
                py,
                str(root / "scripts" / "evaluate.py"),
                "--data_dir",
                args.data_dir,
                "--output_dir",
                str(run_out),
                "--checkpoint",
                str(ckpt_path),
                "--batch_size",
                str(args.batch_size),
                "--device",
                args.device,
                "--split_strategy",
                args.split_strategy,
                "--split_seed",
                str(args.split_seed),
            ]
            if bool(ab["calibrate"]):
                eval_cmd.append("--calibrate_temperature")
            if use_ecfp:
                eval_cmd.append("--use_ecfp_features")
            if use_physchem:
                eval_cmd.append("--use_physchem_features")
            if use_maccs:
                eval_cmd.append("--use_maccs_features")
            eval_cmd.extend(
                [
                    "--ecfp_bits",
                    str(int(args.ecfp_bits)),
                    "--ecfp_radius",
                    str(int(args.ecfp_radius)),
                    "--physchem_dim",
                    str(int(args.physchem_dim)),
                ]
            )
            _run(eval_cmd)

            metrics_path = run_out / "evaluation_metrics.json"
            metrics = _read_eval_metrics(metrics_path, prefer_calibrated=bool(ab["calibrate"]))
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
        "objective_loss",
        "nll_loss",
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
