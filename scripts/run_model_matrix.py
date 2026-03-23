#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ddigat.benchmark import (
    build_benchmark_rows,
    compute_drw_start_epoch,
    resolve_fold_plan,
    write_benchmark_report,
)
from ddigat.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run adaptive cold-drug benchmark matrix.")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_root", type=str, default="./outputs/adaptive_model_matrix")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--preferred_cold_k", type=int, default=5)
    p.add_argument("--fallback_cold_k", type=int, default=5)
    p.add_argument("--cold_protocol", type=str, default="s1")
    p.add_argument("--cold_min_test_pairs", type=int, default=5000)
    p.add_argument("--cold_min_test_labels", type=int, default=45)
    p.add_argument("--cold_dedupe_policy", type=str, default="keep_all")
    p.add_argument("--cold_selection_objective", type=str, default="selected_fold")
    p.add_argument("--preferred_max_resamples", type=int, default=1000)
    p.add_argument("--fallback_max_resamples", type=int, default=200)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--drw_ratio", type=float, default=0.7)
    p.add_argument("--drw_lr_drop", type=float, default=0.2)
    p.add_argument("--include_xgboost_ecfp4", action="store_true")
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _train_command(
    *,
    py: str,
    root: Path,
    args: argparse.Namespace,
    output_dir: Path,
    split_cache_dir: Path,
    cold_k: int,
    cold_fold: int,
    row,
    drw_start_epoch: int,
) -> list[str]:
    cmd = [
        py,
        str(root / "scripts" / "train.py"),
        "--data_dir",
        args.data_dir,
        "--output_dir",
        str(output_dir),
        "--split_cache_dir",
        str(split_cache_dir),
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--device",
        str(args.device),
        "--encoder_type",
        str(row.encoder_type),
        "--split_strategy",
        "cold_drug",
        "--split_seed",
        str(args.split_seed),
        "--cold_k",
        str(cold_k),
        "--cold_fold",
        str(cold_fold),
        "--cold_protocol",
        str(args.cold_protocol),
        "--cold_min_test_pairs",
        str(args.cold_min_test_pairs),
        "--cold_min_test_labels",
        str(args.cold_min_test_labels),
        "--cold_max_resamples",
        str(args.preferred_max_resamples if cold_k == int(args.preferred_cold_k) else args.fallback_max_resamples),
        "--cold_dedupe_policy",
        str(args.cold_dedupe_policy),
        "--cold_selection_objective",
        str(args.cold_selection_objective),
    ]
    if bool(row.use_ecfp_features):
        cmd.append("--use_ecfp_features")
    if float(row.logit_adjust_tau) > 0.0:
        cmd.extend(["--logit_adjust_tau", str(float(row.logit_adjust_tau))])
    if bool(row.enable_drw):
        cmd.extend(
            [
                "--enable_drw",
                "--drw_start_epoch",
                str(drw_start_epoch),
                "--drw_lr_drop",
                str(float(args.drw_lr_drop)),
                "--class_weight_method",
                "inv_sqrt",
                "--class_weight_normalize",
                "sample_mean",
                "--class_weight_clip_min",
                "0.25",
                "--class_weight_clip_max",
                "4.0",
                "--class_weight_eps",
                "1e-12",
            ]
        )
    return cmd


def _eval_command(
    *,
    py: str,
    root: Path,
    args: argparse.Namespace,
    output_dir: Path,
    split_cache_dir: Path,
    cold_k: int,
    cold_fold: int,
    row,
) -> list[str]:
    cmd = [
        py,
        str(root / "scripts" / "evaluate.py"),
        "--data_dir",
        args.data_dir,
        "--output_dir",
        str(output_dir),
        "--split_cache_dir",
        str(split_cache_dir),
        "--checkpoint",
        str(output_dir / "checkpoints" / "best.pt"),
        "--batch_size",
        str(args.batch_size),
        "--device",
        str(args.device),
        "--split_strategy",
        "cold_drug",
        "--split_seed",
        str(args.split_seed),
        "--cold_k",
        str(cold_k),
        "--cold_fold",
        str(cold_fold),
        "--cold_protocol",
        str(args.cold_protocol),
        "--cold_min_test_pairs",
        str(args.cold_min_test_pairs),
        "--cold_min_test_labels",
        str(args.cold_min_test_labels),
        "--cold_max_resamples",
        str(args.preferred_max_resamples if cold_k == int(args.preferred_cold_k) else args.fallback_max_resamples),
        "--cold_dedupe_policy",
        str(args.cold_dedupe_policy),
        "--cold_selection_objective",
        str(args.cold_selection_objective),
    ]
    if bool(row.use_ecfp_features):
        cmd.append("--use_ecfp_features")
    return cmd


def _xgboost_command(
    *,
    py: str,
    root: Path,
    args: argparse.Namespace,
    output_dir: Path,
    split_cache_dir: Path,
    cold_k: int,
    cold_fold: int,
) -> list[str]:
    return [
        py,
        str(root / "scripts" / "run_xgboost_ecfp4.py"),
        "--data_dir",
        args.data_dir,
        "--output_dir",
        str(output_dir),
        "--split_cache_dir",
        str(split_cache_dir),
        "--split_seed",
        str(args.split_seed),
        "--cold_k",
        str(cold_k),
        "--cold_fold",
        str(cold_fold),
        "--cold_protocol",
        str(args.cold_protocol),
        "--cold_min_test_pairs",
        str(args.cold_min_test_pairs),
        "--cold_min_test_labels",
        str(args.cold_min_test_labels),
        "--cold_max_resamples",
        str(args.preferred_max_resamples if cold_k == int(args.preferred_cold_k) else args.fallback_max_resamples),
        "--cold_dedupe_policy",
        str(args.cold_dedupe_policy),
        "--cold_selection_objective",
        str(args.cold_selection_objective),
    ]


def main() -> None:
    args = parse_args()
    root = PROJECT_ROOT
    py = sys.executable
    output_root = ensure_dir(args.output_root)
    split_cache_root = ensure_dir(Path(output_root) / "split_cache")

    fold_plan = resolve_fold_plan(
        data_dir=str(args.data_dir),
        split_cache_root=split_cache_root,
        split_seed=int(args.split_seed),
        preferred_cold_k=int(args.preferred_cold_k),
        fallback_cold_k=int(args.fallback_cold_k),
        cold_protocol=str(args.cold_protocol),
        cold_min_test_pairs=int(args.cold_min_test_pairs),
        cold_min_test_labels=int(args.cold_min_test_labels),
        cold_dedupe_policy=str(args.cold_dedupe_policy),
        cold_selection_objective=str(args.cold_selection_objective),
        preferred_max_resamples=int(args.preferred_max_resamples),
        fallback_max_resamples=int(args.fallback_max_resamples),
    )

    resolved_k = int(fold_plan.resolved_cold_k)
    benchmark_dir = ensure_dir(Path(output_root) / f"benchmark_k{resolved_k}")
    split_cache_dir = ensure_dir(split_cache_root / f"k{resolved_k}")
    runs_dir = ensure_dir(benchmark_dir / "runs")
    rows = build_benchmark_rows(
        tau=float(args.tau),
        include_xgboost_ecfp4=bool(args.include_xgboost_ecfp4),
    )
    drw_start_epoch = compute_drw_start_epoch(epochs=int(args.epochs), ratio=float(args.drw_ratio))

    raw_rows: list[dict[str, object]] = []
    run_records: list[dict[str, object]] = []
    for row in rows:
        for cold_fold in range(resolved_k):
            run_dir = ensure_dir(runs_dir / f"{row.slug}_fold{cold_fold}")
            metrics_path = run_dir / "evaluation_metrics.json"
            if str(row.runner_type) == "xgboost":
                artifact_path = run_dir / "model.ubj"
                if not metrics_path.exists():
                    _run(
                        _xgboost_command(
                            py=py,
                            root=root,
                            args=args,
                            output_dir=run_dir,
                            split_cache_dir=split_cache_dir,
                            cold_k=resolved_k,
                            cold_fold=cold_fold,
                        )
                    )
            else:
                artifact_path = run_dir / "checkpoints" / "best.pt"
                if not artifact_path.exists():
                    _run(
                        _train_command(
                            py=py,
                            root=root,
                            args=args,
                            output_dir=run_dir,
                            split_cache_dir=split_cache_dir,
                            cold_k=resolved_k,
                            cold_fold=cold_fold,
                            row=row,
                            drw_start_epoch=drw_start_epoch,
                        )
                    )

                if not metrics_path.exists():
                    _run(
                        _eval_command(
                            py=py,
                            root=root,
                            args=args,
                            output_dir=run_dir,
                            split_cache_dir=split_cache_dir,
                            cold_k=resolved_k,
                            cold_fold=cold_fold,
                            row=row,
                        )
                    )

            from ddigat.benchmark.model_matrix import read_uncalibrated_metrics

            metrics = read_uncalibrated_metrics(metrics_path)
            raw_row = {
                "model": row.name,
                "model_slug": row.slug,
                "cold_fold": int(cold_fold),
                "resolved_cold_k": int(resolved_k),
                "runner_type": str(row.runner_type),
                "encoder_type": row.encoder_type,
                "use_ecfp_features": bool(row.use_ecfp_features),
                "logit_adjust_tau": float(row.logit_adjust_tau),
                "enable_drw": bool(row.enable_drw),
            }
            for metric_name in metrics:
                if metric_name in {
                    "accuracy",
                    "macro_f1",
                    "micro_f1",
                    "kappa",
                    "macro_roc_auc_ovr",
                    "macro_pr_auc_ovr",
                    "tail_macro_pr_auc_ovr",
                    "ece",
                    "brier_score",
                    "objective_loss",
                    "nll_loss",
                    "n_classes_scored",
                    "n_classes_missing_pos",
                    "n_classes_missing_neg",
                    "tail_n_classes_scored",
                }:
                    raw_row[metric_name] = metrics[metric_name]
            raw_rows.append(raw_row)
            run_records.append(
                {
                    "model": row.name,
                    "cold_fold": int(cold_fold),
                    "run_dir": str(run_dir),
                    "artifact_path": str(artifact_path),
                    "metrics_path": str(metrics_path),
                }
            )

    report_paths = write_benchmark_report(
        benchmark_dir=benchmark_dir,
        raw_rows=raw_rows,
        summary_payload={
            "preferred_cold_k": int(args.preferred_cold_k),
            "fallback_cold_k": int(args.fallback_cold_k),
            "resolved_cold_k": int(resolved_k),
            "fallback_used": bool(fold_plan.fallback_used),
            "cold_max_resamples": int(fold_plan.cold_max_resamples),
            "tau": float(args.tau),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "split_seed": int(args.split_seed),
            "cold_protocol": str(args.cold_protocol),
            "cold_min_test_pairs": int(args.cold_min_test_pairs),
            "cold_min_test_labels": int(args.cold_min_test_labels),
            "cold_dedupe_policy": str(args.cold_dedupe_policy),
            "cold_selection_objective": str(args.cold_selection_objective),
            "drw_ratio": float(args.drw_ratio),
            "drw_start_epoch": int(drw_start_epoch),
            "drw_lr_drop": float(args.drw_lr_drop),
            "runs_expected": int(len(rows) * resolved_k),
            "runs_completed": int(len(raw_rows)),
            "split_cache_dir": str(split_cache_dir),
            "benchmark_dir": str(benchmark_dir),
            "fold_resolution_attempts": fold_plan.attempts,
            "models": [row.name for row in rows],
            "run_records": run_records,
        },
    )
    print(f"benchmark_dir={benchmark_dir}")
    for key, value in report_paths.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
