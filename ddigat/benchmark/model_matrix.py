from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from ddigat.data.tdc_ddi import load_tdc_drugbank_ddi
from ddigat.utils.io import ensure_dir, load_json, save_json


METRIC_COLUMNS = [
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
]


@dataclass(frozen=True)
class BenchmarkRow:
    name: str
    slug: str
    encoder_type: str
    runner_type: str = "gnn"
    use_ecfp_features: bool = False
    logit_adjust_tau: float = 0.0
    enable_drw: bool = False


@dataclass(frozen=True)
class FoldPlan:
    resolved_cold_k: int
    fallback_used: bool
    cold_max_resamples: int
    attempts: list[dict[str, object]]


def compute_drw_start_epoch(epochs: int, ratio: float = 0.7) -> int:
    if int(epochs) <= 0:
        raise ValueError("epochs must be > 0")
    if float(ratio) <= 0.0:
        raise ValueError("ratio must be > 0")
    return int(float(ratio) * int(epochs)) + 1


def build_benchmark_rows(tau: float, include_xgboost_ecfp4: bool = False) -> list[BenchmarkRow]:
    rows = [
        BenchmarkRow(name="GAT only", slug="gat_only", encoder_type="gat"),
        BenchmarkRow(name="GIN only", slug="gin_only", encoder_type="gin"),
        BenchmarkRow(name="GAT + ECFP", slug="gat_ecfp", encoder_type="gat", use_ecfp_features=True),
        BenchmarkRow(
            name="GAT + ECFP + LA",
            slug="gat_ecfp_la",
            encoder_type="gat",
            use_ecfp_features=True,
            logit_adjust_tau=float(tau),
        ),
        BenchmarkRow(
            name="GAT + ECFP + LA + DRW",
            slug="gat_ecfp_la_drw",
            encoder_type="gat",
            use_ecfp_features=True,
            logit_adjust_tau=float(tau),
            enable_drw=True,
        ),
    ]
    if bool(include_xgboost_ecfp4):
        rows.append(
            BenchmarkRow(
                name="XGBoost + ECFP4",
                slug="xgboost_ecfp4",
                encoder_type="xgboost",
                runner_type="xgboost",
                use_ecfp_features=True,
            )
        )
    return rows


def _materialize_cold_splits(
    *,
    data_dir: str,
    split_cache_dir: str,
    split_seed: int,
    cold_k: int,
    cold_protocol: str,
    cold_min_test_pairs: int,
    cold_min_test_labels: int,
    cold_max_resamples: int,
    cold_dedupe_policy: str,
    cold_selection_objective: str,
) -> None:
    for fold_idx in range(int(cold_k)):
        load_tdc_drugbank_ddi(
            data_dir=data_dir,
            output_dir=split_cache_dir,
            split_strategy="cold_drug",
            split_seed=int(split_seed),
            cold_k=int(cold_k),
            cold_fold=int(fold_idx),
            cold_protocol=str(cold_protocol),
            cold_min_test_pairs=int(cold_min_test_pairs),
            cold_min_test_labels=int(cold_min_test_labels),
            cold_max_resamples=int(cold_max_resamples),
            cold_dedupe_policy=str(cold_dedupe_policy),
            cold_selection_objective=str(cold_selection_objective),
        )


def resolve_fold_plan(
    *,
    data_dir: str,
    split_cache_root: str | Path,
    split_seed: int,
    preferred_cold_k: int = 10,
    fallback_cold_k: int = 5,
    cold_protocol: str = "s1",
    cold_min_test_pairs: int = 5000,
    cold_min_test_labels: int = 45,
    cold_dedupe_policy: str = "keep_all",
    cold_selection_objective: str = "selected_fold",
    preferred_max_resamples: int = 1000,
    fallback_max_resamples: int = 200,
    materialize_fn: Callable[..., None] | None = None,
) -> FoldPlan:
    materialize = materialize_fn or _materialize_cold_splits
    attempts: list[dict[str, object]] = []
    candidates = [
        (int(preferred_cold_k), int(preferred_max_resamples), False),
        (int(fallback_cold_k), int(fallback_max_resamples), True),
    ]
    last_error: Exception | None = None

    for cold_k, cold_max_resamples, is_fallback in candidates:
        split_cache_dir = str(Path(split_cache_root) / f"k{cold_k}")
        try:
            materialize(
                data_dir=data_dir,
                split_cache_dir=split_cache_dir,
                split_seed=int(split_seed),
                cold_k=int(cold_k),
                cold_protocol=str(cold_protocol),
                cold_min_test_pairs=int(cold_min_test_pairs),
                cold_min_test_labels=int(cold_min_test_labels),
                cold_max_resamples=int(cold_max_resamples),
                cold_dedupe_policy=str(cold_dedupe_policy),
                cold_selection_objective=str(cold_selection_objective),
            )
        except Exception as exc:
            last_error = exc
            attempts.append(
                {
                    "cold_k": int(cold_k),
                    "cold_max_resamples": int(cold_max_resamples),
                    "fallback_candidate": bool(is_fallback),
                    "success": False,
                    "error": str(exc),
                }
            )
            continue

        attempts.append(
            {
                "cold_k": int(cold_k),
                "cold_max_resamples": int(cold_max_resamples),
                "fallback_candidate": bool(is_fallback),
                "success": True,
            }
        )
        return FoldPlan(
            resolved_cold_k=int(cold_k),
            fallback_used=bool(is_fallback),
            cold_max_resamples=int(cold_max_resamples),
            attempts=attempts,
        )

    if last_error is None:
        raise RuntimeError("Unable to resolve a fold plan.")
    raise RuntimeError(
        f"Unable to materialize cold-drug splits for preferred_cold_k={preferred_cold_k} "
        f"or fallback_cold_k={fallback_cold_k}: {last_error}"
    ) from last_error


def read_uncalibrated_metrics(path: str | Path) -> dict[str, object]:
    payload = load_json(path)
    metrics = dict(payload.get("uncalibrated", {}))
    if "kappa" not in metrics and "cohen_kappa" in metrics:
        metrics["kappa"] = metrics["cohen_kappa"]
    return metrics


def _format_metric(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.4f} +- {std_value:.4f}"


def write_benchmark_report(
    *,
    benchmark_dir: str | Path,
    raw_rows: list[dict[str, object]],
    summary_payload: dict[str, object],
) -> dict[str, str]:
    out_dir = ensure_dir(benchmark_dir)
    df = pd.DataFrame(raw_rows)
    raw_csv = out_dir / "benchmark_results_raw.csv"
    df.to_csv(raw_csv, index=False)

    agg = df.groupby("model")[METRIC_COLUMNS].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(col).strip("_") for col in agg.columns.to_flat_index()]
    agg_csv = out_dir / "benchmark_results_mean_std.csv"
    agg.to_csv(agg_csv, index=False)

    md_path = out_dir / "benchmark_table.md"
    resolved_cold_k = int(summary_payload["resolved_cold_k"])
    fallback_used = bool(summary_payload["fallback_used"])
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(f"resolved_cold_k={resolved_cold_k} fallback_used={str(fallback_used).lower()}\n\n")
        handle.write(
            "| Model | Accuracy | Macro-F1 | Micro-F1 | Kappa | Macro ROC-AUC | Macro PR-AUC | "
            "Tail Macro PR-AUC | ECE | Brier | Objective Loss | NLL Loss | "
            "N Classes Scored | Missing Pos | Missing Neg | Tail Classes Scored |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for _, row in agg.iterrows():
            handle.write(
                f"| {row['model']} | "
                f"{_format_metric(row['accuracy_mean'], row['accuracy_std'])} | "
                f"{_format_metric(row['macro_f1_mean'], row['macro_f1_std'])} | "
                f"{_format_metric(row['micro_f1_mean'], row['micro_f1_std'])} | "
                f"{_format_metric(row['kappa_mean'], row['kappa_std'])} | "
                f"{_format_metric(row['macro_roc_auc_ovr_mean'], row['macro_roc_auc_ovr_std'])} | "
                f"{_format_metric(row['macro_pr_auc_ovr_mean'], row['macro_pr_auc_ovr_std'])} | "
                f"{_format_metric(row['tail_macro_pr_auc_ovr_mean'], row['tail_macro_pr_auc_ovr_std'])} | "
                f"{_format_metric(row['ece_mean'], row['ece_std'])} | "
                f"{_format_metric(row['brier_score_mean'], row['brier_score_std'])} | "
                f"{_format_metric(row['objective_loss_mean'], row['objective_loss_std'])} | "
                f"{_format_metric(row['nll_loss_mean'], row['nll_loss_std'])} | "
                f"{_format_metric(row['n_classes_scored_mean'], row['n_classes_scored_std'])} | "
                f"{_format_metric(row['n_classes_missing_pos_mean'], row['n_classes_missing_pos_std'])} | "
                f"{_format_metric(row['n_classes_missing_neg_mean'], row['n_classes_missing_neg_std'])} | "
                f"{_format_metric(row['tail_n_classes_scored_mean'], row['tail_n_classes_scored_std'])} |\n"
            )

    summary_path = out_dir / "benchmark_summary.json"
    payload = {
        **summary_payload,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_results_csv": str(raw_csv),
        "mean_std_results_csv": str(agg_csv),
        "markdown_table": str(md_path),
        "metric_columns": list(METRIC_COLUMNS),
    }
    save_json(payload, summary_path)
    return {
        "raw_csv": str(raw_csv),
        "agg_csv": str(agg_csv),
        "markdown": str(md_path),
        "summary_json": str(summary_path),
    }
