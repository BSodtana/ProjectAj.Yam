#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SELECTION_ORDER = "tail_macro_pr_auc_ovr > macro_pr_auc_ovr > macro_f1 > lower objective_loss"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(obj: dict, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)


@dataclass(frozen=True)
class SearchConfig:
    tau: float
    lr: float
    enable_drw: bool
    drw_ratio: float | None
    drw_lr_drop: float | None
    drw_start_epoch: int | None
    cold_fold: int

    @property
    def slug(self) -> str:
        parts = [
            f"tau_{_float_slug(self.tau)}",
            f"lr_{_float_slug(self.lr)}",
            f"fold_{self.cold_fold}",
        ]
        if not self.enable_drw:
            parts.append("drw_off")
        else:
            parts.extend(
                [
                    "drw_on",
                    f"ratio_{_float_slug(_require_value(self.drw_ratio, 'drw_ratio'))}",
                    f"drop_{_float_slug(_require_value(self.drw_lr_drop, 'drw_lr_drop'))}",
                    f"start_{_require_value(self.drw_start_epoch, 'drw_start_epoch')}",
                ]
            )
        return "_".join(parts)

    def to_dict(self) -> dict[str, object]:
        return {
            "tau": float(self.tau),
            "lr": float(self.lr),
            "enable_drw": bool(self.enable_drw),
            "drw_ratio": None if self.drw_ratio is None else float(self.drw_ratio),
            "drw_lr_drop": None if self.drw_lr_drop is None else float(self.drw_lr_drop),
            "drw_start_epoch": None if self.drw_start_epoch is None else int(self.drw_start_epoch),
            "cold_fold": int(self.cold_fold),
            "config_slug": self.slug,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-fold tuner for the baseline_v4 GAT+ECFP search space.")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_root", type=str, default="./outputs/tuning_baseline_v4_single_fold")
    p.add_argument("--python_bin", type=str, default=sys.executable)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--out_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=100000)
    p.add_argument("--cold_k", type=int, default=5)
    p.add_argument("--cold_fold", type=int, default=0)
    p.add_argument("--cold_protocol", type=str, default="s1", choices=["s1", "s2"])
    p.add_argument("--cold_min_test_pairs", type=int, default=5000)
    p.add_argument("--cold_min_test_labels", type=int, default=45)
    p.add_argument("--cold_max_resamples", type=int, default=200)
    p.add_argument("--cold_dedupe_policy", type=str, default="keep_all", choices=["keep_all", "keep_first"])
    p.add_argument(
        "--cold_selection_objective",
        type=str,
        default="selected_fold",
        choices=["selected_fold", "global_min", "first_pass"],
    )
    p.add_argument("--taus", type=str, default="0.4,0.5,0.75")
    p.add_argument("--lrs", type=str, default="3e-4,5e-4,1e-3,2e-3")
    p.add_argument("--drw_pairs", type=str, default="0.6:0.2,0.7:0.2,0.8:0.2")
    return p.parse_args()


def _parse_float_list(spec: str) -> list[float]:
    out = []
    for token in spec.split(","):
        text = token.strip()
        if not text:
            continue
        out.append(float(text))
    if not out:
        raise ValueError(f"Expected at least one float in '{spec}'")
    return out


def _parse_drw_pairs(spec: str) -> list[tuple[float, float]]:
    out = []
    for token in spec.split(","):
        text = token.strip()
        if not text:
            continue
        parts = text.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid drw pair '{text}'. Expected format ratio:lr_drop")
        ratio = float(parts[0].strip())
        lr_drop = float(parts[1].strip())
        out.append((ratio, lr_drop))
    if not out:
        raise ValueError(f"Expected at least one DRW pair in '{spec}'")
    return out


def _require_value(value, name: str):
    if value is None:
        raise ValueError(f"{name} is required")
    return value


def _float_slug(value: float) -> str:
    text = format(float(value), "g")
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def _build_search_configs(args: argparse.Namespace) -> list[SearchConfig]:
    taus = _parse_float_list(args.taus)
    lrs = _parse_float_list(args.lrs)
    drw_pairs = _parse_drw_pairs(args.drw_pairs)
    configs: list[SearchConfig] = []
    seen_slugs: set[str] = set()
    for tau in taus:
        for lr in lrs:
            for drw_ratio, drw_lr_drop in drw_pairs:
                drw_start_epoch = int(float(drw_ratio) * int(args.epochs)) + 1
                if drw_start_epoch < 1 or drw_start_epoch > int(args.epochs):
                    raise ValueError(
                        f"Invalid DRW start epoch {drw_start_epoch} for ratio={drw_ratio} with epochs={args.epochs}"
                    )
                cfg = SearchConfig(
                    tau=float(tau),
                    lr=float(lr),
                    enable_drw=True,
                    drw_ratio=float(drw_ratio),
                    drw_lr_drop=float(drw_lr_drop),
                    drw_start_epoch=int(drw_start_epoch),
                    cold_fold=int(args.cold_fold),
                )
                if cfg.slug in seen_slugs:
                    raise ValueError(f"Duplicate config slug detected: {cfg.slug}")
                configs.append(cfg)
                seen_slugs.add(cfg.slug)
    return configs


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _training_is_complete(run_dir: Path) -> bool:
    history_path = run_dir / "training_history.json"
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    if not history_path.exists() or not checkpoint_path.exists():
        return False
    try:
        payload = load_json(history_path)
    except Exception:
        return False
    best_metrics = payload.get("best_metrics", {})
    best_epoch = payload.get("best_epoch", None)
    return isinstance(best_metrics, dict) and len(best_metrics) > 0 and best_epoch is not None


def _evaluation_is_complete(run_dir: Path) -> bool:
    metrics_path = run_dir / "evaluation_metrics.json"
    if not metrics_path.exists():
        return False
    try:
        payload = load_json(metrics_path)
    except Exception:
        return False
    return isinstance(payload.get("uncalibrated", {}), dict) and isinstance(payload.get("summary", {}), dict)


def _load_validation_payload(run_dir: Path) -> dict[str, object]:
    payload = load_json(run_dir / "training_history.json")
    best_metrics = dict(payload.get("best_metrics", {}) or {})
    if "kappa" not in best_metrics and "cohen_kappa" in best_metrics:
        best_metrics["kappa"] = best_metrics["cohen_kappa"]
    return {
        "best_epoch": payload.get("best_epoch"),
        "best_metrics": best_metrics,
        "history_path": str(run_dir / "training_history.json"),
        "checkpoint_path": str(run_dir / "checkpoints" / "best.pt"),
    }


def _build_train_command(
    *,
    py: str,
    args: argparse.Namespace,
    run_dir: Path,
    split_cache_dir: Path,
    cfg: SearchConfig,
) -> list[str]:
    cmd = [
        py,
        str(PROJECT_ROOT / "scripts" / "train.py"),
        "--data_dir",
        str(args.data_dir),
        "--output_dir",
        str(run_dir),
        "--split_cache_dir",
        str(split_cache_dir),
        "--seed",
        str(int(args.seed)),
        "--batch_size",
        str(int(args.batch_size)),
        "--epochs",
        str(int(args.epochs)),
        "--lr",
        str(float(cfg.lr)),
        "--weight_decay",
        str(float(args.weight_decay)),
        "--device",
        str(args.device),
        "--num_workers",
        str(int(args.num_workers)),
        "--patience",
        str(int(args.patience)),
        "--limit",
        str(int(args.limit)),
        "--hidden_dim",
        str(int(args.hidden_dim)),
        "--out_dim",
        str(int(args.out_dim)),
        "--num_layers",
        str(int(args.num_layers)),
        "--heads",
        str(int(args.heads)),
        "--dropout",
        str(float(args.dropout)),
        "--encoder_type",
        "gat",
        "--use_ecfp_features",
        "--logit_adjust_tau",
        str(float(cfg.tau)),
        "--split_strategy",
        "cold_drug",
        "--split_seed",
        str(int(args.split_seed)),
        "--cold_k",
        str(int(args.cold_k)),
        "--cold_fold",
        str(int(cfg.cold_fold)),
        "--cold_protocol",
        str(args.cold_protocol),
        "--cold_min_test_pairs",
        str(int(args.cold_min_test_pairs)),
        "--cold_min_test_labels",
        str(int(args.cold_min_test_labels)),
        "--cold_max_resamples",
        str(int(args.cold_max_resamples)),
        "--cold_dedupe_policy",
        str(args.cold_dedupe_policy),
        "--cold_selection_objective",
        str(args.cold_selection_objective),
    ]
    if cfg.enable_drw:
        cmd.extend(
            [
                "--enable_drw",
                "--drw_start_epoch",
                str(int(_require_value(cfg.drw_start_epoch, "drw_start_epoch"))),
                "--drw_lr_drop",
                str(float(_require_value(cfg.drw_lr_drop, "drw_lr_drop"))),
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


def _build_eval_command(
    *,
    py: str,
    args: argparse.Namespace,
    run_dir: Path,
    split_cache_dir: Path,
    cfg: SearchConfig,
) -> list[str]:
    return [
        py,
        str(PROJECT_ROOT / "scripts" / "evaluate.py"),
        "--data_dir",
        str(args.data_dir),
        "--output_dir",
        str(run_dir),
        "--split_cache_dir",
        str(split_cache_dir),
        "--checkpoint",
        str(run_dir / "checkpoints" / "best.pt"),
        "--batch_size",
        str(int(args.batch_size)),
        "--device",
        str(args.device),
        "--split_strategy",
        "cold_drug",
        "--split_seed",
        str(int(args.split_seed)),
        "--cold_k",
        str(int(args.cold_k)),
        "--cold_fold",
        str(int(cfg.cold_fold)),
        "--cold_protocol",
        str(args.cold_protocol),
        "--cold_min_test_pairs",
        str(int(args.cold_min_test_pairs)),
        "--cold_min_test_labels",
        str(int(args.cold_min_test_labels)),
        "--cold_max_resamples",
        str(int(args.cold_max_resamples)),
        "--cold_dedupe_policy",
        str(args.cold_dedupe_policy),
        "--cold_selection_objective",
        str(args.cold_selection_objective),
        "--use_ecfp_features",
    ]


def _metric_for_sort(value, *, descending: bool) -> float:
    try:
        numeric = float(value)
    except Exception:
        return float("-inf") if descending else float("inf")
    if math.isnan(numeric):
        return float("-inf") if descending else float("inf")
    return numeric


def _rank_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    ranked = list(rows)
    ranked.sort(
        key=lambda row: (
            _metric_for_sort(row.get("val_tail_macro_pr_auc_ovr"), descending=True),
            _metric_for_sort(row.get("val_macro_pr_auc_ovr"), descending=True),
            _metric_for_sort(row.get("val_macro_f1"), descending=True),
            -_metric_for_sort(row.get("val_objective_loss"), descending=False),
        ),
        reverse=True,
    )
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = int(idx)
    return ranked


def _write_screen_reports(output_root: Path, rows: list[dict[str, object]]) -> tuple[Path, Path]:
    raw_path = output_root / "screen_results_raw.csv"
    ranked_path = output_root / "screen_results_ranked.csv"
    _write_csv(rows, raw_path)
    ranked_rows = _rank_rows([dict(row) for row in rows])
    _write_csv(ranked_rows, ranked_path)
    return raw_path, ranked_path


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    ensure_dir(path.parent)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            fieldnames.append(key)
            seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    configs = _build_search_configs(args)
    output_root = ensure_dir(args.output_root)
    runs_dir = ensure_dir(output_root / "runs")
    split_cache_dir = ensure_dir(output_root / "split_cache")
    py = str(args.python_bin)

    rows: list[dict[str, object]] = []
    total_configs = len(configs)
    for index, cfg in enumerate(configs, start=1):
        run_dir = ensure_dir(runs_dir / cfg.slug)
        print(f"[{index}/{total_configs}] {cfg.slug}")

        if not _training_is_complete(run_dir):
            _run(
                _build_train_command(
                    py=py,
                    args=args,
                    run_dir=run_dir,
                    split_cache_dir=split_cache_dir,
                    cfg=cfg,
                )
            )

        validation_payload = _load_validation_payload(run_dir)
        best_metrics = dict(validation_payload["best_metrics"])
        row = {
            **cfg.to_dict(),
            "train_limit": int(args.limit),
            "selection_order": SELECTION_ORDER,
            "run_dir": str(run_dir),
            "history_path": str(validation_payload["history_path"]),
            "checkpoint_path": str(validation_payload["checkpoint_path"]),
            "val_best_epoch": validation_payload["best_epoch"],
            "val_accuracy": best_metrics.get("accuracy"),
            "val_macro_f1": best_metrics.get("macro_f1"),
            "val_micro_f1": best_metrics.get("micro_f1"),
            "val_kappa": best_metrics.get("kappa"),
            "val_macro_roc_auc_ovr": best_metrics.get("macro_roc_auc_ovr"),
            "val_macro_pr_auc_ovr": best_metrics.get("macro_pr_auc_ovr"),
            "val_tail_macro_pr_auc_ovr": best_metrics.get("tail_macro_pr_auc_ovr"),
            "val_ece": best_metrics.get("ece"),
            "val_brier_score": best_metrics.get("brier_score"),
            "val_objective_loss": best_metrics.get("objective_loss"),
            "val_nll_loss": best_metrics.get("nll_loss"),
            "val_n_classes_scored": best_metrics.get("n_classes_scored"),
            "val_n_classes_missing_pos": best_metrics.get("n_classes_missing_pos"),
            "val_n_classes_missing_neg": best_metrics.get("n_classes_missing_neg"),
            "val_tail_n_classes_scored": best_metrics.get("tail_n_classes_scored"),
        }
        rows.append(row)
        _write_screen_reports(output_root, rows)

    ranked_rows = _rank_rows([dict(row) for row in rows])
    winner_row = dict(ranked_rows[0])
    winner_dir = Path(str(winner_row["run_dir"]))
    winner_cfg = next(cfg for cfg in configs if cfg.slug == str(winner_row["config_slug"]))

    winner_payload = {
        "selection_order": SELECTION_ORDER,
        "expected_config_count": int(total_configs),
        "completed_config_count": int(len(rows)),
        "winner": winner_row,
    }
    save_json(winner_payload, output_root / "winner.json")

    if not _evaluation_is_complete(winner_dir):
        _run(
            _build_eval_command(
                py=py,
                args=args,
                run_dir=winner_dir,
                split_cache_dir=split_cache_dir,
                cfg=winner_cfg,
            )
        )

    test_metrics = load_json(winner_dir / "evaluation_metrics.json")
    winner_test_payload = {
        "selection_order": SELECTION_ORDER,
        "winner": winner_row,
        "evaluation_metrics_path": str(winner_dir / "evaluation_metrics.json"),
        "test_metrics": test_metrics,
    }
    save_json(winner_test_payload, output_root / "winner_test_summary.json")

    raw_path, ranked_path = _write_screen_reports(output_root, rows)
    print(f"screen_results_raw={raw_path}")
    print(f"screen_results_ranked={ranked_path}")
    print(f"winner_json={output_root / 'winner.json'}")
    print(f"winner_test_summary={output_root / 'winner_test_summary.json'}")


if __name__ == "__main__":
    main()
