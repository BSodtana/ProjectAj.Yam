from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the DRW transition and tail-class prioritization for benchmark_k5 runs."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("outputs/adaptive_model_matrix/benchmark_k5/runs"),
        help="Directory containing benchmark run folders.",
    )
    parser.add_argument(
        "--run-glob",
        type=str,
        default="gat_ecfp_la_drw_fold*",
        help="Glob used to select DRW runs under --runs-root.",
    )
    parser.add_argument(
        "--epoch-start",
        type=int,
        default=12,
        help="First epoch to show in the transition window.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures/benchmark_k5_drw_transition.png"),
        help="Path to the output PNG.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compute_drw_weights(counts: np.ndarray) -> np.ndarray:
    seen = counts > 0
    weights = np.zeros_like(counts, dtype=np.float64)
    if np.any(seen):
        weights[seen] = 1.0 / np.sqrt(counts[seen].astype(np.float64))
        weights[seen] /= weights[seen].mean()
        weights[seen] = np.clip(weights[seen], 0.25, 4.0)
    return weights


def compute_tail_ids(counts: np.ndarray, fraction: float = 0.2) -> np.ndarray:
    order = np.argsort(counts, kind="stable")
    k = int(np.ceil(float(fraction) * float(counts.size)))
    k = max(1, min(k, int(counts.size)))
    return order[:k]


def build_run_payload(run_dir: Path) -> dict | None:
    history_payload = load_json(run_dir / "training_history.json")
    history = history_payload["history"]
    drw_epochs = [int(row["epoch"]) for row in history if bool(row.get("drw_on"))]
    if not drw_epochs:
        return None

    counts = np.asarray(load_json(run_dir / "diagnostics" / "train_counts.json")["counts"], dtype=np.int64)
    weights = compute_drw_weights(counts)
    tail_ids = compute_tail_ids(counts)
    seen_tail_ids = tail_ids[counts[tail_ids] > 0]
    head_start = int(math.floor(0.8 * counts.size))
    head_ids = np.argsort(counts, kind="stable")[head_start:]
    mid_ids = np.setdiff1d(np.arange(counts.size, dtype=np.int64), np.concatenate([tail_ids, head_ids]), assume_unique=False)

    weighted_mass = counts.astype(np.float64) * weights
    total_samples = float(counts.sum())
    total_weighted_mass = float(weighted_mass.sum())
    tail_weight_ratio = float(weights[seen_tail_ids].mean() / weights[head_ids].mean()) if seen_tail_ids.size else float("nan")

    return {
        "run_dir": str(run_dir),
        "history": history,
        "drw_start_epoch": min(drw_epochs),
        "counts": counts.tolist(),
        "tail_ids": tail_ids.tolist(),
        "seen_tail_ids": seen_tail_ids.tolist(),
        "head_ids": head_ids.tolist(),
        "mid_ids": mid_ids.tolist(),
        "weights": weights.tolist(),
        "tail_share_pre": float(counts[tail_ids].sum() / total_samples) if total_samples > 0 else 0.0,
        "tail_share_post": float(weighted_mass[tail_ids].sum() / total_weighted_mass) if total_weighted_mass > 0 else 0.0,
        "seen_tail_weight_mean": float(weights[seen_tail_ids].mean()) if seen_tail_ids.size else 0.0,
        "mid_weight_mean": float(weights[mid_ids].mean()) if mid_ids.size else 0.0,
        "head_weight_mean": float(weights[head_ids].mean()) if head_ids.size else 0.0,
        "tail_weight_ratio_vs_head": tail_weight_ratio,
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def main() -> None:
    args = parse_args()
    run_dirs = sorted(args.runs_root.glob(args.run_glob))
    payloads = [payload for payload in (build_run_payload(run_dir) for run_dir in run_dirs) if payload is not None]
    if not payloads:
        raise SystemExit(f"No complete DRW runs found under {args.runs_root} with glob {args.run_glob!r}")

    drw_start_epochs = sorted({int(payload["drw_start_epoch"]) for payload in payloads})
    if len(drw_start_epochs) != 1:
        raise SystemExit(f"Inconsistent DRW start epochs across runs: {drw_start_epochs}")
    drw_start_epoch = drw_start_epochs[0]

    epoch_stop = min(len(payload["history"]) for payload in payloads)
    epoch_start = max(1, min(int(args.epoch_start), epoch_stop))
    epochs = list(range(epoch_start, epoch_stop + 1))

    objective_means: list[float] = []
    objective_stds: list[float] = []
    nll_means: list[float] = []
    nll_stds: list[float] = []
    pr_means: list[float] = []
    pr_stds: list[float] = []

    for epoch in epochs:
        epoch_rows = [payload["history"][epoch - 1] for payload in payloads]
        mean, std = mean_std([float(row["train_objective_loss"]) for row in epoch_rows])
        objective_means.append(mean)
        objective_stds.append(std)
        mean, std = mean_std([float(row["train_plain_nll_loss"]) for row in epoch_rows])
        nll_means.append(mean)
        nll_stds.append(std)
        mean, std = mean_std([float(row["valid_macro_pr_auc_ovr"]) for row in epoch_rows])
        pr_means.append(mean)
        pr_stds.append(std)

    seen_tail_mean, seen_tail_std = mean_std([float(payload["seen_tail_weight_mean"]) for payload in payloads])
    mid_mean, mid_std = mean_std([float(payload["mid_weight_mean"]) for payload in payloads])
    head_mean, head_std = mean_std([float(payload["head_weight_mean"]) for payload in payloads])
    tail_share_pre_mean, tail_share_pre_std = mean_std([float(payload["tail_share_pre"]) for payload in payloads])
    tail_share_post_mean, tail_share_post_std = mean_std([float(payload["tail_share_post"]) for payload in payloads])
    tail_ratio_mean, tail_ratio_std = mean_std([float(payload["tail_weight_ratio_vs_head"]) for payload in payloads])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output.with_suffix(".summary.json")

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "font.size": 11,
        }
    )

    fig = plt.figure(figsize=(13.5, 7.5), facecolor="#f6f1e8")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.7, 1.0], height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.32)

    ax_loss = fig.add_subplot(gs[:, 0])
    ax_weight = fig.add_subplot(gs[0, 1])
    ax_share = fig.add_subplot(gs[1, 1])

    color_obj = "#9f2b2b"
    color_nll = "#1f5f8b"
    color_pr = "#6b8f23"
    shade = "#f2d8b3"

    ax_loss.axvspan(drw_start_epoch, epoch_stop + 0.3, color=shade, alpha=0.65, zorder=0)
    ax_loss.axvline(drw_start_epoch, color="#3a2f2f", linestyle="--", linewidth=1.6)
    ax_loss.plot(epochs, objective_means, color=color_obj, linewidth=2.6, label="Train objective loss")
    ax_loss.fill_between(
        epochs,
        np.asarray(objective_means) - np.asarray(objective_stds),
        np.asarray(objective_means) + np.asarray(objective_stds),
        color=color_obj,
        alpha=0.12,
    )
    ax_loss.plot(epochs, nll_means, color=color_nll, linewidth=2.3, label="Train plain NLL")
    ax_loss.fill_between(
        epochs,
        np.asarray(nll_means) - np.asarray(nll_stds),
        np.asarray(nll_means) + np.asarray(nll_stds),
        color=color_nll,
        alpha=0.10,
    )
    ax_loss.set_xlim(epoch_start - 0.2, epoch_stop + 0.2)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Training loss")
    ax_loss.set_title("DRW transition creates a visible loss split")
    ax_loss.grid(axis="y", alpha=0.18)

    ax_pr = ax_loss.twinx()
    ax_pr.plot(epochs, pr_means, color=color_pr, linewidth=2.0, linestyle="-.", label="Valid macro PR-AUC")
    ax_pr.fill_between(
        epochs,
        np.asarray(pr_means) - np.asarray(pr_stds),
        np.asarray(pr_means) + np.asarray(pr_stds),
        color=color_pr,
        alpha=0.10,
    )
    ax_pr.set_ylabel("Validation macro PR-AUC", color=color_pr)
    ax_pr.tick_params(axis="y", colors=color_pr)

    handles_1, labels_1 = ax_loss.get_legend_handles_labels()
    handles_2, labels_2 = ax_pr.get_legend_handles_labels()
    ax_loss.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper right", frameon=False)
    ax_loss.text(
        drw_start_epoch + 0.15,
        max(objective_means) * 0.98,
        "DRW on\nLR x0.2",
        ha="left",
        va="top",
        fontsize=10,
        color="#3a2f2f",
        bbox={"facecolor": "#fff8ef", "edgecolor": "none", "alpha": 0.85, "boxstyle": "round,pad=0.35"},
    )

    categories = ["Seen tail", "Middle", "Head"]
    pre_values = [1.0, 1.0, 1.0]
    post_values = [seen_tail_mean, mid_mean, head_mean]
    post_errors = [seen_tail_std, mid_std, head_std]
    x = np.arange(len(categories))
    width = 0.34
    ax_weight.bar(x - width / 2, pre_values, width=width, color="#d9d9d9", label="Before DRW")
    ax_weight.bar(x + width / 2, post_values, width=width, color="#c24b3a", yerr=post_errors, capsize=4, label="After DRW")
    ax_weight.set_xticks(x, categories)
    ax_weight.set_ylabel("Mean class loss multiplier")
    ax_weight.set_title("Per-class weighting flips toward the tail")
    ax_weight.grid(axis="y", alpha=0.18)
    ax_weight.legend(frameon=False, loc="lower right")
    ax_weight.text(
        0.02,
        0.82,
        f"Seen-tail classes get {tail_ratio_mean:.1f}x the head weight\n(std {tail_ratio_std:.1f}x across runs)",
        transform=ax_weight.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#4a2b20",
        bbox={"facecolor": "#fff8ef", "edgecolor": "none", "alpha": 0.9, "boxstyle": "round,pad=0.35"},
    )

    share_labels = ["Tail share\nbefore DRW", "Tail share\nafter DRW"]
    share_values = [tail_share_pre_mean * 100.0, tail_share_post_mean * 100.0]
    share_errors = [tail_share_pre_std * 100.0, tail_share_post_std * 100.0]
    share_colors = ["#9aa6b2", "#7a8f37"]
    ax_share.bar(np.arange(2), share_values, yerr=share_errors, color=share_colors, width=0.58, capsize=4)
    ax_share.set_xticks(np.arange(2), share_labels)
    ax_share.set_ylabel("Share of weighted sample mass (%)")
    ax_share.set_title("Tail classes claim more of the loss budget")
    ax_share.grid(axis="y", alpha=0.18)
    reallocation = share_values[1] / share_values[0] if share_values[0] > 0 else float("inf")
    ax_share.text(
        0.02,
        0.95,
        f"Tail contribution rises from {share_values[0]:.2f}% to {share_values[1]:.2f}%\n({reallocation:.1f}x more weighted emphasis)",
        transform=ax_share.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#243320",
        bbox={"facecolor": "#fff8ef", "edgecolor": "none", "alpha": 0.9, "boxstyle": "round,pad=0.35"},
    )

    fig.suptitle(
        "Benchmark k=5 DRW transition: the loss stops being class-neutral at epoch 15",
        fontsize=16,
        fontweight="bold",
        color="#2f2926",
        y=0.98,
    )
    fig.text(
        0.03,
        0.02,
        f"Runs used: {len(payloads)} complete DRW folds from {args.runs_root} (folds without a DRW switch are excluded).",
        fontsize=10,
        color="#5a514d",
    )

    fig.savefig(args.output, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    summary = {
        "runs_root": str(args.runs_root),
        "run_glob": args.run_glob,
        "runs_used": [payload["run_dir"] for payload in payloads],
        "drw_start_epoch": drw_start_epoch,
        "epoch_window": [epoch_start, epoch_stop],
        "mean_train_objective_loss": dict(zip([str(epoch) for epoch in epochs], objective_means)),
        "mean_train_plain_nll": dict(zip([str(epoch) for epoch in epochs], nll_means)),
        "mean_valid_macro_pr_auc_ovr": dict(zip([str(epoch) for epoch in epochs], pr_means)),
        "seen_tail_weight_mean": seen_tail_mean,
        "mid_weight_mean": mid_mean,
        "head_weight_mean": head_mean,
        "tail_weight_ratio_vs_head_mean": tail_ratio_mean,
        "tail_weight_ratio_vs_head_std": tail_ratio_std,
        "tail_share_pre_mean": tail_share_pre_mean,
        "tail_share_pre_std": tail_share_pre_std,
        "tail_share_post_mean": tail_share_post_mean,
        "tail_share_post_std": tail_share_post_std,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved figure to {args.output}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
