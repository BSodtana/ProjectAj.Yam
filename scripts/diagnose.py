#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ddigat.data.cache import GraphCache
from ddigat.data.splits import DDIPairDataset, make_pair_dataloader, subsample_dataframe
from ddigat.data.tdc_ddi import load_tdc_drugbank_ddi
from ddigat.explain.attention import get_node_importance_from_attention
from ddigat.explain.faithfulness import deletion_test, insertion_test
from ddigat.model.pair_model import DDIPairModel
from ddigat.train.loop import eval_epoch
from ddigat.utils.calibration import apply_temperature, fit_temperature
from ddigat.utils.io import ensure_dir, save_json, torch_load
from ddigat.utils.logging import get_logger
from ddigat.utils.metrics import evaluate_multiclass_metrics
from ddigat.utils.seed import seed_everything


LOGGER = get_logger("scripts.diagnose")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run DDI-GAT diagnostics and sanity checks.")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--checkpoint", type=str, default="./outputs/checkpoints/best.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--ece_bins", type=int, default=15)
    p.add_argument("--faithfulness_examples", type=int, default=40)
    p.add_argument("--random_baseline_repeats", type=int, default=5)
    p.add_argument("--randomization_steps", type=int, default=50)
    p.add_argument("--randomization_lr", type=float, default=1e-3)
    p.add_argument("--split_strategy", type=str, default="cold_drug", choices=["cold_drug", "tdc"])
    p.add_argument("--split_seed", type=int, default=42)
    return p.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            LOGGER.warning(
                "MPS detected, but PyG ops used by this project are not fully supported on MPS. "
                "Falling back to CPU for stability."
            )
            return torch.device("cpu")
        return torch.device("cpu")
    if device_arg == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        LOGGER.warning(
            "Using MPS with PYTORCH_ENABLE_MPS_FALLBACK=1. "
            "Unsupported ops will run on CPU and execution may be slower."
        )
        return torch.device("mps")
    return torch.device(device_arg)


def build_model_from_checkpoint_payload(payload: dict, device: torch.device) -> DDIPairModel:
    cfg = (payload.get("config") or {}).get("model", {})
    model = DDIPairModel(
        in_dim=int(cfg.get("in_dim", 7)),
        edge_dim=int(cfg.get("edge_dim", 5)),
        encoder_type=str(cfg.get("encoder_type", "gat")),
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        out_dim=int(cfg.get("out_dim", 128)),
        num_layers=int(cfg.get("num_layers", 3)),
        heads=int(cfg.get("heads", 4)),
        dropout=float(cfg.get("dropout", 0.2)),
        mlp_hidden_dim=int(cfg.get("mlp_hidden_dim", 256)),
        num_classes=int(cfg.get("num_classes", 86)),
        pooling=str(cfg.get("pooling", "mean")),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y.astype(int), minlength=num_classes).astype(np.float64)
    inv = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    inv = inv / np.mean(inv)
    inv = np.clip(inv, 0.2, 5.0)
    return torch.tensor(inv, dtype=torch.float32)


def restore_loss_params_from_checkpoint(
    model: DDIPairModel,
    payload: dict,
    train_df: pd.DataFrame,
) -> dict[str, Any]:
    cfg = payload.get("config", {}) or {}
    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    use_class_weights = bool(train_cfg.get("use_class_weights", False))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(train_df["y"].to_numpy(dtype=int), num_classes=model.num_classes)
    model.set_loss_params(class_weights=class_weights, label_smoothing=label_smoothing)
    return {
        "use_class_weights": use_class_weights,
        "label_smoothing": label_smoothing,
        "class_weight_min": None if class_weights is None else float(class_weights.min().item()),
        "class_weight_max": None if class_weights is None else float(class_weights.max().item()),
    }


def _k_schedule(num_nodes: int) -> list[int]:
    base = [0, 1, 2, 5, 10, 15, 20, 30]
    ks = sorted({min(max(0, k), num_nodes) for k in base} | {num_nodes})
    return [k for k in ks if k <= num_nodes]


def _curve_auc(k: list[int], y: list[float]) -> float:
    if len(k) < 2:
        return float(y[0]) if y else float("nan")
    k_arr = np.asarray(k, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    area = np.trapz(y_arr, k_arr)
    norm = max(k_arr.max() - k_arr.min(), 1.0)
    return float(area / norm)


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    batch["graph_a"] = batch["graph_a"].to(device)
    batch["graph_b"] = batch["graph_b"].to(device)
    batch["y"] = batch["y"].to(device)
    return batch


def _random_baseline_curves(
    model: DDIPairModel,
    graph_a,
    graph_b,
    target_class: int,
    k_list: list[int],
    which: str,
    repeats: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[list[float], list[float], float]:
    n = int(graph_a.x.size(0) if which == "A" else graph_b.x.size(0))
    del_curves = []
    ins_curves = []
    permutations: list[tuple[int, ...]] = []
    for _ in range(repeats):
        ranking = tuple(rng.permutation(n).tolist())
        permutations.append(ranking)
        d = deletion_test(model, graph_a, graph_b, target_class, list(ranking), k_list, which=which, device=device)
        ins = insertion_test(model, graph_a, graph_b, target_class, list(ranking), k_list, which=which, device=device)
        del_curves.append(np.asarray(d, dtype=float))
        ins_curves.append(np.asarray(ins, dtype=float))
    unique_ratio = len(set(permutations)) / max(len(permutations), 1)
    del_mean = np.mean(np.stack(del_curves, axis=0), axis=0).tolist()
    ins_mean = np.mean(np.stack(ins_curves, axis=0), axis=0).tolist()
    return del_mean, ins_mean, float(unique_ratio)


@torch.no_grad()
def _predict_with_attention(model: DDIPairModel, graph_a, graph_b, device: torch.device):
    logits, attn = model.forward_with_attention(graph_a.to(device), graph_b.to(device))
    probs = torch.softmax(logits, dim=-1).detach().cpu()[0].numpy()
    pred_class = int(np.argmax(probs))
    return probs, pred_class, attn


def _score_graph_from_attention(
    graph,
    attn_payload: dict | None,
) -> np.ndarray:
    if not attn_payload or attn_payload.get("alpha") is None:
        return np.ones(int(graph.x.size(0)), dtype=float)
    scores = get_node_importance_from_attention(
        graph,
        attn_payload["edge_index"],
        attn_payload["alpha"],
        node_embeddings=attn_payload.get("node_input_embeddings", attn_payload.get("node_embeddings")),
        src_message_embeddings=attn_payload.get("src_message_embeddings"),
        batch=attn_payload.get("batch"),
        remove_self_loops=True,
        normalize_per_graph=True,
        aggregate_heads="mean",
    )
    return scores.numpy()


def _topk_jaccard(rank_a: list[int], rank_b: list[int], k: int) -> float:
    sa = set(rank_a[:k])
    sb = set(rank_b[:k])
    union = sa.union(sb)
    if not union:
        return 1.0
    return float(len(sa.intersection(sb)) / len(union))


def _build_weight_randomized_copy(base_model: DDIPairModel, seed: int) -> DDIPairModel:
    model = copy.deepcopy(base_model)
    torch.manual_seed(seed)
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            try:
                module.reset_parameters()
            except Exception:
                pass
    model.eval()
    return model


def _train_label_permuted_copy(
    base_model: DDIPairModel,
    loader,
    device: torch.device,
    seed: int,
    steps: int,
    lr: float,
) -> DDIPairModel:
    torch.manual_seed(seed)
    model = copy.deepcopy(base_model).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    done = 0
    for batch in loader:
        if batch is None:
            continue
        batch = _move_batch_to_device(batch, device)
        y = batch["y"]
        if y.numel() > 1:
            perm = y[torch.randperm(y.numel(), device=y.device)]
        else:
            rand_shift = torch.randint(
                low=1,
                high=max(int(model.num_classes), 2),
                size=(1,),
                device=y.device,
            )
            perm = (y + rand_shift) % int(model.num_classes)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["graph_a"], batch["graph_b"])
        loss = model.loss_fn(logits, perm)
        loss.backward()
        optimizer.step()
        done += 1
        if done >= steps:
            break
    model.eval()
    return model


def _label_checks(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, num_classes: int) -> dict:
    out: dict[str, Any] = {"num_classes": int(num_classes)}
    for name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        y = df["y"].astype(int).to_numpy()
        unique = sorted(np.unique(y).tolist())
        out[name] = {
            "n": int(len(y)),
            "min": int(np.min(y)) if y.size else None,
            "max": int(np.max(y)) if y.size else None,
            "n_unique": int(len(unique)),
            "missing_classes": [c for c in range(num_classes) if c not in set(unique)],
            "invalid_labels": [int(v) for v in unique if v < 0 or v >= num_classes],
        }
    return out


def _class_distribution_df(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, num_classes: int) -> pd.DataFrame:
    rows = []
    train_counts = train_df["y"].value_counts().to_dict()
    valid_counts = valid_df["y"].value_counts().to_dict()
    test_counts = test_df["y"].value_counts().to_dict()
    for c in range(num_classes):
        tr = int(train_counts.get(c, 0))
        va = int(valid_counts.get(c, 0))
        te = int(test_counts.get(c, 0))
        total = tr + va + te
        rows.append({"class_id": c, "train_count": tr, "valid_count": va, "test_count": te, "total_count": total})
    return pd.DataFrame(rows)


def _imbalance_summary(df: pd.DataFrame, split_col: str) -> dict[str, float]:
    vals = df[split_col].to_numpy(dtype=float)
    nonzero = vals[vals > 0]
    if nonzero.size == 0:
        return {"min_nonzero": float("nan"), "median_nonzero": float("nan"), "max": float("nan"), "max_over_min_nonzero": float("nan")}
    return {
        "min_nonzero": float(np.min(nonzero)),
        "median_nonzero": float(np.median(nonzero)),
        "max": float(np.max(vals)),
        "max_over_min_nonzero": float(np.max(vals) / np.min(nonzero)),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)

    diagnostics_dir = ensure_dir(Path(args.output_dir) / "diagnostics")

    payload = torch_load(args.checkpoint, map_location=device)
    model = build_model_from_checkpoint_payload(payload, device=device)

    train_df, valid_df, test_df, label_map = load_tdc_drugbank_ddi(
        args.data_dir,
        output_dir=args.output_dir,
        split_strategy=args.split_strategy,
        split_seed=args.split_seed,
    )
    if args.limit is not None:
        train_df = subsample_dataframe(train_df, limit=args.limit, seed=args.seed, label_col="y", ensure_class_coverage=True)
        valid_df = subsample_dataframe(valid_df, limit=args.limit, seed=args.seed + 1, label_col="y", ensure_class_coverage=True)
        test_df = subsample_dataframe(test_df, limit=args.limit, seed=args.seed + 2, label_col="y", ensure_class_coverage=True)

    loss_cfg = restore_loss_params_from_checkpoint(model, payload, train_df=train_df)
    LOGGER.info("Restored eval objective: %s", loss_cfg)

    labels_check = _label_checks(train_df, valid_df, test_df, num_classes=len(label_map))
    save_json(labels_check, diagnostics_dir / "labels_check.json")

    dist_df = _class_distribution_df(train_df, valid_df, test_df, num_classes=len(label_map))
    dist_df.to_csv(diagnostics_dir / "class_distribution.csv", index=False)
    dist_summary = {
        "train_imbalance": _imbalance_summary(dist_df, "train_count"),
        "valid_imbalance": _imbalance_summary(dist_df, "valid_count"),
        "test_imbalance": _imbalance_summary(dist_df, "test_count"),
    }

    all_smiles = pd.concat(
        [
            train_df["drug_a_smiles"],
            train_df["drug_b_smiles"],
            valid_df["drug_a_smiles"],
            valid_df["drug_b_smiles"],
            test_df["drug_a_smiles"],
            test_df["drug_b_smiles"],
        ],
        axis=0,
    ).dropna().astype(str).unique().tolist()
    cache = GraphCache(output_dir=args.output_dir)
    cache_stats = cache.build(all_smiles, show_progress=False)

    train_ds = DDIPairDataset(train_df, cache, split_name="train_diag")
    valid_ds = DDIPairDataset(valid_df, cache, split_name="valid_diag")
    test_ds = DDIPairDataset(test_df, cache, split_name="test_diag")
    train_loader = make_pair_dataloader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, seed=args.seed)
    valid_loader = make_pair_dataloader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, seed=args.seed)
    test_loader = make_pair_dataloader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, seed=args.seed)

    test_eval = eval_epoch(model, test_loader, device=device, amp_enabled=True, collect_logits=True)
    metric_report = evaluate_multiclass_metrics(
        y_true=test_eval["y_true"],
        y_prob=test_eval["y_prob"],
        ece_bins=args.ece_bins,
        include_ovr_details=True,
    )
    metric_report["objective_loss"] = float(test_eval["objective_loss"])
    metric_report["nll_loss"] = float(test_eval["nll_loss"])

    y_true = test_eval["y_true"]
    y_prob = test_eval["y_prob"]
    y_pred = np.argmax(y_prob, axis=1).astype(int)
    manual_metrics = {
        "accuracy_manual": float(accuracy_score(y_true, y_pred)),
        "macro_f1_manual_present_only": float(f1_score(y_true, y_pred, average="macro", labels=np.unique(y_true), zero_division=0)),
        "micro_f1_manual_all_labels": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "kappa_manual_all_labels": float(cohen_kappa_score(y_true, y_pred)),
    }
    prob_row_sum = np.sum(y_prob, axis=1)
    prob_checks = {
        "row_sum_mean": float(np.mean(prob_row_sum)),
        "row_sum_std": float(np.std(prob_row_sum)),
        "row_sum_max_abs_err": float(np.max(np.abs(prob_row_sum - 1.0))),
    }

    metrics_sanity = {
        "reported": metric_report,
        "manual": manual_metrics,
        "probability_checks": prob_checks,
    }
    save_json(metrics_sanity, diagnostics_dir / "metrics_sanity.json")

    # Loss sanity on a single concrete batch.
    first_batch = None
    for batch in test_loader:
        if batch is not None:
            first_batch = batch
            break
    if first_batch is None:
        raise RuntimeError("No valid batch available for loss sanity check")
    first_batch = _move_batch_to_device(first_batch, device)
    with torch.no_grad():
        logits = model(first_batch["graph_a"], first_batch["graph_b"])
        objective_loss = model.loss_fn(logits, first_batch["y"])
        weight = model.class_weights.to(device) if model.class_weights is not None else None
        manual_objective = F.cross_entropy(
            logits,
            first_batch["y"],
            weight=weight,
            label_smoothing=float(model.label_smoothing),
            reduction="mean",
        )
        nll_loss = F.cross_entropy(
            logits,
            first_batch["y"],
            weight=None,
            label_smoothing=0.0,
            reduction="mean",
        )
    loss_sanity = {
        "objective_loss_batch": float(objective_loss.item()),
        "manual_objective_loss_batch": float(manual_objective.item()),
        "objective_abs_diff": float(abs(objective_loss.item() - manual_objective.item())),
        "nll_loss_batch": float(nll_loss.item()),
        "eval_objective_loss": float(test_eval["objective_loss"]),
        "eval_nll_loss": float(test_eval["nll_loss"]),
        "loss_config": loss_cfg,
    }
    save_json(loss_sanity, diagnostics_dir / "loss_sanity.json")

    # Calibration sanity.
    valid_eval = eval_epoch(model, valid_loader, device=device, amp_enabled=True, collect_logits=True)
    temperature = fit_temperature(valid_eval["y_logits"], valid_eval["y_true"], device=device, max_iter=100)
    calibrated_logits = apply_temperature(test_eval["y_logits"], temperature)
    calibrated_prob = torch.softmax(torch.tensor(calibrated_logits, dtype=torch.float32), dim=-1).numpy()
    calibrated_metrics = evaluate_multiclass_metrics(
        y_true=test_eval["y_true"],
        y_prob=calibrated_prob,
        ece_bins=args.ece_bins,
        include_ovr_details=True,
    )
    calibration_sanity = {
        "temperature": float(temperature),
        "uncalibrated": {
            "ece": float(metric_report["ece"]),
            "brier_score": float(metric_report["brier_score"]),
            "nll": float(metric_report["nll"]),
            "accuracy": float(metric_report["accuracy"]),
        },
        "calibrated": {
            "ece": float(calibrated_metrics["ece"]),
            "brier_score": float(calibrated_metrics["brier_score"]),
            "nll": float(calibrated_metrics["nll"]),
            "accuracy": float(calibrated_metrics["accuracy"]),
        },
    }
    save_json(calibration_sanity, diagnostics_dir / "calibration_sanity.json")

    # Faithfulness + randomization sanity.
    weight_rand_model = _build_weight_randomized_copy(model, seed=args.seed + 11).to(device)
    weight_rand_model.set_loss_params(class_weights=model.class_weights, label_smoothing=model.label_smoothing)
    permuted_model = _train_label_permuted_copy(
        model,
        loader=train_loader,
        device=device,
        seed=args.seed + 29,
        steps=args.randomization_steps,
        lr=args.randomization_lr,
    )
    permuted_model.set_loss_params(class_weights=model.class_weights, label_smoothing=model.label_smoothing)

    faith_rows: list[dict[str, Any]] = []
    n_done = 0
    ds_idx = 0
    while n_done < args.faithfulness_examples and ds_idx < len(test_ds):
        sample = test_ds[ds_idx]
        ds_idx += 1
        if sample is None:
            continue
        base_probs, target_class, base_attn = _predict_with_attention(model, sample.graph_a, sample.graph_b, device)
        attn_a = base_attn.get("A") if isinstance(base_attn, dict) else None
        attn_b = base_attn.get("B") if isinstance(base_attn, dict) else None
        rank_a = list(np.argsort(-_score_graph_from_attention(sample.graph_a, attn_a)))
        rank_b = list(np.argsort(-_score_graph_from_attention(sample.graph_b, attn_b)))
        k_a = _k_schedule(len(rank_a))
        k_b = _k_schedule(len(rank_b))

        for which, rank, k_list in [("A", rank_a, k_a), ("B", rank_b, k_b)]:
            if len(rank) == 0:
                continue
            del_base = deletion_test(model, sample.graph_a, sample.graph_b, target_class, rank, k_list, which=which, device=device)
            ins_base = insertion_test(model, sample.graph_a, sample.graph_b, target_class, rank, k_list, which=which, device=device)
            rnd_del_base, rnd_ins_base, rnd_unique_ratio = _random_baseline_curves(
                model,
                sample.graph_a,
                sample.graph_b,
                target_class,
                k_list,
                which=which,
                repeats=args.random_baseline_repeats,
                rng=rng,
                device=device,
            )
            auc_del_base = _curve_auc(k_list, del_base)
            auc_ins_base = _curve_auc(k_list, ins_base)
            auc_del_base_rand = _curve_auc(k_list, rnd_del_base)
            auc_ins_base_rand = _curve_auc(k_list, rnd_ins_base)
            delta_ins_base = auc_ins_base - auc_ins_base_rand
            delta_del_base = auc_del_base_rand - auc_del_base

            # Weight-randomized model sanity.
            _, _, wr_attn = _predict_with_attention(weight_rand_model, sample.graph_a, sample.graph_b, device)
            wr_rank = list(
                np.argsort(
                    -_score_graph_from_attention(sample.graph_a if which == "A" else sample.graph_b, wr_attn.get(which) if isinstance(wr_attn, dict) else None)
                )
            )
            del_wr = deletion_test(weight_rand_model, sample.graph_a, sample.graph_b, target_class, wr_rank, k_list, which=which, device=device)
            ins_wr = insertion_test(weight_rand_model, sample.graph_a, sample.graph_b, target_class, wr_rank, k_list, which=which, device=device)
            rnd_del_wr, rnd_ins_wr, _ = _random_baseline_curves(
                weight_rand_model,
                sample.graph_a,
                sample.graph_b,
                target_class,
                k_list,
                which=which,
                repeats=max(2, args.random_baseline_repeats),
                rng=rng,
                device=device,
            )
            delta_ins_wr = _curve_auc(k_list, ins_wr) - _curve_auc(k_list, rnd_ins_wr)
            delta_del_wr = _curve_auc(k_list, rnd_del_wr) - _curve_auc(k_list, del_wr)

            # Label-permuted fine-tuned model sanity.
            _, _, pm_attn = _predict_with_attention(permuted_model, sample.graph_a, sample.graph_b, device)
            pm_rank = list(
                np.argsort(
                    -_score_graph_from_attention(sample.graph_a if which == "A" else sample.graph_b, pm_attn.get(which) if isinstance(pm_attn, dict) else None)
                )
            )
            del_pm = deletion_test(permuted_model, sample.graph_a, sample.graph_b, target_class, pm_rank, k_list, which=which, device=device)
            ins_pm = insertion_test(permuted_model, sample.graph_a, sample.graph_b, target_class, pm_rank, k_list, which=which, device=device)
            rnd_del_pm, rnd_ins_pm, _ = _random_baseline_curves(
                permuted_model,
                sample.graph_a,
                sample.graph_b,
                target_class,
                k_list,
                which=which,
                repeats=max(2, args.random_baseline_repeats),
                rng=rng,
                device=device,
            )
            delta_ins_pm = _curve_auc(k_list, ins_pm) - _curve_auc(k_list, rnd_ins_pm)
            delta_del_pm = _curve_auc(k_list, rnd_del_pm) - _curve_auc(k_list, del_pm)

            topk = min(10, len(rank), len(wr_rank), len(pm_rank))
            row = {
                "pair_id": sample.pair_id,
                "which": which,
                "target_class": int(target_class),
                "pred_prob_target": float(base_probs[target_class]),
                "delta_ins_base": float(delta_ins_base),
                "delta_del_base": float(delta_del_base),
                "delta_ins_weight_randomized": float(delta_ins_wr),
                "delta_del_weight_randomized": float(delta_del_wr),
                "delta_ins_label_permuted": float(delta_ins_pm),
                "delta_del_label_permuted": float(delta_del_pm),
                "random_baseline_unique_ratio": float(rnd_unique_ratio),
                "rank_jaccard_weight_randomized": float(_topk_jaccard(rank, wr_rank, k=max(topk, 1))),
                "rank_jaccard_label_permuted": float(_topk_jaccard(rank, pm_rank, k=max(topk, 1))),
                "orientation_ok_base": bool(delta_ins_base >= -1.0 and delta_del_base >= -1.0),
            }
            faith_rows.append(row)
        n_done += 1

    faith_df = pd.DataFrame(faith_rows)
    faith_df.to_csv(diagnostics_dir / "faithfulness_sanity.csv", index=False)

    randomization_summary = {
        "n_rows": int(len(faith_df)),
        "means": {
            "delta_ins_base": float(faith_df["delta_ins_base"].mean()) if len(faith_df) else float("nan"),
            "delta_del_base": float(faith_df["delta_del_base"].mean()) if len(faith_df) else float("nan"),
            "delta_ins_weight_randomized": float(faith_df["delta_ins_weight_randomized"].mean()) if len(faith_df) else float("nan"),
            "delta_del_weight_randomized": float(faith_df["delta_del_weight_randomized"].mean()) if len(faith_df) else float("nan"),
            "delta_ins_label_permuted": float(faith_df["delta_ins_label_permuted"].mean()) if len(faith_df) else float("nan"),
            "delta_del_label_permuted": float(faith_df["delta_del_label_permuted"].mean()) if len(faith_df) else float("nan"),
            "rank_jaccard_weight_randomized": float(faith_df["rank_jaccard_weight_randomized"].mean()) if len(faith_df) else float("nan"),
            "rank_jaccard_label_permuted": float(faith_df["rank_jaccard_label_permuted"].mean()) if len(faith_df) else float("nan"),
            "random_baseline_unique_ratio": float(faith_df["random_baseline_unique_ratio"].mean()) if len(faith_df) else float("nan"),
        },
    }
    if len(faith_df):
        randomization_summary["collapse_checks"] = {
            "weight_randomization_faithfulness_collapse": bool(
                randomization_summary["means"]["delta_ins_weight_randomized"] < randomization_summary["means"]["delta_ins_base"]
                and randomization_summary["means"]["delta_del_weight_randomized"] < randomization_summary["means"]["delta_del_base"]
            ),
            "label_permutation_faithfulness_collapse": bool(
                randomization_summary["means"]["delta_ins_label_permuted"] < randomization_summary["means"]["delta_ins_base"]
                and randomization_summary["means"]["delta_del_label_permuted"] < randomization_summary["means"]["delta_del_base"]
            ),
            "weight_randomization_similarity_collapse": bool(
                randomization_summary["means"]["rank_jaccard_weight_randomized"] < 0.5
            ),
            "label_permutation_similarity_collapse": bool(
                randomization_summary["means"]["rank_jaccard_label_permuted"] < 0.5
            ),
        }
    save_json(randomization_summary, diagnostics_dir / "randomization_sanity.json")

    diagnostic_summary = {
        "config": {
            "seed": int(args.seed),
            "limit": int(args.limit) if args.limit is not None else None,
            "split_strategy": args.split_strategy,
            "split_seed": int(args.split_seed),
            "device": str(device),
        },
        "cache_stats": cache_stats,
        "label_checks": labels_check,
        "distribution_summary": dist_summary,
        "metrics_key": {
            "accuracy": float(metric_report["accuracy"]),
            "macro_f1": float(metric_report["macro_f1"]),
            "micro_f1": float(metric_report["micro_f1"]),
            "cohen_kappa": float(metric_report["cohen_kappa"]),
            "macro_roc_auc_ovr": float(metric_report["macro_roc_auc_ovr"]),
            "macro_pr_auc_ovr": float(metric_report["macro_pr_auc_ovr"]),
            "ece": float(metric_report["ece"]),
            "brier_score": float(metric_report["brier_score"]),
            "objective_loss": float(metric_report["objective_loss"]),
            "nll_loss": float(metric_report["nll_loss"]),
        },
        "manual_metric_checks": manual_metrics,
        "probability_checks": prob_checks,
        "calibration": calibration_sanity,
        "randomization": randomization_summary,
    }
    save_json(diagnostic_summary, diagnostics_dir / "diagnostic_summary.json")

    print(f"diagnostics_dir={diagnostics_dir}")
    print(f"labels_check={diagnostics_dir / 'labels_check.json'}")
    print(f"class_distribution={diagnostics_dir / 'class_distribution.csv'}")
    print(f"loss_sanity={diagnostics_dir / 'loss_sanity.json'}")
    print(f"metrics_sanity={diagnostics_dir / 'metrics_sanity.json'}")
    print(f"calibration_sanity={diagnostics_dir / 'calibration_sanity.json'}")
    print(f"faithfulness_sanity={diagnostics_dir / 'faithfulness_sanity.csv'}")
    print(f"randomization_sanity={diagnostics_dir / 'randomization_sanity.json'}")
    print(f"diagnostic_summary={diagnostics_dir / 'diagnostic_summary.json'}")


if __name__ == "__main__":
    main()
