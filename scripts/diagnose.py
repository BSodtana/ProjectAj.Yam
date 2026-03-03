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

from ddigat.data.cache import DrugFeatureCache, GraphCache
from ddigat.data.drug_features import PHYS_CHEM_DESCRIPTOR_NAMES, canonical_smiles_digest
from ddigat.data.splits import DDIPairDataset, make_pair_dataloader, subsample_dataframe
from ddigat.data.tdc_ddi import load_tdc_drugbank_ddi
from ddigat.explain.attention import get_node_importance_from_attention
from ddigat.explain.faithfulness import deletion_test, insertion_test
from ddigat.model.pair_model import DDIPairModel
from ddigat.train.loop import eval_epoch
from ddigat.utils.calibration import apply_temperature, fit_temperature
from ddigat.utils.class_weights import (
    assert_class_weight_sanity,
    class_counts_payload,
    compute_class_priors,
    compute_tail_class_ids,
    class_weights_payload,
    compute_class_counts,
    compute_class_weights,
)
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
    p.add_argument("--use_ecfp_features", action="store_true")
    p.add_argument("--use_physchem_features", action="store_true")
    p.add_argument("--use_maccs_features", action="store_true")
    p.add_argument("--ecfp_bits", type=int, default=2048)
    p.add_argument("--ecfp_radius", type=int, default=2)
    p.add_argument("--physchem_dim", type=int, default=0, help="0=auto from checkpoint/extractor")
    p.add_argument("--use_class_weights", action="store_true")
    p.add_argument("--class_weight_method", type=str, choices=["inv_sqrt", "effective_num"], default=None)
    p.add_argument("--class_weight_normalize", type=str, choices=["sample_mean", "mean_seen", "none"], default=None)
    p.add_argument("--class_weight_beta", type=float, default=None)
    p.add_argument("--class_weight_clip_min", type=float, default=None)
    p.add_argument("--class_weight_clip_max", type=float, default=None)
    p.add_argument("--label_smoothing", type=float, default=None)
    p.add_argument("--logit_adjust_tau", type=float, default=None)
    p.add_argument("--split_strategy", type=str, default="cold_drug", choices=["cold_drug", "cold_drug_v2", "tdc"])
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--cold_k", type=int, default=5)
    p.add_argument("--cold_fold", type=int, default=0)
    p.add_argument("--cold_protocol", type=str, default="s1", choices=["s1", "s2"])
    p.add_argument("--cold_min_test_pairs", type=int, default=5000)
    p.add_argument("--cold_min_test_labels", type=int, default=45)
    p.add_argument("--cold_max_resamples", type=int, default=200)
    p.add_argument("--cold_dedupe_policy", type=str, default="keep_all", choices=["keep_all", "keep_first"])
    p.add_argument("--cold_write_legacy_flat_splits", action="store_true")
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


def _resolve_feature_config(model_cfg: dict, args: argparse.Namespace) -> dict[str, int | bool]:
    use_ecfp_features = bool(model_cfg.get("use_ecfp_features", False) or args.use_ecfp_features)
    use_physchem_features = bool(model_cfg.get("use_physchem_features", False) or args.use_physchem_features)
    use_maccs_features = bool(model_cfg.get("use_maccs_features", False) or args.use_maccs_features)
    ecfp_bits = int(model_cfg.get("ecfp_bits", args.ecfp_bits))
    ecfp_radius = int(model_cfg.get("ecfp_radius", args.ecfp_radius))
    default_physchem_dim = int(args.physchem_dim) if int(args.physchem_dim) > 0 else int(len(PHYS_CHEM_DESCRIPTOR_NAMES))
    physchem_dim = int(model_cfg.get("physchem_dim", default_physchem_dim))
    if use_physchem_features and physchem_dim <= 0:
        physchem_dim = int(len(PHYS_CHEM_DESCRIPTOR_NAMES))
    maccs_dim = int(model_cfg.get("maccs_dim", 166))
    return {
        "use_ecfp_features": use_ecfp_features,
        "use_physchem_features": use_physchem_features,
        "use_maccs_features": use_maccs_features,
        "ecfp_bits": ecfp_bits,
        "ecfp_radius": ecfp_radius,
        "physchem_dim": physchem_dim,
        "maccs_dim": maccs_dim,
    }


def build_model_from_checkpoint_payload(payload: dict, device: torch.device, feature_cfg: dict[str, int | bool]) -> DDIPairModel:
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
        use_ecfp_features=bool(feature_cfg["use_ecfp_features"]),
        use_physchem_features=bool(feature_cfg["use_physchem_features"]),
        use_maccs_features=bool(feature_cfg["use_maccs_features"]),
        ecfp_bits=int(feature_cfg["ecfp_bits"]),
        physchem_dim=int(feature_cfg["physchem_dim"]),
        maccs_dim=int(feature_cfg["maccs_dim"]),
        ecfp_proj_dim=int(cfg.get("ecfp_proj_dim", 128)),
        physchem_proj_dim=int(cfg.get("physchem_proj_dim", 32)),
        maccs_proj_dim=int(cfg.get("maccs_proj_dim", 32)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def restore_loss_params_from_checkpoint(
    model: DDIPairModel,
    payload: dict,
    train_df: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cfg = payload.get("config", {}) or {}
    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    use_class_weights = bool(train_cfg.get("use_class_weights", False)) or bool(args.use_class_weights)
    class_weight_method = str(train_cfg.get("class_weight_method", "inv_sqrt"))
    if args.class_weight_method is not None:
        class_weight_method = str(args.class_weight_method)
    class_weight_normalize = str(train_cfg.get("class_weight_normalize", "sample_mean"))
    if args.class_weight_normalize is not None:
        class_weight_normalize = str(args.class_weight_normalize)
    class_weight_beta = float(train_cfg.get("class_weight_beta", 0.9999))
    if args.class_weight_beta is not None:
        class_weight_beta = float(args.class_weight_beta)
    class_weight_clip_min = float(train_cfg.get("class_weight_clip_min", 0.25))
    if args.class_weight_clip_min is not None:
        class_weight_clip_min = float(args.class_weight_clip_min)
    class_weight_clip_max = float(train_cfg.get("class_weight_clip_max", 4.0))
    if args.class_weight_clip_max is not None:
        class_weight_clip_max = float(args.class_weight_clip_max)
    class_weight_eps = float(train_cfg.get("class_weight_eps", 1e-12))
    class_counts_cfg = train_cfg.get("class_counts", None)
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    if args.label_smoothing is not None:
        label_smoothing = float(args.label_smoothing)
    logit_adjust_tau = float(train_cfg.get("logit_adjust_tau", 0.0))
    if args.logit_adjust_tau is not None:
        logit_adjust_tau = float(args.logit_adjust_tau)
    if float(logit_adjust_tau) > 0.0 and str(args.split_strategy) not in {"cold_drug", "cold_drug_v2"}:
        raise ValueError("Logit adjustment diagnostics expect split_strategy in {'cold_drug', 'cold_drug_v2'}.")
    logit_adjust_eps = float(train_cfg.get("logit_adjust_eps", 1e-12))

    if isinstance(class_counts_cfg, list) and len(class_counts_cfg) == int(model.num_classes):
        class_counts = np.asarray([int(v) for v in class_counts_cfg], dtype=np.int64)
    else:
        class_counts = compute_class_counts(train_df["y"].to_numpy(dtype=int), num_classes=model.num_classes)
    _, log_priors = compute_class_priors(class_counts, eps=logit_adjust_eps)
    tail_ids = compute_tail_class_ids(class_counts, fraction=0.2, include_zero_count=True)
    class_weight_info = compute_class_weights(
        class_counts,
        method=class_weight_method,
        normalize=class_weight_normalize,
        beta=class_weight_beta,
        eps=class_weight_eps,
        clip_min=class_weight_clip_min,
        clip_max=class_weight_clip_max,
    )
    assert_class_weight_sanity(
        class_weight_info.weights,
        num_classes=model.num_classes,
        clip_min=class_weight_clip_min,
        clip_max=class_weight_clip_max,
        mean_after_normalization=class_weight_info.mean_after_normalization,
        counts=class_counts,
        normalize=class_weight_normalize,
    )
    class_weights = None
    if use_class_weights:
        class_weights = class_weight_info.weights
    model.set_loss_params(
        class_weights=class_weights,
        label_smoothing=label_smoothing,
        logit_adjust_tau=logit_adjust_tau,
        logit_adjust_log_pi=torch.tensor(log_priors, dtype=torch.float32),
    )
    serializable_cfg = {
        "use_class_weights": use_class_weights,
        "class_weight_method": class_weight_method,
        "class_weight_normalize": class_weight_normalize,
        "class_weight_beta": class_weight_beta,
        "class_weight_clip_min": class_weight_clip_min,
        "class_weight_clip_max": class_weight_clip_max,
        "class_weight_eps": class_weight_eps,
        "label_smoothing": label_smoothing,
        "logit_adjust_tau": float(logit_adjust_tau),
        "logit_adjust_eps": float(logit_adjust_eps),
        "tail_k": int(tail_ids.size),
        "tail_class_ids": [int(v) for v in tail_ids.tolist()],
        "class_weight_min": None if class_weights is None else float(class_weights.min().item()),
        "class_weight_max": None if class_weights is None else float(class_weights.max().item()),
    }
    serializable_cfg["class_counts"] = [int(v) for v in class_counts.tolist()]
    serializable_cfg["mean_after_normalization"] = float(class_weight_info.mean_after_normalization)
    serializable_cfg["mean_after_clipping"] = float(class_weight_info.mean_after_clipping)
    return {
        **serializable_cfg,
        "class_counts_np": class_counts,
        "class_weight_info_obj": class_weight_info,
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
    if "feat_a" in batch:
        batch["feat_a"] = batch["feat_a"].to(device)
    if "feat_b" in batch:
        batch["feat_b"] = batch["feat_b"].to(device)
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
def _predict_with_attention(model: DDIPairModel, graph_a, graph_b, device: torch.device, feat_a=None, feat_b=None):
    fa = None if feat_a is None else feat_a.to(device)
    fb = None if feat_b is None else feat_b.to(device)
    logits, attn = model.forward_with_attention(graph_a.to(device), graph_b.to(device), fa, fb)
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
        logits = model(batch["graph_a"], batch["graph_b"], batch.get("feat_a"), batch.get("feat_b"))
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


def _feature_slices(feature_cfg: dict[str, int | bool]) -> dict[str, tuple[int, int]]:
    slices: dict[str, tuple[int, int]] = {}
    offset = 0
    if bool(feature_cfg["use_ecfp_features"]):
        n = int(feature_cfg["ecfp_bits"])
        slices["ecfp"] = (offset, offset + n)
        offset += n
    if bool(feature_cfg["use_physchem_features"]):
        n = int(feature_cfg["physchem_dim"])
        slices["physchem"] = (offset, offset + n)
        offset += n
    if bool(feature_cfg["use_maccs_features"]):
        n = int(feature_cfg["maccs_dim"])
        slices["maccs"] = (offset, offset + n)
        offset += n
    return slices


def _scan_feature_loader(
    loader,
    expected_dim: int,
    physchem_slice: tuple[int, int] | None,
    ecfp_slice: tuple[int, int] | None,
) -> dict[str, Any]:
    out = {
        "n_batches": 0,
        "n_samples": 0,
        "shape_ok": True,
        "first_batch_shape": None,
        "physchem_all_finite": True,
        "physchem_value_mean": float("nan"),
        "physchem_value_std": float("nan"),
        "physchem_dim_std_min": float("nan"),
        "physchem_dim_std_median": float("nan"),
        "physchem_dim_std_max": float("nan"),
        "physchem_dim_std_lt_1e3_fraction": float("nan"),
        "ecfp_bit_density": float("nan"),
        "ecfp_all_zero_fraction": float("nan"),
    }
    phys_sum = 0.0
    phys_sq_sum = 0.0
    phys_count = 0
    phys_dim_sum: np.ndarray | None = None
    phys_dim_sq_sum: np.ndarray | None = None
    phys_dim_count = 0
    ecfp_sum = 0.0
    ecfp_count = 0
    ecfp_all_zero = 0
    ecfp_molecules = 0
    for batch in loader:
        if batch is None:
            continue
        fa = batch["feat_a"]
        fb = batch["feat_b"]
        out["n_batches"] += 1
        out["n_samples"] += int(fa.size(0))
        if out["first_batch_shape"] is None:
            out["first_batch_shape"] = {
                "feat_a": [int(v) for v in fa.shape],
                "feat_b": [int(v) for v in fb.shape],
            }
        if fa.dim() != 2 or fb.dim() != 2:
            out["shape_ok"] = False
        if int(fa.size(1)) != int(expected_dim) or int(fb.size(1)) != int(expected_dim):
            out["shape_ok"] = False
        if physchem_slice is not None:
            start, end = physchem_slice
            if end > start:
                pa = fa[:, start:end]
                pb = fb[:, start:end]
                if not bool(torch.isfinite(pa).all().item()) or not bool(torch.isfinite(pb).all().item()):
                    out["physchem_all_finite"] = False
                pa_np = pa.detach().cpu().numpy().astype(np.float64, copy=False)
                pb_np = pb.detach().cpu().numpy().astype(np.float64, copy=False)
                vals_2d = np.concatenate([pa_np, pb_np], axis=0)
                vals = vals_2d.reshape(-1)
                phys_sum += float(np.sum(vals))
                phys_sq_sum += float(np.sum(vals * vals))
                phys_count += int(vals.size)
                if phys_dim_sum is None:
                    phys_dim_sum = np.zeros((int(vals_2d.shape[1]),), dtype=np.float64)
                    phys_dim_sq_sum = np.zeros((int(vals_2d.shape[1]),), dtype=np.float64)
                assert phys_dim_sq_sum is not None
                phys_dim_sum += np.sum(vals_2d, axis=0)
                phys_dim_sq_sum += np.sum(vals_2d * vals_2d, axis=0)
                phys_dim_count += int(vals_2d.shape[0])
        if ecfp_slice is not None:
            start, end = ecfp_slice
            if end > start:
                ea = fa[:, start:end]
                eb = fb[:, start:end]
                ea_np = ea.detach().cpu().numpy().astype(np.float64, copy=False)
                eb_np = eb.detach().cpu().numpy().astype(np.float64, copy=False)
                bits = np.concatenate([ea_np.reshape(-1), eb_np.reshape(-1)], axis=0)
                ecfp_sum += float(np.sum(bits))
                ecfp_count += int(bits.size)
                ea_row_sum = np.sum(ea_np, axis=1)
                eb_row_sum = np.sum(eb_np, axis=1)
                ecfp_all_zero += int(np.sum(ea_row_sum <= 0.0) + np.sum(eb_row_sum <= 0.0))
                ecfp_molecules += int(ea_np.shape[0] + eb_np.shape[0])
    if phys_count > 0:
        mu = phys_sum / float(phys_count)
        var = max((phys_sq_sum / float(phys_count)) - (mu * mu), 0.0)
        out["physchem_value_mean"] = float(mu)
        out["physchem_value_std"] = float(np.sqrt(var))
    if phys_dim_count > 0 and phys_dim_sum is not None and phys_dim_sq_sum is not None:
        dim_mu = phys_dim_sum / float(phys_dim_count)
        dim_var = np.maximum((phys_dim_sq_sum / float(phys_dim_count)) - (dim_mu * dim_mu), 0.0)
        dim_std = np.sqrt(dim_var)
        out["physchem_dim_std_min"] = float(np.min(dim_std))
        out["physchem_dim_std_median"] = float(np.median(dim_std))
        out["physchem_dim_std_max"] = float(np.max(dim_std))
        out["physchem_dim_std_lt_1e3_fraction"] = float(np.mean(dim_std < 1e-3))
    if ecfp_count > 0:
        out["ecfp_bit_density"] = float(ecfp_sum / float(ecfp_count))
    if ecfp_molecules > 0:
        out["ecfp_all_zero_fraction"] = float(ecfp_all_zero / float(ecfp_molecules))
    return out


def _cache_hit_rates(stats: dict[str, int], feature_cfg: dict[str, int | bool]) -> dict[str, float]:
    out: dict[str, float] = {}
    for kind, enabled_key in [("ecfp", "use_ecfp_features"), ("physchem", "use_physchem_features"), ("maccs", "use_maccs_features")]:
        if not bool(feature_cfg[enabled_key]):
            continue
        hits = int(stats.get(f"{kind}_hits", 0))
        misses = int(stats.get(f"{kind}_misses", 0))
        denom = max(hits + misses, 1)
        out[f"{kind}_hit_rate"] = float(hits / denom)
    return out


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)

    diagnostics_dir = ensure_dir(Path(args.output_dir) / "diagnostics")

    payload = torch_load(args.checkpoint, map_location=device)
    cfg = payload.get("config", {}) or {}
    train_cfg_payload = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    training_start_unix = None
    if isinstance(train_cfg_payload, dict) and train_cfg_payload.get("training_start_unix") is not None:
        try:
            training_start_unix = float(train_cfg_payload.get("training_start_unix"))
        except Exception:
            training_start_unix = None
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    feature_cfg = _resolve_feature_config(model_cfg if isinstance(model_cfg, dict) else {}, args)
    model = build_model_from_checkpoint_payload(payload, device=device, feature_cfg=feature_cfg)

    train_df, valid_df, test_df, label_map = load_tdc_drugbank_ddi(
        args.data_dir,
        output_dir=args.output_dir,
        split_strategy=args.split_strategy,
        split_seed=args.split_seed,
        cold_k=args.cold_k,
        cold_fold=args.cold_fold,
        cold_protocol=args.cold_protocol,
        cold_min_test_pairs=args.cold_min_test_pairs,
        cold_min_test_labels=args.cold_min_test_labels,
        cold_max_resamples=args.cold_max_resamples,
        cold_dedupe_policy=args.cold_dedupe_policy,
        cold_write_legacy_flat_splits=bool(args.cold_write_legacy_flat_splits),
    )
    if args.limit is not None:
        train_df = subsample_dataframe(train_df, limit=args.limit, seed=args.seed, label_col="y", ensure_class_coverage=True)
        valid_df = subsample_dataframe(valid_df, limit=args.limit, seed=args.seed + 1, label_col="y", ensure_class_coverage=True)
        test_df = subsample_dataframe(test_df, limit=args.limit, seed=args.seed + 2, label_col="y", ensure_class_coverage=True)

    loss_cfg = restore_loss_params_from_checkpoint(model, payload, train_df=train_df, args=args)
    LOGGER.info("Restored eval objective: %s", loss_cfg)
    save_json(
        class_counts_payload(loss_cfg["class_counts_np"], num_classes=model.num_classes),
        diagnostics_dir / "class_counts.json",
    )
    save_json(
        class_weights_payload(
            enabled=bool(loss_cfg["use_class_weights"]),
            method=str(loss_cfg["class_weight_method"]),
            beta=float(loss_cfg["class_weight_beta"]),
            eps=float(loss_cfg["class_weight_eps"]),
            clip_min=float(loss_cfg["class_weight_clip_min"]),
            clip_max=float(loss_cfg["class_weight_clip_max"]),
            computation=loss_cfg["class_weight_info_obj"],
        ),
        diagnostics_dir / "class_weights.json",
    )

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
    feature_cache: DrugFeatureCache | None = None
    feature_scaler_stats: dict[str, Any] | None = None
    scaler_std_stats: dict[str, float] | None = None
    expected_train_scaler_digest: str | None = None
    scaler_mtime_before: float | None = None
    if bool(
        feature_cfg["use_ecfp_features"] or feature_cfg["use_physchem_features"] or feature_cfg["use_maccs_features"]
    ):
        feature_cache = DrugFeatureCache(
            output_dir=args.output_dir,
            use_ecfp=bool(feature_cfg["use_ecfp_features"]),
            use_physchem=bool(feature_cfg["use_physchem_features"]),
            use_maccs=bool(feature_cfg["use_maccs_features"]),
            ecfp_bits=int(feature_cfg["ecfp_bits"]),
            ecfp_radius=int(feature_cfg["ecfp_radius"]),
        )
        if bool(feature_cfg["use_physchem_features"]):
            train_smiles_for_scaler = list(train_df["drug_a_smiles"]) + list(train_df["drug_b_smiles"])
            expected_train_scaler_digest = canonical_smiles_digest(train_smiles_for_scaler)
            if not feature_cache.scaler_path.exists():
                raise FileNotFoundError(
                    f"Missing physchem scaler at {feature_cache.scaler_path}. "
                    "Train with --use_physchem_features first; diagnostics does not recompute scaler."
                )
            feature_scaler_stats = feature_cache.load_physchem_stats()
            if feature_scaler_stats is None:
                raise RuntimeError(f"Failed to load physchem scaler stats from {feature_cache.scaler_path}")
            feature_cache.set_physchem_stats(feature_scaler_stats)
            scaler_dim = int(feature_scaler_stats.get("dim", feature_cache.physchem_dim))
            if scaler_dim != int(feature_cfg["physchem_dim"]):
                raise ValueError(
                    f"Physchem dim mismatch between checkpoint config ({feature_cfg['physchem_dim']}) and scaler ({scaler_dim})"
                )
            scaler_mean = np.asarray(feature_scaler_stats.get("mean", []), dtype=np.float64).reshape(-1)
            scaler_std = np.asarray(feature_scaler_stats.get("std", []), dtype=np.float64).reshape(-1)
            if scaler_mean.size == 0 or scaler_std.size == 0 or scaler_mean.size != scaler_std.size:
                raise ValueError("Physchem scaler file has invalid mean/std vectors")
            scaler_std_floor = float(feature_scaler_stats.get("std_floor", 1e-6))
            if float(np.min(scaler_std)) < (scaler_std_floor - 1e-12):
                raise AssertionError(
                    f"Physchem scaler std below floor detected: min_std={float(np.min(scaler_std)):.6e}, floor={scaler_std_floor:.6e}"
                )
            scaler_std_stats = {
                "std_floor": float(scaler_std_floor),
                "std_min": float(np.min(scaler_std)),
                "std_median": float(np.median(scaler_std)),
                "std_max": float(np.max(scaler_std)),
                "std_lt_1e3_fraction": float(np.mean(scaler_std < 1e-3)),
            }
            scaler_mtime_before = float(feature_cache.scaler_path.stat().st_mtime)
            if training_start_unix is not None and scaler_mtime_before > (training_start_unix + 1e-6):
                raise AssertionError(
                    "Physchem scaler timestamp is newer than checkpoint training_start_unix; expected train-only pre-fit scaler artifact"
                )
        feature_cache.build(all_smiles, show_progress=False)

    train_ds = DDIPairDataset(train_df, cache, feature_cache=feature_cache, split_name="train_diag")
    valid_ds = DDIPairDataset(valid_df, cache, feature_cache=feature_cache, split_name="valid_diag")
    test_ds = DDIPairDataset(test_df, cache, feature_cache=feature_cache, split_name="test_diag")
    train_loader = make_pair_dataloader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, seed=args.seed)
    valid_loader = make_pair_dataloader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, seed=args.seed)
    test_loader = make_pair_dataloader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, seed=args.seed)

    feature_checks: dict[str, Any] = {
        "enabled": bool(feature_cache is not None and feature_cache.enabled),
        "expected_feature_dim": int(0 if feature_cache is None else feature_cache.feature_dim),
    }
    if feature_cache is not None and feature_cache.enabled:
        slices = _feature_slices(feature_cfg)
        physchem_slice = slices.get("physchem")
        ecfp_slice = slices.get("ecfp")
        feature_checks["slices"] = {k: [int(v[0]), int(v[1])] for k, v in slices.items()}
        feature_checks["loader_scans"] = {
            "train": _scan_feature_loader(
                train_loader,
                expected_dim=feature_cache.feature_dim,
                physchem_slice=physchem_slice,
                ecfp_slice=ecfp_slice,
            ),
            "valid": _scan_feature_loader(
                valid_loader,
                expected_dim=feature_cache.feature_dim,
                physchem_slice=physchem_slice,
                ecfp_slice=ecfp_slice,
            ),
            "test": _scan_feature_loader(
                test_loader,
                expected_dim=feature_cache.feature_dim,
                physchem_slice=physchem_slice,
                ecfp_slice=ecfp_slice,
            ),
        }
        feature_checks["all_shapes_ok"] = bool(
            feature_checks["loader_scans"]["train"]["shape_ok"]
            and feature_checks["loader_scans"]["valid"]["shape_ok"]
            and feature_checks["loader_scans"]["test"]["shape_ok"]
        )
        feature_checks["physchem_all_finite"] = bool(
            feature_checks["loader_scans"]["train"]["physchem_all_finite"]
            and feature_checks["loader_scans"]["valid"]["physchem_all_finite"]
            and feature_checks["loader_scans"]["test"]["physchem_all_finite"]
        )
        if bool(feature_cfg["use_physchem_features"]):
            train_scan = feature_checks["loader_scans"]["train"]
            valid_scan = feature_checks["loader_scans"]["valid"]
            test_scan = feature_checks["loader_scans"]["test"]
            train_phys_mean = float(train_scan["physchem_value_mean"])
            train_phys_std = float(train_scan["physchem_value_std"])
            feature_checks["physchem_distribution"] = {
                "train_mean": train_phys_mean,
                "train_std": train_phys_std,
                "valid_mean": float(valid_scan["physchem_value_mean"]),
                "valid_std": float(valid_scan["physchem_value_std"]),
                "test_mean": float(test_scan["physchem_value_mean"]),
                "test_std": float(test_scan["physchem_value_std"]),
                "train_dim_std_min": float(train_scan["physchem_dim_std_min"]),
                "train_dim_std_median": float(train_scan["physchem_dim_std_median"]),
                "train_dim_std_max": float(train_scan["physchem_dim_std_max"]),
                "train_dim_std_lt_1e3_fraction": float(train_scan["physchem_dim_std_lt_1e3_fraction"]),
                "valid_dim_std_min": float(valid_scan["physchem_dim_std_min"]),
                "valid_dim_std_median": float(valid_scan["physchem_dim_std_median"]),
                "valid_dim_std_max": float(valid_scan["physchem_dim_std_max"]),
                "valid_dim_std_lt_1e3_fraction": float(valid_scan["physchem_dim_std_lt_1e3_fraction"]),
                "test_dim_std_min": float(test_scan["physchem_dim_std_min"]),
                "test_dim_std_median": float(test_scan["physchem_dim_std_median"]),
                "test_dim_std_max": float(test_scan["physchem_dim_std_max"]),
                "test_dim_std_lt_1e3_fraction": float(test_scan["physchem_dim_std_lt_1e3_fraction"]),
            }
            if abs(train_phys_mean) > 0.20:
                raise AssertionError(
                    f"Train physchem scaled mean too far from 0: mean={train_phys_mean:.6f}"
                )
            if not (0.50 <= train_phys_std <= 1.50):
                raise AssertionError(
                    f"Train physchem scaled std out of broad sanity range [0.50, 1.50]: std={train_phys_std:.6f}"
                )
        if bool(feature_cfg["use_ecfp_features"]):
            feature_checks["ecfp_density"] = {
                "train_bit_density": float(feature_checks["loader_scans"]["train"]["ecfp_bit_density"]),
                "valid_bit_density": float(feature_checks["loader_scans"]["valid"]["ecfp_bit_density"]),
                "test_bit_density": float(feature_checks["loader_scans"]["test"]["ecfp_bit_density"]),
                "train_all_zero_fraction": float(feature_checks["loader_scans"]["train"]["ecfp_all_zero_fraction"]),
                "valid_all_zero_fraction": float(feature_checks["loader_scans"]["valid"]["ecfp_all_zero_fraction"]),
                "test_all_zero_fraction": float(feature_checks["loader_scans"]["test"]["ecfp_all_zero_fraction"]),
            }
            train_density = float(feature_checks["ecfp_density"]["train_bit_density"])
            if not (0.001 <= train_density <= 0.20):
                raise AssertionError(
                    f"Train ECFP bit density out of sanity range [0.001, 0.20]: density={train_density:.6f}"
                )
        if bool(feature_cfg["use_physchem_features"]):
            scaler_digest = None if feature_scaler_stats is None else str(feature_scaler_stats.get("canonical_smiles_sha256"))
            scaler_source = None if feature_scaler_stats is None else str(feature_scaler_stats.get("source_split"))
            observed_matches_train_digest = bool(
                expected_train_scaler_digest is not None and scaler_digest == expected_train_scaler_digest
            )
            observed_matches_all_digest = bool(scaler_digest == canonical_smiles_digest(all_smiles))
            train_only_no_leakage = bool(scaler_source == "train" and not observed_matches_all_digest)
            if training_start_unix is None:
                raise AssertionError(
                    "Checkpoint config missing train.training_start_unix; cannot verify scaler mtime against training start"
                )
            timestamp_ok_vs_training_start = bool(
                scaler_mtime_before is not None
                and scaler_mtime_before <= (training_start_unix + 1e-6)
            )
            feature_checks["scaler"] = {
                "source_split": scaler_source,
                "expected_train_digest": expected_train_scaler_digest,
                "observed_digest": scaler_digest,
                "matches_current_train_digest": observed_matches_train_digest,
                "matches_all_split_digest": observed_matches_all_digest,
                "train_only_no_leakage": train_only_no_leakage,
                "path": str(feature_cache.scaler_path),
                "mtime_unix_before_diag": scaler_mtime_before,
                "checkpoint_training_start_unix": training_start_unix,
                "mtime_leq_training_start": timestamp_ok_vs_training_start,
                "std_stats": scaler_std_stats,
            }
            if not feature_checks["scaler"]["train_only_no_leakage"]:
                raise AssertionError("Physchem scaler metadata does not match train-only digest/source requirements")
            if not timestamp_ok_vs_training_start:
                raise AssertionError(
                    "Physchem scaler mtime must be <= training_start_unix recorded in checkpoint config"
                )
        if not feature_checks["all_shapes_ok"]:
            raise AssertionError("Feature tensor shape check failed in one or more loaders")
        if bool(feature_cfg["use_physchem_features"]) and not feature_checks["physchem_all_finite"]:
            raise AssertionError("Non-finite physchem features found after standardization")

    test_eval = eval_epoch(
        model,
        test_loader,
        device=device,
        amp_enabled=True,
        collect_logits=True,
        train_class_counts=loss_cfg["class_counts_np"],
    )
    metric_report = evaluate_multiclass_metrics(
        y_true=test_eval["y_true"],
        y_prob=test_eval["y_prob"],
        ece_bins=args.ece_bins,
        include_ovr_details=True,
        train_class_counts=loss_cfg["class_counts_np"],
    )
    metric_report["macro_f1_present_only"] = float(metric_report["macro_f1"])
    metric_report["kappa"] = float(metric_report["cohen_kappa"])
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
    if prob_checks["row_sum_max_abs_err"] > 1e-5:
        raise AssertionError(
            f"Softmax probability rows do not sum to ~1 (max_abs_err={prob_checks['row_sum_max_abs_err']:.6e})"
        )

    metrics_sanity = {
        "reported": metric_report,
        "manual": manual_metrics,
        "probability_checks": prob_checks,
    }
    save_json(metrics_sanity, diagnostics_dir / "metrics_sanity.json")

    # Loss sanity on a concrete batch with weight diversity when possible.
    first_batch = None
    has_weight_diversity = False
    for batch in test_loader:
        if batch is None:
            continue
        if model.class_weights is not None and batch["y"].numel() > 0:
            batch_weights = model.class_weights[batch["y"]]
            if float(torch.max(batch_weights).item() - torch.min(batch_weights).item()) > 1e-12:
                first_batch = batch
                has_weight_diversity = True
                break
        if batch["y"].numel() > 1 and int(torch.unique(batch["y"]).numel()) > 1:
            first_batch = batch
            break
        if first_batch is None:
            first_batch = batch
    if first_batch is None:
        raise RuntimeError("No valid batch available for loss sanity check")
    first_batch = _move_batch_to_device(first_batch, device)
    feature_effect_diag: dict[str, Any] | None = None
    with torch.no_grad():
        logits = model(first_batch["graph_a"], first_batch["graph_b"], first_batch.get("feat_a"), first_batch.get("feat_b"))
        logits_adj = model.adjust_logits(logits)
        if feature_cache is not None and feature_cache.enabled and first_batch.get("feat_a") is not None and first_batch.get("feat_b") is not None:
            zeros_a = torch.zeros_like(first_batch["feat_a"])
            zeros_b = torch.zeros_like(first_batch["feat_b"])
            logits_zero = model(first_batch["graph_a"], first_batch["graph_b"], zeros_a, zeros_b)
            logits_zero_adj = model.adjust_logits(logits_zero)
            abs_diff = torch.abs(logits_adj - logits_zero_adj)
            pred_real = torch.argmax(logits_adj, dim=-1)
            pred_zero = torch.argmax(logits_zero_adj, dim=-1)
            argmax_change_pct = 100.0 * float(torch.mean((pred_real != pred_zero).float()).item())
            mean_abs_logit_diff = float(torch.mean(abs_diff).item())
            num_classes = int(logits_adj.size(-1))
            topk = min(5, num_classes)
            topk_real = torch.topk(logits_adj, k=topk, dim=-1).indices
            topk_zero = torch.topk(logits_zero_adj, k=topk, dim=-1).indices
            topk_overlap = (topk_real.unsqueeze(-1) == topk_zero.unsqueeze(-2)).any(dim=-1).float().mean(dim=-1)
            topk_overlap_mean = float(torch.mean(topk_overlap).item())
            topk_full_match_pct = 100.0 * float(torch.mean((topk_overlap >= (1.0 - 1e-8)).float()).item())

            prob_real = torch.softmax(logits_adj, dim=-1).clamp_min(1e-12)
            prob_zero = torch.softmax(logits_zero_adj, dim=-1).clamp_min(1e-12)
            log_prob_real = torch.log(prob_real)
            log_prob_zero = torch.log(prob_zero)
            kl_real_to_zero = torch.sum(prob_real * (log_prob_real - log_prob_zero), dim=-1)
            kl_zero_to_real = torch.sum(prob_zero * (log_prob_zero - log_prob_real), dim=-1)
            kl_real_to_zero_mean = float(torch.mean(kl_real_to_zero).item())
            kl_zero_to_real_mean = float(torch.mean(kl_zero_to_real).item())
            symmetric_kl_mean = float(torch.mean(0.5 * (kl_real_to_zero + kl_zero_to_real)).item())

            if num_classes >= 2:
                top2_real = torch.topk(logits_adj, k=2, dim=-1).values
                top2_zero = torch.topk(logits_zero_adj, k=2, dim=-1).values
                margin_real = top2_real[:, 0] - top2_real[:, 1]
                margin_zero = top2_zero[:, 0] - top2_zero[:, 1]
                margin_shift = margin_real - margin_zero
                mean_abs_margin_shift = float(torch.mean(torch.abs(margin_shift)).item())
                mean_signed_margin_shift = float(torch.mean(margin_shift).item())
            else:
                mean_abs_margin_shift = 0.0
                mean_signed_margin_shift = 0.0
            feature_effect_diag = {
                "mean_abs_logit_diff": mean_abs_logit_diff,
                "max_abs_logit_diff": float(torch.max(abs_diff).item()),
                "argmax_change_pct": argmax_change_pct,
                "top5_overlap_mean": topk_overlap_mean,
                "top5_full_match_pct": topk_full_match_pct,
                "kl_real_to_zero_mean": kl_real_to_zero_mean,
                "kl_zero_to_real_mean": kl_zero_to_real_mean,
                "symmetric_kl_mean": symmetric_kl_mean,
                "mean_abs_margin_shift": mean_abs_margin_shift,
                "mean_signed_margin_shift": mean_signed_margin_shift,
                "n_samples": int(logits_adj.size(0)),
            }
            if argmax_change_pct <= 0.0:
                LOGGER.warning("Feature influence check: argmax did not change on this batch despite non-zero logit shift")
            if mean_abs_logit_diff <= 1e-6:
                raise AssertionError(
                    "Feature influence check failed: zeroing feature inputs produced negligible logit change"
                )
            if symmetric_kl_mean <= 1e-6 and mean_abs_margin_shift <= 1e-4:
                raise AssertionError(
                    "Feature influence check failed: negligible distribution/margin change after zeroing features"
                )
        objective_loss = model.loss_fn(logits, first_batch["y"])
        weight = model.class_weights.to(device) if model.class_weights is not None else None
        manual_objective = F.cross_entropy(
            logits_adj,
            first_batch["y"],
            weight=weight,
            label_smoothing=float(model.label_smoothing),
            reduction="mean",
        )
        nll_loss = F.cross_entropy(
            logits_adj,
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
        "objective_minus_nll_batch": float(objective_loss.item() - nll_loss.item()),
        "batch_has_weight_diversity": bool(has_weight_diversity),
        "eval_objective_loss": float(test_eval["objective_loss"]),
        "eval_nll_loss": float(test_eval["nll_loss"]),
        "loss_config": {k: v for k, v in loss_cfg.items() if k not in {"class_counts_np", "class_weight_info_obj"}},
    }
    if loss_sanity["objective_abs_diff"] > 1e-7:
        raise AssertionError(
            f"model.loss_fn mismatch vs manual cross_entropy: abs_diff={loss_sanity['objective_abs_diff']:.6e}"
        )
    if float(loss_cfg["label_smoothing"]) > 0.0:
        if abs(loss_sanity["objective_minus_nll_batch"]) <= 1e-8:
            raise AssertionError("Expected objective_loss != nll_loss when label smoothing is enabled")
    if loss_cfg["use_class_weights"]:
        if loss_sanity["batch_has_weight_diversity"]:
            if abs(loss_sanity["objective_minus_nll_batch"]) <= 1e-8:
                raise AssertionError(
                    "Expected objective_loss != nll_loss when class weights are enabled and batch target weights differ"
                )
        else:
            LOGGER.warning(
                "Skipped strict objective-vs-nll inequality assertion: selected batch has uniform target weights."
            )
    LOGGER.info(
        "Loss sanity | objective=%.6f nll=%.6f delta=%.6f",
        loss_sanity["objective_loss_batch"],
        loss_sanity["nll_loss_batch"],
        loss_sanity["objective_minus_nll_batch"],
    )
    save_json(loss_sanity, diagnostics_dir / "loss_sanity.json")
    if feature_effect_diag is not None:
        feature_checks["feature_influence"] = feature_effect_diag

    # Calibration sanity.
    valid_eval = eval_epoch(
        model,
        valid_loader,
        device=device,
        amp_enabled=True,
        collect_logits=True,
        train_class_counts=loss_cfg["class_counts_np"],
    )
    temperature = fit_temperature(valid_eval["y_logits"], valid_eval["y_true"], device=device, max_iter=100)
    calibrated_logits = apply_temperature(test_eval["y_logits"], temperature)
    calibrated_prob = torch.softmax(torch.tensor(calibrated_logits, dtype=torch.float32), dim=-1).numpy()
    calibrated_metrics = evaluate_multiclass_metrics(
        y_true=test_eval["y_true"],
        y_prob=calibrated_prob,
        ece_bins=args.ece_bins,
        include_ovr_details=True,
        train_class_counts=loss_cfg["class_counts_np"],
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
    weight_rand_model.set_loss_params(
        class_weights=model.class_weights,
        label_smoothing=model.label_smoothing,
        logit_adjust_tau=model.logit_adjust_tau,
        logit_adjust_log_pi=model.logit_adjust_log_pi,
    )
    permuted_model = _train_label_permuted_copy(
        model,
        loader=train_loader,
        device=device,
        seed=args.seed + 29,
        steps=args.randomization_steps,
        lr=args.randomization_lr,
    )
    permuted_model.set_loss_params(
        class_weights=model.class_weights,
        label_smoothing=model.label_smoothing,
        logit_adjust_tau=model.logit_adjust_tau,
        logit_adjust_log_pi=model.logit_adjust_log_pi,
    )

    faith_rows: list[dict[str, Any]] = []
    n_done = 0
    ds_idx = 0
    while n_done < args.faithfulness_examples and ds_idx < len(test_ds):
        sample = test_ds[ds_idx]
        ds_idx += 1
        if sample is None:
            continue
        base_probs, target_class, base_attn = _predict_with_attention(
            model,
            sample.graph_a,
            sample.graph_b,
            device,
            feat_a=sample.feat_a.unsqueeze(0),
            feat_b=sample.feat_b.unsqueeze(0),
        )
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
            _, _, wr_attn = _predict_with_attention(
                weight_rand_model,
                sample.graph_a,
                sample.graph_b,
                device,
                feat_a=sample.feat_a.unsqueeze(0),
                feat_b=sample.feat_b.unsqueeze(0),
            )
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
            _, _, pm_attn = _predict_with_attention(
                permuted_model,
                sample.graph_a,
                sample.graph_b,
                device,
                feat_a=sample.feat_a.unsqueeze(0),
                feat_b=sample.feat_b.unsqueeze(0),
            )
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

    feature_cache_stats = {}
    feature_cache_hit_rates = {}
    if feature_cache is not None:
        feature_cache_stats = {k: int(v) for k, v in feature_cache.stats.items()}
        feature_cache_hit_rates = _cache_hit_rates(feature_cache_stats, feature_cfg)
        if bool(feature_cfg["use_physchem_features"]) and scaler_mtime_before is not None:
            scaler_mtime_after = float(feature_cache.scaler_path.stat().st_mtime)
            mtime_unchanged = bool(abs(scaler_mtime_after - scaler_mtime_before) <= 1e-6)
            if "scaler" not in feature_checks:
                feature_checks["scaler"] = {}
            feature_checks["scaler"]["mtime_unix_after_diag"] = scaler_mtime_after
            feature_checks["scaler"]["mtime_unchanged_during_diag"] = mtime_unchanged
            if not mtime_unchanged:
                raise AssertionError("Physchem scaler file was modified during diagnostics/eval; expected immutable train artifact")
    feature_checks["cache_stats"] = feature_cache_stats
    feature_checks["cache_hit_rates"] = feature_cache_hit_rates
    save_json(feature_checks, diagnostics_dir / "feature_checks.json")

    diagnostic_summary = {
        "config": {
            "seed": int(args.seed),
            "limit": int(args.limit) if args.limit is not None else None,
            "split_strategy": args.split_strategy,
            "split_seed": int(args.split_seed),
            "device": str(device),
            "feature_config": {
                "use_ecfp_features": bool(feature_cfg["use_ecfp_features"]),
                "use_physchem_features": bool(feature_cfg["use_physchem_features"]),
                "use_maccs_features": bool(feature_cfg["use_maccs_features"]),
                "ecfp_bits": int(feature_cfg["ecfp_bits"]),
                "ecfp_radius": int(feature_cfg["ecfp_radius"]),
                "physchem_dim": int(feature_cfg["physchem_dim"]),
                "maccs_dim": int(feature_cfg["maccs_dim"]),
            },
        },
        "cache_stats": cache_stats,
        "feature_checks": feature_checks,
        "label_checks": labels_check,
        "distribution_summary": dist_summary,
        "metrics_key": {
            "accuracy": float(metric_report["accuracy"]),
            "macro_f1": float(metric_report["macro_f1"]),
            "micro_f1": float(metric_report["micro_f1"]),
            "kappa": float(metric_report["kappa"]),
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
    print(f"class_counts={diagnostics_dir / 'class_counts.json'}")
    print(f"class_weights={diagnostics_dir / 'class_weights.json'}")
    print(f"labels_check={diagnostics_dir / 'labels_check.json'}")
    print(f"class_distribution={diagnostics_dir / 'class_distribution.csv'}")
    print(f"loss_sanity={diagnostics_dir / 'loss_sanity.json'}")
    print(f"metrics_sanity={diagnostics_dir / 'metrics_sanity.json'}")
    print(f"calibration_sanity={diagnostics_dir / 'calibration_sanity.json'}")
    print(f"faithfulness_sanity={diagnostics_dir / 'faithfulness_sanity.csv'}")
    print(f"randomization_sanity={diagnostics_dir / 'randomization_sanity.json'}")
    print(f"feature_checks={diagnostics_dir / 'feature_checks.json'}")
    print(f"diagnostic_summary={diagnostics_dir / 'diagnostic_summary.json'}")


if __name__ == "__main__":
    main()
