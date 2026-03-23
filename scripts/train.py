#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ddigat.config import default_project_config
from ddigat.data.cache import DrugFeatureCache, GraphCache
from ddigat.data.drug_features import compute_train_feature_stats
from ddigat.data.splits import DDIPairDataset, make_pair_dataloader, subsample_dataframe
from ddigat.data.tdc_ddi import load_tdc_drugbank_ddi
from ddigat.model.pair_model import DDIPairModel
from ddigat.train.loop import fit
from ddigat.utils.class_weights import (
    assert_class_weight_sanity,
    class_counts_payload,
    class_weights_payload,
    compute_class_priors,
    compute_class_weights,
    compute_tail_class_ids,
)
from ddigat.utils.io import ensure_dir, save_json
from ddigat.utils.logging import get_logger
from ddigat.utils.seed import seed_everything


LOGGER = get_logger("scripts.train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Siamese GNN for DrugBank DDI.")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--limit", type=int, default=None, help="Limit rows per split for smoke tests.")
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--out_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--encoder_type", type=str, default="gat", choices=["gat", "gcn", "gin", "mlp"])
    p.add_argument("--use_ecfp_features", action="store_true")
    p.add_argument("--use_physchem_features", action="store_true")
    p.add_argument("--use_maccs_features", action="store_true")
    p.add_argument("--ecfp_bits", type=int, default=2048)
    p.add_argument("--ecfp_radius", type=int, default=2)
    p.add_argument("--physchem_dim", type=int, default=0, help="0=auto from extractor")
    p.add_argument("--use_class_weights", action="store_true")
    p.add_argument("--class_weight_method", type=str, default="inv_sqrt", choices=["inv_sqrt", "effective_num"])
    p.add_argument("--class_weight_normalize", type=str, default="sample_mean", choices=["sample_mean", "mean_seen", "none"])
    p.add_argument("--class_weight_beta", type=float, default=0.9999)
    p.add_argument("--class_weight_clip_min", type=float, default=0.25)
    p.add_argument("--class_weight_clip_max", type=float, default=4.0)
    p.add_argument("--class_weight_eps", type=float, default=1e-12)
    p.add_argument("--enable_drw", action="store_true")
    p.add_argument("--drw_start_epoch", type=int, default=None)
    p.add_argument("--drw_lr_drop", type=float, default=0.2)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument(
        "--logit_adjust_tau",
        type=float,
        default=0.0,
        help="Logit adjustment strength tau. 0.0 disables LA; logits become z + tau*log(pi_train).",
    )
    p.add_argument("--split_strategy", type=str, default="cold_drug", choices=["cold_drug", "cold_drug_v2", "tdc"])
    p.add_argument("--split_seed", type=int, default=42)
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
    p.add_argument("--cold_write_legacy_flat_splits", action="store_true")
    p.add_argument(
        "--split_cache_dir",
        type=str,
        default=None,
        help="Optional shared directory used for persisted split cache; defaults to output_dir.",
    )
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


def compute_drw_class_weights(
    train_counts: torch.Tensor,
    *,
    method: str,
    normalize: str,
    clip_min: float,
    clip_max: float,
    eps: float,
) -> torch.Tensor:
    counts = train_counts.detach().to(dtype=torch.float64, device=torch.device("cpu")).reshape(-1)
    if counts.numel() <= 0:
        raise ValueError("train_counts must be non-empty for DRW.")
    if float(eps) <= 0.0:
        raise ValueError(f"class_weight_eps must be > 0, got {eps}")
    if float(clip_min) < 0.0 or float(clip_max) <= 0.0 or float(clip_min) > float(clip_max):
        raise ValueError(f"Invalid clip bounds: clip_min={clip_min}, clip_max={clip_max}")

    method_norm = str(method).lower().strip()
    if method_norm != "inv_sqrt":
        raise ValueError(f"DRW supports class_weight_method='inv_sqrt' only, got {method}")

    seen = counts > 0
    n_seen = int(seen.sum().item())
    if n_seen <= 0:
        raise ValueError("All classes are unseen in train split; cannot build DRW weights.")

    w = torch.zeros_like(counts, dtype=torch.float64)
    w[seen] = 1.0 / torch.sqrt(counts[seen] + float(eps))

    normalize_mode = str(normalize).lower().strip()
    if normalize_mode in {"sample_mean", "mean_seen"}:
        mean_seen = torch.mean(w[seen])
        if not torch.isfinite(mean_seen).item() or float(mean_seen.item()) <= 0.0:
            raise ValueError(f"Invalid seen-class mean during DRW normalization: {float(mean_seen.item())}")
        w[seen] = w[seen] / mean_seen
    elif normalize_mode in {"none", ""}:
        pass
    else:
        raise ValueError(
            f"Unsupported class_weight_normalize for DRW: {normalize}. Use sample_mean, mean_seen, or none."
        )

    w[seen] = torch.clamp(w[seen], min=float(clip_min), max=float(clip_max))
    w[~seen] = 0.0
    return w.to(dtype=torch.float32)


def main() -> None:
    args = parse_args()
    if float(args.logit_adjust_tau) > 0.0 and str(args.split_strategy) not in {"cold_drug", "cold_drug_v2"}:
        raise ValueError(
            "Logit adjustment is configured for cold-drug splits only. "
            "Use --split_strategy cold_drug or --split_strategy cold_drug_v2."
        )
    device = resolve_device(args.device)
    seed_everything(args.seed)
    ensure_dir(args.output_dir)
    split_cache_dir = args.split_cache_dir or args.output_dir
    is_feature_only = str(args.encoder_type).lower().strip() == "mlp"
    if is_feature_only and not bool(args.use_ecfp_features or args.use_physchem_features or args.use_maccs_features):
        raise ValueError("encoder_type='mlp' requires at least one enabled drug feature pathway.")

    train_df, valid_df, test_df, label_map = load_tdc_drugbank_ddi(
        args.data_dir,
        output_dir=split_cache_dir,
        split_strategy=args.split_strategy,
        split_seed=args.split_seed,
        cold_k=args.cold_k,
        cold_fold=args.cold_fold,
        cold_protocol=args.cold_protocol,
        cold_min_test_pairs=args.cold_min_test_pairs,
        cold_min_test_labels=args.cold_min_test_labels,
        cold_max_resamples=args.cold_max_resamples,
        cold_dedupe_policy=args.cold_dedupe_policy,
        cold_selection_objective=args.cold_selection_objective,
        cold_write_legacy_flat_splits=bool(args.cold_write_legacy_flat_splits),
    )
    if args.limit is not None:
        train_df = subsample_dataframe(train_df, limit=args.limit, seed=args.seed, label_col="y", ensure_class_coverage=True)

    cache = None if is_feature_only else GraphCache(output_dir=args.output_dir)
    all_smiles = (
        list(train_df["drug_a_smiles"])
        + list(train_df["drug_b_smiles"])
        + list(valid_df["drug_a_smiles"])
        + list(valid_df["drug_b_smiles"])
    )
    if cache is not None:
        cache.build(all_smiles, show_progress=True)
    feature_cache: DrugFeatureCache | None = None
    resolved_physchem_dim = 0
    use_any_drug_features = bool(args.use_ecfp_features or args.use_physchem_features or args.use_maccs_features)
    if use_any_drug_features:
        feature_cache = DrugFeatureCache(
            output_dir=args.output_dir,
            use_ecfp=bool(args.use_ecfp_features),
            use_physchem=bool(args.use_physchem_features),
            use_maccs=bool(args.use_maccs_features),
            ecfp_bits=int(args.ecfp_bits),
            ecfp_radius=int(args.ecfp_radius),
        )
        if args.use_physchem_features:
            train_smiles = list(train_df["drug_a_smiles"]) + list(train_df["drug_b_smiles"])
            physchem_stats = compute_train_feature_stats(train_smiles)
            if int(args.physchem_dim) > 0 and int(physchem_stats["dim"]) != int(args.physchem_dim):
                raise ValueError(
                    f"--physchem_dim={args.physchem_dim} does not match extractor dim={physchem_stats['dim']}"
                )
            feature_cache.set_physchem_stats(physchem_stats, persist=True)
            resolved_physchem_dim = int(physchem_stats["dim"])
        feature_cache.build(all_smiles, show_progress=True)

    train_ds = DDIPairDataset(train_df, cache, feature_cache=feature_cache, split_name="train")
    valid_ds = DDIPairDataset(valid_df, cache, feature_cache=feature_cache, split_name="valid")
    train_loader = make_pair_dataloader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, seed=args.seed)
    valid_loader = make_pair_dataloader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, seed=args.seed)

    cfg = default_project_config(data_dir=args.data_dir, output_dir=args.output_dir, device=str(device), num_classes=len(label_map))
    cfg.train.seed = args.seed
    cfg.train.batch_size = args.batch_size
    cfg.train.epochs = args.epochs
    cfg.train.lr = args.lr
    cfg.train.weight_decay = args.weight_decay
    cfg.train.num_workers = args.num_workers
    cfg.train.patience = args.patience
    cfg.train.limit = args.limit
    cfg.train.device = str(device)
    cfg.train.use_class_weights = bool(args.use_class_weights)
    cfg.train.class_weight_method = str(args.class_weight_method)
    cfg.train.class_weight_normalize = str(args.class_weight_normalize)
    cfg.train.class_weight_beta = float(args.class_weight_beta)
    cfg.train.class_weight_clip_min = float(args.class_weight_clip_min)
    cfg.train.class_weight_clip_max = float(args.class_weight_clip_max)
    cfg.train.class_weight_eps = float(args.class_weight_eps)
    cfg.train.enable_drw = bool(args.enable_drw)
    cfg.train.drw_lr_drop = float(args.drw_lr_drop)
    cfg.train.label_smoothing = float(args.label_smoothing)
    cfg.train.logit_adjust_tau = float(args.logit_adjust_tau)
    cfg.train.split_strategy = str(args.split_strategy)
    cfg.train.split_seed = int(args.split_seed)
    cfg.train.cold_k = int(args.cold_k)
    cfg.train.cold_fold = int(args.cold_fold)
    cfg.train.cold_protocol = str(args.cold_protocol)
    cfg.train.cold_min_test_pairs = int(args.cold_min_test_pairs)
    cfg.train.cold_min_test_labels = int(args.cold_min_test_labels)
    cfg.train.cold_max_resamples = int(args.cold_max_resamples)
    cfg.train.cold_dedupe_policy = str(args.cold_dedupe_policy)
    cfg.train.cold_write_legacy_flat_splits = bool(args.cold_write_legacy_flat_splits)
    cfg.model.encoder_type = args.encoder_type
    cfg.model.hidden_dim = args.hidden_dim
    cfg.model.out_dim = args.out_dim
    cfg.model.num_layers = args.num_layers
    cfg.model.heads = args.heads
    cfg.model.dropout = args.dropout
    cfg.model.num_classes = len(label_map)
    cfg.model.use_ecfp_features = bool(args.use_ecfp_features)
    cfg.model.use_physchem_features = bool(args.use_physchem_features)
    cfg.model.use_maccs_features = bool(args.use_maccs_features)
    cfg.model.ecfp_bits = int(args.ecfp_bits)
    cfg.model.ecfp_radius = int(args.ecfp_radius)
    cfg.model.physchem_dim = int(resolved_physchem_dim)
    cfg.model.maccs_dim = 166

    model = DDIPairModel(
        in_dim=cfg.model.in_dim,
        edge_dim=cfg.model.edge_dim,
        encoder_type=cfg.model.encoder_type,
        hidden_dim=cfg.model.hidden_dim,
        out_dim=cfg.model.out_dim,
        num_layers=cfg.model.num_layers,
        heads=cfg.model.heads,
        dropout=cfg.model.dropout,
        mlp_hidden_dim=cfg.model.mlp_hidden_dim,
        num_classes=cfg.model.num_classes,
        pooling=cfg.model.pooling,
        use_ecfp_features=cfg.model.use_ecfp_features,
        use_physchem_features=cfg.model.use_physchem_features,
        use_maccs_features=cfg.model.use_maccs_features,
        ecfp_bits=cfg.model.ecfp_bits,
        physchem_dim=cfg.model.physchem_dim,
        maccs_dim=cfg.model.maccs_dim,
        ecfp_proj_dim=cfg.model.ecfp_proj_dim,
        physchem_proj_dim=cfg.model.physchem_proj_dim,
        maccs_proj_dim=cfg.model.maccs_proj_dim,
    ).to(device)
    diagnostics_dir = ensure_dir(Path(args.output_dir) / "diagnostics")
    if feature_cache is not None:
        feature_diag = {
            "use_ecfp_features": bool(cfg.model.use_ecfp_features),
            "use_physchem_features": bool(cfg.model.use_physchem_features),
            "use_maccs_features": bool(cfg.model.use_maccs_features),
            "ecfp_bits": int(cfg.model.ecfp_bits),
            "ecfp_radius": int(cfg.model.ecfp_radius),
            "physchem_dim": int(cfg.model.physchem_dim),
            "maccs_dim": int(cfg.model.maccs_dim),
            "feature_dim_total": int(feature_cache.feature_dim),
            "cache_stats": {k: int(v) for k, v in feature_cache.stats.items()},
        }
        if feature_cache.scaler_path.exists():
            loaded_stats = feature_cache.load_physchem_stats()
            if isinstance(loaded_stats, dict):
                feature_diag["physchem_scaler"] = loaded_stats
        save_json(feature_diag, diagnostics_dir / "feature_cache.json")
    train_labels_cpu = torch.as_tensor(train_ds.df["y"].to_numpy(dtype=int), dtype=torch.long, device=torch.device("cpu"))
    train_counts_t = torch.bincount(train_labels_cpu, minlength=int(cfg.model.num_classes)).to(dtype=torch.long)
    class_counts = train_counts_t.numpy().astype(int, copy=True)
    if class_counts.shape[0] != int(cfg.model.num_classes):
        raise RuntimeError(
            f"train_counts size mismatch: expected {cfg.model.num_classes}, got {class_counts.shape[0]}"
        )
    drw_start_epoch = args.drw_start_epoch
    if drw_start_epoch is None:
        drw_start_epoch = int(0.7 * int(args.epochs)) + 1
    if int(drw_start_epoch) < 1 or int(drw_start_epoch) > int(args.epochs):
        raise ValueError(f"drw_start_epoch must be in [1, {int(args.epochs)}], got {drw_start_epoch}")
    cfg.train.drw_start_epoch = int(drw_start_epoch)
    cfg.train.class_counts = [int(v) for v in class_counts.tolist()]
    save_json(class_counts_payload(class_counts, num_classes=cfg.model.num_classes), diagnostics_dir / "class_counts.json")
    save_json(class_counts_payload(class_counts, num_classes=cfg.model.num_classes), diagnostics_dir / "train_counts.json")
    priors, log_priors = compute_class_priors(class_counts, eps=float(cfg.train.logit_adjust_eps))
    save_json(
        {
            "eps": float(cfg.train.logit_adjust_eps),
            "num_classes": int(cfg.model.num_classes),
            "priors": [float(v) for v in priors.tolist()],
            "log_priors": [float(v) for v in log_priors.tolist()],
        },
        diagnostics_dir / "train_priors.json",
    )
    tail_ids = compute_tail_class_ids(class_counts, fraction=0.2, include_zero_count=True)
    LOGGER.info(
        "Logit adjustment config | tau=%.4f eps=%.1e | tail_k=%d/%d (bottom 20%% train support, including zero-count classes)",
        float(args.logit_adjust_tau),
        float(cfg.train.logit_adjust_eps),
        int(tail_ids.size),
        int(cfg.model.num_classes),
    )

    class_weight_info = compute_class_weights(
        class_counts,
        method=str(args.class_weight_method),
        normalize=str(args.class_weight_normalize),
        beta=float(args.class_weight_beta),
        eps=float(cfg.train.class_weight_eps),
        clip_min=float(args.class_weight_clip_min),
        clip_max=float(args.class_weight_clip_max),
    )
    assert_class_weight_sanity(
        class_weight_info.weights,
        num_classes=cfg.model.num_classes,
        clip_min=float(args.class_weight_clip_min),
        clip_max=float(args.class_weight_clip_max),
        mean_after_normalization=float(class_weight_info.mean_after_normalization),
        counts=class_counts,
        normalize=str(args.class_weight_normalize),
    )
    save_json(
        class_weights_payload(
            enabled=bool(args.use_class_weights or args.enable_drw),
            method=str(args.class_weight_method),
            beta=float(args.class_weight_beta),
            eps=float(args.class_weight_eps),
            clip_min=float(args.class_weight_clip_min),
            clip_max=float(args.class_weight_clip_max),
            computation=class_weight_info,
        ),
        diagnostics_dir / "class_weights.json",
    )

    drw_class_weights: torch.Tensor | None = None
    if bool(args.enable_drw):
        drw_weights_cpu = compute_drw_class_weights(
            train_counts_t,
            method=str(args.class_weight_method),
            normalize=str(args.class_weight_normalize),
            clip_min=float(args.class_weight_clip_min),
            clip_max=float(args.class_weight_clip_max),
            eps=float(args.class_weight_eps),
        )
        drw_class_weights = drw_weights_cpu.to(device)
        seen_mask = train_counts_t > 0
        seen_mean_after_clip = float(drw_weights_cpu[seen_mask].mean().item())
        LOGGER.info(
            "DRW configured | start_epoch=%d lr_drop=%.4f method=%s normalize=%s "
            "class_w[min/mean/max]=%.4f/%.4f/%.4f seen_mean_after_clip=%.4f (may drift from 1.0 after clipping)",
            int(drw_start_epoch),
            float(args.drw_lr_drop),
            str(args.class_weight_method),
            str(args.class_weight_normalize),
            float(drw_weights_cpu.min().item()),
            float(drw_weights_cpu.mean().item()),
            float(drw_weights_cpu.max().item()),
            seen_mean_after_clip,
        )
        class_weights = None
    elif args.use_class_weights:
        class_weights = class_weight_info.weights
        LOGGER.info(
            "Using class-weighted loss | method=%s normalize=%s beta=%.6f min=%.4f mean=%.4f max=%.4f seen=%d unseen=%d sat_min_seen=%.2f sat_max_seen=%.2f",
            args.class_weight_method,
            args.class_weight_normalize,
            float(args.class_weight_beta),
            float(class_weights.min()),
            float(class_weights.mean()),
            float(class_weights.max()),
            int(class_weight_info.n_seen),
            int(class_weight_info.n_unseen),
            float(class_weight_info.sat_min_seen),
            float(class_weight_info.sat_max_seen),
        )
    else:
        class_weights = None
    log_pi_t = torch.tensor(log_priors, dtype=torch.float32)
    model.set_loss_params(
        class_weights=class_weights,
        label_smoothing=float(args.label_smoothing),
        logit_adjust_tau=float(args.logit_adjust_tau),
        logit_adjust_log_pi=log_pi_t,
    )

    loss_config = {
        "use_class_weights": bool(args.use_class_weights),
        "enable_drw": bool(args.enable_drw),
        "drw_start_epoch": int(drw_start_epoch),
        "drw_lr_drop": float(args.drw_lr_drop),
        "class_weight_method": str(args.class_weight_method),
        "class_weight_normalize": str(args.class_weight_normalize),
        "class_weight_beta": float(args.class_weight_beta),
        "class_weight_clip_min": float(args.class_weight_clip_min),
        "class_weight_clip_max": float(args.class_weight_clip_max),
        "class_weight_eps": float(args.class_weight_eps),
        "label_smoothing": float(args.label_smoothing),
        "logit_adjust_tau": float(args.logit_adjust_tau),
        "logit_adjust_eps": float(cfg.train.logit_adjust_eps),
        "num_classes": int(cfg.model.num_classes),
        "class_counts": [int(v) for v in class_counts.tolist()],
    }
    if drw_class_weights is not None:
        cw_cpu = drw_class_weights.detach().cpu()
        loss_config["drw_class_weight_min"] = float(cw_cpu.min().item())
        loss_config["drw_class_weight_mean"] = float(cw_cpu.mean().item())
        loss_config["drw_class_weight_max"] = float(cw_cpu.max().item())
    LOGGER.info("Loss config: %s", json.dumps(loss_config, sort_keys=True))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    LOGGER.info(
        "Training | train=%d valid=%d | batch_size=%d epochs=%d device=%s",
        len(train_ds),
        len(valid_ds),
        args.batch_size,
        args.epochs,
        device,
    )
    cfg.train.training_start_unix = float(time.time())

    fit_result = fit(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        output_dir=args.output_dir,
        config=cfg.to_dict(),
        label_map=label_map,
        loss_config=loss_config,
        patience=args.patience,
        min_delta=0.0,
        amp_enabled=True,
        train_class_counts=class_counts,
        enable_drw=bool(args.enable_drw),
        drw_start_epoch=int(drw_start_epoch),
        drw_lr_drop=float(args.drw_lr_drop),
        drw_class_weights=drw_class_weights,
    )

    save_json({"history": fit_result["history"], "best_epoch": fit_result["best_epoch"], "best_metrics": fit_result["best_metrics"]}, Path(args.output_dir) / "training_history.json")
    LOGGER.info("Training complete. Best checkpoint: %s", fit_result["checkpoint_path"])


if __name__ == "__main__":
    main()
