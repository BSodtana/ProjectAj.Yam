#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ddigat.config import default_project_config
from ddigat.data.cache import GraphCache
from ddigat.data.splits import DDIPairDataset, make_pair_dataloader, subsample_dataframe
from ddigat.data.tdc_ddi import load_tdc_drugbank_ddi
from ddigat.model.pair_model import DDIPairModel
from ddigat.train.loop import fit
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
    p.add_argument("--encoder_type", type=str, default="gat", choices=["gat", "gcn", "gin"])
    p.add_argument("--use_class_weights", action="store_true")
    p.add_argument("--label_smoothing", type=float, default=0.0)
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


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y.astype(int), minlength=num_classes).astype(np.float64)
    inv = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    # Normalize and clip to avoid extreme instability.
    inv = inv / np.mean(inv)
    inv = np.clip(inv, 0.2, 5.0)
    return torch.tensor(inv, dtype=torch.float32)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    train_df, valid_df, test_df, label_map = load_tdc_drugbank_ddi(
        args.data_dir,
        output_dir=args.output_dir,
        split_strategy=args.split_strategy,
        split_seed=args.split_seed,
    )
    if args.limit is not None:
        train_df = subsample_dataframe(train_df, limit=args.limit, seed=args.seed, label_col="y", ensure_class_coverage=True)

    cache = GraphCache(output_dir=args.output_dir)
    all_smiles = (
        list(train_df["drug_a_smiles"])
        + list(train_df["drug_b_smiles"])
        + list(valid_df["drug_a_smiles"])
        + list(valid_df["drug_b_smiles"])
    )
    cache.build(all_smiles, show_progress=True)

    train_ds = DDIPairDataset(train_df, cache, split_name="train")
    valid_ds = DDIPairDataset(valid_df, cache, split_name="valid")
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
    cfg.train.label_smoothing = float(args.label_smoothing)
    cfg.train.split_strategy = str(args.split_strategy)
    cfg.train.split_seed = int(args.split_seed)
    cfg.model.encoder_type = args.encoder_type
    cfg.model.hidden_dim = args.hidden_dim
    cfg.model.out_dim = args.out_dim
    cfg.model.num_layers = args.num_layers
    cfg.model.heads = args.heads
    cfg.model.dropout = args.dropout
    cfg.model.num_classes = len(label_map)

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
    ).to(device)
    if args.use_class_weights:
        class_weights = compute_class_weights(train_df["y"].to_numpy(dtype=int), num_classes=cfg.model.num_classes)
        LOGGER.info("Using class-weighted loss (min=%.4f, max=%.4f)", float(class_weights.min()), float(class_weights.max()))
    else:
        class_weights = None
    model.set_loss_params(class_weights=class_weights, label_smoothing=float(args.label_smoothing))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    LOGGER.info(
        "Training | train=%d valid=%d | batch_size=%d epochs=%d device=%s",
        len(train_ds),
        len(valid_ds),
        args.batch_size,
        args.epochs,
        device,
    )

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
        patience=args.patience,
        min_delta=0.0,
        amp_enabled=True,
    )

    save_json({"history": fit_result["history"], "best_epoch": fit_result["best_epoch"], "best_metrics": fit_result["best_metrics"]}, Path(args.output_dir) / "training_history.json")
    LOGGER.info("Training complete. Best checkpoint: %s", fit_result["checkpoint_path"])


if __name__ == "__main__":
    main()
