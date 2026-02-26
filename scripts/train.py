#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ddigat.config import default_project_config
from ddigat.data.cache import GraphCache
from ddigat.data.splits import DDIPairDataset, make_pair_dataloader
from ddigat.data.tdc_ddi import load_tdc_drugbank_ddi
from ddigat.model.pair_model import DDIPairModel
from ddigat.train.loop import fit
from ddigat.utils.io import ensure_dir, save_json
from ddigat.utils.logging import get_logger
from ddigat.utils.seed import seed_everything


LOGGER = get_logger("scripts.train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Siamese GAT for DrugBank DDI.")
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
    return p.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    train_df, valid_df, test_df, label_map = load_tdc_drugbank_ddi(args.data_dir, output_dir=args.output_dir)
    if args.limit is not None:
        train_df = train_df.head(args.limit).copy()
        valid_df = valid_df.head(max(1, args.limit // 4)).copy()
        test_df = test_df.head(max(1, args.limit // 4)).copy()

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
    cfg.model.hidden_dim = args.hidden_dim
    cfg.model.out_dim = args.out_dim
    cfg.model.num_layers = args.num_layers
    cfg.model.heads = args.heads
    cfg.model.dropout = args.dropout
    cfg.model.num_classes = len(label_map)

    model = DDIPairModel(
        in_dim=cfg.model.in_dim,
        edge_dim=cfg.model.edge_dim,
        hidden_dim=cfg.model.hidden_dim,
        out_dim=cfg.model.out_dim,
        num_layers=cfg.model.num_layers,
        heads=cfg.model.heads,
        dropout=cfg.model.dropout,
        mlp_hidden_dim=cfg.model.mlp_hidden_dim,
        num_classes=cfg.model.num_classes,
        pooling=cfg.model.pooling,
    ).to(device)
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

