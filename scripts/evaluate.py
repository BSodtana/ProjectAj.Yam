#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ddigat.data.cache import GraphCache
from ddigat.data.splits import DDIPairDataset, make_pair_dataloader
from ddigat.data.tdc_ddi import load_tdc_drugbank_ddi
from ddigat.model.pair_model import DDIPairModel
from ddigat.train.loop import eval_epoch
from ddigat.utils.io import torch_load
from ddigat.utils.logging import get_logger
from ddigat.utils.seed import seed_everything


LOGGER = get_logger("scripts.evaluate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained DDI GAT model.")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--checkpoint", type=str, default="./outputs/checkpoints/best.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1)  # interface consistency
    p.add_argument("--lr", type=float, default=1e-3)  # interface consistency
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_model_from_checkpoint_payload(payload: dict, device: torch.device) -> DDIPairModel:
    cfg = payload.get("config", {})
    model_cfg = (cfg or {}).get("model", {})
    model = DDIPairModel(
        in_dim=int(model_cfg.get("in_dim", 7)),
        edge_dim=int(model_cfg.get("edge_dim", 5)),
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        out_dim=int(model_cfg.get("out_dim", 128)),
        num_layers=int(model_cfg.get("num_layers", 3)),
        heads=int(model_cfg.get("heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        mlp_hidden_dim=int(model_cfg.get("mlp_hidden_dim", 256)),
        num_classes=int(model_cfg.get("num_classes", 86)),
        pooling=str(model_cfg.get("pooling", "mean")),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)

    payload = torch_load(args.checkpoint, map_location=device)
    model = build_model_from_checkpoint_payload(payload, device)

    train_df, valid_df, test_df, _ = load_tdc_drugbank_ddi(args.data_dir, output_dir=args.output_dir)
    if args.limit is not None:
        test_df = test_df.head(args.limit).copy()

    cache = GraphCache(output_dir=args.output_dir)
    test_ds = DDIPairDataset(test_df, cache, split_name="test")
    test_loader = make_pair_dataloader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, seed=args.seed)

    metrics = eval_epoch(model, test_loader, device=device, amp_enabled=True)
    LOGGER.info("Test loss: %.4f", metrics["loss"])
    LOGGER.info("Macro ROC-AUC (OvR): %.6f", metrics["macro_roc_auc_ovr"])
    LOGGER.info("Macro PR-AUC (OvR): %.6f", metrics["macro_pr_auc_ovr"])
    print(f"macro_roc_auc_ovr={metrics['macro_roc_auc_ovr']:.6f}")
    print(f"macro_pr_auc_ovr={metrics['macro_pr_auc_ovr']:.6f}")


if __name__ == "__main__":
    main()
