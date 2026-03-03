#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ddigat.data.cache import GraphCache
from ddigat.data.splits import subsample_dataframe
from ddigat.data.tdc_ddi import load_tdc_drugbank_ddi
from ddigat.utils.logging import get_logger
from ddigat.utils.seed import seed_everything


LOGGER = get_logger("scripts.prepare_data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download/load TDC DrugBank DDI and build graph cache.")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=64)  # kept for interface consistency
    p.add_argument("--epochs", type=int, default=1)  # kept for interface consistency
    p.add_argument("--lr", type=float, default=1e-3)  # kept for interface consistency
    p.add_argument("--device", type=str, default="cpu")  # kept for interface consistency
    p.add_argument("--limit", type=int, default=None, help="Limit rows per split for smoke tests.")
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


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

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

    all_smiles = pd.concat(
        [train_df["drug_a_smiles"], train_df["drug_b_smiles"], valid_df["drug_a_smiles"], valid_df["drug_b_smiles"], test_df["drug_a_smiles"], test_df["drug_b_smiles"]],
        axis=0,
    ).dropna().astype(str).unique().tolist()

    LOGGER.info(
        "Dataset stats | train=%d valid=%d test=%d | unique_smiles=%d | labels=%d",
        len(train_df),
        len(valid_df),
        len(test_df),
        len(all_smiles),
        len(label_map),
    )

    cache = GraphCache(output_dir=args.output_dir)
    stats = cache.build(all_smiles, show_progress=True)
    LOGGER.info("Cache build stats: %s", stats)


if __name__ == "__main__":
    main()
