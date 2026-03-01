from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data

from ddigat.data.cache import GraphCache
from ddigat.utils.logging import get_logger
from ddigat.utils.seed import seed_worker


LOGGER = get_logger(__name__)


@dataclass
class PairSample:
    graph_a: Data
    graph_b: Data
    y: int
    pair_id: str
    smiles_a: str
    smiles_b: str


class DDIPairDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        graph_cache: GraphCache,
        limit: Optional[int] = None,
        split_name: str = "train",
    ) -> None:
        if limit is not None:
            df = df.iloc[:limit].copy()
        self.df = df.reset_index(drop=True)
        self.graph_cache = graph_cache
        self.split_name = split_name

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Optional[PairSample]:
        row = self.df.iloc[idx]
        a = self.graph_cache.get_or_create(str(row["drug_a_smiles"]))
        b = self.graph_cache.get_or_create(str(row["drug_b_smiles"]))
        if a is None or b is None:
            return None
        pair_id = f"{self.split_name}_{idx}"
        return PairSample(
            graph_a=a,
            graph_b=b,
            y=int(row["y"]),
            pair_id=pair_id,
            smiles_a=str(row["drug_a_smiles"]),
            smiles_b=str(row["drug_b_smiles"]),
        )


def subsample_dataframe(
    df: pd.DataFrame,
    limit: Optional[int],
    seed: int = 42,
    label_col: str = "y",
    ensure_class_coverage: bool = True,
) -> pd.DataFrame:
    """Deterministically subsample rows while avoiding order-based label collapse.

    If `ensure_class_coverage` is True and `label_col` exists, one sample per class
    is selected first (when feasible), then the remaining rows are sampled uniformly.
    """
    if limit is None or int(limit) <= 0 or len(df) <= int(limit):
        return df.reset_index(drop=True).copy()

    n = int(limit)
    work_df = df.reset_index(drop=True)
    rng = np.random.default_rng(int(seed))

    if (not ensure_class_coverage) or (label_col not in work_df.columns):
        idx = rng.choice(len(work_df), size=n, replace=False)
        out = work_df.iloc[idx].copy()
        return out.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)

    selected: list[int] = []
    remaining_pool: list[int] = []
    grouped = work_df.groupby(label_col, sort=False).indices
    labels = sorted(grouped.keys())
    n_classes = len(labels)

    if n >= n_classes:
        for lab in labels:
            idxs = np.array(grouped[lab], dtype=int)
            rng.shuffle(idxs)
            selected.append(int(idxs[0]))
            if len(idxs) > 1:
                remaining_pool.extend(idxs[1:].tolist())
    else:
        # If limit is smaller than class count, pick classes proportionally to prevalence.
        class_counts = np.array([len(grouped[lab]) for lab in labels], dtype=float)
        probs = class_counts / np.maximum(class_counts.sum(), 1.0)
        chosen_labels = rng.choice(np.array(labels, dtype=object), size=n, replace=False, p=probs)
        for lab in chosen_labels.tolist():
            idxs = np.array(grouped[lab], dtype=int)
            rng.shuffle(idxs)
            selected.append(int(idxs[0]))
        out = work_df.iloc[selected].copy()
        return out.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)

    remaining_needed = n - len(selected)
    if remaining_needed > 0:
        if len(remaining_pool) < remaining_needed:
            selected_set = set(selected)
            remaining_pool = [i for i in range(len(work_df)) if i not in selected_set]
        extra = rng.choice(np.array(remaining_pool, dtype=int), size=remaining_needed, replace=False).tolist()
        selected.extend(int(i) for i in extra)

    out = work_df.iloc[selected].copy()
    return out.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)


def collate_pair_batch(items: list[Optional[PairSample]]) -> Optional[dict[str, Any]]:
    items = [it for it in items if it is not None]
    if not items:
        return None
    graph_a = Batch.from_data_list([it.graph_a for it in items])
    graph_b = Batch.from_data_list([it.graph_b for it in items])
    y = torch.tensor([it.y for it in items], dtype=torch.long)
    meta = {
        "pair_ids": [it.pair_id for it in items],
        "smiles_a": [it.smiles_a for it in items],
        "smiles_b": [it.smiles_b for it in items],
    }
    return {"graph_a": graph_a, "graph_b": graph_b, "y": y, "meta": meta}


def make_pair_dataloader(
    dataset: DDIPairDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pair_batch,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
