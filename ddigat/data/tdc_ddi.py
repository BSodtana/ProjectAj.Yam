from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from ddigat.utils.io import ensure_dir, load_json, save_json
from ddigat.utils.logging import get_logger


LOGGER = get_logger(__name__)

EXPECTED_COLS = ["drug_a_smiles", "drug_b_smiles", "y"]


def _pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lowered = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    for c in df.columns:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() == cl or cand.lower() in cl:
                return c
    return None


def _coerce_mapping(obj: Any) -> dict[str, str]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return {str(k): str(v) for k, v in obj.items()}
    if isinstance(obj, pd.Series):
        return {str(k): str(v) for k, v in obj.to_dict().items()}
    if isinstance(obj, pd.DataFrame):
        id_col = _pick_column(obj, ["id", "drug_id", "drugbank_id", "drug"])
        smiles_col = _pick_column(obj, ["smiles", "drug_smiles", "canonical_smiles"])
        if id_col and smiles_col:
            return {str(k): str(v) for k, v in zip(obj[id_col], obj[smiles_col])}
    return {}


def _infer_id_to_smiles_mapping(tdc_data_obj: Any) -> dict[str, str]:
    candidates: dict[str, str] = {}
    for attr in [
        "id2smiles",
        "drug2smiles",
        "drugid2smiles",
        "entity1",
        "entity2",
        "drugs",
        "drugbank",
    ]:
        if hasattr(tdc_data_obj, attr):
            candidates.update(_coerce_mapping(getattr(tdc_data_obj, attr)))

    # Last-resort scan over attributes for dict-like id->smiles shapes.
    for attr in dir(tdc_data_obj):
        if attr.startswith("_") or attr in {"path", "name"}:
            continue
        try:
            val = getattr(tdc_data_obj, attr)
        except Exception:
            continue
        mapped = _coerce_mapping(val)
        if mapped and len(mapped) > len(candidates):
            candidates.update(mapped)
    return candidates


def _normalize_split_df(df: pd.DataFrame, id_to_smiles: dict[str, str]) -> pd.DataFrame:
    df = df.copy()

    a_smiles_col = _pick_column(df, ["drug_a_smiles", "drug1_smiles", "drug1", "x1", "smiles1"])
    b_smiles_col = _pick_column(df, ["drug_b_smiles", "drug2_smiles", "drug2", "x2", "smiles2"])
    y_col = _pick_column(df, ["y", "label", "interaction", "event", "type"])

    if y_col is None:
        raise ValueError(f"Could not identify label column in split columns: {list(df.columns)}")

    def _map_if_needed(series: pd.Series) -> pd.Series:
        if series.dtype == object:
            sample = series.dropna().astype(str).head(20).tolist()
            if sample and all(("C" in s or "N" in s or "[" in s or "=" in s) for s in sample):
                return series.astype(str)
        if not id_to_smiles:
            return series.astype(str)
        mapped = series.astype(str).map(id_to_smiles)
        missing = int(mapped.isna().sum())
        if missing > 0:
            LOGGER.warning("Failed to map %d drug ids to SMILES in split", missing)
        return mapped

    if a_smiles_col is None or b_smiles_col is None:
        raise ValueError(f"Could not identify pair columns in split columns: {list(df.columns)}")

    out = pd.DataFrame(
        {
            "drug_a_smiles": _map_if_needed(df[a_smiles_col]),
            "drug_b_smiles": _map_if_needed(df[b_smiles_col]),
            "y": pd.to_numeric(df[y_col], errors="coerce").astype("Int64"),
        }
    )
    out = out.dropna(subset=["drug_a_smiles", "drug_b_smiles", "y"]).copy()
    out["drug_a_smiles"] = out["drug_a_smiles"].astype(str)
    out["drug_b_smiles"] = out["drug_b_smiles"].astype(str)
    out["y"] = out["y"].astype(int)
    return out[EXPECTED_COLS]


def _load_saved_splits(split_dir: Path) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    train_p = split_dir / "train.csv"
    valid_p = split_dir / "valid.csv"
    test_p = split_dir / "test.csv"
    if not (train_p.exists() and valid_p.exists() and test_p.exists()):
        return None
    train_df = pd.read_csv(train_p)
    valid_df = pd.read_csv(valid_p)
    test_df = pd.read_csv(test_p)
    return train_df[EXPECTED_COLS], valid_df[EXPECTED_COLS], test_df[EXPECTED_COLS]


def _load_split_meta(split_dir: Path) -> dict:
    p = split_dir / "meta.json"
    if p.exists():
        try:
            return load_json(p)
        except Exception:
            return {}
    return {}


def _save_split_meta(split_dir: Path, split_strategy: str, split_seed: int) -> None:
    save_json(
        {
            "split_strategy": split_strategy,
            "split_seed": int(split_seed),
            "split_impl_version": 2,
            "schema": EXPECTED_COLS,
        },
        split_dir / "meta.json",
    )


def _coerce_label_map_keys(label_map: dict) -> dict:
    coerced: dict = {}
    for k, v in label_map.items():
        try:
            coerced[int(k)] = v
        except Exception:
            coerced[k] = v
    return coerced


def _load_saved_label_map(split_dir: Path) -> Optional[dict]:
    p = split_dir / "label_map.json"
    if not p.exists():
        return None
    try:
        return _coerce_label_map_keys(load_json(p))
    except Exception as e:
        LOGGER.warning("Failed loading persisted label map from %s: %s", p, e)
        return None


def _log_split_label_coverage(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_classes: int,
) -> None:
    for split_name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        present = sorted(df["y"].astype(int).unique().tolist())
        present_set = set(present)
        missing = [c for c in range(num_classes) if c not in present_set]
        LOGGER.info(
            "%s label coverage | present=%d missing=%d",
            split_name,
            len(present),
            len(missing),
        )
        if missing:
            LOGGER.info("%s missing classes: %s", split_name, missing)


def _make_cold_drug_split(
    full_df: pd.DataFrame,
    seed: int,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    trials: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build realistic cold-drug split: test drugs are unseen in training.

    Strategy:
      1) Split drugs into train-drug set and holdout-drug set.
      2) Train pairs use only train-drug drugs (both sides in train set).
      3) Holdout pairs use only holdout-drug drugs (both sides in holdout set).
      4) Split holdout pairs into valid/test by pair-level random split.
    """
    if not np.isclose(train_ratio + valid_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.")

    drugs = sorted(set(full_df["drug_a_smiles"]).union(set(full_df["drug_b_smiles"])))
    if len(drugs) < 10:
        raise ValueError("Too few unique drugs for cold-drug split.")

    n_total = len(drugs)
    n_train = max(1, int(n_total * train_ratio))
    n_holdout = max(1, n_total - n_train)

    best = None
    best_score = -1
    rng = np.random.default_rng(seed)
    drug_array = np.array(drugs, dtype=object)
    for _ in range(trials):
        perm = rng.permutation(n_total)
        train_drugs = set(drug_array[perm[:n_train]].tolist())
        holdout_drugs = set(drug_array[perm[n_train : n_train + n_holdout]].tolist())

        train_mask = full_df["drug_a_smiles"].isin(train_drugs) & full_df["drug_b_smiles"].isin(train_drugs)
        holdout_mask = full_df["drug_a_smiles"].isin(holdout_drugs) & full_df["drug_b_smiles"].isin(holdout_drugs)

        train_df = full_df.loc[train_mask].copy()
        holdout_df = full_df.loc[holdout_mask].copy()
        score = len(train_df) + len(holdout_df)
        if len(train_df) == 0 or len(holdout_df) < 2:
            continue
        if score > best_score:
            best = (train_df, holdout_df)
            best_score = score

    if best is None:
        raise RuntimeError("Unable to create non-empty cold-drug split.")
    train_df, holdout_df = best
    rng_pairs = np.random.default_rng(seed + 9973)
    perm_pairs = rng_pairs.permutation(len(holdout_df))
    valid_frac = valid_ratio / max(valid_ratio + test_ratio, 1e-8)
    n_valid = int(len(holdout_df) * valid_frac)
    n_valid = min(max(1, n_valid), len(holdout_df) - 1)
    valid_idx = perm_pairs[:n_valid]
    test_idx = perm_pairs[n_valid:]
    valid_df = holdout_df.iloc[valid_idx].copy().reset_index(drop=True)
    test_df = holdout_df.iloc[test_idx].copy().reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    train_drug_set = set(train_df["drug_a_smiles"]).union(set(train_df["drug_b_smiles"]))
    test_drug_set = set(test_df["drug_a_smiles"]).union(set(test_df["drug_b_smiles"]))
    overlap = train_drug_set.intersection(test_drug_set)
    if overlap:
        raise RuntimeError(f"Cold-drug split invariant violated: {len(overlap)} test drugs also in train.")
    LOGGER.info(
        "Cold-drug split created | train=%d valid=%d test=%d (usable from full=%d)",
        len(train_df),
        len(valid_df),
        len(test_df),
        len(full_df),
    )
    return train_df, valid_df, test_df


def _normalize_label_indexing(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_map: dict,
    split_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Ensure labels are zero-based class indices for torch cross-entropy.

    TDC DrugBank DDI label ids can appear as 1..86. Internally we require 0..85.
    """
    frames = [train_df.copy(), valid_df.copy(), test_df.copy()]
    all_y = pd.concat([f["y"] for f in frames], axis=0).astype(int)
    y_min = int(all_y.min())
    y_max = int(all_y.max())
    num_classes = len(label_map)

    # Already zero-based and within bounds.
    if y_min == 0 and y_max <= num_classes - 1:
        return frames[0], frames[1], frames[2], label_map

    # Common TDC case: one-based labels 1..C.
    if y_min >= 1 and y_max <= num_classes:
        LOGGER.info(
            "Converting labels from one-based to zero-based indexing (min=%d, max=%d, classes=%d)",
            y_min,
            y_max,
            num_classes,
        )
        for f in frames:
            f["y"] = f["y"].astype(int) - 1

        shifted_label_map: dict = {}
        for k, v in label_map.items():
            try:
                shifted_label_map[int(k) - 1] = v
            except Exception:
                shifted_label_map[k] = v
        label_map = shifted_label_map

        if split_dir is not None:
            frames[0].to_csv(split_dir / "train.csv", index=False)
            frames[1].to_csv(split_dir / "valid.csv", index=False)
            frames[2].to_csv(split_dir / "test.csv", index=False)
            save_json({str(k): str(v) for k, v in label_map.items()}, split_dir / "label_map.json")
        return frames[0], frames[1], frames[2], label_map

    raise ValueError(
        f"Unexpected label range for {num_classes} classes: min={y_min}, max={y_max}. "
        "Expected zero-based [0..C-1] or one-based [1..C]."
    )


def load_tdc_drugbank_ddi(
    data_dir: str,
    output_dir: str = "outputs",
    split_strategy: str = "cold_drug",
    split_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load TDC DrugBank DDI splits and normalize to standard columns.

    split_strategy:
      - "cold_drug": disjoint drug sets across train/valid/test (recommended realistic split)
      - "tdc": use TDC-provided split directly
    """
    split_dir = ensure_dir(Path(output_dir) / "splits")
    saved = _load_saved_splits(split_dir)
    meta = _load_split_meta(split_dir)
    strategy = split_strategy.lower().strip()
    if strategy not in {"cold_drug", "tdc"}:
        raise ValueError(f"Unsupported split_strategy={split_strategy}. Expected one of: cold_drug, tdc")

    try:
        from tdc.multi_pred import DDI
        from tdc.utils import get_label_map
    except ImportError as e:  # pragma: no cover
        raise ImportError("Please install `tdc` to load the DrugBank DDI dataset.") from e

    if (
        saved is not None
        and meta.get("split_strategy") == strategy
        and int(meta.get("split_seed", split_seed)) == int(split_seed)
        and int(meta.get("split_impl_version", 0)) == 2
    ):
        label_map = _load_saved_label_map(split_dir)
        if label_map is None:
            label_map = get_label_map(name="DrugBank", task="DDI", path=data_dir)
        train_df, valid_df, test_df = saved
        train_df, valid_df, test_df, label_map = _normalize_label_indexing(
            train_df, valid_df, test_df, label_map, split_dir=split_dir
        )
        _log_split_label_coverage(train_df, valid_df, test_df, num_classes=len(label_map))
        LOGGER.info("Loaded persisted %s splits from %s", strategy, split_dir)
        return train_df, valid_df, test_df, label_map

    ensure_dir(data_dir)
    try:
        data = DDI(name="DrugBank", path=data_dir)
    except TypeError:
        data = DDI(name="DrugBank")
    try:
        label_map = get_label_map(name="DrugBank", task="DDI", path=data_dir)
    except TypeError:
        # Compatibility fallback for older signatures.
        label_map = get_label_map(name="DrugBank", task="DDI")

    split = data.get_split()
    if not isinstance(split, dict):
        raise ValueError(f"TDC get_split() returned unexpected type: {type(split)}")

    id_to_smiles = _infer_id_to_smiles_mapping(data)
    LOGGER.info("Inferred id->SMILES mapping entries: %d", len(id_to_smiles))

    try:
        train_raw = split["train"]
        valid_raw = split.get("valid", split.get("val"))
        test_raw = split["test"]
    except KeyError as e:
        raise KeyError(f"Unexpected split keys from TDC: {list(split.keys())}") from e
    if valid_raw is None:
        raise KeyError(f"Validation split missing in TDC split keys: {list(split.keys())}")

    train_df = _normalize_split_df(train_raw, id_to_smiles)
    valid_df = _normalize_split_df(valid_raw, id_to_smiles)
    test_df = _normalize_split_df(test_raw, id_to_smiles)

    if strategy == "cold_drug":
        full_df = pd.concat([train_df, valid_df, test_df], axis=0, ignore_index=True).drop_duplicates()
        train_df, valid_df, test_df = _make_cold_drug_split(full_df=full_df, seed=split_seed)

    train_df, valid_df, test_df, label_map = _normalize_label_indexing(
        train_df, valid_df, test_df, label_map, split_dir=None
    )

    train_df.to_csv(split_dir / "train.csv", index=False)
    valid_df.to_csv(split_dir / "valid.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)
    save_json({str(k): str(v) for k, v in label_map.items()}, split_dir / "label_map.json")
    _save_split_meta(split_dir=split_dir, split_strategy=strategy, split_seed=split_seed)
    _log_split_label_coverage(train_df, valid_df, test_df, num_classes=len(label_map))

    LOGGER.info(
        "Saved normalized %s splits to %s | train=%d valid=%d test=%d",
        strategy,
        split_dir,
        len(train_df),
        len(valid_df),
        len(test_df),
    )
    return train_df, valid_df, test_df, label_map
