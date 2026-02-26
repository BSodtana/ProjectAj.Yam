from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import pandas as pd

from ddigat.utils.io import ensure_dir, save_json
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


def load_tdc_drugbank_ddi(
    data_dir: str,
    output_dir: str = "outputs",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load TDC DrugBank DDI splits and normalize to standard columns."""
    split_dir = ensure_dir(Path(output_dir) / "splits")
    saved = _load_saved_splits(split_dir)

    try:
        from tdc.multi_pred import DDI
        from tdc.utils import get_label_map
    except ImportError as e:  # pragma: no cover
        raise ImportError("Please install `tdc` to load the DrugBank DDI dataset.") from e

    label_map = get_label_map(name="DrugBank", task="DDI")

    if saved is not None:
        LOGGER.info("Loaded persisted splits from %s", split_dir)
        return (*saved, label_map)

    ensure_dir(data_dir)
    try:
        data = DDI(name="DrugBank", path=data_dir)
    except TypeError:
        data = DDI(name="DrugBank")

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

    train_df.to_csv(split_dir / "train.csv", index=False)
    valid_df.to_csv(split_dir / "valid.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)
    save_json({str(k): str(v) for k, v in label_map.items()}, split_dir / "label_map.json")

    LOGGER.info(
        "Saved normalized splits to %s | train=%d valid=%d test=%d",
        split_dir,
        len(train_df),
        len(valid_df),
        len(test_df),
    )
    return train_df, valid_df, test_df, label_map

