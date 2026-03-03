from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from ddigat.utils.io import ensure_dir, load_json, save_json
from ddigat.utils.logging import get_logger


LOGGER = get_logger(__name__)

EXPECTED_COLS = ["drug_a_smiles", "drug_b_smiles", "y"]
SPLIT_IMPL_VERSION_V2 = 2
SPLIT_IMPL_VERSION_V3 = 3
PAIR_KEY_SEP = "||"


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


def _save_split_meta(split_dir: Path, payload: dict) -> None:
    body = {"schema": EXPECTED_COLS}
    body.update(payload)
    save_json(body, split_dir / "meta.json")


def _meta_matches(meta: dict, expected: dict) -> bool:
    for k, v in expected.items():
        if meta.get(k) != v:
            return False
    return True


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


def _load_label_map_from_tdc(data_dir: str) -> dict:
    try:
        from tdc.utils import get_label_map
    except ImportError as e:  # pragma: no cover
        raise ImportError("Please install `tdc` to load the DrugBank DDI dataset.") from e

    try:
        label_map = get_label_map(name="DrugBank", task="DDI", path=data_dir)
    except TypeError:
        # Compatibility fallback for older signatures.
        label_map = get_label_map(name="DrugBank", task="DDI")
    return _coerce_label_map_keys(label_map)


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


def _pair_keys_for_df(df: pd.DataFrame) -> pd.Series:
    a = df["drug_a_smiles"].astype(str)
    b = df["drug_b_smiles"].astype(str)
    return a + PAIR_KEY_SEP + b


def _canonicalize_pairs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["drug_a_smiles"] = out["drug_a_smiles"].astype(str)
    out["drug_b_smiles"] = out["drug_b_smiles"].astype(str)
    out["y"] = out["y"].astype(int)
    a = out["drug_a_smiles"].to_numpy(dtype=object)
    b = out["drug_b_smiles"].to_numpy(dtype=object)
    out["drug_a_smiles"] = np.where(a <= b, a, b)
    out["drug_b_smiles"] = np.where(a <= b, b, a)
    out["pair_key"] = _pair_keys_for_df(out)
    return out


def _prepare_pair_groups_for_cold_drug(
    full_df: pd.DataFrame,
    dedupe_policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if dedupe_policy not in {"keep_all", "keep_first"}:
        raise ValueError(f"Unsupported dedupe policy: {dedupe_policy}. Expected keep_all or keep_first.")

    canonical_rows = _canonicalize_pairs(full_df).reset_index(drop=True)
    grouped = canonical_rows.groupby("pair_key", sort=True)

    first_rows = grouped[["drug_a_smiles", "drug_b_smiles", "y"]].first().reset_index()
    label_nunique = grouped["y"].nunique().rename("label_nunique").reset_index()
    multiplicity = grouped.size().rename("multiplicity").reset_index()

    pair_df = (
        first_rows.merge(label_nunique, on="pair_key", how="left")
        .merge(multiplicity, on="pair_key", how="left")
        .sort_values("pair_key")
        .reset_index(drop=True)
    )
    pair_df["label_nunique"] = pair_df["label_nunique"].astype(int)
    pair_df["multiplicity"] = pair_df["multiplicity"].astype(int)
    pair_df["y"] = pair_df["y"].astype(int)

    conflict_mask = pair_df["label_nunique"] > 1
    conflict_keys = set(pair_df.loc[conflict_mask, "pair_key"].tolist())
    clean_pair_df = pair_df.loc[~conflict_mask, ["pair_key", "drug_a_smiles", "drug_b_smiles", "y", "multiplicity"]].copy()

    clean_rows = canonical_rows.loc[~canonical_rows["pair_key"].isin(conflict_keys)].copy()
    rows_before_dedupe = int(len(clean_rows))
    if dedupe_policy == "keep_first":
        clean_rows = (
            clean_rows.sort_values(["pair_key", "y", "drug_a_smiles", "drug_b_smiles"], kind="mergesort")
            .drop_duplicates(subset=["pair_key"], keep="first")
            .copy()
        )
        clean_pair_df["row_weight"] = 1
    else:
        clean_pair_df["row_weight"] = clean_pair_df["multiplicity"].astype(int)

    hist_counter = Counter(clean_pair_df["multiplicity"].astype(int).tolist())
    preprocess_stats = {
        "input_rows": int(len(canonical_rows)),
        "input_pair_groups": int(len(pair_df)),
        "conflict_pair_groups_dropped": int(conflict_mask.sum()),
        "conflict_rows_dropped": int(canonical_rows["pair_key"].isin(conflict_keys).sum()),
        "kept_pair_groups": int(len(clean_pair_df)),
        "kept_rows_after_policy": int(len(clean_rows)),
        "dedupe_policy": dedupe_policy,
        "dedupe_rows_removed": int(rows_before_dedupe - len(clean_rows)),
        "pair_multiplicity_hist": {str(k): int(v) for k, v in sorted(hist_counter.items())},
    }
    return clean_rows.reset_index(drop=True), clean_pair_df.reset_index(drop=True), preprocess_stats


def _compute_drug_weights(pair_df: pd.DataFrame) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in pair_df[["drug_a_smiles", "drug_b_smiles"]].itertuples(index=False):
        counts[str(row[0])] += 1
        counts[str(row[1])] += 1
    return {k: int(v) for k, v in counts.items()}


def _assign_drugs_to_folds_degree_aware(
    drug_weights: dict[str, int],
    k: int,
    seed: int,
    attempt: int,
) -> tuple[dict[str, int], list[set[str]], list[int], list[int]]:
    if k <= 1:
        raise ValueError("k must be > 1")
    if not drug_weights:
        raise ValueError("No drugs available for fold assignment.")

    rng = np.random.default_rng(int(seed) + (104_729 * int(attempt)))
    drugs = sorted(drug_weights.keys())
    tie_perm = rng.permutation(len(drugs))
    tie_rank = {drugs[int(idx)]: int(rank) for rank, idx in enumerate(tie_perm.tolist())}

    ordered = sorted(
        drugs,
        key=lambda d: (-int(drug_weights[d]), tie_rank[d], d),
    )
    fold_tie = rng.permutation(k).tolist()
    fold_tie_rank = {int(f): int(r) for r, f in enumerate(fold_tie)}

    fold_weights = np.zeros(k, dtype=np.int64)
    fold_counts = np.zeros(k, dtype=np.int64)
    folds: list[set[str]] = [set() for _ in range(k)]

    for drug in ordered:
        fold_idx = min(
            range(k),
            key=lambda f: (int(fold_weights[f]), int(fold_counts[f]), fold_tie_rank[int(f)]),
        )
        folds[fold_idx].add(drug)
        fold_weights[fold_idx] += int(drug_weights[drug])
        fold_counts[fold_idx] += 1

    drug_to_fold: dict[str, int] = {}
    for fold_idx, members in enumerate(folds):
        for drug in members:
            drug_to_fold[drug] = int(fold_idx)
    return drug_to_fold, folds, [int(v) for v in fold_weights.tolist()], [int(v) for v in fold_counts.tolist()]


def _label_coverage(labels: np.ndarray, num_classes: int) -> dict[str, object]:
    if labels.size == 0:
        present = []
    else:
        present = sorted(np.unique(labels.astype(int)).tolist())
    present_set = set(int(v) for v in present)
    missing_ids = [int(c) for c in range(int(num_classes)) if c not in present_set]
    return {
        "present": int(len(present)),
        "missing": int(len(missing_ids)),
        "missing_ids": missing_ids,
    }


def _build_fold_masks_s1(
    pair_df: pd.DataFrame,
    drug_to_fold: dict[str, int],
    fold_idx: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_fold = (int(fold_idx) + 1) % int(k)
    a = pair_df["drug_a_smiles"].astype(str).to_numpy()
    b = pair_df["drug_b_smiles"].astype(str).to_numpy()
    a_fold = np.fromiter((drug_to_fold[s] for s in a), dtype=np.int64, count=len(a))
    b_fold = np.fromiter((drug_to_fold[s] for s in b), dtype=np.int64, count=len(b))

    a_known = (a_fold != fold_idx) & (a_fold != valid_fold)
    b_known = (b_fold != fold_idx) & (b_fold != valid_fold)
    train_mask = a_known & b_known

    valid_mask = ((a_fold == valid_fold) & b_known) | ((b_fold == valid_fold) & a_known)
    test_mask = ((a_fold == fold_idx) & b_known) | ((b_fold == fold_idx) & a_known)
    return train_mask, valid_mask, test_mask


def _build_fold_stats(
    pair_df: pd.DataFrame,
    folds: list[set[str]],
    fold_idx: int,
    k: int,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    test_mask: np.ndarray,
    num_classes: int,
) -> dict[str, object]:
    keys = pair_df["pair_key"].astype(str).to_numpy()
    labels = pair_df["y"].astype(int).to_numpy()
    row_weight = pair_df["row_weight"].astype(int).to_numpy()

    train_keys = set(keys[train_mask].tolist())
    valid_keys = set(keys[valid_mask].tolist())
    test_keys = set(keys[test_mask].tolist())
    if not train_keys.isdisjoint(valid_keys):
        raise AssertionError("train and valid pair_key overlap detected.")
    if not train_keys.isdisjoint(test_keys):
        raise AssertionError("train and test pair_key overlap detected.")
    if not valid_keys.isdisjoint(test_keys):
        raise AssertionError("valid and test pair_key overlap detected.")

    valid_fold = (int(fold_idx) + 1) % int(k)

    train_labels = labels[train_mask]
    valid_labels = labels[valid_mask]
    test_labels = labels[test_mask]
    train_cov = _label_coverage(train_labels, num_classes=num_classes)
    valid_cov = _label_coverage(valid_labels, num_classes=num_classes)
    test_cov = _label_coverage(test_labels, num_classes=num_classes)
    return {
        "fold": int(fold_idx),
        "valid_fold": int(valid_fold),
        "n_drugs": int(len(folds[fold_idx])),
        "train_rows": int(row_weight[train_mask].sum()),
        "valid_rows": int(row_weight[valid_mask].sum()),
        "test_rows": int(row_weight[test_mask].sum()),
        "train_pairs": int(train_mask.sum()),
        "valid_pairs": int(valid_mask.sum()),
        "test_pairs": int(test_mask.sum()),
        "train_labels_present": int(train_cov["present"]),
        "train_labels_missing": int(train_cov["missing"]),
        "valid_labels_present": int(valid_cov["present"]),
        "valid_labels_missing": int(valid_cov["missing"]),
        "test_labels_present": int(test_cov["present"]),
        "test_labels_missing": int(test_cov["missing"]),
        "train_missing_label_ids": train_cov["missing_ids"],
        "valid_missing_label_ids": valid_cov["missing_ids"],
        "test_missing_label_ids": test_cov["missing_ids"],
    }


def _mean_std(values: list[int]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def _evaluate_assignment_s1(
    pair_df: pd.DataFrame,
    drug_to_fold: dict[str, int],
    folds: list[set[str]],
    k: int,
    num_classes: int,
) -> list[dict[str, object]]:
    fold_stats: list[dict[str, object]] = []
    for fold_idx in range(int(k)):
        train_mask, valid_mask, test_mask = _build_fold_masks_s1(
            pair_df=pair_df,
            drug_to_fold=drug_to_fold,
            fold_idx=fold_idx,
            k=k,
        )
        stats = _build_fold_stats(
            pair_df=pair_df,
            folds=folds,
            fold_idx=fold_idx,
            k=k,
            train_mask=train_mask,
            valid_mask=valid_mask,
            test_mask=test_mask,
            num_classes=num_classes,
        )
        fold_stats.append(stats)
    return fold_stats


def _build_selected_fold_split_s1(
    rows_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    drug_to_fold: dict[str, int],
    fold_idx: int,
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_mask, valid_mask, test_mask = _build_fold_masks_s1(
        pair_df=pair_df,
        drug_to_fold=drug_to_fold,
        fold_idx=fold_idx,
        k=k,
    )
    keys = pair_df["pair_key"].astype(str).to_numpy()
    split_label = np.full(shape=(len(pair_df),), fill_value="excluded", dtype=object)
    split_label[train_mask] = "train"
    split_label[valid_mask] = "valid"
    split_label[test_mask] = "test"
    role_map = {str(k): str(v) for k, v in zip(keys.tolist(), split_label.tolist()) if str(v) != "excluded"}

    row_roles = rows_df["pair_key"].map(role_map).fillna("excluded")
    train_df = rows_df.loc[row_roles == "train", EXPECTED_COLS].copy().reset_index(drop=True)
    valid_df = rows_df.loc[row_roles == "valid", EXPECTED_COLS].copy().reset_index(drop=True)
    test_df = rows_df.loc[row_roles == "test", EXPECTED_COLS].copy().reset_index(drop=True)

    pair_key_train = set(rows_df.loc[row_roles == "train", "pair_key"].astype(str).tolist())
    pair_key_valid = set(rows_df.loc[row_roles == "valid", "pair_key"].astype(str).tolist())
    pair_key_test = set(rows_df.loc[row_roles == "test", "pair_key"].astype(str).tolist())
    if not pair_key_train.isdisjoint(pair_key_valid):
        raise AssertionError("train and valid overlap on unordered pair keys.")
    if not pair_key_train.isdisjoint(pair_key_test):
        raise AssertionError("train and test overlap on unordered pair keys.")
    if not pair_key_valid.isdisjoint(pair_key_test):
        raise AssertionError("valid and test overlap on unordered pair keys.")
    return train_df, valid_df, test_df


def _make_cold_drug_split_v3(
    full_df: pd.DataFrame,
    seed: int,
    k: int,
    fold_idx: int,
    protocol: str,
    min_test_pairs: int,
    min_test_labels: int,
    max_resamples: int,
    dedupe_policy: str,
    num_classes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if int(k) < 3:
        raise ValueError("cold_k must be >= 3 for S1 fold construction (train/valid/test).")
    if int(fold_idx) < 0 or int(fold_idx) >= int(k):
        raise ValueError(f"cold_fold must be in [0, {int(k) - 1}], got {fold_idx}")
    if int(max_resamples) <= 0:
        raise ValueError("cold_max_resamples must be > 0")
    if protocol.lower().strip() == "s2":
        raise NotImplementedError("cold_protocol='s2' is not implemented yet. Use cold_protocol='s1'.")
    if protocol.lower().strip() != "s1":
        raise ValueError(f"Unsupported cold_protocol={protocol}. Expected 's1' or 's2'.")

    rows_df, pair_df, preprocess_stats = _prepare_pair_groups_for_cold_drug(
        full_df=full_df,
        dedupe_policy=dedupe_policy,
    )
    if len(pair_df) == 0:
        raise RuntimeError("No usable pairs after ambiguity filtering.")
    drug_weights = _compute_drug_weights(pair_df)
    if len(drug_weights) < int(k):
        raise RuntimeError(
            f"Too few drugs ({len(drug_weights)}) for cold_k={k} after ambiguity filtering."
        )

    accepted: Optional[dict[str, Any]] = None
    best_attempt: Optional[dict[str, Any]] = None
    for attempt in range(int(max_resamples)):
        drug_to_fold, folds, fold_weight_totals, fold_drug_counts = _assign_drugs_to_folds_degree_aware(
            drug_weights=drug_weights,
            k=int(k),
            seed=int(seed),
            attempt=int(attempt),
        )
        fold_stats = _evaluate_assignment_s1(
            pair_df=pair_df,
            drug_to_fold=drug_to_fold,
            folds=folds,
            k=int(k),
            num_classes=int(num_classes),
        )
        fold_pass = []
        for stat in fold_stats:
            ok = (
                int(stat["test_rows"]) >= int(min_test_pairs)
                and int(stat["test_labels_present"]) >= int(min_test_labels)
                and int(stat["train_rows"]) > 0
                and int(stat["valid_rows"]) > 0
                and int(stat["test_rows"]) > 0
            )
            stat["guardrail_pass"] = bool(ok)
            fold_pass.append(bool(ok))

        candidate = {
            "attempt": int(attempt),
            "drug_to_fold": drug_to_fold,
            "folds": folds,
            "fold_weight_totals": fold_weight_totals,
            "fold_drug_counts": fold_drug_counts,
            "fold_stats": fold_stats,
            "min_test_rows": int(min(int(s["test_rows"]) for s in fold_stats)),
            "min_test_labels": int(min(int(s["test_labels_present"]) for s in fold_stats)),
            "sum_test_rows": int(sum(int(s["test_rows"]) for s in fold_stats)),
            "all_guardrails_passed": bool(all(fold_pass)),
        }
        if best_attempt is None:
            best_attempt = candidate
        else:
            best_score = (
                int(best_attempt["min_test_rows"]),
                int(best_attempt["min_test_labels"]),
                int(best_attempt["sum_test_rows"]),
            )
            cand_score = (candidate["min_test_rows"], candidate["min_test_labels"], candidate["sum_test_rows"])
            if cand_score > best_score:
                best_attempt = candidate
        if candidate["all_guardrails_passed"]:
            accepted = candidate
            break

    if accepted is None:
        if best_attempt is None:
            raise RuntimeError("Unable to build any cold-drug assignment.")
        raise RuntimeError(
            "Unable to satisfy cold-drug guardrails after "
            f"{max_resamples} attempts | required test_rows>={min_test_pairs}, "
            f"required test_labels>={min_test_labels}, "
            f"best_min_test_rows={best_attempt['min_test_rows']}, "
            f"best_min_test_labels={best_attempt['min_test_labels']}"
        )

    train_df, valid_df, test_df = _build_selected_fold_split_s1(
        rows_df=rows_df,
        pair_df=pair_df,
        drug_to_fold=accepted["drug_to_fold"],
        fold_idx=int(fold_idx),
        k=int(k),
    )
    fold_stats = accepted["fold_stats"]
    summary = {
        "test_rows": _mean_std([int(s["test_rows"]) for s in fold_stats]),
        "test_pairs": _mean_std([int(s["test_pairs"]) for s in fold_stats]),
        "test_labels_present": _mean_std([int(s["test_labels_present"]) for s in fold_stats]),
        "valid_rows": _mean_std([int(s["valid_rows"]) for s in fold_stats]),
        "train_rows": _mean_std([int(s["train_rows"]) for s in fold_stats]),
    }
    report = {
        "split_strategy": "cold_drug",
        "split_impl_version": int(SPLIT_IMPL_VERSION_V3),
        "seed": int(seed),
        "cold_k": int(k),
        "cold_fold": int(fold_idx),
        "cold_protocol": str(protocol),
        "cold_dedupe_policy": str(dedupe_policy),
        "guardrails": {
            "min_test_pairs": int(min_test_pairs),
            "min_test_labels": int(min_test_labels),
            "max_resamples": int(max_resamples),
        },
        "preprocess": preprocess_stats,
        "assignment": {
            "accepted_attempt": int(accepted["attempt"]),
            "attempts_tried": int(accepted["attempt"]) + 1,
            "fold_weight_totals": [int(v) for v in accepted["fold_weight_totals"]],
            "fold_drug_counts": [int(v) for v in accepted["fold_drug_counts"]],
        },
        "folds": fold_stats,
        "summary": summary,
    }
    return train_df, valid_df, test_df, report


def _make_cold_drug_split_v2(
    full_df: pd.DataFrame,
    seed: int,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    trials: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Legacy cold-drug v2 split behavior, preserved for reproducibility."""
    if not np.isclose(train_ratio + valid_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.")

    drugs = sorted(set(full_df["drug_a_smiles"]).union(set(full_df["drug_b_smiles"])))
    if len(drugs) < 10:
        raise ValueError("Too few unique drugs for cold-drug split.")

    n_total = len(drugs)
    n_train = max(1, int(n_total * train_ratio))
    n_holdout = max(1, n_total - n_train)

    best: Optional[tuple[pd.DataFrame, pd.DataFrame]] = None
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
        raise RuntimeError("Unable to create non-empty cold-drug v2 split.")

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
        raise RuntimeError(f"Cold-drug v2 invariant violated: {len(overlap)} test drugs also in train.")
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

    if y_min == 0 and y_max <= num_classes - 1:
        return frames[0], frames[1], frames[2], label_map

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


def _load_raw_tdc_split(
    data_dir: str,
) -> tuple[dict[str, Any], dict[str, str], dict]:
    try:
        from tdc.multi_pred import DDI
    except ImportError as e:  # pragma: no cover
        raise ImportError("Please install `tdc` to load the DrugBank DDI dataset.") from e

    ensure_dir(data_dir)
    try:
        data = DDI(name="DrugBank", path=data_dir)
    except TypeError:
        data = DDI(name="DrugBank")

    label_map = _load_label_map_from_tdc(data_dir)
    split = data.get_split()
    if not isinstance(split, dict):
        raise ValueError(f"TDC get_split() returned unexpected type: {type(split)}")
    id_to_smiles = _infer_id_to_smiles_mapping(data)
    LOGGER.info("Inferred id->SMILES mapping entries: %d", len(id_to_smiles))
    return split, id_to_smiles, label_map


def load_tdc_drugbank_ddi(
    data_dir: str,
    output_dir: str = "outputs",
    split_strategy: str = "cold_drug",
    split_seed: int = 42,
    cold_k: int = 5,
    cold_fold: int = 0,
    cold_protocol: str = "s1",
    cold_min_test_pairs: int = 5000,
    cold_min_test_labels: int = 45,
    cold_max_resamples: int = 200,
    cold_dedupe_policy: str = "keep_all",
    cold_write_legacy_flat_splits: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load TDC DrugBank DDI splits and normalize to standard columns.

    split_strategy:
      - "cold_drug": v3 group-aware K-fold cold-drug split (default)
      - "cold_drug_v2": legacy split behavior for reproducibility
      - "tdc": use TDC-provided split directly
    """
    strategy = split_strategy.lower().strip()
    if strategy not in {"cold_drug", "cold_drug_v2", "tdc"}:
        raise ValueError(f"Unsupported split_strategy={split_strategy}. Expected one of: cold_drug, cold_drug_v2, tdc")

    cold_protocol = str(cold_protocol).lower().strip()
    cold_dedupe_policy = str(cold_dedupe_policy).lower().strip()
    splits_root = ensure_dir(Path(output_dir) / "splits")

    if strategy == "cold_drug":
        if cold_protocol not in {"s1", "s2"}:
            raise ValueError(f"Unsupported cold_protocol={cold_protocol}. Expected one of: s1, s2")
        if cold_dedupe_policy not in {"keep_all", "keep_first"}:
            raise ValueError(
                f"Unsupported cold_dedupe_policy={cold_dedupe_policy}. Expected one of: keep_all, keep_first"
            )
        if int(cold_k) < 3:
            raise ValueError(f"cold_k must be >= 3, got {cold_k}")
        if int(cold_fold) < 0 or int(cold_fold) >= int(cold_k):
            raise ValueError(f"cold_fold must be in [0, {int(cold_k) - 1}], got {cold_fold}")

        split_root = ensure_dir(
            splits_root / "cold_drug_v3" / f"seed_{int(split_seed)}" / f"k_{int(cold_k)}"
        )
        fold_dir = ensure_dir(split_root / f"fold_{int(cold_fold)}")
        expected_meta = {
            "split_strategy": "cold_drug",
            "split_seed": int(split_seed),
            "split_impl_version": int(SPLIT_IMPL_VERSION_V3),
            "cold_k": int(cold_k),
            "cold_protocol": cold_protocol,
            "cold_min_test_pairs": int(cold_min_test_pairs),
            "cold_min_test_labels": int(cold_min_test_labels),
            "cold_max_resamples": int(cold_max_resamples),
            "cold_dedupe_policy": cold_dedupe_policy,
        }
        saved = _load_saved_splits(fold_dir)
        meta = _load_split_meta(split_root)
        if saved is not None and _meta_matches(meta, expected_meta):
            label_map = _load_saved_label_map(split_root)
            if label_map is None:
                label_map = _load_label_map_from_tdc(data_dir)
            train_df, valid_df, test_df = saved
            train_df, valid_df, test_df, label_map = _normalize_label_indexing(
                train_df, valid_df, test_df, label_map, split_dir=None
            )
            _log_split_label_coverage(train_df, valid_df, test_df, num_classes=len(label_map))
            LOGGER.info("Loaded persisted cold_drug v3 split from %s", fold_dir)
            return train_df, valid_df, test_df, label_map
    else:
        simple_dir = ensure_dir(splits_root / strategy)
        expected_meta = {
            "split_strategy": strategy,
            "split_seed": int(split_seed),
            "split_impl_version": int(SPLIT_IMPL_VERSION_V2),
        }
        saved = _load_saved_splits(simple_dir)
        meta = _load_split_meta(simple_dir)
        if saved is not None and _meta_matches(meta, expected_meta):
            label_map = _load_saved_label_map(simple_dir)
            if label_map is None:
                label_map = _load_label_map_from_tdc(data_dir)
            train_df, valid_df, test_df = saved
            train_df, valid_df, test_df, label_map = _normalize_label_indexing(
                train_df, valid_df, test_df, label_map, split_dir=simple_dir
            )
            _log_split_label_coverage(train_df, valid_df, test_df, num_classes=len(label_map))
            LOGGER.info("Loaded persisted %s splits from %s", strategy, simple_dir)
            return train_df, valid_df, test_df, label_map

        # Reproducibility escape hatch for older flat v2 cache layout.
        if strategy == "cold_drug_v2":
            legacy_meta = _load_split_meta(splits_root)
            legacy_saved = _load_saved_splits(splits_root)
            legacy_ok = (
                legacy_saved is not None
                and int(legacy_meta.get("split_impl_version", 0)) == int(SPLIT_IMPL_VERSION_V2)
                and int(legacy_meta.get("split_seed", split_seed)) == int(split_seed)
                and str(legacy_meta.get("split_strategy", "")).lower().strip() in {"cold_drug", "cold_drug_v2"}
            )
            if legacy_ok:
                label_map = _load_saved_label_map(splits_root)
                if label_map is None:
                    label_map = _load_label_map_from_tdc(data_dir)
                train_df, valid_df, test_df = legacy_saved
                train_df, valid_df, test_df, label_map = _normalize_label_indexing(
                    train_df, valid_df, test_df, label_map, split_dir=splits_root
                )
                _log_split_label_coverage(train_df, valid_df, test_df, num_classes=len(label_map))
                LOGGER.info("Loaded legacy cold_drug v2 splits from %s", splits_root)
                return train_df, valid_df, test_df, label_map

    split, id_to_smiles, label_map = _load_raw_tdc_split(data_dir=data_dir)
    try:
        train_raw = split["train"]
        valid_raw = split.get("valid", split.get("val"))
        test_raw = split["test"]
    except KeyError as e:
        raise KeyError(f"Unexpected split keys from TDC: {list(split.keys())}") from e
    if valid_raw is None:
        raise KeyError(f"Validation split missing in TDC split keys: {list(split.keys())}")

    base_train_df = _normalize_split_df(train_raw, id_to_smiles)
    base_valid_df = _normalize_split_df(valid_raw, id_to_smiles)
    base_test_df = _normalize_split_df(test_raw, id_to_smiles)
    base_train_df, base_valid_df, base_test_df, label_map = _normalize_label_indexing(
        base_train_df, base_valid_df, base_test_df, label_map, split_dir=None
    )

    if strategy == "cold_drug":
        full_df = pd.concat([base_train_df, base_valid_df, base_test_df], axis=0, ignore_index=True)
        train_df, valid_df, test_df, report = _make_cold_drug_split_v3(
            full_df=full_df,
            seed=int(split_seed),
            k=int(cold_k),
            fold_idx=int(cold_fold),
            protocol=cold_protocol,
            min_test_pairs=int(cold_min_test_pairs),
            min_test_labels=int(cold_min_test_labels),
            max_resamples=int(cold_max_resamples),
            dedupe_policy=cold_dedupe_policy,
            num_classes=len(label_map),
        )
        train_df, valid_df, test_df, label_map = _normalize_label_indexing(
            train_df, valid_df, test_df, label_map, split_dir=None
        )
        split_root = ensure_dir(splits_root / "cold_drug_v3" / f"seed_{int(split_seed)}" / f"k_{int(cold_k)}")
        fold_dir = ensure_dir(split_root / f"fold_{int(cold_fold)}")
        train_df.to_csv(fold_dir / "train.csv", index=False)
        valid_df.to_csv(fold_dir / "valid.csv", index=False)
        test_df.to_csv(fold_dir / "test.csv", index=False)
        save_json({str(k): str(v) for k, v in label_map.items()}, split_root / "label_map.json")
        _save_split_meta(
            split_dir=split_root,
            payload={
                "split_strategy": "cold_drug",
                "split_seed": int(split_seed),
                "split_impl_version": int(SPLIT_IMPL_VERSION_V3),
                "cold_k": int(cold_k),
                "cold_protocol": cold_protocol,
                "cold_min_test_pairs": int(cold_min_test_pairs),
                "cold_min_test_labels": int(cold_min_test_labels),
                "cold_max_resamples": int(cold_max_resamples),
                "cold_dedupe_policy": cold_dedupe_policy,
            },
        )
        save_json(report, split_root / "cold_drug_kfold_report.json")
        if cold_write_legacy_flat_splits:
            train_df.to_csv(splits_root / "train.csv", index=False)
            valid_df.to_csv(splits_root / "valid.csv", index=False)
            test_df.to_csv(splits_root / "test.csv", index=False)
            save_json({str(k): str(v) for k, v in label_map.items()}, splits_root / "label_map.json")
            _save_split_meta(
                split_dir=splits_root,
                payload={
                    "split_strategy": "cold_drug",
                    "split_seed": int(split_seed),
                    "split_impl_version": int(SPLIT_IMPL_VERSION_V3),
                    "cold_k": int(cold_k),
                    "cold_fold": int(cold_fold),
                    "cold_protocol": cold_protocol,
                    "cold_min_test_pairs": int(cold_min_test_pairs),
                    "cold_min_test_labels": int(cold_min_test_labels),
                    "cold_max_resamples": int(cold_max_resamples),
                    "cold_dedupe_policy": cold_dedupe_policy,
                    "legacy_flat_copy": True,
                },
            )
        _log_split_label_coverage(train_df, valid_df, test_df, num_classes=len(label_map))
        LOGGER.info(
            "Saved cold_drug v3 split | root=%s fold=%d | train=%d valid=%d test=%d",
            split_root,
            int(cold_fold),
            len(train_df),
            len(valid_df),
            len(test_df),
        )
        return train_df, valid_df, test_df, label_map

    if strategy == "cold_drug_v2":
        full_df = pd.concat([base_train_df, base_valid_df, base_test_df], axis=0, ignore_index=True).drop_duplicates()
        train_df, valid_df, test_df = _make_cold_drug_split_v2(full_df=full_df, seed=int(split_seed))
    else:
        train_df, valid_df, test_df = base_train_df, base_valid_df, base_test_df

    train_df, valid_df, test_df, label_map = _normalize_label_indexing(
        train_df, valid_df, test_df, label_map, split_dir=None
    )
    simple_dir = ensure_dir(splits_root / strategy)
    train_df.to_csv(simple_dir / "train.csv", index=False)
    valid_df.to_csv(simple_dir / "valid.csv", index=False)
    test_df.to_csv(simple_dir / "test.csv", index=False)
    save_json({str(k): str(v) for k, v in label_map.items()}, simple_dir / "label_map.json")
    _save_split_meta(
        split_dir=simple_dir,
        payload={
            "split_strategy": strategy,
            "split_seed": int(split_seed),
            "split_impl_version": int(SPLIT_IMPL_VERSION_V2),
        },
    )
    _log_split_label_coverage(train_df, valid_df, test_df, num_classes=len(label_map))
    LOGGER.info(
        "Saved normalized %s splits to %s | train=%d valid=%d test=%d",
        strategy,
        simple_dir,
        len(train_df),
        len(valid_df),
        len(test_df),
    )
    return train_df, valid_df, test_df, label_map
