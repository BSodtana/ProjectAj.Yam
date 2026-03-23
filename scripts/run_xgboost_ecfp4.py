#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ddigat.data.cache import DrugFeatureCache
from ddigat.data.featurize import canonicalize_smiles
from ddigat.data.splits import subsample_dataframe
from ddigat.data.tdc_ddi import load_tdc_drugbank_ddi
from ddigat.utils.class_weights import compute_class_counts, compute_tail_class_ids
from ddigat.utils.io import ensure_dir, save_json
from ddigat.utils.logging import get_logger
from ddigat.utils.metrics import evaluate_multiclass_metrics, multiclass_nll_from_probs
from ddigat.utils.seed import seed_everything


LOGGER = get_logger("scripts.run_xgboost_ecfp4")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train and evaluate an XGBoost multiclass baseline on ECFP4 pair features."
    )
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./outputs/xgboost_ecfp4")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional dataset subsample. Applied per split in single_split mode and to the full dataset in stratified_kfold mode.",
    )
    p.add_argument("--ece_bins", type=int, default=15)
    p.add_argument("--n_estimators", type=int, default=600)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--max_depth", type=int, default=8)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample_bytree", type=float, default=0.8)
    p.add_argument("--min_child_weight", type=float, default=1.0)
    p.add_argument("--reg_lambda", type=float, default=1.0)
    p.add_argument("--tree_method", type=str, default="hist")
    p.add_argument("--early_stopping_rounds", type=int, default=30)
    p.add_argument("--n_jobs", type=int, default=0, help="0 delegates to XGBoost default thread selection.")
    p.add_argument(
        "--run_mode",
        type=str,
        default="single_split",
        choices=["single_split", "stratified_kfold", "cold_drug_kfold"],
    )
    p.add_argument(
        "--split_strategy",
        type=str,
        default="cold_drug",
        choices=["cold_drug", "cold_drug_v2", "tdc"],
        help=(
            "Applies only to single_split mode. stratified_kfold always rebuilds folds from the normalized full "
            "TDC dataset, and cold_drug_kfold always uses the project cold_drug s1 v3 splitter."
        ),
    )
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
    p.add_argument(
        "--split_cache_dir",
        type=str,
        default=None,
        help="Optional shared directory used for persisted split cache; defaults to output_dir.",
    )
    p.add_argument("--cv_k", type=int, default=5, help="Number of outer folds when run_mode=stratified_kfold.")
    p.add_argument(
        "--cv_valid_fraction",
        type=float,
        default=0.1,
        help="Fraction of each outer-train fold reserved for early stopping when run_mode=stratified_kfold.",
    )
    p.add_argument("--ecfp_bits", type=int, default=2048)
    p.add_argument("--ecfp_radius", type=int, default=2)
    return p.parse_args()


def _require_xgboost():
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError(
            "xgboost is required for scripts/run_xgboost_ecfp4.py. "
            "Install it with `./.venv/bin/pip install xgboost` or add it to your environment first."
        ) from exc
    return XGBClassifier


def _canonical_pair(smiles_a: str, smiles_b: str) -> tuple[str, str] | None:
    canon_a = canonicalize_smiles(str(smiles_a))
    canon_b = canonicalize_smiles(str(smiles_b))
    if canon_a is None or canon_b is None:
        return None
    if canon_a <= canon_b:
        return canon_a, canon_b
    return canon_b, canon_a


def _build_feature_bank(smiles_values: list[str], feature_cache: DrugFeatureCache) -> tuple[sparse.csr_matrix, dict[str, int]]:
    canonical_smiles = sorted(
        {
            canonical
            for smiles in smiles_values
            for canonical in [canonicalize_smiles(str(smiles))]
            if canonical is not None
        }
    )
    if not canonical_smiles:
        raise ValueError("No valid SMILES found for ECFP4 feature-bank construction.")

    rows: list[sparse.csr_matrix] = []
    index: dict[str, int] = {}
    for row_idx, canonical in enumerate(canonical_smiles):
        feat = feature_cache.get_or_create(canonical)
        if feat is None:
            raise ValueError(f"Feature cache could not build ECFP4 for {canonical!r}")
        rows.append(sparse.csr_matrix(feat.reshape(1, -1), dtype=np.float32))
        index[canonical] = int(row_idx)
    bank = sparse.vstack(rows, format="csr", dtype=np.float32)
    return bank, index


def _build_pair_matrix(
    df: pd.DataFrame,
    *,
    feature_bank: sparse.csr_matrix,
    feature_index: dict[str, int],
) -> tuple[sparse.csr_matrix, pd.DataFrame, int]:
    left_ids: list[int] = []
    right_ids: list[int] = []
    keep_rows: list[int] = []
    skipped = 0
    for row_idx, row in enumerate(df.itertuples(index=False)):
        canonical_pair = _canonical_pair(row.drug_a_smiles, row.drug_b_smiles)
        if canonical_pair is None:
            skipped += 1
            continue
        canon_a, canon_b = canonical_pair
        if canon_a not in feature_index or canon_b not in feature_index:
            skipped += 1
            continue
        left_ids.append(int(feature_index[canon_a]))
        right_ids.append(int(feature_index[canon_b]))
        keep_rows.append(int(row_idx))
    if not keep_rows:
        raise ValueError("All rows were skipped while building pair features.")
    left = feature_bank[np.asarray(left_ids, dtype=np.int64)]
    right = feature_bank[np.asarray(right_ids, dtype=np.int64)]
    filtered_df = df.iloc[keep_rows].reset_index(drop=True).copy()
    return sparse.hstack([left, right], format="csr", dtype=np.float32), filtered_df, int(skipped)


def _summary_from_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "objective_loss": float(metrics["objective_loss"]),
        "nll_loss": float(metrics["nll_loss"]),
        "macro_f1": float(metrics["macro_f1"]),
        "micro_f1": float(metrics["micro_f1"]),
        "accuracy": float(metrics["accuracy"]),
        "kappa": float(metrics["kappa"]),
        "macro_pr_auc_ovr": float(metrics["macro_pr_auc_ovr"]),
        "tail_macro_pr_auc_ovr": float(metrics["tail_macro_pr_auc_ovr"]),
        "macro_roc_auc_ovr": float(metrics["macro_roc_auc_ovr"]),
    }


def _encode_training_labels(
    y_train: np.ndarray,
    y_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, dict[int, int]]:
    train_classes = np.unique(y_train.astype(np.int64))
    if train_classes.size <= 1:
        raise ValueError("XGBoost requires at least two train classes after filtering.")
    class_to_index = {int(label): int(idx) for idx, label in enumerate(train_classes.tolist())}
    y_train_encoded = np.asarray([class_to_index[int(v)] for v in y_train.tolist()], dtype=np.int64)
    valid_mask = np.asarray([int(v) in class_to_index for v in y_valid.tolist()], dtype=bool)
    if bool(valid_mask.any()):
        y_valid_encoded = np.asarray([class_to_index[int(v)] for v in y_valid[valid_mask].tolist()], dtype=np.int64)
    else:
        y_valid_encoded = None
    return y_train_encoded, y_valid_encoded, valid_mask, class_to_index


def _expand_probabilities(
    y_prob_compact: np.ndarray,
    *,
    class_to_index: dict[int, int],
    num_classes: int,
) -> np.ndarray:
    full = np.zeros((int(y_prob_compact.shape[0]), int(num_classes)), dtype=np.float64)
    for original_label, compact_idx in class_to_index.items():
        full[:, int(original_label)] = y_prob_compact[:, int(compact_idx)]
    return full


def _collect_smiles(*frames: pd.DataFrame) -> list[str]:
    smiles_values: list[str] = []
    for frame in frames:
        smiles_values.extend(frame["drug_a_smiles"].astype(str).tolist())
        smiles_values.extend(frame["drug_b_smiles"].astype(str).tolist())
    return smiles_values


def _pair_key_from_canonical_pair(pair: tuple[str, str]) -> str:
    return f"{pair[0]}||{pair[1]}"


def _pair_key_set_from_df(df: pd.DataFrame) -> set[str]:
    pair_keys: set[str] = set()
    for row in df.itertuples(index=False):
        canonical_pair = _canonical_pair(row.drug_a_smiles, row.drug_b_smiles)
        if canonical_pair is None:
            continue
        pair_keys.add(_pair_key_from_canonical_pair(canonical_pair))
    return pair_keys


def _drug_set_from_df(df: pd.DataFrame) -> set[str]:
    drug_ids: set[str] = set()
    for row in df.itertuples(index=False):
        canonical_pair = _canonical_pair(row.drug_a_smiles, row.drug_b_smiles)
        if canonical_pair is None:
            continue
        drug_ids.add(str(canonical_pair[0]))
        drug_ids.add(str(canonical_pair[1]))
    return drug_ids


def _split_overlap_summary(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, int]:
    train_pairs = _pair_key_set_from_df(train_df)
    valid_pairs = _pair_key_set_from_df(valid_df)
    test_pairs = _pair_key_set_from_df(test_df)
    train_drugs = _drug_set_from_df(train_df)
    valid_drugs = _drug_set_from_df(valid_df)
    test_drugs = _drug_set_from_df(test_df)
    return {
        "train_unique_pairs": int(len(train_pairs)),
        "valid_unique_pairs": int(len(valid_pairs)),
        "test_unique_pairs": int(len(test_pairs)),
        "train_valid_pair_overlap": int(len(train_pairs & valid_pairs)),
        "train_test_pair_overlap": int(len(train_pairs & test_pairs)),
        "valid_test_pair_overlap": int(len(valid_pairs & test_pairs)),
        "train_unique_drugs": int(len(train_drugs)),
        "valid_unique_drugs": int(len(valid_drugs)),
        "test_unique_drugs": int(len(test_drugs)),
        "train_valid_drug_overlap": int(len(train_drugs & valid_drugs)),
        "train_test_drug_overlap": int(len(train_drugs & test_drugs)),
        "valid_test_drug_overlap": int(len(valid_drugs & test_drugs)),
    }


def _assert_zero_pair_overlap(overlap_summary: dict[str, int], *, run_name: str) -> None:
    offending = {
        name: int(overlap_summary[name])
        for name in ("train_valid_pair_overlap", "train_test_pair_overlap", "valid_test_pair_overlap")
        if int(overlap_summary[name]) > 0
    }
    if offending:
        raise AssertionError(f"{run_name}: pair overlap detected across splits: {offending}")


def _make_stratified_folds(
    y: np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(y, dtype=np.int64)
    n_splits = int(n_splits)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("Stratified folds require a non-empty 1D label array.")
    if n_splits < 2:
        raise ValueError(f"cv_k must be >= 2, got {n_splits}")
    if int(y.shape[0]) < n_splits:
        raise ValueError(f"cv_k={n_splits} exceeds available rows after filtering ({int(y.shape[0])}).")

    rng = np.random.default_rng(int(seed))
    classes, counts = np.unique(y, return_counts=True)
    class_order = classes[np.argsort(-counts, kind="stable")]
    fold_indices: list[list[int]] = [[] for _ in range(n_splits)]
    fold_sizes = np.zeros(n_splits, dtype=np.int64)
    per_class_fold_counts = {int(label): np.zeros(n_splits, dtype=np.int64) for label in classes.tolist()}

    for label in class_order.tolist():
        class_indices = np.flatnonzero(y == int(label))
        shuffled = class_indices.copy()
        rng.shuffle(shuffled)
        for idx in shuffled.tolist():
            class_counts_for_label = per_class_fold_counts[int(label)]
            candidate_folds = np.flatnonzero(class_counts_for_label == class_counts_for_label.min())
            if candidate_folds.size > 1:
                candidate_sizes = fold_sizes[candidate_folds]
                candidate_folds = candidate_folds[candidate_sizes == candidate_sizes.min()]
            chosen_fold = int(candidate_folds[int(rng.integers(candidate_folds.size))])
            fold_indices[chosen_fold].append(int(idx))
            fold_sizes[chosen_fold] += 1
            per_class_fold_counts[int(label)][chosen_fold] += 1

    all_indices = np.arange(y.shape[0], dtype=np.int64)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in range(n_splits):
        test_idx = np.asarray(sorted(fold_indices[fold_idx]), dtype=np.int64)
        if test_idx.size == 0:
            raise ValueError(f"Constructed empty test fold at index {fold_idx}.")
        train_idx = all_indices[~np.isin(all_indices, test_idx)]
        folds.append((train_idx.astype(np.int64), test_idx))
    return folds


def _make_stratified_validation_split(
    y: np.ndarray,
    *,
    valid_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.int64)
    if y.ndim != 1:
        raise ValueError("Validation split requires a 1D label array.")
    if y.size == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)

    valid_fraction = float(valid_fraction)
    if valid_fraction <= 0.0:
        return np.arange(y.shape[0], dtype=np.int64), np.asarray([], dtype=np.int64)
    if valid_fraction >= 1.0:
        raise ValueError(f"cv_valid_fraction must be in [0, 1), got {valid_fraction}")

    rng = np.random.default_rng(int(seed))
    train_idx: list[int] = []
    valid_idx: list[int] = []
    for label in np.unique(y).tolist():
        class_indices = np.flatnonzero(y == int(label))
        shuffled = class_indices.copy()
        rng.shuffle(shuffled)
        if shuffled.size <= 1:
            train_idx.extend(int(v) for v in shuffled.tolist())
            continue
        valid_count = int(round(float(shuffled.size) * valid_fraction))
        valid_count = max(1, valid_count)
        valid_count = min(valid_count, int(shuffled.size) - 1)
        valid_idx.extend(int(v) for v in shuffled[:valid_count].tolist())
        train_idx.extend(int(v) for v in shuffled[valid_count:].tolist())
    return (
        np.asarray(sorted(train_idx), dtype=np.int64),
        np.asarray(sorted(valid_idx), dtype=np.int64),
    )


def _aggregate_fold_summaries(
    fold_summaries: list[dict[str, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    if not fold_summaries:
        raise ValueError("Cannot aggregate empty fold summaries.")
    metric_names = sorted({str(key) for summary in fold_summaries for key in summary.keys()})
    mean_summary: dict[str, float] = {}
    std_summary: dict[str, float] = {}
    for metric_name in metric_names:
        values = np.asarray([float(summary[metric_name]) for summary in fold_summaries], dtype=np.float64)
        mean_summary[metric_name] = float(np.nanmean(values))
        std_summary[metric_name] = float(np.nanstd(values, ddof=0))
    return mean_summary, std_summary


def _build_xgboost_model(
    *,
    XGBClassifier,
    args: argparse.Namespace,
    num_class: int,
    seed: int,
):
    return XGBClassifier(
        objective="multi:softprob",
        num_class=int(num_class),
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        max_depth=int(args.max_depth),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        min_child_weight=float(args.min_child_weight),
        reg_lambda=float(args.reg_lambda),
        tree_method=str(args.tree_method),
        random_state=int(seed),
        n_jobs=None if int(args.n_jobs) == 0 else int(args.n_jobs),
        eval_metric="mlogloss",
        early_stopping_rounds=int(args.early_stopping_rounds) if int(args.early_stopping_rounds) > 0 else None,
        verbosity=1,
    )


def _fit_and_evaluate_split(
    *,
    XGBClassifier,
    args: argparse.Namespace,
    X_train: sparse.csr_matrix,
    y_train: np.ndarray,
    X_valid: sparse.csr_matrix,
    y_valid: np.ndarray,
    X_test: sparse.csr_matrix,
    y_test: np.ndarray,
    num_classes: int,
    run_seed: int,
    run_name: str,
) -> tuple[object, dict[str, object], dict[str, object]]:
    if X_train.shape[0] <= 0:
        raise ValueError(f"{run_name}: training split is empty after invalid-SMILES filtering.")
    if X_test.shape[0] <= 0:
        raise ValueError(f"{run_name}: test split is empty after invalid-SMILES filtering.")

    y_train = np.asarray(y_train, dtype=np.int64)
    y_valid = np.asarray(y_valid, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)

    y_train_encoded, y_valid_encoded, valid_mask, class_to_index = _encode_training_labels(y_train, y_valid)
    X_valid_fit = X_valid[valid_mask] if bool(valid_mask.any()) else None
    if X_valid.shape[0] > 0 and not bool(valid_mask.any()):
        LOGGER.warning("%s: validation split has no train-seen labels after filtering; disabling eval_set.", run_name)
    elif X_valid.shape[0] > int(np.sum(valid_mask)):
        LOGGER.warning(
            "%s: dropped %d validation rows with labels unseen in train before XGBoost fit.",
            run_name,
            int(X_valid.shape[0] - int(np.sum(valid_mask))),
        )

    model = _build_xgboost_model(
        XGBClassifier=XGBClassifier,
        args=args,
        num_class=len(class_to_index),
        seed=int(run_seed),
    )
    fit_kwargs: dict[str, object] = {"verbose": False}
    if X_valid_fit is not None and y_valid_encoded is not None and X_valid_fit.shape[0] > 0:
        fit_kwargs["eval_set"] = [(X_valid_fit, y_valid_encoded)]
    model.fit(X_train, y_train_encoded, **fit_kwargs)

    y_prob_compact = np.asarray(model.predict_proba(X_test), dtype=np.float64)
    y_prob = _expand_probabilities(
        y_prob_compact,
        class_to_index=class_to_index,
        num_classes=num_classes,
    )
    train_class_counts = compute_class_counts(y_train, num_classes=num_classes)
    metrics = evaluate_multiclass_metrics(
        y_true=y_test,
        y_prob=y_prob,
        ece_bins=int(args.ece_bins),
        train_class_counts=train_class_counts,
    )
    objective_loss = multiclass_nll_from_probs(y_true=y_test, y_prob=y_prob)
    metrics["macro_f1_present_only"] = float(metrics["macro_f1"])
    metrics["kappa"] = float(metrics["cohen_kappa"])
    metrics["objective_loss"] = float(objective_loss)
    metrics["nll_loss"] = float(objective_loss)
    metrics["loss"] = float(objective_loss)

    unseen_test_mask = ~np.isin(y_test, np.asarray(sorted(class_to_index.keys()), dtype=np.int64))
    unseen_test_labels = sorted({int(v) for v in y_test[unseen_test_mask].tolist()})
    if unseen_test_labels:
        LOGGER.warning(
            "%s: test split contains %d rows across %d labels unseen in train: %s",
            run_name,
            int(np.sum(unseen_test_mask)),
            int(len(unseen_test_labels)),
            unseen_test_labels,
        )

    tail_ids = compute_tail_class_ids(train_class_counts, fraction=0.2, include_zero_count=True)
    metrics_payload: dict[str, object] = {
        "objective_config": {
            "model_family": "xgboost",
            "pair_representation": "sorted_concat_ecfp4",
            "use_class_weights": False,
            "num_classes_total": int(num_classes),
            "num_classes_trained": int(len(class_to_index)),
            "ecfp_bits": int(args.ecfp_bits),
            "ecfp_radius": int(args.ecfp_radius),
            "n_estimators": int(args.n_estimators),
            "learning_rate": float(args.learning_rate),
            "max_depth": int(args.max_depth),
            "subsample": float(args.subsample),
            "colsample_bytree": float(args.colsample_bytree),
            "min_child_weight": float(args.min_child_weight),
            "reg_lambda": float(args.reg_lambda),
            "tree_method": str(args.tree_method),
            "early_stopping_rounds": int(args.early_stopping_rounds),
            "run_seed": int(run_seed),
        },
        "tail_definition": {
            "fraction": 0.2,
            "include_zero_count": True,
            "tail_k": int(len(tail_ids)),
            "tail_class_ids": [int(v) for v in tail_ids.tolist()],
        },
        "uncalibrated": metrics,
        "summary": _summary_from_metrics(metrics),
    }
    manifest = {
        "n_train": int(X_train.shape[0]),
        "n_valid": int(X_valid.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_valid_used_for_eval_set": 0 if X_valid_fit is None else int(X_valid_fit.shape[0]),
        "trained_class_ids": [int(k) for k in sorted(class_to_index.keys())],
        "best_iteration": None if getattr(model, "best_iteration", None) is None else int(model.best_iteration),
        "n_test_rows_with_unseen_train_labels": int(np.sum(unseen_test_mask)),
        "test_unseen_label_ids": unseen_test_labels,
    }
    return model, metrics_payload, manifest


def _train_evaluate_frames(
    *,
    XGBClassifier,
    args: argparse.Namespace,
    output_dir: Path,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_map: dict,
    run_seed: int,
    run_name: str,
    limit_seed_base: int,
    enforce_zero_pair_overlap: bool,
    feature_cache_dir: Path | None = None,
) -> tuple[object, dict[str, object], dict[str, object]]:
    if args.limit is not None:
        train_df = subsample_dataframe(
            train_df,
            limit=args.limit,
            seed=int(limit_seed_base),
            label_col="y",
            ensure_class_coverage=True,
        )
        valid_df = subsample_dataframe(
            valid_df,
            limit=args.limit,
            seed=int(limit_seed_base) + 1,
            label_col="y",
            ensure_class_coverage=True,
        )
        test_df = subsample_dataframe(
            test_df,
            limit=args.limit,
            seed=int(limit_seed_base) + 2,
            label_col="y",
            ensure_class_coverage=True,
        )

    feature_cache = DrugFeatureCache(
        output_dir=output_dir if feature_cache_dir is None else feature_cache_dir,
        use_ecfp=True,
        use_physchem=False,
        use_maccs=False,
        ecfp_bits=int(args.ecfp_bits),
        ecfp_radius=int(args.ecfp_radius),
    )
    feature_bank, feature_index = _build_feature_bank(
        _collect_smiles(train_df, valid_df, test_df),
        feature_cache=feature_cache,
    )

    X_train, train_df, skipped_train = _build_pair_matrix(
        train_df,
        feature_bank=feature_bank,
        feature_index=feature_index,
    )
    X_valid, valid_df, skipped_valid = _build_pair_matrix(
        valid_df,
        feature_bank=feature_bank,
        feature_index=feature_index,
    )
    X_test, test_df, skipped_test = _build_pair_matrix(
        test_df,
        feature_bank=feature_bank,
        feature_index=feature_index,
    )
    if skipped_train or skipped_valid or skipped_test:
        LOGGER.warning(
            "%s: skipped invalid/unfeaturizable rows | train=%d valid=%d test=%d",
            run_name,
            int(skipped_train),
            int(skipped_valid),
            int(skipped_test),
        )

    split_overlap = _split_overlap_summary(train_df, valid_df, test_df)
    if enforce_zero_pair_overlap:
        _assert_zero_pair_overlap(split_overlap, run_name=run_name)

    model, metrics_payload, fit_manifest = _fit_and_evaluate_split(
        XGBClassifier=XGBClassifier,
        args=args,
        X_train=X_train,
        y_train=train_df["y"].to_numpy(dtype=np.int64),
        X_valid=X_valid,
        y_valid=valid_df["y"].to_numpy(dtype=np.int64),
        X_test=X_test,
        y_test=test_df["y"].to_numpy(dtype=np.int64),
        num_classes=int(len(label_map)),
        run_seed=int(run_seed),
        run_name=run_name,
    )
    manifest_payload = {
        "skipped_invalid_train_rows": int(skipped_train),
        "skipped_invalid_valid_rows": int(skipped_valid),
        "skipped_invalid_test_rows": int(skipped_test),
        "n_features": int(X_train.shape[1]),
        "feature_bank_size": int(feature_bank.shape[0]),
        "feature_cache_stats": {str(k): int(v) for k, v in feature_cache.stats.items()},
        "split_overlap": split_overlap,
        **fit_manifest,
    }
    return model, metrics_payload, manifest_payload


def _write_single_split_artifacts(
    *,
    output_dir: Path,
    model,
    metrics_payload: dict[str, object],
    manifest_payload: dict[str, object],
) -> tuple[Path, Path]:
    model_path = Path(output_dir) / "model.ubj"
    metrics_path = Path(output_dir) / "evaluation_metrics.json"
    manifest_path = Path(output_dir) / "run_manifest.json"
    model.save_model(model_path)
    save_json(metrics_payload, metrics_path)
    save_json(manifest_payload, manifest_path)
    return model_path, metrics_path


def _run_single_split(
    *,
    XGBClassifier,
    args: argparse.Namespace,
    output_dir: Path,
    split_cache_dir: str | Path,
) -> tuple[Path, Path]:
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
    )
    model, metrics_payload, fit_manifest = _train_evaluate_frames(
        XGBClassifier=XGBClassifier,
        args=args,
        output_dir=output_dir,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        label_map=label_map,
        run_seed=int(args.seed),
        run_name="single_split",
        limit_seed_base=int(args.seed),
        enforce_zero_pair_overlap=bool(str(args.split_strategy) == "cold_drug"),
        feature_cache_dir=output_dir,
    )
    manifest_payload = {
        "runner": "xgboost_ecfp4",
        "run_mode": "single_split",
        "seed": int(args.seed),
        "split_strategy": str(args.split_strategy),
        "split_seed": int(args.split_seed),
        "cold_k": int(args.cold_k),
        "cold_fold": int(args.cold_fold),
        "cold_protocol": str(args.cold_protocol),
        **fit_manifest,
    }
    model_path, metrics_path = _write_single_split_artifacts(
        output_dir=output_dir,
        model=model,
        metrics_payload=metrics_payload,
        manifest_payload=manifest_payload,
    )
    LOGGER.info(
        "Saved XGBoost ECFP4 baseline | out=%s | accuracy=%.4f macro_f1=%.4f macro_pr_auc=%.4f",
        output_dir,
        float(metrics_payload["summary"]["accuracy"]),
        float(metrics_payload["summary"]["macro_f1"]),
        float(metrics_payload["summary"]["macro_pr_auc_ovr"]),
    )
    return model_path, metrics_path


def _run_cold_drug_kfold(
    *,
    XGBClassifier,
    args: argparse.Namespace,
    output_dir: Path,
    split_cache_dir: str | Path,
) -> Path:
    if str(args.cold_protocol) != "s1":
        raise ValueError(
            "run_mode=cold_drug_kfold enforces the project benchmark protocol. "
            f"Expected --cold_protocol s1, got {args.cold_protocol!r}."
        )

    fold_summaries: list[dict[str, float]] = []
    fold_manifests: list[dict[str, object]] = []
    fold_metrics_paths: list[str] = []

    for cold_fold in range(int(args.cold_k)):
        fold_dir = ensure_dir(Path(output_dir) / f"fold_{int(cold_fold)}")
        train_df, valid_df, test_df, label_map = load_tdc_drugbank_ddi(
            args.data_dir,
            output_dir=split_cache_dir,
            split_strategy="cold_drug",
            split_seed=args.split_seed,
            cold_k=args.cold_k,
            cold_fold=int(cold_fold),
            cold_protocol="s1",
            cold_min_test_pairs=args.cold_min_test_pairs,
            cold_min_test_labels=args.cold_min_test_labels,
            cold_max_resamples=args.cold_max_resamples,
            cold_dedupe_policy=args.cold_dedupe_policy,
            cold_selection_objective=args.cold_selection_objective,
        )
        model, metrics_payload, fit_manifest = _train_evaluate_frames(
            XGBClassifier=XGBClassifier,
            args=args,
            output_dir=fold_dir,
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            label_map=label_map,
            run_seed=int(args.seed),
            run_name=f"cold_drug_fold_{cold_fold}",
            limit_seed_base=int(args.seed) + (1000 * int(cold_fold)),
            enforce_zero_pair_overlap=True,
            feature_cache_dir=output_dir,
        )

        model_path = Path(fold_dir) / "model.ubj"
        metrics_path = Path(fold_dir) / "evaluation_metrics.json"
        manifest_path = Path(fold_dir) / "run_manifest.json"
        model.save_model(model_path)
        save_json(metrics_payload, metrics_path)

        fold_manifest = {
            "runner": "xgboost_ecfp4",
            "run_mode": "cold_drug_kfold",
            "benchmark_protocol": "cold_drug_s1_v3",
            "seed": int(args.seed),
            "split_strategy": "cold_drug",
            "split_impl": "v3",
            "split_seed": int(args.split_seed),
            "cold_k": int(args.cold_k),
            "cold_fold": int(cold_fold),
            "cold_protocol": "s1",
            "cold_min_test_pairs": int(args.cold_min_test_pairs),
            "cold_min_test_labels": int(args.cold_min_test_labels),
            "cold_max_resamples": int(args.cold_max_resamples),
            "cold_dedupe_policy": str(args.cold_dedupe_policy),
            "cold_selection_objective": str(args.cold_selection_objective),
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            **fit_manifest,
        }
        save_json(fold_manifest, manifest_path)

        fold_summaries.append({"fold_index": int(cold_fold), **metrics_payload["summary"]})
        fold_metrics_paths.append(str(metrics_path))
        fold_manifests.append(
            {
                "fold_index": int(cold_fold),
                "output_dir": str(fold_dir),
                "model_path": str(model_path),
                "metrics_path": str(metrics_path),
                "manifest_path": str(manifest_path),
                "train_test_pair_overlap": int(fit_manifest["split_overlap"]["train_test_pair_overlap"]),
                "train_valid_pair_overlap": int(fit_manifest["split_overlap"]["train_valid_pair_overlap"]),
                "valid_test_pair_overlap": int(fit_manifest["split_overlap"]["valid_test_pair_overlap"]),
                **{
                    key: value
                    for key, value in fit_manifest.items()
                    if key in {"n_train", "n_valid", "n_test", "best_iteration", "split_overlap"}
                },
            }
        )

    mean_summary, std_summary = _aggregate_fold_summaries(
        [{k: v for k, v in summary.items() if k != "fold_index"} for summary in fold_summaries]
    )
    metrics_path = Path(output_dir) / "evaluation_metrics.json"
    save_json(
        {
            "objective_config": {
                "model_family": "xgboost",
                "pair_representation": "sorted_concat_ecfp4",
                "use_class_weights": False,
                "num_classes_total": int(len(label_map)),
                "ecfp_bits": int(args.ecfp_bits),
                "ecfp_radius": int(args.ecfp_radius),
                "n_estimators": int(args.n_estimators),
                "learning_rate": float(args.learning_rate),
                "max_depth": int(args.max_depth),
                "subsample": float(args.subsample),
                "colsample_bytree": float(args.colsample_bytree),
                "min_child_weight": float(args.min_child_weight),
                "reg_lambda": float(args.reg_lambda),
                "tree_method": str(args.tree_method),
                "early_stopping_rounds": int(args.early_stopping_rounds),
                "run_mode": "cold_drug_kfold",
                "benchmark_protocol": "cold_drug_s1_v3",
                "split_strategy": "cold_drug",
                "split_impl": "v3",
                "cold_protocol": "s1",
                "cold_k": int(args.cold_k),
            },
            "aggregation": {
                "method": "mean_std",
                "num_folds": int(args.cold_k),
            },
            "summary": mean_summary,
            "summary_std": std_summary,
            "fold_summaries": fold_summaries,
            "fold_metrics_paths": fold_metrics_paths,
        },
        metrics_path,
    )
    save_json(
        {
            "runner": "xgboost_ecfp4",
            "run_mode": "cold_drug_kfold",
            "benchmark_protocol": "cold_drug_s1_v3",
            "seed": int(args.seed),
            "split_strategy": "cold_drug",
            "split_impl": "v3",
            "split_seed": int(args.split_seed),
            "cold_k": int(args.cold_k),
            "cold_protocol": "s1",
            "cold_min_test_pairs": int(args.cold_min_test_pairs),
            "cold_min_test_labels": int(args.cold_min_test_labels),
            "cold_max_resamples": int(args.cold_max_resamples),
            "cold_dedupe_policy": str(args.cold_dedupe_policy),
            "cold_selection_objective": str(args.cold_selection_objective),
            "all_train_test_pair_overlap_zero": bool(
                all(int(fold["train_test_pair_overlap"]) == 0 for fold in fold_manifests)
            ),
            "folds": fold_manifests,
        },
        Path(output_dir) / "run_manifest.json",
    )
    LOGGER.info(
        "Saved strict cold-drug s1 v3 XGBoost benchmark | out=%s | accuracy=%.4f macro_f1=%.4f macro_pr_auc=%.4f",
        output_dir,
        float(mean_summary["accuracy"]),
        float(mean_summary["macro_f1"]),
        float(mean_summary["macro_pr_auc_ovr"]),
    )
    return metrics_path


def _run_stratified_kfold(
    *,
    XGBClassifier,
    args: argparse.Namespace,
    output_dir: Path,
    split_cache_dir: str | Path,
) -> Path:
    train_df, valid_df, test_df, label_map = load_tdc_drugbank_ddi(
        args.data_dir,
        output_dir=split_cache_dir,
        split_strategy="tdc",
        split_seed=args.split_seed,
    )
    full_df = pd.concat([train_df, valid_df, test_df], axis=0, ignore_index=True)
    if args.limit is not None:
        full_df = subsample_dataframe(
            full_df,
            limit=args.limit,
            seed=args.seed,
            label_col="y",
            ensure_class_coverage=True,
        )

    feature_cache = DrugFeatureCache(
        output_dir=output_dir,
        use_ecfp=True,
        use_physchem=False,
        use_maccs=False,
        ecfp_bits=int(args.ecfp_bits),
        ecfp_radius=int(args.ecfp_radius),
    )
    feature_bank, feature_index = _build_feature_bank(
        _collect_smiles(full_df),
        feature_cache=feature_cache,
    )
    X_all, full_df, skipped_full = _build_pair_matrix(
        full_df,
        feature_bank=feature_bank,
        feature_index=feature_index,
    )
    if skipped_full:
        LOGGER.warning("Skipped %d invalid/unfeaturizable rows before stratified CV.", int(skipped_full))

    y_all = full_df["y"].to_numpy(dtype=np.int64)
    label_counts = compute_class_counts(y_all, num_classes=int(len(label_map)))
    rare_label_ids = np.where((label_counts > 0) & (label_counts < int(args.cv_k)))[0].astype(int).tolist()
    if rare_label_ids:
        LOGGER.warning(
            "Requested stratified %d-fold CV, but %d labels have fewer than %d rows after filtering. "
            "Using the deterministic custom stratifier so the run remains valid.",
            int(args.cv_k),
            int(len(rare_label_ids)),
            int(args.cv_k),
        )

    folds = _make_stratified_folds(
        y_all,
        n_splits=int(args.cv_k),
        seed=int(args.seed),
    )
    fold_summaries: list[dict[str, float]] = []
    fold_manifests: list[dict[str, object]] = []
    fold_metrics_paths: list[str] = []

    for fold_idx, (train_valid_idx, test_idx) in enumerate(folds):
        inner_train_rel_idx, inner_valid_rel_idx = _make_stratified_validation_split(
            y_all[train_valid_idx],
            valid_fraction=float(args.cv_valid_fraction),
            seed=int(args.seed) + int(fold_idx),
        )
        train_idx = train_valid_idx[inner_train_rel_idx]
        valid_idx = train_valid_idx[inner_valid_rel_idx]

        X_train = X_all[train_idx]
        X_valid = X_all[valid_idx]
        X_test = X_all[test_idx]
        y_train = y_all[train_idx]
        y_valid = y_all[valid_idx]
        y_test = y_all[test_idx]

        model, metrics_payload, fit_manifest = _fit_and_evaluate_split(
            XGBClassifier=XGBClassifier,
            args=args,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            X_test=X_test,
            y_test=y_test,
            num_classes=int(len(label_map)),
            run_seed=int(args.seed) + int(fold_idx),
            run_name=f"fold_{fold_idx}",
        )

        fold_dir = ensure_dir(Path(output_dir) / f"fold_{int(fold_idx)}")
        model_path = Path(fold_dir) / "model.ubj"
        metrics_path = Path(fold_dir) / "evaluation_metrics.json"
        manifest_path = Path(fold_dir) / "run_manifest.json"
        model.save_model(model_path)
        save_json(metrics_payload, metrics_path)

        fold_manifest = {
            "runner": "xgboost_ecfp4",
            "run_mode": "stratified_kfold",
            "cv_splitter": "custom_greedy_stratified",
            "cv_fold": int(fold_idx),
            "cv_k": int(args.cv_k),
            "cv_valid_fraction": float(args.cv_valid_fraction),
            "cv_source_split_strategy": "tdc",
            "seed": int(args.seed),
            "split_seed": int(args.split_seed),
            "n_features": int(X_all.shape[1]),
            "feature_bank_size": int(feature_bank.shape[0]),
            "feature_cache_stats": {str(k): int(v) for k, v in feature_cache.stats.items()},
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            **fit_manifest,
        }
        save_json(fold_manifest, manifest_path)

        fold_summaries.append({"fold_index": int(fold_idx), **metrics_payload["summary"]})
        fold_metrics_paths.append(str(metrics_path))
        fold_manifests.append(
            {
                "fold_index": int(fold_idx),
                "output_dir": str(fold_dir),
                "model_path": str(model_path),
                "metrics_path": str(metrics_path),
                "manifest_path": str(manifest_path),
                **fit_manifest,
            }
        )

    mean_summary, std_summary = _aggregate_fold_summaries(
        [{k: v for k, v in summary.items() if k != "fold_index"} for summary in fold_summaries]
    )
    metrics_path = Path(output_dir) / "evaluation_metrics.json"
    save_json(
        {
            "objective_config": {
                "model_family": "xgboost",
                "pair_representation": "sorted_concat_ecfp4",
                "use_class_weights": False,
                "num_classes_total": int(len(label_map)),
                "ecfp_bits": int(args.ecfp_bits),
                "ecfp_radius": int(args.ecfp_radius),
                "n_estimators": int(args.n_estimators),
                "learning_rate": float(args.learning_rate),
                "max_depth": int(args.max_depth),
                "subsample": float(args.subsample),
                "colsample_bytree": float(args.colsample_bytree),
                "min_child_weight": float(args.min_child_weight),
                "reg_lambda": float(args.reg_lambda),
                "tree_method": str(args.tree_method),
                "early_stopping_rounds": int(args.early_stopping_rounds),
                "run_mode": "stratified_kfold",
                "cv_splitter": "custom_greedy_stratified",
                "cv_k": int(args.cv_k),
                "cv_valid_fraction": float(args.cv_valid_fraction),
                "cv_source_split_strategy": "tdc",
            },
            "aggregation": {
                "method": "mean_std",
                "num_folds": int(args.cv_k),
            },
            "summary": mean_summary,
            "summary_std": std_summary,
            "fold_summaries": fold_summaries,
            "fold_metrics_paths": fold_metrics_paths,
        },
        metrics_path,
    )
    save_json(
        {
            "runner": "xgboost_ecfp4",
            "run_mode": "stratified_kfold",
            "cv_splitter": "custom_greedy_stratified",
            "cv_k": int(args.cv_k),
            "cv_valid_fraction": float(args.cv_valid_fraction),
            "cv_source_split_strategy": "tdc",
            "seed": int(args.seed),
            "split_seed": int(args.split_seed),
            "n_total": int(X_all.shape[0]),
            "n_features": int(X_all.shape[1]),
            "feature_bank_size": int(feature_bank.shape[0]),
            "skipped_invalid_full_rows": int(skipped_full),
            "label_min_count": int(label_counts[label_counts > 0].min()) if np.any(label_counts > 0) else 0,
            "label_max_count": int(label_counts.max()) if label_counts.size > 0 else 0,
            "labels_with_count_lt_cv_k": rare_label_ids,
            "feature_cache_stats": {str(k): int(v) for k, v in feature_cache.stats.items()},
            "folds": fold_manifests,
        },
        Path(output_dir) / "run_manifest.json",
    )
    LOGGER.info(
        "Saved stratified %d-fold XGBoost ECFP4 baseline | out=%s | accuracy=%.4f macro_f1=%.4f macro_pr_auc=%.4f",
        int(args.cv_k),
        output_dir,
        float(mean_summary["accuracy"]),
        float(mean_summary["macro_f1"]),
        float(mean_summary["macro_pr_auc_ovr"]),
    )
    return metrics_path


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))
    XGBClassifier = _require_xgboost()

    output_dir = ensure_dir(args.output_dir)
    split_cache_dir = args.split_cache_dir or args.output_dir

    if str(args.run_mode) == "cold_drug_kfold":
        metrics_path = _run_cold_drug_kfold(
            XGBClassifier=XGBClassifier,
            args=args,
            output_dir=output_dir,
            split_cache_dir=split_cache_dir,
        )
        print(f"metrics_path={metrics_path}")
        return

    if str(args.run_mode) == "stratified_kfold":
        metrics_path = _run_stratified_kfold(
            XGBClassifier=XGBClassifier,
            args=args,
            output_dir=output_dir,
            split_cache_dir=split_cache_dir,
        )
        print(f"metrics_path={metrics_path}")
        return

    model_path, metrics_path = _run_single_split(
        XGBClassifier=XGBClassifier,
        args=args,
        output_dir=output_dir,
        split_cache_dir=split_cache_dir,
    )
    print(f"model_path={model_path}")
    print(f"metrics_path={metrics_path}")


if __name__ == "__main__":
    main()
