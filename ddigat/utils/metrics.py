from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)

from ddigat.utils.logging import get_logger
from ddigat.utils.class_weights import compute_tail_class_ids


LOGGER = get_logger(__name__)


@dataclass
class MacroMetricResult:
    value: float
    included_classes: List[int]
    excluded_classes: List[int]

    def as_dict(self, prefix: str) -> dict[str, Any]:
        return {
            prefix: self.value,
            f"{prefix}_included_classes": list(self.included_classes),
            f"{prefix}_excluded_classes": list(self.excluded_classes),
        }


def _validate_classification_arrays(y_true: np.ndarray, y_prob: np.ndarray, metric_name: str) -> None:
    if y_true.ndim != 1:
        raise ValueError(f"{metric_name}: y_true must be shape [N], got {y_true.shape}")
    if y_prob.ndim != 2:
        raise ValueError(f"{metric_name}: y_prob must be shape [N, C], got {y_prob.shape}")
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError(
            f"{metric_name}: y_true/y_prob sample mismatch {y_true.shape[0]} != {y_prob.shape[0]}"
        )
    if y_true.size == 0:
        raise ValueError(f"{metric_name}: empty inputs")


def _safe_macro_ovr_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    scorer,
    metric_name: str,
    class_ids: list[int] | np.ndarray | None = None,
) -> MacroMetricResult:
    _validate_classification_arrays(y_true=y_true, y_prob=y_prob, metric_name=metric_name)

    n_classes = y_prob.shape[1]
    y_true = y_true.astype(int)

    included: list[int] = []
    excluded: list[int] = []
    scores: list[float] = []
    if class_ids is None:
        class_iter = list(range(n_classes))
    else:
        class_iter = [int(c) for c in class_ids]
    for c in class_iter:
        y_bin = (y_true == c).astype(int)
        pos = int(y_bin.sum())
        neg = int((1 - y_bin).sum())
        if pos == 0 or neg == 0:
            excluded.append(c)
            continue
        try:
            s = scorer(y_bin, y_prob[:, c])
        except ValueError as e:
            LOGGER.warning("%s failed for class %d: %s", metric_name, c, e)
            excluded.append(c)
            continue
        if np.isnan(s):
            excluded.append(c)
            continue
        included.append(c)
        scores.append(float(s))

    if excluded:
        LOGGER.info(
            "%s excluded %d classes with missing positives/negatives: %s",
            metric_name,
            len(excluded),
            excluded,
        )
    if not scores:
        raise ValueError(f"{metric_name}: no valid classes available for macro averaging")
    return MacroMetricResult(
        value=float(np.mean(scores)),
        included_classes=included,
        excluded_classes=excluded,
    )


def multiclass_macro_roc_auc_ovr_result(y_true: np.ndarray, y_prob: np.ndarray) -> MacroMetricResult:
    return _safe_macro_ovr_metric(y_true, y_prob, roc_auc_score, "macro_roc_auc_ovr")


def multiclass_macro_roc_auc_ovr(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return multiclass_macro_roc_auc_ovr_result(y_true, y_prob).value


def multiclass_macro_pr_auc_ovr_result(y_true: np.ndarray, y_prob: np.ndarray) -> MacroMetricResult:
    return _safe_macro_ovr_metric(y_true, y_prob, average_precision_score, "macro_pr_auc_ovr")


def multiclass_macro_pr_auc_ovr(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return multiclass_macro_pr_auc_ovr_result(y_true, y_prob).value


def multiclass_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def multiclass_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Field protocol: macro-F1 over classes present in y_true.
    labels = np.unique(y_true.astype(int))
    return float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))


def multiclass_micro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Keep micro-F1 on full prediction behavior; do not restrict labels to y_true.
    return float(f1_score(y_true, y_pred, average="micro", zero_division=0))


def multiclass_cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Default behavior accounts for the full confusion matrix (union of y_true/y_pred labels).
    return float(cohen_kappa_score(y_true, y_pred))


def multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(int)
    n = y_true.shape[0]
    c = y_prob.shape[1]
    y_one_hot = np.zeros((n, c), dtype=np.float32)
    y_one_hot[np.arange(n), y_true] = 1.0
    return float(np.mean(np.sum((y_prob - y_one_hot) ** 2, axis=1)))


def multiclass_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error using max-prob confidence bins."""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if not np.any(mask):
            continue
        bin_acc = float(np.mean(accuracies[mask]))
        bin_conf = float(np.mean(confidences[mask]))
        ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)
    return float(ece)


def multiclass_nll_from_probs(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    _validate_classification_arrays(y_true=y_true, y_prob=y_prob, metric_name="multiclass_nll_from_probs")
    y = y_true.astype(int)
    clipped = np.clip(y_prob, eps, 1.0)
    target_probs = clipped[np.arange(y.shape[0]), y]
    return float(-np.mean(np.log(target_probs)))


def multiclass_nll_from_logits(y_true: np.ndarray, y_logits: np.ndarray) -> float:
    if y_logits.ndim != 2:
        raise ValueError(f"multiclass_nll_from_logits: y_logits must be [N, C], got {y_logits.shape}")
    if y_true.ndim != 1:
        raise ValueError(f"multiclass_nll_from_logits: y_true must be [N], got {y_true.shape}")
    if y_true.shape[0] != y_logits.shape[0]:
        raise ValueError("multiclass_nll_from_logits: sample size mismatch")
    y = y_true.astype(int)
    logits = y_logits.astype(np.float64)
    max_logit = np.max(logits, axis=1, keepdims=True)
    stabilized = logits - max_logit
    logsumexp = max_logit[:, 0] + np.log(np.sum(np.exp(stabilized), axis=1))
    target_logits = logits[np.arange(y.shape[0]), y]
    return float(np.mean(logsumexp - target_logits))


def evaluate_multiclass_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ece_bins: int = 15,
    include_ovr_details: bool = False,
    train_class_counts: np.ndarray | None = None,
    tail_fraction: float = 0.2,
    include_zero_count_tail: bool = True,
) -> Dict[str, Any]:
    _validate_classification_arrays(y_true=y_true, y_prob=y_prob, metric_name="evaluate_multiclass_metrics")
    y_true = y_true.astype(int)
    y_pred = np.argmax(y_prob, axis=1).astype(int)

    n_classes = int(y_prob.shape[1])
    pos_counts = np.bincount(y_true, minlength=n_classes).astype(np.int64)
    neg_counts = int(y_true.shape[0]) - pos_counts
    missing_pos_classes = np.where(pos_counts == 0)[0].astype(int).tolist()
    missing_neg_classes = np.where(neg_counts == 0)[0].astype(int).tolist()

    out: dict[str, Any] = {
        "accuracy": multiclass_accuracy(y_true, y_pred),
        "macro_f1": multiclass_macro_f1(y_true, y_pred),
        "micro_f1": multiclass_micro_f1(y_true, y_pred),
        "cohen_kappa": multiclass_cohen_kappa(y_true, y_pred),
        "brier_score": multiclass_brier_score(y_true, y_prob),
        "ece": multiclass_ece(y_true, y_prob, n_bins=ece_bins),
        "nll": multiclass_nll_from_probs(y_true, y_prob),
        "n_classes_total": n_classes,
        "n_classes_missing_pos": int(len(missing_pos_classes)),
        "n_classes_missing_neg": int(len(missing_neg_classes)),
    }
    try:
        roc_res = multiclass_macro_roc_auc_ovr_result(y_true, y_prob)
        out["macro_roc_auc_ovr"] = roc_res.value
        if include_ovr_details:
            out["macro_roc_auc_ovr_included_classes"] = roc_res.included_classes
            out["macro_roc_auc_ovr_excluded_classes"] = roc_res.excluded_classes
    except Exception as e:
        LOGGER.warning("macro_roc_auc_ovr failed: %s", e)
        out["macro_roc_auc_ovr"] = float("nan")
        if include_ovr_details:
            out["macro_roc_auc_ovr_included_classes"] = []
            out["macro_roc_auc_ovr_excluded_classes"] = []
    try:
        pr_res = multiclass_macro_pr_auc_ovr_result(y_true, y_prob)
        out["macro_pr_auc_ovr"] = pr_res.value
        out["n_classes_scored"] = int(len(pr_res.included_classes))
        if include_ovr_details:
            out["macro_pr_auc_ovr_included_classes"] = pr_res.included_classes
            out["macro_pr_auc_ovr_excluded_classes"] = pr_res.excluded_classes
    except Exception as e:
        LOGGER.warning("macro_pr_auc_ovr failed: %s", e)
        out["macro_pr_auc_ovr"] = float("nan")
        out["n_classes_scored"] = 0
        if include_ovr_details:
            out["macro_pr_auc_ovr_included_classes"] = []
            out["macro_pr_auc_ovr_excluded_classes"] = []

    if train_class_counts is not None:
        counts = np.asarray(train_class_counts, dtype=np.int64).reshape(-1)
        if int(counts.size) != n_classes:
            raise ValueError(f"train_class_counts size mismatch: expected {n_classes}, got {counts.size}")
        tail_ids_np = compute_tail_class_ids(
            counts=counts,
            fraction=float(tail_fraction),
            include_zero_count=bool(include_zero_count_tail),
        )
        tail_ids = [int(v) for v in tail_ids_np.tolist()]
        out["tail_k"] = int(len(tail_ids))
        out["tail_definition"] = (
            f"bottom_{int(round(float(tail_fraction) * 100))}pct_train_support_"
            f"{'including_zero' if include_zero_count_tail else 'positive_only'}"
        )
        out["tail_class_ids"] = tail_ids
        try:
            tail_res = _safe_macro_ovr_metric(
                y_true=y_true,
                y_prob=y_prob,
                scorer=average_precision_score,
                metric_name="tail_macro_pr_auc_ovr",
                class_ids=tail_ids,
            )
            out["tail_macro_pr_auc_ovr"] = float(tail_res.value)
            out["tail_n_classes_scored"] = int(len(tail_res.included_classes))
            if include_ovr_details:
                out["tail_macro_pr_auc_ovr_included_classes"] = tail_res.included_classes
                out["tail_macro_pr_auc_ovr_excluded_classes"] = tail_res.excluded_classes
        except Exception as e:
            LOGGER.warning("tail_macro_pr_auc_ovr failed: %s", e)
            out["tail_macro_pr_auc_ovr"] = float("nan")
            out["tail_n_classes_scored"] = 0
            if include_ovr_details:
                out["tail_macro_pr_auc_ovr_included_classes"] = []
                out["tail_macro_pr_auc_ovr_excluded_classes"] = []
    return out
