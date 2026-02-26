from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ddigat.utils.logging import get_logger


LOGGER = get_logger(__name__)


@dataclass
class MacroMetricResult:
    value: float
    included_classes: List[int]
    excluded_classes: List[int]


def _safe_macro_ovr_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    scorer,
    metric_name: str,
) -> MacroMetricResult:
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

    n_classes = y_prob.shape[1]
    y_true = y_true.astype(int)

    included: list[int] = []
    excluded: list[int] = []
    scores: list[float] = []
    for c in range(n_classes):
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


def multiclass_macro_roc_auc_ovr(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return _safe_macro_ovr_metric(y_true, y_prob, roc_auc_score, "macro_roc_auc_ovr").value


def multiclass_macro_pr_auc_ovr(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return _safe_macro_ovr_metric(y_true, y_prob, average_precision_score, "macro_pr_auc_ovr").value

