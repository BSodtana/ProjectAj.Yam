from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ddigat.utils.logging import get_logger


LOGGER = get_logger(__name__)


@dataclass
class ClassWeightComputation:
    counts: np.ndarray
    weights: torch.Tensor
    normalize: str
    n_seen: int
    n_unseen: int
    mean_after_normalization: float
    mean_after_clipping: float
    mean_seen_after_clipping: float
    min_weight: float
    max_weight: float
    sat_min_seen: float
    sat_max_seen: float
    unique_seen_weights: int
    cv_seen: float


def compute_class_counts(y: np.ndarray, num_classes: int) -> np.ndarray:
    labels = y.astype(int).reshape(-1)
    if labels.size == 0:
        raise ValueError("Cannot compute class counts from an empty label array")
    if int(np.min(labels)) < 0 or int(np.max(labels)) >= int(num_classes):
        raise ValueError(
            f"Labels out of range for num_classes={num_classes}: min={int(np.min(labels))}, max={int(np.max(labels))}"
        )
    return np.bincount(labels, minlength=int(num_classes)).astype(np.int64)


def compute_class_priors(
    counts: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    counts_f = np.asarray(counts, dtype=np.float64).reshape(-1)
    if counts_f.size == 0:
        raise ValueError("counts must be non-empty")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be > 0, got {eps}")
    denom = float(np.sum(counts_f) + float(eps) * float(counts_f.size))
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError(f"Invalid denominator for class prior computation: {denom}")
    priors = (counts_f + float(eps)) / denom
    log_priors = np.log(priors)
    if not np.isfinite(log_priors).all():
        raise ValueError("log_priors contains non-finite values")
    return priors.astype(np.float64), log_priors.astype(np.float64)


def compute_tail_class_ids(
    counts: np.ndarray,
    fraction: float = 0.2,
    include_zero_count: bool = True,
) -> np.ndarray:
    counts_i = np.asarray(counts, dtype=np.int64).reshape(-1)
    if counts_i.size == 0:
        raise ValueError("counts must be non-empty")
    if not (0.0 < float(fraction) <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    class_ids = np.arange(counts_i.size, dtype=np.int64)
    if bool(include_zero_count):
        eligible = class_ids
    else:
        eligible = class_ids[counts_i > 0]
    if eligible.size == 0:
        return np.asarray([], dtype=np.int64)

    order = np.argsort(counts_i[eligible], kind="stable")
    sorted_ids = eligible[order]
    k = int(np.ceil(float(fraction) * float(counts_i.size)))
    k = max(1, min(k, int(sorted_ids.size)))
    return sorted_ids[:k].astype(np.int64)


def compute_class_weights(
    counts: np.ndarray,
    method: str = "inv_sqrt",
    beta: float = 0.9999,
    eps: float = 1e-12,
    clip_min: float = 0.25,
    clip_max: float = 4.0,
    normalize: str = "sample_mean",
) -> ClassWeightComputation:
    counts_f = np.asarray(counts, dtype=np.float64).reshape(-1)
    if counts_f.size == 0:
        raise ValueError("counts must be non-empty")
    seen = counts_f > 0.0
    n_seen = int(np.sum(seen))
    n_unseen = int(counts_f.size - n_seen)
    if n_unseen > 0:
        LOGGER.warning(
            "Class-weight computation: %d/%d classes unseen in train split; assigning unseen weight=1.0 and excluding from normalization/clipping.",
            n_unseen,
            int(counts_f.size),
        )

    raw = np.ones_like(counts_f, dtype=np.float64)
    method_norm = method.lower().strip()
    if method_norm == "inv_sqrt":
        raw[seen] = 1.0 / np.sqrt(np.maximum(counts_f[seen], float(eps)))
    elif method_norm in {"effective_num", "effective_num_cui"}:
        if not (0.0 < float(beta) < 1.0):
            raise ValueError(f"effective_num requires 0 < beta < 1, got beta={beta}")
        denom = 1.0 - np.power(float(beta), counts_f[seen])
        raw[seen] = (1.0 - float(beta)) / np.maximum(denom, float(eps))
    else:
        raise ValueError(f"Unsupported class weight method: {method}")

    normalized = raw.copy()
    norm_mode = str(normalize).lower().strip()
    if norm_mode in {"none", ""}:
        mean_check = 1.0
    elif n_seen <= 0:
        mean_check = 1.0
    elif norm_mode == "mean_seen":
        denom = float(np.mean(raw[seen]))
        if not np.isfinite(denom) or denom <= 0.0:
            raise ValueError(f"Invalid seen-mean normalization denominator: {denom}")
        normalized[seen] = raw[seen] / denom
        mean_check = float(np.mean(normalized[seen]))
    elif norm_mode == "sample_mean":
        sample_total = float(np.sum(counts_f[seen]))
        if not np.isfinite(sample_total) or sample_total <= 0.0:
            raise ValueError(f"Invalid sample_total for class-weight normalization: {sample_total}")
        denom = float(np.sum(raw[seen] * counts_f[seen]) / sample_total)
        if not np.isfinite(denom) or denom <= 0.0:
            raise ValueError(f"Invalid sample-mean normalization denominator: {denom}")
        normalized[seen] = raw[seen] / denom
        mean_check = float(np.sum(normalized[seen] * counts_f[seen]) / sample_total)
    else:
        raise ValueError(f"Unsupported class weight normalization mode: {normalize}")

    clipped = normalized.copy()
    if n_seen > 0:
        clipped[seen] = np.clip(clipped[seen], float(clip_min), float(clip_max))
    weights_t = torch.tensor(clipped, dtype=torch.float32)
    if n_seen > 0:
        ws = clipped[seen]
        sat_min_seen = float(np.mean(np.isclose(ws, float(clip_min), rtol=0.0, atol=1e-12)))
        sat_max_seen = float(np.mean(np.isclose(ws, float(clip_max), rtol=0.0, atol=1e-12)))
        unique_seen_weights = int(len(np.unique(np.round(ws, 12))))
        cv_seen = float(np.std(ws) / (np.mean(ws) + float(eps)))
        mean_seen_after_clipping = float(np.mean(ws))
    else:
        sat_min_seen = 0.0
        sat_max_seen = 0.0
        unique_seen_weights = 0
        cv_seen = 0.0
        mean_seen_after_clipping = 1.0
    return ClassWeightComputation(
        counts=counts.astype(np.int64),
        weights=weights_t,
        normalize=norm_mode,
        n_seen=n_seen,
        n_unseen=n_unseen,
        mean_after_normalization=mean_check,
        mean_after_clipping=float(np.mean(clipped)),
        mean_seen_after_clipping=mean_seen_after_clipping,
        min_weight=float(np.min(clipped)),
        max_weight=float(np.max(clipped)),
        sat_min_seen=sat_min_seen,
        sat_max_seen=sat_max_seen,
        unique_seen_weights=unique_seen_weights,
        cv_seen=cv_seen,
    )


def assert_class_weight_sanity(
    weights: torch.Tensor,
    num_classes: int,
    clip_min: float,
    clip_max: float,
    mean_after_normalization: float,
    counts: np.ndarray | None = None,
    normalize: str = "sample_mean",
    sat_min_max: float = 0.50,
    sat_max_max: float | None = None,
    min_unique_seen_weights: int = 4,
    min_cv_seen: float = 1e-3,
    atol: float = 1e-6,
) -> None:
    if tuple(weights.shape) != (int(num_classes),):
        raise AssertionError(f"class_weights shape must be ({num_classes},), got {tuple(weights.shape)}")
    if not torch.isfinite(weights).all().item():
        raise AssertionError("class_weights contains non-finite values")
    if not bool((weights > 0).all().item()):
        raise AssertionError("class_weights must be strictly > 0")
    norm_mode = str(normalize).lower().strip()
    if norm_mode not in {"none", ""} and abs(float(mean_after_normalization) - 1.0) > float(atol):
        raise AssertionError(
            f"class_weights mean after normalization must be ~1.0, got {mean_after_normalization:.8f}"
        )
    if float(weights.max().item()) > float(clip_max) + 1e-8:
        raise AssertionError(
            f"class_weights max exceeds clip_max={clip_max}: got {float(weights.max().item())}"
        )
    if float(weights.min().item()) < float(clip_min) - 1e-8:
        raise AssertionError(
            f"class_weights min below clip_min={clip_min}: got {float(weights.min().item())}"
        )
    if counts is None:
        return
    counts_np = np.asarray(counts, dtype=np.float64).reshape(-1)
    if counts_np.size != int(num_classes):
        raise AssertionError(
            f"counts shape mismatch for class weight sanity: expected {num_classes}, got {counts_np.size}"
        )
    seen = counts_np > 0.0
    if not np.any(seen):
        return
    w = weights.detach().cpu().numpy().astype(np.float64, copy=False)
    ws = w[seen]
    if not np.all(np.isfinite(ws)):
        raise AssertionError("class_weights contains non-finite values among seen classes")
    if not np.all(ws > 0.0):
        raise AssertionError("class_weights contains non-positive values among seen classes")
    if not np.all(ws >= float(clip_min) - 1e-9) or not np.all(ws <= float(clip_max) + 1e-9):
        raise AssertionError("Seen-class weights out of clip range")
    sat_min = float(np.mean(np.isclose(ws, float(clip_min), rtol=0.0, atol=1e-12)))
    sat_max = float(np.mean(np.isclose(ws, float(clip_max), rtol=0.0, atol=1e-12)))
    uniq = int(len(np.unique(np.round(ws, 12))))
    cv = float(np.std(ws) / (np.mean(ws) + float(np.finfo(np.float64).eps)))
    if sat_min >= float(sat_min_max):
        raise AssertionError(f"Degenerate class weights: {sat_min:.2%} of seen classes clipped at clip_min")
    if sat_max_max is not None and sat_max >= float(sat_max_max):
        raise AssertionError(f"Degenerate class weights: {sat_max:.2%} of seen classes clipped at clip_max")
    if uniq < int(min_unique_seen_weights):
        raise AssertionError(f"Degenerate class weights: too few distinct seen weights (uniq={uniq})")
    if cv <= float(min_cv_seen):
        raise AssertionError(f"Degenerate class weights: near-constant seen weights (cv={cv:.3e})")


def class_counts_payload(counts: np.ndarray, num_classes: int) -> dict[str, Any]:
    counts_i = counts.astype(np.int64)
    zero_classes = np.where(counts_i == 0)[0].astype(int).tolist()
    return {
        "num_classes": int(num_classes),
        "total_samples": int(np.sum(counts_i)),
        "counts": [int(v) for v in counts_i.tolist()],
        "zero_count_classes": zero_classes,
    }


def class_weights_payload(
    *,
    enabled: bool,
    method: str,
    beta: float,
    eps: float,
    clip_min: float,
    clip_max: float,
    computation: ClassWeightComputation | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "enabled": bool(enabled),
        "method": str(method),
        "beta": float(beta),
        "eps": float(eps),
        "clip_min": float(clip_min),
        "clip_max": float(clip_max),
    }
    if computation is None:
        payload.update(
            {
                "weights": None,
                "normalize": None,
                "mean_after_normalization": None,
                "mean_after_clipping": None,
                "mean_seen_after_clipping": None,
                "min_weight": None,
                "max_weight": None,
                "n_seen": None,
                "n_unseen": None,
                "sat_min_seen": None,
                "sat_max_seen": None,
                "unique_seen_weights": None,
                "cv_seen": None,
            }
        )
        return payload
    payload.update(
        {
            "weights": [float(v) for v in computation.weights.cpu().numpy().tolist()],
            "normalize": str(computation.normalize),
            "mean_after_normalization": float(computation.mean_after_normalization),
            "mean_after_clipping": float(computation.mean_after_clipping),
            "mean_seen_after_clipping": float(computation.mean_seen_after_clipping),
            "min_weight": float(computation.min_weight),
            "max_weight": float(computation.max_weight),
            "n_seen": int(computation.n_seen),
            "n_unseen": int(computation.n_unseen),
            "sat_min_seen": float(computation.sat_min_seen),
            "sat_max_seen": float(computation.sat_max_seen),
            "unique_seen_weights": int(computation.unique_seen_weights),
            "cv_seen": float(computation.cv_seen),
        }
    )
    return payload
