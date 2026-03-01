from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class ClassWeightComputation:
    counts: np.ndarray
    weights: torch.Tensor
    mean_after_normalization: float
    mean_after_clipping: float
    min_weight: float
    max_weight: float


def compute_class_counts(y: np.ndarray, num_classes: int) -> np.ndarray:
    labels = y.astype(int).reshape(-1)
    if labels.size == 0:
        raise ValueError("Cannot compute class counts from an empty label array")
    if int(np.min(labels)) < 0 or int(np.max(labels)) >= int(num_classes):
        raise ValueError(
            f"Labels out of range for num_classes={num_classes}: min={int(np.min(labels))}, max={int(np.max(labels))}"
        )
    return np.bincount(labels, minlength=int(num_classes)).astype(np.int64)


def compute_class_weights(
    counts: np.ndarray,
    method: str = "inv_sqrt",
    beta: float = 0.9999,
    eps: float = 1e-12,
    clip_min: float = 0.25,
    clip_max: float = 20.0,
) -> ClassWeightComputation:
    counts_f = counts.astype(np.float64)
    method_norm = method.lower().strip()
    if method_norm == "inv_sqrt":
        raw = 1.0 / np.sqrt(counts_f + float(eps))
    elif method_norm == "effective_num":
        if not (0.0 < float(beta) < 1.0):
            raise ValueError(f"effective_num requires 0 < beta < 1, got beta={beta}")
        denom = 1.0 - np.power(float(beta), counts_f)
        raw = (1.0 - float(beta)) / np.maximum(denom, float(eps))
    else:
        raise ValueError(f"Unsupported class weight method: {method}")

    mean_after_norm = float(np.mean(raw))
    if not np.isfinite(mean_after_norm) or mean_after_norm <= 0.0:
        raise ValueError(f"Invalid class-weight normalization mean: {mean_after_norm}")
    normalized = raw / mean_after_norm
    mean_check = float(np.mean(normalized))

    clipped = np.clip(normalized, float(clip_min), float(clip_max))
    weights_t = torch.tensor(clipped, dtype=torch.float32)
    return ClassWeightComputation(
        counts=counts.astype(np.int64),
        weights=weights_t,
        mean_after_normalization=mean_check,
        mean_after_clipping=float(np.mean(clipped)),
        min_weight=float(np.min(clipped)),
        max_weight=float(np.max(clipped)),
    )


def assert_class_weight_sanity(
    weights: torch.Tensor,
    num_classes: int,
    clip_min: float,
    clip_max: float,
    mean_after_normalization: float,
    atol: float = 1e-6,
) -> None:
    if tuple(weights.shape) != (int(num_classes),):
        raise AssertionError(f"class_weights shape must be ({num_classes},), got {tuple(weights.shape)}")
    if not torch.isfinite(weights).all().item():
        raise AssertionError("class_weights contains non-finite values")
    if not bool((weights > 0).all().item()):
        raise AssertionError("class_weights must be strictly > 0")
    if abs(float(mean_after_normalization) - 1.0) > float(atol):
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
                "mean_after_normalization": None,
                "mean_after_clipping": None,
                "min_weight": None,
                "max_weight": None,
            }
        )
        return payload
    payload.update(
        {
            "weights": [float(v) for v in computation.weights.cpu().numpy().tolist()],
            "mean_after_normalization": float(computation.mean_after_normalization),
            "mean_after_clipping": float(computation.mean_after_clipping),
            "min_weight": float(computation.min_weight),
            "max_weight": float(computation.max_weight),
        }
    )
    return payload
