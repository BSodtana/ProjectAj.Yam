from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(
    logits: np.ndarray,
    y_true: np.ndarray,
    device: str | torch.device = "cpu",
    max_iter: int = 100,
) -> float:
    """Fit scalar temperature on validation logits using NLL minimization."""
    if logits.ndim != 2:
        raise ValueError(f"logits must be [N, C], got {logits.shape}")
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be [N], got {y_true.shape}")
    if logits.shape[0] != y_true.shape[0]:
        raise ValueError("logits/y_true size mismatch")

    dev = torch.device(device)
    logits_t = torch.tensor(logits, dtype=torch.float32, device=dev)
    y_t = torch.tensor(y_true, dtype=torch.long, device=dev)

    log_temp = torch.nn.Parameter(torch.zeros(1, device=dev))
    optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temp = torch.exp(log_temp) + 1e-6
        loss = F.cross_entropy(logits_t / temp, y_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    temp = float(torch.exp(log_temp).detach().cpu().item())
    return max(temp, 1e-6)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError(f"temperature must be >0, got {temperature}")
    return logits / float(temperature)

