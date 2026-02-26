from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from ddigat.utils.io import ensure_dir, torch_save


@dataclass
class EarlyStopping:
    patience: int = 5
    min_delta: float = 0.0
    mode: str = "max"

    def __post_init__(self) -> None:
        if self.mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        self.best: Optional[float] = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            self.bad_epochs = 0
            return False
        improved = (
            value > (self.best + self.min_delta)
            if self.mode == "max"
            else value < (self.best - self.min_delta)
        )
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[dict[str, Any]] = None,
    config: Optional[dict[str, Any]] = None,
    label_map: Optional[dict[str, Any]] = None,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "metrics": metrics or {},
        "config": config or {},
        "label_map": label_map or {},
    }
    torch_save(payload, path)

