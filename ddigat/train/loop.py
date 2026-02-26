from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from ddigat.train.callbacks import EarlyStopping, save_checkpoint
from ddigat.utils.logging import get_logger
from ddigat.utils.metrics import multiclass_macro_pr_auc_ovr, multiclass_macro_roc_auc_ovr


LOGGER = get_logger(__name__)


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    batch["graph_a"] = batch["graph_a"].to(device)
    batch["graph_b"] = batch["graph_b"].to(device)
    batch["y"] = batch["y"].to(device)
    return batch


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    amp_enabled: bool = True,
) -> dict[str, float]:
    model.train()
    use_amp = bool(amp_enabled and device.type == "cuda")

    total_loss = 0.0
    total_samples = 0
    total_batches = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        if batch is None:
            continue
        batch = _move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(batch["graph_a"], batch["graph_b"])
            loss = model.loss_fn(logits, batch["y"])

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = int(batch["y"].size(0))
        total_loss += float(loss.detach().item()) * bs
        total_samples += bs
        total_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(total_samples, 1)
    return {"loss": avg_loss, "n_samples": total_samples, "n_batches": total_batches}


@torch.no_grad()
def eval_epoch(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    amp_enabled: bool = True,
) -> dict[str, Any]:
    model.eval()
    use_amp = bool(amp_enabled and device.type == "cuda")

    total_loss = 0.0
    total_samples = 0
    y_true_all: list[np.ndarray] = []
    y_prob_all: list[np.ndarray] = []
    pair_ids_all: list[str] = []

    pbar = tqdm(loader, desc="eval", leave=False)
    for batch in pbar:
        if batch is None:
            continue
        batch = _move_batch_to_device(batch, device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(batch["graph_a"], batch["graph_b"])
            loss = model.loss_fn(logits, batch["y"])
            probs = torch.softmax(logits, dim=-1)

        bs = int(batch["y"].size(0))
        total_loss += float(loss.detach().item()) * bs
        total_samples += bs
        y_true_all.append(batch["y"].detach().cpu().numpy())
        y_prob_all.append(probs.detach().cpu().numpy())
        pair_ids_all.extend(batch["meta"]["pair_ids"])

    if total_samples == 0:
        raise RuntimeError("eval_epoch observed zero valid samples (all pairs skipped?)")

    y_true = np.concatenate(y_true_all, axis=0)
    y_prob = np.concatenate(y_prob_all, axis=0)
    metrics: dict[str, Any] = {
        "loss": total_loss / total_samples,
        "n_samples": total_samples,
        "y_true": y_true,
        "y_prob": y_prob,
        "pair_ids": pair_ids_all,
    }
    try:
        metrics["macro_roc_auc_ovr"] = multiclass_macro_roc_auc_ovr(y_true, y_prob)
    except Exception as e:
        LOGGER.warning("macro_roc_auc_ovr failed: %s", e)
        metrics["macro_roc_auc_ovr"] = float("nan")
    try:
        metrics["macro_pr_auc_ovr"] = multiclass_macro_pr_auc_ovr(y_true, y_prob)
    except Exception as e:
        LOGGER.warning("macro_pr_auc_ovr failed: %s", e)
        metrics["macro_pr_auc_ovr"] = float("nan")
    return metrics


def fit(
    model: torch.nn.Module,
    train_loader,
    valid_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    output_dir: str | Path,
    config: dict[str, Any],
    label_map: dict[str, Any] | dict[int, Any],
    patience: int = 5,
    min_delta: float = 0.0,
    amp_enabled: bool = True,
) -> dict[str, Any]:
    checkpoint_path = Path(output_dir) / "checkpoints" / "best.pt"
    early_stopper = EarlyStopping(patience=patience, min_delta=min_delta, mode="max")
    scaler = torch.cuda.amp.GradScaler(enabled=bool(amp_enabled and device.type == "cuda"))

    history: list[dict[str, Any]] = []
    best_metrics: dict[str, Any] | None = None
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )
        valid_metrics = eval_epoch(
            model=model,
            loader=valid_loader,
            device=device,
            amp_enabled=amp_enabled,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "valid_loss": valid_metrics["loss"],
            "valid_macro_roc_auc_ovr": valid_metrics.get("macro_roc_auc_ovr"),
            "valid_macro_pr_auc_ovr": valid_metrics.get("macro_pr_auc_ovr"),
        }
        history.append(row)
        LOGGER.info(
            "Epoch %d/%d | train_loss=%.4f valid_loss=%.4f valid_macro_pr_auc=%.4f valid_macro_roc_auc=%.4f",
            epoch,
            epochs,
            row["train_loss"],
            row["valid_loss"],
            row["valid_macro_pr_auc_ovr"],
            row["valid_macro_roc_auc_ovr"],
        )

        current_score = float(valid_metrics.get("macro_pr_auc_ovr", float("-inf")))
        if best_metrics is None or current_score > float(best_metrics.get("macro_pr_auc_ovr", float("-inf"))):
            best_metrics = {
                k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in valid_metrics.items()
                if k not in {"y_true", "y_prob", "pair_ids"}
            }
            best_epoch = epoch
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=best_metrics,
                config=config,
                label_map={str(k): str(v) for k, v in label_map.items()},
            )
            LOGGER.info("Saved best checkpoint to %s", checkpoint_path)

        if early_stopper.step(current_score):
            LOGGER.info("Early stopping triggered at epoch %d (best epoch=%d)", epoch, best_epoch)
            break

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics or {},
        "checkpoint_path": str(checkpoint_path),
    }

