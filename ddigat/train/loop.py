from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ddigat.train.callbacks import EarlyStopping, save_checkpoint
from ddigat.utils.logging import get_logger
from ddigat.utils.metrics import evaluate_multiclass_metrics


LOGGER = get_logger(__name__)


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    batch["graph_a"] = batch["graph_a"].to(device)
    batch["graph_b"] = batch["graph_b"].to(device)
    if "feat_a" in batch:
        batch["feat_a"] = batch["feat_a"].to(device)
    if "feat_b" in batch:
        batch["feat_b"] = batch["feat_b"].to(device)
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

    total_objective_loss = 0.0
    total_plain_nll_loss = 0.0
    total_samples = 0
    total_batches = 0
    weight_diverse_batches = 0
    observed_target_weight_min = float("inf")
    observed_target_weight_max = float("-inf")

    has_class_weights = bool(getattr(model, "class_weights", None) is not None)
    has_multiple_global_weights = False
    if has_class_weights:
        cw = getattr(model, "class_weights")
        has_multiple_global_weights = bool(int(torch.unique(cw.detach().cpu()).numel()) > 1)

    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        if batch is None:
            continue
        batch = _move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(batch["graph_a"], batch["graph_b"], batch.get("feat_a"), batch.get("feat_b"))
            objective_loss = model.loss_fn(logits, batch["y"])
            plain_nll_loss = F.cross_entropy(logits, batch["y"], weight=None, label_smoothing=0.0, reduction="mean")

        if scaler is not None and use_amp:
            scaler.scale(objective_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            objective_loss.backward()
            optimizer.step()

        bs = int(batch["y"].size(0))
        total_objective_loss += float(objective_loss.detach().item()) * bs
        total_plain_nll_loss += float(plain_nll_loss.detach().item()) * bs
        total_samples += bs
        total_batches += 1
        if has_class_weights:
            batch_w = getattr(model, "class_weights").to(batch["y"].device)[batch["y"]]
            batch_w_min = float(torch.min(batch_w).item())
            batch_w_max = float(torch.max(batch_w).item())
            observed_target_weight_min = min(observed_target_weight_min, batch_w_min)
            observed_target_weight_max = max(observed_target_weight_max, batch_w_max)
            if float(batch_w_max - batch_w_min) > 1e-12:
                weight_diverse_batches += 1
        pbar.set_postfix(loss=f"{objective_loss.item():.4f}")

    avg_objective_loss = total_objective_loss / max(total_samples, 1)
    avg_plain_nll_loss = total_plain_nll_loss / max(total_samples, 1)
    weight_diverse_batches_pct = (100.0 * float(weight_diverse_batches) / float(max(total_batches, 1))) if total_batches > 0 else 0.0
    if not (0.0 <= weight_diverse_batches_pct <= 100.0):
        raise AssertionError(f"weight_diverse_batches_pct out of range: {weight_diverse_batches_pct}")
    if has_class_weights and has_multiple_global_weights and total_batches > 0 and weight_diverse_batches <= 0:
        if observed_target_weight_max > observed_target_weight_min:
            LOGGER.warning(
                "No intra-batch weight diversity observed; weighted objective still active "
                "(epoch target-weight range %.6f..%.6f across batches).",
                observed_target_weight_min,
                observed_target_weight_max,
            )
        else:
            LOGGER.warning(
                "No weight-diverse batches observed; all sampled targets shared one weight (%.6f). "
                "This can happen when sampled labels map to the clipped min weight.",
                observed_target_weight_min,
            )
    return {
        "loss": avg_objective_loss,
        "objective_loss": avg_objective_loss,
        "plain_nll_loss": avg_plain_nll_loss,
        "weight_diverse_batches": float(weight_diverse_batches),
        "weight_diverse_batches_pct": float(weight_diverse_batches_pct),
        "n_samples": total_samples,
        "n_batches": total_batches,
    }


@torch.no_grad()
def eval_epoch(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    amp_enabled: bool = True,
    collect_logits: bool = False,
) -> dict[str, Any]:
    model.eval()
    use_amp = bool(amp_enabled and device.type == "cuda")

    total_objective_loss = 0.0
    total_nll_loss = 0.0
    total_samples = 0
    y_true_all: list[np.ndarray] = []
    y_prob_all: list[np.ndarray] = []
    y_logits_all: list[np.ndarray] = []
    pair_ids_all: list[str] = []

    pbar = tqdm(loader, desc="eval", leave=False)
    for batch in pbar:
        if batch is None:
            continue
        batch = _move_batch_to_device(batch, device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(batch["graph_a"], batch["graph_b"], batch.get("feat_a"), batch.get("feat_b"))
            objective_loss = model.loss_fn(logits, batch["y"])
            nll_loss = F.cross_entropy(logits, batch["y"], reduction="mean")
            probs = torch.softmax(logits, dim=-1)
        row_sum_err = float(torch.max(torch.abs(probs.sum(dim=-1) - 1.0)).item())
        if row_sum_err > 1e-5:
            raise AssertionError(f"Softmax rows must sum to ~1.0, max_abs_err={row_sum_err:.6e}")

        bs = int(batch["y"].size(0))
        total_objective_loss += float(objective_loss.detach().item()) * bs
        total_nll_loss += float(nll_loss.detach().item()) * bs
        total_samples += bs
        y_true_all.append(batch["y"].detach().cpu().numpy())
        y_prob_all.append(probs.detach().cpu().numpy())
        if collect_logits:
            y_logits_all.append(logits.detach().cpu().numpy())
        pair_ids_all.extend(batch["meta"]["pair_ids"])

    if total_samples == 0:
        raise RuntimeError("eval_epoch observed zero valid samples (all pairs skipped?)")

    y_true = np.concatenate(y_true_all, axis=0)
    y_prob = np.concatenate(y_prob_all, axis=0)
    metrics: dict[str, Any] = {
        "loss": total_objective_loss / total_samples,
        "objective_loss": total_objective_loss / total_samples,
        "nll_loss": total_nll_loss / total_samples,
        "n_samples": total_samples,
        "y_true": y_true,
        "y_prob": y_prob,
        "pair_ids": pair_ids_all,
    }
    if collect_logits:
        metrics["y_logits"] = np.concatenate(y_logits_all, axis=0)
    metrics.update(evaluate_multiclass_metrics(y_true=y_true, y_prob=y_prob))
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
    loss_config: dict[str, Any] | None = None,
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
            "train_objective_loss": train_metrics["objective_loss"],
            "train_plain_nll_loss": train_metrics["plain_nll_loss"],
            "train_weight_diverse_batches_pct": train_metrics["weight_diverse_batches_pct"],
            "valid_loss": valid_metrics["loss"],
            "valid_macro_roc_auc_ovr": valid_metrics.get("macro_roc_auc_ovr"),
            "valid_macro_pr_auc_ovr": valid_metrics.get("macro_pr_auc_ovr"),
        }
        history.append(row)
        LOGGER.info(
            "Epoch %d/%d | train_loss=%.4f train_objective_loss=%.4f train_plain_nll=%.4f "
            "weight_diverse_batches_pct=%.2f valid_loss=%.4f valid_acc=%.4f valid_macro_f1=%.4f "
            "valid_micro_f1=%.4f valid_kappa=%.4f valid_macro_pr_auc=%.4f valid_macro_roc_auc=%.4f",
            epoch,
            epochs,
            row["train_loss"],
            row["train_objective_loss"],
            row["train_plain_nll_loss"],
            row["train_weight_diverse_batches_pct"],
            row["valid_loss"],
            valid_metrics.get("accuracy", float("nan")),
            valid_metrics.get("macro_f1", float("nan")),
            valid_metrics.get("micro_f1", float("nan")),
            valid_metrics.get("cohen_kappa", float("nan")),
            row["valid_macro_pr_auc_ovr"],
            row["valid_macro_roc_auc_ovr"],
        )

        current_score = float(valid_metrics.get("macro_pr_auc_ovr", float("-inf")))
        if best_metrics is None or current_score > float(best_metrics.get("macro_pr_auc_ovr", float("-inf"))):
            best_metrics = {
                k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in valid_metrics.items()
                if k not in {"y_true", "y_prob", "y_logits", "pair_ids"}
            }
            best_epoch = epoch
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=best_metrics,
                config=config,
                loss_config=loss_config,
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
