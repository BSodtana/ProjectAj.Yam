#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ddigat.data.cache import GraphCache
from ddigat.data.splits import DDIPairDataset, make_pair_dataloader, subsample_dataframe
from ddigat.data.tdc_ddi import load_tdc_drugbank_ddi
from ddigat.model.pair_model import DDIPairModel
from ddigat.train.loop import eval_epoch
from ddigat.utils.class_weights import assert_class_weight_sanity, compute_class_counts, compute_class_weights
from ddigat.utils.calibration import apply_temperature, fit_temperature
from ddigat.utils.io import torch_load
from ddigat.utils.logging import get_logger
from ddigat.utils.metrics import evaluate_multiclass_metrics
from ddigat.utils.seed import seed_everything


LOGGER = get_logger("scripts.evaluate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained DDI GNN model.")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--checkpoint", type=str, default="./outputs/checkpoints/best.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1)  # interface consistency
    p.add_argument("--lr", type=float, default=1e-3)  # interface consistency
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--ece_bins", type=int, default=15)
    p.add_argument("--calibrate_temperature", action="store_true")
    p.add_argument("--split_strategy", type=str, default="cold_drug", choices=["cold_drug", "tdc"])
    p.add_argument("--split_seed", type=int, default=42)
    return p.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            LOGGER.warning(
                "MPS detected, but PyG ops used by this project are not fully supported on MPS. "
                "Falling back to CPU for stability."
            )
            return torch.device("cpu")
        return torch.device("cpu")
    if device_arg == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        LOGGER.warning(
            "Using MPS with PYTORCH_ENABLE_MPS_FALLBACK=1. "
            "Unsupported ops will run on CPU and execution may be slower."
        )
        return torch.device("mps")
    return torch.device(device_arg)


def build_model_from_checkpoint_payload(payload: dict, device: torch.device) -> DDIPairModel:
    cfg = payload.get("config", {})
    model_cfg = (cfg or {}).get("model", {})
    model = DDIPairModel(
        in_dim=int(model_cfg.get("in_dim", 7)),
        edge_dim=int(model_cfg.get("edge_dim", 5)),
        encoder_type=str(model_cfg.get("encoder_type", "gat")),
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        out_dim=int(model_cfg.get("out_dim", 128)),
        num_layers=int(model_cfg.get("num_layers", 3)),
        heads=int(model_cfg.get("heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        mlp_hidden_dim=int(model_cfg.get("mlp_hidden_dim", 256)),
        num_classes=int(model_cfg.get("num_classes", 86)),
        pooling=str(model_cfg.get("pooling", "mean")),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def _to_float_dict(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        try:
            out[k] = float(v)
        except Exception:
            out[k] = v
    return out


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)

    payload = torch_load(args.checkpoint, map_location=device)
    model = build_model_from_checkpoint_payload(payload, device)

    train_df, valid_df, test_df, _ = load_tdc_drugbank_ddi(
        args.data_dir,
        output_dir=args.output_dir,
        split_strategy=args.split_strategy,
        split_seed=args.split_seed,
    )
    if args.limit is not None:
        test_df = subsample_dataframe(test_df, limit=args.limit, seed=args.seed, label_col="y", ensure_class_coverage=True)

    cfg = payload.get("config", {}) or {}
    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    use_class_weights = bool(train_cfg.get("use_class_weights", False))
    class_weight_method = str(train_cfg.get("class_weight_method", "inv_sqrt"))
    class_weight_beta = float(train_cfg.get("class_weight_beta", 0.9999))
    class_weight_clip_min = float(train_cfg.get("class_weight_clip_min", 0.25))
    class_weight_clip_max = float(train_cfg.get("class_weight_clip_max", 20.0))
    class_weight_eps = float(train_cfg.get("class_weight_eps", 1e-12))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    if use_class_weights:
        class_counts = compute_class_counts(train_df["y"].to_numpy(dtype=int), num_classes=model.num_classes)
        class_weight_info = compute_class_weights(
            class_counts,
            method=class_weight_method,
            beta=class_weight_beta,
            eps=class_weight_eps,
            clip_min=class_weight_clip_min,
            clip_max=class_weight_clip_max,
        )
        assert_class_weight_sanity(
            class_weight_info.weights,
            num_classes=model.num_classes,
            clip_min=class_weight_clip_min,
            clip_max=class_weight_clip_max,
            mean_after_normalization=class_weight_info.mean_after_normalization,
        )
        class_weights = class_weight_info.weights
    else:
        class_weights = None
    model.set_loss_params(class_weights=class_weights, label_smoothing=label_smoothing)
    LOGGER.info(
        "Evaluation objective restored | use_class_weights=%s method=%s label_smoothing=%.4f",
        use_class_weights,
        class_weight_method,
        label_smoothing,
    )

    cache = GraphCache(output_dir=args.output_dir)
    test_ds = DDIPairDataset(test_df, cache, split_name="test")
    test_loader = make_pair_dataloader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    test_eval = eval_epoch(
        model,
        test_loader,
        device=device,
        amp_enabled=True,
        collect_logits=args.calibrate_temperature,
    )
    uncalibrated = evaluate_multiclass_metrics(
        y_true=test_eval["y_true"],
        y_prob=test_eval["y_prob"],
        ece_bins=args.ece_bins,
    )
    uncalibrated["macro_f1_present_only"] = float(uncalibrated["macro_f1"])
    uncalibrated["kappa"] = float(uncalibrated["cohen_kappa"])
    uncalibrated["objective_loss"] = float(test_eval["objective_loss"])
    uncalibrated["nll_loss"] = float(test_eval["nll_loss"])
    uncalibrated["loss"] = float(test_eval["objective_loss"])
    LOGGER.info("Uncalibrated metrics: %s", _to_float_dict(uncalibrated))

    out_payload: dict[str, object] = {
        "objective_config": {
            "use_class_weights": use_class_weights,
            "class_weight_method": class_weight_method,
            "class_weight_beta": class_weight_beta,
            "class_weight_clip_min": class_weight_clip_min,
            "class_weight_clip_max": class_weight_clip_max,
            "class_weight_eps": class_weight_eps,
            "label_smoothing": label_smoothing,
        },
        "uncalibrated": _to_float_dict(uncalibrated),
        "summary": {
            "objective_loss": float(uncalibrated["objective_loss"]),
            "nll_loss": float(uncalibrated["nll_loss"]),
            "macro_f1": float(uncalibrated["macro_f1"]),
            "micro_f1": float(uncalibrated["micro_f1"]),
            "accuracy": float(uncalibrated["accuracy"]),
            "kappa": float(uncalibrated["kappa"]),
            "macro_pr_auc_ovr": float(uncalibrated["macro_pr_auc_ovr"]),
            "macro_roc_auc_ovr": float(uncalibrated["macro_roc_auc_ovr"]),
        },
    }

    if args.calibrate_temperature:
        valid_ds = DDIPairDataset(valid_df, cache, split_name="valid")
        valid_loader = make_pair_dataloader(
            valid_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        valid_eval = eval_epoch(model, valid_loader, device=device, amp_enabled=True, collect_logits=True)
        temperature = fit_temperature(
            logits=valid_eval["y_logits"],
            y_true=valid_eval["y_true"],
            device=device,
            max_iter=100,
        )
        calibrated_logits = apply_temperature(test_eval["y_logits"], temperature)
        calibrated_prob = torch.softmax(torch.tensor(calibrated_logits, dtype=torch.float32), dim=-1).numpy()
        calibrated = evaluate_multiclass_metrics(
            y_true=test_eval["y_true"],
            y_prob=calibrated_prob,
            ece_bins=args.ece_bins,
        )
        calibrated["macro_f1_present_only"] = float(calibrated["macro_f1"])
        calibrated["kappa"] = float(calibrated["cohen_kappa"])
        cal_logits_t = torch.tensor(calibrated_logits, dtype=torch.float32, device=device)
        y_t = torch.tensor(test_eval["y_true"], dtype=torch.long, device=device)
        calibrated["nll_loss"] = float(F.cross_entropy(cal_logits_t, y_t).item())
        calibrated["objective_loss"] = float(model.loss_fn(cal_logits_t, y_t).item())
        calibrated["loss"] = float(calibrated["objective_loss"])
        out_payload["temperature"] = float(temperature)
        out_payload["calibrated"] = _to_float_dict(calibrated)
        LOGGER.info("Temperature scaled metrics: %s", _to_float_dict(calibrated))

    out_path = Path(args.output_dir) / "evaluation_metrics.json"
    out_path.write_text(json.dumps(out_payload, indent=2))

    # Headline metrics aligned with field norms.
    print(f"accuracy={uncalibrated['accuracy']:.6f}")
    print(f"macro_f1={uncalibrated['macro_f1']:.6f}")
    print(f"micro_f1={uncalibrated['micro_f1']:.6f}")
    print(f"cohen_kappa={uncalibrated['cohen_kappa']:.6f}")
    print(f"macro_roc_auc_ovr={uncalibrated['macro_roc_auc_ovr']:.6f}")
    print(f"macro_pr_auc_ovr={uncalibrated['macro_pr_auc_ovr']:.6f}")
    print(f"ece={uncalibrated['ece']:.6f}")
    print(f"brier_score={uncalibrated['brier_score']:.6f}")
    print(f"objective_loss={uncalibrated['objective_loss']:.6f}")
    print(f"nll_loss={uncalibrated['nll_loss']:.6f}")
    if args.calibrate_temperature:
        print(f"temperature={out_payload['temperature']:.6f}")
        c = out_payload["calibrated"]
        print(f"calibrated_accuracy={c['accuracy']:.6f}")
        print(f"calibrated_macro_f1={c['macro_f1']:.6f}")
        print(f"calibrated_micro_f1={c['micro_f1']:.6f}")
        print(f"calibrated_cohen_kappa={c['cohen_kappa']:.6f}")
        print(f"calibrated_macro_roc_auc_ovr={c['macro_roc_auc_ovr']:.6f}")
        print(f"calibrated_macro_pr_auc_ovr={c['macro_pr_auc_ovr']:.6f}")
        print(f"calibrated_ece={c['ece']:.6f}")
        print(f"calibrated_brier_score={c['brier_score']:.6f}")
        print(f"calibrated_objective_loss={c['objective_loss']:.6f}")
        print(f"calibrated_nll_loss={c['nll_loss']:.6f}")
    print(f"metrics_saved={out_path}")


if __name__ == "__main__":
    main()
