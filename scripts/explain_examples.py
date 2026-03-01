#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ddigat.data.cache import GraphCache
from ddigat.data.splits import DDIPairDataset, subsample_dataframe
from ddigat.data.tdc_ddi import load_tdc_drugbank_ddi
from ddigat.explain.attention import get_node_importance_from_attention
from ddigat.explain.faithfulness import deletion_test, insertion_test
from ddigat.explain.gnnexplainer import build_explainer_for_graph_contrib, run_gnnexplainer_on_graph
from ddigat.model.pair_model import DDIPairModel
from ddigat.utils.io import ensure_dir, torch_load, torch_save
from ddigat.utils.logging import get_logger
from ddigat.utils.seed import seed_everything
from ddigat.viz.molecule import draw_molecule_importance
from ddigat.viz.plots import plot_faithfulness_curves, plot_node_scores


LOGGER = get_logger("scripts.explain_examples")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate explanation artifacts and summary metrics.")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--checkpoint", type=str, default="./outputs/checkpoints/best.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=1)  # interface consistency
    p.add_argument("--epochs", type=int, default=1)  # interface consistency
    p.add_argument("--lr", type=float, default=1e-3)  # interface consistency
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--gnnexplainer_epochs", type=int, default=30)
    p.add_argument("--run_gnnexplainer", action="store_true")
    p.add_argument("--random_baseline_repeats", type=int, default=5)
    p.add_argument("--topk_stability", type=int, default=10)
    p.add_argument("--stability_noise_std", type=float, default=0.01)
    p.add_argument("--stability_repeats", type=int, default=20)
    p.add_argument("--only_correct", action="store_true", default=False)
    p.add_argument("--split_strategy", type=str, default="cold_drug", choices=["cold_drug", "tdc"])
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--bootstrap_repeats", type=int, default=2000)
    p.add_argument("--bootstrap_ci", type=float, default=95.0)
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
    cfg = (payload.get("config") or {}).get("model", {})
    model = DDIPairModel(
        in_dim=int(cfg.get("in_dim", 7)),
        edge_dim=int(cfg.get("edge_dim", 5)),
        encoder_type=str(cfg.get("encoder_type", "gat")),
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        out_dim=int(cfg.get("out_dim", 128)),
        num_layers=int(cfg.get("num_layers", 3)),
        heads=int(cfg.get("heads", 4)),
        dropout=float(cfg.get("dropout", 0.2)),
        mlp_hidden_dim=int(cfg.get("mlp_hidden_dim", 256)),
        num_classes=int(cfg.get("num_classes", 86)),
        pooling=str(cfg.get("pooling", "mean")),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def _k_schedule(num_nodes: int) -> list[int]:
    base = [0, 1, 2, 5, 10, 15, 20, 30]
    ks = sorted({min(max(0, k), num_nodes) for k in base} | {num_nodes})
    return [k for k in ks if k <= num_nodes]


def _curve_auc(k: list[int], y: list[float]) -> float:
    if len(k) < 2:
        return float(y[0]) if y else float("nan")
    k_arr = np.asarray(k, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    area = np.trapz(y_arr, k_arr)
    norm = max(k_arr.max() - k_arr.min(), 1.0)
    return float(area / norm)


def _topk_jaccard(rank_a: list[int], rank_b: list[int], k: int) -> float:
    sa = set(rank_a[:k])
    sb = set(rank_b[:k])
    union = sa.union(sb)
    if not union:
        return 1.0
    return float(len(sa.intersection(sb)) / len(union))


def _attention_stability(
    scores: np.ndarray,
    k: int,
    repeats: int,
    noise_std: float,
    rng: np.random.Generator,
) -> float:
    if len(scores) == 0:
        return float("nan")
    base_rank = list(np.argsort(-scores))
    js = []
    for _ in range(repeats):
        noisy = scores + rng.normal(0.0, noise_std, size=scores.shape)
        noisy_rank = list(np.argsort(-noisy))
        js.append(_topk_jaccard(base_rank, noisy_rank, k=k))
    return float(np.mean(js))


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size == 0:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _degree_score_correlation(graph, node_scores: np.ndarray) -> float:
    num_nodes = int(graph.x.size(0))
    if num_nodes == 0:
        return float("nan")
    edge_index = graph.edge_index.detach().cpu().long()
    deg = np.zeros(num_nodes, dtype=float)
    if edge_index.numel() > 0:
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()
        np.add.at(deg, src, 1.0)
        np.add.at(deg, dst, 1.0)
    return _safe_pearson(deg, np.asarray(node_scores, dtype=float))


def _naive_attention_scores(num_nodes: int, edge_index: torch.Tensor, alpha: torch.Tensor) -> np.ndarray:
    edge_index = edge_index.detach().cpu().long()
    if alpha.dim() == 2:
        edge_scores = alpha.detach().float().mean(dim=-1).cpu()
    elif alpha.dim() == 1:
        edge_scores = alpha.detach().float().cpu()
    else:
        raise ValueError(f"Unsupported alpha shape {tuple(alpha.shape)}")
    scores = torch.zeros((num_nodes,), dtype=torch.float32)
    scores.scatter_add_(0, edge_index[1], edge_scores)
    if scores.numel():
        min_v = float(scores.min())
        max_v = float(scores.max())
        if max_v > min_v:
            scores = (scores - min_v) / (max_v - min_v)
        elif max_v > 0:
            scores = scores / max_v
    return scores.numpy()


def _bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    repeats: int,
    ci: float,
) -> dict[str, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    if vals.size == 1:
        v = float(vals[0])
        return {"mean": v, "ci_low": v, "ci_high": v}
    means = []
    n = vals.size
    for _ in range(repeats):
        idx = rng.integers(0, n, size=n)
        means.append(float(np.mean(vals[idx])))
    means_arr = np.asarray(means, dtype=float)
    alpha = (100.0 - ci) / 2.0
    low = float(np.percentile(means_arr, alpha))
    high = float(np.percentile(means_arr, 100.0 - alpha))
    return {"mean": float(np.mean(vals)), "ci_low": low, "ci_high": high}


@torch.no_grad()
def _predict_single(model: DDIPairModel, graph_a, graph_b, device: torch.device):
    graph_a = graph_a.to(device)
    graph_b = graph_b.to(device)
    logits, attn = model.forward_with_attention(graph_a, graph_b)
    probs = torch.softmax(logits, dim=-1).detach().cpu()[0]
    pred = int(torch.argmax(probs).item())
    return probs.numpy(), pred, attn


def _random_baseline_curves(
    model: DDIPairModel,
    graph_a,
    graph_b,
    target_class: int,
    k_list: list[int],
    which: str,
    repeats: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    n = int(graph_a.x.size(0) if which == "A" else graph_b.x.size(0))
    del_curves = []
    ins_curves = []
    for _ in range(repeats):
        ranking = list(rng.permutation(n).tolist())
        d = deletion_test(model, graph_a, graph_b, target_class, ranking, k_list, which=which, device=device)
        ins = insertion_test(model, graph_a, graph_b, target_class, ranking, k_list, which=which, device=device)
        del_curves.append(np.asarray(d, dtype=float))
        ins_curves.append(np.asarray(ins, dtype=float))
    del_mean = np.mean(np.stack(del_curves, axis=0), axis=0).tolist()
    ins_mean = np.mean(np.stack(ins_curves, axis=0), axis=0).tolist()
    return del_mean, ins_mean


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    bootstrap_rng = np.random.default_rng(args.seed + 12345)
    device = resolve_device(args.device)

    payload = torch_load(args.checkpoint, map_location=device)
    model = build_model_from_checkpoint_payload(payload, device=device)
    _, _, test_df, _ = load_tdc_drugbank_ddi(
        args.data_dir,
        output_dir=args.output_dir,
        split_strategy=args.split_strategy,
        split_seed=args.split_seed,
    )
    if args.limit is not None:
        test_df = subsample_dataframe(test_df, limit=args.limit, seed=args.seed, label_col="y", ensure_class_coverage=True)

    cache = GraphCache(output_dir=args.output_dir)
    ds = DDIPairDataset(test_df, cache, split_name="test")
    out_root = ensure_dir(Path(args.output_dir) / "explanations")

    per_pair_rows: list[dict] = []
    generated = 0
    idx = 0
    while generated < args.n and idx < len(ds):
        sample = ds[idx]
        idx += 1
        if sample is None:
            continue

        pair_dir = ensure_dir(out_root / sample.pair_id)
        probs, pred_class, attn = _predict_single(model, sample.graph_a, sample.graph_b, device)
        if args.only_correct and pred_class != int(sample.y):
            continue
        # Keep target fixed to the original model prediction for faithfulness consistency.
        target_class = pred_class

        # Attention importances for GAT; fallback to uniform for non-attention encoders.
        attn_a = attn.get("A") if isinstance(attn, dict) else None
        attn_b = attn.get("B") if isinstance(attn, dict) else None
        if attn_a is not None and attn_a.get("alpha") is not None:
            node_scores_a_t = get_node_importance_from_attention(
                sample.graph_a,
                attn_a["edge_index"],
                attn_a["alpha"],
                node_embeddings=attn_a.get("node_input_embeddings", attn_a.get("node_embeddings")),
                src_message_embeddings=attn_a.get("src_message_embeddings"),
                batch=attn_a.get("batch"),
                remove_self_loops=True,
                normalize_per_graph=True,
                aggregate_heads="mean",
            )
            node_scores_a = node_scores_a_t.numpy()
            naive_scores_a = _naive_attention_scores(
                int(sample.graph_a.x.size(0)),
                edge_index=attn_a["edge_index"],
                alpha=attn_a["alpha"],
            )
        else:
            node_scores_a = np.ones(int(sample.graph_a.x.size(0)), dtype=float)
            naive_scores_a = node_scores_a.copy()

        if attn_b is not None and attn_b.get("alpha") is not None:
            node_scores_b_t = get_node_importance_from_attention(
                sample.graph_b,
                attn_b["edge_index"],
                attn_b["alpha"],
                node_embeddings=attn_b.get("node_input_embeddings", attn_b.get("node_embeddings")),
                src_message_embeddings=attn_b.get("src_message_embeddings"),
                batch=attn_b.get("batch"),
                remove_self_loops=True,
                normalize_per_graph=True,
                aggregate_heads="mean",
            )
            node_scores_b = node_scores_b_t.numpy()
            naive_scores_b = _naive_attention_scores(
                int(sample.graph_b.x.size(0)),
                edge_index=attn_b["edge_index"],
                alpha=attn_b["alpha"],
            )
        else:
            node_scores_b = np.ones(int(sample.graph_b.x.size(0)), dtype=float)
            naive_scores_b = node_scores_b.copy()

        degree_corr_old_a = _degree_score_correlation(sample.graph_a, naive_scores_a)
        degree_corr_new_a = _degree_score_correlation(sample.graph_a, node_scores_a)
        degree_corr_old_b = _degree_score_correlation(sample.graph_b, naive_scores_b)
        degree_corr_new_b = _degree_score_correlation(sample.graph_b, node_scores_b)

        draw_molecule_importance(
            sample.smiles_a,
            node_scores_a,
            pair_dir / "attention_A.png",
            top_k=min(10, len(node_scores_a)),
            legend=f"A attn | pred={pred_class}",
        )
        draw_molecule_importance(
            sample.smiles_b,
            node_scores_b,
            pair_dir / "attention_B.png",
            top_k=min(10, len(node_scores_b)),
            legend=f"B attn | pred={pred_class}",
        )
        plot_node_scores(node_scores_a, pair_dir / "attention_A_scores.png", title="Attention node scores A")
        plot_node_scores(node_scores_b, pair_dir / "attention_B_scores.png", title="Attention node scores B")

        # Faithfulness and random baseline.
        rank_a = list(np.argsort(-node_scores_a))
        rank_b = list(np.argsort(-node_scores_b))
        k_a = _k_schedule(len(rank_a))
        k_b = _k_schedule(len(rank_b))
        del_a = deletion_test(model, sample.graph_a, sample.graph_b, target_class, rank_a, k_a, which="A", device=device)
        ins_a = insertion_test(model, sample.graph_a, sample.graph_b, target_class, rank_a, k_a, which="A", device=device)
        del_b = deletion_test(model, sample.graph_a, sample.graph_b, target_class, rank_b, k_b, which="B", device=device)
        ins_b = insertion_test(model, sample.graph_a, sample.graph_b, target_class, rank_b, k_b, which="B", device=device)
        rnd_del_a, rnd_ins_a = _random_baseline_curves(
            model, sample.graph_a, sample.graph_b, target_class, k_a, "A", args.random_baseline_repeats, rng, device
        )
        rnd_del_b, rnd_ins_b = _random_baseline_curves(
            model, sample.graph_a, sample.graph_b, target_class, k_b, "B", args.random_baseline_repeats, rng, device
        )
        plot_faithfulness_curves(k_a, del_a, ins_a, pair_dir / "faithfulness_A.png", title=f"Faithfulness A | target={target_class}")
        plot_faithfulness_curves(k_b, del_b, ins_b, pair_dir / "faithfulness_B.png", title=f"Faithfulness B | target={target_class}")
        plot_faithfulness_curves(
            k_a,
            rnd_del_a,
            rnd_ins_a,
            pair_dir / "faithfulness_A_random_baseline.png",
            title=f"Random baseline A | target={target_class}",
        )
        plot_faithfulness_curves(
            k_b,
            rnd_del_b,
            rnd_ins_b,
            pair_dir / "faithfulness_B_random_baseline.png",
            title=f"Random baseline B | target={target_class}",
        )

        gnn_jaccard_a = float("nan")
        gnn_jaccard_b = float("nan")
        gnn_status_a = "not_run"
        gnn_status_b = "not_run"
        if args.run_gnnexplainer:
            for which in ["A", "B"]:
                if which == "A":
                    graph = sample.graph_a.clone().to(device)
                    fixed = sample.graph_b.clone()
                    smiles = sample.smiles_a
                    att_rank = rank_a
                else:
                    graph = sample.graph_b.clone().to(device)
                    fixed = sample.graph_a.clone()
                    smiles = sample.smiles_b
                    att_rank = rank_b
                bundle = build_explainer_for_graph_contrib(
                    model,
                    fixed_other_graph=fixed,
                    target_class=target_class,
                    which=which,
                    epochs=args.gnnexplainer_epochs,
                )
                explanation, node_scores, edge_mask, status = run_gnnexplainer_on_graph(bundle, graph)
                if which == "A":
                    gnn_status_a = status
                else:
                    gnn_status_b = status
                if node_scores is not None:
                    node_scores_np = node_scores.numpy()
                    draw_molecule_importance(
                        smiles,
                        node_scores_np,
                        pair_dir / f"gnnexplainer_{which}.png",
                        top_k=min(10, int(node_scores.numel())),
                        legend=f"GNNExplainer {which}",
                    )
                    plot_node_scores(
                        node_scores_np,
                        pair_dir / f"gnnexplainer_{which}_scores.png",
                        title=f"GNNExplainer node scores {which}",
                    )
                    gnn_rank = list(np.argsort(-node_scores_np))
                    j = _topk_jaccard(att_rank, gnn_rank, k=min(args.topk_stability, len(att_rank)))
                    if which == "A":
                        gnn_jaccard_a = j
                    else:
                        gnn_jaccard_b = j
                if edge_mask is not None:
                    torch_save(
                        {
                            "status": status,
                            "edge_mask": edge_mask.detach().cpu(),
                            "node_scores": None if node_scores is None else node_scores.detach().cpu(),
                        },
                        pair_dir / f"gnnexplainer_{which}_masks.pt",
                    )

        stability_a = _attention_stability(
            node_scores_a,
            k=min(args.topk_stability, len(rank_a)),
            repeats=args.stability_repeats,
            noise_std=args.stability_noise_std,
            rng=rng,
        )
        stability_b = _attention_stability(
            node_scores_b,
            k=min(args.topk_stability, len(rank_b)),
            repeats=args.stability_repeats,
            noise_std=args.stability_noise_std,
            rng=rng,
        )

        auc_del_a = _curve_auc(k_a, del_a)
        auc_ins_a = _curve_auc(k_a, ins_a)
        auc_del_b = _curve_auc(k_b, del_b)
        auc_ins_b = _curve_auc(k_b, ins_b)
        auc_del_a_random = _curve_auc(k_a, rnd_del_a)
        auc_ins_a_random = _curve_auc(k_a, rnd_ins_a)
        auc_del_b_random = _curve_auc(k_b, rnd_del_b)
        auc_ins_b_random = _curve_auc(k_b, rnd_ins_b)

        row = {
            "pair_id": sample.pair_id,
            "y_true": int(sample.y),
            "target_class": int(target_class),
            "pred_prob": float(probs[target_class]),
            "auc_del_a": auc_del_a,
            "auc_ins_a": auc_ins_a,
            "auc_del_b": auc_del_b,
            "auc_ins_b": auc_ins_b,
            "auc_del_a_random": auc_del_a_random,
            "auc_ins_a_random": auc_ins_a_random,
            "auc_del_b_random": auc_del_b_random,
            "auc_ins_b_random": auc_ins_b_random,
            "delta_ins_a": auc_ins_a - auc_ins_a_random,
            "delta_del_a": auc_del_a_random - auc_del_a,
            "delta_ins_b": auc_ins_b - auc_ins_b_random,
            "delta_del_b": auc_del_b_random - auc_del_b,
            "attention_stability_a": stability_a,
            "attention_stability_b": stability_b,
            "attn_gnn_topk_jaccard_a": gnn_jaccard_a,
            "attn_gnn_topk_jaccard_b": gnn_jaccard_b,
            "gnn_status_a": gnn_status_a,
            "gnn_status_b": gnn_status_b,
            "degree_corr_old_a": degree_corr_old_a,
            "degree_corr_new_a": degree_corr_new_a,
            "degree_corr_old_b": degree_corr_old_b,
            "degree_corr_new_b": degree_corr_new_b,
        }
        per_pair_rows.append(row)

        torch_save(
            {
                "y_true": sample.y,
                "pred_class": pred_class,
                "target_class": target_class,
                "probs": torch.tensor(probs),
                "smiles_a": sample.smiles_a,
                "smiles_b": sample.smiles_b,
                "metrics": row,
            },
            pair_dir / "metadata.pt",
        )
        LOGGER.info("Saved explanations for %s", sample.pair_id)
        generated += 1

    if generated == 0:
        raise RuntimeError("No explanation examples were generated (all selected pairs invalid?)")

    df = pd.DataFrame(per_pair_rows)
    per_pair_csv = out_root / "explain_metrics_per_pair.csv"
    df.to_csv(per_pair_csv, index=False)

    ci_metrics = [
        "delta_ins_a",
        "delta_del_a",
        "delta_ins_b",
        "delta_del_b",
        "attention_stability_a",
        "attention_stability_b",
        "attn_gnn_topk_jaccard_a",
        "attn_gnn_topk_jaccard_b",
        "degree_corr_old_a",
        "degree_corr_new_a",
        "degree_corr_old_b",
        "degree_corr_new_b",
    ]
    bootstrap_ci = {
        metric: _bootstrap_mean_ci(
            df[metric].to_numpy(dtype=float),
            rng=bootstrap_rng,
            repeats=args.bootstrap_repeats,
            ci=args.bootstrap_ci,
        )
        for metric in ci_metrics
    }
    summary = {
        "n_pairs": int(len(df)),
        "mean": {k: float(v) for k, v in df.mean(numeric_only=True).to_dict().items()},
        "std": {k: float(v) for k, v in df.std(numeric_only=True, ddof=1).to_dict().items()},
        "bootstrap_ci": bootstrap_ci,
    }
    summary_json = out_root / "explain_metrics_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2))
    LOGGER.info("Generated explanations for %d pairs in %s", generated, out_root)
    LOGGER.info("Per-pair explanation metrics: %s", per_pair_csv)
    LOGGER.info("Summary explanation metrics: %s", summary_json)
    print(f"n_pairs={len(df)}")
    print(f"per_pair_metrics={per_pair_csv}")
    print(f"summary_metrics={summary_json}")


if __name__ == "__main__":
    main()
