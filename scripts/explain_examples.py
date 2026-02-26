#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ddigat.data.cache import GraphCache
from ddigat.data.splits import DDIPairDataset
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
    p = argparse.ArgumentParser(description="Generate explanation artifacts for example DDI pairs.")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--checkpoint", type=str, default="./outputs/checkpoints/best.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=1)  # interface consistency
    p.add_argument("--epochs", type=int, default=1)  # interface consistency
    p.add_argument("--lr", type=float, default=1e-3)  # interface consistency
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--gnnexplainer_epochs", type=int, default=50)
    return p.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_model_from_checkpoint_payload(payload: dict, device: torch.device) -> DDIPairModel:
    cfg = (payload.get("config") or {}).get("model", {})
    model = DDIPairModel(
        in_dim=int(cfg.get("in_dim", 7)),
        edge_dim=int(cfg.get("edge_dim", 5)),
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


@torch.no_grad()
def _predict_single(model: DDIPairModel, graph_a, graph_b, device: torch.device):
    graph_a = graph_a.to(device)
    graph_b = graph_b.to(device)
    logits, attn = model.forward_with_attention(graph_a, graph_b)
    probs = torch.softmax(logits, dim=-1).detach().cpu()[0]
    pred = int(torch.argmax(probs).item())
    return probs.numpy(), pred, attn


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)

    payload = torch_load(args.checkpoint, map_location=device)
    model = build_model_from_checkpoint_payload(payload, device=device)
    _, _, test_df, _ = load_tdc_drugbank_ddi(args.data_dir, output_dir=args.output_dir)
    if args.limit is not None:
        test_df = test_df.head(args.limit).copy()

    cache = GraphCache(output_dir=args.output_dir)
    ds = DDIPairDataset(test_df, cache, split_name="test")
    out_root = ensure_dir(Path(args.output_dir) / "explanations")

    generated = 0
    idx = 0
    while generated < args.n and idx < len(ds):
        sample = ds[idx]
        idx += 1
        if sample is None:
            continue

        pair_dir = ensure_dir(out_root / sample.pair_id)
        probs, pred_class, attn = _predict_single(model, sample.graph_a, sample.graph_b, device)
        target_class = pred_class

        # Attention importances
        attn_a = attn["A"]
        attn_b = attn["B"]
        node_scores_a = get_node_importance_from_attention(sample.graph_a, attn_a["edge_index"], attn_a["alpha"]).numpy()
        node_scores_b = get_node_importance_from_attention(sample.graph_b, attn_b["edge_index"], attn_b["alpha"]).numpy()

        draw_molecule_importance(sample.smiles_a, node_scores_a, pair_dir / "attention_A.png", top_k=min(10, len(node_scores_a)), legend=f"A attn | pred={pred_class}")
        draw_molecule_importance(sample.smiles_b, node_scores_b, pair_dir / "attention_B.png", top_k=min(10, len(node_scores_b)), legend=f"B attn | pred={pred_class}")
        plot_node_scores(node_scores_a, pair_dir / "attention_A_scores.png", title="Attention node scores A")
        plot_node_scores(node_scores_b, pair_dir / "attention_B_scores.png", title="Attention node scores B")

        # Faithfulness curves (A and B)
        rank_a = list(np.argsort(-node_scores_a))
        rank_b = list(np.argsort(-node_scores_b))
        k_a = _k_schedule(len(rank_a))
        k_b = _k_schedule(len(rank_b))
        del_a = deletion_test(model, sample.graph_a, sample.graph_b, target_class, rank_a, k_a, which="A", device=device)
        ins_a = insertion_test(model, sample.graph_a, sample.graph_b, target_class, rank_a, k_a, which="A", device=device)
        del_b = deletion_test(model, sample.graph_a, sample.graph_b, target_class, rank_b, k_b, which="B", device=device)
        ins_b = insertion_test(model, sample.graph_a, sample.graph_b, target_class, rank_b, k_b, which="B", device=device)

        plot_faithfulness_curves(k_a, del_a, ins_a, pair_dir / "faithfulness_A.png", title=f"Faithfulness A | target={target_class}")
        plot_faithfulness_curves(k_b, del_b, ins_b, pair_dir / "faithfulness_B.png", title=f"Faithfulness B | target={target_class}")

        # GNNExplainer baseline for A and B
        for which in ["A", "B"]:
            if which == "A":
                graph = sample.graph_a.clone().to(device)
                fixed = sample.graph_b.clone()
                smiles = sample.smiles_a
            else:
                graph = sample.graph_b.clone().to(device)
                fixed = sample.graph_a.clone()
                smiles = sample.smiles_b
            bundle = build_explainer_for_graph_contrib(
                model,
                fixed_other_graph=fixed,
                target_class=target_class,
                which=which,
                epochs=args.gnnexplainer_epochs,
            )
            explanation, node_scores, edge_mask = run_gnnexplainer_on_graph(bundle, graph)
            if node_scores is not None:
                draw_molecule_importance(
                    smiles,
                    node_scores.numpy(),
                    pair_dir / f"gnnexplainer_{which}.png",
                    top_k=min(10, int(node_scores.numel())),
                    legend=f"GNNExplainer {which}",
                )
                plot_node_scores(node_scores.numpy(), pair_dir / f"gnnexplainer_{which}_scores.png", title=f"GNNExplainer node scores {which}")
            if edge_mask is not None:
                torch_save(
                    {"edge_mask": edge_mask.detach().cpu(), "node_scores": None if node_scores is None else node_scores},
                    pair_dir / f"gnnexplainer_{which}_masks.pt",
                )

        torch_save(
            {
                "y_true": sample.y,
                "pred_class": pred_class,
                "target_class": target_class,
                "probs": torch.tensor(probs),
                "smiles_a": sample.smiles_a,
                "smiles_b": sample.smiles_b,
            },
            pair_dir / "metadata.pt",
        )
        LOGGER.info("Saved explanations for %s", sample.pair_id)
        generated += 1

    if generated == 0:
        raise RuntimeError("No explanation examples were generated (all selected pairs invalid?)")
    LOGGER.info("Generated explanations for %d pairs in %s", generated, out_root)


if __name__ == "__main__":
    main()

