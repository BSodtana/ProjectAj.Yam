from __future__ import annotations

from typing import Iterable, Literal, Sequence

import torch
from torch_geometric.data import Data


def _clone_graph(data: Data) -> Data:
    return data.clone()


def _mask_graph_nodes(data: Data, masked_nodes: set[int]) -> Data:
    """Mask nodes by zeroing features and dropping incident edges."""
    g = _clone_graph(data)
    n = int(g.x.size(0))
    if not masked_nodes:
        return g
    masked = sorted(i for i in masked_nodes if 0 <= i < n)
    if not masked:
        return g

    x = g.x.clone()
    x[masked] = 0
    g.x = x

    if g.edge_index.numel() > 0:
        src = g.edge_index[0]
        dst = g.edge_index[1]
        keep_mask = (~torch.isin(src, torch.tensor(masked, device=src.device))) & (
            ~torch.isin(dst, torch.tensor(masked, device=dst.device))
        )
        g.edge_index = g.edge_index[:, keep_mask]
        if getattr(g, "edge_attr", None) is not None and g.edge_attr.size(0) == keep_mask.size(0):
            g.edge_attr = g.edge_attr[keep_mask]
    if hasattr(g, "batch"):
        delattr(g, "batch")
    return g


@torch.no_grad()
def _target_prob(model, graph_a: Data, graph_b: Data, target_class: int, device: torch.device) -> float:
    model.eval()
    ga = graph_a.to(device)
    gb = graph_b.to(device)
    logits = model(ga, gb)
    probs = torch.softmax(logits, dim=-1)
    if probs.dim() != 2 or probs.size(0) != 1:
        raise ValueError(f"Expected single-sample logits; got probs shape={tuple(probs.shape)}")
    return float(probs[0, target_class].detach().cpu().item())


def deletion_test(
    model,
    graphA: Data,
    graphB: Data,
    target_class: int,
    node_ranked_list: Sequence[int],
    k_list: Sequence[int],
    which: Literal["A", "B"] = "A",
    device: str | torch.device = "cpu",
) -> list[float]:
    """Delete top-k important nodes progressively and return target probabilities."""
    device = torch.device(device)
    probs: list[float] = []
    ranked = [int(i) for i in node_ranked_list]
    for k in k_list:
        masked = set(ranked[: int(k)])
        if which == "A":
            ga = _mask_graph_nodes(graphA, masked)
            gb = graphB
        else:
            ga = graphA
            gb = _mask_graph_nodes(graphB, masked)
        probs.append(_target_prob(model, ga, gb, target_class, device))
    return probs


def insertion_test(
    model,
    graphA: Data,
    graphB: Data,
    target_class: int,
    node_ranked_list: Sequence[int],
    k_list: Sequence[int],
    which: Literal["A", "B"] = "A",
    device: str | torch.device = "cpu",
) -> list[float]:
    """Start from fully masked graph and restore top-k nodes, returning target probabilities."""
    device = torch.device(device)
    ranked = [int(i) for i in node_ranked_list]

    n = int(graphA.x.size(0) if which == "A" else graphB.x.size(0))
    all_nodes = set(range(n))
    probs: list[float] = []
    for k in k_list:
        keep = set(ranked[: int(k)])
        masked = all_nodes - keep
        if which == "A":
            ga = _mask_graph_nodes(graphA, masked)
            gb = graphB
        else:
            ga = graphA
            gb = _mask_graph_nodes(graphB, masked)
        probs.append(_target_prob(model, ga, gb, target_class, device))
    return probs

