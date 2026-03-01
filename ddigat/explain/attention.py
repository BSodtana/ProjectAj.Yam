from __future__ import annotations

import torch
from torch_geometric.data import Data


def _ensure_2d_alpha(alpha: torch.Tensor) -> torch.Tensor:
    if alpha.dim() == 1:
        return alpha.unsqueeze(-1)
    if alpha.dim() == 2:
        return alpha
    raise ValueError(f"Unsupported alpha shape {tuple(alpha.shape)}")


def _feature_to_head_magnitude(features: torch.Tensor, num_heads: int) -> torch.Tensor:
    if features.dim() == 3:
        if features.size(1) != num_heads:
            if features.size(1) == 1:
                features = features.expand(-1, num_heads, -1)
            else:
                # Fallback for head-mismatch across PyG versions.
                features = features.mean(dim=1, keepdim=True).expand(-1, num_heads, -1)
        return torch.norm(features.float(), p=2, dim=-1)
    if features.dim() == 2:
        mag = torch.norm(features.float(), p=2, dim=-1, keepdim=True)
        return mag.expand(-1, num_heads)
    if features.dim() == 1:
        mag = features.float().abs().view(-1, 1)
        return mag.expand(-1, num_heads)
    raise ValueError(f"Unsupported feature tensor shape for message magnitude: {tuple(features.shape)}")


def _normalize_per_graph(node_scores: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    out = node_scores.clone()
    for g in batch.unique(sorted=True):
        mask = batch == g
        if not torch.any(mask):
            continue
        vals = out[mask]
        min_v = float(vals.min())
        max_v = float(vals.max())
        if max_v > min_v:
            out[mask] = (vals - min_v) / (max_v - min_v)
        elif max_v > 0:
            out[mask] = vals / max_v
        else:
            out[mask] = torch.zeros_like(vals)
    return out


def get_node_importance_from_attention(
    data: Data,
    edge_index: torch.Tensor,
    alpha: torch.Tensor,
    *,
    node_embeddings: torch.Tensor | None = None,
    src_message_embeddings: torch.Tensor | None = None,
    batch: torch.Tensor | None = None,
    remove_self_loops: bool = True,
    normalize_per_graph: bool = True,
    aggregate_heads: str = "mean",
) -> torch.Tensor:
    """Aggregate GAT attention into node importances with message-aware weighting.

    Each edge contributes per-head score `alpha(e,h) * ||message_src(e,h)||`.
    Contributions are accumulated on destination nodes and then heads are aggregated.
    """

    alpha_2d = _ensure_2d_alpha(alpha).detach().float().cpu()
    edge_index_cpu = edge_index.detach().cpu().long()
    num_nodes = int(getattr(data, "num_nodes", data.x.size(0)))
    num_heads = int(alpha_2d.size(1))

    src = edge_index_cpu[0]
    dst = edge_index_cpu[1]

    keep_mask = torch.ones(src.size(0), dtype=torch.bool)
    if remove_self_loops:
        keep_mask &= src != dst
    src = src[keep_mask]
    dst = dst[keep_mask]
    alpha_2d = alpha_2d[keep_mask]

    if src_message_embeddings is not None:
        source_features = src_message_embeddings.detach().float().cpu()
    elif node_embeddings is not None:
        source_features = node_embeddings.detach().float().cpu()
    else:
        source_features = torch.ones((num_nodes, num_heads), dtype=torch.float32)

    source_magnitude = _feature_to_head_magnitude(source_features, num_heads=num_heads)
    edge_source_magnitude = source_magnitude[src]
    edge_contrib = alpha_2d * edge_source_magnitude

    node_scores_heads = torch.zeros((num_nodes, num_heads), dtype=torch.float32)
    scatter_idx = dst.view(-1, 1).expand(-1, num_heads)
    node_scores_heads.scatter_add_(0, scatter_idx, edge_contrib)

    if aggregate_heads == "mean":
        node_scores = node_scores_heads.mean(dim=1)
    elif aggregate_heads == "sum":
        node_scores = node_scores_heads.sum(dim=1)
    else:
        raise ValueError(f"Unsupported aggregate_heads={aggregate_heads}. Use 'mean' or 'sum'.")

    if normalize_per_graph:
        if batch is not None:
            batch_vec = batch.detach().cpu().long()
        else:
            batch_attr = getattr(data, "batch", None)
            if batch_attr is None:
                batch_vec = torch.zeros((num_nodes,), dtype=torch.long)
            else:
                batch_vec = batch_attr.detach().cpu().long()
        node_scores = _normalize_per_graph(node_scores=node_scores, batch=batch_vec)

    return node_scores
