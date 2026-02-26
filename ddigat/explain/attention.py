from __future__ import annotations

import torch
from torch_geometric.data import Data


def get_node_importance_from_attention(
    data: Data,
    edge_index: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Aggregate GAT edge attention into node importances via incoming-edge sums.

    Args:
        data: Graph data object with `num_nodes`.
        edge_index: Edge indices from GAT attention output, shape [2, E].
        alpha: Attention weights shape [E] or [E, heads].
    """
    if alpha.dim() == 2:
        edge_scores = alpha.mean(dim=-1)
    elif alpha.dim() == 1:
        edge_scores = alpha
    else:
        raise ValueError(f"Unsupported alpha shape {tuple(alpha.shape)}")

    edge_scores = edge_scores.detach().float().cpu()
    edge_index = edge_index.detach().cpu()
    num_nodes = int(getattr(data, "num_nodes", data.x.size(0)))

    node_scores = torch.zeros(num_nodes, dtype=torch.float32)
    dst = edge_index[1]
    node_scores.scatter_add_(0, dst, edge_scores)

    # Min-max normalize for visualization/readability.
    if num_nodes > 0:
        min_v = float(node_scores.min())
        max_v = float(node_scores.max())
        if max_v > min_v:
            node_scores = (node_scores - min_v) / (max_v - min_v)
        elif max_v > 0:
            node_scores = node_scores / max_v
    return node_scores

