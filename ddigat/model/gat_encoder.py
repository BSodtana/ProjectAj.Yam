from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_add_pool, global_mean_pool


class GATEncoder(nn.Module):
    """Graph encoder with optional attention extraction from the last GAT layer."""

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("GATEncoder requires num_layers >= 2")
        if pooling not in {"mean", "add"}:
            raise ValueError(f"Unsupported pooling={pooling}; use 'mean' or 'add'")

        self.dropout = float(dropout)
        self.pooling = pooling
        self.num_layers = num_layers

        convs = []
        in_channels = in_dim
        for layer_idx in range(num_layers):
            is_last = layer_idx == num_layers - 1
            out_channels = out_dim if is_last else hidden_dim
            concat = not is_last
            conv = GATConv(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=heads,
                concat=concat,
                dropout=dropout,
                edge_dim=edge_dim,
                add_self_loops=True,
            )
            convs.append(conv)
            in_channels = out_channels if is_last else out_channels * heads
        self.convs = nn.ModuleList(convs)

    def _pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return global_mean_pool(x, batch)
        return global_add_pool(x, batch)

    def encode(
        self, data: Data, return_attention: bool = False
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros((x.size(0),), dtype=torch.long)

        attn_bundle = None
        for i, conv in enumerate(self.convs):
            is_last = i == len(self.convs) - 1
            if is_last and return_attention:
                x, (attn_edge_index, alpha) = conv(
                    x,
                    edge_index,
                    edge_attr=edge_attr,
                    return_attention_weights=True,
                )
                attn_bundle = {
                    "edge_index": attn_edge_index,
                    "alpha": alpha,
                    "batch": batch,
                    "node_embeddings": x,
                }
            else:
                x = conv(x, edge_index, edge_attr=edge_attr)

            if not is_last:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        graph_emb = self._pool(x, batch)
        return graph_emb, attn_bundle

    def forward(self, data: Data) -> torch.Tensor:
        emb, _ = self.encode(data, return_attention=False)
        return emb

