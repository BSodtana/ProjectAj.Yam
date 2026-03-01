from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, GINConv, global_add_pool, global_mean_pool


class _BaseEncoder(nn.Module):
    def __init__(self, pooling: str = "mean") -> None:
        super().__init__()
        if pooling not in {"mean", "add"}:
            raise ValueError(f"Unsupported pooling={pooling}; use 'mean' or 'add'")
        self.pooling = pooling

    def _pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return global_mean_pool(x, batch)
        return global_add_pool(x, batch)

    @staticmethod
    def _batch_or_zeros(data: Data) -> torch.Tensor:
        batch = getattr(data, "batch", None)
        if batch is None:
            return data.x.new_zeros((data.x.size(0),), dtype=torch.long)
        return batch


class GATEncoder(_BaseEncoder):
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
        super().__init__(pooling=pooling)
        if num_layers < 2:
            raise ValueError("GATEncoder requires num_layers >= 2")
        self.dropout = float(dropout)

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

    @staticmethod
    def _reshape_transformed_features(
        transformed: torch.Tensor,
        heads: int,
        out_channels: int,
    ) -> torch.Tensor:
        if transformed.dim() != 2:
            return transformed
        if transformed.size(-1) == heads * out_channels:
            return transformed.view(transformed.size(0), heads, out_channels)
        if heads == 1 and transformed.size(-1) == out_channels:
            return transformed.unsqueeze(1)
        return transformed

    def _compute_src_message_embeddings(self, conv: GATConv, x_in: torch.Tensor) -> Optional[torch.Tensor]:
        transformed: Optional[torch.Tensor] = None
        try:
            if getattr(conv, "lin", None) is not None:
                transformed = conv.lin(x_in)
            elif getattr(conv, "lin_src", None) is not None:
                transformed = conv.lin_src(x_in)
        except Exception:
            transformed = None
        if transformed is None:
            return None
        try:
            return self._reshape_transformed_features(transformed, heads=conv.heads, out_channels=conv.out_channels)
        except Exception:
            return transformed

    def encode(
        self, data: Data, return_attention: bool = False
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        batch = self._batch_or_zeros(data)

        attn_bundle = None
        for i, conv in enumerate(self.convs):
            is_last = i == len(self.convs) - 1
            if is_last and return_attention:
                x_in = x
                x, (attn_edge_index, alpha) = conv(
                    x,
                    edge_index,
                    edge_attr=edge_attr,
                    return_attention_weights=True,
                )
                src_message_embeddings = self._compute_src_message_embeddings(conv, x_in)
                attn_bundle = {
                    "edge_index": attn_edge_index,
                    "alpha": alpha,
                    "batch": batch,
                    "node_embeddings": x,
                    "node_input_embeddings": x_in,
                    "src_message_embeddings": src_message_embeddings,
                }
            else:
                x = conv(x, edge_index, edge_attr=edge_attr)
            if not is_last:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        graph_emb = self._pool(x, batch)
        return graph_emb, attn_bundle


class GCNEncoder(_BaseEncoder):
    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 128,
        num_layers: int = 3,
        heads: int = 1,
        dropout: float = 0.2,
        pooling: str = "mean",
    ) -> None:
        super().__init__(pooling=pooling)
        if num_layers < 2:
            raise ValueError("GCNEncoder requires num_layers >= 2")
        self.dropout = float(dropout)
        dims = [in_dim] + [hidden_dim] * (num_layers - 2) + [out_dim]
        self.convs = nn.ModuleList([GCNConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def encode(
        self, data: Data, return_attention: bool = False
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        x = data.x
        edge_index = data.edge_index
        batch = self._batch_or_zeros(data)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        graph_emb = self._pool(x, batch)
        return graph_emb, None


class GINEncoder(_BaseEncoder):
    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 128,
        num_layers: int = 3,
        heads: int = 1,
        dropout: float = 0.2,
        pooling: str = "mean",
    ) -> None:
        super().__init__(pooling=pooling)
        if num_layers < 2:
            raise ValueError("GINEncoder requires num_layers >= 2")
        self.dropout = float(dropout)
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(nn=mlp))
        self.proj_out = nn.Linear(hidden_dim, out_dim)

    def encode(
        self, data: Data, return_attention: bool = False
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        x = self.proj_in(data.x)
        edge_index = data.edge_index
        batch = self._batch_or_zeros(data)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.proj_out(x)
        graph_emb = self._pool(x, batch)
        return graph_emb, None


def build_encoder(
    encoder_type: str,
    in_dim: int,
    edge_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int,
    heads: int,
    dropout: float,
    pooling: str,
) -> nn.Module:
    et = encoder_type.lower().strip()
    if et == "gat":
        return GATEncoder(
            in_dim=in_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            pooling=pooling,
        )
    if et == "gcn":
        return GCNEncoder(
            in_dim=in_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            pooling=pooling,
        )
    if et == "gin":
        return GINEncoder(
            in_dim=in_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            pooling=pooling,
        )
    raise ValueError(f"Unsupported encoder_type={encoder_type}. Expected one of: gat, gcn, gin")
