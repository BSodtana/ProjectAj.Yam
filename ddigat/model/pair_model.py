from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ddigat.model.gat_encoder import GATEncoder


class DDIPairModel(nn.Module):
    """Siamese GAT encoder + pairwise MLP head for multi-class DDI prediction."""

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
        mlp_hidden_dim: int = 256,
        num_classes: int = 86,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.encoder = GATEncoder(
            in_dim=in_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            pooling=pooling,
        )
        pair_dim = out_dim * 4
        self.classifier = nn.Sequential(
            nn.Linear(pair_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, num_classes),
        )
        self.num_classes = num_classes

    @staticmethod
    def build_pair_features(h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        return torch.cat([h_a, h_b, torch.abs(h_a - h_b), h_a * h_b], dim=-1)

    def forward(self, graph_a, graph_b) -> torch.Tensor:
        h_a, _ = self.encoder.encode(graph_a, return_attention=False)
        h_b, _ = self.encoder.encode(graph_b, return_attention=False)
        pair = self.build_pair_features(h_a, h_b)
        return self.classifier(pair)

    def forward_with_attention(self, graph_a, graph_b):
        h_a, attn_a = self.encoder.encode(graph_a, return_attention=True)
        h_b, attn_b = self.encoder.encode(graph_b, return_attention=True)
        pair = self.build_pair_features(h_a, h_b)
        logits = self.classifier(pair)
        return logits, {"A": attn_a, "B": attn_b}

    def predict_proba(self, graph_a, graph_b) -> torch.Tensor:
        logits = self.forward(graph_a, graph_b)
        return F.softmax(logits, dim=-1)

    @staticmethod
    def loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, y)

