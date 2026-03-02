from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ddigat.model.gnn_encoders import build_encoder


class DDIPairModel(nn.Module):
    """Siamese GAT encoder + pairwise MLP head for multi-class DDI prediction."""

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        encoder_type: str = "gat",
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
        self.encoder_type = encoder_type.lower().strip()
        self.encoder = build_encoder(
            encoder_type=self.encoder_type,
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
        self.label_smoothing = 0.0
        self.class_weights: torch.Tensor | None = None

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

    def set_loss_params(
        self,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        if class_weights is not None:
            w = class_weights.detach().float().clone().reshape(-1)
            if int(w.numel()) != int(self.num_classes):
                raise ValueError(f"class_weights must have shape ({self.num_classes},), got {tuple(w.shape)}")
            if not torch.isfinite(w).all().item():
                raise ValueError("class_weights contains non-finite values")
            if not bool((w > 0).all().item()):
                raise ValueError("class_weights must be strictly positive")
            self.class_weights = w
        else:
            self.class_weights = None
        self.label_smoothing = float(label_smoothing)

    def loss_fn(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
        return F.cross_entropy(
            logits,
            y,
            weight=weight,
            label_smoothing=self.label_smoothing,
        )
