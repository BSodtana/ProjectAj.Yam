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
        use_ecfp_features: bool = False,
        use_physchem_features: bool = False,
        use_maccs_features: bool = False,
        ecfp_bits: int = 2048,
        physchem_dim: int = 0,
        maccs_dim: int = 166,
        ecfp_proj_dim: int = 128,
        physchem_proj_dim: int = 32,
        maccs_proj_dim: int = 32,
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

        self.use_ecfp_features = bool(use_ecfp_features)
        self.use_physchem_features = bool(use_physchem_features)
        self.use_maccs_features = bool(use_maccs_features)
        self.ecfp_bits = int(ecfp_bits)
        self.physchem_dim = int(physchem_dim)
        self.maccs_dim = int(maccs_dim)
        self.ecfp_proj_dim = int(ecfp_proj_dim)
        self.physchem_proj_dim = int(physchem_proj_dim)
        self.maccs_proj_dim = int(maccs_proj_dim)

        if self.use_ecfp_features and self.ecfp_bits <= 0:
            raise ValueError("ecfp_bits must be > 0 when use_ecfp_features=True")
        if self.use_physchem_features and self.physchem_dim <= 0:
            raise ValueError("physchem_dim must be > 0 when use_physchem_features=True")
        if self.use_maccs_features and self.maccs_dim <= 0:
            raise ValueError("maccs_dim must be > 0 when use_maccs_features=True")

        self._feature_slices: dict[str, slice] = {}
        offset = 0
        if self.use_ecfp_features:
            self._feature_slices["ecfp"] = slice(offset, offset + self.ecfp_bits)
            offset += self.ecfp_bits
            self.proj_ecfp = nn.Sequential(
                nn.Linear(self.ecfp_bits, self.ecfp_proj_dim),
                nn.LayerNorm(self.ecfp_proj_dim, elementwise_affine=False),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.proj_ecfp = None

        if self.use_physchem_features:
            self._feature_slices["physchem"] = slice(offset, offset + self.physchem_dim)
            offset += self.physchem_dim
            self.proj_physchem = nn.Sequential(
                nn.Linear(self.physchem_dim, self.physchem_proj_dim),
                nn.LayerNorm(self.physchem_proj_dim, elementwise_affine=False),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.proj_physchem = None

        if self.use_maccs_features:
            self._feature_slices["maccs"] = slice(offset, offset + self.maccs_dim)
            offset += self.maccs_dim
            self.proj_maccs = nn.Sequential(
                nn.Linear(self.maccs_dim, self.maccs_proj_dim),
                nn.LayerNorm(self.maccs_proj_dim, elementwise_affine=False),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.proj_maccs = None

        self.feature_dim = int(offset)
        has_feature_pathways = self.feature_dim > 0
        drug_input_dim = int(out_dim)
        if self.use_ecfp_features:
            drug_input_dim += self.ecfp_proj_dim
        if self.use_physchem_features:
            drug_input_dim += self.physchem_proj_dim
        if self.use_maccs_features:
            drug_input_dim += self.maccs_proj_dim

        if has_feature_pathways:
            self.drug_fusion = nn.Sequential(
                nn.Linear(drug_input_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.drug_fusion = None

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

    def _prepare_feature_tensor(self, h_gnn: torch.Tensor, feat: torch.Tensor | None) -> torch.Tensor:
        if self.feature_dim <= 0:
            return torch.empty((h_gnn.size(0), 0), dtype=h_gnn.dtype, device=h_gnn.device)
        if feat is None:
            return torch.zeros((h_gnn.size(0), self.feature_dim), dtype=h_gnn.dtype, device=h_gnn.device)
        if feat.dim() != 2:
            raise ValueError(f"Expected feat tensor with shape [B, D], got {tuple(feat.shape)}")
        if int(feat.size(0)) != int(h_gnn.size(0)):
            raise ValueError(f"Feature batch size {feat.size(0)} does not match graph batch size {h_gnn.size(0)}")
        if int(feat.size(1)) != int(self.feature_dim):
            raise ValueError(f"Feature dim mismatch: expected {self.feature_dim}, got {int(feat.size(1))}")
        return feat.to(device=h_gnn.device, dtype=h_gnn.dtype)

    def build_drug_embedding(self, h_gnn: torch.Tensor, feat: torch.Tensor | None = None) -> torch.Tensor:
        if self.feature_dim <= 0:
            return h_gnn

        feat_batch = self._prepare_feature_tensor(h_gnn=h_gnn, feat=feat)
        parts = [h_gnn]
        if self.use_ecfp_features and self.proj_ecfp is not None:
            parts.append(self.proj_ecfp(feat_batch[:, self._feature_slices["ecfp"]]))
        if self.use_physchem_features and self.proj_physchem is not None:
            parts.append(self.proj_physchem(feat_batch[:, self._feature_slices["physchem"]]))
        if self.use_maccs_features and self.proj_maccs is not None:
            parts.append(self.proj_maccs(feat_batch[:, self._feature_slices["maccs"]]))
        fused = torch.cat(parts, dim=-1)
        if self.drug_fusion is None:
            return fused
        return self.drug_fusion(fused)

    def forward(
        self,
        graph_a,
        graph_b,
        feat_a: torch.Tensor | None = None,
        feat_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h_a, _ = self.encoder.encode(graph_a, return_attention=False)
        h_b, _ = self.encoder.encode(graph_b, return_attention=False)
        h_a = self.build_drug_embedding(h_a, feat=feat_a)
        h_b = self.build_drug_embedding(h_b, feat=feat_b)
        pair = self.build_pair_features(h_a, h_b)
        return self.classifier(pair)

    def forward_with_attention(
        self,
        graph_a,
        graph_b,
        feat_a: torch.Tensor | None = None,
        feat_b: torch.Tensor | None = None,
    ):
        h_a, attn_a = self.encoder.encode(graph_a, return_attention=True)
        h_b, attn_b = self.encoder.encode(graph_b, return_attention=True)
        h_a = self.build_drug_embedding(h_a, feat=feat_a)
        h_b = self.build_drug_embedding(h_b, feat=feat_b)
        pair = self.build_pair_features(h_a, h_b)
        logits = self.classifier(pair)
        return logits, {"A": attn_a, "B": attn_b}

    def predict_proba(
        self,
        graph_a,
        graph_b,
        feat_a: torch.Tensor | None = None,
        feat_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.forward(graph_a, graph_b, feat_a=feat_a, feat_b=feat_b)
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
