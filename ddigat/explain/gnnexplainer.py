from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from ddigat.utils.logging import get_logger


LOGGER = get_logger(__name__)


class SingleGraphContributionWrapper(nn.Module):
    """Wrap DDIPairModel so PyG Explainer can explain one drug graph while the other is fixed."""

    def __init__(self, pair_model: nn.Module, fixed_other_graph: Data, which: Literal["A", "B"] = "A") -> None:
        super().__init__()
        self.pair_model = pair_model
        self.which = which
        self.fixed_other_graph_cpu = fixed_other_graph.clone().cpu()
        self.register_buffer("_fixed_emb", torch.empty(0), persistent=False)
        self._fixed_emb_ready = False

    def _pair_model_device(self) -> torch.device:
        try:
            return next(self.pair_model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @torch.no_grad()
    def _compute_fixed_embedding(self) -> torch.Tensor:
        """Precompute the fixed graph embedding once to avoid explainer edge-mask leakage.

        PyG's GNNExplainer injects edge masks into MessagePassing modules globally for the
        wrapped model. Since the DDI model shares a single encoder for both graphs, passing
        the fixed graph through the encoder during explanation can trigger edge-mask shape
        mismatches. Precomputing the fixed embedding removes the fixed-graph encoder path.
        """
        device = self._pair_model_device()
        fixed = Batch.from_data_list([self.fixed_other_graph_cpu.clone()]).to(device)
        was_training = self.pair_model.training
        self.pair_model.eval()
        emb, _ = self.pair_model.encoder.encode(fixed, return_attention=False)
        if was_training:
            self.pair_model.train()
        return emb.detach()

    def _get_fixed_embedding(self, device: torch.device) -> torch.Tensor:
        if not self._fixed_emb_ready:
            fixed_emb = self._compute_fixed_embedding().cpu()
            self._fixed_emb = fixed_emb
            self._fixed_emb_ready = True
        return self._fixed_emb.to(device)

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graph.batch = batch
        variable_emb, _ = self.pair_model.encoder.encode(graph, return_attention=False)
        fixed_emb = self._get_fixed_embedding(variable_emb.device)
        if fixed_emb.dim() == 1:
            fixed_emb = fixed_emb.unsqueeze(0)
        if fixed_emb.size(0) != variable_emb.size(0):
            if fixed_emb.size(0) == 1:
                fixed_emb = fixed_emb.expand(variable_emb.size(0), -1)
            else:
                raise ValueError(
                    f"Fixed embedding batch size {fixed_emb.size(0)} incompatible with variable batch size {variable_emb.size(0)}"
                )

        if self.which == "A":
            pair_feat = self.pair_model.build_pair_features(variable_emb, fixed_emb)
        else:
            pair_feat = self.pair_model.build_pair_features(fixed_emb, variable_emb)
        return self.pair_model.classifier(pair_feat)


@dataclass
class GNNExplainerBundle:
    explainer: object
    wrapper: nn.Module
    target_class: int
    which: str


def build_explainer_for_graph_contrib(
    model: nn.Module,
    fixed_other_graph: Data,
    target_class: int,
    which: Literal["A", "B"] = "A",
    epochs: int = 100,
) -> GNNExplainerBundle:
    """Build a PyG `Explainer` + `GNNExplainer` for one graph contribution explanation."""
    try:
        from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
    except ImportError as e:  # pragma: no cover
        raise ImportError("PyG explainability modules are unavailable in this torch_geometric build") from e

    wrapper = SingleGraphContributionWrapper(model, fixed_other_graph=fixed_other_graph, which=which)
    wrapper.eval()

    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=ModelConfig(
            mode="multiclass_classification",
            task_level="graph",
            return_type="raw",
        ),
    )
    return GNNExplainerBundle(explainer=explainer, wrapper=wrapper, target_class=int(target_class), which=which)


@torch.no_grad()
def _aggregate_node_mask(node_mask: torch.Tensor) -> torch.Tensor:
    if node_mask.dim() == 2:
        v = node_mask.detach().float().abs().mean(dim=-1)
    elif node_mask.dim() == 1:
        v = node_mask.detach().float().abs()
    else:
        raise ValueError(f"Unsupported node_mask shape {tuple(node_mask.shape)}")
    if v.numel():
        max_v = float(v.max())
        if max_v > 0:
            v = v / max_v
    return v.cpu()


@torch.no_grad()
def _node_scores_from_edge_mask(edge_index: torch.Tensor, edge_mask: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edge_mask.dim() != 1:
        edge_values = edge_mask.detach().float().abs().view(-1)
    else:
        edge_values = edge_mask.detach().float().abs()
    edge_index = edge_index.detach().cpu().long()
    edge_values = edge_values.detach().cpu()
    src = edge_index[0]
    dst = edge_index[1]
    scores = torch.zeros((num_nodes,), dtype=torch.float32)
    scores.scatter_add_(0, src, edge_values)
    scores.scatter_add_(0, dst, edge_values)
    if scores.numel():
        max_v = float(scores.max())
        if max_v > 0:
            scores = scores / max_v
    return scores


def run_gnnexplainer_on_graph(bundle: GNNExplainerBundle, graph: Data):
    """Run explainer on a single graph.

    Returns:
        explanation, node_scores, edge_mask, status

    status in {"ok", "edge_only_fallback", "no_node_mask", "failed"}.
    """
    batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=graph.x.device)
    kwargs = {
        "x": graph.x,
        "edge_index": graph.edge_index,
        "batch": batch,
        "target": torch.tensor([bundle.target_class], device=graph.x.device),
    }
    if getattr(graph, "edge_attr", None) is not None:
        kwargs["edge_attr"] = graph.edge_attr

    try:
        explanation = bundle.explainer(**kwargs)
    except TypeError:
        try:
            explanation = bundle.explainer(
                graph.x,
                graph.edge_index,
                target=kwargs["target"],
                batch=batch,
                edge_attr=kwargs.get("edge_attr"),
            )
        except Exception as e:  # pragma: no cover - PyG version/runtime specific
            LOGGER.warning("GNNExplainer failed with fallback signature: %s", e)
            return None, None, None, "failed"
    except Exception as e:  # pragma: no cover - PyG version/runtime specific
        LOGGER.warning("GNNExplainer failed: %s", e)
        return None, None, None, "failed"

    node_mask = getattr(explanation, "node_mask", None)
    edge_mask = getattr(explanation, "edge_mask", None)
    if node_mask is not None:
        node_scores = _aggregate_node_mask(node_mask)
        return explanation, node_scores, edge_mask, "ok"
    if edge_mask is not None:
        fallback = _node_scores_from_edge_mask(
            edge_index=graph.edge_index,
            edge_mask=edge_mask,
            num_nodes=int(graph.x.size(0)),
        )
        return explanation, fallback, edge_mask, "edge_only_fallback"
    return explanation, None, edge_mask, "no_node_mask"
