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
        self.fixed_other_graph_cpu = fixed_other_graph.clone().cpu()
        self.which = which

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graph.batch = batch

        # Explainer calls graph-level models with one graph in practice here.
        fixed = self.fixed_other_graph_cpu.clone()
        fixed = Batch.from_data_list([fixed]).to(x.device)

        if self.which == "A":
            logits = self.pair_model(graph, fixed)
        else:
            logits = self.pair_model(fixed, graph)
        return logits


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


def run_gnnexplainer_on_graph(bundle: GNNExplainerBundle, graph: Data):
    """Run the explainer on a single graph and return explanation + aggregated node mask."""
    batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=graph.x.device)
    kwargs = {"x": graph.x, "edge_index": graph.edge_index, "batch": batch, "target": torch.tensor([bundle.target_class], device=graph.x.device)}
    if getattr(graph, "edge_attr", None) is not None:
        kwargs["edge_attr"] = graph.edge_attr

    try:
        explanation = bundle.explainer(**kwargs)
    except TypeError:
        # Compatibility fallback for older/newer signatures.
        explanation = bundle.explainer(graph.x, graph.edge_index, target=kwargs["target"], batch=batch, edge_attr=kwargs.get("edge_attr"))

    node_mask = getattr(explanation, "node_mask", None)
    edge_mask = getattr(explanation, "edge_mask", None)
    node_scores = _aggregate_node_mask(node_mask) if node_mask is not None else None
    return explanation, node_scores, edge_mask

