import importlib.util
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Data

from ddigat.model.pair_model import DDIPairModel


def _dummy_graph() -> Data:
    return Data(
        x=torch.zeros((1, 7), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 5), dtype=torch.float32),
    )


class PairModelMLPTest(unittest.TestCase):
    def test_build_pair_features_is_swap_invariant(self) -> None:
        h_a = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float32)
        h_b = torch.tensor([[4.0, -5.0, 0.5]], dtype=torch.float32)
        pair_ab = DDIPairModel.build_pair_features(h_a, h_b)
        pair_ba = DDIPairModel.build_pair_features(h_b, h_a)
        self.assertTrue(torch.allclose(pair_ab, pair_ba))

    def test_mlp_requires_feature_pathway(self) -> None:
        with self.assertRaises(ValueError):
            DDIPairModel(
                in_dim=7,
                edge_dim=5,
                encoder_type="mlp",
                num_classes=4,
            )

    def test_mlp_forward_with_ecfp_features(self) -> None:
        model = DDIPairModel(
            in_dim=7,
            edge_dim=5,
            encoder_type="mlp",
            hidden_dim=16,
            out_dim=8,
            mlp_hidden_dim=16,
            num_classes=4,
            use_ecfp_features=True,
            ecfp_bits=16,
            ecfp_proj_dim=8,
        )
        graph_a = _dummy_graph()
        graph_b = _dummy_graph()
        feat_a = torch.randn(2, 16)
        feat_b = torch.randn(2, 16)
        logits = model(graph_a, graph_b, feat_a=feat_a, feat_b=feat_b)
        self.assertEqual(tuple(logits.shape), (2, 4))

    def test_mlp_forward_is_swap_invariant(self) -> None:
        model = DDIPairModel(
            in_dim=7,
            edge_dim=5,
            encoder_type="mlp",
            hidden_dim=16,
            out_dim=8,
            mlp_hidden_dim=16,
            num_classes=4,
            use_ecfp_features=True,
            ecfp_bits=16,
            ecfp_proj_dim=8,
            dropout=0.0,
        )
        model.eval()
        graph_a = _dummy_graph()
        graph_b = _dummy_graph()
        feat_a = torch.randn(2, 16)
        feat_b = torch.randn(2, 16)
        logits_ab = model(graph_a, graph_b, feat_a=feat_a, feat_b=feat_b)
        logits_ba = model(graph_b, graph_a, feat_a=feat_b, feat_b=feat_a)
        self.assertTrue(torch.allclose(logits_ab, logits_ba, atol=1e-6, rtol=1e-6))

    def test_gat_forward_is_swap_invariant(self) -> None:
        model = DDIPairModel(
            in_dim=7,
            edge_dim=5,
            encoder_type="gat",
            hidden_dim=8,
            out_dim=8,
            num_layers=2,
            heads=1,
            dropout=0.0,
            mlp_hidden_dim=16,
            num_classes=4,
        )
        model.eval()
        graph_a = _dummy_graph()
        graph_b = _dummy_graph()
        graph_a.x = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        graph_b.x = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        logits_ab = model(graph_a, graph_b)
        logits_ba = model(graph_b, graph_a)
        self.assertTrue(torch.allclose(logits_ab, logits_ba, atol=1e-6, rtol=1e-6))

    def test_evaluate_checkpoint_builder_supports_mlp(self) -> None:
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
        spec = importlib.util.spec_from_file_location("evaluate_script", script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Failed to load scripts/evaluate.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        model = DDIPairModel(
            in_dim=7,
            edge_dim=5,
            encoder_type="mlp",
            hidden_dim=16,
            out_dim=8,
            mlp_hidden_dim=16,
            num_classes=4,
            use_ecfp_features=True,
            ecfp_bits=16,
            ecfp_proj_dim=8,
        )
        payload = {
            "config": {
                "model": {
                    "in_dim": 7,
                    "edge_dim": 5,
                    "encoder_type": "mlp",
                    "hidden_dim": 16,
                    "out_dim": 8,
                    "num_layers": 3,
                    "heads": 1,
                    "dropout": 0.2,
                    "mlp_hidden_dim": 16,
                    "num_classes": 4,
                    "pooling": "mean",
                    "use_ecfp_features": True,
                    "ecfp_bits": 16,
                    "ecfp_proj_dim": 8,
                }
            },
            "model_state_dict": model.state_dict(),
        }
        built = module.build_model_from_checkpoint_payload(
            payload,
            torch.device("cpu"),
            feature_cfg={
                "use_ecfp_features": True,
                "use_physchem_features": False,
                "use_maccs_features": False,
                "ecfp_bits": 16,
                "physchem_dim": 0,
                "maccs_dim": 166,
            },
        )
        self.assertEqual(built.encoder_type, "mlp")
        self.assertTrue(built.is_feature_only)


if __name__ == "__main__":
    unittest.main()
