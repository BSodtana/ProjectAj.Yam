import tempfile
import unittest
from pathlib import Path

from ddigat.benchmark import (
    build_benchmark_rows,
    compute_drw_start_epoch,
    resolve_fold_plan,
    write_benchmark_report,
)
from ddigat.utils.io import load_json


class ModelMatrixBenchmarkTest(unittest.TestCase):
    def test_build_benchmark_rows_has_expected_rows(self) -> None:
        rows = build_benchmark_rows(tau=0.5)
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0].name, "GAT only")
        self.assertEqual(rows[-1].name, "GAT + ECFP + LA + DRW")

    def test_build_benchmark_rows_can_include_xgboost_ecfp4(self) -> None:
        rows = build_benchmark_rows(tau=0.5, include_xgboost_ecfp4=True)
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[-1].name, "XGBoost + ECFP4")
        self.assertEqual(rows[-1].runner_type, "xgboost")

    def test_compute_drw_start_epoch_matches_prior_recipe(self) -> None:
        self.assertEqual(compute_drw_start_epoch(epochs=20, ratio=0.7), 15)

    def test_resolve_fold_plan_prefers_10_when_available(self) -> None:
        calls: list[int] = []

        def materialize(**kwargs) -> None:
            calls.append(int(kwargs["cold_k"]))

        plan = resolve_fold_plan(
            data_dir="./data",
            split_cache_root="/tmp/split_cache",
            split_seed=42,
            materialize_fn=materialize,
        )
        self.assertEqual(plan.resolved_cold_k, 10)
        self.assertFalse(plan.fallback_used)
        self.assertEqual(calls, [10])

    def test_resolve_fold_plan_falls_back_to_5(self) -> None:
        calls: list[int] = []

        def materialize(**kwargs) -> None:
            cold_k = int(kwargs["cold_k"])
            calls.append(cold_k)
            if cold_k == 10:
                raise RuntimeError("guardrail failure")

        plan = resolve_fold_plan(
            data_dir="./data",
            split_cache_root="/tmp/split_cache",
            split_seed=42,
            materialize_fn=materialize,
        )
        self.assertEqual(plan.resolved_cold_k, 5)
        self.assertTrue(plan.fallback_used)
        self.assertEqual(calls, [10, 5])

    def test_write_benchmark_report_writes_resolved_summary(self) -> None:
        raw_rows = []
        for fold in range(2):
            raw_rows.append(
                {
                    "model": "GAT only",
                    "accuracy": 0.5 + 0.1 * fold,
                    "macro_f1": 0.4 + 0.1 * fold,
                    "micro_f1": 0.5 + 0.1 * fold,
                    "kappa": 0.3 + 0.1 * fold,
                    "macro_roc_auc_ovr": 0.8 + 0.01 * fold,
                    "macro_pr_auc_ovr": 0.6 + 0.01 * fold,
                    "tail_macro_pr_auc_ovr": 0.2 + 0.01 * fold,
                    "ece": 0.1 + 0.01 * fold,
                    "brier_score": 0.7 + 0.01 * fold,
                    "objective_loss": 1.0 + 0.1 * fold,
                    "nll_loss": 1.0 + 0.1 * fold,
                    "n_classes_scored": 85,
                    "n_classes_missing_pos": 1,
                    "n_classes_missing_neg": 0,
                    "tail_n_classes_scored": 17,
                }
            )
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = write_benchmark_report(
                benchmark_dir=tmp_dir,
                raw_rows=raw_rows,
                summary_payload={
                    "resolved_cold_k": 5,
                    "fallback_used": True,
                },
            )
            summary = load_json(paths["summary_json"])
            self.assertEqual(summary["resolved_cold_k"], 5)
            self.assertTrue(summary["fallback_used"])
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            self.assertIn("resolved_cold_k=5", markdown)


if __name__ == "__main__":
    unittest.main()
