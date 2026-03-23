import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ddigat.data.cache import DrugFeatureCache


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_xgboost_ecfp4.py"
SPEC = importlib.util.spec_from_file_location("run_xgboost_ecfp4", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load script module from {SCRIPT_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class XGBoostECFP4ScriptTest(unittest.TestCase):
    def test_canonical_pair_returns_none_for_invalid_smiles(self) -> None:
        self.assertIsNone(MODULE._canonical_pair("CCO", "not_a_smiles"))

    def test_build_pair_matrix_skips_invalid_rows(self) -> None:
        df = pd.DataFrame(
            {
                "drug_a_smiles": ["CCO", "CCO"],
                "drug_b_smiles": ["O", "not_a_smiles"],
                "y": [0, 1],
            }
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            feature_cache = DrugFeatureCache(
                output_dir=tmp_dir,
                use_ecfp=True,
                use_physchem=False,
                use_maccs=False,
                ecfp_bits=2048,
                ecfp_radius=2,
            )
            smiles_values = list(df["drug_a_smiles"]) + list(df["drug_b_smiles"])
            feature_bank, feature_index = MODULE._build_feature_bank(smiles_values, feature_cache=feature_cache)
            matrix, filtered_df, skipped = MODULE._build_pair_matrix(
                df,
                feature_bank=feature_bank,
                feature_index=feature_index,
            )
            self.assertEqual(matrix.shape[0], 1)
            self.assertEqual(len(filtered_df), 1)
            self.assertEqual(int(skipped), 1)

    def test_encode_training_labels_makes_labels_contiguous(self) -> None:
        y_train = MODULE.np.asarray([1, 4, 4, 9], dtype=MODULE.np.int64)
        y_valid = MODULE.np.asarray([4, 7, 9], dtype=MODULE.np.int64)
        y_train_encoded, y_valid_encoded, valid_mask, class_to_index = MODULE._encode_training_labels(y_train, y_valid)
        self.assertEqual(y_train_encoded.tolist(), [0, 1, 1, 2])
        self.assertEqual(valid_mask.tolist(), [True, False, True])
        self.assertEqual(y_valid_encoded.tolist(), [1, 2])
        self.assertEqual(class_to_index, {1: 0, 4: 1, 9: 2})

    def test_expand_probabilities_restores_full_class_space(self) -> None:
        compact = MODULE.np.asarray([[0.2, 0.3, 0.5]], dtype=MODULE.np.float64)
        full = MODULE._expand_probabilities(compact, class_to_index={1: 0, 4: 1, 9: 2}, num_classes=12)
        self.assertEqual(full.shape, (1, 12))
        self.assertAlmostEqual(float(full[0, 1]), 0.2)
        self.assertAlmostEqual(float(full[0, 4]), 0.3)
        self.assertAlmostEqual(float(full[0, 9]), 0.5)
        self.assertAlmostEqual(float(full.sum()), 1.0)

    def test_make_stratified_folds_balances_classes_when_min_count_lt_k(self) -> None:
        y = np.asarray([0] * 8 + [1] * 4 + [2] * 3 + [3], dtype=np.int64)
        folds = MODULE._make_stratified_folds(y, n_splits=5, seed=42)
        self.assertEqual(len(folds), 5)

        seen_test_indices: list[int] = []
        per_label_test_counts: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}
        for train_idx, test_idx in folds:
            self.assertGreater(len(train_idx), 0)
            self.assertGreater(len(test_idx), 0)
            self.assertEqual(len(np.intersect1d(train_idx, test_idx)), 0)
            seen_test_indices.extend(int(v) for v in test_idx.tolist())
            labels, counts = np.unique(y[test_idx], return_counts=True)
            count_map = {int(label): int(count) for label, count in zip(labels.tolist(), counts.tolist())}
            for label in per_label_test_counts:
                per_label_test_counts[label].append(count_map.get(label, 0))

        self.assertEqual(sorted(seen_test_indices), list(range(len(y))))
        for counts in per_label_test_counts.values():
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_make_stratified_validation_split_keeps_singletons_in_train(self) -> None:
        y = np.asarray([0, 0, 0, 1, 1, 2], dtype=np.int64)
        train_idx, valid_idx = MODULE._make_stratified_validation_split(y, valid_fraction=0.4, seed=7)
        self.assertEqual(sorted(train_idx.tolist() + valid_idx.tolist()), list(range(len(y))))
        self.assertEqual(len(np.intersect1d(train_idx, valid_idx)), 0)
        self.assertIn(5, train_idx.tolist())
        self.assertNotIn(5, valid_idx.tolist())

    def test_aggregate_fold_summaries_returns_mean_and_std(self) -> None:
        mean_summary, std_summary = MODULE._aggregate_fold_summaries(
            [
                {"accuracy": 0.4, "macro_f1": 0.2},
                {"accuracy": 0.6, "macro_f1": 0.4},
                {"accuracy": 0.8, "macro_f1": 0.6},
            ]
        )
        self.assertAlmostEqual(float(mean_summary["accuracy"]), 0.6)
        self.assertAlmostEqual(float(mean_summary["macro_f1"]), 0.4)
        self.assertAlmostEqual(float(std_summary["accuracy"]), np.std([0.4, 0.6, 0.8], ddof=0))
        self.assertAlmostEqual(float(std_summary["macro_f1"]), np.std([0.2, 0.4, 0.6], ddof=0))

    def test_split_overlap_summary_detects_reversed_pair_leakage(self) -> None:
        train_df = pd.DataFrame(
            {
                "drug_a_smiles": ["CCO"],
                "drug_b_smiles": ["O"],
                "y": [0],
            }
        )
        valid_df = pd.DataFrame(
            {
                "drug_a_smiles": ["CCN"],
                "drug_b_smiles": ["N"],
                "y": [1],
            }
        )
        test_df = pd.DataFrame(
            {
                "drug_a_smiles": ["O"],
                "drug_b_smiles": ["CCO"],
                "y": [2],
            }
        )
        summary = MODULE._split_overlap_summary(train_df, valid_df, test_df)
        self.assertEqual(int(summary["train_test_pair_overlap"]), 1)
        self.assertEqual(int(summary["train_valid_pair_overlap"]), 0)
        self.assertGreaterEqual(int(summary["train_test_drug_overlap"]), 2)

    def test_assert_zero_pair_overlap_raises_for_leaky_split(self) -> None:
        with self.assertRaises(AssertionError):
            MODULE._assert_zero_pair_overlap(
                {
                    "train_valid_pair_overlap": 0,
                    "train_test_pair_overlap": 1,
                    "valid_test_pair_overlap": 0,
                },
                run_name="unit_test",
            )


if __name__ == "__main__":
    unittest.main()
