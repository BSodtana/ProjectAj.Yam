import unittest

import pandas as pd

from ddigat.data.tdc_ddi import (
    _canonicalize_pairs,
    _make_cold_drug_split_v2,
    _make_cold_drug_split_v3,
    _prepare_pair_groups_for_cold_drug,
)


def _pair_keys(df: pd.DataFrame) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for row in df[["drug_a_smiles", "drug_b_smiles"]].itertuples(index=False):
        a = str(row[0])
        b = str(row[1])
        out.add((a, b) if a <= b else (b, a))
    return out


def _drug_set(df: pd.DataFrame) -> set[str]:
    return set(df["drug_a_smiles"].astype(str).tolist()).union(set(df["drug_b_smiles"].astype(str).tolist()))


def _build_small_df() -> pd.DataFrame:
    rows = [
        {"drug_a_smiles": "B", "drug_b_smiles": "A", "y": 1},
        {"drug_a_smiles": "A", "drug_b_smiles": "B", "y": 2},  # conflict with above
        {"drug_a_smiles": "C", "drug_b_smiles": "D", "y": 3},
        {"drug_a_smiles": "D", "drug_b_smiles": "C", "y": 3},
        {"drug_a_smiles": "D", "drug_b_smiles": "C", "y": 3},
        {"drug_a_smiles": "E", "drug_b_smiles": "F", "y": 4},
    ]
    return pd.DataFrame(rows)


def _build_dense_df(n_drugs: int = 12, n_classes: int = 6) -> pd.DataFrame:
    drugs = [f"D{i:02d}" for i in range(n_drugs)]
    rows: list[dict[str, object]] = []
    for i, a in enumerate(drugs):
        for j, b in enumerate(drugs):
            if i >= j:
                continue
            rows.append({"drug_a_smiles": a, "drug_b_smiles": b, "y": int((i + j) % n_classes)})

    # Add duplicate non-conflict evidence rows.
    rows.append({"drug_a_smiles": "D02", "drug_b_smiles": "D03", "y": 5})
    rows.append({"drug_a_smiles": "D03", "drug_b_smiles": "D02", "y": 5})

    # Add one conflict pair to ensure ambiguity filtering is active.
    rows.append({"drug_a_smiles": "D00", "drug_b_smiles": "D01", "y": 0})
    rows.append({"drug_a_smiles": "D01", "drug_b_smiles": "D00", "y": 1})
    return pd.DataFrame(rows)


class ColdDrugSplitTest(unittest.TestCase):
    def test_canonical_pair_key_symmetry(self) -> None:
        df = _build_small_df()
        out = _canonicalize_pairs(df)
        ab = out[(out["drug_a_smiles"] == "A") & (out["drug_b_smiles"] == "B")]
        self.assertGreaterEqual(len(ab), 2)
        self.assertEqual(int(ab["pair_key"].nunique()), 1)

    def test_conflict_drop_and_multiplicity_keep_all(self) -> None:
        rows_df, pair_df, stats = _prepare_pair_groups_for_cold_drug(_build_small_df(), dedupe_policy="keep_all")
        self.assertEqual(int(stats["conflict_pair_groups_dropped"]), 1)
        self.assertEqual(int(stats["kept_pair_groups"]), 2)
        self.assertEqual(int(stats["kept_rows_after_policy"]), 4)
        # C-D has multiplicity 3 after conflict filtering.
        cd = pair_df[(pair_df["drug_a_smiles"] == "C") & (pair_df["drug_b_smiles"] == "D")]
        self.assertEqual(int(cd.iloc[0]["multiplicity"]), 3)
        self.assertEqual(int(cd.iloc[0]["row_weight"]), 3)
        # Conflicting A-B pair removed from row-level output.
        keys = set(rows_df["pair_key"].astype(str).tolist())
        self.assertNotIn("A||B", keys)

    def test_keep_first_dedup_policy(self) -> None:
        rows_df, pair_df, stats = _prepare_pair_groups_for_cold_drug(_build_small_df(), dedupe_policy="keep_first")
        self.assertEqual(int(stats["kept_pair_groups"]), 2)
        self.assertEqual(int(stats["kept_rows_after_policy"]), 2)
        self.assertEqual(int(pair_df["row_weight"].sum()), 2)
        self.assertEqual(int(rows_df["pair_key"].nunique()), 2)

    def test_s1_split_is_pair_disjoint_and_group_atomic_keep_all(self) -> None:
        df = _build_dense_df()
        train_df, valid_df, test_df, report = _make_cold_drug_split_v3(
            full_df=df,
            seed=42,
            k=5,
            fold_idx=0,
            protocol="s1",
            min_test_pairs=1,
            min_test_labels=1,
            max_resamples=20,
            dedupe_policy="keep_all",
            num_classes=6,
        )
        train_keys = _pair_keys(train_df)
        valid_keys = _pair_keys(valid_df)
        test_keys = _pair_keys(test_df)
        self.assertTrue(train_keys.isdisjoint(valid_keys))
        self.assertTrue(train_keys.isdisjoint(test_keys))
        self.assertTrue(valid_keys.isdisjoint(test_keys))

        # D02-D03 appears 3 times in source (1 base + 2 duplicates) and must stay in one split.
        dup_key = ("D02", "D03")
        counts = 0
        owner_splits = 0
        for split_df in [train_df, valid_df, test_df]:
            c = sum(1 for row in split_df[["drug_a_smiles", "drug_b_smiles"]].itertuples(index=False) if tuple(sorted((str(row[0]), str(row[1])))) == dup_key)
            counts += c
            if c > 0:
                owner_splits += 1
        self.assertEqual(counts, 3)
        self.assertEqual(owner_splits, 1)

        # Conflict key is removed globally.
        all_keys = train_keys | valid_keys | test_keys
        self.assertNotIn(("D00", "D01"), all_keys)
        self.assertIn("summary", report)

    def test_s1_split_is_strict_cold_drug_disjoint_by_drug_identity(self) -> None:
        df = _build_dense_df()
        train_df, valid_df, test_df, _ = _make_cold_drug_split_v3(
            full_df=df,
            seed=42,
            k=5,
            fold_idx=0,
            protocol="s1",
            min_test_pairs=1,
            min_test_labels=1,
            max_resamples=20,
            dedupe_policy="keep_all",
            num_classes=6,
            selection_objective="selected_fold",
        )
        train_drugs = _drug_set(train_df)
        for split_df in [train_df, valid_df, test_df]:
            for row in split_df[["drug_a_smiles", "drug_b_smiles"]].itertuples(index=False):
                a = str(row[0])
                b = str(row[1])
                unseen_vs_train = int(a not in train_drugs) + int(b not in train_drugs)
                if split_df is train_df:
                    self.assertEqual(unseen_vs_train, 0)
                else:
                    # Strict S1 protocol: valid/test pairs contain exactly one unseen drug.
                    self.assertEqual(unseen_vs_train, 1)

    def test_keep_first_collapses_duplicates(self) -> None:
        df = _build_dense_df()
        train_df, valid_df, test_df, _ = _make_cold_drug_split_v3(
            full_df=df,
            seed=42,
            k=5,
            fold_idx=0,
            protocol="s1",
            min_test_pairs=1,
            min_test_labels=1,
            max_resamples=20,
            dedupe_policy="keep_first",
            num_classes=6,
        )
        dup_key = ("D02", "D03")
        counts = 0
        for split_df in [train_df, valid_df, test_df]:
            counts += sum(
                1
                for row in split_df[["drug_a_smiles", "drug_b_smiles"]].itertuples(index=False)
                if tuple(sorted((str(row[0]), str(row[1])))) == dup_key
            )
        self.assertEqual(counts, 1)

    def test_guardrail_failure_raises(self) -> None:
        df = _build_dense_df()
        with self.assertRaises(RuntimeError):
            _make_cold_drug_split_v3(
                full_df=df,
                seed=42,
                k=5,
                fold_idx=0,
                protocol="s1",
                min_test_pairs=1_000_000,
                min_test_labels=1,
                max_resamples=3,
                dedupe_policy="keep_all",
                num_classes=6,
            )

    def test_cold_drug_v2_regression_shape(self) -> None:
        df = _build_dense_df(n_drugs=14, n_classes=6)
        train_df, valid_df, test_df = _make_cold_drug_split_v2(full_df=df, seed=42)
        self.assertGreater(len(train_df), 0)
        self.assertGreater(len(valid_df), 0)
        self.assertGreater(len(test_df), 0)
        train_drugs = set(train_df["drug_a_smiles"]).union(set(train_df["drug_b_smiles"]))
        test_drugs = set(test_df["drug_a_smiles"]).union(set(test_df["drug_b_smiles"]))
        self.assertTrue(train_drugs.isdisjoint(test_drugs))


if __name__ == "__main__":
    unittest.main()
