from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np

from ddigat.data.featurize import canonicalize_smiles

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors
except ImportError as e:  # pragma: no cover - dependency/runtime specific
    raise ImportError("RDKit is required for ddigat.data.drug_features") from e


PHYS_CHEM_DESCRIPTOR_NAMES: tuple[str, ...] = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "HeavyAtomCount",
)
PHYSCHEM_STD_FLOOR: float = 1e-6

_PHYS_CHEM_FNS = (
    Descriptors.MolWt,
    Descriptors.MolLogP,
    rdMolDescriptors.CalcTPSA,
    rdMolDescriptors.CalcNumHBD,
    rdMolDescriptors.CalcNumHBA,
    rdMolDescriptors.CalcNumRotatableBonds,
    getattr(rdMolDescriptors, "CalcRingCount", Descriptors.RingCount),
    getattr(rdMolDescriptors, "CalcNumAromaticRings", Descriptors.NumAromaticRings),
    getattr(rdMolDescriptors, "CalcFractionCSP3", Descriptors.FractionCSP3),
    getattr(rdMolDescriptors, "CalcHeavyAtomCount", Descriptors.HeavyAtomCount),
)


def _mol_from_smiles(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


def _canonical_unique_smiles(smiles_list: Iterable[str]) -> list[str]:
    canonical: set[str] = set()
    for smiles in smiles_list:
        c = canonicalize_smiles(str(smiles))
        if c is not None:
            canonical.add(c)
    return sorted(canonical)


def canonical_smiles_digest(smiles_list: Iterable[str]) -> str:
    canonical_sorted = _canonical_unique_smiles(smiles_list)
    payload = "\n".join(canonical_sorted)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def smiles_to_ecfp(smiles: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    mol = _mol_from_smiles(smiles)
    arr = np.zeros((int(n_bits),), dtype=np.uint8)
    if mol is None:
        return arr.astype(np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        int(radius),
        nBits=int(n_bits),
        useChirality=False,
    )
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32, copy=False)


def smiles_to_physchem(smiles: str) -> np.ndarray:
    mol = _mol_from_smiles(smiles)
    n_dim = len(PHYS_CHEM_DESCRIPTOR_NAMES)
    if mol is None:
        return np.full((n_dim,), np.nan, dtype=np.float32)

    values = []
    for fn in _PHYS_CHEM_FNS:
        try:
            values.append(float(fn(mol)))
        except Exception:
            values.append(float("nan"))
    arr = np.asarray(values, dtype=np.float32)
    arr[~np.isfinite(arr)] = np.nan
    return arr


def smiles_to_maccs(smiles: str) -> np.ndarray:
    mol = _mol_from_smiles(smiles)
    arr = np.zeros((166,), dtype=np.uint8)
    if mol is None:
        return arr.astype(np.float32)
    fp = MACCSkeys.GenMACCSKeys(mol)
    # RDKit returns 167 bits where bit 0 is unused; keep 166 informational keys.
    raw = np.zeros((167,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, raw)
    arr = raw[1:]
    return arr.astype(np.float32, copy=False)


def compute_train_feature_stats(train_smiles_list: Iterable[str]) -> dict[str, object]:
    canonical_sorted = _canonical_unique_smiles(train_smiles_list)
    rows: list[np.ndarray] = []
    invalid = 0
    for smiles in canonical_sorted:
        feat = smiles_to_physchem(smiles)
        if np.all(np.isfinite(feat)):
            rows.append(feat.astype(np.float64, copy=False))
        else:
            invalid += 1

    dim = len(PHYS_CHEM_DESCRIPTOR_NAMES)
    if rows:
        mat = np.stack(rows, axis=0)
        mean = np.mean(mat, axis=0)
        std = np.std(mat, axis=0)
        std = np.maximum(std, PHYSCHEM_STD_FLOOR)
    else:
        mean = np.zeros((dim,), dtype=np.float64)
        std = np.ones((dim,), dtype=np.float64)

    return {
        "source_split": "train",
        "descriptor_names": list(PHYS_CHEM_DESCRIPTOR_NAMES),
        "dim": int(dim),
        "mean": [float(v) for v in mean.tolist()],
        "std": [float(v) for v in std.tolist()],
        "std_floor": float(PHYSCHEM_STD_FLOOR),
        "n_unique_canonical_smiles": int(len(canonical_sorted)),
        "n_valid_molecules": int(len(rows)),
        "n_invalid_molecules": int(invalid),
        "canonical_smiles_sha256": hashlib.sha256("\n".join(canonical_sorted).encode("utf-8")).hexdigest(),
    }
