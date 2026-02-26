from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch_geometric.data import Data

from ddigat.utils.logging import get_logger


LOGGER = get_logger(__name__)

try:
    from rdkit import Chem
except ImportError as e:  # pragma: no cover - dependency/runtime specific
    raise ImportError("RDKit is required for ddigat.data.featurize") from e


_HYBRIDIZATION_MAP = {
    Chem.rdchem.HybridizationType.UNSPECIFIED: 0,
    Chem.rdchem.HybridizationType.S: 1,
    Chem.rdchem.HybridizationType.SP: 2,
    Chem.rdchem.HybridizationType.SP2: 3,
    Chem.rdchem.HybridizationType.SP3: 4,
    Chem.rdchem.HybridizationType.SP3D: 5,
    Chem.rdchem.HybridizationType.SP3D2: 6,
    Chem.rdchem.HybridizationType.OTHER: 7,
}

_BOND_TYPE_MAP = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 4,
}

_BOND_STEREO_MAP = {
    Chem.rdchem.BondStereo.STEREONONE: 0,
    Chem.rdchem.BondStereo.STEREOANY: 1,
    Chem.rdchem.BondStereo.STEREOZ: 2,
    Chem.rdchem.BondStereo.STEREOE: 3,
    Chem.rdchem.BondStereo.STEREOCIS: 4,
    Chem.rdchem.BondStereo.STEREOTRANS: 5,
}


def canonicalize_smiles(smiles: str) -> Optional[str]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _atom_features(atom: "Chem.rdchem.Atom") -> list[float]:
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetDegree()),
        float(int(atom.GetIsAromatic())),
        float(atom.GetFormalCharge()),
        float(_HYBRIDIZATION_MAP.get(atom.GetHybridization(), 0)),
        float(int(atom.IsInRing())),
        float(atom.GetTotalNumHs()),
    ]


def _bond_features(bond: "Chem.rdchem.Bond") -> list[float]:
    return [
        float(_BOND_TYPE_MAP.get(bond.GetBondType(), 0)),
        float(int(bond.GetIsConjugated())),
        float(int(bond.GetIsAromatic())),
        float(int(bond.IsInRing())),
        float(_BOND_STEREO_MAP.get(bond.GetStereo(), 0)),
    ]


def smiles_to_pyg(smiles: str) -> Optional[Data]:
    """Convert a SMILES string into a PyG `Data` graph.

    Returns `None` for invalid SMILES and logs the failure.
    """
    if not isinstance(smiles, str):
        LOGGER.warning("Non-string SMILES encountered: %r", smiles)
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        LOGGER.warning("RDKit failed to parse SMILES: %s", smiles)
        return None

    canonical = Chem.MolToSmiles(mol, canonical=True)
    atoms = mol.GetAtoms()
    if mol.GetNumAtoms() == 0:
        LOGGER.warning("Zero-atom molecule skipped: %s", smiles)
        return None

    x = torch.tensor([_atom_features(atom) for atom in atoms], dtype=torch.float32)

    edges: list[list[int]] = []
    edge_attrs: list[list[float]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = _bond_features(bond)
        edges.append([i, j])
        edge_attrs.append(bf)
        edges.append([j, i])
        edge_attrs.append(bf)

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 5), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.smiles = smiles
    data.canonical_smiles = canonical
    data.num_nodes = x.size(0)
    return data

