from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.cm as cm
import numpy as np

from ddigat.utils.io import ensure_dir
from ddigat.utils.logging import get_logger


LOGGER = get_logger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
except ImportError as e:  # pragma: no cover
    raise ImportError("RDKit is required for molecule visualization") from e


def _normalize_scores(scores: Sequence[float]) -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return arr
    min_v = float(np.nanmin(arr))
    max_v = float(np.nanmax(arr))
    if max_v > min_v:
        arr = (arr - min_v) / (max_v - min_v)
    elif max_v > 0:
        arr = arr / max_v
    else:
        arr = np.zeros_like(arr)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return arr


def draw_molecule_importance(
    smiles: str,
    node_scores: Sequence[float],
    out_path: str | Path,
    top_k: Optional[int] = None,
    legend: Optional[str] = None,
    size: tuple[int, int] = (600, 420),
) -> None:
    """Save RDKit 2D depiction with atom highlights from importance scores."""
    ensure_dir(Path(out_path).parent)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES for drawing: {smiles}")
    Chem.rdDepictor.Compute2DCoords(mol)

    scores = _normalize_scores(node_scores)
    n_atoms = mol.GetNumAtoms()
    if scores.shape[0] != n_atoms:
        raise ValueError(f"node_scores length ({scores.shape[0]}) != num_atoms ({n_atoms})")

    indices = np.arange(n_atoms)
    if top_k is not None and top_k < n_atoms:
        top_idx = set(indices[np.argsort(-scores)[:top_k]].tolist())
    else:
        top_idx = set(indices.tolist())

    cmap = cm.get_cmap("YlOrRd")
    highlight_atoms = []
    highlight_colors = {}
    highlight_radii = {}
    for idx in range(n_atoms):
        s = float(scores[idx])
        if s <= 0 and idx not in top_idx:
            continue
        highlight_atoms.append(idx)
        rgba = cmap(s)
        rgb = (float(rgba[0]), float(rgba[1]), float(rgba[2]))
        if idx not in top_idx:
            rgb = tuple(0.7 * c + 0.3 for c in rgb)
        highlight_colors[idx] = rgb
        highlight_radii[idx] = 0.25 + 0.35 * s

    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    if legend:
        opts.legendFontSize = 18
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        highlightAtomRadii=highlight_radii,
        legend=legend or "",
    )
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    with Path(out_path).open("wb") as f:
        f.write(png)

