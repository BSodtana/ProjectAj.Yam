from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from ddigat.data.drug_features import (
    PHYS_CHEM_DESCRIPTOR_NAMES,
    PHYSCHEM_STD_FLOOR,
    smiles_to_ecfp,
    smiles_to_maccs,
    smiles_to_physchem,
)
from ddigat.data.featurize import canonicalize_smiles, smiles_to_pyg
from ddigat.utils.io import ensure_dir, load_json, save_json, torch_load, torch_save
from ddigat.utils.logging import get_logger


LOGGER = get_logger(__name__)


class GraphCache:
    """Persistent cache of featurized PyG graphs keyed by canonical SMILES hash."""

    def __init__(self, output_dir: str | Path):
        self.root = ensure_dir(Path(output_dir) / "processed_graphs")
        self.index_path = self.root / "index.json"
        self.index = self._load_index()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "saved": 0,
            "skipped_invalid": 0,
            "load_failures": 0,
        }

    def _load_index(self) -> dict[str, str]:
        if self.index_path.exists():
            try:
                return load_json(self.index_path)
            except Exception as e:
                LOGGER.warning("Failed loading cache index %s: %s", self.index_path, e)
        return {}

    def _save_index(self) -> None:
        save_json(self.index, self.index_path)

    @staticmethod
    def _hash_smiles(canonical_smiles: str) -> str:
        return hashlib.sha256(canonical_smiles.encode("utf-8")).hexdigest()[:24]

    def _path_for_canonical(self, canonical_smiles: str) -> Path:
        digest = self._hash_smiles(canonical_smiles)
        return self.root / f"{digest}.pt"

    def get_graph_path(self, smiles: str) -> Optional[Path]:
        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            return None
        p = self.index.get(canonical)
        if p is None:
            return None
        return Path(p)

    def get_or_create(self, smiles: str):
        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            self.stats["skipped_invalid"] += 1
            return None

        cached = self.index.get(canonical)
        if cached and Path(cached).exists():
            self.stats["hits"] += 1
            try:
                return torch_load(cached)
            except Exception as e:
                self.stats["load_failures"] += 1
                LOGGER.warning("Cache load failed for %s (%s): %s", smiles, cached, e)

        self.stats["misses"] += 1
        data = smiles_to_pyg(canonical)
        if data is None:
            self.stats["skipped_invalid"] += 1
            return None
        out_path = self._path_for_canonical(canonical)
        torch_save(data, out_path)
        self.index[canonical] = str(out_path)
        self._save_index()
        self.stats["saved"] += 1
        return data

    def build(self, smiles_list: Iterable[str], show_progress: bool = True) -> dict[str, int]:
        try:
            from tqdm.auto import tqdm
        except Exception:  # pragma: no cover
            tqdm = None

        iterable = smiles_list
        if show_progress and tqdm is not None:
            smiles_list = list(smiles_list)
            iterable = tqdm(smiles_list, desc="Caching graphs")
        for s in iterable:
            self.get_or_create(s)
        return dict(self.stats)


class DrugFeatureCache:
    """Persistent cache for optional SMILES-derived features by canonical SMILES hash."""

    def __init__(
        self,
        output_dir: str | Path,
        use_ecfp: bool = False,
        use_physchem: bool = False,
        use_maccs: bool = False,
        ecfp_bits: int = 2048,
        ecfp_radius: int = 2,
        physchem_stats: Optional[dict[str, object]] = None,
    ) -> None:
        self.root = ensure_dir(Path(output_dir) / "drug_features")
        self.ecfp_dir = ensure_dir(self.root / "ecfp")
        self.physchem_dir = ensure_dir(self.root / "physchem")
        self.maccs_dir = ensure_dir(self.root / "maccs")
        self.scaler_path = self.root / "physchem_scaler.json"

        self.use_ecfp = bool(use_ecfp)
        self.use_physchem = bool(use_physchem)
        self.use_maccs = bool(use_maccs)
        self.ecfp_bits = int(ecfp_bits)
        self.ecfp_radius = int(ecfp_radius)
        self.maccs_dim = 166
        self.physchem_dim = len(PHYS_CHEM_DESCRIPTOR_NAMES)
        self.physchem_mean: np.ndarray | None = None
        self.physchem_std: np.ndarray | None = None

        if physchem_stats is None and self.scaler_path.exists():
            physchem_stats = self.load_physchem_stats()
        if physchem_stats is not None:
            self.set_physchem_stats(physchem_stats)

        self.stats = {
            "requests": 0,
            "ecfp_hits": 0,
            "ecfp_misses": 0,
            "ecfp_saved": 0,
            "physchem_hits": 0,
            "physchem_misses": 0,
            "physchem_saved": 0,
            "maccs_hits": 0,
            "maccs_misses": 0,
            "maccs_saved": 0,
            "skipped_invalid": 0,
            "load_failures": 0,
        }

    @property
    def enabled(self) -> bool:
        return bool(self.use_ecfp or self.use_physchem or self.use_maccs)

    @property
    def feature_dim(self) -> int:
        dim = 0
        if self.use_ecfp:
            dim += self.ecfp_bits
        if self.use_physchem:
            dim += self.physchem_dim
        if self.use_maccs:
            dim += self.maccs_dim
        return int(dim)

    @staticmethod
    def _hash_smiles(canonical_smiles: str) -> str:
        return hashlib.sha256(canonical_smiles.encode("utf-8")).hexdigest()[:24]

    def _path_for(self, kind: str, canonical_smiles: str) -> Path:
        digest = self._hash_smiles(canonical_smiles)
        if kind == "ecfp":
            return self.ecfp_dir / f"{digest}.npy"
        if kind == "physchem":
            return self.physchem_dir / f"{digest}.npy"
        if kind == "maccs":
            return self.maccs_dir / f"{digest}.npy"
        raise ValueError(f"Unsupported feature kind: {kind}")

    def _inc(self, key: str) -> None:
        self.stats[key] = int(self.stats.get(key, 0)) + 1

    def set_physchem_stats(self, stats: dict[str, object], persist: bool = False) -> None:
        mean = np.asarray(stats.get("mean", []), dtype=np.float32).reshape(-1)
        std = np.asarray(stats.get("std", []), dtype=np.float32).reshape(-1)
        if mean.size == 0 or std.size == 0 or mean.size != std.size:
            raise ValueError("physchem stats must contain same-length non-empty mean/std vectors")
        std = np.maximum(std, float(PHYSCHEM_STD_FLOOR))
        self.physchem_mean = mean
        self.physchem_std = std
        self.physchem_dim = int(mean.size)
        if persist:
            save_json({str(k): v for k, v in stats.items()}, self.scaler_path)

    def load_physchem_stats(self) -> Optional[dict[str, object]]:
        if not self.scaler_path.exists():
            return None
        try:
            loaded = load_json(self.scaler_path)
            if isinstance(loaded, dict):
                return loaded
        except Exception as e:
            LOGGER.warning("Failed loading physchem scaler stats %s: %s", self.scaler_path, e)
        return None

    def save_physchem_stats(self, stats: dict[str, object]) -> None:
        save_json({str(k): v for k, v in stats.items()}, self.scaler_path)

    def _get_or_create_raw(self, canonical_smiles: str, kind: str) -> np.ndarray:
        path = self._path_for(kind=kind, canonical_smiles=canonical_smiles)
        if path.exists():
            self._inc(f"{kind}_hits")
            try:
                return np.load(path, allow_pickle=False)
            except Exception as e:
                self._inc("load_failures")
                LOGGER.warning("Feature cache load failed for %s (%s): %s", canonical_smiles, path, e)

        self._inc(f"{kind}_misses")
        if kind == "ecfp":
            arr = smiles_to_ecfp(canonical_smiles, n_bits=self.ecfp_bits, radius=self.ecfp_radius).astype(np.uint8)
        elif kind == "physchem":
            arr = smiles_to_physchem(canonical_smiles).astype(np.float32)
        elif kind == "maccs":
            arr = smiles_to_maccs(canonical_smiles).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported feature kind: {kind}")

        np.save(path, arr, allow_pickle=False)
        self._inc(f"{kind}_saved")
        return arr

    def _standardize_physchem(self, raw_physchem: np.ndarray) -> np.ndarray:
        x = raw_physchem.astype(np.float32, copy=False).reshape(-1)
        if self.physchem_mean is None or self.physchem_std is None:
            return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size != int(self.physchem_mean.size):
            raise ValueError(
                f"physchem feature dim mismatch: got {x.size}, expected {int(self.physchem_mean.size)} from scaler stats"
            )
        z = (x - self.physchem_mean) / self.physchem_std
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    def get_or_create(self, smiles: str) -> Optional[np.ndarray]:
        self._inc("requests")
        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            self._inc("skipped_invalid")
            return None

        if not self.enabled:
            return np.empty((0,), dtype=np.float32)

        parts: list[np.ndarray] = []
        if self.use_ecfp:
            ecfp = self._get_or_create_raw(canonical, kind="ecfp").astype(np.float32, copy=False)
            parts.append(ecfp)
        if self.use_physchem:
            physchem_raw = self._get_or_create_raw(canonical, kind="physchem")
            physchem = self._standardize_physchem(physchem_raw)
            parts.append(physchem)
        if self.use_maccs:
            maccs = self._get_or_create_raw(canonical, kind="maccs").astype(np.float32, copy=False)
            parts.append(maccs)

        if not parts:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    def build(self, smiles_list: Iterable[str], show_progress: bool = True) -> dict[str, int]:
        try:
            from tqdm.auto import tqdm
        except Exception:  # pragma: no cover
            tqdm = None

        iterable = smiles_list
        if show_progress and tqdm is not None:
            smiles_list = list(smiles_list)
            iterable = tqdm(smiles_list, desc="Caching drug features")
        for smiles in iterable:
            self.get_or_create(str(smiles))
        return {k: int(v) for k, v in self.stats.items()}
