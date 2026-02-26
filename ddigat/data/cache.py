from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Optional

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

