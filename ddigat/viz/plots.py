from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ddigat.utils.io import ensure_dir


def plot_faithfulness_curves(
    k_list: Sequence[int],
    deletion_probs: Sequence[float],
    insertion_probs: Sequence[float],
    out_path: str | Path,
    title: str,
) -> None:
    ensure_dir(Path(out_path).parent)
    k = np.asarray(k_list)
    d = np.asarray(deletion_probs)
    ins = np.asarray(insertion_probs)

    plt.figure(figsize=(6, 4))
    plt.plot(k, d, marker="o", label="Deletion")
    plt.plot(k, ins, marker="o", label="Insertion")
    plt.xlabel("Top-k nodes perturbed/restored")
    plt.ylabel("Target class probability")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_node_scores(
    node_scores: Sequence[float],
    out_path: str | Path,
    title: str,
    max_nodes: int = 50,
) -> None:
    ensure_dir(Path(out_path).parent)
    scores = np.asarray(node_scores, dtype=float)
    n = min(len(scores), max_nodes)
    idx = np.arange(n)
    vals = scores[:n]
    plt.figure(figsize=(7, 3))
    plt.bar(idx, vals)
    plt.xlabel("Node index")
    plt.ylabel("Score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

