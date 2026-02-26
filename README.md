# Interpretable GAT for Multi-Typed Drug-Drug Interaction (DrugBank / TDC)

Colab-first, reproducible PyTorch Geometric project for multi-class prediction of 86 DrugBank DDI interaction types with:

- Siamese GAT encoder (shared weights)
- Pairwise interaction head (86-way softmax)
- Attention-based interpretability
- Faithfulness checks (deletion/insertion)
- PyG `GNNExplainer` baseline (single-graph contribution explanation with the other drug fixed)

## Quick Start (CLI)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
# PyTDC 0.4.1 depends on `rdkit-pypi` (unavailable on macOS ARM); install without deps:
python -m pip install --no-deps PyTDC==0.4.1
```

Optional PyG wheels fallback (if your environment needs compiled deps):

```bash
python - <<'PY'
import torch
print(torch.__version__)
PY
# Then install from https://data.pyg.org/whl/ matching your torch/CUDA build if needed.
```

### 1) Prepare data + graph cache (smoke test)

```bash
python scripts/prepare_data.py --data_dir ./data --output_dir ./outputs --limit 2000
```

### 2) Train (smoke test)

```bash
python scripts/train.py --data_dir ./data --output_dir ./outputs --limit 2000 --epochs 1
```

Expected checkpoint:
- `outputs/checkpoints/best.pt`

### 3) Evaluate

```bash
python scripts/evaluate.py --data_dir ./data --output_dir ./outputs
```

Prints:
- Macro ROC-AUC (OvR)
- Macro PR-AUC (OvR)

### 4) Explain examples

```bash
python scripts/explain_examples.py --data_dir ./data --output_dir ./outputs --n 2
```

Artifacts in:
- `outputs/explanations/<pair_id>/attention_A.png`
- `outputs/explanations/<pair_id>/attention_B.png`
- `outputs/explanations/<pair_id>/faithfulness_A.png`
- `outputs/explanations/<pair_id>/faithfulness_B.png`
- `outputs/explanations/<pair_id>/gnnexplainer_A.png`
- `outputs/explanations/<pair_id>/gnnexplainer_B.png`

## Colab

Use `notebooks/colab_ddi_gat.py` as a Colab cell-style notebook (`# %%` cells). It:

1. Installs dependencies
2. Loads TDC DrugBank DDI
3. Builds graph cache
4. Trains a smoke-test model by default (subset)
5. Evaluates ROC-AUC / PR-AUC
6. Generates explanation artifacts for 5 examples
7. Saves everything under `/content/outputs/`

## Reproducibility

- Fixed seeds (`random`, `numpy`, `torch`, `cuda`)
- Deterministic-ish backend settings where possible
- Saved TDC splits to `outputs/splits/*.csv`
- Graph featurization cache by canonical SMILES hash
- Version-pinned `requirements.txt`

## Limitations

- TDC column schemas can vary across versions; loader includes defensive normalization heuristics.
- PyG explainability APIs differ across versions; this project targets `torch_geometric>=2.5`.
- Full training on the entire DrugBank DDI dataset can be slow on CPU-only environments.

## Licensing Note

This repository contains original project code. Dataset usage and redistribution are subject to TDC / DrugBank terms. RDKit, PyTorch, PyG, and other dependencies retain their respective licenses.
