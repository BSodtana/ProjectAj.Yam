# Interpretable GAT for Multi-Typed Drug-Drug Interaction (DrugBank / TDC)

Colab-first, reproducible PyTorch Geometric project for multi-class prediction of 86 DrugBank DDI interaction types with:

- Siamese GAT encoder (shared weights)
- Pairwise interaction head (86-way softmax)
- Attention-based interpretability
- Faithfulness checks (deletion/insertion)
- PyG `GNNExplainer` baseline (single-graph contribution explanation with the other drug fixed)
- Realistic split support with default cold-drug protocol (test drugs unseen in training)

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

Class-imbalance objective options (CLIPPED weighted CE):

```bash
# Baseline objective (unweighted CE)
python scripts/train.py --data_dir ./data --output_dir ./outputs

# Inverse-sqrt frequency weights + clipping
python scripts/train.py --data_dir ./data --output_dir ./outputs \
  --use_class_weights \
  --class_weight_method inv_sqrt \
  --class_weight_clip_min 0.25 \
  --class_weight_clip_max 20.0

# Class-balanced effective-number weights (Cui et al., CVPR 2019) + clipping
python scripts/train.py --data_dir ./data --output_dir ./outputs \
  --use_class_weights \
  --class_weight_method effective_num \
  --class_weight_beta 0.9999 \
  --class_weight_clip_min 0.25 \
  --class_weight_clip_max 20.0
```

Optional SMILES-derived feature branches (cached by canonical SMILES hash):

```bash
# ECFP4 (2048-bit) + standardized RDKit physchem descriptors
python scripts/train.py --data_dir ./data --output_dir ./outputs \
  --use_ecfp_features \
  --use_physchem_features \
  --ecfp_bits 2048 \
  --ecfp_radius 2

# Add optional MACCS keys (166-bit)
python scripts/train.py --data_dir ./data --output_dir ./outputs \
  --use_ecfp_features \
  --use_physchem_features \
  --use_maccs_features
```

Feature cache artifacts:
- `outputs/drug_features/ecfp/<hash>.npy`
- `outputs/drug_features/physchem/<hash>.npy`
- `outputs/drug_features/maccs/<hash>.npy`
- `outputs/drug_features/physchem_scaler.json` (train-split mean/std only)

Implementation note:
- Uses PyTorch `cross_entropy` / `CrossEntropyLoss` semantics with `weight=` and `label_smoothing=`.
- With `reduction="mean"`, weighted CE follows PyTorch’s weight-normalized mean behavior.
- Effective-number weighting follows Cui et al. (2019), then mean-normalization and clipping.
- Future option (not implemented): logit adjustment (Menon et al., 2020).

Expected checkpoint:
- `outputs/checkpoints/best.pt`

### 3) Evaluate

```bash
python scripts/evaluate.py --data_dir ./data --output_dir ./outputs
```

Prints:
- Accuracy
- Macro-F1 / Micro-F1
- Cohen's kappa
- Macro ROC-AUC (OvR)
- Macro PR-AUC (OvR)
- ECE / Brier score
- Objective loss (training-consistent CE with configured class weights/label smoothing)
- NLL loss (plain CE, no class weights / no label smoothing)

Optional temperature scaling calibration report:

```bash
python scripts/evaluate.py --data_dir ./data --output_dir ./outputs --calibrate_temperature
```

### 4) Explain examples

```bash
python scripts/explain_examples.py --data_dir ./data --output_dir ./outputs --n 200
```

Artifacts in:
- `outputs/explanations/<pair_id>/attention_A.png`
- `outputs/explanations/<pair_id>/attention_B.png`
- `outputs/explanations/<pair_id>/faithfulness_A.png`
- `outputs/explanations/<pair_id>/faithfulness_B.png`
- `outputs/explanations/<pair_id>/gnnexplainer_A.png`
- `outputs/explanations/<pair_id>/gnnexplainer_B.png`
- `outputs/explanations/explain_metrics_per_pair.csv`
- `outputs/explanations/explain_metrics_summary.json`

### 5) Run 3-seed ablation table

```bash
python scripts/run_ablations.py --data_dir ./data --output_dir ./outputs --seeds 42,43,44 --limit 50000 --epochs 10
```

Feature-isolation ablation suite (same objective across variants):

```bash
python scripts/run_ablations.py --data_dir ./data --output_dir ./outputs \
  --ablation_suite feature \
  --seeds 42,43,44 \
  --baseline_use_class_weights \
  --baseline_class_weight_method inv_sqrt \
  --calibrate_temperature
```

Outputs:
- `outputs/ablations/ablation_results_raw.csv`
- `outputs/ablations/ablation_results_mean_std.csv`
- `outputs/ablations/ablation_table.md`

### 6) Run diagnostics mode

```bash
python scripts/diagnose.py --data_dir ./data --output_dir ./outputs --limit 2000 --seed 42

# Weighted objective sanity run
python scripts/diagnose.py --data_dir ./data --output_dir ./outputs --limit 2000 --seed 42 \
  --use_class_weights \
  --class_weight_method inv_sqrt \
  --class_weight_clip_max 20
```

Outputs:
- `outputs/diagnostics/class_counts.json`
- `outputs/diagnostics/class_weights.json`
- `outputs/diagnostics/labels_check.json`
- `outputs/diagnostics/class_distribution.csv`
- `outputs/diagnostics/loss_sanity.json`
- `outputs/diagnostics/metrics_sanity.json`
- `outputs/diagnostics/calibration_sanity.json`
- `outputs/diagnostics/faithfulness_sanity.csv`
- `outputs/diagnostics/randomization_sanity.json`
- `outputs/diagnostics/diagnostic_summary.json`

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
- Drug feature caches (`ecfp`, `physchem`, `maccs`) by canonical SMILES hash
- Train-only physchem scaler persisted to `outputs/drug_features/physchem_scaler.json`
- Version-pinned `requirements.txt`

## Limitations

- TDC column schemas can vary across versions; loader includes defensive normalization heuristics.
- PyG explainability APIs differ across versions; this project targets `torch_geometric>=2.5`.
- Full training on the entire DrugBank DDI dataset can be slow on CPU-only environments.

## Licensing Note

This repository contains original project code. Dataset usage and redistribution are subject to TDC / DrugBank terms. RDKit, PyTorch, PyG, and other dependencies retain their respective licenses.
