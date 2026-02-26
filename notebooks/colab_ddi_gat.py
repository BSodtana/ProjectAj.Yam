# %% [markdown]
# # Interpretable Siamese GAT for DrugBank Multi-Typed DDI (TDC)
#
# Colab-first runnable notebook script (`# %%` cells).
# Default mode runs a smoke test on a subset for speed, and can be toggled to full training.

# %%
import os
import sys
import subprocess
from pathlib import Path
from IPython.display import display


def run(cmd, check=True):
    print("+", " ".join(map(str, cmd)))
    return subprocess.run(cmd, check=check)


# %%
# Install core dependencies (Colab-friendly).
run([sys.executable, "-m", "pip", "install", "-q", "torch", "numpy", "pandas", "scikit-learn", "matplotlib", "tqdm"])
run([sys.executable, "-m", "pip", "install", "-q", "tdc", "rdkit"])
pyg_ok = subprocess.run([sys.executable, "-c", "import torch_geometric"], capture_output=True)
if pyg_ok.returncode != 0:
    run([sys.executable, "-m", "pip", "install", "-q", "torch_geometric"])
    pyg_ok = subprocess.run([sys.executable, "-c", "import torch_geometric"], capture_output=True)
    if pyg_ok.returncode != 0:
        import torch

        torch_version = torch.__version__.split("+")[0]
        cuda_tag = "cpu"
        if torch.version.cuda:
            cuda_tag = "cu" + torch.version.cuda.replace(".", "")
        whl_url = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_tag}.html"
        print("PyG fallback wheel URL:", whl_url)
        run([sys.executable, "-m", "pip", "install", "-q", "pyg_lib", "torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv", "-f", whl_url])
        run([sys.executable, "-m", "pip", "install", "-q", "torch_geometric"])


# %%
# Adjust this path if you mount Drive or clone the repo elsewhere in Colab.
PROJECT_ROOT = Path("/content/ProjectAj.Yam")
if not PROJECT_ROOT.exists():
    # Fallback: assume notebook file is executed inside the repo.
    PROJECT_ROOT = Path.cwd()
os.chdir(PROJECT_ROOT)
print("PROJECT_ROOT =", PROJECT_ROOT)

DATA_DIR = Path("/content/data")
OUTPUT_DIR = Path("/content/outputs")
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Reproducible subset by default (fast smoke test).
RUN_FULL_DATASET = False
TRAIN_LIMIT = None if RUN_FULL_DATASET else 10_000
EPOCHS = 1 if not RUN_FULL_DATASET else 10
BATCH_SIZE = 64
SEED = 42
DEVICE = "auto"


# %%
# Verify dataset loading via TDC API and print normalized split stats.
from ddigat.data.tdc_ddi import load_tdc_drugbank_ddi

train_df, valid_df, test_df, label_map = load_tdc_drugbank_ddi(str(DATA_DIR), output_dir=str(OUTPUT_DIR))
print(
    {
        "train": len(train_df),
        "valid": len(valid_df),
        "test": len(test_df),
        "num_labels": len(label_map),
        "columns": list(train_df.columns),
    }
)
display(train_df.head())


# %%
# Build graph cache (prepare_data CLI).
prep_cmd = [
    sys.executable,
    "scripts/prepare_data.py",
    "--data_dir",
    str(DATA_DIR),
    "--output_dir",
    str(OUTPUT_DIR),
    "--seed",
    str(SEED),
]
if TRAIN_LIMIT is not None:
    prep_cmd += ["--limit", str(TRAIN_LIMIT)]
run(prep_cmd)


# %%
# Train model (smoke test by default: 1 epoch on subset).
train_cmd = [
    sys.executable,
    "scripts/train.py",
    "--data_dir",
    str(DATA_DIR),
    "--output_dir",
    str(OUTPUT_DIR),
    "--seed",
    str(SEED),
    "--batch_size",
    str(BATCH_SIZE),
    "--epochs",
    str(EPOCHS),
    "--device",
    DEVICE,
]
if TRAIN_LIMIT is not None:
    train_cmd += ["--limit", str(TRAIN_LIMIT)]
run(train_cmd)


# %%
# Evaluate validation/test-quality metrics (macro ROC-AUC and macro PR-AUC, OvR).
eval_cmd = [
    sys.executable,
    "scripts/evaluate.py",
    "--data_dir",
    str(DATA_DIR),
    "--output_dir",
    str(OUTPUT_DIR),
    "--checkpoint",
    str(OUTPUT_DIR / "checkpoints" / "best.pt"),
    "--batch_size",
    str(BATCH_SIZE),
    "--device",
    DEVICE,
]
if TRAIN_LIMIT is not None:
    eval_cmd += ["--limit", str(max(1000, TRAIN_LIMIT // 4))]
run(eval_cmd)


# %%
# Generate explanation artifacts for 5 example pairs.
explain_cmd = [
    sys.executable,
    "scripts/explain_examples.py",
    "--data_dir",
    str(DATA_DIR),
    "--output_dir",
    str(OUTPUT_DIR),
    "--checkpoint",
    str(OUTPUT_DIR / "checkpoints" / "best.pt"),
    "--n",
    "5",
    "--device",
    DEVICE,
    "--gnnexplainer_epochs",
    "30",
]
if TRAIN_LIMIT is not None:
    explain_cmd += ["--limit", str(max(500, TRAIN_LIMIT // 4))]
run(explain_cmd)


# %%
# Inspect generated output structure.
for p in sorted(OUTPUT_DIR.rglob("*")):
    if p.is_file():
        print(p)
