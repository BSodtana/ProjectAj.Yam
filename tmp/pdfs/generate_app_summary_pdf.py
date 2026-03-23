from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas


OUT_PATH = Path("output/pdf/projectajyam_app_summary_one_page.pdf")


def draw_wrapped_text(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    font_name: str = "Helvetica",
    font_size: int = 10,
    leading: float = 12.0,
) -> float:
    lines = simpleSplit(text, font_name, font_size, width)
    c.setFont(font_name, font_size)
    for line in lines:
        c.drawString(x, y, line)
        y -= leading
    return y


def draw_section_heading(
    c: canvas.Canvas,
    heading: str,
    x: float,
    y: float,
    font_size: int = 14,
) -> float:
    c.setFont("Helvetica-Bold", font_size)
    c.drawString(x, y, heading)
    return y - 16


def draw_bullets(
    c: canvas.Canvas,
    bullets: list[str],
    x: float,
    y: float,
    width: float,
    font_size: int = 11,
    leading: float = 14.0,
) -> float:
    bullet_indent = 10
    text_width = width - bullet_indent
    for item in bullets:
        lines = simpleSplit(item, "Helvetica", font_size, text_width)
        c.setFont("Helvetica", font_size)
        if not lines:
            continue
        c.drawString(x, y, "-")
        c.drawString(x + bullet_indent, y, lines[0])
        y -= leading
        for line in lines[1:]:
            c.drawString(x + bullet_indent, y, line)
            y -= leading
    return y


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(OUT_PATH), pagesize=letter)
    page_w, page_h = letter
    margin_x = 44
    top_y = page_h - 44
    content_w = page_w - (2 * margin_x)
    y = top_y

    c.setTitle("ProjectAj.Yam App Summary (One Page)")

    c.setFont("Helvetica-Bold", 22)
    c.drawString(margin_x, y, "ProjectAj.Yam App Summary")
    y -= 24

    c.setFont("Helvetica", 9)
    c.drawString(margin_x, y, "Evidence source: README.md, scripts/, ddigat/ modules in this repository.")
    y -= 18

    y = draw_section_heading(c, "What It Is", margin_x, y)
    what_it_is = (
        "ProjectAj.Yam is a Colab-first, reproducible PyTorch Geometric app for multi-class "
        "Drug-Drug Interaction prediction on DrugBank/TDC. It trains a Siamese graph neural "
        "network and produces evaluation and interpretability artifacts."
    )
    y = draw_wrapped_text(c, what_it_is, margin_x, y, content_w, font_size=12, leading=15)
    y -= 4

    y = draw_section_heading(c, "Who It Is For", margin_x, y)
    who_for = (
        "Primary user/persona: Not found explicitly in repo. Inferred from scripts and dependencies: "
        "ML engineers or researchers running DDI experiments with Python, PyTorch, and PyG."
    )
    y = draw_wrapped_text(c, who_for, margin_x, y, content_w, font_size=12, leading=15)
    y -= 4

    y = draw_section_heading(c, "What It Does", margin_x, y)
    features = [
        "Loads and normalizes TDC DrugBank DDI data, then creates train/valid/test splits (cold_drug default).",
        "Converts SMILES into PyG molecular graphs and caches graphs by canonical SMILES hash.",
        "Trains a Siamese pair model (GAT default, with GCN/GIN options) for 86-way DDI classification.",
        "Supports optional ECFP, physchem, and MACCS feature branches, plus class weighting, DRW, and logit adjustment.",
        "Evaluates with accuracy, macro/micro-F1, kappa, macro ROC-AUC, macro PR-AUC, ECE, and Brier score.",
        "Generates explanation artifacts (attention, deletion/insertion faithfulness, optional GNNExplainer) and diagnostics/ablation outputs.",
    ]
    y = draw_bullets(c, features, margin_x, y, content_w, font_size=11, leading=14)
    y -= 2

    y = draw_section_heading(c, "How It Works (Architecture)", margin_x, y)
    architecture = [
        "CLI entry points: scripts/prepare_data.py, scripts/train.py, scripts/evaluate.py, scripts/explain_examples.py.",
        "Data layer: ddigat/data/tdc_ddi.py loads TDC and persists split files; ddigat/data/splits.py builds datasets/dataloaders.",
        "Cache layer: ddigat/data/cache.py stores processed graphs and optional drug features under outputs/.",
        "Model/training layer: ddigat/model/pair_model.py + ddigat/model/gnn_encoders.py + ddigat/train/loop.py.",
        "Data flow: TDC dataset -> normalized splits -> cached graphs/features -> DataLoader batches -> model training -> best.pt + evaluation_metrics.json + explanation images.",
    ]
    y = draw_bullets(c, architecture, margin_x, y, content_w, font_size=11, leading=14)
    y -= 2

    y = draw_section_heading(c, "How To Run (Minimal)", margin_x, y)
    run_steps = [
        "1) python -m venv .venv && source .venv/bin/activate",
        "2) python -m pip install --upgrade pip setuptools wheel && python -m pip install -r requirements.txt",
        "3) python -m pip install --no-deps PyTDC==0.4.1",
        "4) python scripts/prepare_data.py --data_dir ./data --output_dir ./outputs --limit 2000",
        "5) python scripts/train.py --data_dir ./data --output_dir ./outputs --limit 2000 --epochs 1",
        "6) python scripts/evaluate.py --data_dir ./data --output_dir ./outputs",
    ]
    y = draw_bullets(c, run_steps, margin_x, y, content_w, font_size=11, leading=14)

    min_allowed_y = 30
    if y < min_allowed_y:
        raise RuntimeError(f"Content overflow detected (y={y:.1f}). Reduce content to keep one page.")

    c.showPage()
    c.save()


if __name__ == "__main__":
    main()
