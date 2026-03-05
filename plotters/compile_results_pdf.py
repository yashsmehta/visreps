"""Compile all coarseness bar figures into a PDF catalog with methodology captions.

Usage:
    python plotters/compile_results_pdf.py
"""

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parent

FIGURES_DIR = {
    "nsd": ROOT / "nsd" / "figures",
    "nsd_synthetic": ROOT / "nsd_synthetic" / "figures",
    "tvsd": ROOT / "tvsd" / "figures",
    "things": ROOT / "things" / "figures",
}

DATASET_TITLES = {
    "nsd": "Natural Scenes Dataset (NSD) — Human fMRI",
    "nsd_synthetic": "NSD-Synthetic — Synthetic fMRI",
    "tvsd": "TVSD — Macaque Electrophysiology",
    "things": "THINGS — Human Behavioral Similarity",
}

ANALYSIS_TITLES = {
    "rsa": "Representational Similarity Analysis (RSA)",
    "encoding_score": "Encoding Score (Voxelwise Prediction)",
}

PCA_MODEL_NAMES = {
    "alexnet": "AlexNet",
    "clip": "CLIP",
    "dino": "DINO",
    "vit": "ViT",
}

# Wrap width for methodology text at 10pt font on letter page with margins
WRAP_WIDTH = 105

# Page dimensions (US Letter, portrait)
PAGE_W = 8.5
PAGE_H = 11.0
MARGIN = 0.65


def parse_figure_info(filepath, dataset):
    """Extract metadata from a coarseness_bars figure filename."""
    name = filepath.stem
    if not name.startswith("coarseness_bars_"):
        return None

    remainder = name[len("coarseness_bars_"):]

    if "_encoding" in remainder:
        analysis = "encoding_score"
        remainder = remainder.replace("_encoding", "")
    else:
        analysis = "rsa"

    if "_finegrained" in remainder:
        layout = "finegrained"
        remainder = remainder.replace("_finegrained", "")
    else:
        layout = "default"

    pca_model = remainder

    if dataset in ("nsd", "nsd_synthetic"):
        if layout == "finegrained":
            regions = "V1, V2, V3, hV4, FFA, PPA (fine-grained ROIs)"
        else:
            regions = "Early Visual Stream, Ventral Visual Stream"
    elif dataset == "tvsd":
        regions = "V1, V4, IT"
    else:
        regions = "N/A (concept-level behavioral data)"

    if dataset in ("nsd", "nsd_synthetic"):
        subjects = "8 human subjects"
    elif dataset == "tvsd":
        subjects = "2 macaque monkeys"
    else:
        subjects = None

    return {
        "filepath": filepath,
        "dataset": dataset,
        "analysis": analysis,
        "pca_model": pca_model,
        "pca_display": PCA_MODEL_NAMES.get(pca_model, pca_model),
        "regions": regions,
        "subjects": subjects,
        "layout": layout,
    }


def _wrap(text):
    """Wrap paragraphs to WRAP_WIDTH, preserving blank-line separators."""
    lines = text.split("\n")
    wrapped = []
    for line in lines:
        if line.strip() == "":
            wrapped.append("")
        else:
            wrapped.append(textwrap.fill(line, width=WRAP_WIDTH))
    return "\n".join(wrapped)


def get_methodology_text(info):
    """Generate the methodology caption for a figure."""
    dataset = info["dataset"]
    analysis = info["analysis"]
    pca_display = info["pca_display"]
    parts = []

    # -- Header --
    header_items = [f"PCA Label Model: {pca_display}", f"Regions: {info['regions']}"]
    if info["subjects"]:
        header_items.append(f"Subjects: {info['subjects']}")
    header_line = "  |  ".join(header_items)
    header_line += "\nSeeds: 3 (seeds 1, 2, 3)  |  Training Epoch: 20"
    parts.append(header_line)

    # -- Train/test split --
    if dataset in ("nsd", "nsd_synthetic"):
        split = (
            "Train/Test Split: ~9,000 unique train stimuli per subject; "
            "1,000 shared test stimuli (intersection across all subjects)."
        )
    elif dataset == "tvsd":
        split = (
            "Train/Test Split: ~22,000 train stimuli; 100 test stimuli per subject "
            "(pre-split). Stimuli are THINGS object images."
        )
    else:
        split = (
            "Train/Test Split: ~1,854 concepts total. Fixed 80/20 concept-level split (seed=42): "
            "~370 concepts for layer selection, ~1,484 concepts for evaluation. "
            "Model activations extracted for all images then averaged per concept."
        )
    parts.append(split)

    # -- Best layer selection --
    if analysis == "rsa":
        if dataset == "things":
            layer_sel = (
                "Best Layer Selection: All ~370 selection-set concepts used (no subsampling cap). "
                "Activations projected via Sparse Random Projection (SRP, k=4,096). "
                "Pearson dissimilarity RDMs (1 - r) built per layer. "
                "Layer with highest RDM correlation selected."
            )
        else:
            layer_sel = (
                "Best Layer Selection: 1,000 train stimuli subsampled (seed=42) per subject per region. "
                "Activations projected via Sparse Random Projection (SRP, k=4,096). "
                "Pearson dissimilarity RDMs (1 - r) built per layer. "
                "Layer with highest RDM correlation selected independently per (subject, region)."
            )
    else:
        layer_sel = (
            "Best Layer Selection: 80/20 fit/validation split of train data (seed=42). "
            "Both activations (X) and neural responses (Y) z-normalized using fit-only statistics "
            "(mean, std + 1e-8). "
            "RidgeCV with 5-fold CV, 20 alpha candidates (1e-10 to 1e10), fit_intercept=False. "
            "Layer with highest mean Pearson r across voxels on validation set selected. "
            "SRP (k=4,096) used throughout."
        )
    parts.append(layer_sel)

    # -- Test scoring --
    if analysis == "rsa":
        if dataset == "things":
            scoring = (
                "Test Scoring: Best layer re-extracted WITHOUT SRP for exact full-resolution activations. "
                "Activations concept-averaged over the ~1,484 evaluation-set images. "
                "Pearson dissimilarity RDMs built from uncompressed, concept-averaged activations."
            )
        else:
            scoring = (
                "Test Scoring: Best layer re-extracted WITHOUT SRP for exact full-resolution activations. "
                "Pearson dissimilarity RDMs built from uncompressed test activations."
            )
    else:
        scoring = (
            "Test Scoring: Best layer refit on full train set (z-normalized with full-train statistics). "
            "Predictions on test set. Score = mean Pearson r across all voxels. "
            "SRP (k=4,096) used throughout (no re-extraction)."
        )
    parts.append(scoring)

    # -- Bootstrap --
    if analysis == "rsa":
        bootstrap = (
            "Bootstrap 95% CIs: 1,000 iterations, 90% test stimulus subsample per iteration (seed=42). "
            "Both model and neural RDMs subindexed to the same subset. "
            "Distributions element-wise averaged across all runs (seeds x subjects). "
            "CI bounds at 2.5th and 97.5th percentiles."
        )
    else:
        bootstrap = (
            "Bootstrap 95% CIs: 1,000 iterations on cached predictions (no refitting). "
            "90% test subsample per iteration (seed=42). "
            "Mean Pearson r across voxels recomputed per subsample. "
            "Distributions element-wise averaged across all runs (seeds x subjects). "
            "CI bounds at 2.5th and 97.5th percentiles."
        )
    parts.append(bootstrap)

    # -- Aggregation --
    if info["subjects"]:
        agg = (
            "Point Estimate: Mean score across all seeds and subjects. "
            "SEM fallback (per-seed means, 1.96 x SEM) when bootstrap CIs unavailable."
        )
    else:
        agg = (
            "Point Estimate: Mean score across all seeds. "
            "SEM fallback (per-seed means, 1.96 x SEM) when bootstrap CIs unavailable."
        )
    parts.append(agg)

    return _wrap("\n\n".join(parts))


def collect_figures():
    """Collect all coarseness_bars figures organized by (dataset, analysis)."""
    catalog = {}
    for dataset, fig_dir in FIGURES_DIR.items():
        if not fig_dir.exists():
            continue
        for png_file in sorted(fig_dir.glob("coarseness_bars_*.png")):
            info = parse_figure_info(png_file, dataset)
            if info is None:
                continue
            key = (dataset, info["analysis"])
            catalog.setdefault(key, []).append(info)
    return catalog


def add_figure_page(pdf, info, section_header=None):
    """Add a page with an optional section header, the figure, and methodology text."""
    img = mpimg.imread(str(info["filepath"]))
    h, w = img.shape[:2]
    img_aspect = h / w
    methodology = get_methodology_text(info)

    usable_w = PAGE_W - 2 * MARGIN
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))

    # Track vertical position (in figure fraction, from top)
    y_top = 1.0 - MARGIN / PAGE_H

    # -- Optional section header --
    if section_header:
        fig.text(
            MARGIN / PAGE_W, y_top, section_header,
            fontsize=14, fontweight="bold", va="top", ha="left",
        )
        y_top -= 0.4 / PAGE_H
        # Thin separator line
        fig.add_artist(plt.Line2D(
            [MARGIN / PAGE_W, 1 - MARGIN / PAGE_W],
            [y_top, y_top],
            transform=fig.transFigure, color="#bbbbbb", linewidth=0.7,
        ))
        y_top -= 0.15 / PAGE_H

    # -- Image --
    # Scale image to fill page width; cap height so text has room
    img_h_inches = usable_w * img_aspect
    max_img_h = 5.2 if section_header else 5.8
    img_h_inches = min(img_h_inches, max_img_h)
    img_h_frac = img_h_inches / PAGE_H

    img_bottom = y_top - img_h_frac
    ax_img = fig.add_axes([MARGIN / PAGE_W, img_bottom, usable_w / PAGE_W, img_h_frac])
    ax_img.imshow(img)
    ax_img.axis("off")

    y_top = img_bottom - 0.2 / PAGE_H  # small gap below image

    # -- Methodology text --
    text_bottom = MARGIN / PAGE_H
    text_h = y_top - text_bottom

    if text_h > 0.02:
        ax_text = fig.add_axes([MARGIN / PAGE_W, text_bottom, usable_w / PAGE_W, text_h])
        ax_text.axis("off")
        ax_text.text(
            0, 1, methodology,
            transform=ax_text.transAxes,
            fontsize=10, va="top", ha="left",
            family="sans-serif", linespacing=1.4,
        )

    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def compile_pdf(output_path):
    """Compile all coarseness bar figures into a single PDF."""
    catalog = collect_figures()
    dataset_order = ["nsd", "nsd_synthetic", "tvsd", "things"]
    analysis_order = ["rsa", "encoding_score"]
    total = 0

    with PdfPages(output_path) as pdf:
        for dataset in dataset_order:
            for analysis in analysis_order:
                key = (dataset, analysis)
                if key not in catalog:
                    continue

                section_header = (
                    f"{DATASET_TITLES[dataset]}  —  {ANALYSIS_TITLES[analysis]}"
                )

                for i, info in enumerate(catalog[key]):
                    # Section header only on the first figure of each section
                    add_figure_page(
                        pdf, info,
                        section_header=section_header if i == 0 else None,
                    )
                    total += 1

    print(f"PDF compiled: {output_path}")
    print(f"Total figures: {total}")


if __name__ == "__main__":
    output = ROOT / "results_catalog.pdf"
    compile_pdf(output)
