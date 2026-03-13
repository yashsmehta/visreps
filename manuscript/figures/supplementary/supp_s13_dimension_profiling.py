"""
Supplementary Figure S13: THINGS Semantic Dimension Profiling.

Shows horizontal bar chart of Spearman rho between per-concept advantage
(coarse minus 1000-way) and each of the 66 THINGS behavioral dimensions.
Displays top 25 dimensions by |difference|.

Adapted from experiments/things_visualizations/plot_dimension_alignment.py
and experiments/things_visualizations/characterize_discrepant.py.

Run from project root:
    python manuscript/figures/supplementary/supp_s13_dimension_profiling.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, false_discovery_control

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from manuscript.figures.fig_utils import setup_style
from experiments.things_visualizations.utils import load_data
from experiments.things_visualizations.per_row_rdm import per_row_correlations
from visreps.analysis.rsa import compute_rdm

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "supp_s13_dimension_profiling.png")

TOP_N = 25
COLOR_COARSE = "#2d7f2d"  # green for coarse advantage
COLOR_FINE = "#b2182b"    # red for 1000-class advantage


def compute_per_concept_diff(data):
    """Compute per-concept RDM row correlation difference (CLIP 4-class - 1K)."""
    rdm_behav = compute_rdm(torch.tensor(data["embeddings"], dtype=torch.float32)).numpy()
    rdm_clip4 = compute_rdm(torch.tensor(data["clip4_acts"], dtype=torch.float32)).numpy()
    rdm_1k = compute_rdm(torch.tensor(data["thousand_acts"], dtype=torch.float32)).numpy()

    corr_clip4 = per_row_correlations(rdm_clip4, rdm_behav)
    corr_1k = per_row_correlations(rdm_1k, rdm_behav)
    return corr_clip4 - corr_1k


def dimension_profiling(diff, embeddings, dimension_labels):
    """Correlate per-concept advantage with each of 66 behavioral dimensions.

    Returns (rhos, pvals, rejected) arrays of shape (n_dims,).
    """
    n_dims = len(dimension_labels)
    rhos = np.zeros(n_dims)
    pvals = np.zeros(n_dims)

    for d in range(n_dims):
        rho, p = spearmanr(diff, embeddings[:, d])
        rhos[d] = rho if not np.isnan(rho) else 0.0
        pvals[d] = p if not np.isnan(p) else 1.0

    # FDR correction (Benjamini-Hochberg)
    rejected = pvals <= false_discovery_control(pvals, method="bh")
    return rhos, pvals, rejected


def shorten_label(label):
    """Shorten long dimension labels for display."""
    remap = {
        "music-related / hearing-related / hobby-related / loud": "Music / auditory",
        "transportation/movement-related": "Transportation / movement",
        "electronics / technology": "Electronics / technology",
        "hobby-related / game-related / playing-related": "Games / hobbies",
        "paper-related / flat": "Paper / flat",
        "metallic/artificial": "Metallic / artificial",
        "house-related/furnishing-related": "Home / furnishing",
        "valuable/precious": "Valuable / precious",
        "circular / round": "Circular / round",
        "colorful / playful": "Colorful / playful",
        "wood-related/brown": "Wood / brown",
        "body-/people-related": "Body / people",
        "fluid-related / drink-related": "Fluid / drink",
        "body part-related": "Body parts",
    }
    if label in remap:
        return remap[label]
    # General cleanup
    return label.replace("-related", "").strip().capitalize()


def main():
    setup_style()

    print("Loading THINGS visualization data...")
    data = load_data()
    labels = [str(l) for l in data["dimension_labels"]]
    n_dims = len(labels)

    print(f"Computing per-concept RDM row correlations ({len(data['concept_names'])} concepts)...")
    diff = compute_per_concept_diff(data)
    print(f"  CLIP-4 wins on {(diff > 0).sum()}/{len(diff)} concepts")

    print(f"Computing dimension profiling ({n_dims} dimensions)...")
    rhos, pvals, rejected = dimension_profiling(diff, data["embeddings"], labels)
    n_sig = rejected.sum()
    print(f"  {n_sig}/{n_dims} significant (FDR q < 0.05)")

    # Sort by absolute correlation, take top N
    order = np.argsort(np.abs(rhos))[::-1][:TOP_N]
    # Within top N, sort by actual rho value for visual clarity
    order = order[np.argsort(rhos[order])]

    # Build figure
    fig, ax = plt.subplots(figsize=(6.0, 0.32 * TOP_N + 0.8))
    y = np.arange(TOP_N)
    colors = [COLOR_COARSE if rhos[i] > 0 else COLOR_FINE for i in order]

    ax.barh(y, rhos[order], color=colors, edgecolor="none", alpha=0.85, zorder=3)

    # Y-tick labels with significance markers and shortened names
    tick_labels = []
    for i in order:
        marker = " *" if rejected[i] else ""
        tick_labels.append(f"{shorten_label(labels[i])}{marker}")

    ax.set_yticks(y)
    ax.set_yticklabels(tick_labels, fontsize=7.5)
    ax.set_xlabel(r"Spearman $\rho$ (concept advantage vs dimension loading)", fontsize=8.5)
    ax.axvline(0, color="black", lw=0.8, zorder=4)
    ax.grid(axis="x", alpha=0.25, zorder=0)

    import seaborn as sns
    sns.despine(ax=ax, right=True, top=True, offset=4)

    # Annotation
    ax.text(0.98, 0.03,
            f"* FDR-corrected $p$ < 0.05 ({n_sig}/{n_dims} significant)\n"
            f"Green = high loading favors coarse model\n"
            f"Red = high loading favors 1000-class",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
            fontstyle="italic", color="#666666")

    plt.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")

    # Print top dimensions
    print(f"\nTop {TOP_N} dimensions by |rho|:")
    for i in np.argsort(np.abs(rhos))[::-1][:10]:
        sig = "*" if rejected[i] else ""
        print(f"  {labels[i]:<40} rho={rhos[i]:+.4f}  p={pvals[i]:.2e} {sig}")


if __name__ == "__main__":
    main()
