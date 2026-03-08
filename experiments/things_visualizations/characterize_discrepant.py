"""
Characterize discrepant concepts: where does CLIP 4-class beat 1K and vice versa?

Three complementary views of per-concept RDM differences:
  A. Image collages for top/bottom concepts
  B. Dendrograms comparing behavioral/model clustering for tail groups
  C. Semantic dimension profiling (66 THINGS dimensions)

Depends on per_row_rdm.py for the core per-row correlation computation.

Input:  experiments/things_visualizations/data/things_viz_data.npz
Output: experiments/things_visualizations/figures/collage_clip4_wins.png
        experiments/things_visualizations/figures/collage_1k_wins.png
        experiments/things_visualizations/figures/dendrograms_tails.png
        experiments/things_visualizations/figures/dimension_profiling.png

Run from project root:
  python experiments/things_visualizations/characterize_discrepant.py
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, false_discovery_control
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from PIL import Image, ImageDraw

from experiments.things_visualizations.utils import (
    load_data, load_thumbnail, get_font, save_fig, FIG_DIR,
    COLOR_CLIP4, COLOR_1K, COLOR_NEUTRAL,
)
from experiments.things_visualizations.per_row_rdm import per_row_correlations
from visreps.analysis.rsa import compute_rdm

TAIL_N = 20  # number of concepts per tail group
COLLAGE_COLS = 5
THUMB_SIZE = 128


# ── Shared data computation ──────────────────────────────────────────

def build_rdms(data):
    """Build Pearson dissimilarity RDMs for behavioral, CLIP-4, and 1K."""
    rdm_behav = compute_rdm(torch.tensor(data["embeddings"], dtype=torch.float32)).numpy()
    rdm_clip4 = compute_rdm(torch.tensor(data["clip4_acts"], dtype=torch.float32)).numpy()
    rdm_1k = compute_rdm(torch.tensor(data["thousand_acts"], dtype=torch.float32)).numpy()
    return rdm_behav, rdm_clip4, rdm_1k


def compute_differences(data):
    """Compute per-concept RDM row correlations and differences."""
    rdm_behav, rdm_clip4, rdm_1k = build_rdms(data)
    corr_clip4 = per_row_correlations(rdm_clip4, rdm_behav)
    corr_1k = per_row_correlations(rdm_1k, rdm_behav)
    return corr_clip4 - corr_1k


def get_tail_indices(diff):
    """Return (top_idx, bot_idx) for CLIP-4 wins and 1K wins."""
    order = np.argsort(diff)
    return order[-TAIL_N:][::-1], order[:TAIL_N]


# ── Part A: Image collages ───────────────────────────────────────────

def make_collage(concept_names, diffs, image_paths, title, output_path):
    """Create a grid of representative images for given concepts."""
    n = len(concept_names)
    rows = int(np.ceil(n / COLLAGE_COLS))
    label_height = 40
    cell_w = THUMB_SIZE
    cell_h = THUMB_SIZE + label_height
    title_height = 60
    margin = 10

    canvas_w = COLLAGE_COLS * cell_w + (COLLAGE_COLS + 1) * margin
    canvas_h = rows * cell_h + (rows + 1) * margin + title_height
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Title
    title_font = get_font(28)
    bbox = draw.textbbox((0, 0), title, font=title_font)
    tw = bbox[2] - bbox[0]
    draw.text(((canvas_w - tw) // 2, 15), title, fill=(30, 30, 30), font=title_font)

    label_font = get_font(12)
    for i, (name, d, path) in enumerate(zip(concept_names, diffs, image_paths)):
        row, col = divmod(i, COLLAGE_COLS)
        x = margin + col * (cell_w + margin)
        y = title_height + margin + row * (cell_h + margin)

        thumb = load_thumbnail(str(path), size=THUMB_SIZE, border_width=2,
                               border_color=(80, 80, 80))
        canvas.paste(thumb, (x, y))

        # Label: concept name + delta rho
        label = f"{name}\n({d:+.3f})"
        draw.text((x + 2, y + THUMB_SIZE + 2), label, fill=(40, 40, 40), font=label_font)

    canvas.save(output_path, dpi=(300, 300))
    print(f"Saved: {output_path}")


def plot_collages(concept_names, diff, image_paths, top_idx, bot_idx):
    """Create collages for top and bottom concepts."""
    for idx, title, suffix in [
        (top_idx, f"Top {TAIL_N} Concepts: CLIP 4-class > 1K", "collage_clip4_wins.png"),
        (bot_idx, f"Top {TAIL_N} Concepts: 1K > CLIP 4-class", "collage_1k_wins.png"),
    ]:
        names = [concept_names[i] for i in idx]
        diffs = diff[idx]
        paths = [image_paths[i] for i in idx]
        make_collage(names, diffs, paths, title, os.path.join(FIG_DIR, suffix))


# ── Part B: Dendrograms ──────────────────────────────────────────────

def plot_dendrograms(concept_names, data, top_idx, bot_idx):
    """Side-by-side dendrograms for top and bottom tail groups."""
    rdm_behav, rdm_clip4, rdm_1k = build_rdms(data)
    groups = [
        ("CLIP 4-class wins", top_idx, COLOR_CLIP4),
        ("1K wins", bot_idx, COLOR_1K),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    rdm_dict = {"Behavioral": rdm_behav, "CLIP 4-class": rdm_clip4, "1000-class": rdm_1k}

    for row, (group_title, idx, accent) in enumerate(groups):
        names = [concept_names[i] for i in idx]

        # Build mini-RDMs
        mini_rdms = {}
        for key, rdm in rdm_dict.items():
            mini_rdms[key] = rdm[np.ix_(idx, idx)]

        # Cluster on behavioral RDM
        behav_condensed = squareform(mini_rdms["Behavioral"], checks=False)
        Z = linkage(behav_condensed, method="average")

        for col, (rdm_title, mini_rdm) in enumerate(mini_rdms.items()):
            ax = axes[row, col]
            condensed = squareform(mini_rdm, checks=False)
            Z_panel = linkage(condensed, method="average") if rdm_title != "Behavioral" else Z

            dendrogram(
                Z_panel, ax=ax, labels=names, orientation="right",
                leaf_font_size=7, color_threshold=0,
                above_threshold_color=accent if rdm_title != "Behavioral" else COLOR_NEUTRAL,
            )

            panel_title = rdm_title
            if row == 0:
                ax.set_title(panel_title, fontsize=13, fontweight="bold", pad=8)
            if col == 0:
                ax.set_ylabel(group_title, fontsize=12, fontweight="bold",
                              color=accent, labelpad=10)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.set_xticks([])

    fig.suptitle("Hierarchical Clustering of Tail Concepts", fontsize=15,
                 fontweight="bold", y=0.98)
    save_fig(fig, os.path.join(FIG_DIR, "dendrograms_tails.png"))


# ── Part C: Semantic dimension profiling ─────────────────────────────

def plot_dimension_profiling(diff, embeddings, dimension_labels):
    """Correlate per-concept advantage with each of 66 behavioral dimensions."""
    labels = [str(l) for l in dimension_labels]
    n_dims = len(labels)

    # Spearman correlation of diff with each dimension's concept loadings
    rhos = np.zeros(n_dims)
    pvals = np.zeros(n_dims)
    for d in range(n_dims):
        rho, p = spearmanr(diff, embeddings[:, d])
        rhos[d] = rho if not np.isnan(rho) else 0.0
        pvals[d] = p if not np.isnan(p) else 1.0

    # FDR correction (Benjamini-Hochberg)
    rejected = pvals <= false_discovery_control(pvals, method="bh")

    # Sort by absolute correlation magnitude
    order = np.argsort(np.abs(rhos))[::-1]

    fig, ax = plt.subplots(figsize=(8, 0.34 * n_dims + 1.5))
    y = np.arange(n_dims)
    colors = [COLOR_CLIP4 if rhos[i] > 0 else COLOR_1K for i in order]
    ax.barh(y, rhos[order], color=colors, edgecolor="none", alpha=0.8, zorder=3)

    # Labels with significance markers
    tick_labels = []
    for i in order:
        marker = " *" if rejected[i] else ""
        tick_labels.append(f"{labels[i]}{marker}")

    ax.set_yticks(y)
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_xlabel("Spearman $\\rho$ (concept advantage vs dimension loading)", fontsize=10)
    ax.set_title("Semantic Dimension Profiling of Per-Concept Advantage\n"
                 "(CLIP 4-class $-$ 1K)", fontsize=12, fontweight="bold")
    ax.axvline(0, color="black", lw=0.8, zorder=4)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    # Legend text
    n_sig = rejected.sum()
    ax.text(0.98, 0.02,
            f"* FDR-corrected p < 0.05 ({n_sig}/{n_dims} significant)\n"
            f"green = high loading → CLIP-4 advantage\n"
            f"red = high loading → 1K advantage",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            fontstyle="italic", color=COLOR_NEUTRAL)

    # Print top dimensions
    print(f"\nDimension profiling: {n_sig}/{n_dims} significant (FDR q<0.05)")
    for i in order[:10]:
        sig = "*" if rejected[i] else ""
        print(f"  {labels[i]:<25} rho={rhos[i]:+.4f}  p={pvals[i]:.2e} {sig}")

    save_fig(fig, os.path.join(FIG_DIR, "dimension_profiling.png"))


# ── Main ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    data = load_data()
    concept_names = [str(c) for c in data["concept_names"]]

    print("Computing per-concept RDM row correlations...")
    diff = compute_differences(data)
    print(f"  {len(concept_names)} concepts, CLIP-4 wins: {(diff > 0).sum()}/{len(concept_names)}")

    top_idx, bot_idx = get_tail_indices(diff)

    print("\n[A] Creating image collages...")
    plot_collages(concept_names, diff, data["rep_image_paths"], top_idx, bot_idx)

    print("\n[B] Creating dendrograms...")
    plot_dendrograms(concept_names, data, top_idx, bot_idx)

    print("\n[C] Semantic dimension profiling...")
    plot_dimension_profiling(diff, data["embeddings"], data["dimension_labels"])

    print("\nDone!")


if __name__ == "__main__":
    main()
