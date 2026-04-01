"""Figure 1a: Label Space — shared PCA scatter with 2/4/1000-way coloring.

Three panels showing the same PCA coordinates colored by different
granularity levels, with image insets and decision boundary lines.

Usage (from project root):
    python manuscript/figures/fig1/plot_label_space.py
    python manuscript/figures/fig1/plot_label_space.py --recompute
"""

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import seaborn as sns

sys.path.insert(0, ".")

from manuscript.figures.fig1.utils import (
    CACHE_PATH, PALETTE_2, PALETTE_4, INSET_CLASSES,
    setup_style, median_split_labels, add_top_row_insets,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_top_panel(ax, pcs, labels, n_classes, colors, title,
                   point_size=22, alpha=0.75, show_ylabel=True,
                   decision_lines=None, subtitle=None):
    """Draw one shared-PCA scatter panel."""
    rng = np.random.RandomState(42)
    order = rng.permutation(len(labels))
    point_colors = np.array([colors[labels[i] % len(colors)] for i in order])

    ax.scatter(pcs[order, 0], pcs[order, 1],
               c=point_colors, s=point_size, alpha=alpha,
               edgecolors="white", linewidths=0.3,
               rasterized=True, zorder=2)

    ax.set_xlabel("PC 1", fontsize=10, labelpad=1)
    if show_ylabel:
        ax.set_ylabel("PC 2", fontsize=10, labelpad=1)
    if subtitle:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=18)
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes,
                fontsize=8.5, color="#666666", ha="center", va="bottom",
                fontstyle="italic")
    else:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis="both", length=0)

    for idx in [0, 1]:
        lo, hi = pcs[:, idx].min(), pcs[:, idx].max()
        margin = (hi - lo) * 0.04
        (ax.set_xlim if idx == 0 else ax.set_ylim)(lo - margin, hi + margin)

    sns.despine(ax=ax, offset=2, left=not show_ylabel)

    if decision_lines is not None:
        split_kw = dict(color="#222222", linestyle="--", linewidth=1.3,
                        alpha=0.6, zorder=5)
        for line in decision_lines:
            if line["type"] == "vline":
                ax.axvline(line["pos"], **split_kw)
            elif line["type"] == "hline_segment":
                ax.plot(line["x"], [line["pos"], line["pos"]],
                        clip_on=True, **split_kw)


def _make_1k_colors(class_labels):
    """Generate 1000 distinct colors with forced inset class colors."""
    rng_colors = np.random.RandomState(7)
    base_cmap = plt.cm.tab20
    colors_1k = []
    for i in range(1000):
        base = np.array(base_cmap(i % 20))
        jitter = rng_colors.uniform(-0.08, 0.08, 3)
        base[:3] = np.clip(base[:3] + jitter, 0, 1)
        colors_1k.append(tuple(base))
    rng_colors.shuffle(colors_1k)

    inset_1k_hex = [
        "#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#f781bf", "#1b9e77",
    ]
    for k, entry in enumerate(INSET_CLASSES):
        cls_idx = entry[0]
        pos = np.where(class_labels == cls_idx)[0]
        if len(pos) > 0:
            colors_1k[pos[0]] = tuple(mcolors.to_rgba(inset_1k_hex[k]))
    return colors_1k


def plot_label_space(save=True):
    """Generate Figure 1a: label space PCA scatter."""
    setup_style()

    print(f"Loading top-row cache: {CACHE_PATH}")
    raw = np.load(CACHE_PATH, allow_pickle=True)
    top_pcs = raw["pcs"]
    top_class_labels = raw["class_labels"]

    labels_2 = median_split_labels(top_pcs, 2)
    labels_4 = median_split_labels(top_pcs, 4)
    colors_1k = _make_1k_colors(top_class_labels)

    # Layout: [1000-class | divider | 2-class | 4-class]
    fig = plt.figure(figsize=(14.8, 4.8))
    gs = gridspec.GridSpec(1, 4, figure=fig,
                           width_ratios=[1.15, 0.02, 1, 1], wspace=0.12,
                           left=0.01, right=0.99, top=0.97, bottom=0.06)
    ax_1k = fig.add_subplot(gs[0, 0])
    ax_div = fig.add_subplot(gs[0, 1])
    ax_2 = fig.add_subplot(gs[0, 2])
    ax_4 = fig.add_subplot(gs[0, 3])

    plot_top_panel(ax_1k, top_pcs, top_class_labels, 1000, colors_1k,
                   "", point_size=20, alpha=0.70, show_ylabel=True)
    plot_top_panel(ax_2, top_pcs, labels_2, 2, PALETTE_2, "",
                   show_ylabel=False)
    plot_top_panel(ax_4, top_pcs, labels_4, 4, PALETTE_4, "",
                   show_ylabel=False)

    # Subtle vertical divider between fine-grained and coarse panels
    ax_div.set_xlim(0, 1)
    ax_div.set_ylim(0, 1)
    ax_div.axvline(0.5, ymin=0.08, ymax=0.92, color="#cccccc",
                   linewidth=1.2, linestyle="-", alpha=0.7)
    ax_div.set_axis_off()

    # Image insets
    from dotenv import load_dotenv
    load_dotenv()
    imagenet_dir = os.environ.get("IMAGENET_DATA_DIR", "")
    if imagenet_dir and os.path.isdir(imagenet_dir):
        for ax, labels, colors in [
            (ax_1k, top_class_labels, colors_1k),
            (ax_2, labels_2, PALETTE_2),
            (ax_4, labels_4, PALETTE_4),
        ]:
            add_top_row_insets(ax, top_pcs, top_class_labels,
                               labels, colors, INSET_CLASSES,
                               imagenet_dir, zoom=0.44, thumb_size=75)
    else:
        print(f"WARNING: ImageNet dir not found ({imagenet_dir}), skipping insets")

    # Nudge 1000-class panel left xlim to prevent chair inset overlapping PC2 label
    xl = ax_1k.get_xlim()
    ax_1k.set_xlim(xl[0] - (xl[1] - xl[0]) * 0.06, xl[1])

    if save:
        out_png = os.path.join(SCRIPT_DIR, "figure1a.png")
        fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white",
                    edgecolor="none")
        print(f"Saved -> {out_png}")

        out_svg = os.path.join(SCRIPT_DIR, "figure1a.svg")
        fig.savefig(out_svg, format="svg", bbox_inches="tight", facecolor="white",
                    edgecolor="none")
        print(f"Saved -> {out_svg}")
        plt.close(fig)

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute shared PCA data (requires GPU + CLIP)")
    args = parser.parse_args()

    if args.recompute or not os.path.exists(CACHE_PATH):
        from manuscript.figures.fig1.compute_pca_cache import compute_and_cache
        compute_and_cache()

    plot_label_space()
