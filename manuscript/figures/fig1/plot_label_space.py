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

from manuscript.figures.fig2.utils import (
    TOP_ROW_CACHE, PALETTE_2, PALETTE_4, INSET_CLASSES,
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

    ax.set_xlabel("PC 1", fontsize=10, labelpad=4)
    if show_ylabel:
        ax.set_ylabel("PC 2", fontsize=10, labelpad=4)
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
        margin = (hi - lo) * 0.10
        (ax.set_xlim if idx == 0 else ax.set_ylim)(lo - margin, hi + margin)

    sns.despine(ax=ax, offset=5, left=not show_ylabel)

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

    # Load cached PCA data (lives in fig2/ alongside the representation data)
    print(f"Loading top-row cache: {TOP_ROW_CACHE}")
    raw = np.load(TOP_ROW_CACHE, allow_pickle=True)
    top_pcs = raw["pcs"]
    top_class_labels = raw["class_labels"]

    labels_2 = median_split_labels(top_pcs, 2)
    labels_4 = median_split_labels(top_pcs, 4)
    colors_1k = _make_1k_colors(top_class_labels)

    # Layout: [2-class | 4-class | 1000-class]
    fig = plt.figure(figsize=(14, 4.8))
    gs = gridspec.GridSpec(1, 3, figure=fig,
                           width_ratios=[1, 1, 1.15], wspace=0.15,
                           left=0.04, right=0.97, top=0.86, bottom=0.10)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    plot_top_panel(axes[0], top_pcs, labels_2, 2, PALETTE_2, "",
                   show_ylabel=True)
    plot_top_panel(axes[1], top_pcs, labels_4, 4, PALETTE_4, "",
                   show_ylabel=False)
    plot_top_panel(axes[2], top_pcs, top_class_labels, 1000, colors_1k,
                   "", point_size=20, alpha=0.70, show_ylabel=False)

    # Image insets
    from dotenv import load_dotenv
    load_dotenv()
    imagenet_dir = os.environ.get("IMAGENET_DATA_DIR", "")
    if imagenet_dir and os.path.isdir(imagenet_dir):
        for i, (labels, colors) in enumerate([
            (labels_2, PALETTE_2),
            (labels_4, PALETTE_4),
            (top_class_labels, colors_1k),
        ]):
            add_top_row_insets(axes[i], top_pcs, top_class_labels,
                               labels, colors, INSET_CLASSES,
                               imagenet_dir, zoom=0.40, thumb_size=68)
    else:
        print(f"WARNING: ImageNet dir not found ({imagenet_dir}), skipping insets")

    # Unified header
    pos0 = axes[0].get_position()
    pos1 = axes[1].get_position()
    pos2 = axes[2].get_position()
    header_y = pos0.y1 + 0.06
    sub_y = pos0.y1 + 0.02

    cx0 = (pos0.x0 + pos0.x1) / 2
    cx1 = (pos1.x0 + pos1.x1) / 2
    cx2 = (pos2.x0 + pos2.x1) / 2

    fig.text(cx0, header_y, "2 classes", ha="center", va="bottom",
             fontsize=12, fontweight="bold", color="#1a1a1a",
             transform=fig.transFigure)
    fig.text(cx1, header_y, "4 classes", ha="center", va="bottom",
             fontsize=12, fontweight="bold", color="#1a1a1a",
             transform=fig.transFigure)
    fig.text(cx2, header_y, "1000 classes", ha="center", va="bottom",
             fontsize=12, fontweight="bold", color="#1a1a1a",
             transform=fig.transFigure)

    fig.text(cx0, sub_y, "median split on PC 1", ha="center", va="bottom",
             fontsize=8, color="#888888", fontstyle="italic",
             transform=fig.transFigure)
    fig.text(cx1, sub_y, "+ split on PC 2", ha="center", va="bottom",
             fontsize=8, color="#888888", fontstyle="italic",
             transform=fig.transFigure)
    fig.text(cx2, sub_y, "default ImageNet", ha="center", va="bottom",
             fontsize=8, color="#888888", fontstyle="italic",
             transform=fig.transFigure)

    gap_cx = (pos1.x1 + pos2.x0) / 2
    fig.text(gap_cx, header_y + 0.005,
             "8  \u00b7  16  \u00b7  32  \u00b7  64 classes",
             ha="center", va="bottom",
             fontsize=9.5, fontweight="normal", color="#999999",
             transform=fig.transFigure)

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

    if args.recompute or not os.path.exists(TOP_ROW_CACHE):
        from manuscript.figures.fig2.pc_scatter_explore import compute_and_cache
        compute_and_cache()

    plot_label_space()
