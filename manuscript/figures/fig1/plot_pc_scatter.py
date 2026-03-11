"""Figure 1C — PC1/PC2 scatter: pretrained (1000-way) vs. coarse-trained.

Side-by-side scatter plots showing how coarse training fundamentally changes
the geometry of learned representations. Uses cached data from the
representation_analysis/2pcs_compare pipeline.

Usage (from project root):
    python manuscript/figures/fig1/plot_pc_scatter.py
    python manuscript/figures/fig1/plot_pc_scatter.py --layer fc1
    python manuscript/figures/fig1/plot_pc_scatter.py --layer fc2 --n_classes 4 --tag clip
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))),
    "experiments", "representation_analysis", "2pcs_compare",
)

LAYER_LABELS = {"conv4": "Conv 4", "fc1": "FC 1", "fc2": "FC 2"}
TAG_LABELS = {"clip": "CLIP", "alexnet": "AlexNet", "dino": "DINO"}

PALETTES = {
    2: ["#2176AE", "#E84855"],
    4: ["#00A896", "#7B68EE", "#E8963E", "#D64045"],
}

# ── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style="ticks", context="paper", font_scale=1.0)
RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
}


def plot_scatter_panel(ax, pcs, var_explained, labels, n_classes, colors,
                       title, point_size=0.7, alpha=0.25):
    """Draw one PC1 vs PC2 scatter panel."""
    # Shuffle draw order so no class dominates the foreground
    order = np.random.permutation(len(labels))
    pcs_s, labels_s = pcs[order], labels[order]

    for c in range(n_classes):
        mask = labels_s == c
        ax.scatter(
            pcs_s[mask, 0], pcs_s[mask, 1],
            c=colors[c], s=point_size, alpha=alpha, edgecolors="none",
            rasterized=True, zorder=2,
        )

    ax.set_xlabel(f"PC 1 ({var_explained[0]:.1f}% var.)",
                  fontsize=7.5, labelpad=3)
    ax.set_ylabel(f"PC 2 ({var_explained[1]:.1f}% var.)",
                  fontsize=7.5, labelpad=3)

    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=5)
    ax.tick_params(axis="both", labelsize=6.5, length=2.5, width=0.5, pad=2)

    # Tight limits with small margin around data extent
    for axis_fn, idx in [(ax.set_xlim, 0), (ax.set_ylim, 1)]:
        lo, hi = pcs[:, idx].min(), pcs[:, idx].max()
        margin = (hi - lo) * 0.04
        axis_fn(lo - margin, hi + margin)

    sns.despine(ax=ax, offset=4)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    # Consistent decimal formatting
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_classes", type=int, default=4)
    parser.add_argument("--tag", type=str, default="clip",
                        help="PCA source model tag used in run_analysis.py")
    parser.add_argument("--layer", type=str, default="fc2",
                        choices=["conv4", "fc1", "fc2"])
    args = parser.parse_args()

    # Resolve data file — try tag-specific name first, then generic
    tagged_path = os.path.join(DATA_DIR, f"data_{args.n_classes}way_{args.tag}.npz")
    generic_path = os.path.join(DATA_DIR, f"data_{args.n_classes}way.npz")
    data_path = tagged_path if os.path.exists(tagged_path) else generic_path
    if not os.path.exists(data_path):
        print(f"Data not found: tried {tagged_path} and {generic_path}")
        print("Run: python experiments/representation_analysis/2pcs_compare/run_analysis.py "
              f"--n_classes {args.n_classes}")
        return

    data = np.load(data_path, allow_pickle=True)
    layer = args.layer
    n_classes = int(data["n_classes"])
    pca_labels = data["pca_labels"]

    pretrained_pcs = data[f"{layer}_pretrained_pcs"]
    pretrained_var = data[f"{layer}_pretrained_var"]
    trained_pcs = data[f"{layer}_trained_pcs"]
    trained_var = data[f"{layer}_trained_var"]

    colors = PALETTES.get(n_classes, PALETTES[4][:n_classes])
    tag_label = TAG_LABELS.get(args.tag, args.tag.upper())
    layer_label = LAYER_LABELS.get(layer, layer)

    np.random.seed(42)  # reproducible point ordering
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    plot_scatter_panel(
        axes[0], pretrained_pcs, pretrained_var, pca_labels, n_classes, colors,
        title=f"Fine-grained (1000-way)",
    )
    plot_scatter_panel(
        axes[1], trained_pcs, trained_var, pca_labels, n_classes, colors,
        title=f"Coarsened ({n_classes}-way)",
    )

    # Panel labels
    for i, (ax, letter) in enumerate(zip(axes, "ab")):
        ax.text(-0.14, 1.12, letter, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top", ha="left")

    # Shared legend — circle markers matching the scatter
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[c],
               markeredgecolor="none", markersize=5.5, label=f"Class {c}")
        for c in range(n_classes)
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=n_classes, fontsize=7,
        frameon=False, handletextpad=0.2, columnspacing=0.8,
        bbox_to_anchor=(0.5, 0.005),
    )

    plt.tight_layout(rect=[0, 0.055, 1, 1])
    plt.subplots_adjust(wspace=0.40)
    out = os.path.join(SCRIPT_DIR,
                       f"pc_scatter_{n_classes}way_{args.tag}_{layer}.png")
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white",
                edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
