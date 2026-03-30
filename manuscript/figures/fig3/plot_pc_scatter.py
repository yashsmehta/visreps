"""PC / UMAP scatter: Behavioral vs 8 classes (CLIP repr.) vs 1000-class.

Three-panel scatter showing how concept representations are organized
in behavioral similarity space vs two model spaces. Points colored by
6 broad THINGS super-categories.

Usage (from project root):
    python manuscript/figures/fig3/plot_pc_scatter.py                    # PCA, fc1 (default)
    python manuscript/figures/fig3/plot_pc_scatter.py --layer fc2        # PCA, fc2
    python manuscript/figures/fig3/plot_pc_scatter.py --method umap      # UMAP, fc1
    python manuscript/figures/fig3/plot_pc_scatter.py --no-behavioral    # model panels only
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ACTIVATIONS = os.path.join(SCRIPT_DIR, "activations.npz")
BEHAV_DATA = os.path.normpath(os.path.join(SCRIPT_DIR, "../../..",
    "experiments/things_visualizations/data/things_viz_data.npz"))
CATEGORY_FILE = os.path.expanduser(
    "~/.cache/bonner-datasets/hebart2019.things/03_category-level/category27_manual.tsv"
)

# ── 6 super-categories (grouping the 27 THINGS manual categories) ─────
SUPER_CATEGORIES = {
    "Animal": ["animal", "bird", "insect"],
    "Food":   ["food", "fruit", "vegetable", "dessert", "drink"],
    "Clothing": ["clothing", "clothing accessory"],
    "Tool":   ["tool", "kitchen tool", "kitchen appliance"],
    "Vehicle": ["vehicle", "part of car"],
    "Plant":  ["plant"],
}

SUPER_COLORS = {
    "Animal":   "#d62728",
    "Food":     "#e88a1a",
    "Clothing": "#7b4fae",
    "Tool":     "#2578b2",
    "Vehicle":  "#27a34a",
    "Plant":    "#d65fad",
}

SUPER_ORDER = ["Animal", "Food", "Clothing", "Tool", "Vehicle", "Plant"]


# ── Category assignment ───────────────────────────────────────────────

def load_super_categories(n_concepts):
    """Assign each THINGS concept to one of 6 super-categories."""
    cat_df = pd.read_csv(CATEGORY_FILE, sep="\t")
    cat_names = list(cat_df.columns)

    fine_to_super = {}
    for i, super_name in enumerate(SUPER_ORDER):
        for fine in SUPER_CATEGORIES[super_name]:
            fine_to_super[fine] = i

    labels = np.full(n_concepts, -1, dtype=int)
    for idx, (_, row) in enumerate(cat_df.iterrows()):
        for cat_name in cat_names:
            if row[cat_name] == 1 and cat_name in fine_to_super:
                labels[idx] = fine_to_super[cat_name]
                break

    assigned = (labels >= 0).sum()
    print(f"Super-categories: {assigned}/{n_concepts} assigned "
          f"({n_concepts - assigned} unassigned)")
    return labels


# ── Dimensionality reduction ──────────────────────────────────────────

def compute_pca(features, n_pcs=2):
    """PCA via truncated SVD. Returns (projections, var_explained%).

    Uses SVD on the centered data matrix directly — O(n^2 * d) — rather than
    eigendecomposition of the d×d covariance matrix — O(d^3). Much faster
    when n_samples (1854) << n_features (4096).
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_pcs)
    projections = pca.fit_transform(features)
    var_explained = pca.explained_variance_ratio_ * 100
    return projections, var_explained


def compute_umap(features, n_neighbors=50, min_dist=0.3, seed=42):
    """UMAP to 2D with PCA pre-reduction."""
    from sklearn.decomposition import PCA
    import umap

    if features.shape[1] > 50:
        features = PCA(n_components=50, random_state=seed).fit_transform(features)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist,
        metric="euclidean", random_state=seed, verbose=False,
    )
    return reducer.fit_transform(features.astype(np.float32))


def l2_normalize(features):
    """Row-wise L2 normalization."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return features / norms


# ── Scatter panel ─────────────────────────────────────────────────────

def plot_scatter_panel(ax, coords, labels, title, subtitle=None,
                       xlabel="PC 1", ylabel="PC 2",
                       point_size=10, alpha=0.58):
    """Draw one 2D scatter colored by super-category."""
    rng = np.random.RandomState(42)
    order = rng.permutation(len(labels))

    # All points as grey background layer
    ax.scatter(coords[order, 0], coords[order, 1],
               c="#c8c8c8", s=point_size * 0.5, alpha=0.45,
               edgecolors="none", rasterized=True, zorder=1)

    # Colored overlay for assigned super-categories
    for i, name in enumerate(SUPER_ORDER):
        mask = labels[order] == i
        if not mask.any():
            continue
        ax.scatter(coords[order[mask], 0], coords[order[mask], 1],
                   c=SUPER_COLORS[name], s=point_size, alpha=alpha,
                   edgecolors="white", linewidths=0.12,
                   rasterized=True, zorder=2, label=name)

    ax.set_xlabel(xlabel, fontsize=10, labelpad=3)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=3)

    # Title with optional lighter-weight subtitle
    if subtitle:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=18,
                     color="#1a1a1a")
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
                fontsize=9.5, fontweight="normal", color="#555555",
                ha="center", va="bottom")
    else:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8,
                     color="#1a1a1a")

    ax.tick_params(axis="both", labelsize=8.5, direction="out", pad=2)
    sns.despine(ax=ax, offset=4)
    # Data-driven margins
    for idx in [0, 1]:
        lo, hi = coords[:, idx].min(), coords[:, idx].max()
        margin = (hi - lo) * 0.08
        (ax.set_xlim if idx == 0 else ax.set_ylim)(lo - margin, hi + margin)

    # Smart tick placement: pick ~4-5 nice round ticks within data range
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))

    # Clean tick label formatting — strip trailing zeros
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:g}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:g}"))


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", default="fc1", choices=["fc1", "fc2"])
    parser.add_argument("--method", default="pca", choices=["pca", "umap"])
    parser.add_argument("--no-behavioral", action="store_true",
                        help="Skip behavioral panel, produce 2-panel plot")
    args = parser.parse_args()

    # Load activations
    data = np.load(ACTIVATIONS, allow_pickle=True)
    clip8 = data[f"clip8_{args.layer}"]
    thousand = data[f"thousand_{args.layer}"]
    n_concepts = len(data["concept_names"])
    print(f"Loaded {n_concepts} concepts, layer={args.layer}, method={args.method}")

    labels = load_super_categories(n_concepts)

    # Reduce dimensions
    if args.method == "pca":
        clip8_2d, clip8_var = compute_pca(l2_normalize(clip8))
        thousand_2d, thousand_var = compute_pca(l2_normalize(thousand))
        clip8_xl = f"PC 1"
        clip8_yl = f"PC 2"
        thousand_xl = f"PC 1"
        thousand_yl = f"PC 2"
    else:
        print("Computing UMAP (clip8)...")
        clip8_2d = compute_umap(l2_normalize(clip8))
        print("Computing UMAP (1000-class)...")
        thousand_2d = compute_umap(l2_normalize(thousand))
        clip8_xl = thousand_xl = "UMAP 1"
        clip8_yl = thousand_yl = "UMAP 2"

    # Behavioral embeddings
    show_behavioral = not args.no_behavioral
    if show_behavioral:
        behav_data = np.load(BEHAV_DATA, allow_pickle=True)
        embeddings = behav_data["embeddings"]
        if args.method == "pca":
            behav_2d, behav_var = compute_pca(embeddings)
            behav_xl, behav_yl = "PC 1", "PC 2"
        else:
            print("Computing UMAP (behavioral)...")
            behav_2d = compute_umap(embeddings)
            behav_xl, behav_yl = "UMAP 1", "UMAP 2"

    # ── Plot ──
    sns.set_theme(style="ticks", context="paper", font_scale=1.05)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    n_panels = 3 if show_behavioral else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels, 3.7))

    layer_label = args.layer.upper()

    panel_idx = 0
    if show_behavioral:
        plot_scatter_panel(axes[panel_idx], behav_2d, labels,
                           "Behavioral", subtitle="(ground truth)",
                           xlabel=behav_xl, ylabel=behav_yl)
        panel_idx += 1

    plot_scatter_panel(axes[panel_idx], clip8_2d, labels,
                       f"8 classes (CLIP repr.)", subtitle=f"({layer_label})",
                       xlabel=clip8_xl, ylabel=clip8_yl)
    panel_idx += 1

    plot_scatter_panel(axes[panel_idx], thousand_2d, labels,
                       f"1000-class", subtitle=f"({layer_label})",
                       xlabel=thousand_xl, ylabel=thousand_yl)

    # Panel labels
    for i, ax in enumerate(axes):
        label = chr(ord("a") + i)
        ax.text(-0.10, 1.18, label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="left",
                fontfamily="sans-serif", color="#000000")

    # Shared legend
    handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=SUPER_COLORS[name],
               markeredgecolor="white", markeredgewidth=0.4,
               markersize=7.5, label=name)
        for name in SUPER_ORDER
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(SUPER_ORDER), fontsize=8, frameon=False,
               handletextpad=0.4, columnspacing=1.4,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.subplots_adjust(wspace=0.42)

    suffix = f"_{args.method}_{args.layer}"
    if not show_behavioral:
        suffix += "_no_behav"
    out = os.path.join(SCRIPT_DIR, f"pc_scatter{suffix}.png")
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white",
                edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
