"""
Category-annotated RDM comparison: Behavioral | CLIP 4-class | 1K | Difference.

Concepts sorted by THINGS 27 categories (manual assignments) with colored
sidebar annotations and a legend. Within each category, concepts are ordered
by hierarchical clustering on the behavioral RDM sub-block.

Input:  experiments/things_visualizations/data/things_viz_data.npz
        ~/.cache/bonner-datasets/hebart2019.things/03_category-level/category27_manual.tsv
Output: experiments/things_visualizations/figures/rdm_categorized.png
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

from experiments.things_visualizations.utils import load_data, FIG_DIR
from visreps.analysis.rsa import compute_rdm, compute_rdm_correlation

sns.set_theme(style="ticks", context="paper", font_scale=1.0)


CATEGORY_FILE = os.path.expanduser(
    "~/.cache/bonner-datasets/hebart2019.things/03_category-level/category27_manual.tsv"
)


def rank_transform(rdm):
    """Rank upper triangle, mirror to lower, scale to [0, 1]."""
    n = rdm.shape[0]
    triu = np.triu_indices(n, k=1)
    ranks = rankdata(rdm[triu]) / rdm[triu].size
    ranked = np.zeros_like(rdm)
    ranked[triu] = ranks
    ranked.T[triu] = ranks
    return ranked


def load_categories():
    """Load THINGS 27-category manual assignments.

    Returns a list of category names (one per concept, in dataset order).
    Each concept is assigned to its first matching category, or 'Other'.
    """
    cat_df = pd.read_csv(CATEGORY_FILE, sep="\t")
    categories = []
    cat_names = list(cat_df.columns)
    for _, row in cat_df.iterrows():
        assigned = [c for c in cat_names if row[c] == 1]
        categories.append(assigned[0] if assigned else "Other")
    return np.array(categories)


def build_category_sort_order(categories, behav_rdm):
    """Sort concepts by category (alphabetical, 'Other' last), then by
    hierarchical clustering within each category block."""
    unique_cats = sorted(set(categories) - {"Other"})
    if "Other" in categories:
        unique_cats.append("Other")

    sorted_indices = []
    block_boundaries = []  # (start_idx, category_name)
    offset = 0

    for cat in unique_cats:
        member_idx = np.where(categories == cat)[0]
        if len(member_idx) <= 2:
            # Too few for clustering — just append in original order
            order = member_idx
        else:
            # Hierarchical clustering on behavioral sub-RDM
            sub_rdm = behav_rdm[np.ix_(member_idx, member_idx)]
            sub_condensed = squareform(sub_rdm, checks=False)
            sub_order = leaves_list(linkage(sub_condensed, method="average"))
            order = member_idx[sub_order]

        block_boundaries.append((offset, cat, len(order)))
        sorted_indices.extend(order)
        offset += len(order)

    return np.array(sorted_indices), block_boundaries, unique_cats


def draw_category_sidebar(ax, block_boundaries, n_concepts, cat_colors, cat_to_idx,
                          side="left", width_frac=0.022, gap_frac=0.008):
    """Draw a colored sidebar along the edge of an RDM panel."""
    w = n_concepts * width_frac
    gap = n_concepts * gap_frac
    for start, cat, size in block_boundaries:
        color = cat_colors[cat_to_idx[cat]]
        if side == "left":
            rect = mpatches.Rectangle(
                (-w - gap, start - 0.5), w, size,
                facecolor=color, edgecolor="none", clip_on=False)
        else:  # bottom
            rect = mpatches.Rectangle(
                (start - 0.5, n_concepts - 0.5 + gap), size, w,
                facecolor=color, edgecolor="none", clip_on=False)
        ax.add_patch(rect)


def draw_boundary_lines(ax, block_boundaries, n_concepts, color="#444444",
                        lw=0.3, alpha=0.6):
    """Draw thin lines at category boundaries."""
    for start, _, size in block_boundaries:
        if start > 0:
            ax.axhline(start - 0.5, color=color, lw=lw, alpha=alpha)
            ax.axvline(start - 0.5, color=color, lw=lw, alpha=alpha)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    data = load_data()

    # Load categories and build sort order
    categories = load_categories()
    behav_rdm_unsorted = compute_rdm(
        torch.tensor(data["embeddings"], dtype=torch.float32)
    ).numpy()
    sort_idx, block_boundaries, unique_cats = build_category_sort_order(
        categories, behav_rdm_unsorted
    )

    # Curated saturated palette — chosen for contrast against dark magma background
    PALETTE_28 = [
        "#e31a1c", "#1f78b4", "#33a02c", "#6a3d9a", "#ff7f00",  # animal, bird, body part, clothing, clothing acc
        "#b15928", "#e377c2", "#1b9e77", "#d95f02", "#7570b3",  # container, dessert, drink, electronic, food
        "#a6d854", "#e6ab02", "#d4a76a", "#bcbd22", "#17becf",  # fruit, furniture, home decor, insect, kitchen app
        "#e7298a", "#9467bd", "#c44e52", "#2ca02c", "#8c564b",  # kitchen tool, med equip, music, office, part of car
        "#637939", "#fd8d3c", "#6baed6", "#9e9ac8", "#e7969c",  # plant, sports, tool, toy, vegetable
        "#3182bd", "#74c476",                                    # vehicle, weapon
        "#bdbdbd",                                               # Other
    ]
    cat_colors = PALETTE_28[:len(unique_cats)]
    cat_to_idx = {cat: i for i, cat in enumerate(unique_cats)}

    # Build RDMs (sorted)
    rdm_data = {
        "Behavioral": data["embeddings"],
        "CLIP 4-class": data["clip4_acts"],
        "1000-class": data["thousand_acts"],
    }
    rdms = {}
    for key, arr in rdm_data.items():
        rdm = compute_rdm(torch.tensor(arr, dtype=torch.float32)).numpy()
        rdms[key] = rdm[np.ix_(sort_idx, sort_idx)]

    # RSA scores
    rsa_scores = {}
    for key in ["CLIP 4-class", "1000-class"]:
        rsa_scores[key] = compute_rdm_correlation(
            torch.tensor(rdms[key]), torch.tensor(rdms["Behavioral"]),
            correlation="Spearman"
        )

    # Difference RDM: |behav - 1K| - |behav - clip4| (positive = clip4 closer to ground truth)
    ranked_behav = rank_transform(rdms["Behavioral"])
    diff_rdm = (
        np.abs(ranked_behav - rank_transform(rdms["1000-class"]))
        - np.abs(ranked_behav - rank_transform(rdms["CLIP 4-class"]))
    )

    # Rank-transform the main RDMs for display
    rdms_ranked = {key: rank_transform(rdm) for key, rdm in rdms.items()}

    n = rdms_ranked["Behavioral"].shape[0]

    # ── Plot ─────────────────────────────────────────────────────────
    # Nature double-column width: 183mm ~ 7.2in.
    # Layout: [panel_a | panel_b | panel_c | cb1 | gap | panel_d | cb2]
    # with legend row beneath.

    fig = plt.figure(figsize=(7.2, 3.2))
    gs_outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[1, 0.20],
        hspace=0.12,
    )

    # Top row: panels + colorbars (colorbars are vertically centered, 70% height)
    gs_top = gridspec.GridSpecFromSubplotSpec(
        3, 7, subplot_spec=gs_outer[0],
        width_ratios=[1, 1, 1, 0.035, 0.07, 1, 0.035],
        height_ratios=[0.15, 0.70, 0.15],
        wspace=0.05, hspace=0,
    )
    ax0 = fig.add_subplot(gs_top[0:3, 0])
    ax1 = fig.add_subplot(gs_top[0:3, 1])
    ax2 = fig.add_subplot(gs_top[0:3, 2])
    ax_cb1 = fig.add_subplot(gs_top[1, 3])   # center 70% only
    # gs_top[:, 4] = spacer
    ax3 = fig.add_subplot(gs_top[0:3, 5])
    ax_cb2 = fig.add_subplot(gs_top[1, 6])   # center 70% only

    # Bottom row: legend
    ax_legend = fig.add_subplot(gs_outer[1])
    ax_legend.axis("off")

    rdm_axes = [ax0, ax1, ax2, ax3]
    panels = [
        ("Behavioral", rdms_ranked["Behavioral"], None, "magma"),
        ("CLIP 4-class", rdms_ranked["CLIP 4-class"], rsa_scores["CLIP 4-class"], "magma"),
        ("1000-class", rdms_ranked["1000-class"], rsa_scores["1000-class"], "magma"),
        ("Difference", diff_rdm, None, "RdBu_r"),
    ]
    panel_labels = ["a", "b", "c", "d"]

    diff_vlim = np.percentile(np.abs(diff_rdm), 99)
    ims = []
    for ax, (title, rdm, rsa, cmap), plabel in zip(rdm_axes, panels, panel_labels):
        kwargs = {"cmap": cmap, "interpolation": "nearest", "aspect": "equal",
                  "rasterized": True}
        if cmap == "RdBu_r":
            kwargs["vmin"] = -diff_vlim
            kwargs["vmax"] = diff_vlim
        else:
            kwargs["vmin"] = 0
            kwargs["vmax"] = 1

        im = ax.imshow(rdm, **kwargs)
        ims.append(im)

        # Title + subtitle
        ax.set_title(title, fontsize=8, fontweight="bold", pad=10, color="#1a1a1a",
                     fontfamily="sans-serif")
        if rsa is not None:
            ax.text(0.5, 1.01, f"$\\rho_s$ = {rsa:.3f}", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=6, color="#555555")
        elif "Behavioral" in title:
            ax.text(0.5, 1.01, "(ground truth)", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=6, color="#555555",
                    fontstyle="italic")
        elif "Difference" in title:
            ax.text(0.5, 1.01, "$|$err$_{1K}|$ $-$ $|$err$_{4}|$", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=6, color="#555555")

        # Panel label
        ax.text(-0.02, 1.13, plabel, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top", ha="right", color="#000000")

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Category sidebar + boundary lines
        line_color = "white" if cmap == "magma" else "#999999"
        line_alpha = 0.6 if cmap == "magma" else 0.4
        draw_boundary_lines(ax, block_boundaries, n, color=line_color,
                            lw=0.25, alpha=line_alpha)
        draw_category_sidebar(ax, block_boundaries, n, cat_colors, cat_to_idx,
                              side="left", width_frac=0.026, gap_frac=0.005)
        draw_category_sidebar(ax, block_boundaries, n, cat_colors, cat_to_idx,
                              side="bottom", width_frac=0.026, gap_frac=0.005)

    # ── Colorbars in dedicated gridspec columns ──────────────────────
    cb1 = fig.colorbar(ims[0], cax=ax_cb1)
    cb1.ax.tick_params(labelsize=5.5, length=2, width=0.4, pad=2)
    cb1.outline.set_linewidth(0.4)
    cb1.ax.yaxis.set_major_locator(mticker.FixedLocator([0, 0.5, 1.0]))
    cb1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    cb2 = fig.colorbar(ims[3], cax=ax_cb2)
    cb2.ax.tick_params(labelsize=5.5, length=2, width=0.4, pad=2)
    cb2.outline.set_linewidth(0.4)
    # Symmetric ticks for difference colorbar — round DOWN to stay within range
    diff_tick = np.floor(diff_vlim * 10) / 10  # e.g. 0.67 -> 0.6
    cb2.ax.yaxis.set_major_locator(mticker.FixedLocator([-diff_tick, 0, diff_tick]))
    cb2.ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:+.1f}" if x != 0 else "0.0"
    ))

    # ── Category legend ──────────────────────────────────────────────
    LABEL_MAP = {
        "body part": "Body part", "clothing accessory": "Clothing acc.",
        "electronic device": "Electronic", "home decor": "Home decor",
        "kitchen appliance": "Kitchen appl.", "kitchen tool": "Kitchen tool",
        "medical equipment": "Medical equip.", "musical instrument": "Music instr.",
        "office supply": "Office supply", "part of car": "Car part",
        "sports equipment": "Sports equip.",
    }
    legend_handles = []
    for cat in unique_cats:
        color = cat_colors[cat_to_idx[cat]]
        label = LABEL_MAP.get(cat, cat.capitalize())
        legend_handles.append(mpatches.Patch(facecolor=color, edgecolor="#bbbbbb",
                                             linewidth=0.4, label=label))

    # Thin separator line above legend
    ax_legend.axhline(y=0.95, xmin=0.03, xmax=0.97, color="#c0c0c0",
                      lw=0.6, clip_on=False)

    leg = ax_legend.legend(
        handles=legend_handles, loc="center", fontsize=5.5,
        frameon=False, ncol=7, columnspacing=0.9,
        handlelength=1.0, handleheight=0.7, labelspacing=0.4,
        handletextpad=0.4,
        title="Semantic categories",
        title_fontproperties={"size": 6.5, "weight": "bold"},
    )
    leg._legend_box.sep = 4

    out = os.path.join(FIG_DIR, "rdm_categorized.png")
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved: {out}")
    for key, rsa in rsa_scores.items():
        print(f"  {key} RSA: {rsa:.4f}")


if __name__ == "__main__":
    main()
