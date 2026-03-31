"""Shared THINGS plotting utilities for manuscript figures.

Extracted from fig6/figure6.py so that both fig4 (THINGS behavioral)
and fig5 (per-concept analysis) can reuse the same data loading and panel
drawing functions.
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import seaborn as sns

from experiments.things_visualizations.utils import load_data
from experiments.things_visualizations.plot_rdms_categorized import (
    rank_transform, load_categories, build_category_sort_order,
    draw_category_sidebar, draw_boundary_lines,
)
from experiments.things_visualizations.per_row_scatter_categories import (
    per_row_correlations, short_cat_label,
)
from visreps.analysis.rsa import compute_rdm, compute_rdm_correlation

# ── Constants ─────────────────────────────────────────────────────────────

# Category palette (matches plot_rdms_categorized.py)
PALETTE_28 = [
    "#e31a1c", "#1f78b4", "#33a02c", "#6a3d9a", "#ff7f00",
    "#b15928", "#e377c2", "#1b9e77", "#d95f02", "#7570b3",
    "#a6d854", "#e6ab02", "#d4a76a", "#bcbd22", "#17becf",
    "#e7298a", "#9467bd", "#c44e52", "#2ca02c", "#8c564b",
    "#637939", "#fd8d3c", "#6baed6", "#9e9ac8", "#e7969c",
    "#3182bd", "#74c476", "#bdbdbd",
]

# Scatter colors — greens for 8-class advantage, warm orange/reds for 1K advantage
# Dark forest → bright green → lime → teal → olive — max distinguishability
GREEN_COLORS = ["#1b5e20", "#2e7d32", "#8bc34a", "#00897b", "#6a8e3e"]
GREEN_MARKERS = ["o", "s", "^", "D", "v"]
ORANGE_COLORS = ["#c1121f", "#e07a28", "#d4a373", "#f4a261"]
ORANGE_MARKERS = ["o", "s", "^", "D"]


# ═══════════════════════════════════════════════════════════════════════════
# Shared data computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_things_data(data=None):
    """Compute all RDMs, per-row correlations, and category data once.

    If data is None, loads it via load_data().
    Returns a dict with all precomputed results needed by RDM and scatter panels.
    """
    if data is None:
        print("Loading THINGS data...")
        data = load_data()

    print("  Computing RDMs...")
    rdm_behav = compute_rdm(torch.tensor(data["embeddings"], dtype=torch.float32)).numpy()
    rdm_clip8 = compute_rdm(torch.tensor(data["clip8_acts"], dtype=torch.float32)).numpy()
    rdm_1k = compute_rdm(torch.tensor(data["thousand_acts"], dtype=torch.float32)).numpy()

    print("  Loading categories and sorting...")
    categories = load_categories()
    sort_idx, block_boundaries, unique_cats = build_category_sort_order(
        categories, rdm_behav
    )

    # Sorted RDMs
    rdms_sorted = {
        "Behavioral": rdm_behav[np.ix_(sort_idx, sort_idx)],
        "8 classes (CLIP repr.)": rdm_clip8[np.ix_(sort_idx, sort_idx)],
        "1000-class": rdm_1k[np.ix_(sort_idx, sort_idx)],
    }

    # RSA scores
    rsa_scores = {}
    for key in ["8 classes (CLIP repr.)", "1000-class"]:
        rsa_scores[key] = compute_rdm_correlation(
            torch.tensor(rdms_sorted[key]), torch.tensor(rdms_sorted["Behavioral"]),
            correlation="Spearman"
        )

    # Difference RDM
    ranked_behav = rank_transform(rdms_sorted["Behavioral"])
    diff_rdm = (
        np.abs(ranked_behav - rank_transform(rdms_sorted["1000-class"]))
        - np.abs(ranked_behav - rank_transform(rdms_sorted["8 classes (CLIP repr.)"]))
    )

    # Rank-transform for display
    rdms_ranked = {key: rank_transform(rdm) for key, rdm in rdms_sorted.items()}

    # Per-row correlations (for scatter panel)
    print("  Computing per-row correlations...")
    corr_clip8 = per_row_correlations(rdm_clip8, rdm_behav)
    corr_1k = per_row_correlations(rdm_1k, rdm_behav)

    return {
        "categories": categories,
        "sort_idx": sort_idx,
        "block_boundaries": block_boundaries,
        "unique_cats": unique_cats,
        "rdms_ranked": rdms_ranked,
        "diff_rdm": diff_rdm,
        "rsa_scores": rsa_scores,
        "corr_clip8": corr_clip8,
        "corr_1k": corr_1k,
    }


# ═══════════════════════════════════════════════════════════════════════════
# RDM panels
# ═══════════════════════════════════════════════════════════════════════════

def plot_rdm_panels(axes, precomputed, show_difference=True, colorbar_axes=None):
    """Draw RDM panels using precomputed data.

    axes: list of 3 or 4 axes (Behavioral, 8 classes (CLIP repr.), 1000-class, [Difference])
    colorbar_axes: tuple (ax_cb_magma, ax_cb_diff) or None to skip colorbars.
    """
    block_boundaries = precomputed["block_boundaries"]
    unique_cats = precomputed["unique_cats"]
    rdms_ranked = precomputed["rdms_ranked"]
    diff_rdm = precomputed["diff_rdm"]
    rsa_scores = precomputed["rsa_scores"]

    cat_colors = PALETTE_28[:len(unique_cats)]
    cat_to_idx = {cat: i for i, cat in enumerate(unique_cats)}
    n = rdms_ranked["Behavioral"].shape[0]

    panels = [
        ("Behavioral", rdms_ranked["Behavioral"], None, "magma"),
        ("8 classes (CLIP repr.)", rdms_ranked["8 classes (CLIP repr.)"], rsa_scores["8 classes (CLIP repr.)"], "magma"),
        ("1000-class", rdms_ranked["1000-class"], rsa_scores["1000-class"], "magma"),
    ]
    if show_difference:
        panels.append(("Difference", diff_rdm, None, "RdBu_r"))

    diff_vlim = np.percentile(np.abs(diff_rdm), 99)
    ims = []
    for ax, (title, rdm, rsa, cmap) in zip(axes, panels):
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
        ax.set_title(title, fontsize=10, fontweight="bold", pad=16)
        if rsa is not None:
            ax.text(0.5, 1.015, f"$\\rho_s$ = {rsa:.3f}", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=7.5, color="#444444")
        elif "Behavioral" in title:
            ax.text(0.5, 1.015, "(ground truth)", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=7.5, color="#444444",
                    fontstyle="italic")

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        line_color = "white" if cmap == "magma" else "#999999"
        line_alpha = 0.6 if cmap == "magma" else 0.4
        draw_boundary_lines(ax, block_boundaries, n, color=line_color,
                            lw=0.25, alpha=line_alpha)
        draw_category_sidebar(ax, block_boundaries, n, cat_colors, cat_to_idx,
                              side="left", width_frac=0.045, gap_frac=0.005)
        draw_category_sidebar(ax, block_boundaries, n, cat_colors, cat_to_idx,
                              side="bottom", width_frac=0.045, gap_frac=0.005)

    if colorbar_axes is not None:
        ax_cb1, ax_cb2 = colorbar_axes
        cb1 = plt.colorbar(ims[0], cax=ax_cb1)
        cb1.ax.tick_params(labelsize=7.5, length=3, width=0.5, pad=3)
        cb1.outline.set_linewidth(0.5)
        cb1.ax.yaxis.set_major_locator(mticker.FixedLocator([0, 0.5, 1.0]))

        if show_difference and len(ims) > 3:
            cb2 = plt.colorbar(ims[3], cax=ax_cb2)
            cb2.ax.tick_params(labelsize=5.5, length=2, width=0.4, pad=2)
            cb2.outline.set_linewidth(0.4)
            diff_tick = np.floor(diff_vlim * 10) / 10
            cb2.ax.yaxis.set_major_locator(
                mticker.FixedLocator([-diff_tick, 0, diff_tick]))


# ═══════════════════════════════════════════════════════════════════════════
# Scatter + histogram panel
# ═══════════════════════════════════════════════════════════════════════════

def plot_scatter_panel(ax_scatter, ax_hist, precomputed, super_config=None):
    """Draw category-averaged scatter and per-concept histogram.

    Panel B: Category-averaged scatter (8-class vs 1000-class per-concept rho).
    Panel C: Per-concept KDE of delta rho, with category overlays.

    If super_config is provided, groups by super-categories instead of the
    27 fine THINGS categories. super_config should be a dict with:
        - "fine_to_super": dict mapping fine category -> super category
        - "palette": dict mapping super category -> color
        - "kde_categories": list of super-categories to show in Panel C KDE
    """
    categories = precomputed["categories"]
    corr_clip8 = precomputed["corr_clip8"]
    corr_1k = precomputed["corr_1k"]
    diff = corr_clip8 - corr_1k

    # Map to super-categories if config provided
    if super_config is not None:
        fine_to_super = super_config["fine_to_super"]
        palette = super_config["palette"]
        kde_cats = super_config.get("kde_categories", [])
        categories = np.array([fine_to_super.get(c, "Other") for c in categories])
    else:
        palette = None
        kde_cats = []

    df = pd.DataFrame({
        "corr_clip8": corr_clip8,
        "corr_1k": corr_1k,
        "diff": diff,
        "category": categories,
    })

    cat_df = (df.groupby("category")
              .agg(mean_clip8=("corr_clip8", "mean"),
                   mean_1k=("corr_1k", "mean"),
                   mean_diff=("diff", "mean"),
                   n=("diff", "size"))
              .reset_index())
    cat_df = cat_df.sort_values("mean_diff", ascending=False).reset_index(
        drop=True)

    # ── Markers (one per category) ──
    ALL_MARKERS = [
        "o", "s", "^", "D", "v", "P", "*", "X", "p", "h",
        "<", ">", "d", "8", "H",
        (4, 1, 0), (5, 1, 0), (6, 1, 0), (6, 0, 0), (7, 0, 0),
        (8, 0, 0), (4, 0, 45), (4, 1, 45), (7, 1, 0), (3, 0, 180),
        (5, 0, 36), (8, 1, 0),
    ]
    cat_markers = {cat: ALL_MARKERS[i]
                   for i, cat in enumerate(cat_df["category"])}

    # ── Scatter (x = 8-class, y = 1000-class) ──
    pad = 0.06
    lims = [min(cat_df["mean_1k"].min(), cat_df["mean_clip8"].min()) - pad,
            max(cat_df["mean_1k"].max(), cat_df["mean_clip8"].max()) + pad]
    ax_scatter.plot(lims, lims, color="#cccccc", lw=0.8, ls="--", zorder=0.5)

    if palette is not None:
        # Super-category mode: color by palette, all dots equal
        for _, row in cat_df.iterrows():
            cat = row["category"]
            ax_scatter.scatter(
                row["mean_clip8"], row["mean_1k"],
                c=[palette.get(cat, "#bdbdbd")], s=140, marker=cat_markers[cat],
                alpha=0.92, edgecolors="#333333", linewidths=1.0, zorder=3)
    else:
        # Legacy 27-category mode with discrete advantage coloring
        DISC_COLORS = {
            "dark_orange": "#e65100", "light_orange": "#f4a261",
            "grey": "#c8c8c8", "light_green": "#a5d6a7", "dark_green": "#2e7d32",
        }
        def _discrete_color(val):
            if val < -0.3: return DISC_COLORS["dark_orange"]
            elif val < 0: return DISC_COLORS["light_orange"]
            elif val == 0: return DISC_COLORS["grey"]
            elif val <= 0.3: return DISC_COLORS["light_green"]
            else: return DISC_COLORS["dark_green"]

        for i, (_, row) in enumerate(cat_df.iterrows()):
            ax_scatter.scatter(
                row["mean_clip8"], row["mean_1k"],
                c=[_discrete_color(row["mean_diff"])], s=90,
                marker=cat_markers[row["category"]], alpha=0.92,
                edgecolors="white", linewidths=0.5, zorder=2)

    ax_scatter.set_xlim(lims)
    ax_scatter.set_ylim(lims)
    ax_scatter.set_xlabel(
        r"Per-category $\rho_s$ (8-class)", fontsize=11)
    ax_scatter.set_ylabel(
        r"Per-category $\rho_s$ (1000-class)", fontsize=11)
    ax_scatter.set_title("Per-Category Alignment", fontsize=12.5,
                          fontweight="bold", pad=10)
    ax_scatter.tick_params(axis="both", labelsize=9.5, length=4, width=0.8)
    ax_scatter.set_aspect("equal")
    sns.despine(ax=ax_scatter, offset=5)

    # Region annotations — above/below the diagonal
    ax_scatter.text(0.55, 0.95, "1K better",
                     transform=ax_scatter.transAxes,
                     ha="center", va="top", fontsize=10, color="#c62828",
                     fontweight="bold", alpha=0.55)
    ax_scatter.text(0.75, 0.06, "8-class better",
                     transform=ax_scatter.transAxes,
                     ha="center", va="bottom", fontsize=10, color="#2e7d32",
                     fontweight="bold", alpha=0.55)

    # ── Legend (flat list, no section headers) ──
    if palette is not None:
        # Sort by mean_diff descending (same order as cat_df)
        legend_elements = []
        for _, row in cat_df.iterrows():
            cat = row["category"]
            legend_elements.append(
                Line2D([0], [0], marker=cat_markers[cat], color="w",
                       markerfacecolor=palette.get(cat, "#bdbdbd"),
                       markersize=7, markeredgecolor="#222222",
                       markeredgewidth=0.8,
                       label=cat))
    else:
        legend_elements = []

    if legend_elements:
        leg = ax_scatter.legend(handles=legend_elements, fontsize=7.5, frameon=True,
                                 loc="upper left", handletextpad=0.4,
                                 framealpha=0.95, edgecolor="#bbbbbb",
                                 fancybox=False, borderpad=0.4,
                                 labelspacing=0.25, handlelength=1.2,
                                 bbox_to_anchor=(0.0, 1.0))
        leg.get_frame().set_linewidth(0.5)

    # ── KDE panel (overall + category overlays) ──
    from scipy.stats import gaussian_kde

    # Overall KDE
    x_grid = np.linspace(diff.min() - 0.15, diff.max() + 0.15, 500)
    kde_all = gaussian_kde(diff, bw_method="scott")
    y_all = kde_all(x_grid)

    # Split fill at zero: green (positive) and orange (negative)
    mask_pos = x_grid >= 0
    mask_neg = x_grid <= 0
    ax_hist.fill_between(x_grid[mask_pos], y_all[mask_pos], alpha=0.18,
                          color="#2e7d32", zorder=1)
    ax_hist.fill_between(x_grid[mask_neg], y_all[mask_neg], alpha=0.18,
                          color="#e65100", zorder=1)
    ax_hist.plot(x_grid, y_all, color="#444444", lw=2.0, zorder=2,
                  label="All concepts")

    # Zero line
    ax_hist.axvline(0, color="#555555", lw=0.7, ls="-", zorder=5)

    # Category-specific KDEs
    if palette is not None and kde_cats:
        kde_config = {cat: palette.get(cat, "#bdbdbd") for cat in kde_cats}
    else:
        # Legacy fine-category KDEs
        kde_config = {
            "animal": "#2e7d32", "food": "#00897b",
            "plant": "#8bc34a", "body part": "#c1121f",
        }

    # KDEs with sqrt-proportional scaling: heights scale by
    # sqrt(n_cat / n_total) to preserve ordering while keeping small
    # categories visible (same principle as bubble-chart area scaling).
    n_total = len(diff)
    for cat, color in kde_config.items():
        cat_mask = df["category"] == cat
        cat_diff = diff[cat_mask]
        if len(cat_diff) < 5:
            continue
        kde_cat = gaussian_kde(cat_diff, bw_method="scott")
        y_cat = kde_cat(x_grid)
        weight = np.sqrt(len(cat_diff) / n_total)
        y_cat_scaled = y_cat * weight
        ax_hist.plot(x_grid, y_cat_scaled, color=color, ls="-", lw=2.0,
                      zorder=3, label=f"{cat} (n={len(cat_diff)})")

    # Formatting
    ax_hist.set_xlabel(
        r"$\Delta\rho_s$ (8-class $-$ 1000-class)", fontsize=11)
    ax_hist.set_ylabel("")
    ax_hist.tick_params(axis="both", labelsize=9.5, length=4, width=0.8)
    ax_hist.yaxis.set_major_formatter(mticker.NullFormatter())
    ax_hist.yaxis.set_ticks([])
    sns.despine(ax=ax_hist, offset=5, left=True)

    # Headroom
    ymax = ax_hist.get_ylim()[1]
    ax_hist.set_ylim(top=ymax * 1.18)

    # Percentage annotations
    n_win = (diff > 0).sum()
    pct_win = 100 * n_win / len(diff)
    pct_lose = 100 - pct_win
    ax_hist.text(0.80, 0.95, f"{pct_win:.0f}%",
                  transform=ax_hist.transAxes, fontsize=14, va="top",
                  ha="center", color="#1b5e20", fontweight="bold")
    ax_hist.text(0.16, 0.95, f"{pct_lose:.0f}%",
                  transform=ax_hist.transAxes, fontsize=14, va="top",
                  ha="center", color="#c62828", fontweight="bold")

    # Legend
    leg_kde = ax_hist.legend(fontsize=8.5, frameon=True, loc="upper right",
                              framealpha=0.95, edgecolor="#cccccc",
                              fancybox=False, borderpad=0.4,
                              labelspacing=0.30, handlelength=1.8,
                              bbox_to_anchor=(1.0, 0.82))
    leg_kde.get_frame().set_linewidth(0.4)
