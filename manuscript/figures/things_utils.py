"""Shared THINGS plotting utilities for manuscript figures.

Extracted from fig5/figure5.py so that both fig4 (THINGS behavioral)
and fig5 (summary overview) can reuse the same data loading and panel
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
        "CLIP 8-class": rdm_clip8[np.ix_(sort_idx, sort_idx)],
        "1000-class": rdm_1k[np.ix_(sort_idx, sort_idx)],
    }

    # RSA scores
    rsa_scores = {}
    for key in ["CLIP 8-class", "1000-class"]:
        rsa_scores[key] = compute_rdm_correlation(
            torch.tensor(rdms_sorted[key]), torch.tensor(rdms_sorted["Behavioral"]),
            correlation="Spearman"
        )

    # Difference RDM
    ranked_behav = rank_transform(rdms_sorted["Behavioral"])
    diff_rdm = (
        np.abs(ranked_behav - rank_transform(rdms_sorted["1000-class"]))
        - np.abs(ranked_behav - rank_transform(rdms_sorted["CLIP 8-class"]))
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

    axes: list of 3 or 4 axes (Behavioral, CLIP 8-class, 1000-class, [Difference])
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
        ("CLIP 8-class", rdms_ranked["CLIP 8-class"], rsa_scores["CLIP 8-class"], "magma"),
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

def plot_scatter_panel(ax_scatter, ax_hist, precomputed):
    """Draw per-concept scatter and histogram using precomputed data."""
    categories = precomputed["categories"]
    corr_clip8 = precomputed["corr_clip8"]
    corr_1k = precomputed["corr_1k"]
    diff = corr_clip8 - corr_1k

    df = pd.DataFrame({
        "corr_clip8": corr_clip8,
        "corr_1k": corr_1k,
        "diff": diff,
        "category": categories,
    })

    cat_medians = (df[df["category"] != "Other"]
                   .groupby("category")["diff"].median()
                   .sort_values(ascending=False))

    buffer_threshold = 0.05
    positive_cats = cat_medians[cat_medians > 0].head(5).index.tolist()
    negative_cats = (cat_medians[cat_medians < -0.05]
                     .sort_values(ascending=True).index.tolist())

    cat_style = {}
    for i, cat in enumerate(positive_cats):
        cat_style[cat] = (GREEN_COLORS[i], GREEN_MARKERS[i])
    for i, cat in enumerate(negative_cats[:len(ORANGE_COLORS)]):
        cat_style[cat] = (ORANGE_COLORS[i], ORANGE_MARKERS[i])

    # Assign per-concept style
    colors, markers, is_highlighted = [], [], []
    for _, row in df.iterrows():
        if abs(row["diff"]) < buffer_threshold:
            colors.append("#d9d9d9")
            markers.append("o")
            is_highlighted.append(False)
        elif row["category"] in cat_style:
            c, m = cat_style[row["category"]]
            colors.append(c)
            markers.append(m)
            is_highlighted.append(True)
        else:
            colors.append("#cdcdcd")
            markers.append("o")
            is_highlighted.append(False)
    df["color"] = colors
    df["marker"] = markers
    df["highlighted"] = is_highlighted

    # ── Scatter (x = CLIP 8-class, y = 1000-class) ──
    lims = [min(df["corr_1k"].min(), df["corr_clip8"].min()) - 0.08,
            max(df["corr_1k"].max(), df["corr_clip8"].max()) + 0.05]
    xx = np.linspace(lims[0], lims[1], 200)
    # Buffer band around diagonal
    ax_scatter.fill_between(xx, xx - buffer_threshold, xx + buffer_threshold,
                             color="#f0f0f0", alpha=0.6, zorder=0, lw=0)
    ax_scatter.plot(lims, lims, color="#999999", lw=0.7, ls="--", zorder=0.5)

    grey_df = df[~df["highlighted"]]
    ax_scatter.scatter(grey_df["corr_clip8"], grey_df["corr_1k"],
                        c="#c8c8c8", s=18, marker="o",
                        alpha=0.35, edgecolors="none", rasterized=True, zorder=1)

    for cat in positive_cats + negative_cats:
        if cat not in cat_style:
            continue
        cat_mask = (df["category"] == cat) & df["highlighted"]
        subset = df[cat_mask]
        if subset.empty:
            continue
        c, m = cat_style[cat]
        ax_scatter.scatter(subset["corr_clip8"], subset["corr_1k"],
                            c=c, s=38, marker=m, alpha=0.85,
                            edgecolors="white", linewidths=0.5,
                            rasterized=True, zorder=2)

    ax_scatter.set_xlim(lims)
    ax_scatter.set_ylim(lims)
    ax_scatter.set_xlabel(r"Per-concept $\rho_s$ (CLIP 8-class)", fontsize=10.5)
    ax_scatter.set_ylabel(r"Per-concept $\rho_s$ (1000-class)", fontsize=10.5)
    ax_scatter.set_title("Per-Concept Alignment", fontsize=12,
                          fontweight="semibold", pad=10)
    ax_scatter.tick_params(axis="both", labelsize=9, length=4, width=0.8)
    ax_scatter.set_aspect("equal")
    sns.despine(ax=ax_scatter, offset=5)

    # Subtle region annotations
    ax_scatter.text(0.92, 0.07, "8-class better",
                     transform=ax_scatter.transAxes,
                     ha="right", va="bottom", fontsize=9.5, color="#1a7a3a",
                     fontstyle="italic", alpha=0.65)
    ax_scatter.text(0.05, 0.52, "1K better",
                     transform=ax_scatter.transAxes,
                     ha="left", va="top", fontsize=9.5, color="#c1121f",
                     fontstyle="italic", alpha=0.65)

    # Legend
    legend_elements = []
    legend_elements.append(Line2D([0], [0], marker="None", color="w",
                                   label="8-class advantage"))
    for i, cat in enumerate(positive_cats):
        med = cat_medians[cat]
        legend_elements.append(
            Line2D([0], [0], marker=GREEN_MARKERS[i], color="w",
                   markerfacecolor=GREEN_COLORS[i], markersize=7.5,
                   markeredgecolor="white", markeredgewidth=0.4,
                   label=f"  {short_cat_label(cat)} ({med:+.2f})"))
    legend_elements.append(Line2D([0], [0], marker="None", color="w",
                                   label="1K advantage"))
    for i, cat in enumerate(negative_cats[:len(ORANGE_COLORS)]):
        med = cat_medians[cat]
        legend_elements.append(
            Line2D([0], [0], marker=ORANGE_MARKERS[i], color="w",
                   markerfacecolor=ORANGE_COLORS[i], markersize=7.5,
                   markeredgecolor="white", markeredgewidth=0.4,
                   label=f"  {short_cat_label(cat)} ({med:+.2f})"))

    leg = ax_scatter.legend(handles=legend_elements, fontsize=8.5, frameon=True,
                             loc="upper left", handletextpad=0.4,
                             framealpha=0.95, edgecolor="#bbbbbb", fancybox=False,
                             borderpad=0.5, labelspacing=0.25,
                             handlelength=1.5, bbox_to_anchor=(0.0, 1.0))
    leg.get_frame().set_linewidth(0.5)
    for text in leg.get_texts():
        label = text.get_text()
        if label in ("8-class advantage", "1K advantage"):
            text.set_fontweight("bold")
            text.set_fontsize(9.0)

    # ── Histogram (works as standalone or inset) ──
    bins = np.linspace(diff.min() - 0.02, diff.max() + 0.02, 36)
    c_green_hist = "#1a7a3a"
    c_orange_hist = "#d95e1a"
    bin_colors = [c_green_hist if (b_lo + b_hi) / 2 > 0 else c_orange_hist
                  for b_lo, b_hi in zip(bins[:-1], bins[1:])]
    _, _, patches = ax_hist.hist(diff, bins=bins, edgecolor="white", linewidth=0.4)
    for patch, c in zip(patches, bin_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)

    ax_hist.axvspan(-buffer_threshold, buffer_threshold,
                     color="#f4f4f4", alpha=0.7, zorder=0, lw=0)
    ax_hist.axvline(0, color="#333333", lw=0.8, ls="-", zorder=3)

    # Detect if this is an inset (small axes) or standalone
    bbox = ax_hist.get_position()
    is_inset = (bbox.width < 0.2)  # heuristic: insets are small

    if is_inset:
        ax_hist.set_xlabel(r"$\Delta\rho_s$", fontsize=6, labelpad=2)
        ax_hist.set_ylabel("")
        ax_hist.tick_params(axis="both", labelsize=5, length=2, width=0.4, pad=1)
        ax_hist.yaxis.set_major_locator(mticker.MaxNLocator(3, integer=True))
        pct_fs = 8.5
    else:
        ax_hist.set_xlabel(r"$\Delta\rho_s$ (CLIP 8-class $-$ 1000-class)",
                           fontsize=10.5)
        ax_hist.set_ylabel("Count", fontsize=10.5)
        ax_hist.tick_params(axis="both", labelsize=9, length=4, width=0.8)
        pct_fs = 13

    sns.despine(ax=ax_hist, offset=2 if is_inset else 5)

    # Add headroom above tallest bar
    if not is_inset:
        ymax = ax_hist.get_ylim()[1]
        ax_hist.set_ylim(top=ymax * 1.12)

    n_win = (diff > 0).sum()
    pct_win = 100 * n_win / len(diff)
    pct_lose = 100 - pct_win
    ax_hist.text(0.78, 0.88, f"{pct_win:.0f}%",
                  transform=ax_hist.transAxes, fontsize=pct_fs, va="top",
                  ha="center", color=c_green_hist, fontweight="bold")
    ax_hist.text(0.18, 0.88, f"{pct_lose:.0f}%",
                  transform=ax_hist.transAxes, fontsize=pct_fs, va="top",
                  ha="center", color=c_orange_hist, fontweight="bold")
