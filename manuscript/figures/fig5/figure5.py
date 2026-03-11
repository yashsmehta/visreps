"""Figure 5: Behavioral Alignment (THINGS) — The Surprise.

Composite figure with four panel groups:
  B — Coarseness bar plots (CLIP-PCA, single panel)
  C — Category-annotated RDMs (Behavioral | CLIP 4-class | 1000-class | Difference)
  D — Per-concept scatter with category coloring + histogram
  E — Dual PCA reconstruction control

Layout (3 rows):
  Row 1: [B: bars] [E: reconstruction]
  Row 2: [C1-C4: RDMs with colorbars]
  Row 3: [D: scatter] [D: histogram]

Usage:
    python manuscript/figures/fig5/figure5.py
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.transforms as transforms
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import rankdata, spearmanr
import seaborn as sns

sys.path.insert(0, "plotters")
from plotter_utils import get_condition_summary

sys.path.insert(0, "manuscript/figures")
from fig_utils import COARSE_CFGS, setup_style

sys.path.insert(0, ".")
from experiments.reconstruction_analysis.plot_utils import (
    query_reconstruction_curve, query_untrained_baseline,
    aggregate_curve, plot_dual_curves,
)
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
N_COARSE = len(COARSE_CFGS)

CATEGORY_FILE = os.path.expanduser(
    "~/.cache/bonner-datasets/hebart2019.things/03_category-level/category27_manual.tsv"
)

COARSE_RECON_CONFIG = {"N/A": (64, "/data/ymehta3/vit_pca")}

OUTPUT_DIR = "manuscript/figures/fig5"

blues = sns.color_palette("Blues", n_colors=N_COARSE + 1)[1:]
UNTRAINED_COLOR = "#AAAAAA"
BASELINE_COLOR = "#FFA500"
BAR_WIDTH = 0.72

# Category palette (matches plot_rdms_categorized.py)
PALETTE_28 = [
    "#e31a1c", "#1f78b4", "#33a02c", "#6a3d9a", "#ff7f00",
    "#b15928", "#e377c2", "#1b9e77", "#d95f02", "#7570b3",
    "#a6d854", "#e6ab02", "#d4a76a", "#bcbd22", "#17becf",
    "#e7298a", "#9467bd", "#c44e52", "#2ca02c", "#8c564b",
    "#637939", "#fd8d3c", "#6baed6", "#9e9ac8", "#e7969c",
    "#3182bd", "#74c476", "#bdbdbd",
]

# Scatter colors (matches per_row_scatter_categories.py)
GREEN_COLORS = ["#0b6623", "#2e86ab", "#52b788", "#8ecae6", "#95d5b2"]
GREEN_MARKERS = ["o", "s", "^", "D", "v"]
ORANGE_COLORS = ["#c1121f", "#e07a28", "#d4a373", "#f4a261"]
ORANGE_MARKERS = ["o", "s", "^", "D"]


# ═══════════════════════════════════════════════════════════════════════════
# Shared data computation (done once, passed to panels)
# ═══════════════════════════════════════════════════════════════════════════

def compute_things_data(data):
    """Compute all RDMs, per-row correlations, and category data once.

    Returns a dict with all precomputed results needed by panels C and D.
    """
    print("  Computing RDMs...")
    rdm_behav = compute_rdm(torch.tensor(data["embeddings"], dtype=torch.float32)).numpy()
    rdm_clip4 = compute_rdm(torch.tensor(data["clip4_acts"], dtype=torch.float32)).numpy()
    rdm_1k = compute_rdm(torch.tensor(data["thousand_acts"], dtype=torch.float32)).numpy()

    print("  Loading categories and sorting...")
    categories = load_categories()
    sort_idx, block_boundaries, unique_cats = build_category_sort_order(
        categories, rdm_behav
    )

    # Sorted RDMs
    rdms_sorted = {
        "Behavioral": rdm_behav[np.ix_(sort_idx, sort_idx)],
        "CLIP 4-class": rdm_clip4[np.ix_(sort_idx, sort_idx)],
        "1000-class": rdm_1k[np.ix_(sort_idx, sort_idx)],
    }

    # RSA scores
    rsa_scores = {}
    for key in ["CLIP 4-class", "1000-class"]:
        rsa_scores[key] = compute_rdm_correlation(
            torch.tensor(rdms_sorted[key]), torch.tensor(rdms_sorted["Behavioral"]),
            correlation="Spearman"
        )

    # Difference RDM
    ranked_behav = rank_transform(rdms_sorted["Behavioral"])
    diff_rdm = (
        np.abs(ranked_behav - rank_transform(rdms_sorted["1000-class"]))
        - np.abs(ranked_behav - rank_transform(rdms_sorted["CLIP 4-class"]))
    )

    # Rank-transform for display
    rdms_ranked = {key: rank_transform(rdm) for key, rdm in rdms_sorted.items()}

    # Per-row correlations (for scatter panel)
    print("  Computing per-row correlations...")
    corr_clip4 = per_row_correlations(rdm_clip4, rdm_behav)
    corr_1k = per_row_correlations(rdm_1k, rdm_behav)

    return {
        "categories": categories,
        "sort_idx": sort_idx,
        "block_boundaries": block_boundaries,
        "unique_cats": unique_cats,
        "rdms_ranked": rdms_ranked,
        "diff_rdm": diff_rdm,
        "rsa_scores": rsa_scores,
        "corr_clip4": corr_clip4,
        "corr_1k": corr_1k,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Panel B: Coarseness bar plot
# ═══════════════════════════════════════════════════════════════════════════

def _draw_fancy_bar(ax, x, height, color, hatch=""):
    ax.bar(x, height, width=BAR_WIDTH, color=color, edgecolor="black",
           linewidth=0.8, hatch=hatch, zorder=3)


def _draw_break_marks(ax, x):
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    spine_y, dy, dx, gap = -0.022, 0.028, 0.20, 0.13
    ax.plot([x - gap - dx - 0.1, x + gap + dx + 0.1], [spine_y, spine_y],
            color="white", linewidth=5, transform=trans, clip_on=False, zorder=9)
    for offset in [-gap, gap]:
        ax.plot([x + offset - dx, x + offset + dx], [spine_y - dy, spine_y + dy],
                color="black", linewidth=1.8, clip_on=False, zorder=10,
                transform=trans)


def plot_bars_panel(ax, pca_model="clip"):
    """Draw THINGS coarseness bar plot on the given axis."""
    folder = f"pca_labels_{pca_model}"
    nd = "things-behavior"
    region = "N/A"

    un = get_condition_summary(nd, region, "imagenet1k", 1000,
                               "spearman", epoch=0, analysis="rsa")
    has_untrained = not np.isnan(un["mean"])

    if has_untrained:
        X_COARSE = np.arange(1.5, 1.5 + N_COARSE)
        X_UNTRAINED = 0.0
    else:
        X_COARSE = np.arange(N_COARSE, dtype=float)
        X_UNTRAINED = None
    X_BASELINE = X_COARSE[-1] + 2
    BREAK_X = (X_COARSE[-1] + X_BASELINE) / 2

    all_means, all_ci_lo, all_ci_hi = [], [], []
    all_x, all_colors, all_hatches, all_labels = [], [], [], []

    if has_untrained:
        all_means.append(un["mean"])
        all_ci_lo.append(un["ci_low"])
        all_ci_hi.append(un["ci_high"])
        all_x.append(X_UNTRAINED)
        all_colors.append(UNTRAINED_COLOR)
        all_hatches.append("")
        all_labels.append("Untrained")

    for i, cfg in enumerate(COARSE_CFGS):
        s = get_condition_summary(nd, region, folder, cfg,
                                  "spearman", epoch=20, analysis="rsa")
        all_means.append(s["mean"])
        all_ci_lo.append(s["ci_low"])
        all_ci_hi.append(s["ci_high"])
        all_x.append(X_COARSE[i])
        all_colors.append(blues[i])
        all_hatches.append("/")
        all_labels.append(str(cfg))

    bl = get_condition_summary(nd, region, "imagenet1k", 1000,
                               "spearman", epoch=20, analysis="rsa")
    all_means.append(bl["mean"])
    all_ci_lo.append(bl["ci_low"])
    all_ci_hi.append(bl["ci_high"])
    all_x.append(X_BASELINE)
    all_colors.append(BASELINE_COLOR)
    all_hatches.append("")
    all_labels.append("1000")

    all_means = np.array(all_means)
    all_ci_lo = np.array(all_ci_lo)
    all_ci_hi = np.array(all_ci_hi)
    all_x = np.array(all_x)
    err_lo = all_means - all_ci_lo
    err_hi = all_ci_hi - all_means

    valid = ~np.isnan(all_means)
    if valid.any():
        vlo = np.nanmin(all_ci_lo[~np.isnan(all_ci_lo)]) if (~np.isnan(all_ci_lo)).any() else np.nanmin(all_means)
        vhi = np.nanmax(all_ci_hi[~np.isnan(all_ci_hi)]) if (~np.isnan(all_ci_hi)).any() else np.nanmax(all_means)
    else:
        vlo, vhi = 0, 0.1
    dr = max(vhi - vlo, 0.01)
    y_bottom = max(0, vlo - 0.20 * dr)
    y_top = vhi + 0.20 * dr

    for k in range(len(all_x)):
        if not np.isnan(all_means[k]):
            _draw_fancy_bar(ax, all_x[k], all_means[k], all_colors[k], all_hatches[k])

    for k in range(len(all_x)):
        if (not np.isnan(err_lo[k]) and not np.isnan(err_hi[k])
                and err_lo[k] >= 0 and err_hi[k] >= 0
                and (err_lo[k] > 0 or err_hi[k] > 0)):
            ax.errorbar(all_x[k], all_means[k],
                        yerr=[[err_lo[k]], [err_hi[k]]],
                        fmt="none", ecolor="black", elinewidth=1.0,
                        capsize=4, capthick=1.0, zorder=5)

    _draw_break_marks(ax, BREAK_X)

    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, fontsize=8, ha="center")
    ax.tick_params(axis="x", direction="out", bottom=False)
    ax.tick_params(axis="y", which="major", direction="out", labelsize=9)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    first_x = X_UNTRAINED if has_untrained else all_x[0]
    ax.set_xlim(first_x - 0.6, X_BASELINE + 0.7)
    ax.set_ylim(y_bottom, y_top)
    ax.set_xlabel("Number of Classes", fontsize=9, labelpad=4)
    ax.set_ylabel(r"Spearman $\rho$", fontsize=9, labelpad=4)
    ax.set_title("THINGS Behavioral Alignment\n(CLIP-PCA Labels)", fontsize=10,
                 fontweight="semibold", pad=6)
    sns.despine(ax=ax, right=True, top=True, offset=4)


# ═══════════════════════════════════════════════════════════════════════════
# Panel C: Category-annotated RDMs
# ═══════════════════════════════════════════════════════════════════════════

def plot_rdm_panels(axes, ax_cb1, ax_cb2, precomputed):
    """Draw 4 RDM panels using precomputed data."""
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
        ("CLIP 4-class", rdms_ranked["CLIP 4-class"], rsa_scores["CLIP 4-class"], "magma"),
        ("1000-class", rdms_ranked["1000-class"], rsa_scores["1000-class"], "magma"),
        ("Difference", diff_rdm, None, "RdBu_r"),
    ]

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
        ax.set_title(title, fontsize=8, fontweight="bold", pad=8)
        if rsa is not None:
            ax.text(0.5, 1.01, f"$\\rho_s$ = {rsa:.3f}", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=6, color="#555555")
        elif "Behavioral" in title:
            ax.text(0.5, 1.01, "(ground truth)", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=6, color="#555555",
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
                              side="left", width_frac=0.026, gap_frac=0.005)
        draw_category_sidebar(ax, block_boundaries, n, cat_colors, cat_to_idx,
                              side="bottom", width_frac=0.026, gap_frac=0.005)

    cb1 = plt.colorbar(ims[0], cax=ax_cb1)
    cb1.ax.tick_params(labelsize=5.5, length=2, width=0.4, pad=2)
    cb1.outline.set_linewidth(0.4)
    cb1.ax.yaxis.set_major_locator(mticker.FixedLocator([0, 0.5, 1.0]))

    cb2 = plt.colorbar(ims[3], cax=ax_cb2)
    cb2.ax.tick_params(labelsize=5.5, length=2, width=0.4, pad=2)
    cb2.outline.set_linewidth(0.4)
    diff_tick = np.floor(diff_vlim * 10) / 10
    cb2.ax.yaxis.set_major_locator(mticker.FixedLocator([-diff_tick, 0, diff_tick]))


# ═══════════════════════════════════════════════════════════════════════════
# Panel D: Per-concept scatter
# ═══════════════════════════════════════════════════════════════════════════

def plot_scatter_panel(ax_scatter, ax_hist, precomputed):
    """Draw per-concept scatter and histogram using precomputed data."""
    categories = precomputed["categories"]
    corr_clip4 = precomputed["corr_clip4"]
    corr_1k = precomputed["corr_1k"]
    diff = corr_clip4 - corr_1k

    df = pd.DataFrame({
        "corr_clip4": corr_clip4,
        "corr_1k": corr_1k,
        "diff": diff,
        "category": categories,
    })

    cat_medians = (df[df["category"] != "Other"]
                   .groupby("category")["diff"].median()
                   .sort_values(ascending=False))

    buffer_threshold = 0.05
    positive_cats = cat_medians[cat_medians > 0].head(5).index.tolist()
    negative_cats = cat_medians[cat_medians < -0.05].sort_values(ascending=True).index.tolist()

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

    # ── Scatter ──
    lims = [min(df["corr_1k"].min(), df["corr_clip4"].min()) - 0.08,
            max(df["corr_1k"].max(), df["corr_clip4"].max()) + 0.05]
    xx = np.linspace(lims[0], lims[1], 200)
    ax_scatter.fill_between(xx, xx - buffer_threshold, xx + buffer_threshold,
                             color="#f4f4f4", alpha=0.8, zorder=0, lw=0)
    ax_scatter.plot(lims, lims, color="#b0b0b0", lw=0.6, ls="--", zorder=0.5)

    grey_df = df[~df["highlighted"]]
    ax_scatter.scatter(grey_df["corr_1k"], grey_df["corr_clip4"],
                        c=grey_df["color"].values, s=6, marker="o",
                        alpha=0.18, edgecolors="none", rasterized=True, zorder=1)

    for cat in positive_cats + negative_cats:
        if cat not in cat_style:
            continue
        cat_mask = (df["category"] == cat) & df["highlighted"]
        subset = df[cat_mask]
        if subset.empty:
            continue
        c, m = cat_style[cat]
        ax_scatter.scatter(subset["corr_1k"], subset["corr_clip4"],
                            c=c, s=14, marker=m, alpha=0.65,
                            edgecolors="white", linewidths=0.15,
                            rasterized=True, zorder=2)

    ax_scatter.set_xlim(lims)
    ax_scatter.set_ylim(lims)
    ax_scatter.set_xlabel(r"Per-concept $\rho_s$ (1000-class)")
    ax_scatter.set_ylabel(r"Per-concept $\rho_s$ (CLIP 4-class)")
    ax_scatter.set_title("Per-Concept Alignment", fontsize=10,
                          fontweight="semibold", pad=6)
    ax_scatter.set_aspect("equal")
    sns.despine(ax=ax_scatter, offset=4)

    # Legend
    legend_elements = []
    legend_elements.append(Line2D([0], [0], marker="None", color="w",
                                   label="4-class advantage"))
    for i, cat in enumerate(positive_cats):
        med = cat_medians[cat]
        legend_elements.append(
            Line2D([0], [0], marker=GREEN_MARKERS[i], color="w",
                   markerfacecolor=GREEN_COLORS[i], markersize=5,
                   markeredgecolor="none",
                   label=f"  {short_cat_label(cat)} ({med:+.2f})"))
    legend_elements.append(Line2D([0], [0], marker="None", color="w", label=" "))
    legend_elements.append(Line2D([0], [0], marker="None", color="w",
                                   label="1K advantage"))
    for i, cat in enumerate(negative_cats[:len(ORANGE_COLORS)]):
        med = cat_medians[cat]
        legend_elements.append(
            Line2D([0], [0], marker=ORANGE_MARKERS[i], color="w",
                   markerfacecolor=ORANGE_COLORS[i], markersize=5,
                   markeredgecolor="none",
                   label=f"  {short_cat_label(cat)} ({med:+.2f})"))

    leg = ax_scatter.legend(handles=legend_elements, fontsize=5.5, frameon=True,
                             loc="lower right", handletextpad=0.3,
                             framealpha=0.95, edgecolor="#e0e0e0", fancybox=True,
                             borderpad=0.5, labelspacing=0.22)
    leg.get_frame().set_linewidth(0.3)
    for text in leg.get_texts():
        label = text.get_text()
        if label in ("4-class advantage", "1K advantage"):
            text.set_fontweight("bold")
            text.set_fontsize(6)

    # ── Histogram ──
    bins = np.linspace(diff.min() - 0.02, diff.max() + 0.02, 48)
    c_green_hist = "#1a8a42"
    c_orange_hist = "#d95e1a"
    bin_colors = [c_green_hist if (b_lo + b_hi) / 2 > 0 else c_orange_hist
                  for b_lo, b_hi in zip(bins[:-1], bins[1:])]
    _, _, patches = ax_hist.hist(diff, bins=bins, edgecolor="white", linewidth=0.3)
    for patch, c in zip(patches, bin_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.70)

    ax_hist.axvspan(-buffer_threshold, buffer_threshold,
                     color="#f4f4f4", alpha=0.7, zorder=0, lw=0)
    ax_hist.axvline(0, color="#b0b0b0", lw=0.5, ls="--", zorder=3)
    med_val = np.median(diff)
    ax_hist.axvline(med_val, color="#333333", lw=1.0, ls="-", zorder=3)
    ax_hist.annotate(f"Median = {med_val:.3f}", xy=(med_val, 0.96),
                      xycoords=ax_hist.get_xaxis_transform(),
                      xytext=(8, 0), textcoords="offset points",
                      fontsize=6, va="top", ha="left", color="#555555",
                      fontstyle="italic")

    ax_hist.set_xlabel(r"$\Delta\rho_s$ (CLIP 4-class $-$ 1000-class)")
    ax_hist.set_ylabel("Count")
    sns.despine(ax=ax_hist, offset=4)

    n_win = (diff > 0).sum()
    pct = 100 * n_win / len(diff)
    ax_hist.text(0.97, 0.92,
                  f"4-class > 1K: {n_win}/{len(diff)} ({pct:.0f}%)",
                  transform=ax_hist.transAxes, fontsize=6.5, va="top", ha="right",
                  color=c_green_hist, fontweight="bold")


# ═══════════════════════════════════════════════════════════════════════════
# Main figure assembly
# ═══════════════════════════════════════════════════════════════════════════

def main():
    setup_style()
    plt.rcParams["axes.linewidth"] = 0.8

    # Load THINGS data once, compute all derived quantities once
    print("Loading THINGS data...")
    data = load_data()
    print("Computing derived data (RDMs, correlations)...")
    precomputed = compute_things_data(data)

    # ── Layout ──
    fig = plt.figure(figsize=(14, 14.5))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.32, wspace=0.30,
                           height_ratios=[1, 1.1, 1.0])

    # Row 0: B + E
    ax_b = fig.add_subplot(gs[0, 0])
    ax_e = fig.add_subplot(gs[0, 1])

    # Row 1: RDMs
    gs_rdm = gridspec.GridSpecFromSubplotSpec(
        1, 7, subplot_spec=gs[1, :],
        width_ratios=[1, 1, 1, 0.04, 0.08, 1, 0.04],
        wspace=0.05,
    )
    ax_rdm = [fig.add_subplot(gs_rdm[0, i]) for i in [0, 1, 2, 5]]
    ax_cb1 = fig.add_subplot(gs_rdm[0, 3])
    ax_cb2 = fig.add_subplot(gs_rdm[0, 6])

    # Row 2: Scatter + Histogram
    ax_scatter = fig.add_subplot(gs[2, 0])
    ax_hist = fig.add_subplot(gs[2, 1])

    # ── Draw panels ──
    print("Drawing Panel B (bars)...")
    plot_bars_panel(ax_b, pca_model="clip")

    print("Drawing Panel E (reconstruction)...")
    fine_df = query_reconstruction_curve("things-behavior", "N/A")
    fine_agg = aggregate_curve(fine_df)
    cfg_id, checkpoint_dir = COARSE_RECON_CONFIG["N/A"]
    coarse_df = query_reconstruction_curve("things-behavior", "N/A",
                                            cfg_id=cfg_id,
                                            checkpoint_dir=checkpoint_dir)
    coarse_agg = aggregate_curve(coarse_df)
    untrained = query_untrained_baseline("things-behavior", "N/A")
    if fine_agg.empty and coarse_agg.empty:
        ax_e.text(0.5, 0.5, "No data", ha="center", va="center",
                   transform=ax_e.transAxes, fontsize=12, color="#888")
    else:
        plot_dual_curves(ax_e, fine_agg, coarse_agg, untrained,
                         title="Reconstruction Control",
                         coarse_label="Coarse model (ViT-PCA 64-way)")
        ax_e.legend(fontsize=7, loc="lower right", frameon=True,
                     edgecolor="#cccccc", fancybox=False, handletextpad=0.3)

    print("Drawing Panel C (RDMs)...")
    plot_rdm_panels(ax_rdm, ax_cb1, ax_cb2, precomputed)

    print("Drawing Panel D (scatter)...")
    plot_scatter_panel(ax_scatter, ax_hist, precomputed)

    # ── Panel labels ──
    for ax, label in zip([ax_b, ax_e, ax_rdm[0], ax_scatter],
                          ["B", "E", "C", "D"]):
        ax.text(-0.10, 1.08, label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="left")

    # ── Save ──
    fig.suptitle("Figure 5: Behavioral Alignment (THINGS)",
                 fontsize=14, fontweight="bold", y=0.99)
    out = f"{OUTPUT_DIR}/figure5.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
