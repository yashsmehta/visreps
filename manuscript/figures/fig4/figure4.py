"""Figure 4: THINGS Behavioral Alignment.

Layout:
  Row 0 (top): [Comparison | Coarseness | Scatter | Histogram]
  Row 1 (bottom): [3 RDMs side by side + colorbar] spanning full width

Usage:
    python manuscript/figures/fig4/figure4.py
"""

import os
import sys
import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns

sys.path.insert(0, "plotters")
from plotter_utils import get_condition_summary

sys.path.insert(0, "manuscript/figures")
from fig_utils import (
    DB_PATH,
    COARSE_CFGS, BREAK_1K_POS,
    UNTRAINED_LINE_STYLE, MARKER_SIZE, EDGE_COLOR, EDGE_WIDTH,
    setup_style, compute_jitter,
    format_coarseness_axes,
)
from things_utils import compute_things_data, plot_rdm_panels, plot_scatter_panel

# ── Config ────────────────────────────────────────────────────────────────
OUTPUT_DIR = "manuscript/figures/fig4"

# ── Figure 4 color scheme ──
ARCHITECTURES = [
    ("alexnet", "pca_labels_alexnet", "AlexNet"),
    ("clip",    "pca_labels_clip",    "CLIP"),
    ("pixels",  "pca_labels_pixels",  "Pixels"),
]
ARCH_STYLE = {
    "alexnet": {"color": "#6baed6", "marker": "o"},   # medium blue
    "clip":    {"color": "#08519c", "marker": "s"},    # dark blue
    "pixels":  {"color": "#c0a898", "marker": "v"},    # muted tan
}
BASELINE_1K_COLOR = "#d4822e"  # warm amber (darkened for white text)
COARSE_BAR_COLOR = "#08519c"   # dark blue (CLIP)

# ── Pretrained model comparison config ────────────────────────────────────
# Groups: (display_name, db_model_name, architecture)
PRETRAINED_GROUPS = {
    "Supervised": [
        ("AlexNet",    "AlexNet",         "cnn"),
        ("VGG-16",     "VGG16",           "cnn"),
        ("ResNet-50",  "ResNet50",        "cnn"),
        ("ConvNeXt",   "ConvNeXt_Base",   "cnn"),
        ("ViT-B/16",   "ViTBase",         "vit"),
    ],
    "Self-supervised": [
        ("DINOv1",     "DINOv1_ResNet50", "cnn"),
        ("DINOv2",     "DINOv2_ViT_B14",  "vit"),
        ("DINOv3",     "DINOv3_ViT_L16",  "vit"),
    ],
    "Vision-language": [
        ("CLIP-B/32",  "CLIP_ViT_B32",    "vit"),
        ("CLIP-L/14",  "CLIP_ViT_L14",    "vit"),
    ],
}
GROUP_COLORS = {
    "Supervised":      "#4a8c6f",   # sage green
    "Self-supervised": "#6b5b95",   # muted lavender
    "Vision-language": "#c26b3a",   # burnt sienna
}
ARCH_MARKERS = {"cnn": "p", "vit": "*"}


# ── Comparison panel ─────────────────────────────────────────────────────

DATA_EFF_CSV = os.path.join("experiments", "coarse_grain_benefits",
                            "data_efficiency", "data_efficiency.csv")

# Data scales for the bar groups (label, CSV dataset key)
DATA_SCALES = [
    ("5",     "imagenet-mini-5"),
    ("10",    "imagenet-mini-10"),
    ("50",    "imagenet-mini-50"),
    ("~1300", "imagenet-full"),
]


def _fetch_comparison_data():
    """Fetch all data needed for the comparison panel."""
    # ── Data efficiency bars (best epoch per dataset × condition) ──
    eff_df = pd.read_csv(DATA_EFF_CSV)
    best_eff = eff_df.loc[
        eff_df.groupby(["dataset", "condition"])["score"].idxmax()
    ].reset_index(drop=True)

    # ── Pretrained model scatter ──
    conn = sqlite3.connect(DB_PATH)
    pretrained_df = pd.read_sql("""
        SELECT model_name, score, ci_low, ci_high
        FROM results
        WHERE neural_dataset = 'things-behavior'
          AND analysis = 'rsa'
          AND compare_method = 'spearman'
          AND cfg_id = 'pretrained'
    """, conn)
    conn.close()
    scores = pretrained_df.set_index("model_name")

    all_points = []
    for group_name, models in PRETRAINED_GROUPS.items():
        color = GROUP_COLORS[group_name]
        for display, db_name, arch in models:
            if db_name not in scores.index:
                continue
            row = scores.loc[db_name]
            all_points.append({
                "display": display, "score": row["score"],
                "ci_low": row["ci_low"], "ci_high": row["ci_high"],
                "color": color, "marker": ARCH_MARKERS[arch],
                "group": group_name,
            })
    all_points.sort(key=lambda p: p["score"], reverse=True)

    return best_eff, all_points


def _draw_neurips_bar(ax, x, height, width, color, y_base=0.25,
                      edgecolor="black", linewidth=0.8, zorder=3):
    """Draw a bar with rounded top corners in NeurIPS style."""
    import matplotlib.patches as mpatches
    # Start slightly below y_base so bottom rounding is hidden by axis
    bar_bottom = y_base - 0.015
    bar_height = height - bar_bottom
    rect = mpatches.FancyBboxPatch(
        (x - width / 2, bar_bottom), width, bar_height,
        boxstyle=mpatches.BoxStyle("Round", pad=0.02, rounding_size=0.08),
        facecolor=color, edgecolor=edgecolor,
        linewidth=linewidth, mutation_aspect=0.05, zorder=zorder,
    )
    ax.add_patch(rect)


def plot_comparison_panel(ax, standalone=False):
    """Panel A: Data-efficiency bars (8-class vs 1000-class at 4 data scales)
    + grouped scatter for pretrained models.

    If standalone=True, uses larger fonts and more spacing.
    """
    best_eff, all_points = _fetch_comparison_data()

    s = 1.0 if standalone else 0.72

    # ── Layout x-positions ──
    # 4 bar groups (5, 10, 50, full) then pretrained scatter
    n_groups = len(DATA_SCALES)
    group_spacing = 1.6
    bar_w = 0.5
    bar_gap = 0.06
    group_centers = [i * group_spacing for i in range(n_groups)]

    # Pretrained scatter starts after the last bar group
    scatter_start = group_centers[-1] + group_spacing * 1.1
    group_positions = {
        "Supervised":      scatter_start,
        "Self-supervised": scatter_start + 1.2,
        "Vision-language": scatter_start + 2.4,
    }
    jitter_spread = 0.22

    # ── Draw paired bars at each data scale ──
    conditions = [(8, COARSE_BAR_COLOR), (1000, BASELINE_1K_COLOR)]
    for gi, (label, ds_key) in enumerate(DATA_SCALES):
        cx = group_centers[gi]
        for ci, (cond, color) in enumerate(conditions):
            offset = (ci - 0.5) * (bar_w + bar_gap)
            x = cx + offset
            row = best_eff[(best_eff["dataset"] == ds_key) &
                           (best_eff["condition"] == cond)]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            score = row["score"]
            _draw_neurips_bar(ax, x, score, bar_w, color,
                              y_base=0.2, edgecolor="#333333", linewidth=0.7)
            err_lo = score - row["ci_low"]
            err_hi = row["ci_high"] - score
            ax.errorbar(x, score, yerr=[[err_lo], [err_hi]], fmt="none",
                        ecolor="#333333", capsize=3.0 * s, capthick=0.7,
                        elinewidth=0.7, zorder=4)

    # "(full)" sub-label beneath the last group
    full_cx = group_centers[-1]
    ax.annotate("(full)", xy=(full_cx, 0), xycoords=("data", "axes fraction"),
                xytext=(0, -22), textcoords="offset points",
                ha="center", va="top", fontsize=6.5 * s, color="#555555")

    # ── Dashed reference line from full-data coarse bar into scatter region ──
    full_coarse = best_eff[(best_eff["dataset"] == "imagenet-full") &
                           (best_eff["condition"] == 8)]
    if len(full_coarse) > 0:
        ref_score = full_coarse.iloc[0]["score"]
        x_ref_start = group_centers[-1] + bar_w
        x_ref_end = list(group_positions.values())[-1] + 0.6
        ax.plot([x_ref_start, x_ref_end], [ref_score, ref_score],
                color=COARSE_BAR_COLOR, linestyle=(0, (5, 3)),
                linewidth=0.7, alpha=0.40, zorder=1)

    # Subtle vertical separator between bars and scatter
    sep_x = (group_centers[-1] + scatter_start) / 2
    ax.axvline(sep_x, color="#e8e8e8", linewidth=0.5, linestyle="-",
               ymin=0.0, ymax=1.0, zorder=0)

    # ── Draw pretrained points grouped by paradigm ──
    pt_size_base = 95 * s * s
    pt_size_star = 130 * s * s
    for pt in all_points:
        gx = group_positions[pt["group"]]
        group_pts = [p for p in all_points if p["group"] == pt["group"]]
        idx = group_pts.index(pt)
        n = len(group_pts)
        if n == 1:
            x_jit = gx
        else:
            x_jit = gx + np.linspace(-jitter_spread, jitter_spread, n)[idx]
        pt["x_plot"] = x_jit

        ax.plot([x_jit, x_jit], [pt["ci_low"], pt["ci_high"]],
                color=pt["color"], linewidth=1.0, alpha=0.35, zorder=4,
                solid_capstyle="round")
        sz = pt_size_star if pt["marker"] == "*" else pt_size_base
        ax.scatter(x_jit, pt["score"], marker=pt["marker"], c=pt["color"],
                   s=sz, edgecolors="white", linewidths=0.6, zorder=5)

    # ── Model name labels ──
    fs_model = 7.9 * s
    x_offset = 0.18 * s
    for pt in all_points:
        ax.text(pt["x_plot"] + x_offset, pt["score"], pt["display"],
                ha="left", va="center", fontsize=fs_model, color="#2a2a2a")

    # ── Axis formatting ──
    xlim_right = list(group_positions.values())[-1] + (1.5 if standalone else 1.1)
    ax.set_xlim(-0.6, xlim_right)
    ax.set_ylim(0.2, 0.65)
    ax.set_ylabel(r"Spearman $\rho$", fontsize=10 * s, labelpad=6)
    ax.set_title("THINGS Behavioral Similarity",
                 fontsize=11 * s, fontweight="semibold", pad=10)

    # X-ticks: bar group labels + pretrained group labels
    all_xticks = group_centers + [group_positions[g] for g in group_positions]
    all_xlabels = [lbl for lbl, _ in DATA_SCALES] + [
        "Supervised", "Self-\nSupervised", "Vision-\nLanguage"]
    ax.set_xticks(all_xticks)
    ax.set_xticklabels(all_xlabels, fontsize=7.5 * s)

    # x-axis label for the bar region
    bar_mid = (group_centers[0] + group_centers[-1]) / 2
    ax.annotate("Images per class", xy=(bar_mid, 0),
                xycoords=("data", "axes fraction"),
                xytext=(0, -35), textcoords="offset points",
                ha="center", va="top", fontsize=8.5 * s, color="#333333")

    # Subtle horizontal grid
    ax.yaxis.grid(True, which="major", color="#EBEBEB", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Y-axis
    from matplotlib.ticker import MultipleLocator, FuncFormatter
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:.1f}"))
    ax.tick_params(axis="y", which="major", direction="out", length=5,
                   width=1.2, labelsize=9 * s)
    ax.tick_params(axis="y", which="minor", direction="out", length=3,
                   width=0.8)
    ax.tick_params(axis="x", which="major", length=4, width=1.0, direction="out")

    sns.despine(ax=ax, right=True, top=True, offset=5)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_linewidth(1.2)

    # ── Legend ──
    import matplotlib.patches as mpatches
    leg_handles = [
        mpatches.Patch(facecolor=COARSE_BAR_COLOR, edgecolor="#333333",
                       linewidth=0.6, label="Coarse (8-class)"),
        mpatches.Patch(facecolor=BASELINE_1K_COLOR, edgecolor="#333333",
                       linewidth=0.6, label="Fine (1000-class)"),
        Line2D([], [], marker="p", color="none", markerfacecolor="#777777",
               markeredgecolor="white", markeredgewidth=0.5,
               markersize=8, label="CNN"),
        Line2D([], [], marker="*", color="none", markerfacecolor="#777777",
               markeredgecolor="white", markeredgewidth=0.4,
               markersize=10, label="ViT"),
    ]
    leg = ax.legend(handles=leg_handles, fontsize=7.5 * s, frameon=True,
                    loc="upper left", edgecolor="#dddddd", fancybox=False,
                    framealpha=0.95, handletextpad=0.5, borderpad=0.4,
                    labelspacing=0.3, ncol=2)
    leg.get_frame().set_linewidth(0.3)


def save_standalone_comparison():
    """Save a standalone version of the comparison panel."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    plot_comparison_panel(ax, standalone=True)
    plt.tight_layout(pad=0.8)
    out = f"{OUTPUT_DIR}/model_comparison.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white",
                edgecolor="none")
    print(f"Saved standalone -> {out}")
    plt.close(fig)


# ── Coarseness data fetching ─────────────────────────────────────────────

def fetch_things_arch_data(folder):
    means, ci_lo, ci_hi = [], [], []
    for cfg in COARSE_CFGS:
        s = get_condition_summary("things-behavior", "N/A", folder, cfg,
                                  "spearman", epoch=20, analysis="rsa")
        means.append(s["mean"])
        ci_lo.append(s["ci_low"])
        ci_hi.append(s["ci_high"])
    return np.array(means), np.array(ci_lo), np.array(ci_hi)


def plot_coarseness_raw(ax):
    """Plot raw Spearman rho coarseness for THINGS behavioral."""
    # 1000-way baseline
    bl = get_condition_summary("things-behavior", "N/A", "imagenet1k", 1000,
                               "spearman", epoch=20, analysis="rsa")
    bl_mean = bl["mean"]

    # Untrained baseline
    un = get_condition_summary("things-behavior", "N/A", "imagenet1k", 1000,
                               "spearman", epoch=0, analysis="rsa")
    if not np.isnan(un["mean"]):
        ax.axhline(un["mean"], **UNTRAINED_LINE_STYLE, zorder=1)

    # 1000-way horizontal reference line + diamond
    if not np.isnan(bl_mean):
        ax.axhline(bl_mean, color=BASELINE_1K_COLOR, linestyle="-",
                    linewidth=0.6, alpha=0.35, zorder=2)
        bl_err_lo = max(bl["mean"] - bl["ci_low"], 0) if not np.isnan(bl["ci_low"]) else 0
        bl_err_hi = max(bl["ci_high"] - bl["mean"], 0) if not np.isnan(bl["ci_high"]) else 0
        ax.errorbar(BREAK_1K_POS, bl_mean, yerr=[[bl_err_lo], [bl_err_hi]],
                     fmt="D", color=BASELINE_1K_COLOR, markersize=MARKER_SIZE + 1,
                     markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                     capsize=2.5, capthick=0.8, ecolor=BASELINE_1K_COLOR,
                     elinewidth=1.0, zorder=5)

    # Coarse architectures (AlexNet, CLIP, Pixels — no ViT)
    for arch_idx, (arch_key, folder, _) in enumerate(ARCHITECTURES):
        style = ARCH_STYLE[arch_key]
        means, ci_lo, ci_hi = fetch_things_arch_data(folder)
        jitter = compute_jitter(arch_idx, len(ARCHITECTURES))
        for i, cfg in enumerate(COARSE_CFGS):
            if np.isnan(means[i]):
                continue
            e_lo = max(means[i] - ci_lo[i], 0) if not np.isnan(ci_lo[i]) else 0
            e_hi = max(ci_hi[i] - means[i], 0) if not np.isnan(ci_hi[i]) else 0
            ax.errorbar(cfg * jitter, means[i], yerr=[[e_lo], [e_hi]],
                         fmt=style["marker"], color=style["color"],
                         markersize=MARKER_SIZE + 1,
                         markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                         capsize=2.5, capthick=0.8,
                         ecolor=style["color"], elinewidth=1.0, zorder=4)

    format_coarseness_axes(ax, "", show_ylabel=True, show_xlabel=True)
    ax.set_title("Alignment vs. Granularity",
                 fontsize=9.5, fontweight="semibold", pad=8)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    setup_style()
    plt.rcParams.update({
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
    })

    # ── Standalone comparison plot ──
    save_standalone_comparison()

    # ── Full figure 4 ──
    fig = plt.figure(figsize=(15, 9))

    # Top row: 4 columns; Bottom row: RDMs spanning full width
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.05, 1.15],
                           hspace=0.38, left=0.06, right=0.96,
                           top=0.95, bottom=0.04)

    # ── Top row: 3 panels (histogram is inset in scatter) ──
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0],
                                              wspace=0.38,
                                              width_ratios=[1, 1.6, 1.1])

    # Panel A: Coarseness (Alignment vs. Granularity)
    ax_coarse = fig.add_subplot(gs_top[0, 0])
    plot_coarseness_raw(ax_coarse)

    # Panel B: Model Comparison (data efficiency + pretrained)
    ax_compare = fig.add_subplot(gs_top[0, 1])
    plot_comparison_panel(ax_compare)

    # Coarseness legend — local color scheme
    legend_handles = []
    for arch_key, _, display in ARCHITECTURES:
        style = ARCH_STYLE[arch_key]
        h = Line2D([], [], marker=style["marker"], color="none",
                   markerfacecolor=style["color"],
                   markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                   markersize=MARKER_SIZE, label=display)
        legend_handles.append(h)
    legend_handles.append(Line2D([], [], marker="D", color="none",
                                  markerfacecolor=BASELINE_1K_COLOR,
                                  markeredgecolor=EDGE_COLOR,
                                  markeredgewidth=EDGE_WIDTH,
                                  markersize=MARKER_SIZE, label="1K (ImageNet)"))
    legend_handles.append(Line2D([], [], **UNTRAINED_LINE_STYLE, label="Untrained"))

    leg_c = ax_coarse.legend(
        handles=legend_handles,
        fontsize=6, frameon=True, loc="lower left",
        edgecolor="#dddddd", fancybox=False, framealpha=0.95,
        handletextpad=0.25, columnspacing=0.5, ncol=2,
        borderpad=0.3, bbox_to_anchor=(0.0, 0.0))
    leg_c.get_frame().set_linewidth(0.5)

    # Panel C: Scatter with histogram inset
    ax_scatter = fig.add_subplot(gs_top[0, 2])

    # Compute THINGS data (RDMs + scatter data)
    print("Computing THINGS data for scatter and RDMs...")
    precomputed = compute_things_data()

    # Create histogram inset in upper-left of scatter
    ax_hist = ax_scatter.inset_axes([0.03, 0.58, 0.38, 0.38])

    plot_scatter_panel(ax_scatter, ax_hist, precomputed)

    # Style the inset: clean white background, subtle border
    ax_hist.patch.set_alpha(0.97)
    ax_hist.patch.set_facecolor("white")
    for spine in ax_hist.spines.values():
        spine.set_color("#aaaaaa")
        spine.set_linewidth(0.6)

    # ── Bottom row: 3 RDMs + colorbar ──
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1],
                                              wspace=0.06,
                                              width_ratios=[1, 1, 1, 0.05])

    ax_rdm1 = fig.add_subplot(gs_bot[0, 0])
    ax_rdm2 = fig.add_subplot(gs_bot[0, 1])
    ax_rdm3 = fig.add_subplot(gs_bot[0, 2])
    ax_cb = fig.add_subplot(gs_bot[0, 3])

    rdm_axes = [ax_rdm1, ax_rdm2, ax_rdm3]
    plot_rdm_panels(rdm_axes, precomputed, show_difference=False,
                    colorbar_axes=(ax_cb, ax_cb))

    # ── Panel labels ──
    for ax, label, x_off in zip(
        [ax_coarse, ax_compare, ax_scatter, ax_rdm1],
        ["A", "B", "C", "D"],
        [-0.14, -0.08, -0.10, -0.04]):
        ax.text(x_off, 1.10, label, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", ha="left",
                family="sans-serif")

    # ── Save ──
    out = f"{OUTPUT_DIR}/figure4.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
