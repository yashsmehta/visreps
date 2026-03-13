"""Figure 4: THINGS Behavioral Alignment.

Layout:
  Row 0 (top): [Schematic placeholder | Coarseness | Model Comparison]
  Row 1 (bottom): [3 RDMs side by side + colorbar] spanning full width

Panel A: Schematic of THINGS behavioral similarity task (placeholder)
Panel B: Alignment vs. Granularity (raw Spearman rho, log x-axis)
Panel C: Model comparison — coarse vs 1000-way bars + pretrained scatter
Panel D: 3 RDMs (human behavioral, CLIP 4-class, 1000-class)

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
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter
import seaborn as sns

sys.path.insert(0, "plotters")
from plotter_utils import get_condition_summary

sys.path.insert(0, "manuscript/figures")
from fig_utils import (
    DB_PATH,
    COARSE_CFGS, BREAK_1K_POS,
    UNTRAINED_LINE_STYLE, MARKER_SIZE, EDGE_COLOR, EDGE_WIDTH,
    setup_style, compute_jitter,
    format_coarseness_axes, draw_schematic_placeholder,
)
from things_utils import compute_things_data, plot_rdm_panels

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
BASELINE_1K_COLOR = "#d4822e"  # warm amber
COARSE_BAR_COLOR = "#08519c"   # dark blue (CLIP)

# ── Pretrained model comparison config ────────────────────────────────────
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
    """Panel B: Raw Spearman rho coarseness for THINGS behavioral."""
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

    # Coarse architectures
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


# ── Model comparison panel (simplified: 2 bars + pretrained scatter) ─────

def _fetch_clip8_score():
    """Fetch CLIP 8-class score on THINGS."""
    s = get_condition_summary("things-behavior", "N/A", "pca_labels_clip", 8,
                              "spearman", epoch=20, analysis="rsa")
    return {
        "mean": s["mean"],
        "ci_low": s["ci_low"],
        "ci_high": s["ci_high"],
        "label": "CLIP 8-class",
    }


def _fetch_pretrained_data():
    """Fetch pretrained model scores for scatter."""
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
    return all_points


def _draw_neurips_bar(ax, x, height, width, color, y_base=0.25,
                      edgecolor="black", linewidth=0.8, zorder=3):
    """Draw a bar with rounded top corners in NeurIPS style."""
    bar_bottom = y_base - 0.015
    bar_height = height - bar_bottom
    rect = mpatches.FancyBboxPatch(
        (x - width / 2, bar_bottom), width, bar_height,
        boxstyle=mpatches.BoxStyle("Round", pad=0.02, rounding_size=0.08),
        facecolor=color, edgecolor=edgecolor,
        linewidth=linewidth, mutation_aspect=0.05, zorder=zorder,
    )
    ax.add_patch(rect)


def plot_comparison_panel(ax):
    """Panel C: Coarse vs 1000-way bars + pretrained model scatter."""
    # ── Fetch data ──
    bl = get_condition_summary("things-behavior", "N/A", "imagenet1k", 1000,
                               "spearman", epoch=20, analysis="rsa")
    best_coarse = _fetch_clip8_score()
    all_points = _fetch_pretrained_data()

    s = 0.78  # scale factor

    # ── Layout: 2 bars then pretrained scatter ──
    bar_w = 0.48
    bar_positions = [0.0, 0.6]  # coarse, 1000-way
    bars = [
        (bar_positions[0], best_coarse["mean"], best_coarse["ci_low"],
         best_coarse["ci_high"], COARSE_BAR_COLOR, best_coarse["label"]),
        (bar_positions[1], bl["mean"], bl["ci_low"],
         bl["ci_high"], BASELINE_1K_COLOR, "1000-class"),
    ]

    for x, score, ci_lo, ci_hi, color, label in bars:
        if np.isnan(score):
            continue
        _draw_neurips_bar(ax, x, score, bar_w, color,
                          y_base=0.2, edgecolor="#333333", linewidth=0.7)
        err_lo = max(score - ci_lo, 0) if not np.isnan(ci_lo) else 0
        err_hi = max(ci_hi - score, 0) if not np.isnan(ci_hi) else 0
        ax.errorbar(x, score, yerr=[[err_lo], [err_hi]], fmt="none",
                    ecolor="#333333", capsize=3.0 * s, capthick=0.7,
                    elinewidth=0.7, zorder=4)

    # ── Dashed reference line from coarse bar into scatter region ──
    scatter_start = 2.0
    group_positions = {
        "Supervised":      scatter_start,
        "Self-supervised": scatter_start + 1.4,
        "Vision-language": scatter_start + 2.8,
    }
    jitter_spread = 0.25

    if not np.isnan(best_coarse["mean"]):
        x_ref_start = bar_positions[0] + bar_w / 2 + 0.1
        x_ref_end = list(group_positions.values())[-1] + 0.6
        ax.plot([x_ref_start, x_ref_end],
                [best_coarse["mean"], best_coarse["mean"]],
                color=COARSE_BAR_COLOR, linestyle=(0, (5, 3)),
                linewidth=0.7, alpha=0.40, zorder=1)

    # Subtle vertical separator between bars and scatter
    sep_x = (bar_positions[-1] + scatter_start) / 2
    ax.axvline(sep_x, color="#d0d0d0", linewidth=0.6, linestyle="-",
               ymin=0.0, ymax=1.0, zorder=0)

    # ── Draw pretrained scatter ──
    pt_size_base = 110 * s * s
    pt_size_star = 155 * s * s
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
                color=pt["color"], linewidth=1.2, alpha=0.45, zorder=4,
                solid_capstyle="round")
        sz = pt_size_star if pt["marker"] == "*" else pt_size_base
        ax.scatter(x_jit, pt["score"], marker=pt["marker"], c=pt["color"],
                   s=sz, edgecolors="white", linewidths=0.6, zorder=5)

    # ── Model name labels ──
    fs_model = 6.5 * s
    x_offset = 0.22 * s
    min_gap = 0.013
    for group_name in PRETRAINED_GROUPS:
        group_pts = sorted(
            [p for p in all_points if p["group"] == group_name],
            key=lambda p: p["score"], reverse=True)
        used_y = []
        for pt in group_pts:
            y = pt["score"]
            for uy in used_y:
                if abs(y - uy) < min_gap:
                    y = uy - min_gap
            used_y.append(y)
            ax.text(pt["x_plot"] + x_offset, y, pt["display"],
                    ha="left", va="center", fontsize=fs_model, color="#333333",
                    fontstyle="italic")

    # ── Axis formatting ──
    xlim_right = list(group_positions.values())[-1] + 1.4
    ax.set_xlim(-0.55, xlim_right)
    ax.set_ylim(0.2, 0.66)
    ax.set_ylabel(r"Spearman $\rho$", fontsize=10 * s, labelpad=5)
    ax.set_title("Model Comparison",
                 fontsize=11 * s, fontweight="semibold", pad=8)

    # X-ticks
    bar_xticks = bar_positions
    bar_xlabels = ["CLIP\n8-class", "1000-\nclass"]
    scatter_xticks = [group_positions[g] for g in group_positions]
    scatter_xlabels = ["Supervised", "Self-\nsupervised", "Vision-\nlanguage"]
    ax.set_xticks(bar_xticks + scatter_xticks)
    ax.set_xticklabels(bar_xlabels + scatter_xlabels, fontsize=7.5 * s)

    # Subtle horizontal grid
    ax.yaxis.grid(True, which="major", color="#EBEBEB", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Y-axis
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.tick_params(axis="y", which="major", direction="out", length=5,
                   width=1.2, labelsize=9 * s)
    ax.tick_params(axis="y", which="minor", direction="out", length=3,
                   width=0.8)
    ax.tick_params(axis="x", which="major", length=4, width=1.0, direction="out")

    sns.despine(ax=ax, right=True, top=True, offset=5)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_linewidth(1.2)

    # ── Legend ──
    leg_handles = [
        mpatches.Patch(facecolor=COARSE_BAR_COLOR, edgecolor="#333333",
                       linewidth=0.6, label=best_coarse["label"]),
        mpatches.Patch(facecolor=BASELINE_1K_COLOR, edgecolor="#333333",
                       linewidth=0.6, label="1000-class"),
        Line2D([], [], marker="p", color="none", markerfacecolor="#777777",
               markeredgecolor="white", markeredgewidth=0.5,
               markersize=8, label="CNN"),
        Line2D([], [], marker="*", color="none", markerfacecolor="#777777",
               markeredgecolor="white", markeredgewidth=0.4,
               markersize=10, label="ViT"),
    ]
    leg = ax.legend(handles=leg_handles, fontsize=6.5 * s, frameon=True,
                    loc="upper left", edgecolor="#dddddd", fancybox=False,
                    framealpha=0.95, handletextpad=0.3, borderpad=0.3,
                    labelspacing=0.2, ncol=2, columnspacing=0.5,
                    bbox_to_anchor=(-0.01, 1.01))
    leg.get_frame().set_linewidth(0.3)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    setup_style()
    plt.rcParams.update({
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
    })

    fig = plt.figure(figsize=(14.5, 8.5))
    fig.patch.set_facecolor("white")

    # Top row: 3 panels; Bottom row: RDMs spanning full width
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.05, 1.15],
                           hspace=0.35, left=0.05, right=0.96,
                           top=0.95, bottom=0.04)

    gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0],
                                              wspace=0.38,
                                              width_ratios=[0.85, 0.85, 1.7])

    # Panel A: Schematic placeholder
    ax_schematic = fig.add_subplot(gs_top[0, 0])
    draw_schematic_placeholder(ax_schematic,
                               "THINGS\nBehavioral Similarity\n(schematic)")

    # Panel B: Coarseness (Alignment vs. Granularity)
    ax_coarse = fig.add_subplot(gs_top[0, 1])
    plot_coarseness_raw(ax_coarse)

    # Coarseness legend
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

    # Panel C: Model Comparison
    ax_compare = fig.add_subplot(gs_top[0, 2])
    plot_comparison_panel(ax_compare)

    # ── Bottom row: 3 RDMs + colorbar ──
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1],
                                              wspace=0.06,
                                              width_ratios=[1, 1, 1, 0.05])

    ax_rdm1 = fig.add_subplot(gs_bot[0, 0])
    ax_rdm2 = fig.add_subplot(gs_bot[0, 1])
    ax_rdm3 = fig.add_subplot(gs_bot[0, 2])
    ax_cb = fig.add_subplot(gs_bot[0, 3])

    rdm_axes = [ax_rdm1, ax_rdm2, ax_rdm3]
    for rdm_ax in rdm_axes:
        rdm_ax.set_facecolor("white")
    ax_cb.set_facecolor("white")

    print("Computing THINGS data for RDMs...")
    precomputed = compute_things_data()
    plot_rdm_panels(rdm_axes, precomputed, show_difference=False,
                    colorbar_axes=(ax_cb, ax_cb))

    # ── Panel labels ──
    for ax, label, x_off in zip(
        [ax_schematic, ax_coarse, ax_compare, ax_rdm1],
        ["A", "B", "C", "D"],
        [-0.08, -0.14, -0.08, -0.04]):
        ax.text(x_off, 1.10, label, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", ha="left",
                family="sans-serif")

    # ── Save ──
    out = f"{OUTPUT_DIR}/figure4.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
