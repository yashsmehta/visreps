"""Panel B: Alignment vs. Granularity (THINGS behavioral).

Scatter of coarse-trained models (2-64 classes) across three PCA label sources
(AlexNet, CLIP, Pixels) with a 1000-way dashed baseline.
"""

import sys

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, FixedLocator, NullLocator
import seaborn as sns

sys.path.insert(0, "plotters")
from plotter_utils import get_condition_summary

sys.path.insert(0, "manuscript/figures")
from fig_utils import (
    COARSE_CFGS, MARKER_SIZE, EDGE_COLOR, EDGE_WIDTH, UNTRAINED_LINE_STYLE,
    BREAK_1K_POS, compute_jitter, draw_xaxis_break, draw_torn_ci_band,
)

COARSE_CFGS_SET = set(COARSE_CFGS)

# ── Style constants ──────────────────────────────────────────────────────
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
BASELINE_1K_COLOR = "#e8963e"  # warm amber


def _fetch_arch_data(folder):
    """Fetch coarseness scores for one PCA label source."""
    means, ci_lo, ci_hi = [], [], []
    for cfg in COARSE_CFGS:
        s = get_condition_summary("things-behavior", "N/A", folder, cfg,
                                  "spearman", epoch=20, analysis="rsa")
        means.append(s["mean"])
        ci_lo.append(s["ci_low"])
        ci_hi.append(s["ci_high"])
    return np.array(means), np.array(ci_lo), np.array(ci_hi)


def plot_coarseness(ax):
    """Plot coarseness scatter + 1000-way dashed baseline + architecture legend."""
    # 1000-way baseline
    bl = get_condition_summary("things-behavior", "N/A", "imagenet1k", 1000,
                               "spearman", epoch=20, analysis="rsa")
    bl_mean = bl["mean"]

    if np.isnan(bl_mean) or bl_mean == 0:
        ax.text(0.5, 0.5, "No baseline", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888")
        return

    all_y_vals = [bl_mean]

    # Coarse architectures
    for arch_idx, (arch_key, folder, _) in enumerate(ARCHITECTURES):
        style = ARCH_STYLE[arch_key]
        means, ci_lo, ci_hi = _fetch_arch_data(folder)
        all_y_vals.extend([m for m in means if not np.isnan(m)])
        jitter = compute_jitter(arch_idx, len(ARCHITECTURES))
        for i, cfg in enumerate(COARSE_CFGS):
            if np.isnan(means[i]):
                continue
            e_lo = max(means[i] - ci_lo[i], 0) if not np.isnan(ci_lo[i]) else 0
            e_hi = max(ci_hi[i] - means[i], 0) if not np.isnan(ci_hi[i]) else 0
            ax.errorbar(cfg * jitter, means[i], yerr=[[e_lo], [e_hi]],
                        fmt=style["marker"], color=style["color"],
                        markersize=MARKER_SIZE,
                        markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                        capsize=1.5, capthick=0.5,
                        ecolor=style["color"], elinewidth=0.7, zorder=4)

    # 1000-way baseline: CI band + bounded dashed mean + orange diamond
    bl_err_lo = max(bl_mean - bl["ci_low"], 0) if not np.isnan(bl["ci_low"]) else 0
    bl_err_hi = max(bl["ci_high"] - bl_mean, 0) if not np.isnan(bl["ci_high"]) else 0

    # Pale orange CI band across the coarse region (torn later)
    if not np.isnan(bl["ci_low"]) and not np.isnan(bl["ci_high"]):
        ax.fill_between([1.5, BREAK_1K_POS], bl["ci_low"], bl["ci_high"],
                        facecolor=BASELINE_1K_COLOR, alpha=0.12,
                        edgecolor="none", zorder=1)
        all_y_vals.extend([bl["ci_low"], bl["ci_high"]])

    # Dashed mean line bounded to coarse region (torn later)
    ax.plot([1.5, BREAK_1K_POS], [bl_mean, bl_mean],
            color=BASELINE_1K_COLOR, linestyle="--",
            linewidth=1.0, alpha=0.6, zorder=2, clip_on=False)

    ax.errorbar(BREAK_1K_POS, bl_mean,
                yerr=[[bl_err_lo], [bl_err_hi]],
                fmt="D", color=BASELINE_1K_COLOR, markersize=MARKER_SIZE,
                markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                capsize=1.5, capthick=0.5,
                ecolor=BASELINE_1K_COLOR, elinewidth=0.7, zorder=5)

    # Untrained baseline (epoch=0) — zorder=3 so mask at 2.6 doesn't erase it
    un = get_condition_summary("things-behavior", "N/A", "imagenet1k", 1000,
                               "spearman", epoch=0, analysis="rsa")
    if not np.isnan(un["mean"]):
        ax.axhline(un["mean"], **UNTRAINED_LINE_STYLE, zorder=3)
        all_y_vals.append(un["mean"])

    y_min, y_max = min(all_y_vals), max(all_y_vals)
    y_range = y_max - y_min

    # ── Axis formatting (broken x-axis matching fig3) ──
    ax.set_xscale("log", base=2)
    all_x = COARSE_CFGS + [BREAK_1K_POS]
    label_map = {v: str(v) for v in COARSE_CFGS}
    label_map[BREAK_1K_POS] = "1000"
    ax.xaxis.set_major_locator(FixedLocator(all_x))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda val, pos: label_map.get(int(round(val)), "")))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.tick_params(axis="x", which="major", direction="out", labelsize=12)
    ax.set_xlim(1.5, BREAK_1K_POS * 1.5)

    ax.tick_params(axis="y", which="major", direction="out")
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", direction="out")
    ax.yaxis.grid(True, which="major", color="#F0F0F0", linewidth=0.3, zorder=0)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".")))
    ax.set_ylim(y_min - y_range * 0.12, y_max + y_range * 0.10)

    # Tear through the 1000-way CI band (after ylim is final)
    if not np.isnan(bl["ci_low"]) and not np.isnan(bl["ci_high"]):
        draw_torn_ci_band(ax, bl["ci_low"], bl["ci_high"], BASELINE_1K_COLOR)

    ax.set_xlabel("Granularity", fontsize=10.8, labelpad=6)
    ax.set_ylabel(r"RSA (Spearman $\rho$)", fontsize=10.8, labelpad=3)
    sns.despine(ax=ax, right=True, top=True, offset=4)
    draw_xaxis_break(ax)
    ax.set_title("Alignment vs. Granularity",
                 fontsize=11, fontweight="semibold", pad=8)

    # ── Legend ──
    legend_handles = [
        Line2D([], [], marker=ARCH_STYLE[k]["marker"], color="none",
               markerfacecolor=ARCH_STYLE[k]["color"],
               markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
               markersize=5, label=d)
        for k, _, d in ARCHITECTURES
    ]
    leg = ax.legend(
        handles=legend_handles, fontsize=8,
        frameon=True, fancybox=False, framealpha=0.92,
        edgecolor="#dddddd", borderpad=0.35,
        handletextpad=0.3, labelspacing=0.2,
        title="Coarse label source", title_fontsize=7.5,
        loc="center left", bbox_to_anchor=(0.0, 0.42))
    leg._legend_box.align = "left"
