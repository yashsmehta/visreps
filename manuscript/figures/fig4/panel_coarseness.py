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
from fig_utils import COARSE_CFGS, MARKER_SIZE, EDGE_COLOR, EDGE_WIDTH, UNTRAINED_LINE_STYLE, compute_jitter

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

    # 1000-way dashed reference line
    ax.axhline(bl_mean, color=BASELINE_1K_COLOR, linestyle="--",
               linewidth=1.1, alpha=0.85, zorder=2)

    # Untrained baseline (epoch=0)
    un = get_condition_summary("things-behavior", "N/A", "imagenet1k", 1000,
                               "spearman", epoch=0, analysis="rsa")
    if not np.isnan(un["mean"]):
        ax.axhline(un["mean"], **UNTRAINED_LINE_STYLE, zorder=2)
        all_y_vals.append(un["mean"])

    y_min, y_max = min(all_y_vals), max(all_y_vals)
    y_range = y_max - y_min
    ax.text(180 * 0.95, bl_mean + y_range * 0.015, "Trained, 1000 classes",
            fontsize=6, fontstyle="italic", color=BASELINE_1K_COLOR,
            ha="right", va="bottom", zorder=10)

    # ── Axis formatting ──
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(COARSE_CFGS))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda val, pos: str(int(val)) if int(round(val)) in COARSE_CFGS_SET else ""))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.tick_params(axis="x", which="major", direction="out")
    ax.set_xlim(1.5, 180)

    ax.tick_params(axis="y", which="major", direction="out")
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", direction="out")
    ax.yaxis.grid(True, which="major", color="#F0F0F0", linewidth=0.3, zorder=0)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".")))
    ax.set_ylim(y_min - y_range * 0.12, y_max + y_range * 0.10)

    ax.set_xlabel("Training classes", fontsize=9, labelpad=6)
    ax.set_ylabel(r"RSA (Spearman $\rho$)", fontsize=9, labelpad=3)
    sns.despine(ax=ax, right=True, top=True, offset=4)
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
