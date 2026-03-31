"""Panel renderer: feedback coarseness margin (FCM) as rounded bar plot.

FCM = (1 - log2(k*) / log2(1000)) * 100%, where k* is the minimum number
of training classes whose alignment CI overlaps the 1000-way baseline.
Higher FCM = the feedback signal can be coarser while preserving alignment.

Visual style adapted from experiments/neurips_2025/fig2/bar_plot_nsd.py.
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, FuncFormatter

sys.path.insert(0, "manuscript/figures/fig3")
from shared import (
    COARSE_CFGS, ARCH_STYLE,
    fetch_baseline_ci, fetch_coarse_ci,
)

# Architectures shown in FCM panels (no Pixels)
FCM_ARCHS = [
    ("alexnet", "pca_labels_alexnet", "AlexNet"),
    ("clip",    "pca_labels_clip",    "CLIP"),
]

BASELINE_BITS = np.log2(1000)  # ~9.97


def _find_min_bits(dataset, folder, region, bl_ci_low):
    """Find minimum training classes whose CI overlaps with baseline CI.

    Returns log2(classes) for the first coarse condition whose ci_high >= bl_ci_low,
    or NaN if no condition reaches the baseline.
    """
    for cfg in COARSE_CFGS:
        _, _, ci_high = fetch_coarse_ci(dataset, folder, region, cfg)
        if not np.isnan(ci_high) and ci_high >= bl_ci_low:
            return np.log2(cfg)
    return np.nan


def plot_fcm(ax, dataset, region, show_ylabel=True, show_xlabel=True):
    """Rounded bar chart: feedback coarseness margin per label source.

    Style matches NeurIPS 2025 bar plots (FancyBboxPatch with rounded corners,
    hatching, thick spines).
    """
    bl_mean, bl_ci_low, _ = fetch_baseline_ci(dataset, region)
    if np.isnan(bl_mean) or bl_mean == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color="#888")
        return

    fcm_vals = []
    for arch_key, folder, display in FCM_ARCHS:
        min_bits = _find_min_bits(dataset, folder, region, bl_ci_low)
        if np.isnan(min_bits):
            fcm_vals.append((arch_key, display, np.nan))
        else:
            fcm = (1 - min_bits / BASELINE_BITS) * 100
            fcm_vals.append((arch_key, display, fcm))

    positions = np.arange(len(fcm_vals))
    bar_w = 0.6

    # Set hatch color to grey (NeurIPS style)
    original_hatch_color = plt.rcParams.get("hatch.color")
    plt.rcParams["hatch.color"] = "grey"

    for i, (arch_key, display, fcm) in enumerate(fcm_vals):
        c = ARCH_STYLE[arch_key]["color"]
        if np.isnan(fcm):
            ax.text(positions[i], 5, "N/A", ha="center", va="bottom",
                    fontsize=7, color="#999", fontstyle="italic")
            continue
        x0 = positions[i] - bar_w / 2
        rect = mpatches.FancyBboxPatch(
            (x0, 0), bar_w, fcm,
            boxstyle=mpatches.BoxStyle("Round", pad=0.02, rounding_size=0.1),
            facecolor=c, edgecolor="black",
            linewidth=0.8, hatch="/", mutation_aspect=0.05, zorder=3,
        )
        ax.add_patch(rect)

    # Restore hatch color
    if original_hatch_color is not None:
        plt.rcParams["hatch.color"] = original_hatch_color

    # ── X-axis ──
    ax.set_xticks(positions)
    ax.set_xticklabels([d for _, d, _ in fcm_vals], fontsize=8)
    ax.tick_params(axis="x", direction="out", bottom=False, top=False,
                   length=4, width=1.0)
    ax.set_xlim(-0.6, len(fcm_vals) - 0.4)
    if show_xlabel:
        ax.set_xlabel("Label source", fontsize=9, labelpad=4)

    # ── Y-axis ──
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    def _pct_formatter(v, _):
        if np.isclose(v, 0):
            return ""
        return f"{int(v)}%"

    ax.yaxis.set_major_formatter(FuncFormatter(_pct_formatter))
    ax.tick_params(axis="y", which="major", direction="out", length=5,
                   width=1.0, labelsize=7)
    ax.tick_params(axis="y", which="minor", direction="out", length=3,
                   width=0.7)

    if show_ylabel:
        ax.set_ylabel("Compression (%)", fontsize=9, labelpad=4)
    else:
        ax.set_ylabel("")

    # ── Spines (NeurIPS style: thick bottom + left) ──
    sns.despine(ax=ax, right=True, top=True, offset=5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
