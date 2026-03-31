"""Panel renderer: minimum bits of supervision to match 1000-way alignment."""

import sys

import numpy as np
import seaborn as sns
from matplotlib.ticker import FixedLocator, FuncFormatter

sys.path.insert(0, "manuscript/figures/fig3")
from shared import (
    COARSE_CFGS, ARCH_STYLE, BASELINE_1K_COLOR,
    fetch_baseline_ci, fetch_coarse_ci,
)

# Architectures shown in bits panels (no Pixels)
BITS_ARCHS = [
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


def plot_bits(ax, dataset, region, show_ylabel=True, show_xlabel=True):
    """Lollipop chart: bits of supervision needed to match 1000-way, per architecture."""
    bl_mean, bl_ci_low, _ = fetch_baseline_ci(dataset, region)
    if np.isnan(bl_mean) or bl_mean == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color="#888")
        return

    bits_vals = []
    for arch_key, folder, display in BITS_ARCHS:
        bits = _find_min_bits(dataset, folder, region, bl_ci_low)
        bits_vals.append((arch_key, display, bits))

    # Place lollipops close together, centered in the panel
    x_pos = np.array([0.35, 0.65])

    for i, (arch_key, display, h) in enumerate(bits_vals):
        c = ARCH_STYLE[arch_key]["color"]
        marker = ARCH_STYLE[arch_key]["marker"]
        if np.isnan(h):
            ax.text(x_pos[i], 1.5, "N/A", ha="center", va="bottom",
                    fontsize=7, color="#999", fontstyle="italic")
            continue
        # Stem
        ax.plot([x_pos[i], x_pos[i]], [0, h], color=c,
                linewidth=2.0, solid_capstyle="round", zorder=3)
        # Dot
        ax.plot(x_pos[i], h, marker=marker, color=c, markersize=10,
                markeredgecolor="white", markeredgewidth=1.0, zorder=4)

    # Class count labels above each lollipop, in arch color
    for i, (arch_key, _, h) in enumerate(bits_vals):
        if np.isnan(h):
            continue
        c = ARCH_STYLE[arch_key]["color"]
        n_classes = int(2 ** h)
        ax.text(x_pos[i], h + 0.45, f"{n_classes} cls",
                ha="center", va="bottom", fontsize=7,
                fontweight="semibold", color=c, zorder=5)

    # 1000-way reference line at top with label
    ax.axhline(BASELINE_BITS, color=BASELINE_1K_COLOR, linestyle="--",
               linewidth=1.1, alpha=0.85, zorder=2)
    ax.text(0.97, BASELINE_BITS - 0.25, "1000 cls",
            ha="right", va="top", fontsize=7, fontweight="semibold",
            color=BASELINE_1K_COLOR, zorder=5)

    # X-axis: architecture labels, centered under each lollipop
    ax.set_xlim(0, 1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([d for _, d, _ in bits_vals], fontsize=8)
    ax.tick_params(axis="x", length=3.5, width=0.7)
    if show_xlabel:
        ax.set_xlabel("Label source", fontsize=9, labelpad=4)

    # Y-axis: consistent 0–10.5, integer ticks 1–10
    ax.set_ylim(0, BASELINE_BITS + 0.5)
    ax.yaxis.set_major_locator(FixedLocator(range(1, 11)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: str(int(v))))
    ax.tick_params(axis="y", which="major", direction="out", length=3.5, width=0.6,
                   labelsize=7)
    ax.yaxis.grid(True, which="major", color="#F0F0F0", linewidth=0.3, zorder=0)

    if show_ylabel:
        ax.set_ylabel("Bits of supervision", fontsize=9, labelpad=4)
    else:
        ax.set_ylabel("")
    sns.despine(ax=ax, right=True, top=True, offset=3)
