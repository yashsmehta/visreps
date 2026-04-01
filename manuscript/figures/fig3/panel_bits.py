"""Panel renderer: horizontal lollipop showing min classes to match 1000-way.

Each lollipop is a thin stem from the y-axis to x=k*, where k* is the minimum
number of training classes whose 95% CI overlaps the 1000-way baseline.
A vertical dashed orange line at x=BREAK_1K_POS marks the 1000-way reference.
"""

import sys

import numpy as np
import seaborn as sns
from matplotlib.ticker import FixedLocator, NullLocator

sys.path.insert(0, "manuscript/figures/fig3")
from shared import (
    COARSE_CFGS, ARCH_STYLE, BASELINE_1K_COLOR,
    fetch_baseline_ci, fetch_coarse_ci,
)

sys.path.insert(0, "manuscript/figures")
from fig_utils import BREAK_1K_POS, EDGE_COLOR, EDGE_WIDTH

# AlexNet on bottom (y=0), CLIP on top (y=1)
LOLLIPOP_ARCHS = [
    ("alexnet", "pca_labels_alexnet", "AlexNet"),
    ("clip",    "pca_labels_clip",    "CLIP"),
]

STEM_LW = 2.2
MARKER_SZ = 6.5
X_START = 1.5  # matches xlim left — stems start at the y-axis


def _draw_lollipop_break(ax):
    """Draw interlocking jagged break through the gray lollipop box.

    The gray background keeps jagged edges on both sides of the break.
    A white polygon fills the gap between the interlocking edges.
    Uses pure axes-fraction coordinates for visual consistency on log axes.
    """
    from matplotlib.patches import Polygon
    import math

    # Compute break midpoint as axes fraction (log2 scale)
    x_lo, x_hi = 1.5, BREAK_1K_POS * 1.5
    mid_data = math.exp((math.log(64) + math.log(BREAK_1K_POS)) / 2)
    mid_frac = (math.log2(mid_data) - math.log2(x_lo)) / (math.log2(x_hi) - math.log2(x_lo))

    trans = ax.transAxes

    # Zigzag parameters (axes fraction)
    n_teeth = 4
    tooth_dx = 0.010   # how far each tooth protrudes into the gap
    gap = 0.045        # base separation between edges
    y_pts = np.linspace(-0.1, 1.1, n_teeth * 2 + 1)
    zigzag = np.array([(-1) ** k for k in range(len(y_pts))])

    # Interlocking: left edge bumps right when right edge bumps left
    left_edge_x = mid_frac - gap / 2 + tooth_dx * zigzag
    right_edge_x = mid_frac + gap / 2 - tooth_dx * zigzag  # inverted

    # Single white polygon filling the gap between the two jagged edges
    # Trace left edge top-to-bottom, then right edge bottom-to-top
    verts = (list(zip(left_edge_x, y_pts))
             + list(zip(right_edge_x[::-1], y_pts[::-1])))
    ax.add_patch(Polygon(verts, facecolor="white", edgecolor="none",
                         transform=trans, clip_on=False, zorder=9))

    # Subtle edge lines on the gray side for definition
    ax.plot(left_edge_x, y_pts, transform=trans, color='#bbbbbb',
            linewidth=0.6, clip_on=True, zorder=11)
    ax.plot(right_edge_x, y_pts, transform=trans, color='#bbbbbb',
            linewidth=0.6, clip_on=True, zorder=11)


def _find_min_classes(dataset, folder, region, bl_ci_low):
    """Return the first coarse cfg whose ci_high >= bl_ci_low, else NaN."""
    for cfg in COARSE_CFGS:
        _, _, ci_high = fetch_coarse_ci(dataset, folder, region, cfg)
        if not np.isnan(ci_high) and ci_high >= bl_ci_low:
            return cfg
    return np.nan


def plot_lollipop(ax, dataset, region, show_ylabel=True):
    """Horizontal lollipop strip: min classes to match 1000-way, per label source."""
    bl_mean, bl_ci_low, _ = fetch_baseline_ci(dataset, region)
    if np.isnan(bl_mean) or bl_mean == 0:
        ax.axis("off")
        return

    # Dim gray background
    ax.set_facecolor("#f5f5f5")

    y_positions = list(range(len(LOLLIPOP_ARCHS)))

    for i, (arch_key, folder, display) in enumerate(LOLLIPOP_ARCHS):
        style = ARCH_STYLE[arch_key]
        k_star = _find_min_classes(dataset, folder, region, bl_ci_low)
        y = y_positions[i]

        if np.isnan(k_star):
            # Faded stem to 64 indicating no match found
            ax.plot([X_START, 64], [y, y], color=style["color"], linewidth=STEM_LW,
                    alpha=0.25, solid_capstyle="round", zorder=3)
            ax.text(64 * 1.15, y, "—", fontsize=7, color="#999",
                    va="center", ha="left")
            continue

        # Thin stem from y-axis to k*
        ax.plot([X_START, k_star], [y, y], color=style["color"], linewidth=STEM_LW,
                solid_capstyle="round", zorder=3)
        # Marker at k*
        ax.plot(k_star, y, marker=style["marker"], color=style["color"],
                markersize=MARKER_SZ, markeredgecolor=EDGE_COLOR,
                markeredgewidth=EDGE_WIDTH, zorder=4)

    # 1000-way reference: vertical dashed line
    ax.axvline(BREAK_1K_POS, color=BASELINE_1K_COLOR, linestyle="--",
               linewidth=1.0, alpha=0.7, zorder=2)

    # ── Axis formatting ──
    ax.set_ylim(-0.6, len(LOLLIPOP_ARCHS) - 0.4)

    # Y-axis: architecture labels with ticks
    ax.set_yticks(y_positions)
    if show_ylabel:
        ax.set_yticklabels([d for _, _, d in LOLLIPOP_ARCHS], fontsize=7.5)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="y", length=3, width=0.7, pad=3)

    # Hide x-axis ticks/labels (scatter below handles it)
    ax.tick_params(axis="x", labelbottom=False, length=0, bottom=False)

    # Left spine visible (aligns with scatter y-axis), others hidden
    sns.despine(ax=ax, bottom=True, right=True, top=True, left=False)
    ax.spines["left"].set_linewidth(0.7)

    # Break in the gray box matching the scatter x-axis break
    _draw_lollipop_break(ax)
