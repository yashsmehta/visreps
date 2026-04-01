"""Shared constants, data fetching, and axis formatting for Figure 3.

Used by panel_raw.py, panel_bits.py, and figure3.py.
"""

import sys
from functools import lru_cache

import numpy as np
from matplotlib.ticker import AutoMinorLocator, FixedLocator, FuncFormatter, NullLocator

sys.path.insert(0, "plotters")
from plotter_utils import get_condition_summary

sys.path.insert(0, "manuscript/figures")
from fig_utils import COARSE_CFGS, MARKER_SIZE, EDGE_COLOR, EDGE_WIDTH, compute_jitter

# ── Style constants ──────────────────────────────────────────────────────
ARCHITECTURES = [
    ("alexnet", "pca_labels_alexnet", "AlexNet"),
    ("clip",    "pca_labels_clip",    "CLIP"),
    ("pixels",  "pca_labels_pixels",  "Pixels"),
]
ARCH_STYLE = {
    "alexnet": {"color": "#6baed6", "marker": "o"},
    "clip":    {"color": "#08519c", "marker": "s"},
    "pixels":  {"color": "#c0a898", "marker": "v"},
}
BASELINE_1K_COLOR = "#e8963e"
RAW_XLIM_RIGHT = 180


# ── Data fetching ────────────────────────────────────────────────────────

@lru_cache(maxsize=32)
def fetch_arch_data(dataset, folder, region):
    """Fetch means and asymmetric error bars for coarse-label conditions.

    Returns (means, errs_lo, errs_hi) arrays of length len(COARSE_CFGS).
    Uses bootstrap CIs (via get_condition_summary) for both NSD and TVSD.
    """
    means, ci_lo, ci_hi = [], [], []
    for cfg in COARSE_CFGS:
        s = get_condition_summary(dataset, region, folder, cfg,
                                  "spearman", epoch=20, analysis="rsa")
        means.append(s["mean"])
        ci_lo.append(s["ci_low"])
        ci_hi.append(s["ci_high"])
    means = np.array(means)
    errs_lo = np.array([max(m - lo, 0) if not np.isnan(lo) else 0
                        for m, lo in zip(means, ci_lo)])
    errs_hi = np.array([max(hi - m, 0) if not np.isnan(hi) else 0
                        for m, hi in zip(means, ci_hi)])
    return means, errs_lo, errs_hi


@lru_cache(maxsize=16)
def fetch_baseline(dataset, region, epoch=20):
    """Fetch 1000-way baseline mean score."""
    s = get_condition_summary(dataset, region, "imagenet1k", 1000,
                              "spearman", epoch=epoch, analysis="rsa")
    return s["mean"]


@lru_cache(maxsize=16)
def fetch_baseline_ci(dataset, region, epoch=20):
    """Fetch 1000-way baseline (mean, ci_low, ci_high)."""
    s = get_condition_summary(dataset, region, "imagenet1k", 1000,
                              "spearman", epoch=epoch, analysis="rsa")
    return s["mean"], s["ci_low"], s["ci_high"]


@lru_cache(maxsize=16)
def fetch_coarse_ci(dataset, folder, region, cfg):
    """Fetch (mean, ci_low, ci_high) for a single coarse condition."""
    s = get_condition_summary(dataset, region, folder, cfg,
                              "spearman", epoch=20, analysis="rsa")
    return s["mean"], s["ci_low"], s["ci_high"]


# ── Axis formatting ──────────────────────────────────────────────────────

def format_xaxis(ax, show_xlabel, xlim_right=RAW_XLIM_RIGHT):
    """Log-2 x-axis with COARSE_CFGS ticks."""
    ax.set_xscale("log", base=2)
    coarse_set = set(COARSE_CFGS)
    ax.xaxis.set_major_locator(FixedLocator(COARSE_CFGS))
    if show_xlabel:
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda val, pos: str(int(val)) if int(round(val)) in coarse_set else ""))
        ax.set_xlabel(r"Training cls ($2^i$ supervision bits)", fontsize=9, labelpad=4)
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: ""))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.tick_params(axis="x", which="major", length=3.5, width=0.7, labelsize=8)
    ax.set_xlim(1.5, xlim_right)


def format_yaxis(ax, fmt_str=".2f", tick_interval=None):
    """Y-axis with minor ticks, grid, and trimmed numeric labels."""
    if tick_interval is not None:
        from matplotlib.ticker import MultipleLocator
        ax.yaxis.set_major_locator(MultipleLocator(tick_interval))
    ax.tick_params(axis="y", which="major", direction="out", length=3.5, width=0.6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", direction="out", length=2, width=0.4)
    ax.yaxis.grid(True, which="major", color="#F0F0F0", linewidth=0.3, zorder=0)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:{fmt_str}}".rstrip("0").rstrip(".")))
