"""Panel D: Low Data Regime — THINGS alignment with 10K training images.

Mirrors Panel B style: log2 x-axis with coarse markers (2-64, CLIP labels),
plus two horizontal dashed lines for 1000-way at 10K and full 1.2M ImageNet.
"""

import os
import sys

import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, FixedLocator, NullLocator
import seaborn as sns

sys.path.insert(0, "plotters")
from plotter_utils import get_condition_summary

sys.path.insert(0, "manuscript/figures")
from fig_utils import COARSE_CFGS, MARKER_SIZE, EDGE_COLOR, EDGE_WIDTH

COARSE_CFGS_SET = set(COARSE_CFGS)

# ── Constants ────────────────────────────────────────────────────────────
BASELINE_1K_COLOR = "#e8963e"
CLIP_STYLE = {"color": "#08519c", "marker": "s"}

LEGACY_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../experiments/coarse_grain_benefits/data_efficiency/"
    "legacy_results/data_efficiency_results.csv")


def _load_data_10k():
    """Load THINGS scores from legacy CSV (CLIP labels, epoch 200, 10K images)."""
    df = pd.read_csv(LEGACY_CSV)
    things = df[(df["benchmark"] == "things") &
                (df["dataset"] == "imagenet-mini-10") &
                (df["epoch"] == 200)]
    results = {}
    for cond in COARSE_CFGS + [1000]:
        cdf = things[things["condition"] == cond]
        if cdf.empty:
            continue
        best = cdf.loc[cdf["score"].idxmax()]
        results[cond] = {
            "score": best["score"],
            "ci_low": best["ci_low"],
            "ci_high": best["ci_high"],
        }
    return results


def _load_data_1m():
    """Load 1.2M 1000-class THINGS result (mean over seeds)."""
    s = get_condition_summary("things-behavior", "N/A", "imagenet1k", 1000,
                              "spearman", epoch=20, analysis="rsa")
    if np.isnan(s["mean"]):
        return None
    return {"score": s["mean"], "ci_low": s["ci_low"], "ci_high": s["ci_high"]}


def plot_data_efficiency(ax, ref_ax=None):
    """Plot low data regime panel. Syncs y-axis with ref_ax if provided."""
    data_10k = _load_data_10k()
    data_1m = _load_data_1m()

    # Sync y-axis with Panel B if available (limits already set by plot_coarseness)
    if ref_ax is not None:
        y_lo, y_hi = ref_ax.get_ylim()
    else:
        all_y = [d["score"] for d in data_10k.values()]
        if data_1m:
            all_y.append(data_1m["score"])
        y_min, y_max = (min(all_y), max(all_y)) if all_y else (0, 1)
        y_range = y_max - y_min
        y_lo, y_hi = y_min - y_range * 0.12, y_max + y_range * 0.10
    ax.set_ylim(y_lo, y_hi)

    # Dashed line: 1000-way trained on full 1.2M ImageNet
    if data_1m:
        ax.axhline(y=data_1m["score"], color=BASELINE_1K_COLOR, linestyle="--",
                   linewidth=1.1, alpha=0.85, zorder=2)
        ax.text(180 * 0.95, data_1m["score"] + (y_hi - y_lo) * 0.015,
                "1,000 classes (100% training data)",
                fontsize=6, fontstyle="italic", color=BASELINE_1K_COLOR,
                ha="right", va="bottom", zorder=10)

    # Dashed line: 1000-way trained on 10K images
    if 1000 in data_10k:
        d = data_10k[1000]
        ax.axhline(y=d["score"], color=BASELINE_1K_COLOR, linestyle="--",
                   linewidth=1.1, alpha=0.85, zorder=2)
        ax.text(180 * 0.95, d["score"] + (y_hi - y_lo) * 0.015,
                "1,000 classes (1% training data)",
                fontsize=6, fontstyle="italic", color=BASELINE_1K_COLOR,
                ha="right", va="bottom", zorder=10)

    # Coarse markers
    for cond in COARSE_CFGS:
        if cond not in data_10k:
            continue
        d = data_10k[cond]
        e_lo = max(d["score"] - d["ci_low"], 0) if not np.isnan(d["ci_low"]) else 0
        e_hi = max(d["ci_high"] - d["score"], 0) if not np.isnan(d["ci_high"]) else 0
        ax.errorbar(cond, d["score"], yerr=[[e_lo], [e_hi]],
                    fmt=CLIP_STYLE["marker"], color=CLIP_STYLE["color"],
                    markersize=MARKER_SIZE,
                    markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                    capsize=1.5, capthick=0.5,
                    ecolor=CLIP_STYLE["color"], elinewidth=0.7, zorder=4)

    # ── Axis formatting ──
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(COARSE_CFGS))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda val, pos: str(int(val)) if int(round(val)) in COARSE_CFGS_SET else ""))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.tick_params(axis="x", which="major", direction="out")
    ax.set_xlim(1.5, 180)

    ax.set_ylabel("")
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".")))
    ax.tick_params(axis="y", which="major", direction="out")
    ax.tick_params(axis="y", which="minor", direction="out")
    ax.yaxis.grid(True, which="major", color="#F0F0F0", linewidth=0.3, zorder=0)
    ax.set_axisbelow(True)

    ax.set_xlabel("Training classes", fontsize=9, labelpad=6)
    sns.despine(ax=ax, right=True, top=True, offset=4)

    ax.set_title("Low Data Regime", fontsize=11, fontweight="semibold", pad=8)
    ax.text(0.5, 0.96, "10K images (1% of ImageNet)", transform=ax.transAxes,
            fontsize=7.5, color="#888888", ha="center", va="top")
