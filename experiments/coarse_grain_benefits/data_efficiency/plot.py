"""
Plot data efficiency results: THINGS behavioral alignment vs dataset size.

Usage (from project root):
    python experiments/coarse_grain_benefits/data_efficiency/plot.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "data_efficiency.csv")
OUT_PATH = os.path.join(SCRIPT_DIR, "data_efficiency.png")

# Style — matches manuscript Figure 4
sns.set_theme(style="ticks", context="paper", font_scale=1.2)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
})

DATASET_ORDER = ["imagenet-mini-10", "imagenet-mini-50", "imagenet-full"]
DATASET_LABELS = {"imagenet-mini-10": "10", "imagenet-mini-50": "50",
                  "imagenet-full": "~1300"}
CONDITION_LABELS = {8: "Coarse (8-class)", 1000: "Fine (1000-class)"}
COLORS = {8: "#08519c", 1000: "#d4822e"}
BAR_EDGE_COLOR = "#333333"
BAR_EDGE_WIDTH = 0.6

BAR_WIDTH = 0.32
BAR_GAP = 0.04
Y_MIN = 0.0


def rounded_top_bar(x, y_bottom, y_top, width, radius):
    """Bar with flat bottom and rounded top corners only."""
    hw = width / 2
    r = min(radius, (y_top - y_bottom) / 2, hw)
    left, right = x - hw, x + hw

    verts = [
        (left, y_bottom),
        (left, y_top - r),
        (left, y_top),
        (left + r, y_top),
        (right - r, y_top),
        (right, y_top),
        (right, y_top - r),
        (right, y_bottom),
        (left, y_bottom),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CLOSEPOLY,
    ]
    return MplPath(verts, codes)


def main():
    df = pd.read_csv(CSV_PATH)

    # Take max score across epochs for each (dataset, condition)
    best = df.loc[df.groupby(["dataset", "condition"])["score"].idxmax()].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(3.8, 3.2))

    x = np.arange(len(DATASET_ORDER))
    conditions = [8, 1000]
    legend_patches = []

    # Subtle horizontal gridlines
    ax.yaxis.grid(True, linestyle="-", alpha=0.15, color="#aaaaaa", zorder=0)
    ax.set_axisbelow(True)

    for i, cond in enumerate(conditions):
        scores, ci_lows, ci_highs = [], [], []
        for ds in DATASET_ORDER:
            row = best[(best["dataset"] == ds) & (best["condition"] == cond)]
            if len(row) > 0:
                row = row.iloc[0]
                scores.append(row["score"])
                ci_lows.append(row["score"] - row["ci_low"])
                ci_highs.append(row["ci_high"] - row["score"])
            else:
                scores.append(0)
                ci_lows.append(0)
                ci_highs.append(0)

        offset = (i - 0.5) * (BAR_WIDTH + BAR_GAP)
        positions = x + offset

        for j, (pos, score) in enumerate(zip(positions, scores)):
            path = rounded_top_bar(pos, Y_MIN, score, BAR_WIDTH, radius=0.03)
            patch = mpatches.PathPatch(
                path, facecolor=COLORS[cond], edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH, zorder=2,
            )
            ax.add_patch(patch)

        # Error bars
        ax.errorbar(positions, scores, yerr=[ci_lows, ci_highs],
                    fmt="none", ecolor="#333333", capsize=2, linewidth=0.7,
                    capthick=0.7, zorder=4)

        legend_patches.append(mpatches.Patch(facecolor=COLORS[cond],
                                              edgecolor=BAR_EDGE_COLOR,
                                              linewidth=BAR_EDGE_WIDTH,
                                              label=CONDITION_LABELS[cond]))

    ax.set_xlabel("Images per class", fontsize=9.5, labelpad=5)
    ax.set_ylabel("THINGS alignment (Spearman $\\rho$)", fontsize=9.5, labelpad=5)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in DATASET_ORDER], fontsize=9)
    ax.tick_params(axis="y", labelsize=8.5)

    # "(full)" sub-label beneath ~1300, smaller and grey
    full_idx = DATASET_ORDER.index("imagenet-full")
    ax.annotate("(full)", xy=(full_idx, 0), xycoords=("data", "axes fraction"),
                xytext=(0, -24), textcoords="offset points",
                ha="center", va="top", fontsize=7, color="#555555")

    ax.set_ylim(Y_MIN, 0.66)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax.set_xlim(-0.45, len(DATASET_ORDER) - 0.55)

    ax.legend(handles=legend_patches, frameon=False, fontsize=7.5, loc="upper left",
              handlelength=1.0, handleheight=0.7, borderpad=0.2, labelspacing=0.3)

    sns.despine(ax=ax, offset=5)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved to {OUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
