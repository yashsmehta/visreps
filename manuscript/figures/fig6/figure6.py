"""Figure 6: Data Efficiency — Coarse vs Fine-Grained Training.

Layout:
  [Schematic placeholder]  [Data-efficiency bars]

Panel A: Schematic of the data-efficiency paradigm (placeholder)
Panel B: Paired bars — coarse (8-class) vs fine (1000-class) at 4 data
         scales (5, 10, 50, ~1300 images per class)

Usage:
    python manuscript/figures/fig6/figure6.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter
import seaborn as sns

sys.path.insert(0, "manuscript/figures")
from fig_utils import setup_style, draw_schematic_placeholder

# ── Config ────────────────────────────────────────────────────────────────
OUTPUT_DIR = "manuscript/figures/fig6"
COARSE_BAR_COLOR = "#08519c"   # dark blue
BASELINE_1K_COLOR = "#d4822e"  # warm amber

DATA_EFF_CSV = os.path.join("experiments", "coarse_grain_benefits",
                            "data_efficiency", "data_efficiency.csv")

DATA_SCALES = [
    ("5",     "imagenet-mini-5"),
    ("10",    "imagenet-mini-10"),
    ("50",    "imagenet-mini-50"),
    ("~1300", "imagenet-full"),
]


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


def plot_data_efficiency(ax):
    """Panel B: Paired bars at 4 data scales."""
    eff_df = pd.read_csv(DATA_EFF_CSV)
    best_eff = eff_df.loc[
        eff_df.groupby(["dataset", "condition"])["score"].idxmax()
    ].reset_index(drop=True)

    n_groups = len(DATA_SCALES)
    group_spacing = 1.5
    bar_w = 0.48
    bar_gap = 0.06
    group_centers = [i * group_spacing for i in range(n_groups)]

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
                        ecolor="#333333", capsize=3.5, capthick=0.7,
                        elinewidth=0.7, zorder=4)

    # "(full)" sub-label beneath the last group
    full_cx = group_centers[-1]
    ax.annotate("(full)", xy=(full_cx, 0), xycoords=("data", "axes fraction"),
                xytext=(0, -18), textcoords="offset points",
                ha="center", va="top", fontsize=8, color="#666666")

    # ── Axis formatting ──
    ax.set_xlim(-0.8, group_centers[-1] + 0.8)
    ax.set_ylim(0.2, 0.66)
    ax.set_ylabel(r"Spearman $\rho$", fontsize=11, labelpad=5)
    ax.set_title("Data Efficiency: Coarse vs. Fine",
                 fontsize=12, fontweight="semibold", pad=10)

    ax.set_xticks(group_centers)
    ax.set_xticklabels([lbl for lbl, _ in DATA_SCALES], fontsize=9)
    ax.set_xlabel("Images per class", fontsize=11, labelpad=8)

    ax.yaxis.grid(True, which="major", color="#EBEBEB", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.tick_params(axis="y", which="major", direction="out", length=5,
                   width=1.2, labelsize=10)
    ax.tick_params(axis="y", which="minor", direction="out", length=3,
                   width=0.8)
    ax.tick_params(axis="x", which="major", length=4, width=1.0, direction="out")

    sns.despine(ax=ax, right=True, top=True, offset=5)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_linewidth(1.2)

    # Legend
    leg_handles = [
        mpatches.Patch(facecolor=COARSE_BAR_COLOR, edgecolor="#333333",
                       linewidth=0.6, label="Coarse (8-class)"),
        mpatches.Patch(facecolor=BASELINE_1K_COLOR, edgecolor="#333333",
                       linewidth=0.6, label="Fine (1000-class)"),
    ]
    leg = ax.legend(handles=leg_handles, fontsize=9, frameon=True,
                    loc="upper left", edgecolor="#dddddd", fancybox=False,
                    framealpha=0.95, handletextpad=0.4, borderpad=0.4,
                    labelspacing=0.3)
    leg.get_frame().set_linewidth(0.4)


def main():
    setup_style()

    fig = plt.figure(figsize=(11, 5.0))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30,
                           width_ratios=[0.8, 1.2],
                           left=0.08, right=0.96, top=0.90, bottom=0.12)

    # Panel A: Schematic placeholder
    ax_schematic = fig.add_subplot(gs[0, 0])
    draw_schematic_placeholder(ax_schematic,
                               "Data Efficiency\nParadigm\n(schematic)")

    # Panel B: Data-efficiency bars
    ax_bars = fig.add_subplot(gs[0, 1])
    plot_data_efficiency(ax_bars)

    # Panel labels
    for ax, label, x_off in zip(
        [ax_schematic, ax_bars], ["A", "B"], [-0.08, -0.10]):
        ax.text(x_off, 1.08, label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="left",
                family="sans-serif")

    out = f"{OUTPUT_DIR}/figure6.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white",
                edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
