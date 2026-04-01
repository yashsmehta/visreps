"""Supplementary Figure S11: WordNet Hierarchy Labels — Coarseness Comparison.

Single-row bar plot (5 panels):
  TVSD V1 | TVSD IT | NSD Early | NSD Ventral | THINGS

Each panel shows WordNet coarseness levels (2, 3, 4, 10, 20, 57) as blue-gradient
bars, with horizontal reference lines for the 1000-way model and best PCA-coarse model.

Usage:
    python manuscript/figures/supplementary/supp_s11_wordnet.py
"""

import sys
import sqlite3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
from matplotlib.lines import Line2D
import seaborn as sns

sys.path.insert(0, ".")
sys.path.insert(0, "plotters")
sys.path.insert(0, "manuscript/figures")

from plotter_utils import get_condition_summary, get_bootstrap_ci
from fig_utils import setup_style, find_best_coarse_model, DB_PATH

OUTPUT_DIR = "manuscript/figures/supplementary"

# ── WordNet config ────────────────────────────────────────────────────────
WORDNET_CFGS = [2, 3, 4, 10, 20, 57]
WORDNET_FOLDER = "pca_labels_wordnet"

# Blue gradient for WordNet bars (light → dark, 6 levels)
WORDNET_COLORS = {
    2:  "#c6dbef",
    3:  "#9ecae1",
    4:  "#6baed6",
    10: "#4292c6",
    20: "#2171b5",
    57: "#084594",
}

# Reference line styles
COLOR_1K = "#e6550d"         # orange-red (matches main figures)
COLOR_BEST_COARSE = "#2ca02c"  # green

# ── Panels ────────────────────────────────────────────────────────────────
PANELS = [
    ("tvsd", "V1",                    "V1"),
    ("tvsd", "IT",                    "IT"),
    ("nsd",  "early visual stream",   "Early Visual"),
    ("nsd",  "ventral visual stream", "Ventral Visual"),
    ("things-behavior", "N/A",        "THINGS"),
]


def _draw_rounded_bar(ax, x, height, width, color, hatch="", zorder=3,
                      edgecolor="#555555", alpha=1.0):
    rect = mpatches.FancyBboxPatch(
        (x - width / 2, 0), width, height,
        boxstyle=mpatches.BoxStyle("Round", pad=0.012, rounding_size=0.05),
        facecolor=color, edgecolor=edgecolor, alpha=alpha,
        linewidth=0.5, hatch=hatch, mutation_aspect=0.04, zorder=zorder,
    )
    ax.add_patch(rect)


def get_wordnet_summary(neural_dataset, region, cfg_id):
    """Get mean score (+ cross-subject SEM) for a WordNet condition."""
    conn = sqlite3.connect(DB_PATH)
    df = __import__("pandas").read_sql("""
        SELECT score FROM results
        WHERE neural_dataset = ? AND region = ? AND cfg_id = ?
          AND pca_labels_folder = ? AND compare_method = 'spearman'
          AND reconstruct_from_pcs = 0 AND epoch = 20
    """, conn, params=[neural_dataset, region, cfg_id, WORDNET_FOLDER])
    conn.close()

    if df.empty:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    mean = df["score"].mean()
    if len(df) > 1:
        sem = df["score"].std() / np.sqrt(len(df))
        ci_low = mean - 1.96 * sem
        ci_high = mean + 1.96 * sem
    else:
        ci_low, ci_high = np.nan, np.nan

    return {"mean": mean, "ci_low": ci_low, "ci_high": ci_high}


def plot_wordnet_panel(ax, neural_dataset, region, title, show_ylabel=True):
    """Draw one panel: WordNet bars + 1K and best-coarse reference lines."""
    bar_width = 0.55
    n = len(WORDNET_CFGS)
    x = np.arange(n, dtype=float)

    # Fetch WordNet scores
    vals = [get_wordnet_summary(neural_dataset, region, cfg) for cfg in WORDNET_CFGS]

    # Reference lines
    baseline_1k = get_condition_summary(
        neural_dataset, region, "imagenet1k", 1000, "spearman", epoch=20, analysis="rsa")
    best_coarse = find_best_coarse_model(neural_dataset, region)

    # y-axis range
    all_means = [v["mean"] for v in vals if not np.isnan(v["mean"])]
    ref_vals = [baseline_1k["mean"], best_coarse["mean"]]
    ref_vals = [v for v in ref_vals if not np.isnan(v)]
    y_max = max(all_means + ref_vals) * 1.22 if (all_means or ref_vals) else 0.1
    ax.set_ylim(0, y_max)

    # Reference lines
    if not np.isnan(baseline_1k["mean"]):
        ax.axhline(baseline_1k["mean"], color=COLOR_1K, linestyle="-",
                   linewidth=1.0, zorder=1, alpha=0.7)
    if not np.isnan(best_coarse["mean"]):
        ax.axhline(best_coarse["mean"], color=COLOR_BEST_COARSE, linestyle="--",
                   linewidth=0.9, zorder=1, alpha=0.7)

    # Draw bars
    plt.rcParams["hatch.color"] = "#666666"
    for i, (cfg, val) in enumerate(zip(WORDNET_CFGS, vals)):
        mean = val["mean"]
        if np.isnan(mean):
            continue
        _draw_rounded_bar(ax, x[i], mean, bar_width, WORDNET_COLORS[cfg],
                          hatch="/", edgecolor="#444444")
        ci_lo, ci_hi = val.get("ci_low", np.nan), val.get("ci_high", np.nan)
        if not np.isnan(ci_lo) and not np.isnan(ci_hi):
            err_lo = max(mean - ci_lo, 0)
            err_hi = max(ci_hi - mean, 0)
            if err_lo > 0 or err_hi > 0:
                ax.errorbar(x[i], mean, yerr=[[err_lo], [err_hi]],
                            fmt="none", ecolor="#333333", elinewidth=0.6,
                            capsize=2, capthick=0.6, zorder=5)

    # Axis formatting
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in WORDNET_CFGS], fontsize=7)
    ax.set_xlabel("WordNet classes", fontsize=8, labelpad=4)
    ax.set_xlim(-0.55, n - 0.45)
    ax.tick_params(axis="x", direction="out", bottom=False, length=0, pad=2)

    if show_ylabel:
        ax.set_ylabel(r"Spearman $\rho$", fontsize=8, labelpad=3)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=5, color="#2c2c2c")

    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, pos: "" if np.isclose(v, 0) else f"{v:.2f}"))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="major", direction="out", length=3,
                   width=0.5, labelsize=6.5)
    ax.tick_params(axis="y", which="minor", direction="out", length=1.8, width=0.4)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, right=True, top=True, offset=3)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["left"].set_linewidth(0.6)


def main():
    setup_style()
    plt.rcParams.update({
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "font.size": 7,
    })

    fig = plt.figure(figsize=(14, 3.6))
    gs = gridspec.GridSpec(
        1, 5, figure=fig,
        wspace=0.12,
        top=0.76, bottom=0.17, left=0.05, right=0.98,
    )

    # Dataset section headers
    header_y = 0.87
    header_kw = dict(fontsize=9.5, fontweight="bold", ha="center", va="bottom",
                     color="#2c2c2c", fontfamily="sans-serif")

    axes = [fig.add_subplot(gs[0, i]) for i in range(5)]

    for idx, (ax, (nd, region, title)) in enumerate(zip(axes, PANELS)):
        print(f"Drawing: {nd} / {region}")
        plot_wordnet_panel(ax, nd, region, title, show_ylabel=(idx == 0))

    # Section headers
    def _mid_x(ax):
        return ax.get_position().x0 + ax.get_position().width / 2

    tvsd_center = (_mid_x(axes[0]) + _mid_x(axes[1])) / 2
    nsd_center = (_mid_x(axes[2]) + _mid_x(axes[3])) / 2
    things_center = _mid_x(axes[4])

    fig.text(tvsd_center, header_y, "TVSD (Macaque)", **header_kw)
    fig.text(nsd_center, header_y, "NSD (Human fMRI)", **header_kw)
    fig.text(things_center, header_y, "THINGS (Behavioral)", **header_kw)

    # Vertical separators
    def _gap_x(ax_l, ax_r):
        return (ax_l.get_position().x1 + ax_r.get_position().x0) / 2

    for sx in [_gap_x(axes[1], axes[2]), _gap_x(axes[3], axes[4])]:
        fig.add_artist(Line2D(
            [sx, sx], [0.10, 0.80], transform=fig.transFigure,
            color="#DCDCDC", linewidth=0.5, linestyle="-", zorder=0))

    # Shared legend centered at top
    legend_handles = [
        mpatches.Patch(facecolor="#4292c6", edgecolor="#444444", linewidth=0.5,
                       hatch="/", label="WordNet coarse"),
        Line2D([], [], color=COLOR_1K, linestyle="-", linewidth=1.0,
               alpha=0.7, label="1000-way"),
        Line2D([], [], color=COLOR_BEST_COARSE, linestyle="--", linewidth=0.9,
               alpha=0.7, label="Best PCA-coarse"),
    ]
    fig.legend(handles=legend_handles, fontsize=7, loc="upper center",
               bbox_to_anchor=(0.5, 0.99), frameon=True, edgecolor="#dddddd",
               fancybox=False, handletextpad=0.4, borderpad=0.4,
               labelspacing=0.3, framealpha=0.94, ncol=3)

    out = f"{OUTPUT_DIR}/supp_s11_wordnet.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
