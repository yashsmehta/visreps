"""Supplementary Figure S8: Seed Variability.

Two separate figures:
  (a) Neural: 2x2 grid — TVSD (V1, IT) top row, NSD (Early, Ventral) bottom row
  (b) Behavioral: single THINGS panel

Each panel shows individual seed scores (dots with distinct markers) and a
dashed mean line, across CLIP coarse conditions (8, 16, 32) and 1000-way,
with a broken x-axis between them.

Usage:
    python manuscript/figures/supplementary/supp_s8_seed_variability.py
"""

import sqlite3
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, FixedLocator, FuncFormatter, NullLocator
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, ".")
sys.path.insert(0, "manuscript/figures")
from fig_utils import (
    setup_style, BREAK_1K_POS, draw_xaxis_break,
    EDGE_COLOR, EDGE_WIDTH, MARKER_SIZE,
)

OUTPUT_DIR = "manuscript/figures/supplementary/S9_seed_variability"
DB_PATH = "results.db"

# Coarse conditions (CLIP labels, full range) + 1000-way baseline
COARSE_CFGS = [2, 4, 8, 16, 32, 64]
CLIP_COLOR = "#08519c"       # CLIP dark blue (matches S1A / Fig 3)
BASELINE_1K_COLOR = "#e8963e"

# Seed markers
SEED_MARKERS = {1: "o", 2: "s", 3: "^"}
SEED_JITTER = {1: -0.06, 2: 0.0, 3: 0.06}  # multiplicative jitter in log-space

# Neural panel definitions: (row, col, dataset, region, show_ylabel, show_xlabel, show_untrained_label)
NEURAL_PANELS = [
    (0, 0, "tvsd", "V1",                    True,  False, False),
    (0, 1, "tvsd", "IT",                    False, False, False),
    (1, 0, "nsd",  "early visual stream",   True,  True,  True),
    (1, 1, "nsd",  "ventral visual stream", False, True,  True),
]


def get_best_layer_results(conn, cfg_id, pca_labels, pca_folder_filter,
                           neural_dataset, region, seed):
    """Get the best-layer score for a single (condition, seed)."""
    where_parts = [
        f"cfg_id = {cfg_id}",
        "compare_method = 'spearman'",
        f"neural_dataset = '{neural_dataset}'",
        f"seed = {seed}",
        "epoch = 20",
        "reconstruct_from_pcs = 0",
        "analysis = 'rsa'",
    ]
    if region != "N/A":
        where_parts.append(f"region = '{region}'")
    if pca_labels:
        where_parts.append("pca_labels = 1")
        where_parts.append(f"pca_labels_folder LIKE '%{pca_folder_filter}%'")
    else:
        where_parts.append("pca_labels = 0")
        where_parts.append("pca_labels_folder = 'imagenet1k'")

    where = " AND ".join(where_parts)

    q = f"""
    SELECT r.subject_idx, r.score
    FROM results r
    INNER JOIN (
        SELECT subject_idx, MAX(score) as max_score
        FROM results
        WHERE {where}
        GROUP BY subject_idx
    ) best ON r.subject_idx = best.subject_idx AND r.score = best.max_score
    WHERE {where}
    """
    df = pd.read_sql(q, conn)
    if df.empty:
        return np.nan

    df = df.drop_duplicates(subset="subject_idx", keep="first")
    return df["score"].mean()


def _get_untrained_mean(conn, neural_dataset, region):
    """Get mean untrained (epoch=0) score across subjects."""
    where_parts = [
        "cfg_id = 1000",
        "compare_method = 'spearman'",
        f"neural_dataset = '{neural_dataset}'",
        "epoch = 0",
        "reconstruct_from_pcs = 0",
        "analysis = 'rsa'",
    ]
    if region != "N/A":
        where_parts.append(f"region = '{region}'")
    where = " AND ".join(where_parts)

    df = pd.read_sql(f"SELECT score FROM results WHERE {where}", conn)
    if df.empty:
        return np.nan
    return df["score"].mean()


def _collect_panel_data(conn, neural_dataset, region):
    """Collect per-seed scores for coarse (8, 16, 32) and 1000-way."""
    coarse_data = {}  # cfg -> [seed1, seed2, seed3]
    for cfg in COARSE_CFGS:
        scores = []
        for seed in [1, 2, 3]:
            s = get_best_layer_results(conn, cfg, True, "clip",
                                       neural_dataset, region, seed)
            scores.append(s)
        coarse_data[cfg] = scores

    baseline_scores = []
    for seed in [1, 2, 3]:
        s = get_best_layer_results(conn, 1000, False, "",
                                   neural_dataset, region, seed)
        baseline_scores.append(s)

    untrained_mean = _get_untrained_mean(conn, neural_dataset, region)

    return coarse_data, baseline_scores, untrained_mean


def _format_broken_xaxis(ax, show_xlabel):
    """Log-2 x-axis with coarse ticks + broken gap before 1000."""
    ax.set_xscale("log", base=2)
    all_x = COARSE_CFGS + [BREAK_1K_POS]
    label_map = {v: str(v) for v in COARSE_CFGS}
    label_map[BREAK_1K_POS] = "1000"
    ax.xaxis.set_major_locator(FixedLocator(all_x))
    if show_xlabel:
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda val, pos: label_map.get(int(round(val)), "")))
        ax.set_xlabel("Granularity", fontsize=9, labelpad=4)
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: ""))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.tick_params(axis="x", which="major", length=3.5, width=0.7, labelsize=8)
    ax.set_xlim(1.5, BREAK_1K_POS * 1.5)


def _format_yaxis(ax):
    """Y-axis with minor ticks, grid, and trimmed labels."""
    ax.tick_params(axis="y", which="major", direction="out", length=3.5, width=0.6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", direction="out", length=2, width=0.4)
    ax.yaxis.grid(True, which="major", color="#F0F0F0", linewidth=0.3, zorder=0)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".")))


def plot_seed_panel(ax, conn, neural_dataset, region,
                    show_ylabel=True, show_xlabel=True,
                    show_untrained_label=False):
    """Draw one seed variability panel: scatter dots + dashed mean lines."""
    coarse_data, baseline_scores, untrained_mean = _collect_panel_data(
        conn, neural_dataset, region)

    all_y = []
    if not np.isnan(untrained_mean):
        all_y.append(untrained_mean)

    # Coarse conditions (8, 16, 32) — all in CLIP color
    for cfg in COARSE_CFGS:
        scores = coarse_data[cfg]
        valid = [s for s in scores if not np.isnan(s)]
        all_y.extend(valid)

        for seed_idx, seed in enumerate([1, 2, 3]):
            if np.isnan(scores[seed_idx]):
                continue
            jitter = 2 ** (SEED_JITTER[seed] * 0.6)
            ax.scatter(
                cfg * jitter, scores[seed_idx],
                marker=SEED_MARKERS[seed], color=CLIP_COLOR,
                s=50, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH, zorder=4,
            )

        # Dashed mean line
        if valid:
            mean_val = np.mean(valid)
            half_w = cfg * 0.12
            ax.plot([cfg / (1 + 0.10), cfg * (1 + 0.10)],
                    [mean_val, mean_val],
                    color=CLIP_COLOR, linestyle="--", linewidth=1.2,
                    alpha=0.7, zorder=3)

    # 1000-way baseline — orange at break position
    valid_bl = [s for s in baseline_scores if not np.isnan(s)]
    all_y.extend(valid_bl)

    for seed_idx, seed in enumerate([1, 2, 3]):
        if np.isnan(baseline_scores[seed_idx]):
            continue
        jitter = 2 ** (SEED_JITTER[seed] * 0.6)
        ax.scatter(
            BREAK_1K_POS * jitter, baseline_scores[seed_idx],
            marker=SEED_MARKERS[seed], color=BASELINE_1K_COLOR,
            s=50, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH, zorder=4,
        )

    if valid_bl:
        mean_bl = np.mean(valid_bl)
        ax.plot([BREAK_1K_POS / (1 + 0.10), BREAK_1K_POS * (1 + 0.10)],
                [mean_bl, mean_bl],
                color=BASELINE_1K_COLOR, linestyle="--", linewidth=1.2,
                alpha=0.7, zorder=3)

    # Untrained baseline — gray dashed line
    if not np.isnan(untrained_mean):
        ax.axhline(untrained_mean, color="#AAAAAA", linestyle="--",
                    linewidth=1.25, alpha=0.6, zorder=2)
        if show_untrained_label:
            ax.text(0.97, untrained_mean, "Untrained",
                    fontsize=8, fontstyle="italic", color="#999999",
                    ha="right", va="bottom",
                    transform=ax.get_yaxis_transform(), zorder=10)

    # Axis formatting
    if not all_y:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color="#888")
        return

    y_min, y_max = min(all_y), max(all_y)
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.15, y_max + y_range * 0.12)

    _format_broken_xaxis(ax, show_xlabel)
    draw_xaxis_break(ax)
    _format_yaxis(ax)

    if show_ylabel:
        ax.set_ylabel(r"RSA (Spearman $\rho$)", fontsize=9, labelpad=4)
    else:
        ax.set_ylabel("")
    sns.despine(ax=ax, right=True, top=True, offset=3)


def _build_legend():
    """Legend: just the 3 seed markers."""
    return [
        Line2D([], [], marker=SEED_MARKERS[s], color="#666666",
               linestyle="None", markersize=6, markeredgecolor="white",
               markeredgewidth=0.5, label=f"Seed {s}")
        for s in [1, 2, 3]
    ]


def generate_neural():
    """S8A: Neural — 2x2 grid (TVSD top | NSD bottom) x (Early | Higher)."""
    setup_style()
    plt.rcParams.update({
        "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    })

    conn = sqlite3.connect(DB_PATH)

    fig = plt.figure(figsize=(11, 7))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.30, wspace=0.25,
                           left=0.09, right=0.96, top=0.87, bottom=0.10)

    axes = {}
    for row, col, ds, region, ylabel, xlabel, untrained_label in NEURAL_PANELS:
        ax = fig.add_subplot(gs[row, col])
        print(f"Drawing: {ds} / {region}")
        plot_seed_panel(ax, conn, ds, region,
                        show_ylabel=ylabel, show_xlabel=xlabel,
                        show_untrained_label=untrained_label)
        axes[(row, col)] = ax

    # Region sub-titles
    region_titles = {
        (0, 0): "V1", (0, 1): "IT",
        (1, 0): "Early visual stream", (1, 1): "Ventral visual stream",
    }
    for key, label in region_titles.items():
        axes[key].set_title(label, fontsize=9.5, fontweight="medium",
                            color="#333333", pad=7)

    # Row headers (left side, following S1A)
    fig.canvas.draw()
    for row, title in [(0, "TVSD"), (1, "NSD")]:
        pos = axes[(row, 0)].get_position()
        fig.text(0.02, (pos.y0 + pos.y1) / 2 + 0.015, title,
                 fontsize=11, fontweight="bold", color="#1a1a1a",
                 ha="center", va="center", rotation=90)

    # Column headers
    for col, label in [(0, "Early Visual Cortex"), (1, "Higher Visual Cortex")]:
        pos = axes[(0, col)].get_position()
        fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.050, label,
                 fontsize=11, fontweight="bold", color="#333333",
                 ha="center", va="bottom")

    # Panel labels (a–d)
    for (row, col), label in zip([(0, 0), (0, 1), (1, 0), (1, 1)], "abcd"):
        pos = axes[(row, col)].get_position()
        fig.text(pos.x0 - 0.018, pos.y1 + 0.022, label,
                 fontsize=13, fontweight="bold", va="bottom", ha="left")

    # Legend in panel b — right side, positioned between data and untrained line
    axes[(0, 1)].legend(
        handles=_build_legend(), fontsize=7.5, frameon=True,
        fancybox=False, framealpha=0.92, edgecolor="#dddddd",
        borderpad=0.5, handletextpad=0.4, labelspacing=0.3,
        loc="right", bbox_to_anchor=(1.0, 0.35))

    conn.close()

    out = f"{OUTPUT_DIR}/S9_seed_variability_neural.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


def generate_behavioral():
    """S8B: THINGS behavioral — single seed variability panel."""
    setup_style()
    plt.rcParams.update({
        "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    })

    conn = sqlite3.connect(DB_PATH)

    fig, ax = plt.subplots(figsize=(5, 3.5))

    print("Drawing: things-behavior / N/A")
    plot_seed_panel(ax, conn, "things-behavior", "N/A",
                    show_ylabel=True, show_xlabel=True,
                    show_untrained_label=True)

    ax.set_title("THINGS (Behavioral)", fontsize=11, fontweight="bold",
                 color="#333333", pad=8)

    ax.legend(handles=_build_legend(), fontsize=7.5, frameon=True,
              fancybox=False, framealpha=0.92, edgecolor="#dddddd",
              borderpad=0.5, handletextpad=0.4, labelspacing=0.3,
              loc="center left", bbox_to_anchor=(0.0, 0.35))

    conn.close()

    out = f"{OUTPUT_DIR}/S9_seed_variability_behavioral.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


def main():
    generate_neural()
    generate_behavioral()


if __name__ == "__main__":
    main()
