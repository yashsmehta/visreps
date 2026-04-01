"""Supplementary Figure S17: Seed Variability.

Shows how alignment scores vary across 3 random seeds for:
  - 1000-class standard ImageNet model
  - CLIP coarse-grained models at 8, 16, and 32 classes

Across three benchmarks (ordered from invasive to behavioral):
  (A) TVSD IT (macaque electrophysiology)
  (B) NSD ventral visual stream (human fMRI)
  (C) THINGS behavioral similarity

Each panel plots individual seed scores (dots) with 95% bootstrap CIs
(error bars) and the cross-seed mean (horizontal line).

Usage:
    python manuscript/figures/supplementary/supp_s17_seed_variability.py
"""

import sqlite3
import sys

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, ".")
from manuscript.figures.fig_utils import (
    BASELINE_1K_COLOR,
    GRAN_COLORS,
    setup_style,
)

OUTPUT = "manuscript/figures/supplementary/supp_s17_seed_variability.png"
DB_PATH = "results.db"

# Model conditions: (label, cfg_id, pca_labels, pca_folder_filter)
CONDITIONS = [
    ("1000", 1000, False, ""),
    ("CLIP-8", 8, True, "clip"),
    ("CLIP-16", 16, True, "clip"),
    ("CLIP-32", 32, True, "clip"),
]

# Panels ordered: invasive → fMRI → behavior
PANELS = [
    ("tvsd", "IT", "TVSD IT"),
    ("nsd", "ventral visual stream", "NSD Ventral"),
    ("things-behavior", "N/A", "THINGS Behavioral"),
]

PANEL_LABELS = ["a", "b", "c"]

SEED_MARKERS = {1: "o", 2: "s", 3: "^"}

# Color for each condition
CONDITION_COLORS = {
    "1000": BASELINE_1K_COLOR,
    "CLIP-8": GRAN_COLORS[8],
    "CLIP-16": GRAN_COLORS[16],
    "CLIP-32": GRAN_COLORS[32],
}

# X-axis labels
CONDITION_XLABELS = ["1000", "8\n(CLIP)", "16\n(CLIP)", "32\n(CLIP)"]


def get_best_layer_results(conn, cfg_id, pca_labels, pca_folder_filter,
                           neural_dataset, region, seed):
    """Get the best-layer score + bootstrap CIs for a single (condition, seed).

    Returns (score, ci_low, ci_high) averaged across subjects.
    """
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
    SELECT r.subject_idx, r.score, r.ci_low, r.ci_high
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
        return np.nan, np.nan, np.nan

    df = df.drop_duplicates(subset="subject_idx", keep="first")
    return (df["score"].mean(), df["ci_low"].mean(), df["ci_high"].mean())


def main():
    setup_style()
    conn = sqlite3.connect(DB_PATH)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))

    # ── Collect data ──────────────────────────────────────────────────────
    all_data = {}
    all_ci_low = {}
    all_ci_high = {}
    panel_ranges = {}

    for ax_idx, (nd, region, title) in enumerate(PANELS):
        panel_vals = []
        for cond_idx, (label, cfg_id, pca, folder) in enumerate(CONDITIONS):
            scores, ci_lows, ci_highs = [], [], []
            for seed in [1, 2, 3]:
                s, cl, ch = get_best_layer_results(conn, cfg_id, pca, folder,
                                                   nd, region, seed)
                scores.append(s)
                ci_lows.append(cl)
                ci_highs.append(ch)
                for v in [s, cl, ch]:
                    if not np.isnan(v):
                        panel_vals.append(v)
            all_data[(ax_idx, cond_idx)] = scores
            all_ci_low[(ax_idx, cond_idx)] = ci_lows
            all_ci_high[(ax_idx, cond_idx)] = ci_highs

        ymin = min(panel_vals)
        ymax = max(panel_vals)
        yrange = ymax - ymin
        pad_bot = max(yrange * 0.22, 0.003)
        pad_top = max(yrange * 0.30, 0.003)
        panel_ranges[ax_idx] = (ymin - pad_bot, ymax + pad_top)

    # ── Draw ──────────────────────────────────────────────────────────────
    bar_width = 0.54
    jitter = [-0.10, 0.0, 0.10]

    for ax_idx, (nd, region, title) in enumerate(PANELS):
        ax = axes[ax_idx]
        ybot, ytop = panel_ranges[ax_idx]
        ax.set_ylim(ybot, ytop)

        for cond_idx, (label, cfg_id, pca, folder) in enumerate(CONDITIONS):
            seed_scores = all_data[(ax_idx, cond_idx)]
            ci_lows = all_ci_low[(ax_idx, cond_idx)]
            ci_highs = all_ci_high[(ax_idx, cond_idx)]
            color = CONDITION_COLORS[label]
            mean_val = np.nanmean(seed_scores)
            x = cond_idx

            # Translucent bar
            ax.bar(x, mean_val - ybot, bottom=ybot, width=bar_width,
                   color=color, alpha=0.22, edgecolor="none", zorder=2)

            # Per-seed error bars + dots
            for seed_idx in range(3):
                score = seed_scores[seed_idx]
                cl = ci_lows[seed_idx]
                ch = ci_highs[seed_idx]
                if np.isnan(score):
                    continue
                xj = x + jitter[seed_idx]

                # Error bar (95% bootstrap CI)
                yerr_lo = score - cl if not np.isnan(cl) else 0
                yerr_hi = ch - score if not np.isnan(ch) else 0
                ax.errorbar(
                    xj, score,
                    yerr=[[yerr_lo], [yerr_hi]],
                    fmt="none", ecolor=color, elinewidth=0.9,
                    capsize=2.0, capthick=0.7, alpha=0.40, zorder=3,
                )

                # Seed dot
                ax.scatter(
                    xj, score,
                    marker=SEED_MARKERS[seed_idx + 1],
                    color=color, s=46, edgecolor="white",
                    linewidth=0.6, zorder=4,
                )

            # Mean tick with white halo
            line = ax.hlines(mean_val, x - 0.18, x + 0.18, colors="#222222",
                             linewidth=1.6, zorder=5)
            line.set_path_effects([
                pe.withStroke(linewidth=3.0, foreground="white"),
                pe.Normal(),
            ])

        ax.set_xlim(-0.55, len(CONDITIONS) - 0.45)

        # X-axis
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(CONDITION_XLABELS, fontsize=7, linespacing=0.85)

        # Title and panel label
        ax.set_title(title, fontsize=9.5, fontweight="semibold", pad=8)
        ax.text(-0.14, 1.15, PANEL_LABELS[ax_idx], transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top", ha="left")

        # Y-axis
        if ax_idx == 0:
            ax.set_ylabel(r"Spearman $\rho$", fontsize=9, labelpad=4)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5]))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="y", which="major", direction="out", length=4,
                       width=0.8, labelsize=7.5)
        ax.tick_params(axis="y", which="minor", direction="out", length=2.5,
                       width=0.5)
        ax.tick_params(axis="x", which="major", length=0, width=0, pad=3)
        ax.yaxis.grid(True, which="major", color="#F0F0F0", linewidth=0.4,
                      zorder=0)
        ax.set_axisbelow(True)

        ax.set_xlabel("Number of classes", fontsize=8, labelpad=8)
        sns.despine(ax=ax, right=True, top=True, offset=4)

    # ── Legend ────────────────────────────────────────────────────────────
    seed_handles = [
        Line2D([], [], marker=SEED_MARKERS[s], color="#666666",
               linestyle="None", markersize=5.5, markeredgecolor="white",
               markeredgewidth=0.4, label=f"Seed {s}")
        for s in [1, 2, 3]
    ]
    seed_handles.append(
        Line2D([], [], color="#222222", linewidth=1.6, label="Mean")
    )
    seed_handles.append(
        Line2D([], [], color="#888888", linewidth=0.9, marker="|",
               markersize=6, markeredgewidth=0.7, alpha=0.5,
               label="95% bootstrap CI")
    )
    fig.legend(handles=seed_handles, loc="upper center", ncol=5,
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.02),
               columnspacing=1.0, handletextpad=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.92], w_pad=3.0)
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight", facecolor="white",
                edgecolor="none")
    plt.close()
    print(f"Saved: {OUTPUT}")
    conn.close()


if __name__ == "__main__":
    main()
