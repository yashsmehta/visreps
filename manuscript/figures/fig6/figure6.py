"""Figure 6: Architecture Generalization — THINGS Behavioral Alignment.

Layout (single row, 3 panels):
  a: ResNet-50 | b: ConvNeXt | c: ViT-B/16
  Each panel shows THINGS coarseness (CLIP labels, epoch 20, seed 1)
  with a 1000-class baseline bar.

Usage:
    python manuscript/figures/fig6/figure6.py
"""

import os
import sys
import json
import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import (
    AutoMinorLocator, FuncFormatter, FixedLocator, NullLocator,
)
from matplotlib.transforms import blended_transform_factory
import seaborn as sns

sys.path.insert(0, "manuscript/figures")
from fig_utils import (
    COARSE_CFGS, MARKER_SIZE, EDGE_COLOR, EDGE_WIDTH, BREAK_1K_POS,
    setup_style, draw_xaxis_break, draw_torn_ci_band,
)

# ── Config ────────────────────────────────────────────────────────────────
OUTPUT_DIR = "manuscript/figures/fig6"
DB_PATH = "results.db"

ARCH_MODELS = [
    ("ResNet50",      "ResNet-50"),
    ("ConvNeXt_Base", "ConvNeXt"),
    ("ViTBase",       "ViT-B/16"),
]

# Match Figure 3 color scheme
CLIP_STYLE = {"color": "#08519c", "marker": "s"}   # dark blue square
BASELINE_1K_COLOR = "#e8963e"                        # warm amber
BAR_CENTER = 250
BAR_WIDTH_FRAC = 0.15


# ── Data fetching ─────────────────────────────────────────────────────────

def fetch_things_arch_data(model_name, epoch=20, seed=1):
    """Fetch THINGS-behavior scores for a specific architecture (CLIP labels).

    Returns (coarse_dict, baseline_dict_or_None, untrained_score_or_None).
    coarse_dict: {cfg_id: {score, ci_low, ci_high}} for coarse conditions.
    """
    conn = sqlite3.connect(DB_PATH)
    results = {}

    # ── Coarse conditions (CLIP labels) ──
    for cfg in COARSE_CFGS:
        df = pd.read_sql("""
            SELECT r.run_id, r.score, r.ci_low, r.ci_high,
                   bd.scores AS boot_scores
            FROM results r
            LEFT JOIN bootstrap_distributions bd
                ON r.run_id = bd.run_id AND bd.compare_method = 'spearman'
            WHERE r.neural_dataset = 'things-behavior'
              AND r.model_name = ? AND r.cfg_id = ? AND r.epoch = ?
              AND r.seed = ? AND r.pca_labels_folder = 'pca_labels_clip'
              AND r.compare_method = 'spearman' AND r.reconstruct_from_pcs = 0
            ORDER BY r.score DESC LIMIT 1
        """, conn, params=[model_name, cfg, epoch, seed])
        if not df.empty:
            row = df.iloc[0]
            ci_low, ci_high = row["ci_low"], row["ci_high"]
            if row["boot_scores"] is not None:
                boots = np.array(json.loads(row["boot_scores"]))
                ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
            results[cfg] = {"score": row["score"],
                            "ci_low": ci_low, "ci_high": ci_high}

    # ── 1000-way baseline ──
    df_1k = pd.read_sql("""
        SELECT r.run_id, r.score, r.ci_low, r.ci_high,
               bd.scores AS boot_scores
        FROM results r
        LEFT JOIN bootstrap_distributions bd
            ON r.run_id = bd.run_id AND bd.compare_method = 'spearman'
        WHERE r.neural_dataset = 'things-behavior'
          AND r.model_name = ? AND r.cfg_id = 1000 AND r.epoch = ?
          AND r.seed = ? AND r.compare_method = 'spearman'
          AND r.reconstruct_from_pcs = 0
        ORDER BY r.score DESC LIMIT 1
    """, conn, params=[model_name, epoch, seed])
    baseline = None
    if not df_1k.empty:
        row = df_1k.iloc[0]
        ci_low, ci_high = row["ci_low"], row["ci_high"]
        if row["boot_scores"] is not None:
            boots = np.array(json.loads(row["boot_scores"]))
            ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
        baseline = {"score": row["score"],
                    "ci_low": ci_low, "ci_high": ci_high}

    # ── Untrained (epoch=0) ──
    df_un = pd.read_sql("""
        SELECT score FROM results
        WHERE neural_dataset = 'things-behavior'
          AND model_name = ? AND epoch = 0 AND seed = ?
          AND compare_method = 'spearman' AND reconstruct_from_pcs = 0
        ORDER BY score DESC LIMIT 1
    """, conn, params=[model_name, seed])
    untrained = df_un.iloc[0]["score"] if not df_un.empty else None

    conn.close()
    return results, baseline, untrained


# ── Plotting ──────────────────────────────────────────────────────────────

def _draw_bar_break(ax):
    """Draw // break marks between the coarse scatter region and the bar."""
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    mid = np.exp((np.log(64) + np.log(BAR_CENTER)) / 2)
    rect_hw = mid * 0.16
    rect = mpatches.FancyBboxPatch(
        (mid / 1.16, -0.05), width=rect_hw * 1.5, height=0.10,
        boxstyle="square,pad=0", facecolor="white", edgecolor="none",
        transform=trans, clip_on=False, zorder=9)
    ax.add_patch(rect)
    for x_shift in [0.93, 1.07]:
        x_c = mid * x_shift
        ax.plot([x_c / 1.04, x_c * 1.04], [-0.028, 0.028],
                transform=trans, color="#555555", linewidth=0.7,
                clip_on=False, zorder=11)


def plot_things_coarseness(ax, model_name, display_name,
                           show_ylabel=True, show_xlabel=True,
                           forced_ylim=None):
    """Plot THINGS coarseness panel — clean compact style with dashed reference lines."""
    results, baseline, untrained = fetch_things_arch_data(model_name)

    if not results and baseline is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888")
        return

    all_y_vals = []

    # ── Coarse scatter points ──
    for cfg in COARSE_CFGS:
        if cfg not in results:
            continue
        r = results[cfg]
        all_y_vals.append(r["score"])
        err_lo = max(r["score"] - r["ci_low"], 0) if pd.notna(r["ci_low"]) else 0
        err_hi = max(r["ci_high"] - r["score"], 0) if pd.notna(r["ci_high"]) else 0
        ax.errorbar(cfg, r["score"],
                    yerr=[[err_lo], [err_hi]],
                    fmt=CLIP_STYLE["marker"], color=CLIP_STYLE["color"],
                    markersize=MARKER_SIZE,
                    markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                    capsize=1.5, capthick=0.5,
                    ecolor=CLIP_STYLE["color"], elinewidth=0.7, zorder=4)

    # ── Collect y-range ──
    if baseline:
        all_y_vals.append(baseline["score"])
    if untrained is not None:
        all_y_vals.append(untrained)

    y_min = min(all_y_vals)
    y_max = max(all_y_vals)
    y_range = y_max - y_min if y_max > y_min else 0.05

    # ── 1000-way baseline: CI band + bounded dashed mean + orange diamond ──
    if baseline:
        bl_err_lo = max(baseline["score"] - baseline["ci_low"], 0) if pd.notna(baseline["ci_low"]) else 0
        bl_err_hi = max(baseline["ci_high"] - baseline["score"], 0) if pd.notna(baseline["ci_high"]) else 0

        # CI band: pale orange horizontal span over coarse region (torn later)
        if pd.notna(baseline["ci_low"]) and pd.notna(baseline["ci_high"]):
            ax.fill_between([1.5, BREAK_1K_POS],
                            baseline["ci_low"], baseline["ci_high"],
                            facecolor=BASELINE_1K_COLOR, alpha=0.12,
                            edgecolor="none", zorder=1)

        # Dashed mean line: bounded to coarse region (torn later)
        ax.plot([1.5, BREAK_1K_POS],
                [baseline["score"], baseline["score"]],
                color=BASELINE_1K_COLOR, linestyle="--",
                linewidth=1.0, alpha=0.6, zorder=2, clip_on=False)

        ax.errorbar(BREAK_1K_POS, baseline["score"],
                    yerr=[[bl_err_lo], [bl_err_hi]],
                    fmt="D", color=BASELINE_1K_COLOR, markersize=MARKER_SIZE,
                    markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                    capsize=1.5, capthick=0.5,
                    ecolor=BASELINE_1K_COLOR, elinewidth=0.7, zorder=5)

    # ── Untrained dashed line (zorder=3 so mask at 2.6 doesn't erase it) ──
    if untrained is not None:
        ax.axhline(untrained, color="#AAAAAA", linestyle="--",
                   linewidth=0.9, alpha=0.7, zorder=3)
        y_offset = y_range * 0.015
        ax.text(0.97, untrained + y_offset, "Untrained",
                fontsize=6, fontstyle="italic", color="#AAAAAA",
                ha="right", va="bottom",
                transform=ax.get_yaxis_transform(), zorder=10)

    # ── Axis formatting — broken x-axis with 1000 ──
    ax.set_xscale("log", base=2)
    all_x = COARSE_CFGS + [BREAK_1K_POS]
    label_map = {v: str(v) for v in COARSE_CFGS}
    label_map[BREAK_1K_POS] = "1000"
    ax.xaxis.set_major_locator(FixedLocator(all_x))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda val, pos: label_map.get(int(round(val)), "")))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.tick_params(axis="x", which="major", length=3.5, width=0.6, labelsize=10)
    ax.set_xlim(1.5, BREAK_1K_POS * 1.5)

    ax.tick_params(axis="y", which="major", direction="out", length=3.5,
                   width=0.6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", direction="out", length=2,
                   width=0.4)
    ax.yaxis.grid(True, which="major", color="#F0F0F0", linewidth=0.3,
                  zorder=0)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".")))

    if forced_ylim is not None:
        yl = forced_ylim[0] if forced_ylim[0] is not None else y_min - y_range * 0.12
        yh = forced_ylim[1] if forced_ylim[1] is not None else y_max + y_range * 0.10
        ax.set_ylim(yl, yh)
    else:
        ax.set_ylim(y_min - y_range * 0.12, y_max + y_range * 0.10)

    # ── Tear through the 1000-way CI band (after ylim is final) ──
    if baseline and pd.notna(baseline["ci_low"]) and pd.notna(baseline["ci_high"]):
        draw_torn_ci_band(ax, baseline["ci_low"], baseline["ci_high"],
                          BASELINE_1K_COLOR)

    if show_xlabel:
        ax.set_xlabel("Granularity", fontsize=9, labelpad=6)
    if show_ylabel:
        ax.set_ylabel(r"RSA (Spearman $\rho$)", fontsize=9, labelpad=3)
    else:
        ax.set_ylabel("")
    sns.despine(ax=ax, right=True, top=True, offset=3)
    draw_xaxis_break(ax)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    setup_style()
    plt.rcParams.update({
        "axes.labelsize": 10.5,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })

    # ── Figure layout: single row, 3 panels ──
    fig = plt.figure(figsize=(13, 4.8))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.28,
                           left=0.07, right=0.97, top=0.82, bottom=0.15)

    # Pre-fetch data to compute shared y-limits for ResNet-50 & ConvNeXt
    arch_data = {}
    for model_name, display_name in ARCH_MODELS:
        arch_data[model_name] = fetch_things_arch_data(model_name)

    # Shared ylim for ResNet-50 and ConvNeXt (indices 0 and 1)
    shared_y_vals = []
    for mn in ["ResNet50", "ConvNeXt_Base"]:
        results, baseline, untrained = arch_data[mn]
        for r in results.values():
            shared_y_vals.extend([r["score"] - (r["score"] - r["ci_low"])
                                  if pd.notna(r["ci_low"]) else r["score"],
                                  r["ci_high"]
                                  if pd.notna(r["ci_high"]) else r["score"]])
        if baseline:
            shared_y_vals.append(baseline["score"])
        if untrained is not None:
            shared_y_vals.append(untrained)

    shared_ymin = min(shared_y_vals)
    shared_ymax = max(shared_y_vals)
    shared_range = shared_ymax - shared_ymin
    shared_ylim = (0.1, shared_ymax + shared_range * 0.08)

    axes = []
    for i, (model_name, display_name) in enumerate(ARCH_MODELS):
        ax = fig.add_subplot(gs[0, i])
        ylim = shared_ylim if i < 2 else (0.1, None)  # All start at 0.1
        plot_things_coarseness(ax, model_name, display_name,
                               show_ylabel=(i == 0), show_xlabel=True,
                               forced_ylim=ylim)
        axes.append(ax)

    # Architecture subtitles above each panel
    for i, (_, display_name) in enumerate(ARCH_MODELS):
        pos = axes[i].get_position()
        x_center = (pos.x0 + pos.x1) / 2
        fig.text(x_center, pos.y1 + 0.012, display_name,
                 fontsize=9, color="#888888",
                 ha="center", va="bottom", family="sans-serif")

    # Legend in first panel
    coarse_handle = Line2D([], [], marker=CLIP_STYLE["marker"],
                           color="none",
                           markerfacecolor=CLIP_STYLE["color"],
                           markeredgecolor=EDGE_COLOR,
                           markeredgewidth=EDGE_WIDTH,
                           markersize=5.5, label="Coarse labels\n(CLIP)")
    axes[0].legend(handles=[coarse_handle],
                   fontsize=7.5, frameon=True, fancybox=False,
                   framealpha=0.92, edgecolor="#dddddd",
                   borderpad=0.4, handletextpad=0.3,
                   labelspacing=0.25,
                   loc="center left",
                   bbox_to_anchor=(0.0, 0.35))

    # ── Panel labels: a, b, c ──
    top_y = axes[0].get_position().y1
    label_y = top_y + 0.035
    for i, label in enumerate(["a", "b", "c"]):
        pos = axes[i].get_position()
        fig.text(pos.x0 - 0.03, label_y, label,
                 fontsize=14, fontweight="bold", va="bottom", ha="left",
                 family="sans-serif")

    # ── Save ──
    out = f"{OUTPUT_DIR}/figure6.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white",
                edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
