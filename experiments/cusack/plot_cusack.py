"""Coarseness plots for Cusack infant fMRI dataset.

Generates two separate 1×2 figures (EVC, VVC):
  - cusack_2month.png
  - cusack_9month.png

Style identical to manuscript Figure 2: AlexNet + CLIP + Pixels,
warm amber 1000-way bar, untrained dashed line (if data available),
axis break, in-plot legend.

Usage:
    python experiments/cusack/plot_cusack.py
"""

import sys
import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, AutoMinorLocator
import seaborn as sns

sys.path.insert(0, "manuscript/figures")
from fig_utils import (
    COARSE_CFGS, MARKER_SIZE, EDGE_COLOR, EDGE_WIDTH,
    setup_style, compute_jitter,
)

DB_PATH = "results.db"
OUTPUT_DIR = "experiments/cusack"

# ── Color scheme (identical to Figure 2) ─────────────────────────────────
ARCHITECTURES = [
    ("alexnet", "pca_labels_alexnet", "AlexNet"),
    ("clip",    "pca_labels_clip",    "CLIP"),
    ("pixels",  "pca_labels_pixels",  "Pixels"),
]
ARCH_STYLE = {
    "alexnet": {"color": "#6baed6", "marker": "o"},   # medium blue
    "clip":    {"color": "#08519c", "marker": "s"},    # dark blue
    "pixels":  {"color": "#c0a898", "marker": "v"},    # muted tan
}
BASELINE_1K_COLOR = "#e8963e"    # warm amber
UNTRAINED_BAR_COLOR = "#999999"  # medium gray

# ── Bar positions (identical to Figure 2) ────────────────────────────────
BAR_CENTER = 250
BAR_LEFT = BAR_CENTER / 1.16     # untrained  (~216)
BAR_RIGHT = BAR_CENTER * 1.16    # trained    (~290)
BAR_WIDTH_FRAC = 0.15

REGIONS = [("evc", "EVC"), ("vvc", "VVC")]


# ── Data fetching ────────────────────────────────────────────────────────

def _query_scores(region, pca_labels_folder, cfg_id, subject_idx, epoch=20):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT seed, score, layer
        FROM results
        WHERE neural_dataset = 'cusack'
          AND region = ? AND pca_labels_folder = ? AND cfg_id = ?
          AND subject_idx = ? AND compare_method = 'spearman'
          AND analysis = 'rsa' AND epoch = ?
    """, conn, params=[region, pca_labels_folder, cfg_id, subject_idx, epoch])
    conn.close()
    return df


def _seed_sem(df):
    if df.empty:
        return np.nan, 0
    seed_scores = df.groupby("seed")["score"].max()
    mean = seed_scores.mean()
    sem = seed_scores.std() / np.sqrt(len(seed_scores)) if len(seed_scores) > 1 else 0
    return mean, sem


# ── Drawing helpers (from Figure 2) ──────────────────────────────────────

def _draw_bar_break(ax):
    """Draw // break marks between the coarse scatter region and the bars."""
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    mid = np.exp((np.log(64) + np.log(BAR_LEFT)) / 2)
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


def _make_tick_formatter(label_map):
    def _fmt(val, pos):
        for k, lbl in label_map.items():
            if abs(val - k) < k * 0.05:
                return lbl
        return ""
    return _fmt


# ── Panel plotting (mirrors plot_raw_coarseness from Figure 2) ───────────

def plot_panel(ax, region, subject_idx, show_ylabel=True, show_xlabel=True,
               show_untrained_label=True):
    # ── Baselines ──
    bl_df = _query_scores(region, "imagenet1k", 1000, subject_idx, epoch=20)
    bl_mean, bl_sem = _seed_sem(bl_df)

    un_df = _query_scores(region, "imagenet1k", 1000, subject_idx, epoch=0)
    un_mean, un_sem = _seed_sem(un_df)

    if np.isnan(bl_mean) or bl_mean == 0:
        ax.text(0.5, 0.5, "No baseline", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888")
        return

    # ── 1) Architecture scatter points ──
    all_y_vals = [bl_mean]
    if not np.isnan(un_mean):
        all_y_vals.append(un_mean)

    for arch_idx, (arch_key, folder, _) in enumerate(ARCHITECTURES):
        style = ARCH_STYLE[arch_key]
        jitter = compute_jitter(arch_idx, len(ARCHITECTURES))

        for i, cfg in enumerate(COARSE_CFGS):
            df = _query_scores(region, folder, cfg, subject_idx)
            m, s = _seed_sem(df)
            if np.isnan(m):
                continue
            all_y_vals.append(m)
            err = 1.96 * s
            ax.errorbar(cfg * jitter, m,
                        yerr=[[err], [err]],
                        fmt=style["marker"], color=style["color"],
                        markersize=MARKER_SIZE,
                        markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                        capsize=1.5, capthick=0.5,
                        ecolor=style["color"], elinewidth=0.7, zorder=4)

    # ── 2) Y-axis range (no forced zero) ──
    y_min = min(all_y_vals)
    y_max = max(all_y_vals)
    y_range = y_max - y_min
    y_bottom = y_min - y_range * 0.12

    # ── 3) Untrained dashed line + label ──
    if not np.isnan(un_mean):
        ax.axhline(un_mean, color="#AAAAAA", linestyle="--",
                    linewidth=0.9, alpha=0.7, zorder=1)
        if show_untrained_label:
            y_offset = (y_max - y_min) * 0.03
            ax.text(0.02, un_mean + y_offset, " Untrained",
                    fontsize=6, fontstyle="italic", color="#AAAAAA",
                    ha="left", va="bottom",
                    transform=blended_transform_factory(ax.transAxes, ax.transData),
                    zorder=10)

    # ── 4) 1000-way trained bar ──
    bl_err = 1.96 * bl_sem
    ax.bar(BAR_CENTER, bl_mean - y_bottom, bottom=y_bottom,
           width=BAR_CENTER * BAR_WIDTH_FRAC,
           color=BASELINE_1K_COLOR, edgecolor="#c07830",
           linewidth=0.4, zorder=3)
    ax.errorbar(BAR_CENTER, bl_mean,
                yerr=[[bl_err], [bl_err]],
                fmt="none", ecolor="#555555", elinewidth=0.7,
                capsize=2.2, capthick=0.6, zorder=5)

    # ── 5) Axis formatting (identical to Figure 2) ──
    ax.set_xscale("log", base=2)

    all_ticks = COARSE_CFGS + [BAR_CENTER]
    label_map = {v: str(v) for v in COARSE_CFGS}
    label_map[BAR_CENTER] = "1000"
    ax.xaxis.set_major_locator(FixedLocator(all_ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(_make_tick_formatter(label_map)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.tick_params(axis="x", which="major", length=3.5, width=0.6)
    if not show_xlabel:
        ax.set_xticklabels([""] * len(all_ticks))
    ax.set_xlim(1.5, BAR_CENTER * 1.35)

    # Y-axis
    ax.tick_params(axis="y", which="major", direction="out", length=3.5, width=0.6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", direction="out", length=2, width=0.4)
    ax.yaxis.grid(True, which="major", color="#F0F0F0", linewidth=0.3, zorder=0)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".")))

    # Small top margin
    cur_ylim = ax.get_ylim()
    ax.set_ylim(cur_ylim[0], cur_ylim[1] + y_range * 0.03)

    if show_xlabel:
        ax.set_xlabel("ImageNet training classes", fontsize=9, labelpad=6)
    if show_ylabel:
        ax.set_ylabel(r"RSA (Spearman $\rho$)", fontsize=9, labelpad=3)
    else:
        ax.set_ylabel("")
    sns.despine(ax=ax, right=True, top=True, offset=3)

    _draw_bar_break(ax)


# ── Figure generation ────────────────────────────────────────────────────

def make_figure(subject_idx, subject_label, filename):
    setup_style()
    plt.rcParams.update({
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })

    fig = plt.figure(figsize=(8, 3.2))
    axes = [fig.add_subplot(1, 2, i + 1) for i in range(2)]

    for col, (region, _) in enumerate(REGIONS):
        plot_panel(axes[col], region, subject_idx,
                   show_ylabel=(col == 0), show_xlabel=True,
                   show_untrained_label=True)

    # ── Column headers (bold, above panels — Figure 2 style) ──
    col_headers = ["Early Visual Cortex", "Ventral Visual Cortex"]
    for col, header in enumerate(col_headers):
        pos = axes[col].get_position()
        x_center = (pos.x0 + pos.x1) / 2
        y_top = pos.y1
        fig.text(x_center, y_top + 0.058, header,
                 fontsize=11.5, fontweight="bold", color="#1a1a1a",
                 ha="center", va="bottom", family="sans-serif")

    # ── Region subtitles (gray, below header) ──
    region_subtitles = ["EVC", "VVC"]
    for col, subtitle in enumerate(region_subtitles):
        pos = axes[col].get_position()
        x_center = (pos.x0 + pos.x1) / 2
        y_top = pos.y1
        fig.text(x_center, y_top + 0.012, subtitle,
                 fontsize=8, color="#888888",
                 ha="center", va="bottom", family="sans-serif")

    # ── In-plot legend in left panel (Figure 2 style) ──
    arch_handles = []
    for arch_key, _, display in ARCHITECTURES:
        style = ARCH_STYLE[arch_key]
        h = Line2D([], [], marker=style["marker"], color="none",
                   markerfacecolor=style["color"],
                   markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                   markersize=5.5, label=display)
        arch_handles.append(h)
    leg = axes[0].legend(
        handles=arch_handles, fontsize=7.5,
        frameon=True, fancybox=False, framealpha=0.92,
        edgecolor="#dddddd", borderpad=0.4,
        handletextpad=0.3, labelspacing=0.25,
        title="Coarse label source",
        title_fontsize=7,
        loc="center left",
        bbox_to_anchor=(0.0, 0.40),
    )
    leg._legend_box.align = "left"

    # ── Title ──
    fig.suptitle(f"Cusack — {subject_label}",
                 fontsize=12, fontweight="bold", y=1.12)

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/{filename}"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved → {out}")
    plt.close()


def main():
    make_figure("2month", "2-month-olds", "cusack_2month.png")
    make_figure("9month", "9-month-olds", "cusack_9month.png")


if __name__ == "__main__":
    main()
