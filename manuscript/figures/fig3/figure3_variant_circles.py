"""Figure 3 variant: Divided circle visualization for class counts.

Identical to figure3.py but replaces dot-density grids below the x-axis
with divided circle (pie) icons. Each circle is split into N equal slices
(N = number of classes), with alternating dark/light fills. For large N
(e.g., 1000), the slices are so thin the circle appears nearly solid.

Usage:
    python manuscript/figures/fig3/figure3_variant_circles.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge, Circle
import seaborn as sns

sys.path.insert(0, "plotters")
from plotter_utils import get_condition_summary, query_best_scores

sys.path.insert(0, "manuscript/figures")
from fig_utils import (
    COARSE_CFGS, BREAK_1K_POS,
    UNTRAINED_LINE_STYLE, MARKER_SIZE, EDGE_COLOR, EDGE_WIDTH,
    setup_style, compute_jitter,
    format_coarseness_axes,
)

OUTPUT_DIR = "manuscript/figures/fig3"

# ── Figure 3 color scheme (AlexNet/CLIP = blue shades, 1K = amber) ──
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
BASELINE_1K_COLOR = "#e8963e"  # warm amber


# ── NSD data fetching ────────────────────────────────────────────────────

def fetch_nsd_arch_data(folder, region):
    means, ci_lo, ci_hi = [], [], []
    for cfg in COARSE_CFGS:
        s = get_condition_summary("nsd", region, folder, cfg,
                                  "spearman", epoch=20, analysis="rsa")
        means.append(s["mean"])
        ci_lo.append(s["ci_low"])
        ci_hi.append(s["ci_high"])
    return np.array(means), np.array(ci_lo), np.array(ci_hi)


def fetch_nsd_baseline(region, epoch=20):
    s = get_condition_summary("nsd", region, "imagenet1k", 1000,
                              "spearman", epoch=epoch, analysis="rsa")
    return s["mean"], s["ci_low"], s["ci_high"]


# ── TVSD data fetching ───────────────────────────────────────────────────

def _sem_summary(df):
    seed_means = df.groupby("seed")["score"].mean()
    mean = seed_means.mean()
    sem = seed_means.std() / np.sqrt(len(seed_means)) if len(seed_means) > 1 else 0
    return mean, sem


def fetch_tvsd_arch_data(folder, region):
    means, sems = [], []
    for cfg in COARSE_CFGS:
        df = query_best_scores("tvsd", region, folder, cfg,
                               "spearman", epoch=20, analysis="rsa")
        if df.empty:
            means.append(np.nan)
            sems.append(0)
            continue
        m, s = _sem_summary(df)
        means.append(m)
        sems.append(s)
    return np.array(means), np.array(sems)


def fetch_tvsd_baseline(region, epoch=20):
    df = query_best_scores("tvsd", region, "imagenet1k", 1000,
                           "spearman", epoch=epoch, analysis="rsa")
    if df.empty:
        return np.nan, np.nan, np.nan
    m, s = _sem_summary(df)
    return m, m - 1.96 * s, m + 1.96 * s


# ── Coarseness plotting ─────────────────────────────────────────────────

def plot_raw_coarseness(ax, dataset, region, show_ylabel=True, show_xlabel=True):
    """Plot coarseness panel with raw Spearman rho values."""

    if dataset == "nsd":
        bl_mean, bl_ci_lo, bl_ci_hi = fetch_nsd_baseline(region, epoch=20)
        un_mean, un_ci_lo, un_ci_hi = fetch_nsd_baseline(region, epoch=0)
    else:
        bl_mean, bl_ci_lo, bl_ci_hi = fetch_tvsd_baseline(region, epoch=20)
        un_mean, un_ci_lo, un_ci_hi = fetch_tvsd_baseline(region, epoch=0)

    if np.isnan(bl_mean) or bl_mean == 0:
        ax.text(0.5, 0.5, "No baseline", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888")
        return

    # Untrained dashed line
    if not np.isnan(un_mean):
        ax.axhline(un_mean, **UNTRAINED_LINE_STYLE, zorder=1)

    # 1000-way horizontal reference line + diamond
    ax.axhline(bl_mean, color=BASELINE_1K_COLOR, linestyle="-",
               linewidth=0.6, alpha=0.35, zorder=2)
    bl_err_lo = max(bl_mean - bl_ci_lo, 0) if not np.isnan(bl_ci_lo) else 0
    bl_err_hi = max(bl_ci_hi - bl_mean, 0) if not np.isnan(bl_ci_hi) else 0
    ax.errorbar(BREAK_1K_POS, bl_mean, yerr=[[bl_err_lo], [bl_err_hi]],
                fmt="D", color=BASELINE_1K_COLOR, markersize=MARKER_SIZE,
                markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                capsize=2, capthick=0.7, ecolor=BASELINE_1K_COLOR,
                elinewidth=1.0, zorder=5)

    # Architectures (AlexNet, CLIP, Pixels — no ViT)
    for arch_idx, (arch_key, folder, _) in enumerate(ARCHITECTURES):
        style = ARCH_STYLE[arch_key]
        if dataset == "nsd":
            means, ci_lo, ci_hi = fetch_nsd_arch_data(folder, region)
            errs_lo = np.array([max(means[i] - ci_lo[i], 0) if not np.isnan(ci_lo[i]) else 0
                                for i in range(len(means))])
            errs_hi = np.array([max(ci_hi[i] - means[i], 0) if not np.isnan(ci_hi[i]) else 0
                                for i in range(len(means))])
        else:
            means, sems = fetch_tvsd_arch_data(folder, region)
            errs_lo = 1.96 * sems
            errs_hi = 1.96 * sems

        jitter = compute_jitter(arch_idx, len(ARCHITECTURES))

        for i, cfg in enumerate(COARSE_CFGS):
            if np.isnan(means[i]):
                continue
            ax.errorbar(cfg * jitter, means[i],
                        yerr=[[errs_lo[i]], [errs_hi[i]]],
                        fmt=style["marker"], color=style["color"],
                        markersize=MARKER_SIZE,
                        markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                        capsize=2, capthick=0.7,
                        ecolor=style["color"], elinewidth=1.0, zorder=4)

    format_coarseness_axes(ax, "", show_ylabel=show_ylabel,
                           show_xlabel=show_xlabel)


# ── Class count divided circles ──────────────────────────────────────────

def draw_class_count_circles(fig, bottom_axes):
    """Draw divided circle icons below x-axis ticks to visualize class counts.

    Each circle is split into N equal pie slices (N = number of classes),
    with alternating dark/light fills. For large N (1000), slices are so
    thin the circle appears nearly solid.
    """
    fig.canvas.draw()  # finalize positions before reading coordinates

    class_counts = COARSE_CFGS + [1000]
    x_data_positions = COARSE_CFGS + [BREAK_1K_POS]

    dark_color = '#5a7d9a'    # muted steel blue
    light_color = '#e8edf2'   # very light blue-gray
    border_color = '#9a9a9a'

    circle_side = 0.048   # figure fraction per circle (diameter)
    gap = 0.036           # gap below axes y0

    for ax in bottom_axes:
        ax_pos = ax.get_position()

        for x_data, n_classes in zip(x_data_positions, class_counts):
            # Convert data x-position to figure x-coordinate
            display_pt = ax.transData.transform([x_data, 0])
            fig_pt = fig.transFigure.inverted().transform(display_pt)
            cx = fig_pt[0]

            left = cx - circle_side / 2
            bottom = ax_pos.y0 - gap - circle_side

            if bottom < 0.01 or left < 0:
                continue

            inset = fig.add_axes([left, bottom, circle_side, circle_side],
                                 aspect='equal')
            inset.set_xlim(-1.12, 1.12)
            inset.set_ylim(-1.12, 1.12)
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_visible(False)
            inset.set_facecolor('none')

            # Draw pie wedges
            angle_per_slice = 360.0 / n_classes
            # For small N, add subtle edge lines between slices
            slice_edge = border_color if n_classes <= 16 else 'none'
            slice_lw = 0.3 if n_classes <= 16 else 0
            for i in range(n_classes):
                theta1 = 90 + i * angle_per_slice  # start from top
                theta2 = 90 + (i + 1) * angle_per_slice
                color = dark_color if i % 2 == 0 else light_color
                wedge = Wedge(center=(0, 0), r=1.0,
                              theta1=theta1, theta2=theta2,
                              facecolor=color, edgecolor=slice_edge,
                              linewidth=slice_lw)
                inset.add_patch(wedge)

            # Circle border
            border = Circle((0, 0), 1.0, fill=False,
                            edgecolor=border_color, linewidth=0.7)
            inset.add_patch(border)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    setup_style()
    plt.rcParams.update({
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    fig = plt.figure(figsize=(11.7, 9.0))

    # 3 rows (TVSD, separator, NSD) x 2 regions
    outer = gridspec.GridSpec(3, 2, figure=fig,
                              hspace=0.35, wspace=0.25,
                              height_ratios=[1, 0.005, 1],
                              left=0.09, right=0.96, top=0.92, bottom=0.19)

    axes = {}

    # ── Row 0: TVSD ──
    ax_v1 = fig.add_subplot(outer[0, 0])
    plot_raw_coarseness(ax_v1, "tvsd", "V1",
                        show_ylabel=True, show_xlabel=False)
    axes[(0, 0)] = ax_v1

    ax_it = fig.add_subplot(outer[0, 1])
    plot_raw_coarseness(ax_it, "tvsd", "IT",
                        show_ylabel=False, show_xlabel=False)
    axes[(0, 1)] = ax_it

    # ── Row 2: NSD ──
    ax_early = fig.add_subplot(outer[2, 0])
    plot_raw_coarseness(ax_early, "nsd", "early visual stream",
                        show_ylabel=True, show_xlabel=True)
    axes[(2, 0)] = ax_early

    ax_ventral = fig.add_subplot(outer[2, 1])
    plot_raw_coarseness(ax_ventral, "nsd", "ventral visual stream",
                        show_ylabel=False, show_xlabel=True)
    axes[(2, 1)] = ax_ventral

    # ── Region titles ──
    for ax_key, title in [
        ((0, 0), "V1 (Early)"),
        ((0, 1), "IT (Late)"),
        ((2, 0), "Early Visual Stream"),
        ((2, 1), "Ventral Visual Stream"),
    ]:
        axes[ax_key].set_title(title, fontsize=10, fontweight="bold",
                                pad=8, color="#1a1a1a")

    # ── Row labels ──
    tvsd_mid_y = (axes[(0, 0)].get_position().y0 + axes[(0, 0)].get_position().y1) / 2
    nsd_mid_y = (axes[(2, 0)].get_position().y0 + axes[(2, 0)].get_position().y1) / 2
    fig.text(0.02, tvsd_mid_y, "TVSD  (Macaque)", fontsize=10.5, fontweight="bold",
             ha="center", va="center", color="#333333", rotation=90)
    fig.text(0.02, nsd_mid_y, "NSD  (Human)", fontsize=10.5, fontweight="bold",
             ha="center", va="center", color="#333333", rotation=90)

    # ── Panel labels (A-D) ──
    label_order = [(0, 0), (0, 1), (2, 0), (2, 1)]
    for i, key in enumerate(label_order):
        label = chr(ord("A") + i)
        ax = axes[key]
        ax.text(-0.12, 1.14, label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="left",
                family="sans-serif")

    # ── Remove per-axes legends ──
    for key, ax in axes.items():
        legend = ax.get_legend()
        if legend:
            legend.remove()

    # ── Horizontal separator ──
    tvsd_bottom = axes[(0, 0)].get_position().y0
    nsd_top = axes[(2, 0)].get_position().y1
    mid_y = (tvsd_bottom + nsd_top) / 2
    line_x0 = axes[(0, 0)].get_position().x0 - 0.01
    line_x1 = axes[(0, 1)].get_position().x1 + 0.01
    fig.add_artist(plt.Line2D([line_x0, line_x1], [mid_y, mid_y],
                               transform=fig.transFigure, color="#cccccc",
                               linewidth=0.7, linestyle="-", clip_on=False))

    # ── Remove "Classes" xlabel from bottom row — circles replace it ──
    axes[(2, 0)].set_xlabel("")
    axes[(2, 1)].set_xlabel("")

    # ── Class count divided circles (below bottom row) ──
    draw_class_count_circles(fig, [axes[(2, 0)], axes[(2, 1)]])

    # ── Per-axis "Number of training classes" labels ──
    for ax in [axes[(2, 0)], axes[(2, 1)]]:
        ax_pos = ax.get_position()
        ax_cx = (ax_pos.x0 + ax_pos.x1) / 2
        label_y = ax_pos.y0 - 0.038 - 0.042 - 0.008
        fig.text(ax_cx, max(label_y, 0.045),
                 "Number of training classes", fontsize=8, ha="center",
                 va="top", color="#555555")

    # ── Shared legend ──
    fig_left = axes[(0, 0)].get_position().x0
    fig_right = axes[(0, 1)].get_position().x1
    fig_center = (fig_left + fig_right) / 2

    handles = []
    for arch_key, _, display in ARCHITECTURES:
        style = ARCH_STYLE[arch_key]
        h = Line2D([], [], marker=style["marker"], color="none",
                   markerfacecolor=style["color"],
                   markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                   markersize=MARKER_SIZE, label=display)
        handles.append(h)
    handles.append(Line2D([], [], marker="D", color="none",
                          markerfacecolor=BASELINE_1K_COLOR,
                          markeredgecolor=EDGE_COLOR,
                          markeredgewidth=EDGE_WIDTH,
                          markersize=MARKER_SIZE, label="1K (ImageNet)"))
    handles.append(Line2D([], [], **UNTRAINED_LINE_STYLE, label="Untrained"))

    fig.legend(handles=handles, loc="center", fontsize=8, frameon=False,
               handletextpad=0.3, columnspacing=0.8, ncol=5,
               borderpad=0.3, handlelength=1.5,
               bbox_to_anchor=(fig_center, 0.02))

    # ── Save ──
    out = f"{OUTPUT_DIR}/figure3_variant_circles.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
