"""Figure 3: Combined TVSD + NSD Neural Alignment.

2 rows × 3 columns:
  Column 0: dataset schematics (placeholders)
  Columns 1-2: data panels
  Row 0 (TVSD):  Schematic | V1 coarseness | IT coarseness
  Row 1 (NSD):   Schematic | Early coarseness | Ventral coarseness

Coarseness panels show raw Spearman ρ values.
AlexNet/CLIP as blue shades, Pixels as brown, 1K as warm amber.
Per-layer profiles and ViT moved to supplementary.

Usage:
    python manuscript/figures/fig3/figure3.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns

sys.path.insert(0, "plotters")
from plotter_utils import get_condition_summary, query_best_scores

sys.path.insert(0, "manuscript/figures")
from fig_utils import (
    COARSE_CFGS, BREAK_1K_POS,
    UNTRAINED_LINE_STYLE, MARKER_SIZE, EDGE_COLOR, EDGE_WIDTH,
    setup_style, compute_jitter,
    format_coarseness_axes, draw_schematic_placeholder,
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
    """Plot coarseness panel with raw Spearman ρ values."""

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


# ── Class count dot grids ──────────────────────────────────────────────

def _ordered_grid_positions(n_classes, margin=0.22):
    """Generate centered, evenly-spaced dot positions for N classes.

    Uses clean rectangular layouts (powers of 2 map to natural grids).
    A generous inner margin ensures clear white space between dots and
    the box border so dots don't crowd the edges.
    Returns (N, 2) array of (x, y) positions in [0, 1]² space.
    """
    # Hand-picked layouts: (rows, cols) for each coarse granularity
    layouts = {2: (1, 2), 4: (2, 2), 8: (2, 4), 16: (4, 4),
               32: (4, 8), 64: (8, 8)}
    rows, cols = layouts.get(n_classes, (int(np.ceil(np.sqrt(n_classes))),
                                         int(np.ceil(np.sqrt(n_classes)))))
    x_pos = np.linspace(margin, 1 - margin, cols) if cols > 1 else [0.5]
    y_pos = np.linspace(margin, 1 - margin, rows) if rows > 1 else [0.5]
    xx, yy = np.meshgrid(x_pos, y_pos)
    return np.column_stack([xx.ravel(), yy.ravel()])[:n_classes]


def draw_class_count_grids(fig, bottom_axes, grid_dim=32):
    """Draw dot density grids below x-axis ticks to visualize class counts.

    Low counts (2–64) use ordered grid layouts so individual dots are clearly
    countable and evenly spaced. N=1000 uses random fill on a 32×32 grid,
    creating a solid packed field that contrasts dramatically with the sparse
    low-count boxes.
    """
    fig.canvas.draw()  # finalize positions before reading coordinates

    class_counts = COARSE_CFGS + [1000]
    x_data_positions = COARSE_CFGS + [BREAK_1K_POS]

    fill_color = '#8fa8be'   # muted slate blue — understated, not competing
    bg_color = '#fafafa'     # near-white background
    border_color = '#b5b5b5'

    grid_side = 0.038  # figure fraction per grid square (scaled for wider figure)
    gap = 0.038        # gap below axes y0 (tight — "Classes" label removed)

    # Dot sizes scale with class count: prominent circles for sparse grids
    # (easy to count), tiny dots for N=1000 (overlap into solid fill).
    dot_sizes = {2: 16, 4: 12, 8: 9, 16: 7, 32: 5.5, 64: 4.5, 1000: 4.0}

    # Pre-generate dot positions for each class count
    # Low counts: ordered grid.  N=1000: random fill on dense grid.
    dot_positions = {}
    for n_classes in class_counts:
        if n_classes <= 64:
            dot_positions[n_classes] = _ordered_grid_positions(n_classes)
        else:
            inner_margin = 0.12
            pos = np.linspace(inner_margin, 1 - inner_margin, grid_dim)
            xx, yy = np.meshgrid(pos, pos)
            all_pos = np.column_stack([xx.ravel(), yy.ravel()])
            rng = np.random.RandomState(42 + n_classes)
            n_fill = min(n_classes, len(all_pos))
            idx = rng.choice(len(all_pos), n_fill, replace=False)
            dot_positions[n_classes] = all_pos[idx]

    for ax in bottom_axes:
        ax_pos = ax.get_position()

        for x_data, n_classes in zip(x_data_positions, class_counts):
            # Convert data x-position to figure x-coordinate
            display_pt = ax.transData.transform([x_data, 0])
            fig_pt = fig.transFigure.inverted().transform(display_pt)
            cx = fig_pt[0]

            left = cx - grid_side / 2
            bottom = ax_pos.y0 - gap - grid_side

            if bottom < 0.01 or left < 0:
                continue

            inset = fig.add_axes([left, bottom, grid_side, grid_side])
            inset.set_facecolor(bg_color)

            # Draw dots (size scales with class count)
            pts = dot_positions[n_classes]
            inset.scatter(pts[:, 0], pts[:, 1],
                          s=dot_sizes[n_classes], c=fill_color,
                          edgecolors='none', linewidths=0, zorder=2)

            inset.set_xlim(0, 1)
            inset.set_ylim(0, 1)
            inset.set_xticks([])
            inset.set_yticks([])
            if n_classes == 1000:
                # Solid fill, no border — reads as a completely packed block
                inset.set_facecolor(fill_color)
                for spine in inset.spines.values():
                    spine.set_visible(False)
            else:
                for spine in inset.spines.values():
                    spine.set_linewidth(0.5)
                    spine.set_color(border_color)


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

    fig = plt.figure(figsize=(14.5, 9.0))

    # 3 rows (TVSD, separator, NSD) × 3 columns (schematic, region1, region2)
    outer = gridspec.GridSpec(3, 3, figure=fig,
                              hspace=0.35, wspace=0.28,
                              height_ratios=[1, 0.005, 1],
                              width_ratios=[0.8, 1, 1],
                              left=0.07, right=0.96, top=0.92, bottom=0.19)

    axes = {}

    # ── Row 0: TVSD ──
    ax_tvsd_schem = fig.add_subplot(outer[0, 0])
    draw_schematic_placeholder(ax_tvsd_schem,
                               "TVSD schematic\n(Macaque electrophysiology,\nV1 / V4 / IT)")
    axes[(0, 0)] = ax_tvsd_schem

    ax_v1 = fig.add_subplot(outer[0, 1])
    plot_raw_coarseness(ax_v1, "tvsd", "V1",
                        show_ylabel=True, show_xlabel=False)
    axes[(0, 1)] = ax_v1

    ax_it = fig.add_subplot(outer[0, 2])
    plot_raw_coarseness(ax_it, "tvsd", "IT",
                        show_ylabel=False, show_xlabel=False)
    axes[(0, 2)] = ax_it

    # ── Row 2: NSD ──
    ax_nsd_schem = fig.add_subplot(outer[2, 0])
    draw_schematic_placeholder(ax_nsd_schem,
                               "NSD schematic\n(Human fMRI,\n8 subjects, early/ventral\nvisual stream)")
    axes[(2, 0)] = ax_nsd_schem

    ax_early = fig.add_subplot(outer[2, 1])
    plot_raw_coarseness(ax_early, "nsd", "early visual stream",
                        show_ylabel=True, show_xlabel=True)
    axes[(2, 1)] = ax_early

    ax_ventral = fig.add_subplot(outer[2, 2])
    plot_raw_coarseness(ax_ventral, "nsd", "ventral visual stream",
                        show_ylabel=False, show_xlabel=True)
    axes[(2, 2)] = ax_ventral

    # ── Region titles ──
    for ax_key, title in [
        ((0, 1), "V1 (Early)"),
        ((0, 2), "IT (Late)"),
        ((2, 1), "Early Visual Stream"),
        ((2, 2), "Ventral Visual Stream"),
    ]:
        axes[ax_key].set_title(title, fontsize=10, fontweight="bold",
                                pad=8, color="#1a1a1a")

    # ── Row labels ──
    tvsd_mid_y = (axes[(0, 0)].get_position().y0 + axes[(0, 0)].get_position().y1) / 2
    nsd_mid_y = (axes[(2, 0)].get_position().y0 + axes[(2, 0)].get_position().y1) / 2
    fig.text(0.015, tvsd_mid_y, "TVSD  (Macaque)", fontsize=10.5, fontweight="bold",
             ha="center", va="center", color="#333333", rotation=90)
    fig.text(0.015, nsd_mid_y, "NSD  (Human)", fontsize=10.5, fontweight="bold",
             ha="center", va="center", color="#333333", rotation=90)

    # ── Panel labels (A–F) ──
    label_order = [(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2)]
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
    line_x1 = axes[(0, 2)].get_position().x1 + 0.01
    fig.add_artist(plt.Line2D([line_x0, line_x1], [mid_y, mid_y],
                               transform=fig.transFigure, color="#cccccc",
                               linewidth=0.7, linestyle="-", clip_on=False))

    # ── Remove "Classes" xlabel from bottom row — dot grids replace it ──
    axes[(2, 1)].set_xlabel("")
    axes[(2, 2)].set_xlabel("")

    # ── Class count dot grids (below bottom data panels in columns 1-2) ──
    draw_class_count_grids(fig, [axes[(2, 1)], axes[(2, 2)]])

    # ── Per-axis "Number of training classes" labels ──
    for ax in [axes[(2, 1)], axes[(2, 2)]]:
        ax_pos = ax.get_position()
        ax_cx = (ax_pos.x0 + ax_pos.x1) / 2
        label_y = ax_pos.y0 - 0.038 - 0.038 - 0.008
        fig.text(ax_cx, max(label_y, 0.045),
                 "Number of training classes", fontsize=8, ha="center",
                 va="top", color="#555555")

    # ── Shared legend (centered across data panels in columns 1-2) ──
    fig_left = axes[(0, 1)].get_position().x0
    fig_right = axes[(0, 2)].get_position().x1
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
    out = f"{OUTPUT_DIR}/figure3.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
