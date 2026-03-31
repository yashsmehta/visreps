"""Figure 3: Neural alignment across species — TVSD + NSD.

2 rows x 4 columns + schematic row on top:
  Columns grouped by dataset: TVSD (macaque) | NSD (human)
  Within each pair: bits-to-match | raw Spearman rho
  Rows: early visual cortex (top) | higher visual cortex (bottom)

  Schematics with example stimuli + species icons span each dataset pair.
  Brain region insets (nilearn for human, SVG for macaque) on bits panels.

Panel modules:
  - panel_bits.py: Minimum bits of supervision to match 1000-way
  - panel_raw.py: Raw Spearman rho scatter (all architectures)
  - schematic_utils.py: Dataset schematics + brain insets
  - shared.py: Style constants, data fetching, axis formatting

Usage:
    python manuscript/figures/fig3/figure3.py
"""

import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

sys.path.insert(0, "manuscript/figures")
from fig_utils import MARKER_SIZE, EDGE_COLOR, EDGE_WIDTH, setup_style

sys.path.insert(0, "manuscript/figures/fig3")
from shared import ARCHITECTURES, ARCH_STYLE
from panel_raw import plot_raw
from panel_bits import plot_fcm
from schematic_utils import draw_tvsd_schematic, draw_nsd_schematic, add_brain_inset

OUTPUT_DIR = "manuscript/figures/fig3"


def main():
    setup_style()
    plt.rcParams.update({
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    })

    fig = plt.figure(figsize=(14.5, 9.0))

    # Outer grid: 3 rows (schematics + 2 data rows) x 2 cols (TVSD | NSD)
    outer = gridspec.GridSpec(3, 2, figure=fig,
                              height_ratios=[0.40, 1, 1],
                              width_ratios=[1, 1],
                              hspace=0.38, wspace=0.18,
                              left=0.06, right=0.97, top=0.89, bottom=0.08)

    # Inner grids: bits panels at 50% width of raw panels
    inner_tvsd = [gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[row, 0],
                  wspace=0.35, width_ratios=[0.25, 0.75]) for row in (1, 2)]
    inner_nsd = [gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[row, 1],
                 wspace=0.35, width_ratios=[0.25, 0.75]) for row in (1, 2)]

    # ── Schematics (row 0) ──
    ax_tvsd_schem = fig.add_subplot(outer[0, 0])
    draw_tvsd_schematic(ax_tvsd_schem)

    ax_nsd_schem = fig.add_subplot(outer[0, 1])
    draw_nsd_schematic(ax_nsd_schem)

    # ── Data panels (rows 1–2) ──
    # Row 1 = early visual cortex, row 2 = higher visual cortex
    # Columns: 0=bits, 1=raw (TVSD); 2=bits, 3=raw (NSD)
    panel_specs = [
        # (row, col, dataset, region, plot_fn, ylabel, xlabel, inner_grid, inner_col, extra_kw)
        (1, 0, "tvsd", "V1",                    plot_fcm,  True,  False, inner_tvsd[0], 0, {}),
        (1, 1, "tvsd", "V1",                    plot_raw,  False, False, inner_tvsd[0], 1, {}),
        (1, 2, "nsd",  "early visual stream",   plot_fcm,  False, False, inner_nsd[0],  0, {}),
        (1, 3, "nsd",  "early visual stream",   plot_raw,  False, False, inner_nsd[0],  1, {}),
        (2, 0, "tvsd", "IT",                    plot_fcm,  True,  True,  inner_tvsd[1], 0, {}),
        (2, 1, "tvsd", "IT",                    plot_raw,  False, True,  inner_tvsd[1], 1, {"show_untrained_label": True}),
        (2, 2, "nsd",  "ventral visual stream", plot_fcm,  False, True,  inner_nsd[1],  0, {}),
        (2, 3, "nsd",  "ventral visual stream", plot_raw,  False, True,  inner_nsd[1],  1, {"show_untrained_label": True}),
    ]

    axes = {}
    for row, col, ds, region, fn, ylabel, xlabel, inner, icol, extra_kw in panel_specs:
        ax = fig.add_subplot(inner[0, icol])
        fn(ax, ds, region, show_ylabel=ylabel, show_xlabel=xlabel, **extra_kw)
        axes[(row, col)] = ax

    # ── Column-pair headers (dataset name + stimulus type subtitle) ──
    for cols, schem_ax, title, subtitle in [
        ((0, 1), ax_tvsd_schem, "TVSD", "Object images"),
        ((2, 3), ax_nsd_schem,  "NSD",  "Natural scenes"),
    ]:
        left = axes[(1, cols[0])].get_position().x0
        right = axes[(1, cols[1])].get_position().x1
        x_center = (left + right) / 2
        y_top = schem_ax.get_position().y1
        fig.text(x_center, y_top + 0.035, title,
                 fontsize=14, fontweight="bold",
                 color="#1a1a1a", ha="center", va="bottom")
        fig.text(x_center, y_top + 0.015, subtitle,
                 fontsize=10, color="#777777", fontstyle="italic",
                 ha="center", va="bottom")

    # ── Row labels ──
    for row, label in [(1, "Early Visual\nCortex"), (2, "Higher Visual\nCortex")]:
        pos = axes[(row, 0)].get_position()
        fig.text(0.012, (pos.y0 + pos.y1) / 2, label,
                 fontsize=10, fontweight="bold", color="#444444",
                 ha="center", va="center", rotation=90, linespacing=1.3)

    # ── Sub-column labels ──
    sub_labels = {
        (1, 0): "Coarse feedback\ncompression",  (1, 1): "V1",
        (1, 2): "Coarse feedback\ncompression",  (1, 3): "Early visual stream",
        (2, 1): "IT",
        (2, 3): "Ventral visual stream",
    }
    for key, label in sub_labels.items():
        pos = axes[key].get_position()
        fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.018, label,
                 fontsize=9, color="#666666", ha="center", va="bottom")

    # ── Panel labels (a–h) ──
    for i, key in enumerate([(1, 0), (1, 1), (1, 2), (1, 3),
                             (2, 0), (2, 1), (2, 2), (2, 3)]):
        pos = axes[key].get_position()
        fig.text(pos.x0 - 0.015, pos.y1 + 0.028, chr(ord("a") + i),
                 fontsize=13, fontweight="bold", va="bottom", ha="left")

    # ── Vertical separator between TVSD and NSD ──
    tvsd_right = axes[(1, 1)].get_position().x1
    nsd_left = axes[(1, 2)].get_position().x0
    sep_x = (tvsd_right + nsd_left) / 2
    bottom_y = axes[(2, 0)].get_position().y0 - 0.02
    top_y = ax_tvsd_schem.get_position().y1 + 0.01
    fig.add_artist(plt.Line2D(
        [sep_x, sep_x], [bottom_y, top_y],
        transform=fig.transFigure, color="#dddddd",
        linewidth=0.8, zorder=0))

    # ── Legend ──
    handles = [Line2D([], [], marker=ARCH_STYLE[k]["marker"], color="none",
                      markerfacecolor=ARCH_STYLE[k]["color"],
                      markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                      markersize=6, label=d)
               for k, _, d in ARCHITECTURES]
    axes[(1, 1)].legend(handles=handles, fontsize=7.5, frameon=True,
                        fancybox=False, framealpha=0.92, edgecolor="#dddddd",
                        borderpad=0.5, handletextpad=0.4, labelspacing=0.3,
                        title="Coarse label source", title_fontsize=7.5,
                        loc="lower right")

    # ── Save ──
    out = f"{OUTPUT_DIR}/figure3.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
