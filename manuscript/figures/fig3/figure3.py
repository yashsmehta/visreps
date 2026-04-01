"""Figure 3: Neural alignment across species — TVSD + NSD.

2 rows (TVSD | NSD) x 3 cols (schematic | early cortex | higher cortex).
Each data cell: horizontal lollipop strip (min classes to match 1K) above
               raw Spearman rho scatter with broken x-axis.

Usage:
    python manuscript/figures/fig3/figure3.py
"""

import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

sys.path.insert(0, "manuscript/figures")
from fig_utils import EDGE_COLOR, EDGE_WIDTH, setup_style

sys.path.insert(0, "manuscript/figures/fig3")
from shared import ARCHITECTURES, ARCH_STYLE
from panel_raw import plot_raw
from panel_bits import plot_lollipop
from schematic_utils import draw_tvsd_schematic, draw_nsd_schematic

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

    fig = plt.figure(figsize=(14, 8.5))

    # 2 rows (TVSD | NSD) x 3 cols (schematic | early | higher), equal widths
    outer = gridspec.GridSpec(2, 3, figure=fig,
                              height_ratios=[1, 1],
                              width_ratios=[1, 1, 1],
                              hspace=0.25, wspace=0.20,
                              left=0.06, right=0.97, top=0.88, bottom=0.08)

    # ── Schematics (col 0, horizontal layout) ──
    ax_tvsd_schem = fig.add_subplot(outer[0, 0])
    draw_tvsd_schematic(ax_tvsd_schem)

    ax_nsd_schem = fig.add_subplot(outer[1, 0])
    draw_nsd_schematic(ax_nsd_schem)

    # ── Data panels (cols 1–2): each cell = lollipop + scatter ──
    panel_defs = [
        # (row, col, dataset, region, show_ylabel, show_xlabel)
        (0, 1, "tvsd", "V1",                    True,  False),
        (0, 2, "tvsd", "IT",                    False, False),
        (1, 1, "nsd",  "early visual stream",   True,  True),
        (1, 2, "nsd",  "ventral visual stream", False, True),
    ]

    axes_scatter = {}
    axes_lollipop = {}

    for orow, ocol, ds, region, ylabel, xlabel in panel_defs:
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[orow, ocol],
            height_ratios=[0.14, 0.86], hspace=0.10)

        ax_raw = fig.add_subplot(inner[1, 0])
        show_untrained = (orow == 1)  # bottom row shows untrained label
        plot_raw(ax_raw, ds, region,
                 show_ylabel=ylabel, show_xlabel=xlabel,
                 show_untrained_label=show_untrained)
        axes_scatter[(orow, ocol)] = ax_raw

        ax_lol = fig.add_subplot(inner[0, 0], sharex=ax_raw)
        plot_lollipop(ax_lol, ds, region, show_ylabel=True)
        axes_lollipop[(orow, ocol)] = ax_lol

    # ── Force-align lollipop plot areas to scatter plot areas ──
    for _ in range(2):
        fig.canvas.draw()
        for key in axes_lollipop:
            scat_pos = axes_scatter[key].get_position()
            lol_pos = axes_lollipop[key].get_position()
            axes_lollipop[key].set_position(
                [scat_pos.x0, lol_pos.y0, scat_pos.width, lol_pos.height])

    # ── Row headers (dataset name + stimulus type, above schematics) ──
    for schem_ax, title, subtitle in [
        (ax_tvsd_schem, "TVSD", "Object images"),
        (ax_nsd_schem,  "NSD",  "Natural scenes"),
    ]:
        pos = schem_ax.get_position()
        x_center = (pos.x0 + pos.x1) / 2
        fig.text(x_center, pos.y1 + 0.030, title,
                 fontsize=13, fontweight="bold",
                 color="#1a1a1a", ha="center", va="bottom")
        fig.text(x_center, pos.y1 + 0.010, subtitle,
                 fontsize=9, color="#777777", fontstyle="italic",
                 ha="center", va="bottom")

    # ── Column headers (cortical level, above top-row data panels) ──
    for col, label in [(1, "Early Visual Cortex"), (2, "Higher Visual Cortex")]:
        pos = axes_lollipop[(0, col)].get_position()
        x_center = (pos.x0 + pos.x1) / 2
        fig.text(x_center, pos.y1 + 0.045, label,
                 fontsize=11, fontweight="bold", color="#333333",
                 ha="center", va="bottom")

    # ── Region sub-labels (above lollipops) ──
    region_labels = {
        (0, 1): "V1",       (0, 2): "IT",
        (1, 1): "Early visual stream", (1, 2): "Ventral visual stream",
    }
    for key, label in region_labels.items():
        pos = axes_lollipop[key].get_position()
        fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.012, label,
                 fontsize=9, color="#666666", ha="center", va="bottom")

    # ── Panel labels (a–f) ──
    # Schematics (a, d) use schematic axes; data panels (b, c, e, f) use lollipop axes
    schem_panels = [(0, ax_tvsd_schem, "a"), (1, ax_nsd_schem, "d")]
    for _, schem_ax, label in schem_panels:
        pos = schem_ax.get_position()
        fig.text(pos.x0 - 0.010, pos.y1 + 0.012, label,
                 fontsize=13, fontweight="bold", va="bottom", ha="left")

    data_labels = [((0, 1), "b"), ((0, 2), "c"), ((1, 1), "e"), ((1, 2), "f")]
    for key, label in data_labels:
        pos = axes_lollipop[key].get_position()
        fig.text(pos.x0 - 0.018, pos.y1 + 0.022, label,
                 fontsize=13, fontweight="bold", va="bottom", ha="left")

    # ── Legend ──
    handles = [Line2D([], [], marker=ARCH_STYLE[k]["marker"], color="none",
                      markerfacecolor=ARCH_STYLE[k]["color"],
                      markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                      markersize=6, label=d)
               for k, _, d in ARCHITECTURES]
    axes_scatter[(0, 1)].legend(handles=handles, fontsize=7.5, frameon=True,
                        fancybox=False, framealpha=0.92, edgecolor="#dddddd",
                        borderpad=0.5, handletextpad=0.4, labelspacing=0.3,
                        title="Coarse label source", title_fontsize=7.5,
                        loc="right", bbox_to_anchor=(1.0, 0.35))

    # ── Save ──
    out = f"{OUTPUT_DIR}/figure3.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
