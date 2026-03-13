"""Figure 1 — Method Overview (placeholder schematics).

Generates a three-panel schematic layout:
  A: Coarse-graining procedure
  B: DNN training + RSA comparison
  C: Evaluation domains
"""

import sys
sys.path.insert(0, ".")

import matplotlib.pyplot as plt
from manuscript.figures.fig_utils import setup_style, draw_schematic_placeholder

OUTPUT = "manuscript/figures/fig1/figure1.png"


def main():
    setup_style()

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.06, wspace=0.15)

    # Panel A
    draw_schematic_placeholder(
        axes[0],
        "Coarse-graining procedure\n"
        "(ImageNet \u2192 PCA \u2192 median splits \u2192 2\u201364 classes)",
    )

    # Panel B
    draw_schematic_placeholder(
        axes[1],
        "DNN training + RSA comparison\n"
        "(Same arch/data/protocol, varying class count;\n"
        "model RDMs vs neural/behavioral RDMs)",
    )

    # Panel C
    draw_schematic_placeholder(
        axes[2],
        "Evaluation domains\n"
        "(TVSD: macaque electrophysiology,\n"
        "NSD: human fMRI,\n"
        "THINGS: behavioral similarity)",
    )

    # Panel labels
    for ax, label in zip(axes, ["A", "B", "C"]):
        ax.text(
            -0.02, 1.08, label,
            transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", ha="left",
        )

    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
