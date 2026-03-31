"""Figure 2: Categorical Nature of Representations.

Learned representation mosaic — 1000-way vs 4-way CNN FC1 PCA projections
with image thumbnails at per-class centroid positions.

Usage (from project root):
    python manuscript/figures/fig2/figure2.py
"""

import sys

sys.path.insert(0, ".")

from manuscript.figures.fig2.plot_representations import plot_representations


def main():
    plot_representations()


if __name__ == "__main__":
    main()
