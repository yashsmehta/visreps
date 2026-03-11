"""
Reconstruction Analysis Plotter
================================
Plots RSA score vs. pca_k for both the 1000-way and coarse models, showing how
brain alignment grows as more PCs are included in the reconstruction.

Each panel uses the region-specific coarse model (e.g., CLIP-PCA for ventral
visual stream, AlexNet-PCA for early visual stream). Untrained baseline shown
as a dotted horizontal line.

Produces one figure per neural dataset (NSD 1x2, TVSD 1x3, THINGS 1x1).
"""

from experiments.reconstruction_analysis.plot_utils import (
    DATASET_LAYOUTS, plot_dual_figure,
)

# Per-region coarse model configs: region -> (cfg_id, checkpoint_dir)
# Must match the models used in run_reconstruction.py COARSE_CONFIG.
COARSE_CONFIG = {
    "nsd": {
        "early visual stream": (64, "/data/ymehta3/alexnet_pca"),
        "ventral visual stream": (16, "/data/ymehta3/clip_pca"),
    },
    "tvsd": {
        "V1": (64, "/data/ymehta3/alexnet_pca"),
        "V4": (64, "/data/ymehta3/alexnet_pca"),
        "IT": (64, "/data/ymehta3/alexnet_pca"),
    },
    "things-behavior": {
        "N/A": (64, "/data/ymehta3/vit_pca"),
    },
}


if __name__ == "__main__":
    for ds in DATASET_LAYOUTS:
        plot_dual_figure(
            ds, COARSE_CONFIG[ds], "reconstruction", coarse_label="Coarse model",
        )
