"""THINGS coarseness bar plot for manuscript (one architecture per figure).

Single panel bar plot: Untrained | coarse (2-64) | break | 1000.
Uses the same shared drawing logic as plotters/things/plot_coarseness.py.

Usage:
    python manuscript/figures/plot_coarseness_things.py --pca_labels alexnet
    python manuscript/figures/plot_coarseness_things.py --pca_labels pixels
"""

import sys
import argparse

sys.path.insert(0, "plotters")
from plot_helpers import plot_coarseness_bars, PCA_MODELS

OUTPUT_DIR = "manuscript/figures"

# Extended choices including pixels
_CHOICES = list(PCA_MODELS) + (["pixels"] if "pixels" not in PCA_MODELS else [])


def main():
    # Extend PCA_MODELS with pixels at runtime (not at import time)
    if "pixels" not in PCA_MODELS:
        PCA_MODELS["pixels"] = "Pixels"

    p = argparse.ArgumentParser()
    p.add_argument("--pca_labels", default="alexnet", choices=_CHOICES)
    args = p.parse_args()

    dcfg = {
        "neural_dataset": "things-behavior",
        "regions": ["N/A"],
        "region_labels": {"N/A": "THINGS Behavior"},
        "has_subjects": False,
        "analysis": "rsa",
        "compare_method": "spearman",
        "output_suffix": "",
    }

    plot_coarseness_bars(dcfg, args.pca_labels, OUTPUT_DIR, dataset_label="THINGS")


if __name__ == "__main__":
    main()
