"""
Reconstruction Case Study: 2-Class Model vs. PC Reconstruction
===============================================================
Overlays reconstruction curve with a 2-class model baseline to show
the model learns more than just the dominant PC.

Usage:
    python experiments/reconstruction_analysis/plot_case_study.py
    python experiments/reconstruction_analysis/plot_case_study.py --pca_labels clip
"""

import argparse
import sqlite3

import pandas as pd

from experiments.reconstruction_analysis.plot_utils import (
    DB_PATH, DATASET_LAYOUTS,
    get_bootstrap_ci, _region_filter, plot_figure,
)

PCA_LABELS_TO_CHECKPOINT = {
    "alexnet": "/data/ymehta3/alexnet_pca",
    "clip": "/data/ymehta3/clip_pca",
    "dino": "/data/ymehta3/dino_pca",
    "vit": "/data/ymehta3/vit_pca",
}


def query_2class_baseline(neural_dataset, region=None, checkpoint_dir=None):
    """2-class model scores (cfg_id=2, pca_labels=1, no reconstruction)."""
    rfrag, rparams = _region_filter(region)
    extra = ""
    extra_params = []
    if checkpoint_dir is not None:
        extra = " AND checkpoint_dir = ?"
        extra_params = [checkpoint_dir]

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT run_id, seed, subject_idx, layer, score FROM results"
        " WHERE cfg_id = 2 AND pca_labels = 1 AND reconstruct_from_pcs = 0"
        f" AND analysis = 'rsa' AND compare_method = 'spearman' AND neural_dataset = ?{rfrag}{extra}",
        conn, params=[neural_dataset] + rparams + extra_params,
    )
    conn.close()

    if df.empty:
        return float("nan"), float("nan"), float("nan")

    best = df.loc[df.groupby(["seed", "subject_idx"])["score"].idxmax()]
    _, ci_low, ci_high = get_bootstrap_ci(best["run_id"].tolist())
    return best["score"].mean(), ci_low, ci_high


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstruction case study: 2-class model vs. PC projection")
    parser.add_argument("--pca_labels", default="alexnet",
                        choices=list(PCA_LABELS_TO_CHECKPOINT),
                        help="Label source for the 2-class model (default: alexnet)")
    args = parser.parse_args()

    checkpoint_dir = PCA_LABELS_TO_CHECKPOINT[args.pca_labels]

    for ds in DATASET_LAYOUTS:
        plot_figure(ds, query_2class_baseline, "2-class model", "case_study",
                    checkpoint_dir=checkpoint_dir)
