"""
Reconstruction Analysis Plotter
================================
Plots RSA score vs. pca_k for the 1000-way model with reconstruct_from_pcs=True,
overlaid with baselines: 1000-way (all dims), best coarse model, untrained.

Produces one figure per neural dataset (NSD 1x2, TVSD 1x3, THINGS 1x1).
"""

import sqlite3

import pandas as pd

from experiments.reconstruction_analysis.plot_utils import (
    DB_PATH, COMPARE_METHOD, DATASET_LAYOUTS,
    get_bootstrap_ci, _region_filter, plot_figure,
)


def query_coarse_baseline(neural_dataset, region=None):
    """Best coarse-grained model's grand-mean score across all label sources."""
    rfrag, rparams = _region_filter(region)
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT run_id, checkpoint_dir, cfg_id, seed, subject_idx, layer, score "
        "FROM results WHERE reconstruct_from_pcs = 0 AND pca_labels = 1"
        " AND cfg_id IN (2, 4, 8, 16, 32, 64) AND analysis = 'rsa'"
        f" AND compare_method = 'spearman' AND neural_dataset = ?{rfrag}",
        conn, params=[neural_dataset] + rparams,
    )
    conn.close()

    if df.empty:
        return float("nan"), float("nan"), float("nan")

    # Best layer per (checkpoint_dir, cfg_id, seed, subject_idx)
    best = df.loc[df.groupby(["checkpoint_dir", "cfg_id", "seed", "subject_idx"])["score"].idxmax()]
    # Find best (checkpoint_dir, cfg_id) combo by grand mean
    grand = best.groupby(["checkpoint_dir", "cfg_id"])["score"].mean()
    best_combo = grand.idxmax()
    best_runs = best[(best["checkpoint_dir"] == best_combo[0]) & (best["cfg_id"] == best_combo[1])]

    _, ci_low, ci_high = get_bootstrap_ci(best_runs["run_id"].tolist())
    return best_runs["score"].mean(), ci_low, ci_high


if __name__ == "__main__":
    for ds in DATASET_LAYOUTS:
        plot_figure(ds, query_coarse_baseline, "Best coarse model", "reconstruction")
