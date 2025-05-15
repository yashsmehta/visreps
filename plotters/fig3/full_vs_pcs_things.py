"""Comparison plots helper + CLI

Refactored for conciseness, simplicity, and separation of data prep from plotting.
"""

from __future__ import annotations

import os
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, FormatStrFormatter

import plotters.utils_plotter as plt_utils
import plotters.fig4.full_vs_pcs_nsd as fig4_plotter

def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    print(f"  • Loading: {os.path.basename(path)}")
    return pd.read_csv(path)

if __name__ == "__main__":
    cfg = {
        "metric": "Spearman",
        "layers": ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2"],
        "results_csv": "/home/ymehta3/research/VisionAI/visreps/logs/full-vs-pcs_things.csv",
        "pca_plot_n_classes": [2, 4, 8, 16, 32, 64], # PCA n_classes to plot
        "dataset": "things"
    }

    data_df = _load_csv(cfg["results_csv"])
    pca_df, full_df = plt_utils.split_and_select_df(data_df,
                                            dataset=cfg["dataset"],
                                            metric=cfg["metric"],
                                            epoch=20,
                                            layers=cfg["layers"])
    _, initial_df = plt_utils.split_and_select_df(data_df,
                                dataset=cfg["dataset"],
                                metric=cfg["metric"],
                                epoch=0,
                                layers=cfg["layers"])

    pca_df = plt_utils.avg_over_seed(pca_df)
    full_df = plt_utils.avg_over_seed(full_df)
    initial_df = plt_utils.avg_over_seed(initial_df)

    layer_order = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2"]

    # 1. epoch-0 scores (full model, no PCA labels)
    initial_scores = (
        initial_df.set_index("layer")["score"]
                .reindex(layer_order)
    )

    # 2. epoch-20 scores for the full model
    final_scores_full_model = (
        full_df.set_index("layer")["score"]
            .reindex(layer_order)
    )

    # 3. epoch-20 scores for each PCA size
    pca_scores_final_epoch = {
        n: (
            pca_df[pca_df["pca_n_classes"] == n]
                .set_index("layer")["score"]
                .reindex(layer_order)
        )
        for n in cfg["pca_plot_n_classes"]
    }

    # 4. y-axis limits (pad 5 %)
    all_scores = pd.concat(
        [initial_scores, final_scores_full_model, *pca_scores_final_epoch.values()]
    ).dropna()
    pad   = 0.05 * (all_scores.max() - all_scores.min() or 1)
    min_y = all_scores.min() - pad
    max_y = all_scores.max() + pad

    # 5. output directory for plots
    out_dir = "/home/ymehta3/research/VisionAI/visreps/plotters/fig3"
    os.makedirs(out_dir, exist_ok=True)

    # ----------------------------------------
    # call the plotter
    # ----------------------------------------
    fig4_plotter.create_comparison_plots(
        initial_scores=initial_scores,
        final_scores_full_model=final_scores_full_model,
        pca_scores_final_epoch=pca_scores_final_epoch,
        layer_order=layer_order,
        neural_dataset=cfg["dataset"],
        compare_rsm_correlation=cfg["metric"],
        min_y=min_y,
        max_y=max_y,
        pca_sizes_for_plot=cfg["pca_plot_n_classes"],
        out_dir=out_dir,
    )