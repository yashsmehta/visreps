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

# -----------------------------------------------------------------------------
# PLOTTER ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

def create_comparison_plots(
    initial_scores: pd.Series,      # epoch 0, full model, indexed by layer
    final_scores_full_model: pd.Series, # full model, indexed by layer
    pca_scores_final_epoch: Dict[int, pd.Series], # dict key: n_classes, val: series indexed by layer
    layer_order: List[str],
    neural_dataset: str,            # For filename
    compare_rsm_correlation: str, # For y-label and filename
    region_name: str,               # For filename
    min_y: float | None,
    max_y: float | None,
    pca_sizes_for_plot: List[int], # To iterate for PCA lines
    out_dir: str
):
    """Generate and save a comparison plot using pre-processed data."""

    # --- Config & Style (Style related parts only) ---------------------------
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Define a base linewidth; other lines will use this, 1000 classes will be scaled
    base_linewidth = 1.75 * 1.25 # Increased by 1.25x
    markersize = 6

    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 12, 'axes.titlesize': 11, 'axes.labelsize': 14, # General label sizes
        'xtick.labelsize': 12, 'ytick.labelsize': 15, # Tick label sizes - INCREASED
        'legend.fontsize': 12, # Legend font size
        'figure.titlesize': 13,
        'lines.linewidth': base_linewidth, # Default linewidth
        'lines.markersize': markersize,    # Default markersize
        'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.7 # Ensure grid is on and styled
    })

    colours = {
        "initial": "#7f8c8d", # Grey for Untrained
        "final": "#FFA500",   # Adjusted to DarkOrange for 1000 Classes
        "pca": dict(zip(pca_sizes_for_plot, sns.color_palette('Blues', n_colors=len(pca_sizes_for_plot)))),
    }

    # --- Plotting (Single Plot) ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 5.5))
    x_ticks_pos = range(len(layer_order))

    # Plot initial (full model, epoch 0) - uses pre-calculated initial_scores
    # Data is already reindexed in main, so direct plot if not empty/all NaN.
    if not initial_scores.empty and not initial_scores.isna().all():
        ax.plot(x_ticks_pos, initial_scores, color=colours["initial"], marker="x", linestyle="-", label="Untrained", zorder=3, linewidth=base_linewidth, markersize=markersize)
    
    # Plot final (full model, epoch 18) - uses pre-calculated final_scores_full_model
    if not final_scores_full_model.empty and not final_scores_full_model.isna().all():
        ax.plot(x_ticks_pos, final_scores_full_model, color=colours["final"], marker="o", linestyle="-", label="1000 Classes", zorder=3, linewidth=base_linewidth * 1.2, markersize=markersize)

    # Plot PCA variants (epoch 18) - uses pre-calculated pca_scores_final_epoch
    for n_classes in pca_sizes_for_plot:
        pca_series = pca_scores_final_epoch.get(n_classes)
        if pca_series is not None and not pca_series.empty and not pca_series.isna().all():
            ax.plot(x_ticks_pos, pca_series, color=colours["pca"][n_classes], marker="s", linestyle="-", label=f"{n_classes}", zorder=2, linewidth=base_linewidth, markersize=markersize)

    ax.set_xticks(x_ticks_pos)
    ax.set_xticklabels(layer_order, rotation=0, ha="center") # X-axis labels horizontal, fontsize controlled by rcParams
    for label in ax.get_xticklabels(): # Make x-axis tick labels bold
        label.set_fontweight('bold')
    ax.set_ylabel(f"RSA ({compare_rsm_correlation.capitalize()})", labelpad=10) # Y-label fontsize controlled by rcParams 'axes.labelsize'
    
    ax.tick_params(axis='x', direction='out', bottom=True, top=False, length=4, color='black', width=1.2) # Keep bottom ticks visible
    
    ax.tick_params(axis='y', which='major', direction='out', left=True, right=False, length=5, color='black', width=1.2) # Y-tick labelsize controlled by rcParams
    if max_y is not None and min_y is not None and (max_y - min_y) > 0.2:
         ax.yaxis.set_major_locator(MultipleLocator(0.1)) 
    elif max_y is not None and min_y is not None and (max_y - min_y) > 0.05:
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
    else:
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6, prune='both'))

    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='y', which='minor', direction='out', left=True, right=False, length=3, color='black', width=1.0,
                   labelsize=int(plt.rcParams['ytick.labelsize'] * 0.75), labelleft=True) # Label minor ticks at 60% size
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.2f")) # Format for minor y-axis ticks

    if min_y is not None and max_y is not None:
        ax.set_ylim(min_y, max_y)
    else:
        current_min_y, current_max_y = ax.get_ylim()
        padding = (current_max_y - current_min_y) * 0.05
        ax.set_ylim(current_min_y - padding, current_max_y + padding)

    # --- Legend & Layout ---
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique_labels_dict = {label: handle for handle, label in zip(handles, labels)} 
        fig.legend(unique_labels_dict.values(), unique_labels_dict.keys(), 
                   loc='center left', bbox_to_anchor=(1.005, 0.5), # Legend to the right, further reduced gap
                   ncol=1, frameon=False) # Legend fontsize controlled by rcParams
        plt.tight_layout(rect=[0.02, 0.02, 1.02, 0.95]) # Adjust rect to make space for legend on the right, further reduce gap
    else:
        plt.tight_layout(rect=[0.02, 0.02, 1.03, 0.95]) # Default tight layout if no legend

    ax.spines['bottom'].set_linewidth(1.2) # Keep prominent bottom spine
    ax.spines['left'].set_linewidth(1.2)   # Keep prominent left spine
    # For whitegrid, top and right spines are part of the grid box, usually thinner.
    # If you want them thicker like bottom/left, you'd add:
    # ax.spines['top'].set_linewidth(1.2)
    # ax.spines['right'].set_linewidth(1.2)
    # However, this might make the grid lines themselves appear thicker if they are tied to spine properties in this style.
    # For now, only enhancing bottom and left as specifically requested for "ticks" previously.
    
    # --- Save Figure ---
    os.makedirs(out_dir, exist_ok=True)
    
    region_str_part = region_name.replace(" visual stream", "").replace(" ", "")
    fname = f"comparison_{neural_dataset.lower()}_{region_str_part}_{compare_rsm_correlation.lower()}.png"
    
    path = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  • Saved: {path}")

    return [path]

# -----------------------------------------------------------------------------
# CLI -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    print(f"  • Loading: {os.path.basename(path)}")
    return pd.read_csv(path)

if __name__ == "__main__":
    cfg = {
        "metric": "Spearman",
        "region": "ventral visual stream",
        "subject_idx": [0, 1, 2, 3, 4, 5, 6, 7],
        "layers": ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2"],
        "results_csv": "/home/ymehta3/research/VisionAI/visreps/logs/full-vs-pcs_nsd.csv",
        "pca_plot_n_classes": [2, 4, 8, 16, 32, 64], # PCA n_classes to plot
        "dataset": "nsd"
    }

    data_df = _load_csv(cfg["results_csv"])
    pca_df, full_df = plt_utils.split_and_select_df(data_df,
                                            dataset=cfg["dataset"],
                                            metric=cfg["metric"],
                                            region=cfg["region"],
                                            epoch=20,
                                            subject_idx=cfg["subject_idx"],
                                            layers=cfg["layers"])
    _, initial_df = plt_utils.split_and_select_df(data_df,
                                dataset=cfg["dataset"],
                                metric=cfg["metric"],
                                region=cfg["region"],
                                epoch=0,
                                subject_idx=cfg["subject_idx"])

    pca_df = plt_utils.avg_over_subject_idx_seed(pca_df)
    full_df = plt_utils.avg_over_subject_idx_seed(full_df)
    initial_df = plt_utils.avg_over_subject_idx_seed(initial_df)

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
    out_dir = "/home/ymehta3/research/VisionAI/visreps/plotters/fig4"
    os.makedirs(out_dir, exist_ok=True)

    # ----------------------------------------
    # call the plotter
    # ----------------------------------------
    create_comparison_plots(
        initial_scores=initial_scores,
        final_scores_full_model=final_scores_full_model,
        pca_scores_final_epoch=pca_scores_final_epoch,
        layer_order=layer_order,
        neural_dataset=cfg["dataset"],
        compare_rsm_correlation=cfg["metric"],
        region_name=cfg["region"],
        min_y=min_y,
        max_y=max_y,
        pca_sizes_for_plot=cfg["pca_plot_n_classes"],
        out_dir=out_dir,
    )