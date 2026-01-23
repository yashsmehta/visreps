"""Comparison plots helper + CLI

Refactored for conciseness, simplicity, and separation of data prep from plotting.
"""

from __future__ import annotations

import os
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import constr
import seaborn as sns
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, FormatStrFormatter

from plotters.utils import plotter_utils as plt_utils

# -----------------------------------------------------------------------------
# PLOTTER ----------------------------------------------------------------------
# -----------------------------------------------------------------------------
COLORS = {
    "initial": "#7f8c8d", # Grey for Untrained
    "final": "#FFA500",   # Adjusted to DarkOrange for 1000 Classes
    "pca": dict(zip([2, 4, 8, 16, 32, 64], sns.color_palette('Blues', n_colors=6))),
}

def create_reconstructed_rsa_plot(
    reconstruction_data: np.ndarray,
    untrained_data: pd.DataFrame | None = None,
    best_pc_data: pd.DataFrame | None = None,
    output_path: str | None = None,
    cfg: Dict | None = None,
):
    """Generate and save a comparison plot using pre-processed data.

    Args:
        reconstruction_data (np.ndarray): A 3x20 numpy array (seeds x pca_k) for reconstruction scores.
        untrained_data (pd.DataFrame | None): Data for the untrained model (mean/std over seeds for a layer).
        best_pc_data (pd.DataFrame | None): Data for the best PC model (mean/std over seeds for a layer).
        output_path (str | None): Path to save the plot.
        cfg (Dict | None): Configuration dictionary for plot details.
    """
    if cfg is None:
        cfg = {}

    # Use a style suitable for scientific publications
    plt.style.use("seaborn-v0_8-paper")
    
    # Store original rcParams to restore them later
    original_rc_params = plt.rcParams.copy()
    try:
        plt.rcParams.update({
            'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
            'font.size': 11,       # Base size for text elements not otherwise specified
            'axes.titlesize': 11,  # Title font size
            'axes.labelsize': 10,  # X and Y axis labels font size
            'xtick.labelsize': 12, # X tick labels font size - User updated
            'ytick.labelsize': 12, # Y tick labels font size - User updated
            'legend.fontsize': 8,  # Adjusted for more legend items
            'axes.linewidth': 1.0  # Linewidth for axes frame (spines)
        })

        fig, ax = plt.subplots(figsize=(cfg.get("figsize", (5, 4))))

        # Plotting reconstruction data
        if reconstruction_data.shape != (3, 20):
            raise ValueError(f"reconstruction_data must have shape (3, 20), but got {reconstruction_data.shape}")

        pca_k_values = np.arange(1, 21)  # pca_k from 1 to 20
        mean_scores_recon = np.mean(reconstruction_data, axis=0)
        std_scores_recon = np.std(reconstruction_data, axis=0)

        line_color_recon = COLORS["final"] # Use "final" as it's the main line like 1000 classes
        fill_color_recon = sns.light_palette(line_color_recon, n_colors=3)[1]

        ax.plot(pca_k_values, mean_scores_recon, marker='^', linestyle='-', color=line_color_recon, 
                label="1000 classes (reconstructed)", markersize=6, linewidth=2, # 1.875 * 1.3
                markeredgecolor='white', markeredgewidth=0.5, zorder=3)
        ax.fill_between(pca_k_values, mean_scores_recon - std_scores_recon, mean_scores_recon + std_scores_recon, 
                        alpha=0.3, color=fill_color_recon, edgecolor=line_color_recon, linewidth=0.5, zorder=3)

        # Plot Best PC Data (plotting before Untrained to get desired legend order)
        if best_pc_data is not None and not best_pc_data.empty:
            layer_name_best_pc = cfg.get('layers_pc', ['unknown_layer'])[0]
            n_classes_best_pc = int(cfg.get('best_pc_n_classes', [0])[0]) 
            best_pc_layer_scores = best_pc_data[best_pc_data['layer'] == layer_name_best_pc]['score']

            if not best_pc_layer_scores.empty:
                mean_best_pc = best_pc_layer_scores.mean()
                std_best_pc = best_pc_layer_scores.std()
                
                pc_color = COLORS["pca"].get(n_classes_best_pc, '#808080') 
                pc_fill_color = sns.light_palette(pc_color, n_colors=3, as_cmap=False)[1] if isinstance(pc_color, str) else sns.set_hls_values(pc_color, l=0.85)

                ax.plot(pca_k_values, np.full_like(pca_k_values, mean_best_pc, dtype=float),
                        color=pc_color, linestyle='--', linewidth=2.275, 
                        label=f"{n_classes_best_pc} classes (full)", zorder=2)
                ax.fill_between(pca_k_values, mean_best_pc - std_best_pc, mean_best_pc + std_best_pc, 
                               color=pc_fill_color, alpha=0.25, zorder=1)

        # Plot Untrained Data
        if untrained_data is not None and not untrained_data.empty:
            layer_name_untrained = cfg.get('layers_1k', ['unknown_layer'])[0]
            untrained_layer_scores = untrained_data[untrained_data['layer'] == layer_name_untrained]['score']
            if not untrained_layer_scores.empty:
                mean_untrained = untrained_layer_scores.mean()
                std_untrained = untrained_layer_scores.std()
                ax.plot(pca_k_values, np.full_like(pca_k_values, mean_untrained, dtype=float), 
                        color=COLORS["initial"], linestyle=':', linewidth=2.275, 
                        label="Untrained (full)", zorder=2)
                ax.fill_between(pca_k_values, mean_untrained - std_untrained, mean_untrained + std_untrained, 
                                color=COLORS["initial"], alpha=0.2, zorder=1)

        ax.set_xlabel("Number of PCs for Reconstruction") # Fontsize from rcParams
        ax.set_ylabel(f"RSA ({cfg.get('metric', 'Score')})") # Fontsize from rcParams
        
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.set_xlim(left=0.5, right=20.5) # Start x-axis at 0.5, slight pad on right
        
        all_data_points_recon = mean_scores_recon # Use mean for y_lim calcs, stddev already included in fill
        if untrained_data is not None and not untrained_data.empty and 'mean_untrained' in locals():
            all_data_points_recon = np.append(all_data_points_recon, [mean_untrained - std_untrained, mean_untrained + std_untrained])
        if best_pc_data is not None and not best_pc_data.empty and 'mean_best_pc' in locals():
             all_data_points_recon = np.append(all_data_points_recon, [mean_best_pc - std_best_pc, mean_best_pc + std_best_pc])
        
        min_val_data = np.min(all_data_points_recon) if all_data_points_recon.size > 0 else 0
        max_val_data = np.max(all_data_points_recon) if all_data_points_recon.size > 0 else 1

        padding = (max_val_data - min_val_data) * 0.05 if (max_val_data - min_val_data) > 0 else 0.05 # Ensure padding if range is 0
        
        min_y = cfg.get("min_y", min_val_data - padding)
        max_y = cfg.get("max_y", max_val_data + padding)
        ax.set_ylim(min_y, max_y)

        # Y-axis: Dynamic major and minor ticks
        y_range = max_y - min_y
        if y_range > 0.2: # Consistent with full_vs_pcs_nsd.py
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
        elif y_range > 0.05: # Consistent with full_vs_pcs_nsd.py
            ax.yaxis.set_major_locator(MultipleLocator(0.05))
        else:
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6, prune='both')) # Consistent

        ax.yaxis.set_minor_locator(AutoMinorLocator(2)) # Consistent
        
        # Tick parameters
        ax.tick_params(axis='x', which='major', direction='out', length=4, width=plt.rcParams['axes.linewidth'], top=False, right=False)
        ax.tick_params(axis='x', which='minor', direction='out', length=2, width=plt.rcParams['axes.linewidth']*0.75, top=False, right=False)
        
        ax.tick_params(axis='y', which='major', direction='out', length=4, width=plt.rcParams['axes.linewidth'], top=False, right=False)
        # Minor y-ticks: size and format (labelsize from rcParams['ytick.labelsize'])
        minor_tick_labelsize = int(plt.rcParams['ytick.labelsize'] * 0.75)
        ax.tick_params(axis='y', which='minor', direction='out', length=2, width=plt.rcParams['axes.linewidth']*0.75, 
                       labelsize=minor_tick_labelsize, labelleft=True, top=False, right=False) 
        ax.yaxis.set_minor_formatter(FormatStrFormatter("%.3f")) # Consistent

        # Grid
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.5)

        # Spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Bottom and left spines will use 'axes.linewidth' from rcParams
        ax.spines['bottom'].set_linewidth(plt.rcParams['axes.linewidth'])
        ax.spines['left'].set_linewidth(plt.rcParams['axes.linewidth'])
        
        # Adjust legend position if it gets too crowded
        handles, labels = ax.get_legend_handles_labels()
        if handles:
             ax.legend(handles, labels, frameon=True, facecolor='white', edgecolor='black', 
                       loc='best', fontsize=plt.rcParams['legend.fontsize'])

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300) # Ensure DPI for publication quality
            print(f"Plot saved to {output_path}")
        else:
            plt.show()
    finally:
        plt.rcParams.update(original_rc_params) # Restore original rcParams


_SKIP_ALWAYS = {"log_interval", "checkpoint_interval", "cfg_id", "score"}
_PCA_COLS   = ("pca_labels", "pca_n_classes", "reconstruct_from_pcs", "pca_k")

def _avg_over_subject_idx(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse `subject_idx`; keep `seed` (if any) and PCA columns.
    """
    if df.empty or "subject_idx" not in df:
        return df.copy()

    d = df.copy()
    d["subject_idx"] = pd.to_numeric(d["subject_idx"], errors="coerce")
    d = d.dropna(subset=["subject_idx"])
    if d.empty:
        return d

    skip = _SKIP_ALWAYS | {"subject_idx"}
    group_cols = [c for c in d.columns if c not in skip]

    out = (
        d.groupby(group_cols, dropna=False, observed=False)["score"]
          .mean()
          .reset_index()
    )

    keep = ["layer", "score"]
    if "seed" in out.columns:
        keep.append("seed")
    keep += [c for c in _PCA_COLS if c in out.columns]
    return out[keep]

def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    print(f"  • Loading: {os.path.basename(path)}")
    return pd.read_csv(path)

if __name__ == "__main__":
    cfg = {
        "metric": "Spearman",
        "region": "ventral visual stream",
        "layers_1k": ["fc1"],
        "layers_pc": ["fc1"],
        "subject_idx": [0, 1, 2, 3, 4, 5, 6, 7],
        "results_csv": "/home/ymehta3/research/VisionAI/visreps/logs/pc_reconstruction_analysis.csv",
        "best_pc_n_classes": [64], # PCA n_classes to plot
        "dataset": "nsd",
    }

    out_path = f"/home/ymehta3/research/VisionAI/visreps/plotters/fig2/reconstructed_rsa_{cfg['dataset']}_{cfg['region'][:6]}.png"
    data_df = _load_csv(cfg["results_csv"])
    _, full_df = plt_utils.split_and_select_df(data_df,
                                            dataset=cfg["dataset"],
                                            metric=cfg["metric"],
                                            region=cfg["region"],
                                            epoch=20,
                                            subject_idx=cfg["subject_idx"],
                                            reconstruct_from_pcs=True,
                                            layers=cfg["layers_1k"])

    full_df = _avg_over_subject_idx(full_df)
    print("full_df: ", full_df)

    best_pc_df, _ = plt_utils.split_and_select_df(data_df,
                                        dataset=cfg["dataset"],
                                        metric=cfg["metric"],
                                        region=cfg["region"],
                                        epoch=20,
                                        subject_idx=cfg["subject_idx"],
                                        pca_n_classes=cfg["best_pc_n_classes"],
                                        reconstruct_from_pcs=False,
                                        layers=cfg["layers_pc"])

    best_pc_df = _avg_over_subject_idx(best_pc_df)
    print("best_pc_df: ", best_pc_df)

    untrained_results_csv = f"/home/ymehta3/research/VisionAI/visreps/logs/full-vs-pcs_{cfg['dataset']}.csv"
    data_df = _load_csv(untrained_results_csv)
    _, untrained_df = plt_utils.split_and_select_df(data_df,
                                            dataset=cfg["dataset"],
                                            metric=cfg["metric"],
                                            region=cfg["region"],
                                            epoch=0,
                                            subject_idx=cfg["subject_idx"],
                                            reconstruct_from_pcs=False,
                                            layers=cfg["layers_1k"])

    # This is the untrained results: should be plotted as a horizontal dashed line (mean + shaded std-dev).
    untrained_df = _avg_over_subject_idx(untrained_df)
    print("untrained_df: ", untrained_df)

    # Prepare reconstruction_data (3 seeds x 20 pca_k values)
    # Assuming seeds are 1, 2, 3 and pca_k are 1 to 20
    # And the layer is fixed based on cfg["layers_1k"][0]
    
    # Filter for the specific layer
    layer_to_plot = cfg["layers_1k"][0]
    reconstruction_subset = full_df[full_df['layer'] == layer_to_plot]

    # Pivot table to get seeds as rows, pca_k as columns
    try:
        pivot_df = reconstruction_subset.pivot_table(
            index='seed', 
            columns='pca_k', 
            values='score'
        )
        # Ensure pca_k columns are sorted and cover 1-20
        expected_pca_k_cols = list(range(1, 21))
        pivot_df = pivot_df.reindex(columns=expected_pca_k_cols)
        
        # Fill NaN with a placeholder if necessary, or raise error
        # For now, let's assume data is complete for seeds 1,2,3 and pca_k 1-20
        # If seeds are not exactly [1,2,3], this needs adjustment
        # Forcing index to be 0,1,2 if seeds are e.g. 1,2,3
        # This assumes seeds are consecutive and start from 1, or are just three distinct seeds.
        # A more robust way would be to map seed values [1,2,3] to [0,1,2] indices.
        
        # Let's ensure we have 3 seeds. If not, this will fail or give wrong shape.
        if len(pivot_df.index) != 3:
            print(f"Warning: Expected 3 seeds, but found {len(pivot_df.index)}. Seeds: {pivot_df.index.tolist()}")
            # Handle case with fewer/more than 3 seeds if necessary
            # For now, proceed if it's 3, otherwise error out in the plot function
        
        reconstruction_data_np = pivot_df.to_numpy()
        
        if reconstruction_data_np.shape[1] != 20:
            raise ValueError(f"Pivot table for reconstruction data does not have 20 pca_k columns. Shape: {reconstruction_data_np.shape}")

        print(f"Shape of reconstruction_data_np: {reconstruction_data_np.shape}")
        print("Sample of reconstruction_data_np (first 5 pca_k values):")
        print(reconstruction_data_np[:, :5])

        plot_cfg = {
            "metric": cfg["metric"],
            "title": f'{cfg["metric"]} Score vs. PCs ({cfg["region"]}, Layer: {layer_to_plot}) - Reconstructed',
            "layers_1k": cfg["layers_1k"],
            "layers_pc": cfg["layers_pc"],
            "best_pc_n_classes": cfg["best_pc_n_classes"],
        }
        
        create_reconstructed_rsa_plot(
            reconstruction_data=reconstruction_data_np,
            untrained_data=untrained_df,
            best_pc_data=best_pc_df,
            output_path=out_path,
            cfg=plot_cfg
        )

    except KeyError as e:
        print(f"Error during pivot: {e}. This might happen if 'seed' or 'pca_k' is not in full_df or has unexpected values.")
        print("Columns in full_df:", full_df.columns)
        print("Unique seeds:", full_df['seed'].unique() if 'seed' in full_df else "N/A")
        print("Unique pca_k:", full_df['pca_k'].unique() if 'pca_k' in full_df else "N/A")

    # print(full_df)
