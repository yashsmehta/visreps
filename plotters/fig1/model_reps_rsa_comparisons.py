import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.stats
from pathlib import Path
import seaborn as sns
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, FormatStrFormatter

# Adapted from visreps.analysis.rsa.py
_CORR_FUNCS = {
    "pearson": scipy.stats.pearsonr,
    "spearman": scipy.stats.spearmanr,
    "kendall": scipy.stats.kendalltau,
}

def compute_rsm_correlation(
    rsm1: torch.Tensor, rsm2: torch.Tensor, *, correlation: str = "Kendall"
) -> float:
    """Correlation between two RSMs using Pearson / Spearman / Kendall.

    Returns NaN if correlation cannot be computed (e.g., zero variance).
    """
    if rsm1.shape != rsm2.shape or rsm1.ndim != 2:
        raise ValueError("RSMs must share the same 2-D shape")

    n = rsm1.size(0)
    if n <= 1:
        return float("nan")

    idx = torch.triu_indices(n, n, offset=1)
    v1 = rsm1[idx[0], idx[1]].cpu().numpy()
    v2 = rsm2[idx[0], idx[1]].cpu().numpy()
    
    if v1.size == 0:
        return float("nan")

    corr_method = correlation.lower()
    if corr_method not in _CORR_FUNCS:
        raise ValueError("correlation must be 'Pearson', 'Spearman', or 'Kendall'")

    try:
        if np.all(v1 == v1[0]) or np.all(v2 == v2[0]):
            return float("nan")
        val, p_value = _CORR_FUNCS[corr_method](v1, v2)
        if np.isnan(val):
            return float("nan")
        return float(val)
    except Exception:
        return float("nan")

def load_rsms(file_path):
    """Loads RSMs from an .npz file."""
    try:
        data = np.load(file_path, allow_pickle=True)
        keys = list(data.keys())
        if not keys:
             if 'arr_0' in data and isinstance(data['arr_0'].item(), dict):
                 return data['arr_0'].item()
             else:
                raise ValueError("NPZ file does not seem to contain a dictionary of RSMs or is in an unexpected format.")
        rsms = {key: data[key] for key in data.keys()}
        return rsms
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading RSMs from {file_path}: {e}")
        return None

def plot_rsa_scores_grouped(
    layer_names, 
    scores_f1f2,  # For "1k reconstructed corr (between seeds)"
    scores_t1t2,  # For "{n_pca_cls} classes corr (between seeds)"
    scores_f1t1,  # For "1k vs {n_pca_cls} classes corr" (cross marker)
    n_pca_cls_val, 
    n_pcs_for_title, 
    correlation_method, 
    output_dir="plotters/fig1"
):
    """Plots stylized RSA scores and saves the plot."""
    if not layer_names or (not any(not np.isnan(s) for s in scores_f1f2 if isinstance(s, float)) and \
                           not any(not np.isnan(s) for s in scores_t1t2 if isinstance(s, float)) and \
                           not any(not np.isnan(s) for s in scores_f1t1 if isinstance(s, float))):
        print("No valid (non-NaN) data to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use("seaborn-v0_8-paper")
    original_rc_params = plt.rcParams.copy()

    try:
        plt.rcParams.update({
            'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
            'font.size': 17,       # Base size: 11 * 1.5 = 16.5 -> 17
            'axes.titlesize': 17,  # Title font size (though title is removed): 11 * 1.5 = 16.5 -> 17
            'axes.labelsize': 15,  # X and Y axis labels: 10 * 1.5 = 15
            'xtick.labelsize': 18, # X tick labels: 12 * 1.5 = 18
            'ytick.labelsize': 18, # Y tick labels: 12 * 1.5 = 18
            'legend.fontsize': 12, # Legend font size: 8 * 1.5 = 12
            'axes.linewidth': 1.0  # Linewidth for axes frame (spines)
        })

        # Define colors
        color_f1f2 = 'silver'
        color_t1t2 = 'dimgray'
        color_f1t1 = '#FF6B6B' # Softer red

        x_indices = np.arange(len(layer_names))
        bar_width = 0.25
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot F1 vs F2
        label_f1f2 = "Inter-seed (1K rec.)"
        ax.bar(x_indices - bar_width, scores_f1f2, bar_width, label=label_f1f2, color=color_f1f2)

        # Plot T1 vs T2
        label_t1t2 = f"Inter-seed ({n_pca_cls_val} CLS)"
        ax.bar(x_indices, scores_t1t2, bar_width, label=label_t1t2, color=color_t1t2)
        
        # Plot F1 vs T1
        label_f1t1 = f"1K vs {n_pca_cls_val} CLS"
        ax.bar(x_indices + bar_width, scores_f1t1, bar_width, label=label_f1t1, color=color_f1t1)

        ax.set_ylabel(f"RSA ({correlation_method})")
        # ax.set_xlabel("Layer") # X-axis title removed

        ax.set_xticks(x_indices)
        ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=plt.rcParams['xtick.labelsize'])
        
        # X-axis tick parameters
        ax.tick_params(axis='x', which='major', direction='out', length=4, width=plt.rcParams['axes.linewidth'], top=False, right=False)

        # Y-axis limits and ticks
        ax.set_ylim(0, 1) # Set Y-axis from 0 to 1
        ax.yaxis.set_major_locator(MultipleLocator(0.5)) # Major ticks at 0, 0.5, 1
        ax.yaxis.set_minor_locator(MultipleLocator(0.25)) # Minor ticks at 0.25, 0.75
        ax.yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))

        # Y-axis tick parameters
        ax.tick_params(axis='y', which='major', direction='out', length=4, width=plt.rcParams['axes.linewidth'], right=False, top=False)
        ax.tick_params(axis='y', which='minor', direction='out', length=2, width=plt.rcParams['axes.linewidth']*0.75, 
                       labelsize=int(plt.rcParams['ytick.labelsize'] * 0.75), labelleft=True, right=False, top=False)
        
        # Grid
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.5)

        # Spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(plt.rcParams['axes.linewidth'])
        ax.spines['left'].set_linewidth(plt.rcParams['axes.linewidth'])
        
        # Legend
        # handles, labels = ax.get_legend_handles_labels()
        # if handles:
        #     ax.legend(handles, labels, frameon=True, facecolor='white', edgecolor='black', 
        #               loc='center left', bbox_to_anchor=(1, 0.5), fontsize=plt.rcParams['legend.fontsize']) # Legend to the right
        
        fig.tight_layout() # Reverted layout adjustment
        
        plot_filename = f"rsa_comparison_pc{n_pcs_for_title}_{correlation_method}_bars.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300)
        print(f"Plot saved to {plot_path}")

    finally:
        plt.rcParams.update(original_rc_params)

def main():
    parser = argparse.ArgumentParser(description="Compare RSMs based on n_pcs and plot stylized layer-wise RSA scores.")
    parser.add_argument("--n_pca_cls", type=int, default=4,  # Changed default to a typical value like 2
                        help="PCA classes integer value (e.g., 2, 4), used to form directory name 'pc<value>' and for labels. Default: 2.")
    parser.add_argument("--correlation_method", default="Kendall", choices=["Pearson", "Spearman", "Kendall"],
                        help="Correlation method for comparing RSMs (default: Kendall).")
    parser.add_argument("--output_dir", default="plotters/fig1", help="Directory to save the output plot (default: plotters/fig1).")
    parser.add_argument("--base_rsm_dir", default="model_checkpoints/RSMs", help="Base directory for RSM files.")
    
    args = parser.parse_args()

    pca_dir_name = f"pca{args.n_pca_cls}cls"
    base_path = Path(args.base_rsm_dir) / pca_dir_name
    
    # Calculate n_pcs (k value for pca_k_{k} in filenames)
    # Ensure n_pca_cls is at least 1 for log2, though typically it's >= 2
    if args.n_pca_cls < 1:
        print("Error: --n_pca_cls must be 1 or greater.")
        return
    n_pcs = int(np.log2(args.n_pca_cls)) if args.n_pca_cls > 0 else 0 # log2(1)=0, for n_pca_cls=1, n_pcs=0

    path_f1 = base_path / f"rsms_nsd_pca_labels_False_pca_k_{n_pcs}_cfgid_1_seed_1.npz"
    path_f2 = base_path / f"rsms_nsd_pca_labels_False_pca_k_{n_pcs}_cfgid_1_seed_2.npz"
    path_t1 = base_path / f"rsms_nsd_pca_labels_True_cfgid_{n_pcs}_seed_1.npz"
    path_t2 = base_path / f"rsms_nsd_pca_labels_True_cfgid_{n_pcs}_seed_2.npz"

    print(f"Attempting to load RSMs for {pca_dir_name} (using k={n_pcs}):")
    print(f"  File F1 (PCA_F, seed1): {path_f1}")
    print(f"  File F2 (PCA_F, seed2): {path_f2}")
    print(f"  File T1 (PCA_T, seed1): {path_t1}")
    print(f"  File T2 (PCA_T, seed2): {path_t2}")

    rsms_f1 = load_rsms(path_f1)
    rsms_f2 = load_rsms(path_f2)
    rsms_t1 = load_rsms(path_t1)
    rsms_t2 = load_rsms(path_t2)

    critical_files_loaded = True
    missing_files = []
    if rsms_f1 is None: missing_files.append(f"F1 ({path_f1})")
    if rsms_f2 is None: missing_files.append(f"F2 ({path_f2})")
    if rsms_t1 is None: missing_files.append(f"T1 ({path_t1})")
    if rsms_t2 is None: missing_files.append(f"T2 ({path_t2})")

    if missing_files:
        print(f"Failed to load one or more critical RSM files: {', '.join(missing_files)}. Exiting.")
        return

    common_layers_set = set(rsms_f1.keys()) & set(rsms_f2.keys()) & set(rsms_t1.keys()) & set(rsms_t2.keys())
    if "neural" in common_layers_set:
        common_layers_set.remove("neural")
        
    plot_layers = sorted(list(common_layers_set))
        
    if not plot_layers:
        print("No common layers found across all required RSM files (F1, F2, T1, T2), excluding 'neural'. Exiting.")
        return

    print(f"Found common layers for plotting: {plot_layers}")

    scores_f1_f2 = []  # Was plot_scores_seed
    scores_f1_t1 = []  # Was plot_scores_pca
    scores_t1_t2 = []  # Was plot_scores_seed_pca_comp

    for layer in plot_layers:
        # print(f"Processing layer: {layer}")
        rsm_f1_layer = torch.from_numpy(rsms_f1[layer]).float()
        rsm_f2_layer = torch.from_numpy(rsms_f2[layer]).float()
        rsm_t1_layer = torch.from_numpy(rsms_t1[layer]).float()
        rsm_t2_layer = torch.from_numpy(rsms_t2[layer]).float()

        score_val_f1f2 = compute_rsm_correlation(rsm_f1_layer, rsm_f2_layer, correlation=args.correlation_method)
        scores_f1_f2.append(score_val_f1f2) 
        # print(f"  F1 vs F2 RSA: {f'{score_val_f1f2:.4f}' if not np.isnan(score_val_f1f2) else 'NaN'}")

        score_val_f1t1 = compute_rsm_correlation(rsm_f1_layer, rsm_t1_layer, correlation=args.correlation_method)
        scores_f1_t1.append(score_val_f1t1) 
        # print(f"  F1 vs T1 RSA: {f'{score_val_f1t1:.4f}' if not np.isnan(score_val_f1t1) else 'NaN'}")
        
        score_val_t1t2 = compute_rsm_correlation(rsm_t1_layer, rsm_t2_layer, correlation=args.correlation_method)
        scores_t1_t2.append(score_val_t1t2)
        # print(f"  T1 vs T2 RSA: {f'{score_val_t1t2:.4f}' if not np.isnan(score_val_t1t2) else 'NaN'}")

    all_f1f2_nan = all(np.isnan(s) for s in scores_f1_f2 if isinstance(s, float))
    all_f1t1_nan = all(np.isnan(s) for s in scores_f1_t1 if isinstance(s, float))
    all_t1t2_nan = all(np.isnan(s) for s in scores_t1_t2 if isinstance(s, float))

    if not plot_layers or (all_f1f2_nan and all_f1t1_nan and all_t1t2_nan):
        print("No valid (non-NaN) RSA scores computed for any layer or comparison type. Plot will not be generated.")
    else:
        print(f"Plotting RSA scores for {len(plot_layers)} layers using {args.correlation_method} correlation.")
        plot_rsa_scores_grouped(
            layer_names=plot_layers, 
            scores_f1f2=scores_f1_f2, 
            scores_t1t2=scores_t1_t2, 
            scores_f1t1=scores_f1_t1, 
            n_pca_cls_val=args.n_pca_cls, 
            n_pcs_for_title=n_pcs,  # This is the 'k' from pca_k_k
            correlation_method=args.correlation_method, 
            output_dir=args.output_dir
        )

if __name__ == "__main__":
    main() 