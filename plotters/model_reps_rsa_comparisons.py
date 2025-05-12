import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.stats
from pathlib import Path
import seaborn as sns

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
    output_dir="plots"
):
    """Plots stylized RSA scores and saves the plot."""
    if not layer_names or (not any(not np.isnan(s) for s in scores_f1f2 if isinstance(s, float)) and \
                           not any(not np.isnan(s) for s in scores_t1t2 if isinstance(s, float)) and \
                           not any(not np.isnan(s) for s in scores_f1t1 if isinstance(s, float))):
        print("No valid (non-NaN) data to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    sns.set_theme(style="whitegrid", context="paper")

    x_indices = np.arange(len(layer_names))

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot F1 vs F2: "1k reconstructed corr (between seeds)" - horizontal line marker
    label_f1f2 = "1k reconstructed corr (between seeds)"
    ax.plot(x_indices, scores_f1f2, marker='_', linestyle='None', markersize=20, markeredgewidth=3, label=label_f1f2, color=sns.color_palette()[0])

    # Plot T1 vs T2: "{n_pca_cls} classes corr (between seeds)" - horizontal line marker
    label_t1t2 = f"{n_pca_cls_val} classes corr (between seeds)"
    ax.plot(x_indices, scores_t1t2, marker='_', linestyle='None', markersize=20, markeredgewidth=3, label=label_t1t2, color=sns.color_palette()[1])
    
    # Plot F1 vs T1: "1k vs {n_pca_cls} classes corr" - cross marker
    label_f1t1 = f"1k vs {n_pca_cls_val} classes corr"
    ax.plot(x_indices, scores_f1t1, marker='x', linestyle='None', markersize=10, markeredgewidth=2, label=label_f1t1, color=sns.color_palette()[2])

    ax.set_ylabel(f"RSA Score ({correlation_method})")
    ax.set_xlabel("Layer")
    ax.set_title(f"Inter-Model Layer RSA Comparison (pc{n_pcs_for_title})")
    ax.set_xticks(x_indices)
    ax.set_xticklabels(layer_names, rotation=45, ha="right")
    ax.legend(frameon=False)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    sns.despine(trim=True)

    fig.tight_layout()
    
    plot_filename = f"rsa_comparison_pc{n_pcs_for_title}_{correlation_method}_stylized.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Compare RSMs based on n_pcs and plot stylized layer-wise RSA scores.")
    parser.add_argument("--n_pca_cls", type=int, default=4,  # Changed default to a typical value like 2
                        help="PCA classes integer value (e.g., 2, 4), used to form directory name 'pc<value>' and for labels. Default: 2.")
    parser.add_argument("--correlation_method", default="Kendall", choices=["Pearson", "Spearman", "Kendall"],
                        help="Correlation method for comparing RSMs (default: Kendall).")
    parser.add_argument("--output_dir", default="plots", help="Directory to save the output plot (default: plots).")
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