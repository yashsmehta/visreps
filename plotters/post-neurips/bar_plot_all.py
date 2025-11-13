import sys
import pandas as pd
import numpy as np
import os
from scipy import stats
from plotting_utils import plot_brain_score_barplot, get_best_layer_scores

# ============================================================================
# CONFIGURATION
# ============================================================================
REGION_TO_PLOT = 'ventral visual stream'  # Options: 'early visual stream', 'ventral visual stream', etc.
ARCHITECTURES_TO_PLOT = ['alexnet', 'dino', 'clip', 'dreamsim']  # Which architectures to include
MAX_PCA_CLASSES = 128  # Maximum PCA class count (e.g., 64 means 2-64)


# --- Main Script ---
if __name__ == "__main__":
    # ---------------- config ----------------
    base_log_path = 'logs/'
    pca_csv = 'all_pca_classes.csv'
    dreamsim_pca_csv = 'dreamsim_pca.csv'
    k1k_csv = 'imagenet1k.csv'
    epoch_to_plot = 20
    
    # Generate pca_classes_to_plot based on MAX_PCA_CLASSES
    all_pca_classes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    pca_classes_to_plot = [c for c in all_pca_classes if c <= MAX_PCA_CLASSES]
    
    out_png = f"plotters/post-neurips/barplt_best_layer_region_{REGION_TO_PLOT.lower().replace(' ','_')}.png"

    # ---------------- load PCA data (all layers) ----------------
    df_pca = pd.read_csv(os.path.join(base_log_path, pca_csv))
    df_pca['layer'] = df_pca['layer'].str.lower()
    df_pca = df_pca[
        (df_pca['region'].str.lower() == REGION_TO_PLOT.lower()) &
        (df_pca['epoch'] == epoch_to_plot) &
        (df_pca['pca_labels'] == True) &
        (df_pca['pca_n_classes'].isin(pca_classes_to_plot))
    ]
    
    # ---------------- load DreamSim PCA data (if needed) ----------------
    df_dreamsim = None
    if 'dreamsim' in ARCHITECTURES_TO_PLOT:
        try:
            df_dreamsim = pd.read_csv(os.path.join(base_log_path, dreamsim_pca_csv))
            df_dreamsim['layer'] = df_dreamsim['layer'].str.lower()
            df_dreamsim = df_dreamsim[
                (df_dreamsim['region'].str.lower() == REGION_TO_PLOT.lower()) &
                (df_dreamsim['epoch'] == epoch_to_plot) &
                (df_dreamsim['pca_labels'] == True) &
                (df_dreamsim['pca_n_classes'].isin(pca_classes_to_plot))
            ]
        except FileNotFoundError:
            print(f"Warning: {dreamsim_pca_csv} not found, skipping DreamSim-PCA")
            ARCHITECTURES_TO_PLOT.remove('dreamsim')

    # ---------------- load 1K data (all layers) ----------------
    df_1k = pd.read_csv(os.path.join(base_log_path, k1k_csv))
    df_1k['layer'] = df_1k['layer'].str.lower()
    df_1k = df_1k[
        (df_1k['region'].str.lower() == REGION_TO_PLOT.lower()) &
        (df_1k['epoch'] == epoch_to_plot) &
        (df_1k['pca_labels'] == False)
    ]


    # ---------------- structure data by (arch, n_classes) ----------------
    scores_by_arch_class = {}
    
    # Map architecture names to their pca_labels_folder patterns
    arch_folder_map = {
        'alexnet': ['pca_labels_imagenet1k', 'pca_labels_alexnet'],
        'clip': ['pca_labels_clip'],
        'dino': ['pca_labels_dino'],
    }
    
    # Load data for each requested architecture from df_pca
    for arch in ARCHITECTURES_TO_PLOT:
        if arch == 'dreamsim':
            # Handle DreamSim separately from its own CSV
            if df_dreamsim is not None and len(df_dreamsim) > 0:
                dreamsim_best = get_best_layer_scores(df_dreamsim, ['pca_n_classes'])
                for n_cls, (scores, best_layer) in dreamsim_best.items():
                    scores_by_arch_class[(arch, n_cls)] = scores
                    print(f"{arch.upper()}-PCA {n_cls}: best layer = {best_layer}")
        elif arch in arch_folder_map:
            # Handle architectures from all_pca_classes.csv
            for folder in arch_folder_map[arch]:
                df_arch = df_pca[df_pca['pca_labels_folder'] == folder]
                if len(df_arch) > 0:
                    arch_best = get_best_layer_scores(df_arch, ['pca_n_classes'])
                    for n_cls, (scores, best_layer) in arch_best.items():
                        scores_by_arch_class[(arch, n_cls)] = scores
                        print(f"{arch.upper()}-PCA {n_cls}: best layer = {best_layer} (from {folder})")

    # ImageNet-1K: find best layer
    if len(df_1k) > 0:
        layer_means = df_1k.groupby('layer')['score'].mean()
        best_layer_1k = layer_means.idxmax()
        best_layer_df = df_1k[df_1k['layer'] == best_layer_1k]
        scores_by_arch_class[('1K', None)] = best_layer_df['score'].tolist()
        print(f"ImageNet-1K: best layer = {best_layer_1k}")

    # ---------------- significance testing ----------------
    print("\n=== Paired t-tests vs ImageNet-1K (p<0.01 threshold) ===")
    scores_1k_test = scores_by_arch_class.get(('1K', None), None)
    if scores_1k_test is not None:
        print(f"1K baseline: mean={np.mean(scores_1k_test):.4f}")
        for key, vals in scores_by_arch_class.items():
            arch, n_cls = key
            if arch != '1K' and len(vals) == len(scores_1k_test):
                t_stat, p_val = stats.ttest_rel(vals, scores_1k_test)
                sig_mark = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                print(f"{arch:10s} {str(n_cls):5s}: mean={np.mean(vals):.4f}, t={t_stat:6.3f}, p={p_val:.4f} {sig_mark}")

    # ---------------- plot ----------------
    plot_brain_score_barplot(
        scores_by_arch_class, 
        pca_classes_to_plot, 
        ARCHITECTURES_TO_PLOT, 
        REGION_TO_PLOT, 
        out_png,
        enable_significance=True
    )