"""Bar plot for semantic alignment scores (Gemini embeddings)."""

import sys
import pandas as pd
import numpy as np
import os
from scipy import stats
from plotters.utils import plot_brain_score_barplot, get_best_layer_scores


# ============================================================================
# CONFIGURATION
# ============================================================================
ARCHITECTURES_TO_PLOT = ['alexnet', 'dino']
MAX_PCA_CLASSES = 128 


def extract_architecture(row):
    """Extract architecture name from pca_labels_folder or checkpoint_dir."""
    folder = row['pca_labels_folder']
    if 'clip' in folder:
        return 'clip'
    elif 'dino' in folder:
        return 'dino'
    elif 'alexnet' in folder or 'imagenet1k' in folder:
        return 'alexnet'
    return 'unknown'


if __name__ == "__main__":
    base_log_path = 'logs/'
    semantic_csv = 'semantic_align.csv'
    epoch_to_plot = 20
    
    all_pca_classes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    pca_classes_to_plot = [c for c in all_pca_classes if c <= MAX_PCA_CLASSES]
    
    out_png = "plotters/post-neurips/barplt_semantic_alignment.png"
    
    # Load semantic alignment data
    df = pd.read_csv(os.path.join(base_log_path, semantic_csv))
    df['layer'] = df['layer'].str.lower()
    df = df[df['epoch'] == epoch_to_plot]
    
    # Extract architecture
    df['architecture'] = df.apply(extract_architecture, axis=1)
    
    # Separate 1K baseline from PCA
    df_1k = df[df['cfg_id'] == 1000]
    df_pca = df[(df['cfg_id'] != 1000) & (df['cfg_id'].isin(pca_classes_to_plot))]
    
    # Build scores_by_arch_class dict
    scores_by_arch_class = {}
    
    # Process PCA data: group by (architecture, cfg_id) and find best layer
    for arch in ARCHITECTURES_TO_PLOT:
        df_arch = df_pca[df_pca['architecture'] == arch]
        if len(df_arch) > 0:
            arch_best = get_best_layer_scores(df_arch, ['cfg_id'])
            for n_cls, (scores, best_layer) in arch_best.items():
                scores_by_arch_class[(arch, n_cls)] = scores
                print(f"{arch.upper()}-PCA {n_cls}: best layer = {best_layer}, score = {np.mean(scores):.4f}")
    
    # Process 1K baseline: find best layer across all architectures
    if len(df_1k) > 0:
        layer_means = df_1k.groupby('layer')['score'].mean()
        best_layer_1k = layer_means.idxmax()
        best_layer_df = df_1k[df_1k['layer'] == best_layer_1k]
        scores_by_arch_class[('1K', None)] = best_layer_df['score'].tolist()
        print(f"ImageNet-1K: best layer = {best_layer_1k}, mean score = {np.mean(best_layer_df['score']):.4f}")
    
    # Print comparison (no paired t-tests since only 1 score per condition per arch)
    print("\n=== Semantic Alignment Scores (vs ImageNet-1K) ===")
    scores_1k_test = scores_by_arch_class.get(('1K', None), None)
    if scores_1k_test is not None:
        mean_1k = np.mean(scores_1k_test)
        print(f"1K baseline: mean={mean_1k:.4f}")
        for key, vals in scores_by_arch_class.items():
            arch, n_cls = key
            if arch != '1K':
                mean_val = np.mean(vals)
                diff = mean_val - mean_1k
                print(f"{arch:10s} {str(n_cls):5s}: mean={mean_val:.4f}, diff={diff:+.4f}")
    
    # Plot (disable significance testing since we have single scores)
    plot_brain_score_barplot(
        scores_by_arch_class, 
        pca_classes_to_plot, 
        ARCHITECTURES_TO_PLOT, 
        "Semantic Alignment (Gemini)",
        out_png,
        enable_significance=False,
        ylabel="Semantic Similarity (RSA)"
    )

