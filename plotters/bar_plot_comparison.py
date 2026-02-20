"""Bar plot comparing brain alignment scores across architectures and class counts.

Two modes:
  MODE = "rsa"       — RSA alignment from all_rsa_coarsegrain.csv / all_rsa_1k.csv
  MODE = "semantic"  — Semantic alignment (Gemini) from semantic_align.csv

Usage:
    source .venv/bin/activate && python plotters/bar_plot_comparison.py
"""

import sys
import pandas as pd
import numpy as np
import os
from scipy import stats
sys.path.insert(0, "plotters")
from plotter_utils import plot_brain_score_barplot, get_best_layer_scores

# ============================================================================
# CONFIGURATION — change MODE to switch between RSA and semantic alignment
# ============================================================================
MODE = "rsa"  # "rsa" or "semantic"

# Shared settings
MAX_PCA_CLASSES = 64 if MODE == "rsa" else 128
ARCHITECTURES_TO_PLOT = ['alexnet', 'vit', 'clip', 'dino'] if MODE == "rsa" else ['alexnet', 'dino']

# RSA-only settings
REGION_TO_PLOT = 'ventral visual stream'  # only used when MODE == "rsa"

# Mode-specific CSV / column / output config
MODE_CONFIG = {
    "rsa": {
        "pca_csv": "all_rsa_coarsegrain.csv",
        "k1k_csv": "all_rsa_1k.csv",
        "class_column": "pca_n_classes",
        "enable_significance": True,
        "ylabel": "Brain Similarity (RSA)",
    },
    "semantic": {
        "semantic_csv": "semantic_align.csv",
        "class_column": "cfg_id",
        "enable_significance": False,
        "ylabel": "Semantic Similarity (RSA)",
    },
}


def extract_architecture(row):
    """Extract architecture name from pca_labels_folder (semantic mode)."""
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
    epoch_to_plot = 20
    cfg = MODE_CONFIG[MODE]

    all_pca_classes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    pca_classes_to_plot = [c for c in all_pca_classes if c <= MAX_PCA_CLASSES]

    scores_by_arch_class = {}

    if MODE == "rsa":
        out_png = f"plotters/figures/bar_plot_all_{REGION_TO_PLOT.lower().replace(' ','_')}.png"
        region_or_title = REGION_TO_PLOT

        # Load PCA data
        df_pca = pd.read_csv(os.path.join(base_log_path, cfg["pca_csv"]))
        df_pca['layer'] = df_pca['layer'].str.lower()
        df_pca = df_pca[
            (df_pca['region'].str.lower() == REGION_TO_PLOT.lower()) &
            (df_pca['epoch'] == epoch_to_plot) &
            (df_pca['pca_labels'] == True) &
            (df_pca['pca_n_classes'].isin(pca_classes_to_plot))
        ]

        # Architecture folder mapping
        arch_folder_map = {
            'alexnet': ['pca_labels_imagenet1k', 'pca_labels_alexnet'],
            'clip': ['pca_labels_clip'],
            'dino': ['pca_labels_dino'],
        }

        for arch in ARCHITECTURES_TO_PLOT:
            if arch in arch_folder_map:
                for folder in arch_folder_map[arch]:
                    df_arch = df_pca[df_pca['pca_labels_folder'] == folder]
                    if len(df_arch) > 0:
                        arch_best = get_best_layer_scores(df_arch, ['pca_n_classes'])
                        for n_cls, (scores, best_layer) in arch_best.items():
                            scores_by_arch_class[(arch, n_cls)] = scores
                            print(f"{arch.upper()}-PCA {n_cls}: best layer = {best_layer} (from {folder})")

        # Load 1K baseline
        df_1k = pd.read_csv(os.path.join(base_log_path, cfg["k1k_csv"]))
        df_1k['layer'] = df_1k['layer'].str.lower()
        df_1k = df_1k[
            (df_1k['region'].str.lower() == REGION_TO_PLOT.lower()) &
            (df_1k['epoch'] == epoch_to_plot) &
            (df_1k['pca_labels'] == False)
        ]

        if len(df_1k) > 0:
            layer_means = df_1k.groupby('layer')['score'].mean()
            best_layer_1k = layer_means.idxmax()
            best_layer_df = df_1k[df_1k['layer'] == best_layer_1k]
            scores_by_arch_class[('1K', None)] = best_layer_df['score'].tolist()
            print(f"ImageNet-1K: best layer = {best_layer_1k}")

        # Significance testing
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

    elif MODE == "semantic":
        out_png = "plotters/figures/bar_plot_semantic_alignment.png"
        region_or_title = "Semantic Alignment (Gemini)"

        df = pd.read_csv(os.path.join(base_log_path, cfg["semantic_csv"]))
        df['layer'] = df['layer'].str.lower()
        df = df[df['epoch'] == epoch_to_plot]
        df['architecture'] = df.apply(extract_architecture, axis=1)

        df_1k = df[df['cfg_id'] == 1000]
        df_pca = df[(df['cfg_id'] != 1000) & (df['cfg_id'].isin(pca_classes_to_plot))]

        for arch in ARCHITECTURES_TO_PLOT:
            df_arch = df_pca[df_pca['architecture'] == arch]
            if len(df_arch) > 0:
                arch_best = get_best_layer_scores(df_arch, ['cfg_id'])
                for n_cls, (scores, best_layer) in arch_best.items():
                    scores_by_arch_class[(arch, n_cls)] = scores
                    print(f"{arch.upper()}-PCA {n_cls}: best layer = {best_layer}, score = {np.mean(scores):.4f}")

        if len(df_1k) > 0:
            layer_means = df_1k.groupby('layer')['score'].mean()
            best_layer_1k = layer_means.idxmax()
            best_layer_df = df_1k[df_1k['layer'] == best_layer_1k]
            scores_by_arch_class[('1K', None)] = best_layer_df['score'].tolist()
            print(f"ImageNet-1K: best layer = {best_layer_1k}, mean score = {np.mean(best_layer_df['score']):.4f}")

        # Print comparison
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

    # Plot
    plot_brain_score_barplot(
        scores_by_arch_class,
        pca_classes_to_plot,
        ARCHITECTURES_TO_PLOT,
        region_or_title,
        out_png,
        enable_significance=cfg["enable_significance"],
        ylabel=cfg["ylabel"],
    )
