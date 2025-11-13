import sys
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter
import pandas as pd
import numpy as np
import os
from scipy import stats

# ============================================================================
# CONFIGURATION
# ============================================================================
REGION_TO_PLOT = 'ventral visual stream'  # Options: 'early visual stream', 'ventral visual stream', etc.
ARCHITECTURES_TO_PLOT = ['alexnet', 'dino', 'clip', 'dreamsim']  # Which architectures to include
MAX_PCA_CLASSES = 128  # Maximum PCA class count (e.g., 64 means 2-64)


def plot_brain_score_barplot(scores_by_arch_class: dict[tuple, list[float]],
                             pca_classes: list[int],
                             architectures: list[str],
                             region_name: str,
                             out_png: str) -> None:
    """
    Grouped bar-plot brain similarity for multiple PCA architectures and 1K baseline.

    Parameters
    ----------
    scores_by_arch_class : dict[tuple, list[float]]
        Keys = (architecture, n_classes) tuples, e.g., ('alexnet', 2), ('1K', None).
        Values = list of per-subject scores (8 subjects).
    pca_classes : list[int]
        Ordered list of PCA class counts, e.g., [2, 4, 8, ..., 64].
    architectures : list[str]
        List of architectures to plot (e.g., ['alexnet', 'dino', 'clip', 'dreamsim']).
    region_name : str
        Brain region name to display as title.
    out_png : str
        Destination PNG filename.
    """
    # ── define colors (Nature-style palette) ─────────────────────────────────
    color_map = {
        'alexnet': '#1f77b4',   # blue
        'dino': '#ff7f0e',      # orange
        'clip': '#2d7f2d',      # dark green
        'dreamsim': '#9467bd',  # purple
    }
    k1k_color = '#666666'  # neutral gray for baseline
    
    # ── plotting ─────────────────────────────────────────────────────────────
    sns.set_theme(style='ticks', context='paper', font_scale=1.2)
    fig, ax = plt.subplots(figsize=(16, 6))
    
    n_archs = len(architectures)
    bar_width = 0.24
    intra_group_gap = 0.04  # gap between bars within same group
    group_gap = 0.30  # gap between different class groups
    
    # ── get 1K scores for paired t-tests ─────────────────────────────────────
    scores_1k = scores_by_arch_class.get(('1K', None), None)
    
    # ── plot grouped bars with significance tests ────────────────────────────
    for i, n_cls in enumerate(pca_classes):
        base_pos = i * (n_archs * bar_width + (n_archs - 1) * intra_group_gap + group_gap)
        
        for arch_idx, arch in enumerate(architectures):
            if (arch, n_cls) in scores_by_arch_class:
                scores = scores_by_arch_class[(arch, n_cls)]
                mean_val = np.mean(scores)
                
                bar_pos = base_pos + arch_idx * (bar_width + intra_group_gap)
                
                rect = mpatches.FancyBboxPatch(
                    (bar_pos, 0), bar_width, mean_val,
                    boxstyle=mpatches.BoxStyle('Round', pad=.02, rounding_size=.08),
                    facecolor=color_map[arch], edgecolor='black',
                    linewidth=1.0, mutation_aspect=.05
                )
                ax.add_patch(rect)
                
                # Paired t-test with 1K
                if scores_1k is not None and len(scores) == len(scores_1k):
                    t_stat, p_val = stats.ttest_rel(scores, scores_1k)
                    if p_val < 0.01:
                        ax.text(bar_pos + bar_width/2, 0.015, '*',
                               ha='center', va='bottom', fontsize=18, fontweight='bold', color='white')
    
    # ── plot 1K as horizontal dashed line ────────────────────────────────────
    if ('1K', None) in scores_by_arch_class:
        scores_1k = scores_by_arch_class[('1K', None)]
        mean_1k = np.mean(scores_1k)
        ax.axhline(y=mean_1k, color=k1k_color, linestyle='--', linewidth=2.5, 
                   label='ImageNet-1K', zorder=2, alpha=0.9)
    
    # ── axis formatting ──────────────────────────────────────────────────────
    # X-axis ticks at center of each group
    tick_positions = []
    tick_labels = []
    
    for i, n_cls in enumerate(pca_classes):
        base_pos = i * (n_archs * bar_width + (n_archs - 1) * intra_group_gap + group_gap)
        group_width = n_archs * bar_width + (n_archs - 1) * intra_group_gap
        group_center = base_pos + group_width / 2
        tick_positions.append(group_center)
        tick_labels.append(str(n_cls))
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontweight='bold')
    ax.tick_params(axis='x', direction='out', bottom=True, top=False, length=5, color='black', 
                   width=1.5, pad=8, labelsize=16)
    
    # Y-axis formatting (Nature style)
    ax.tick_params(axis='y', which='major', direction='out', left=True, right=False, 
                   labelsize=13, length=6, color='black', width=1.5, pad=6)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    
    def hide_zero_formatter(x, pos):
        return '' if np.isclose(x, 0) else f'{x:.2f}'
    ax.yaxis.set_major_formatter(FuncFormatter(hide_zero_formatter))
    
    ax.tick_params(axis='y', which='minor', direction='out', left=True, right=False, 
                   length=3, color='black', width=1.0)
    
    # Set limits
    all_means = [np.mean(v) for v in scores_by_arch_class.values() if len(v) > 0]
    current_max_y = max(all_means) if all_means else 0
    ax.set_ylim(0, current_max_y + 0.025 if current_max_y > 0 else 0.1)
    
    max_pos = (len(pca_classes) - 1) * (n_archs * bar_width + (n_archs - 1) * intra_group_gap + group_gap)
    max_pos += n_archs * bar_width + (n_archs - 1) * intra_group_gap + 0.5
    ax.set_xlim(-0.5, max_pos)
    ax.set_ylabel('Brain Similarity (RSA)', fontsize=15, labelpad=12, fontweight='normal')
    
    # Title with region name
    ax.set_title(region_name.title(), fontsize=18, fontweight='bold', pad=15)
    
    # Legend (Nature style - clean and minimal) - dynamic based on architectures
    legend_handles = []
    arch_name_map = {
        'alexnet': 'AlexNet',
        'dino': 'DINO',
        'clip': 'CLIP',
        'dreamsim': 'DreamSim',
    }
    for arch in architectures:
        arch_display = arch_name_map.get(arch, arch.capitalize())
        patch = mpatches.Patch(facecolor=color_map[arch], edgecolor='black', 
                               linewidth=1.0, label=f'{arch_display} classes')
        legend_handles.append(patch)
    
    k1k_line = mlines.Line2D([], [], color=k1k_color, linestyle='--', 
                             linewidth=2.5, label='ImageNet-1K')
    legend_handles.append(k1k_line)
    
    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5), 
             frameon=True, fontsize=14, framealpha=0.95, edgecolor='black', fancybox=False)
    
    sns.despine(right=True, top=True, offset=8)
    ax.spines['bottom'].set_linewidth(1.8)
    ax.spines['left'].set_linewidth(1.8)
    
    plt.tight_layout(pad=1.2, rect=[0, 0, 0.85, 1])  # Leave space for legend on right
    plt.savefig(out_png, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Plot saved → {out_png}")

# --- Example Usage ---
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

    # ---------------- helper: find best layer for each condition ----------------
    def get_best_layer_scores(df, group_cols):
        """
        For each unique combination of group_cols, find the layer with highest mean score.
        Returns dict mapping group_key -> list of subject scores from best layer.
        """
        result = {}
        for group_vals, group_df in df.groupby(group_cols):
            # Unpack tuple if single column groupby
            if len(group_cols) == 1:
                group_vals = group_vals[0] if isinstance(group_vals, tuple) else group_vals
            
            # Find best layer for this group
            layer_means = group_df.groupby('layer')['score'].mean()
            best_layer = layer_means.idxmax()
            
            # Get subject scores from best layer
            best_layer_df = group_df[group_df['layer'] == best_layer]
            scores = best_layer_df['score'].tolist()
            
            result[group_vals] = (scores, best_layer)
        
        return result

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
    plot_brain_score_barplot(scores_by_arch_class, pca_classes_to_plot, ARCHITECTURES_TO_PLOT, REGION_TO_PLOT, out_png)