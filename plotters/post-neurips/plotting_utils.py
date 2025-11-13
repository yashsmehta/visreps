"""Shared plotting utilities for brain and semantic alignment visualizations."""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter
from scipy import stats


def get_best_layer_scores(df, group_cols):
    """
    For each unique combination of group_cols, find the layer with highest mean score.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: layer, score, and group_cols
    group_cols : list[str]
        Column names to group by (e.g., ['pca_n_classes'] or ['subject_idx', 'pca_n_classes'])
    
    Returns
    -------
    dict
        Maps group_key -> (scores_list, best_layer_name)
    """
    result = {}
    for group_vals, group_df in df.groupby(group_cols):
        if len(group_cols) == 1:
            group_vals = group_vals[0] if isinstance(group_vals, tuple) else group_vals
        
        layer_means = group_df.groupby('layer')['score'].mean()
        best_layer = layer_means.idxmax()
        
        best_layer_df = group_df[group_df['layer'] == best_layer]
        scores = best_layer_df['score'].tolist()
        
        result[group_vals] = (scores, best_layer)
    
    return result


def plot_brain_score_barplot(scores_by_arch_class, pca_classes, architectures, 
                             region_name, out_png, enable_significance=True, ylabel="Brain Similarity (RSA)"):
    """
    Grouped bar-plot for alignment scores across architectures and class counts.
    
    Parameters
    ----------
    scores_by_arch_class : dict[tuple, list[float]]
        Keys = (architecture, n_classes) tuples, e.g., ('alexnet', 2), ('1K', None).
        Values = list of scores (multiple subjects or single score).
    pca_classes : list[int]
        Ordered list of PCA class counts, e.g., [2, 4, 8, ..., 64].
    architectures : list[str]
        List of architectures to plot (e.g., ['alexnet', 'dino', 'clip']).
    region_name : str
        Brain region or analysis name to display as title.
    out_png : str
        Destination PNG filename.
    enable_significance : bool
        If True, perform paired t-tests vs 1K baseline (requires multiple scores).
    ylabel : str
        Y-axis label (default: "Brain Similarity (RSA)").
    """
    color_map = {
        'alexnet': '#1f77b4',
        'dino': '#ff7f0e',
        'clip': '#2d7f2d',
        'dreamsim': '#9467bd',
    }
    k1k_color = '#666666'
    
    sns.set_theme(style='ticks', context='paper', font_scale=1.2)
    fig, ax = plt.subplots(figsize=(16, 6))
    
    n_archs = len(architectures)
    bar_width = 0.24
    intra_group_gap = 0.04
    group_gap = 0.30
    
    scores_1k = scores_by_arch_class.get(('1K', None), None)
    
    # Plot grouped bars with optional significance tests
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
                
                # Significance test (only if enabled and sufficient data)
                if enable_significance and scores_1k is not None and len(scores) == len(scores_1k) and len(scores) > 1:
                    t_stat, p_val = stats.ttest_rel(scores, scores_1k)
                    if p_val < 0.01:
                        ax.text(bar_pos + bar_width/2, 0.015, '*',
                               ha='center', va='bottom', fontsize=18, fontweight='bold', color='white')
    
    # Plot 1K baseline as horizontal line
    if ('1K', None) in scores_by_arch_class:
        scores_1k = scores_by_arch_class[('1K', None)]
        mean_1k = np.mean(scores_1k)
        ax.axhline(y=mean_1k, color=k1k_color, linestyle='--', linewidth=2.5, 
                   label='ImageNet-1K', zorder=2, alpha=0.9)
    
    # X-axis formatting
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
    
    # Y-axis formatting
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
    ax.set_ylabel(ylabel, fontsize=15, labelpad=12, fontweight='normal')
    
    ax.set_title(region_name.title(), fontsize=18, fontweight='bold', pad=15)
    
    # Legend
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
    
    plt.tight_layout(pad=1.2, rect=[0, 0, 0.85, 1])
    plt.savefig(out_png, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Plot saved → {out_png}")

