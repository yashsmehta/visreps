import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


def process_results(results: List[Dict]) -> Dict:
    """Process results into a format suitable for plotting."""
    processed = {
        'no_pca': {'correlations': None, 'covariances': None},
        'pca': {}
    }
    
    for result in results:
        if not result.get('pca_labels', False):
            processed['no_pca']['correlations'] = np.array(result['mean_correlations'])
            processed['no_pca']['covariances'] = np.array(result['mean_covariances'])
        else:
            n_classes = result.get('pca_n_classes')
            if n_classes:
                processed['pca'][n_classes] = {
                    'correlations': np.array(result['mean_correlations']),
                    'covariances': np.array(result['mean_covariances'])
                }
    
    return processed


def plot_binned_correlations(results: Dict, n_bins=8, save_path="plotters/plssvd_plot.png"):
    """Plot binned correlations for different PCA configurations."""
    # Set up the plot style with a clean, modern look
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("notebook", font_scale=1.2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('white')
    
    # Custom color palette - using distinct shades of blue
    base_color = "#95a5a6"  # Clean gray for normal training
    # Shades of blue from light to dark with more contrast
    pca_colors = [
        "#74b9ff",  # Light blue
        "#0984e3",  # Medium blue
        "#0652DD",  # Deep blue
        "#1B1464"   # Dark navy
    ]
    
    def plot_data(data: np.ndarray, ax, ylabel: str, title: str):
        data = np.array(data)  # Ensure data is numpy array
        n_points = len(data)
        ranks = np.arange(1, n_points + 1)
        
        # Compute log-spaced bin edges
        float_edges = np.logspace(0, np.log10(n_points), n_bins + 1)
        bin_edges = np.unique(np.round(float_edges).astype(int))
        
        if bin_edges[0] < 1:
            bin_edges[0] = 1
        if bin_edges[-1] < n_points:
            bin_edges = np.append(bin_edges, n_points)
        
        if len(bin_edges) - 1 < n_bins:
            bin_edges = np.linspace(1, n_points, n_bins + 1, dtype=int)
        
        binned_values = []
        binned_ranks = []
        
        for i in range(len(bin_edges) - 1):
            start = bin_edges[i] - 1  # Convert to 0-based indexing
            end = bin_edges[i + 1] - 1  # Convert to 0-based indexing
            if i == len(bin_edges) - 2:
                end += 1  # Include the last point
            
            values_in_bin = data[start:end]
            ranks_in_bin = ranks[start:end]
            
            if len(values_in_bin) > 0:
                binned_values.append(np.mean(values_in_bin))
                binned_ranks.append(np.mean(ranks_in_bin))
        
        return ranks, data, np.array(binned_ranks), np.array(binned_values)
    
    # Plot no PCA results
    if results['no_pca']['correlations'] is not None:
        for ax, data, ylabel, title in [
            (ax1, results['no_pca']['correlations'], 'Correlation', 'Cross-Decomposition Correlations'),
            (ax2, results['no_pca']['covariances'], 'Covariance', 'Cross-Decomposition Covariances')
        ]:
            ranks, raw, binned_ranks, binned_values = plot_data(data, ax, ylabel, title)
            # Plot raw data with very low alpha
            ax.loglog(ranks, raw, color=base_color, alpha=0.05, linewidth=0.5)
            # Plot binned with solid line
            ax.loglog(binned_ranks, binned_values, color=base_color, linewidth=2.5, 
                     marker='o', markersize=6, label='Normal Training',
                     markeredgewidth=1, markeredgecolor='white')
    
    # Plot PCA results with different colors
    for (n_classes, data), color in zip(sorted(results['pca'].items()), pca_colors):
        for ax, values, ylabel, title in [
            (ax1, data['correlations'], 'Correlation', 'Cross-Decomposition Correlations'),
            (ax2, data['covariances'], 'Covariance', 'Cross-Decomposition Covariances')
        ]:
            ranks, raw, binned_ranks, binned_values = plot_data(values, ax, ylabel, title)
            # Plot raw data with very low alpha
            ax.loglog(ranks, raw, color=color, alpha=0.05, linewidth=0.5)
            # Plot binned with solid line
            ax.loglog(binned_ranks, binned_values, color=color, linewidth=2.5, 
                     marker='o', markersize=6, label=f'PCA {n_classes} classes',
                     markeredgewidth=1, markeredgecolor='white')
    
    # Customize plots with a cleaner look
    for ax, ylabel, title in [
        (ax1, 'Correlation', 'Cross-Decomposition Correlations'),
        (ax2, 'Covariance', 'Cross-Decomposition Covariances')
    ]:
        # Set background color
        ax.set_facecolor('#f8f9fa')
        
        # Customize grid
        ax.grid(True, which="major", ls="-", alpha=0.1, color='#2c3e50')
        ax.grid(True, which="minor", ls=":", alpha=0.05, color='#2c3e50')
        
        # Customize spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#2c3e50')
        ax.spines['bottom'].set_color('#2c3e50')
        
        # Customize labels
        ax.set_xlabel('Rank', fontsize=12, color='#2c3e50', fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, color='#2c3e50', fontweight='bold')
        ax.tick_params(colors='#2c3e50', which='both')
        
        # Move legend outside and style it
        legend = ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left',
                          frameon=True, fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_linewidth(0)
        
        # Set title with more emphasis
        ax.set_title(title, fontsize=14, pad=15, color='#2c3e50', fontweight='bold')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make room for legend
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


if __name__ == "__main__":
    results_file = "plotters/plssvd_results.json"
    with open(results_file, 'r') as f:
        results = json.load(f)
    processed_results = process_results(results)
    plot_binned_correlations(processed_results)
