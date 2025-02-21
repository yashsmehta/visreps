import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List

def filter_results(results: List[Dict], region: str, layer: str, epoch: int) -> Dict:
    """Filter results for specific region, layer and epoch and average across subjects."""
    filtered = {
        'no_pca': {'correlations': None, 'covariances': None},
        'pca': {}
    }
    
    # First collect all matching results to average across subjects
    no_pca_corrs = []
    no_pca_covs = []
    pca_results = {}
    
    for result in results:
        if (result.get('region') != region or 
            result.get('layer') != layer or 
            result.get('epoch') != str(epoch)):
            continue
            
        if not result.get('pca_labels', False):
            no_pca_corrs.append(np.array(result['mean_correlations']))
            no_pca_covs.append(np.array(result['mean_covariances']))
        else:
            n_classes = result.get('pca_n_classes')
            if n_classes:
                if n_classes not in pca_results:
                    pca_results[n_classes] = {'correlations': [], 'covariances': []}
                pca_results[n_classes]['correlations'].append(np.array(result['mean_correlations']))
                pca_results[n_classes]['covariances'].append(np.array(result['mean_covariances']))
    
    # Average across subjects
    if no_pca_corrs:
        filtered['no_pca']['correlations'] = np.mean(no_pca_corrs, axis=0)
        filtered['no_pca']['covariances'] = np.mean(no_pca_covs, axis=0)
    
    for n_classes, data in pca_results.items():
        filtered['pca'][n_classes] = {
            'correlations': np.mean(data['correlations'], axis=0),
            'covariances': np.mean(data['covariances'], axis=0)
        }
    
    return filtered

def plot_epoch_comparison(results: List[Dict], region: str='early visual stream', 
                         layer: str='conv4', epochs: List[int]=[0, 10], n_bins=8, 
                         save_path="plots/cross_decomposition/plssvd_{region}_{layer}.png"):
    """Plot binned correlations comparing different epochs."""
    save_path = save_path.format(region=region, layer=layer)
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("notebook", font_scale=1.4)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.patch.set_facecolor('white')
    
    # Color schemes with more polished colors
    baseline_colors = {
        0: "#7f8c8d",  # Refined gray for untrained
        10: "#e74c3c"  # Vibrant red for trained
    }
    
    # Blues for PCA variants at epoch 10 - more vibrant palette
    pca_colors = ["#85c1e9", "#3498db", "#2874a6", "#1a5276", "#154360", "#0a2942"]
    pca_classes = [2, 4, 8, 16, 32, 64]
    
    def plot_data(data: np.ndarray, n_bins: int):
        data = np.array(data)
        n_points = len(data)
        ranks = np.arange(1, n_points + 1)
        
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
            start = bin_edges[i] - 1
            end = bin_edges[i + 1] - 1
            if i == len(bin_edges) - 2:
                end += 1
            
            values_in_bin = data[start:end]
            ranks_in_bin = ranks[start:end]
            
            if len(values_in_bin) > 0:
                binned_values.append(np.mean(values_in_bin))
                binned_ranks.append(np.mean(ranks_in_bin))
        
        return ranks, data, np.array(binned_ranks), np.array(binned_values)
    
    # Plot baseline epochs
    labels = {0: "Untrained", 10: "Trained"}
    for epoch in [0, 10]:
        filtered_results = filter_results(results, region, layer, epoch)
        
        if filtered_results['no_pca']['correlations'] is not None:
            for ax_idx, (data, ylabel) in enumerate([
                (filtered_results['no_pca']['correlations'], 'Correlation'),
                (filtered_results['no_pca']['covariances'], 'Covariance')
            ]):
                ax = axes[ax_idx]
                ranks, raw, binned_ranks, binned_values = plot_data(data, n_bins)
                ax.loglog(ranks, raw, color=baseline_colors[epoch], alpha=0.05, linewidth=0.5)
                ax.loglog(binned_ranks, binned_values, color=baseline_colors[epoch], 
                         linewidth=3.0, marker='o', markersize=8, 
                         label=labels[epoch],
                         markeredgewidth=1.5, markeredgecolor='white')
    
    # Plot PCA results for epoch 10
    filtered_results = filter_results(results, region, layer, 10)
    for (n_classes, color) in zip(pca_classes, pca_colors):
        if n_classes in filtered_results['pca']:
            data = filtered_results['pca'][n_classes]
            for ax_idx, (values, ylabel) in enumerate([
                (data['correlations'], 'Correlation'),
                (data['covariances'], 'Covariance')
            ]):
                ax = axes[ax_idx]
                ranks, raw, binned_ranks, binned_values = plot_data(values, n_bins)
                ax.loglog(ranks, raw, color=color, alpha=0.05, linewidth=0.5)
                ax.loglog(binned_ranks, binned_values, color=color, linewidth=2.5,
                         marker='o', markersize=7, 
                         label=f'PCA {n_classes}',
                         markeredgewidth=1.5, markeredgecolor='white')
    
    # Customize plots
    for ax_idx, (ax, title) in enumerate(zip(axes, ['Correlation', 'Covariance'])):
        ax.set_facecolor('#ffffff')
        ax.grid(True, which="major", ls="-", alpha=0.15, color='#2c3e50')
        ax.grid(True, which="minor", ls=":", alpha=0.1, color='#2c3e50')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#2c3e50')
        ax.spines['bottom'].set_color('#2c3e50')
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        ax.set_xlabel('Rank', fontsize=14, color='#2c3e50', fontweight='bold')
        ax.set_ylabel(title, fontsize=14, color='#2c3e50', fontweight='bold')
        ax.set_title(title, fontsize=16, pad=15, color='#2c3e50', fontweight='bold')
        ax.tick_params(colors='#2c3e50', which='both', width=1.5, length=6)
        ax.tick_params(which='minor', width=1, length=4)
        
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Set y-axis limit for correlation plot
        if ax_idx == 0:  # Correlation plot
            ax.set_ylim(bottom=1e-3)
        
        if ax_idx == 1:  # Only show legend for the second plot
            legend = ax.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left',
                             frameon=True, fancybox=True, shadow=True)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.9)
            legend.get_frame().set_linewidth(0)
            legend.get_frame().set_edgecolor('#2c3e50')
    
    # Add super title
    plt.suptitle(f'Cross-Decomposition Analysis: {region} ({layer})', 
                 fontsize=18, color='#2c3e50', fontweight='bold', y=1.02)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

if __name__ == "__main__":
    results_file = "logs/eval/cross_decomposition/plssvd_results.pkl"
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    regions = ['early visual stream', 'midventral visual stream', 'ventral visual stream']
    layers = ['conv4', 'fc2']
    for region in regions:
        for layer in layers:
            plot_epoch_comparison(results, region=region, layer=layer, epochs=[0, 10]) 
