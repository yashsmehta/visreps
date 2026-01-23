"""
Visualization functions for dimensionality analysis.

All functions take pre-computed results and create figures.
"""

import numpy as np
import matplotlib.pyplot as plt


# Style constants
COLORS = {'pretrained': '#1f77b4', 'trained': '#ff7f0e'}
FIGSIZE_WIDE = (14, 5)
FIGSIZE_TALL = (10, 8)


def _setup_ax(ax, xlabel, ylabel, title):
    """Common axis setup."""
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_facecolor('#FAFAFA')


def plot_metric_comparison(results, metric_key, layers, model_names, ylabel, title, output_path):
    """Generic comparison plot for any metric across layers.

    Args:
        results: Dict with model_name -> {layer -> value}
        metric_key: Not used (results already extracted)
        layers: List of layer names
        model_names: List of model names
        ylabel: Y-axis label
        title: Plot title
        output_path: Where to save
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(layers))

    vals = {name: np.array([results[name][layer] for layer in layers])
            for name in model_names}

    # Plot 1: Line plot
    ax = axes[0]
    for i, name in enumerate(model_names):
        color = list(COLORS.values())[i]
        ax.plot(x, vals[name], 'o-', linewidth=2, markersize=8, color=color, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _setup_ax(ax, 'Layer', ylabel, title)

    # Plot 2: Ratio
    ax = axes[1]
    ratio = vals[model_names[0]] / np.maximum(vals[model_names[1]], 1e-10)
    colors = ['#2ecc71' if r > 1 else '#e74c3c' for r in ratio]
    bars = ax.bar(x, ratio, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    for bar, r in zip(bars, ratio):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{r:.2f}x', ha='center', va='bottom', fontsize=9)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    _setup_ax(ax, 'Layer', f'Ratio ({model_names[0][:3]} / {model_names[1][:3]})', 'Compression Ratio')

    # Plot 3: Bar comparison
    ax = axes[2]
    width = 0.35
    for i, name in enumerate(model_names):
        color = list(COLORS.values())[i]
        ax.bar(x + (i - 0.5) * width, vals[name], width, label=name, color=color, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    _setup_ax(ax, 'Layer', ylabel, 'Side-by-Side Comparison')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_eigenspectrum(eigs_dict, layers_to_plot, model_names, output_path, n_components=100):
    """Plot eigenspectrum for selected layers.

    Args:
        eigs_dict: Dict with model_name -> {layer -> eigenvalues}
        layers_to_plot: Which layers to show
        model_names: List of model names
        output_path: Where to save
        n_components: How many components to plot
    """
    n_plots = len(layers_to_plot)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers_to_plot):
        for i, name in enumerate(model_names):
            eigs = eigs_dict[name][layer]
            n_plot = min(n_components, len(eigs))
            color = list(COLORS.values())[i]
            ax.plot(range(1, n_plot + 1), eigs[:n_plot] / eigs[0],
                    linewidth=2, color=color, label=name)

        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        _setup_ax(ax, 'Component', 'Normalized Eigenvalue', f'{layer} Eigenspectrum')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_sparsity_comparison(sparsity_results, layers, model_names, output_path):
    """Plot Hoyer sparsity comparison.

    Args:
        sparsity_results: Dict with model_name -> {layer -> {'mean': x, 'std': y}}
        layers: List of layer names
        model_names: List of model names
        output_path: Where to save
    """
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    x = np.arange(len(layers))

    means = {name: np.array([sparsity_results[name][layer]['mean'] for layer in layers])
             for name in model_names}
    stds = {name: np.array([sparsity_results[name][layer]['std'] for layer in layers])
            for name in model_names}

    # Plot 1: Sparsity trajectory with error bars
    ax = axes[0]
    for i, name in enumerate(model_names):
        color = list(COLORS.values())[i]
        ax.errorbar(x, means[name], yerr=stds[name], fmt='o-', linewidth=2,
                    markersize=8, color=color, label=name, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _setup_ax(ax, 'Layer', 'Hoyer Sparsity', 'Activation Sparsity (0=dense, 1=sparse)')

    # Plot 2: Difference
    ax = axes[1]
    diff = means[model_names[1]] - means[model_names[0]]
    colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in diff]
    bars = ax.bar(x, diff, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    for bar, d in zip(bars, diff):
        va = 'bottom' if d >= 0 else 'top'
        offset = 0.005 if d >= 0 else -0.005
        ax.text(bar.get_x() + bar.get_width()/2, d + offset,
                f'{d:+.3f}', ha='center', va=va, fontsize=9)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    _setup_ax(ax, 'Layer', f'Sparsity Diff ({model_names[1][:3]} - {model_names[0][:3]})',
              'Sparsity Change')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_summary_table(results, layers, model_names):
    """Print a summary table of all metrics.

    Args:
        results: Dict with metric_name -> {model_name -> {layer -> value}}
        layers: List of layer names
        model_names: List of model names
    """
    print("\n" + "=" * 80)
    print("DIMENSIONALITY ANALYSIS SUMMARY")
    print("=" * 80)

    for metric_name, metric_results in results.items():
        print(f"\n{metric_name}:")
        print("-" * 60)

        header = f"{'Layer':<8}"
        for name in model_names:
            header += f" | {name[:15]:<15}"
        header += " | Ratio"
        print(header)
        print("-" * 60)

        for layer in layers:
            row = f"{layer:<8}"
            vals = []
            for name in model_names:
                val = metric_results[name][layer]
                if isinstance(val, dict):
                    val = val.get('mean', val.get('dimension', 0))
                vals.append(val)
                row += f" | {val:<15.2f}"

            if len(vals) == 2 and vals[1] != 0:
                ratio = vals[0] / vals[1]
                row += f" | {ratio:.2f}x"
            print(row)
