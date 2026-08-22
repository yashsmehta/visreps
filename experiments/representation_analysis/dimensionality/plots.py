"""
Visualization for dimensionality analysis: coarse-grain vs fine-grain.
"""

import numpy as np
import matplotlib.pyplot as plt


def _setup_ax(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_facecolor('#FAFAFA')


def plot_metric_across_models(metric_dict, layers, model_names, colors,
                              ylabel, title, output_path, log_y=False):
    """Line plot of a single scalar metric across layers, one line per model.

    metric_dict: {model_name -> {layer -> value}}
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(layers))

    for name, color in zip(model_names, colors):
        vals = np.array([metric_dict[name][l] for l in layers], dtype=float)
        ax.plot(x, vals, 'o-', linewidth=2.2, markersize=7,
                color=color, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    if log_y:
        ax.set_yscale('log')
    ax.legend(fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3)
    _setup_ax(ax, 'Layer', ylabel, title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_eigenspectrum_loglog(eigs_dict, alpha_dict, layers_to_plot,
                              model_names, colors, output_path,
                              n_components=None, y_floor=1e-8):
    """Log-log eigenspectrum per layer with fitted power-law slopes annotated.

    eigs_dict:  {model_name -> {layer -> eigenvalues (descending)}}
    alpha_dict: {model_name -> {layer -> {'alpha':..,'rank_min':..,'rank_max':..}}}
    y_floor:    Lower clip on lambda_n / lambda_1 (mask values below).
    """
    n_plots = len(layers_to_plot)
    ncols = min(n_plots, 4)
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.2 * nrows),
                             squeeze=False)
    axes = axes.flatten()

    for ax, layer in zip(axes, layers_to_plot):
        for name, color in zip(model_names, colors):
            eigs = np.asarray(eigs_dict[name][layer])
            n_total = len(eigs)
            n_plot = n_components if n_components else n_total
            n_plot = min(n_plot, n_total)
            eigs_norm = eigs[:n_plot] / max(eigs[0], 1e-30)
            ranks = np.arange(1, n_plot + 1)
            keep = eigs_norm > y_floor

            info = alpha_dict[name][layer]
            label = f"{name}  α={info['alpha']:.2f}" if not np.isnan(info['alpha']) \
                    else f"{name}  α=–"
            ax.plot(ranks[keep], eigs_norm[keep],
                    linewidth=2.0, color=color, label=label)

            # Overlay the fitted line over its actual fit range
            if not np.isnan(info['alpha']) and info['n_fit'] >= 5:
                rmin, rmax = info['rank_min'], info['rank_max']
                xs = np.array([rmin, rmax], dtype=float)
                # Use intercept in absolute eigenvalue scale; renormalize by lambda_1
                ys = np.exp(info['intercept']) * xs ** (-info['alpha']) / max(eigs[0], 1e-30)
                ax.plot(xs, ys, linestyle='--', linewidth=1.2,
                        color=color, alpha=0.7)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(bottom=y_floor)
        ax.legend(fontsize=8.5, loc='lower left', frameon=True)
        ax.grid(True, which='both', alpha=0.25)
        _setup_ax(ax, 'Rank n (log)', r'$\lambda_n / \lambda_1$ (log)',
                  f'{layer} eigenspectrum')

    for ax in axes[n_plots:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_power_law_exponent(alpha_dict, layers, model_names, colors, output_path):
    """Convenience wrapper kept for callers; same shape as metric plot.

    alpha_dict: {model_name -> {layer -> {'alpha':..}}}
    """
    flat = {n: {l: alpha_dict[n][l]['alpha'] for l in layers}
            for n in model_names}
    plot_metric_across_models(
        flat, layers, model_names, colors,
        ylabel=r'Power-law exponent $\alpha$',
        title=r'Eigenspectrum decay $\lambda_n \propto n^{-\alpha}$',
        output_path=output_path, log_y=False,
    )


def plot_sparsity_across_models(sparsity_results, layers, model_names, colors,
                                output_path):
    """Hoyer sparsity mean ± std across layers, one line per model.

    sparsity_results: {model_name -> {layer -> {'mean': x, 'std': y}}}
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(layers))

    for name, color in zip(model_names, colors):
        means = np.array([sparsity_results[name][l]['mean'] for l in layers])
        stds  = np.array([sparsity_results[name][l]['std']  for l in layers])
        ax.errorbar(x, means, yerr=stds, fmt='o-', linewidth=2.0, markersize=6,
                    color=color, label=name, capsize=3, alpha=0.95)

    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3)
    _setup_ax(ax, 'Layer', 'Hoyer Sparsity (0=dense, 1=sparse)',
              'Activation Sparsity across Layers')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_summary_table(results, layers, model_names):
    """Print a tabular summary for each metric."""
    print("\n" + "=" * 90)
    print("DIMENSIONALITY ANALYSIS SUMMARY")
    print("=" * 90)

    for metric_name, metric_results in results.items():
        print(f"\n{metric_name}:")
        print("-" * 90)

        header = f"{'Layer':<8}"
        for name in model_names:
            header += f" | {name[:16]:<16}"
        print(header)
        print("-" * 90)

        for layer in layers:
            row = f"{layer:<8}"
            for name in model_names:
                val = metric_results[name][layer]
                if isinstance(val, dict):
                    val = val.get('mean', val.get('dimension', val.get('alpha', 0)))
                row += f" | {val:<16.3f}"
            print(row)
