"""
Generate dimensionality figures from the cached results.npz.

Usage:
    python plot.py                 # sample-normalized view (default)
    python plot.py --raw           # raw (unnormalized) view -- shows outlier
                                   # domination diagnostic
    python plot.py --refit-alpha   # refit power-law alpha from cached eigenvalues
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import OUTPUT_DIR
from metrics import power_law_exponent
from plots import (
    plot_metric_across_models, plot_eigenspectrum_loglog,
    plot_sparsity_across_models, plot_summary_table,
)

EIGENSPECTRUM_LAYERS = ['conv2', 'conv3', 'conv5', 'fc2']


def load_cache(cache_path):
    d = np.load(cache_path, allow_pickle=True)
    model_names = [str(x) for x in d['model_names']]
    layers = [str(x) for x in d['layers']]
    colors = [str(x) for x in d['colors']] if 'colors' in d.files \
             else ["#9ecae1", "#4292c6", "#08519c", "#ff7f0e"]
    all_results = d['all_results'][0]
    return model_names, layers, colors, all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", action="store_true",
                        help="Plot raw (unnormalized) view (diagnostic).")
    parser.add_argument("--refit-alpha", action="store_true",
                        help="Refit power-law alpha from cached eigenvalues.")
    args = parser.parse_args()

    out = os.path.join(OUTPUT_DIR, "dimensionality")
    cache_path = os.path.join(out, "results.npz")
    if not os.path.exists(cache_path):
        sys.exit(f"Cache not found: {cache_path}\nRun `python run.py` first.")

    model_names, layers, colors, all_results = load_cache(cache_path)

    # Key suffix for selecting raw vs sample-normalized (the default).
    suffix = "_raw" if args.raw else ""
    tag    = " (raw)" if args.raw else " (sample-normalized)"
    file_tag = "_raw" if args.raw else ""

    pr_key, n90_key, alpha_key, eig_key = (
        f"pr{suffix}", f"n90{suffix}", f"alpha{suffix}", f"eigenvalues{suffix}",
    )

    # Tolerate old caches that only have the default keys.
    has_keys = all(k in all_results[model_names[0]]
                   for k in [pr_key, n90_key, alpha_key, eig_key])
    if not has_keys:
        sys.exit(f"Cache missing keys for `{suffix or 'default'}` view.\n"
                 f"Re-run `python run.py` to rebuild the cache.")

    if args.refit_alpha:
        for n in model_names:
            for l in layers:
                eigs = all_results[n][eig_key][l]
                all_results[n][alpha_key][l] = power_law_exponent(eigs)

    summary = {
        'Participation Ratio': {n: all_results[n][pr_key] for n in model_names},
        'Two-NN Dimension':    {n: {l: all_results[n]['twonn'][l]['dimension']
                                    for l in layers} for n in model_names},
        'Components (90% var)': {n: all_results[n][n90_key] for n in model_names},
        'Power-law alpha':     {n: {l: all_results[n][alpha_key][l]['alpha']
                                    for l in layers} for n in model_names},
    }
    print(f"\n=== View: {tag.strip()} ===")
    plot_summary_table(summary, layers, model_names)

    plot_metric_across_models(
        summary['Participation Ratio'], layers, model_names, colors,
        ylabel='Participation Ratio',
        title=f'Effective Dimensionality across Layers{tag}',
        output_path=os.path.join(out, f"participation_ratio{file_tag}.png"),
        log_y=True,
    )
    plot_metric_across_models(
        summary['Two-NN Dimension'], layers, model_names, colors,
        ylabel='Two-NN Intrinsic Dimension',
        title='Manifold Dimensionality across Layers',
        output_path=os.path.join(out, "intrinsic_dimension.png"),
        log_y=False,
    )
    plot_metric_across_models(
        summary['Power-law alpha'], layers, model_names, colors,
        ylabel=r'Power-law exponent $\alpha$  ($\lambda_n \propto n^{-\alpha}$)',
        title=f'Eigenspectrum Decay Rate across Layers{tag}',
        output_path=os.path.join(out, f"power_law_alpha{file_tag}.png"),
        log_y=False,
    )
    plot_eigenspectrum_loglog(
        {n: all_results[n][eig_key]   for n in model_names},
        {n: all_results[n][alpha_key] for n in model_names},
        layers_to_plot=EIGENSPECTRUM_LAYERS,
        model_names=model_names, colors=colors,
        output_path=os.path.join(out, f"eigenspectrum_loglog{file_tag}.png"),
    )
    plot_sparsity_across_models(
        {n: all_results[n]['sparsity'] for n in model_names},
        layers, model_names, colors,
        output_path=os.path.join(out, "sparsity.png"),
    )

    print(f"\n  Figures written to: {out}")


if __name__ == "__main__":
    main()
