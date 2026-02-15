"""
Dimensionality analysis: Compare pretrained vs fine-tuned AlexNet.

Computes:
- Participation Ratio (effective dimensionality from eigenspectrum)
- Two-NN Intrinsic Dimension (manifold dimensionality)
- Hoyer Sparsity (activation sparsity)
"""

import os
import sys
import numpy as np
from tqdm import tqdm

# Add parent to path for utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    load_data_and_models_all_layers, ensure_output_dir,
    MODEL_NAMES, ALL_LAYERS, OUTPUT_DIR
)

from metrics import (
    participation_ratio, eigenspectrum, n_components_for_variance,
    two_nn_dimension, hoyer_sparsity, fraction_active
)
from plots import (
    plot_metric_comparison, plot_eigenspectrum,
    plot_sparsity_comparison, plot_summary_table
)


def compute_all_metrics(feats_dict, layers, n_samples_twonn=2000):
    """Compute all dimensionality metrics for a model.

    Args:
        feats_dict: Dict of layer -> features array
        layers: List of layer names to analyze
        n_samples_twonn: Subsample size for Two-NN (for speed)

    Returns:
        Dict with metric results for each layer
    """
    results = {
        'pr': {},           # Participation ratio
        'n90': {},          # Components for 90% variance
        'twonn': {},        # Two-NN intrinsic dimension
        'sparsity': {},     # Hoyer sparsity stats
        'eigenvalues': {},  # Raw eigenvalues for plotting
    }

    for layer in tqdm(layers, desc="  Computing metrics"):
        X = feats_dict[layer]

        # Participation ratio
        results['pr'][layer] = participation_ratio(X)

        # Components for 90% variance
        results['n90'][layer] = n_components_for_variance(X, threshold=0.9)

        # Two-NN intrinsic dimension
        dim, std = two_nn_dimension(X, n_samples=n_samples_twonn)
        results['twonn'][layer] = {'dimension': dim, 'std': std}

        # Hoyer sparsity
        sparsity_vals = hoyer_sparsity(X)
        frac_active_vals = fraction_active(X)
        results['sparsity'][layer] = {
            'mean': np.mean(sparsity_vals),
            'std': np.std(sparsity_vals),
            'frac_active': np.mean(frac_active_vals)
        }

        # Eigenvalues (for spectrum plots)
        results['eigenvalues'][layer] = eigenspectrum(X)

    return results


def main():
    ensure_output_dir()
    output_dir = os.path.join(OUTPUT_DIR, "dimensionality")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Dimensionality Analysis")
    print("=" * 60)

    # Load data
    (feats_pretrained, feats_32way,
     pca_labels, sem_labels, synsets, img_paths, loader) = load_data_and_models_all_layers()

    layers = ALL_LAYERS
    model_names = MODEL_NAMES

    # Compute metrics for both models
    print(f"\nAnalyzing {model_names[0]}...")
    results_pre = compute_all_metrics(feats_pretrained, layers)

    print(f"\nAnalyzing {model_names[1]}...")
    results_32way = compute_all_metrics(feats_32way, layers)

    # Organize results by metric
    all_results = {
        'Participation Ratio': {
            model_names[0]: results_pre['pr'],
            model_names[1]: results_32way['pr']
        },
        'Two-NN Dimension': {
            model_names[0]: {l: results_pre['twonn'][l]['dimension'] for l in layers},
            model_names[1]: {l: results_32way['twonn'][l]['dimension'] for l in layers}
        },
        'Components (90% var)': {
            model_names[0]: results_pre['n90'],
            model_names[1]: results_32way['n90']
        }
    }

    # Print summary
    plot_summary_table(all_results, layers, model_names)

    # Generate plots
    print("\nGenerating plots...")

    # 1. Participation Ratio
    plot_metric_comparison(
        all_results['Participation Ratio'], 'pr', layers, model_names,
        'Participation Ratio', 'Effective Dimensionality (PR)',
        os.path.join(output_dir, "participation_ratio.png")
    )
    print(f"  Saved: participation_ratio.png")

    # 2. Two-NN Intrinsic Dimension
    plot_metric_comparison(
        all_results['Two-NN Dimension'], 'twonn', layers, model_names,
        'Intrinsic Dimension', 'Manifold Dimensionality (Two-NN)',
        os.path.join(output_dir, "intrinsic_dimension.png")
    )
    print(f"  Saved: intrinsic_dimension.png")

    # 3. Eigenspectrum
    eigs_dict = {
        model_names[0]: results_pre['eigenvalues'],
        model_names[1]: results_32way['eigenvalues']
    }
    plot_eigenspectrum(
        eigs_dict, ['conv2', 'conv5', 'fc2'], model_names,
        os.path.join(output_dir, "eigenspectrum.png")
    )
    print(f"  Saved: eigenspectrum.png")

    # 4. Sparsity
    sparsity_results = {
        model_names[0]: results_pre['sparsity'],
        model_names[1]: results_32way['sparsity']
    }
    plot_sparsity_comparison(
        sparsity_results, layers, model_names,
        os.path.join(output_dir, "sparsity.png")
    )
    print(f"  Saved: sparsity.png")

    print("\n" + "=" * 60)
    print(f"Done! Outputs saved to: {output_dir}")
    print("=" * 60)

    return {
        'pretrained': results_pre,
        '32way': results_32way
    }


if __name__ == "__main__":
    main()
