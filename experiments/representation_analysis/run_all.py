"""
Run All Representation Analyses.

This script runs all analysis modules in sequence:
1. Dimensionality (eigenspectrum, participation ratio) - across ALL layers
2. RSM comparison (visual heatmap + correlation) - FC2 only
3. Variance ratio (within/between class) - FC2 only
4. Nearest neighbor retrieval - FC2 only
5. Fine-grained structure preservation (UMAP within animals) - FC2 only

Each analysis can also be run independently from its own file.
"""

import os
import numpy as np

from utils import (
    load_data_and_models, load_data_and_models_all_layers,
    ensure_output_dir, OUTPUT_DIR, SEED, ALL_LAYERS
)


def main():
    np.random.seed(SEED)
    ensure_output_dir()

    print("=" * 60)
    print("Running All Representation Analyses")
    print("=" * 60)

    # =========================================================================
    # 1. Dimensionality Analysis (all layers)
    # =========================================================================
    print("\n" + "=" * 60)
    print("1. Dimensionality Analysis (All Layers)")
    print("=" * 60)

    from dimensionality_analysis import (
        analyze_dimensionality_all_layers,
        analyze_dimensionality_single_layer
    )

    # Load data with all layers
    (feats_dict_pretrained, feats_dict_4way,
     pca_labels, sem_labels, synsets, img_paths, loader) = load_data_and_models_all_layers()

    # Multi-layer analysis
    analyze_dimensionality_all_layers(
        feats_dict_pretrained, feats_dict_4way,
        os.path.join(OUTPUT_DIR, "dimensionality_all_layers.png")
    )

    # Single-layer FC2 analysis for backward compatibility
    feats_list = [feats_dict_pretrained['fc2'], feats_dict_4way['fc2']]
    analyze_dimensionality_single_layer(
        feats_list,
        os.path.join(OUTPUT_DIR, "dimensionality.png"),
        layer_name="fc2"
    )

    # =========================================================================
    # Remaining analyses use FC2 features only
    # =========================================================================

    # Import analysis functions
    from rsm_comparison import analyze_rsm
    from variance_ratio import analyze_variance_ratio
    from nearest_neighbors import analyze_nearest_neighbors
    from fine_grained_structure import analyze_fine_grained_structure

    # 2. RSM Comparison
    print("\n" + "=" * 60)
    print("2. RSM Comparison (FC2)")
    print("=" * 60)
    analyze_rsm(feats_list, pca_labels, os.path.join(OUTPUT_DIR, "rsm_comparison.png"))

    # 3. Variance Ratio
    print("\n" + "=" * 60)
    print("3. Variance Ratio (FC2)")
    print("=" * 60)
    analyze_variance_ratio(feats_list, pca_labels, os.path.join(OUTPUT_DIR, "variance_ratio.png"))

    # 4. Nearest Neighbor Retrieval
    print("\n" + "=" * 60)
    print("4. Nearest Neighbor Retrieval (FC2)")
    print("=" * 60)
    analyze_nearest_neighbors(feats_list, pca_labels, synsets, img_paths,
                             os.path.join(OUTPUT_DIR, "nearest_neighbors.png"))

    # 5. Fine-grained Structure (Animals)
    print("\n" + "=" * 60)
    print("5. Fine-grained Structure - Animals (FC2)")
    print("=" * 60)
    analyze_fine_grained_structure(feats_list, sem_labels, synsets,
                                   os.path.join(OUTPUT_DIR, "fine_grained_animals.png"))

    print("\n" + "=" * 60)
    print("All analyses complete!")
    print("=" * 60)
    print(f"\nOutput files saved to: {OUTPUT_DIR}/")
    print("  - dimensionality_all_layers.png (NEW: all layers)")
    print("  - dimensionality.png (FC2 only)")
    print("  - rsm_comparison.png")
    print("  - variance_ratio.png")
    print("  - nearest_neighbors.png")
    print("  - fine_grained_animals.png")


if __name__ == "__main__":
    main()
