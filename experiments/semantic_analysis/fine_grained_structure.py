"""
Fine-Grained Structure Analysis: UMAP visualization within animals.

Shows whether fine-grained synset structure is preserved in the representation space.
"""

import os
import warnings
warnings.filterwarnings("ignore", message="n_jobs value.*overridden to 1 by setting random_state")

import numpy as np
import matplotlib.pyplot as plt
import umap

from utils import (
    load_data_and_models, ensure_output_dir,
    MODEL_NAMES, OUTPUT_DIR, SEED
)


def analyze_fine_grained_structure(feats_list, sem_labels, synsets, output_path):
    """UMAP within animals, colored by fine-grained synset.

    Focuses on animal images (semantic label 0) to see if fine-grained
    distinctions (e.g., dog breeds) are preserved in representation space.

    Args:
        feats_list: List of feature arrays for each model
        sem_labels: Array of semantic category labels (0=animals)
        synsets: Array of synset IDs
        output_path: Where to save the figure

    Returns:
        n_animals: Number of animal images analyzed
    """
    # Filter to animals only (semantic label 0)
    animal_mask = sem_labels == 0
    n_animals = animal_mask.sum()
    print(f"  Animals: {n_animals} images")

    if n_animals < 50:
        print("  Not enough animal images for meaningful UMAP visualization")
        return n_animals

    # Get unique synsets within animals
    animal_synsets = synsets[animal_mask]
    unique_synsets, counts = np.unique(animal_synsets, return_counts=True)

    # Keep top 15 most common synsets for coloring
    top_synsets = unique_synsets[np.argsort(counts)[::-1][:15]]
    synset_to_idx = {s: i for i, s in enumerate(top_synsets)}

    # Assign colors
    cmap = plt.cm.tab20(np.linspace(0, 1, 20))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, feats, name in zip(axes, feats_list, MODEL_NAMES):
        feats_animal = feats[animal_mask]

        # L2 normalize
        norms = np.linalg.norm(feats_animal, axis=1, keepdims=True)
        feats_animal = feats_animal / np.maximum(norms, 1e-8)

        # UMAP
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='cosine',
                           random_state=SEED, verbose=False)
        coords = reducer.fit_transform(feats_animal.astype(np.float32))

        # Plot top synsets
        for synset in top_synsets:
            mask = animal_synsets == synset
            color = cmap[synset_to_idx[synset]]
            ax.scatter(coords[mask, 0], coords[mask, 1],
                      c=[color], alpha=0.6, s=15, label=synset[:10])

        # Plot remaining as gray
        other_mask = ~np.isin(animal_synsets, top_synsets)
        ax.scatter(coords[other_mask, 0], coords[other_mask, 1],
                  c='lightgray', alpha=0.3, s=5, label='other')

        ax.set_xlabel('UMAP 1', fontsize=11)
        ax.set_ylabel('UMAP 2', fontsize=11)
        ax.set_title(f'{name}', fontsize=13, fontweight='bold')
        ax.set_facecolor('#FAFAFA')

    # Shared legend
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles[:15], labels_legend[:15], loc='center right',
               bbox_to_anchor=(1.12, 0.5), fontsize=8, title='Synset (Animal)')

    plt.suptitle('Fine-Grained Structure Within Animals (top 15 synsets)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

    return n_animals


def main():
    np.random.seed(SEED)
    ensure_output_dir()

    print("=" * 60)
    print("Fine-Grained Structure Analysis (Animals)")
    print("=" * 60)

    feats_list, _, sem_labels, synsets, _, _ = load_data_and_models()

    output_path = os.path.join(OUTPUT_DIR, "fine_grained_animals.png")
    n_animals = analyze_fine_grained_structure(feats_list, sem_labels, synsets, output_path)

    print("\nDone!")
    return n_animals


if __name__ == "__main__":
    main()
