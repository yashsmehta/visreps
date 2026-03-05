"""Analyze what semantic categories are enriched at each pole of a principal component.

Usage:
    python experiments/pca_visualization/pc_semantic_analysis.py --model dino --pc 1 --level 6
"""
import os
import sys
import argparse
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiments.wordnet.wordnet_utils import setup as setup_wordnet
from nltk.corpus import wordnet as wn

HISTOGRAM_DIR = "experiments/pca_visualization/pc_histogram"


def load_data(model_name, dataset="imagenet"):
    """Load features and eigenvectors, return PC-projected scores and image names."""
    eigenvectors_path = f"datasets/obj_cls/imagenet/eigenvectors_{model_name}.npz"
    pca_data = np.load(eigenvectors_path)

    for pattern in [f"features_{model_name}.npz", f"features_{model_name}_features.npz"]:
        features_path = f"datasets/obj_cls/{dataset}/{pattern}"
        if os.path.exists(features_path):
            break

    features_data = np.load(features_path, allow_pickle=True)
    features_key = [k for k in features_data if 'features' in k and k != 'image_names'][0]

    names = features_data['image_names']
    if names.size > 0 and isinstance(names[0], (bytes, np.bytes_)):
        names = np.array([n.decode('utf-8') for n in names])

    features = features_data[features_key].reshape(len(names), -1)
    return features, pca_data['eigenvectors'], pca_data['mean'], names


def get_ancestor_at_level(filename, level):
    """Get WordNet ancestor name at a given hierarchy level for an ImageNet image."""
    wnid = os.path.basename(filename).split('_')[0]
    try:
        synset = wn.synset_from_pos_and_offset('n', int(wnid[1:]))
    except Exception:
        return "unknown"
    paths = synset.hypernym_paths()
    if not paths or level >= len(paths[0]):
        return synset.name().split('.')[0]
    return paths[0][level].name().split('.')[0]


def analyze_pc(scores, ancestors, percentile=20):
    """Compare category enrichment at PC poles vs baseline."""
    low_thresh = np.percentile(scores, percentile)
    high_thresh = np.percentile(scores, 100 - percentile)

    low_mask = scores <= low_thresh
    high_mask = scores >= high_thresh

    baseline_counts = Counter(ancestors)
    n_total = len(ancestors)

    results = {}
    for pole_name, mask in [('low', low_mask), ('high', high_mask)]:
        pole_ancestors = [ancestors[i] for i in range(len(ancestors)) if mask[i]]
        pole_counts = Counter(pole_ancestors)
        n_pole = len(pole_ancestors)
        min_count = max(1, int(n_pole * 0.005))

        enriched = []
        for cat, count in pole_counts.items():
            if count < min_count:
                continue
            pole_pct = count / n_pole * 100
            base_pct = baseline_counts[cat] / n_total * 100
            enrichment = pole_pct - base_pct
            if enrichment > 0:
                enriched.append({
                    'category': cat, 'count': count,
                    'pole_pct': pole_pct, 'base_pct': base_pct,
                    'enrichment': enrichment,
                })
        enriched.sort(key=lambda x: x['enrichment'], reverse=True)
        results[pole_name] = {'enriched': enriched, 'n': n_pole}

    return results


def print_results(results, model, pc):
    """Print enriched categories for each pole."""
    print(f"\n{'='*60}")
    print(f"PC{pc} Semantic Analysis ({model.upper()})")
    print(f"{'='*60}")

    for pole_name in ['low', 'high']:
        pole = results[pole_name]
        print(f"\n--- {pole_name.upper()} POLE (n={pole['n']:,}) ---")
        print(f"{'Category':<25} {'Count':>6} {'Pole%':>7} {'Base%':>7} {'Enrich':>8}")
        print("-" * 55)
        for r in pole['enriched']:
            print(f"{r['category']:<25} {r['count']:>6} {r['pole_pct']:>6.1f}% "
                  f"{r['base_pct']:>6.1f}% {r['enrichment']:>+7.1f}%")


def plot_histogram(scores, ancestors, results, model, pc):
    """Overlapping histograms for top 3 enriched categories from each pole."""
    top_cats = {
        'low': [r['category'] for r in results['low']['enriched'][:3]],
        'high': [r['category'] for r in results['high']['enriched'][:3]],
    }
    colors = {
        'low': ['#1f77b4', '#6baed6', '#9ecae1'],
        'high': ['#d62728', '#fc8d62', '#fdae6b'],
    }

    plt.figure(figsize=(12, 6))
    for pole in ['low', 'high']:
        for i, cat in enumerate(top_cats[pole]):
            cat_scores = [scores[j] for j, a in enumerate(ancestors) if a == cat]
            if cat_scores:
                plt.hist(cat_scores, bins=50, alpha=0.5, density=True,
                         label=f"{cat} ({pole})", color=colors[pole][i])

    plt.xlabel(f'PC{pc} Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'PC{pc} Distribution by Category ({model.upper()})', fontsize=14)
    plt.legend(loc='upper right')
    plt.tight_layout()

    os.makedirs(HISTOGRAM_DIR, exist_ok=True)
    output_path = os.path.join(HISTOGRAM_DIR, f"pc{pc}_histogram_{model}.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved histogram to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze PC semantics via WordNet")
    parser.add_argument('--model', type=str, default='alexnet',
                        choices=['alexnet', 'vit', 'clip', 'dino'])
    parser.add_argument('--dataset', type=str, default='imagenet-mini-50',
                        choices=['imagenet', 'imagenet-mini-50'])
    parser.add_argument('--pc', type=int, default=1, help='PC to analyze (1-indexed)')
    parser.add_argument('--level', type=int, default=6, help='WordNet hierarchy level (0=root)')
    parser.add_argument('--percentile', type=int, default=20)
    args = parser.parse_args()

    setup_wordnet()

    features, eigenvectors, mean, image_names = load_data(args.model, args.dataset)
    print(f"Loaded {len(image_names):,} images")

    scores = ((features - mean) @ eigenvectors[:, args.pc - 1]).flatten()

    # Compute ancestors once for all images
    ancestors = [get_ancestor_at_level(name, args.level) for name in image_names]

    results = analyze_pc(scores, ancestors, args.percentile)
    print_results(results, args.model, args.pc)
    plot_histogram(scores, ancestors, results, args.model, args.pc)


if __name__ == '__main__':
    main()
