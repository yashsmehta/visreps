"""
Analyze what semantic categories are enriched at each pole of a principal component.

Usage:
    python experiments/wordnet/pc_semantic_analysis.py --model dino --pc 1 --level 6
"""
import os
import sys
import argparse
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiments.wordnet.wordnet import setup as setup_wordnet
from nltk.corpus import wordnet as wn


def load_data(model_name, dataset):
    """Load features and eigenvectors."""
    eigenvectors_path = f"datasets/obj_cls/imagenet/eigenvectors_{model_name}.npz"
    pca_data = np.load(eigenvectors_path)

    for pattern in [f"features_{model_name}.npz", f"features_{model_name}_features.npz"]:
        features_path = f"datasets/obj_cls/{dataset}/{pattern}"
        if os.path.exists(features_path):
            break

    print(f"Loading features from {features_path}")
    features_data = np.load(features_path, allow_pickle=True)
    features_key = [k for k in features_data.keys() if 'features' in k and k != 'image_names'][0]

    names = features_data['image_names']
    if names.size > 0 and isinstance(names[0], (bytes, np.bytes_)):
        names = np.array([n.decode('utf-8') for n in names])

    features = features_data[features_key].reshape(len(names), -1)
    return features, pca_data['eigenvectors'], pca_data['mean'], names


def get_synset_from_filename(filename):
    """Extract synset from ImageNet filename."""
    wnid = os.path.basename(filename).split('_')[0]
    try:
        return wn.synset_from_pos_and_offset('n', int(wnid[1:]))
    except:
        return None


def get_ancestor_at_level(synset, level):
    """Get ancestor at a specific level (0=root)."""
    if synset is None:
        return None
    paths = synset.hypernym_paths()
    if not paths or level >= len(paths[0]):
        return synset
    return paths[0][level]


def get_ancestors_at_level(image_names, level):
    """Get ancestors at a specific level for all images."""
    ancestors = []
    for name in image_names:
        synset = get_synset_from_filename(name)
        ancestor = get_ancestor_at_level(synset, level)
        ancestors.append(ancestor.name() if ancestor else "unknown")
    return ancestors


def compute_enrichment_vs_baseline(pole_ancestors, baseline_counts, n_baseline, min_count_threshold):
    """Compute enrichment of each category vs baseline, with minimum count filter."""
    pole_counts = Counter(pole_ancestors)
    n_pole = len(pole_ancestors)

    results = []
    for cat, count in pole_counts.items():
        # Filter: must have at least 0.1% of the pole
        if count < min_count_threshold:
            continue

        pole_pct = count / n_pole * 100
        baseline_pct = baseline_counts.get(cat, 0) / n_baseline * 100
        enrichment = pole_pct - baseline_pct

        results.append({
            'category': cat.split('.')[0],
            'count': count,
            'pole_pct': pole_pct,
            'baseline_pct': baseline_pct,
            'enrichment': enrichment,
        })

    return results


def analyze_pc(scores, image_names, level, percentile=20):
    """Analyze PC at a fixed WordNet level, comparing poles to baseline."""
    low_thresh = np.percentile(scores, percentile)
    high_thresh = np.percentile(scores, 100 - percentile)

    low_mask = scores <= low_thresh
    high_mask = scores >= high_thresh

    n_low = sum(low_mask)
    n_high = sum(high_mask)

    # Minimum count threshold: 0.5% of the pole
    min_count_low = max(1, int(n_low * 0.005))
    min_count_high = max(1, int(n_high * 0.005))

    # Get ancestors for all images (baseline), low pole, and high pole
    all_ancestors = get_ancestors_at_level(image_names, level)
    baseline_counts = Counter(all_ancestors)
    n_baseline = len(all_ancestors)

    low_names = [image_names[i] for i, m in enumerate(low_mask) if m]
    high_names = [image_names[i] for i, m in enumerate(high_mask) if m]

    low_ancestors = get_ancestors_at_level(low_names, level)
    high_ancestors = get_ancestors_at_level(high_names, level)

    # Compute enrichment vs baseline
    low_enriched = compute_enrichment_vs_baseline(
        low_ancestors, baseline_counts, n_baseline, min_count_low)
    high_enriched = compute_enrichment_vs_baseline(
        high_ancestors, baseline_counts, n_baseline, min_count_high)

    # Sort by enrichment (descending)
    low_enriched.sort(key=lambda x: x['enrichment'], reverse=True)
    high_enriched.sort(key=lambda x: x['enrichment'], reverse=True)

    return {
        'level': level,
        'low_enriched': low_enriched,
        'high_enriched': high_enriched,
        'n_low': n_low,
        'n_high': n_high,
        'n_total': len(image_names),
        'min_count_low': min_count_low,
        'min_count_high': min_count_high,
        'all_ancestors': all_ancestors,  # For histogram plotting
    }


def print_results(results, model, pc, dataset):
    """Print all categories with positive enrichment for LLM analysis."""
    print(f"\n{'='*70}")
    print(f"PC{pc} Semantic Analysis ({model.upper()})")
    print(f"Dataset: {dataset} | WordNet level: {results['level']}")
    print(f"Total images: {results['n_total']:,}")
    print(f"{'='*70}")

    # Filter to only positive enrichment
    low_positive = [r for r in results['low_enriched'] if r['enrichment'] > 0]
    high_positive = [r for r in results['high_enriched'] if r['enrichment'] > 0]

    print(f"\n--- LOW POLE (n={results['n_low']:,}, min_count={results['min_count_low']}) ---")
    print(f"{'Category':<25} {'Count':>6} {'Pole%':>7} {'Base%':>7} {'Enrich':>8}")
    print("-" * 55)
    for r in low_positive:
        print(f"{r['category']:<25} {r['count']:>6} {r['pole_pct']:>6.1f}% {r['baseline_pct']:>6.1f}% {r['enrichment']:>+7.1f}%")

    print(f"\n--- HIGH POLE (n={results['n_high']:,}, min_count={results['min_count_high']}) ---")
    print(f"{'Category':<25} {'Count':>6} {'Pole%':>7} {'Base%':>7} {'Enrich':>8}")
    print("-" * 55)
    for r in high_positive:
        print(f"{r['category']:<25} {r['count']:>6} {r['pole_pct']:>6.1f}% {r['baseline_pct']:>6.1f}% {r['enrichment']:>+7.1f}%")

    print(f"\n{'='*70}\n")


def plot_histogram(scores, results, model, pc, output_dir="experiments/semantic_analysis/pc_histogram"):
    """Plot overlapping histograms for top 3 positively enriched categories from each pole."""
    all_ancestors = results['all_ancestors']

    # Get top 3 with positive enrichment from each pole
    low_positive = [r for r in results['low_enriched'] if r['enrichment'] > 0]
    high_positive = [r for r in results['high_enriched'] if r['enrichment'] > 0]
    top_low = [r['category'] for r in low_positive[:3]]
    top_high = [r['category'] for r in high_positive[:3]]

    # Colors: blues for low pole, reds for high pole
    low_colors = ['#1f77b4', '#6baed6', '#9ecae1']  # dark to light blue
    high_colors = ['#d62728', '#fc8d62', '#fdae6b']  # dark to light red

    plt.figure(figsize=(12, 6))

    # Plot histograms for low-pole enriched categories
    for i, cat in enumerate(top_low):
        # Find indices where ancestor matches this category
        cat_scores = [scores[j] for j, anc in enumerate(all_ancestors)
                      if anc.split('.')[0] == cat]
        if cat_scores:
            plt.hist(cat_scores, bins=50, alpha=0.5, label=f"{cat} (low)",
                     color=low_colors[i], density=True)

    # Plot histograms for high-pole enriched categories
    for i, cat in enumerate(top_high):
        cat_scores = [scores[j] for j, anc in enumerate(all_ancestors)
                      if anc.split('.')[0] == cat]
        if cat_scores:
            plt.hist(cat_scores, bins=50, alpha=0.5, label=f"{cat} (high)",
                     color=high_colors[i], density=True)

    plt.xlabel(f'PC{pc} Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'PC{pc} Distribution by Category ({model.upper()})', fontsize=14)
    plt.legend(loc='upper right')
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"pc{pc}_histogram_{model}.png")
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

    results = analyze_pc(scores, image_names, args.level, args.percentile)
    print_results(results, args.model, args.pc, args.dataset)
    plot_histogram(scores, results, args.model, args.pc)


if __name__ == '__main__':
    main()
