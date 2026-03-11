"""
Single-figure summary of representation geometry across granularity levels.

Produces a 2x2 figure with mean +/- SEM across 3 seeds:
  (a) top-left:     FC1 Eigenspectrum (log-log)
  (b) top-right:    Effective Dimensionality / Participation Ratio (log y)
  (c) bottom-left:  Two-NN Intrinsic Dimension
  (d) bottom-right: Activation Sparsity

Color scheme:
  - Untrained:  gray dashed
  - 2-64 way:   blue gradient (light=coarse, dark=fine)
  - 1000-way:   orange

Usage (from project root):
    python experiments/representation_analysis/representation_summary.py           # compute + save + plot
    python experiments/representation_analysis/representation_summary.py --plot-only  # re-plot from saved data
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "dimensionality"))

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

from visreps.models.utils import FeatureExtractor
from visreps.models.custom_model import CustomCNN
from visreps.dataloaders.obj_cls import get_obj_cls_loader
from metrics import hoyer_sparsity, participation_ratio, two_nn_dimension, eigenspectrum


# ── Configuration ──────────────────────────────────────────────────────
DATASET = "imagenet-mini-50"
ALL_LAYERS = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2']
SEED_LETTERS = ['a', 'b', 'c']

# Coarse-grained levels (blue gradient)
COARSE_IDS = [2, 4, 8, 16, 32, 64]

# All levels including untrained and 1000-way
ALL_IDS = ['untrained'] + COARSE_IDS + [1000]

def get_checkpoint_paths(cfg_id):
    if cfg_id == 1000:
        return [f"/data/ymehta3/default/cfg1000{s}/checkpoint_epoch_20.pth" for s in SEED_LETTERS]
    return [f"/data/ymehta3/alexnet_pca/cfg{cfg_id}{s}/checkpoint_epoch_20.pth" for s in SEED_LETTERS]

EIGENSPECTRUM_LAYER = 'fc1'
N_EIGEN_COMPONENTS = 100
N_SAMPLES_TWONN = 2000
CONV_POOL_SIZE = 3
SEED = 42
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "figs")


# ── Styling ────────────────────────────────────────────────────────────
UNTRAINED_STYLE = {'color': '0.5', 'linestyle': (0, (5, 2.5)), 'linewidth': 1.6, 'markersize': 4.5, 'zorder': 1}
FINE_STYLE = {'color': '#e8590c', 'linestyle': '-', 'linewidth': 1.8, 'markersize': 5.5, 'zorder': 10}

def get_coarse_colors(n):
    """Blue gradient: light (fewest classes) to dark (most classes)."""
    cmap = matplotlib.colormaps['Blues']
    return [cmap(0.25 + 0.65 * i / (n - 1)) for i in range(n)]


def get_style(cfg_id):
    """Return style dict for a given cfg_id."""
    if cfg_id == 'untrained':
        return UNTRAINED_STYLE
    elif cfg_id == 1000:
        return FINE_STYLE
    else:
        idx = COARSE_IDS.index(cfg_id)
        colors = get_coarse_colors(len(COARSE_IDS))
        return {'color': colors[idx], 'linestyle': '-', 'linewidth': 1.4, 'markersize': 4.5, 'zorder': 5}


def get_label(cfg_id):
    if cfg_id == 'untrained':
        return 'Untrained'
    return f'{cfg_id}-way'


# ── Model loading & feature extraction ─────────────────────────────────
def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return checkpoint['model'].to(device).eval()


def create_untrained_model(device):
    """Create a randomly initialized CustomCNN (same architecture, no training)."""
    model = CustomCNN(num_classes=1000)
    return model.to(device).eval()


def extract_all_layers(model, loader, device, layers=ALL_LAYERS):
    """Extract post-ReLU features from all layers."""
    return_nodes = {layer: layer for layer in layers}
    extractor = FeatureExtractor(model, return_nodes=return_nodes,
                                 post_relu=True, extract_pre_and_post=False)
    extractor.to(device).eval()

    adaptive_pool = torch.nn.AdaptiveAvgPool2d((CONV_POOL_SIZE, CONV_POOL_SIZE))
    features = {layer: [] for layer in layers}

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="    Extracting", leave=False):
            feats = extractor(images.to(device))
            for layer in layers:
                layer_feats = feats[layer]
                if layer_feats.dim() == 4:
                    layer_feats = adaptive_pool(layer_feats)
                layer_feats = layer_feats.view(layer_feats.size(0), -1)
                features[layer].append(layer_feats.cpu().numpy())

    return {layer: np.vstack(features[layer]) for layer in layers}


# ── Metrics ────────────────────────────────────────────────────────────
def l2_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-10)


def compute_metrics(feats_dict, layers):
    """Compute sparsity (raw), PR/Two-NN/eigenspectrum (L2-normalized)."""
    results = {
        'sparsity': {},
        'pr': {},
        'twonn': {},
        'eigenvalues': {},
        'erank': {},
    }

    for layer in layers:
        X = feats_dict[layer]
        results['sparsity'][layer] = np.mean(hoyer_sparsity(X))

        X_norm = l2_normalize(X)
        results['pr'][layer] = participation_ratio(X_norm)

        dim, _ = two_nn_dimension(X_norm, n_samples=N_SAMPLES_TWONN, seed=SEED)
        results['twonn'][layer] = dim

        eigs = eigenspectrum(X_norm)
        results['eigenvalues'][layer] = eigs
        results['erank'][layer] = effective_rank(eigs)

    return results


def effective_rank(eigenvalues):
    eigs = eigenvalues[eigenvalues > 0]
    p = eigs / eigs.sum()
    return np.exp(-np.sum(p * np.log(p)))


def aggregate_seeds(all_seed_results, layers, metric_key):
    n_seeds = len(all_seed_results)
    vals = np.array([[r[metric_key][l] for l in layers] for r in all_seed_results])
    means = vals.mean(axis=0)
    sems = vals.std(axis=0) / np.sqrt(n_seeds)
    return means, sems


# ── Plotting ───────────────────────────────────────────────────────────
LAYER_LABELS = {
    'conv1': 'Conv1', 'conv2': 'Conv2', 'conv3': 'Conv3',
    'conv4': 'Conv4', 'conv5': 'Conv5', 'fc1': 'FC1', 'fc2': 'FC2',
}


def _despine(ax):
    sns.despine(ax=ax, offset=5)


def _plot_layer_metric(ax, all_results, layers, metric_key, ylabel, panel_label,
                       log_y=False):
    """Plot a metric across layers with shaded SEM bands."""
    x = np.arange(len(layers))

    for cfg_id in ALL_IDS:
        style = get_style(cfg_id)
        label = get_label(cfg_id)
        means, sems = aggregate_seeds(all_results[cfg_id], layers, metric_key)
        n_seeds = len(all_results[cfg_id])
        if n_seeds > 1:
            ax.fill_between(x, means - sems, means + sems,
                            color=style['color'], alpha=0.12, linewidth=0)
        ax.plot(x, means, color=style['color'], linestyle=style['linestyle'],
                linewidth=style['linewidth'], marker='o',
                markersize=style['markersize'], markeredgecolor='white',
                markeredgewidth=0.8, label=label, zorder=style.get('zorder', 5))

    ax.set_xticks(x)
    ax.set_xticklabels([LAYER_LABELS.get(l, l) for l in layers])
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Layer')
    ax.set_title(panel_label, loc='left', fontsize=10, fontweight='bold', pad=8)
    if log_y:
        ax.set_yscale('log')
    _despine(ax)


def make_figure(all_results, layers, output_path):
    """Create the 2x2 summary figure."""
    sns.set_theme(style='ticks', context='paper', font_scale=1.05)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
    })

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))

    # ── (a) FC1 Eigenspectrum (top-left, log-log) ─────────────────────
    ax = axes[0, 0]
    layer = EIGENSPECTRUM_LAYER
    for cfg_id in ALL_IDS:
        style = get_style(cfg_id)
        label = get_label(cfg_id)
        spectra = []
        for r in all_results[cfg_id]:
            eigs = r['eigenvalues'][layer]
            n_plot = min(N_EIGEN_COMPONENTS, len(eigs))
            spectra.append(eigs[:n_plot] / eigs.sum())
        spectra = np.array(spectra)
        mean_spec = spectra.mean(axis=0)
        sem_spec = spectra.std(axis=0) / np.sqrt(len(spectra)) if len(spectra) > 1 else np.zeros_like(mean_spec)
        comps = np.arange(1, len(mean_spec) + 1)
        if len(spectra) > 1:
            ax.fill_between(comps, mean_spec - sem_spec, mean_spec + sem_spec,
                            color=style['color'], alpha=0.10, linewidth=0)
        ax.plot(comps, mean_spec, color=style['color'],
                linestyle=style['linestyle'], linewidth=style['linewidth'] + 0.2,
                label=label, zorder=style.get('zorder', 5))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Component')
    ax.set_ylabel('Fraction of variance')
    ax.set_title('(a) FC1 eigenspectrum', loc='left', fontsize=10, fontweight='bold', pad=8)
    _despine(ax)

    # ── (b) Effective Dimensionality (top-right, log y) ────────────────
    _plot_layer_metric(axes[0, 1], all_results, layers, 'pr',
                       'Participation ratio', '(b) Effective dimensionality',
                       log_y=True)

    # ── (c) Two-NN Intrinsic Dimension (bottom-left) ──────────────────
    _plot_layer_metric(axes[1, 0], all_results, layers, 'twonn',
                       'Intrinsic dimension', '(c) Two-NN intrinsic dimension')

    # ── (d) Activation Sparsity (bottom-right) ────────────────────────
    _plot_layer_metric(axes[1, 1], all_results, layers, 'sparsity',
                       'Hoyer sparsity', '(d) Activation sparsity')
    axes[1, 1].set_ylim(0.1, 1.02)

    # ── Legend below figure as horizontal strip ─────────────────────────
    handles, labels = axes[0, 0].get_legend_handles_labels()

    plt.tight_layout(h_pad=2.5, w_pad=2.5)
    fig.subplots_adjust(bottom=0.11)

    fig.legend(
        handles, labels, loc='lower center',
        bbox_to_anchor=(0.5, -0.005), frameon=False,
        fontsize=8, ncol=4, columnspacing=1.8,
        handlelength=2.2, handletextpad=0.5, labelspacing=0.4,
    )
    plt.savefig(output_path, dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


# ── Data I/O ──────────────────────────────────────────────────────────
DATA_PATH = os.path.join(OUTPUT_DIR, "representation_summary_data.json")


def save_results(all_results, layers):
    """Serialize all_results to JSON (eigenvalues → list for JSON compat)."""
    out = {"layers": layers, "results": {}}
    for cfg_id, seed_list in all_results.items():
        key = str(cfg_id)
        out["results"][key] = []
        for metrics in seed_list:
            entry = {}
            for metric_name, layer_dict in metrics.items():
                entry[metric_name] = {}
                for layer, val in layer_dict.items():
                    if isinstance(val, np.ndarray):
                        entry[metric_name][layer] = val.tolist()
                    else:
                        entry[metric_name][layer] = float(val)
            out["results"][key].append(entry)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(DATA_PATH, 'w') as f:
        json.dump(out, f)
    print(f"Saved data: {DATA_PATH}")


def load_results():
    """Load saved results, converting eigenvalue lists back to np.arrays."""
    with open(DATA_PATH) as f:
        data = json.load(f)
    layers = data["layers"]
    all_results = {}
    for key, seed_list in data["results"].items():
        cfg_id = key if key == 'untrained' else int(key)
        all_results[cfg_id] = []
        for entry in seed_list:
            metrics = {}
            for metric_name, layer_dict in entry.items():
                metrics[metric_name] = {}
                for layer, val in layer_dict.items():
                    if isinstance(val, list):
                        metrics[metric_name][layer] = np.array(val)
                    else:
                        metrics[metric_name][layer] = val
            all_results[cfg_id].append(metrics)
    return all_results, layers


# ── Main ───────────────────────────────────────────────────────────────
def compute_all(layers):
    """Run the full extraction + metrics pipeline for all models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Representation Geometry Summary (all granularities)")
    print("=" * 60)
    print(f"Device: {device}")

    # Load dataset
    print(f"\nLoading {DATASET}...")
    cfg = {"dataset": DATASET, "batchsize": 256, "num_workers": 8}
    _, loaders = get_obj_cls_loader(cfg, shuffle=False, train_test_split=False)
    loader = loaders['all']
    print(f"Images: {len(loader.dataset)}")

    all_results = {cfg_id: [] for cfg_id in ALL_IDS}

    # 1. Untrained model (single random init — no seeds to average)
    print(f"\n[Untrained] Random init CustomCNN")
    model = create_untrained_model(device)
    print(f"  Extracting features...")
    feats = extract_all_layers(model, loader, device, layers)
    print(f"  Computing metrics...")
    metrics = compute_metrics(feats, layers)
    all_results['untrained'] = [metrics]
    del model, feats
    torch.cuda.empty_cache()

    # 2. Coarse-grained models (2-64, 3 seeds each)
    for cfg_id in COARSE_IDS:
        paths = get_checkpoint_paths(cfg_id)
        for i, ckpt_path in enumerate(paths):
            seed_letter = SEED_LETTERS[i]
            print(f"\n[{cfg_id}-way seed={seed_letter}] {ckpt_path}")
            model = load_checkpoint(ckpt_path, device)
            print(f"  Extracting features...")
            feats = extract_all_layers(model, loader, device, layers)
            print(f"  Computing metrics...")
            metrics = compute_metrics(feats, layers)
            all_results[cfg_id].append(metrics)
            del model, feats
            torch.cuda.empty_cache()

    # 3. Fine-grained 1000-way (3 seeds)
    paths = get_checkpoint_paths(1000)
    for i, ckpt_path in enumerate(paths):
        seed_letter = SEED_LETTERS[i]
        print(f"\n[1000-way seed={seed_letter}] {ckpt_path}")
        model = load_checkpoint(ckpt_path, device)
        print(f"  Extracting features...")
        feats = extract_all_layers(model, loader, device, layers)
        print(f"  Computing metrics...")
        metrics = compute_metrics(feats, layers)
        all_results[1000].append(metrics)
        del model, feats
        torch.cuda.empty_cache()

    # Print summary
    print(f"\n{'Layer':<8}", end="")
    for cfg_id in ALL_IDS:
        print(f" {str(cfg_id):>9}", end="")
    print("  (Participation Ratio)")
    print("-" * (8 + 10 * len(ALL_IDS)))
    for layer in layers:
        print(f"{layer:<8}", end="")
        for cfg_id in ALL_IDS:
            pr = np.mean([r['pr'][layer] for r in all_results[cfg_id]])
            print(f" {pr:9.1f}", end="")
        print()

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true',
                        help='Re-plot from saved data (skip computation)')
    args = parser.parse_args()

    np.random.seed(SEED)
    layers = ALL_LAYERS

    if args.plot_only:
        print(f"Loading saved data from {DATA_PATH}...")
        all_results, layers = load_results()
    else:
        all_results = compute_all(layers)
        save_results(all_results, layers)

    # Generate figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "representation_summary.png")
    make_figure(all_results, layers, output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
