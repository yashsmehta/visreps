"""
Compare representation geometry metrics for pre-ReLU vs post-ReLU activations.

Produces a 2x4 figure: top row = pre-ReLU, bottom row = post-ReLU.
Columns: (1) FC1 Eigenspectrum, (2) Effective Dimensionality,
         (3) Two-NN Intrinsic Dimension, (4) Activation Sparsity.

Uses a subset of models for clarity: untrained, 2-way, 32-way, 1000-way.

Usage (from project root):
    python experiments/representation_analysis/pre_post_relu_comparison.py
    python experiments/representation_analysis/pre_post_relu_comparison.py --plot-only
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
SEED_LETTERS = ['a']

# Subset of models for clarity
COMPARE_IDS = ['untrained', 2, 32, 1000]

COARSE_IDS = [2, 4, 8, 16, 32, 64]

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
STYLES = {
    'untrained': {'color': '0.5', 'linestyle': (0, (5, 2.5)), 'linewidth': 1.6, 'markersize': 5},
    2:           {'color': matplotlib.colormaps['Blues'](0.3), 'linestyle': '-', 'linewidth': 1.5, 'markersize': 5},
    32:          {'color': matplotlib.colormaps['Blues'](0.75), 'linestyle': '-', 'linewidth': 1.5, 'markersize': 5},
    1000:        {'color': '#e8590c', 'linestyle': '-', 'linewidth': 1.8, 'markersize': 5.5},
}

LABELS = {
    'untrained': 'Untrained',
    2: '2-way',
    32: '32-way',
    1000: '1000-way',
}

LAYER_LABELS = {
    'conv1': 'Conv1', 'conv2': 'Conv2', 'conv3': 'Conv3',
    'conv4': 'Conv4', 'conv5': 'Conv5', 'fc1': 'FC1', 'fc2': 'FC2',
}


# ── Model loading & feature extraction ─────────────────────────────────
def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return checkpoint['model'].to(device).eval()


def create_untrained_model(device):
    model = CustomCNN(num_classes=1000)
    return model.to(device).eval()


def extract_pre_post(model, loader, device, layers=ALL_LAYERS):
    """Extract both pre-ReLU and post-ReLU features from all layers."""
    return_nodes = {layer: layer for layer in layers}
    extractor = FeatureExtractor(model, return_nodes=return_nodes,
                                 extract_pre_and_post=True)
    extractor.to(device).eval()

    adaptive_pool = torch.nn.AdaptiveAvgPool2d((CONV_POOL_SIZE, CONV_POOL_SIZE))

    # Collect all output keys from the extractor
    all_keys = list(extractor.return_nodes.values())
    features = {k: [] for k in all_keys}

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="    Extracting", leave=False):
            feats = extractor(images.to(device))
            for k in all_keys:
                layer_feats = feats[k]
                if layer_feats.dim() == 4:
                    layer_feats = adaptive_pool(layer_feats)
                layer_feats = layer_feats.view(layer_feats.size(0), -1)
                features[k].append(layer_feats.cpu().numpy())

    stacked = {k: np.vstack(features[k]) for k in all_keys}

    # Separate into pre and post dicts keyed by original layer name
    pre_feats = {}
    post_feats = {}
    for layer in layers:
        pre_key = f"{layer}_pre"
        post_key = f"{layer}_post"
        if pre_key in stacked and post_key in stacked:
            pre_feats[layer] = stacked[pre_key]
            post_feats[layer] = stacked[post_key]
        else:
            # No activation found — same for both (e.g. last FC without ReLU)
            pre_feats[layer] = stacked.get(layer, stacked.get(pre_key, stacked.get(post_key)))
            post_feats[layer] = pre_feats[layer]

    return pre_feats, post_feats


# ── Metrics ────────────────────────────────────────────────────────────
def l2_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-10)


def compute_metrics(feats_dict, layers):
    results = {
        'sparsity': {},
        'pr': {},
        'twonn': {},
        'eigenvalues': {},
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
    return results


def aggregate_seeds(seed_results, layers, metric_key):
    n_seeds = len(seed_results)
    vals = np.array([[r[metric_key][l] for l in layers] for r in seed_results])
    means = vals.mean(axis=0)
    sems = vals.std(axis=0) / np.sqrt(n_seeds) if n_seeds > 1 else np.zeros_like(means)
    return means, sems


# ── Plotting ───────────────────────────────────────────────────────────
def _despine(ax):
    sns.despine(ax=ax, offset=5)


def _plot_eigenspectrum(ax, results_dict, title):
    layer = EIGENSPECTRUM_LAYER
    for cfg_id in COMPARE_IDS:
        style = STYLES[cfg_id]
        spectra = []
        for r in results_dict[cfg_id]:
            eigs = r['eigenvalues'][layer]
            n_plot = min(N_EIGEN_COMPONENTS, len(eigs))
            spectra.append(eigs[:n_plot] / eigs.sum())
        spectra = np.array(spectra)
        mean_spec = spectra.mean(axis=0)
        comps = np.arange(1, len(mean_spec) + 1)
        if len(spectra) > 1:
            sem = spectra.std(axis=0) / np.sqrt(len(spectra))
            ax.fill_between(comps, mean_spec - sem, mean_spec + sem,
                            color=style['color'], alpha=0.10, linewidth=0)
        ax.plot(comps, mean_spec, color=style['color'],
                linestyle=style['linestyle'], linewidth=style['linewidth'] + 0.2,
                label=LABELS[cfg_id])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Component')
    ax.set_ylabel('Fraction of variance')
    ax.set_title(title, loc='left', fontsize=10, fontweight='bold', pad=8)
    _despine(ax)


def _plot_layer_metric(ax, results_dict, layers, metric_key, ylabel, title,
                       log_y=False):
    x = np.arange(len(layers))
    for cfg_id in COMPARE_IDS:
        style = STYLES[cfg_id]
        means, sems = aggregate_seeds(results_dict[cfg_id], layers, metric_key)
        n_seeds = len(results_dict[cfg_id])
        if n_seeds > 1:
            ax.fill_between(x, means - sems, means + sems,
                            color=style['color'], alpha=0.12, linewidth=0)
        ax.plot(x, means, color=style['color'], linestyle=style['linestyle'],
                linewidth=style['linewidth'], marker='o',
                markersize=style['markersize'], markeredgecolor='white',
                markeredgewidth=0.8, label=LABELS[cfg_id])

    ax.set_xticks(x)
    ax.set_xticklabels([LAYER_LABELS.get(l, l) for l in layers])
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Layer')
    ax.set_title(title, loc='left', fontsize=10, fontweight='bold', pad=8)
    if log_y:
        ax.set_yscale('log')
    _despine(ax)


def make_figure(pre_results, post_results, layers, output_path):
    sns.set_theme(style='ticks', context='paper', font_scale=1.05)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
    })

    # 4 rows (one per metric) × 2 columns (pre-ReLU, post-ReLU)
    # Share y-axes across columns for fair comparison
    fig, axes = plt.subplots(4, 2, figsize=(10, 14),
                             sharey='row')

    col_titles = ['Pre-ReLU', 'Post-ReLU']
    for col, (results, title) in enumerate([(pre_results, col_titles[0]),
                                             (post_results, col_titles[1])]):
        # Row 0: Eigenspectrum
        panel = chr(ord('a') + col)
        _plot_eigenspectrum(axes[0, col], results,
                            f'({panel}) Eigenspectrum')

        # Row 1: Effective dimensionality
        panel = chr(ord('c') + col)
        _plot_layer_metric(axes[1, col], results, layers, 'pr',
                           'Participation ratio',
                           f'({panel}) Eff. dimensionality', log_y=True)

        # Row 2: Two-NN intrinsic dimension
        panel = chr(ord('e') + col)
        _plot_layer_metric(axes[2, col], results, layers, 'twonn',
                           'Intrinsic dimension',
                           f'({panel}) Two-NN intrinsic dim.')

        # Row 3: Activation sparsity
        panel = chr(ord('g') + col)
        _plot_layer_metric(axes[3, col], results, layers, 'sparsity',
                           'Hoyer sparsity',
                           f'({panel}) Activation sparsity')
        axes[3, col].set_ylim(0, 1.02)

    # Column headers
    for col, title in enumerate(col_titles):
        axes[0, col].annotate(title, xy=(0.5, 1.25), xycoords='axes fraction',
                              fontsize=13, fontweight='bold', ha='center',
                              va='bottom', color='0.2')

    # ── Legend ─────────────────────────────────────────────────────────
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, -0.005), frameon=False,
               fontsize=9, ncol=4, columnspacing=2.0,
               handlelength=2.2, handletextpad=0.5)

    plt.tight_layout(h_pad=2.5, w_pad=3.0)
    fig.subplots_adjust(bottom=0.05, top=0.94)
    plt.savefig(output_path, dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


# ── Data I/O ──────────────────────────────────────────────────────────
DATA_PATH = os.path.join(OUTPUT_DIR, "pre_post_relu_data.json")


def save_results(pre_results, post_results, layers):
    out = {"layers": layers, "pre": {}, "post": {}}
    for tag, results in [("pre", pre_results), ("post", post_results)]:
        for cfg_id, seed_list in results.items():
            key = str(cfg_id)
            out[tag][key] = []
            for metrics in seed_list:
                entry = {}
                for metric_name, layer_dict in metrics.items():
                    entry[metric_name] = {}
                    for layer, val in layer_dict.items():
                        if isinstance(val, np.ndarray):
                            entry[metric_name][layer] = val.tolist()
                        else:
                            entry[metric_name][layer] = float(val)
                out[tag][key].append(entry)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(DATA_PATH, 'w') as f:
        json.dump(out, f)
    print(f"Saved data: {DATA_PATH}")


def load_results():
    with open(DATA_PATH) as f:
        data = json.load(f)
    layers = data["layers"]
    pre_results = {}
    post_results = {}
    for tag, results_dict in [("pre", pre_results), ("post", post_results)]:
        for key, seed_list in data[tag].items():
            cfg_id = key if key == 'untrained' else int(key)
            results_dict[cfg_id] = []
            for entry in seed_list:
                metrics = {}
                for metric_name, layer_dict in entry.items():
                    metrics[metric_name] = {}
                    for layer, val in layer_dict.items():
                        if isinstance(val, list):
                            metrics[metric_name][layer] = np.array(val)
                        else:
                            metrics[metric_name][layer] = val
                results_dict[cfg_id].append(metrics)
    return pre_results, post_results, layers


# ── Main ───────────────────────────────────────────────────────────────
def compute_all(layers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("Pre vs Post-ReLU Representation Comparison")
    print("=" * 60)
    print(f"Device: {device}")

    print(f"\nLoading {DATASET}...")
    cfg = {"dataset": DATASET, "batchsize": 256, "num_workers": 8}
    _, loaders = get_obj_cls_loader(cfg, shuffle=False, train_test_split=False)
    loader = loaders['all']
    print(f"Images: {len(loader.dataset)}")

    pre_results = {cfg_id: [] for cfg_id in COMPARE_IDS}
    post_results = {cfg_id: [] for cfg_id in COMPARE_IDS}

    # Untrained
    print(f"\n[Untrained] Random init CustomCNN")
    model = create_untrained_model(device)
    pre_feats, post_feats = extract_pre_post(model, loader, device, layers)
    pre_results['untrained'] = [compute_metrics(pre_feats, layers)]
    post_results['untrained'] = [compute_metrics(post_feats, layers)]
    del model, pre_feats, post_feats
    torch.cuda.empty_cache()

    # Trained models
    for cfg_id in [c for c in COMPARE_IDS if c != 'untrained']:
        paths = get_checkpoint_paths(cfg_id)
        for i, ckpt_path in enumerate(paths):
            print(f"\n[{cfg_id}-way seed={SEED_LETTERS[i]}] {ckpt_path}")
            model = load_checkpoint(ckpt_path, device)
            pre_feats, post_feats = extract_pre_post(model, loader, device, layers)
            pre_results[cfg_id].append(compute_metrics(pre_feats, layers))
            post_results[cfg_id].append(compute_metrics(post_feats, layers))
            del model, pre_feats, post_feats
            torch.cuda.empty_cache()

    return pre_results, post_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()

    np.random.seed(SEED)
    layers = ALL_LAYERS

    if args.plot_only:
        print(f"Loading saved data from {DATA_PATH}...")
        pre_results, post_results, layers = load_results()
    else:
        pre_results, post_results = compute_all(layers)
        save_results(pre_results, post_results, layers)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "pre_post_relu_comparison.png")
    make_figure(pre_results, post_results, layers, output_path)
    print("\nDone!")


if __name__ == "__main__":
    main()
