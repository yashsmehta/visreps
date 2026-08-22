"""
Extract features + compute dimensionality metrics, then cache to results.npz.

After this runs once, use `plot.py` to (re)generate figures without re-extracting.

Models: 2/8/32-way coarse (CLIP-PCA labels) vs 1000-way fine-grain — all
CustomCNN, same training pipeline; only label granularity differs.

Per layer (conv1..conv5, fc1, fc2):
  - participation ratio
  - Two-NN intrinsic dimension
  - Hoyer sparsity
  - eigenspectrum + power-law exponent alpha
"""

import os
import sys
import numpy as np
import torch
from tqdm import tqdm

# Add representation_analysis dir + project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from utils import extract_all_layers, ensure_output_dir, ALL_LAYERS, OUTPUT_DIR
from visreps.dataloaders.obj_cls import get_obj_cls_loader
from experiments.coarse_grain_benefits.utils import load_model_by_config

from metrics import (
    participation_ratio, eigenspectrum, n_components_for_variance,
    two_nn_dimension, hoyer_sparsity, fraction_active, power_law_exponent,
)

# ---------------------------------------------------------------------------
# Model configuration: edit this list to add/remove models.
# Blue shades go light->dark with increasing granularity; fine-grain in orange.
# ---------------------------------------------------------------------------
MODELS = [
    {"name": "2-way (CLIP)",  "cfg_id": 2,    "seed": 1,
     "checkpoint_dir": "/data/ymehta3/clip_pca", "color": "#9ecae1"},
    {"name": "8-way (CLIP)",  "cfg_id": 8,    "seed": 1,
     "checkpoint_dir": "/data/ymehta3/clip_pca", "color": "#4292c6"},
    {"name": "32-way (CLIP)", "cfg_id": 32,   "seed": 1,
     "checkpoint_dir": "/data/ymehta3/clip_pca", "color": "#08519c"},
    {"name": "1000-way",      "cfg_id": 1000, "seed": 1,
     "checkpoint_dir": None,  # -> /data/ymehta3/default/
     "color": "#ff7f0e"},
]

DATASET = "imagenet-mini-50"


def compute_all_metrics(feats_dict, layers, n_samples_twonn=2000):
    """Compute all dimensionality metrics for one model across layers.

    Eigenspectrum-derived quantities (PR, n90, alpha, eigenvalues) are stored
    in *two* variants:
      - default (sample-normalized=True): each sample L2-normalized to unit
        norm before PCA. Removes per-sample magnitude bias caused by a small
        fraction of extreme-activation inputs (e.g. saturated post-BN spikes
        in well-trained 1000-way classifiers). This is the cross-model-fair
        view of representation shape.
      - 'raw_*' variants: no sample normalization. Useful for diagnostics.
    """
    results = {
        'pr': {}, 'n90': {}, 'alpha': {}, 'eigenvalues': {},
        'pr_raw': {}, 'n90_raw': {}, 'alpha_raw': {}, 'eigenvalues_raw': {},
        'twonn': {}, 'sparsity': {}, 'sample_norms': {},
    }

    for layer in tqdm(layers, desc="  metrics", leave=False):
        X = feats_dict[layer]

        # Sample-normalized (default for plots)
        eigs = eigenspectrum(X, sample_normalize=True)
        results['eigenvalues'][layer] = eigs
        results['pr'][layer]    = participation_ratio(X, sample_normalize=True)
        results['n90'][layer]   = n_components_for_variance(X, threshold=0.9,
                                                            sample_normalize=True)
        results['alpha'][layer] = power_law_exponent(eigs)

        # Raw (no sample normalization) — kept for diagnostics
        eigs_raw = eigenspectrum(X, sample_normalize=False)
        results['eigenvalues_raw'][layer] = eigs_raw
        results['pr_raw'][layer]    = participation_ratio(X, sample_normalize=False)
        results['n90_raw'][layer]   = n_components_for_variance(X, threshold=0.9,
                                                                sample_normalize=False)
        results['alpha_raw'][layer] = power_law_exponent(eigs_raw)

        # Magnitude diagnostics: per-sample L2 norms.
        norms = np.linalg.norm(X, axis=1)
        results['sample_norms'][layer] = {
            'median': float(np.median(norms)),
            'max':    float(norms.max()),
            'p99':    float(np.quantile(norms, 0.99)),
            'frac_gt_10x_median': float((norms > 10 * np.median(norms)).mean()),
            'frac_gt_100x_median': float((norms > 100 * np.median(norms)).mean()),
        }

        # Magnitude-independent metrics
        dim, std = two_nn_dimension(X, n_samples=n_samples_twonn)
        results['twonn'][layer] = {'dimension': dim, 'std': std}

        sparsity_vals = hoyer_sparsity(X)
        frac_active_vals = fraction_active(X)
        results['sparsity'][layer] = {
            'mean': float(np.mean(sparsity_vals)),
            'std': float(np.std(sparsity_vals)),
            'frac_active': float(np.mean(frac_active_vals)),
        }

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ensure_output_dir()
    output_dir = os.path.join(OUTPUT_DIR, "dimensionality")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Dimensionality: extract features + compute metrics")
    print("=" * 60)

    # 1. Load dataset once (shared across all models)
    print(f"\nLoading {DATASET}...")
    cfg = {"dataset": DATASET, "batchsize": 256, "num_workers": 8}
    _, loaders = get_obj_cls_loader(cfg, shuffle=False, train_test_split=False)
    loader = loaders['all']
    print(f"Images: {len(loader.dataset)}")

    # 2. Extract features + compute metrics for each model
    layers = ALL_LAYERS
    all_results = {}

    for spec in MODELS:
        name = spec["name"]
        print(f"\n=== {name} (cfg{spec['cfg_id']}, seed {spec['seed']}) ===")
        model = load_model_by_config(
            cfg_id=spec["cfg_id"],
            seed=spec["seed"],
            checkpoint_dir=spec["checkpoint_dir"],
            device=device,
        )
        feats = extract_all_layers(model, loader, device, layers)
        all_results[name] = compute_all_metrics(feats, layers)
        del feats, model
        torch.cuda.empty_cache()

    # 3. Cache everything needed for plotting
    cache_path = os.path.join(output_dir, "results.npz")
    np.savez(
        cache_path,
        model_names=np.array([m["name"]  for m in MODELS], dtype=object),
        colors     =np.array([m["color"] for m in MODELS], dtype=object),
        layers     =np.array(layers, dtype=object),
        all_results=np.array([all_results], dtype=object),
    )
    print(f"\n  Saved cache: {cache_path}")
    print("\n  Next: run `python plot.py` to generate figures from this cache.")


if __name__ == "__main__":
    main()
