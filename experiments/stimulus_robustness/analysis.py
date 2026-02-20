"""Stimulus robustness analysis: verify that coarse > fine-grain brain
alignment is robust to stimulus subsampling, not an artifact.

Computes per-layer Spearman RSA under three conditions:
  1. All stimuli (~900 shared NSD stimuli)
  2. Train half (50% of stimuli)
  3. Test half (50% of stimuli)

Models:
  - AlexNet-1K (baseline fine-grain)
  - 4 coarse-grain PCA models (AlexNet-PCA, ViT-PCA, CLIP-PCA, DINO-PCA)

Usage:
    source /home/ymehta3/research/VisionAI/visreps/.venv/bin/activate && \
    python experiments/stimulus_robustness/analysis.py
"""

import gc
import json
import numpy as np
import torch
from omegaconf import OmegaConf

from visreps.models.utils import load_model, configure_feature_extractor, get_activations
from visreps.dataloaders.neural import get_neural_loader
from visreps.analysis.alignment import prepare_data_for_alignment
from visreps.analysis.rsa import compute_rdm, compute_rdm_correlation


# ── Configuration ─────────────────────────────────────────────────────────
SEED = 1
SUBJECT_IDX = 0
RETURN_NODES = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2"]
OUTPUT_PATH = "experiments/stimulus_robustness/data.json"

MODELS = {
    "early visual stream": [
        {"name": "AlexNet-1K",     "checkpoint_dir": "/data/ymehta3/default",     "cfg_id": 1000},
        {"name": "AlexNet-PCA-64", "checkpoint_dir": "/data/ymehta3/alexnet_pca", "cfg_id": 64},
        {"name": "ViT-PCA-16",    "checkpoint_dir": "/data/ymehta3/vit_pca",     "cfg_id": 16},
        {"name": "CLIP-PCA-16",   "checkpoint_dir": "/data/ymehta3/clip_pca",    "cfg_id": 16},
        {"name": "DINO-PCA-16",   "checkpoint_dir": "/data/ymehta3/dino_pca",    "cfg_id": 16},
    ],
    "ventral visual stream": [
        {"name": "AlexNet-1K",     "checkpoint_dir": "/data/ymehta3/default",     "cfg_id": 1000},
        {"name": "AlexNet-PCA-64", "checkpoint_dir": "/data/ymehta3/alexnet_pca", "cfg_id": 64},
        {"name": "ViT-PCA-64",    "checkpoint_dir": "/data/ymehta3/vit_pca",     "cfg_id": 64},
        {"name": "CLIP-PCA-64",   "checkpoint_dir": "/data/ymehta3/clip_pca",    "cfg_id": 64},
        {"name": "DINO-PCA-64",   "checkpoint_dir": "/data/ymehta3/dino_pca",    "cfg_id": 64},
    ],
}


# ── Helpers ───────────────────────────────────────────────────────────────
def make_cfg(checkpoint_dir, cfg_id, region):
    """Build minimal OmegaConf config compatible with the eval pipeline."""
    return OmegaConf.create({
        "load_model_from": "checkpoint",
        "checkpoint_dir": checkpoint_dir,
        "cfg_id": cfg_id,
        "seed": SEED,
        "checkpoint_model": "checkpoint_epoch_20.pth",
        "return_nodes": RETURN_NODES,
        "extract_pre_and_post": True,
        "neural_dataset": "nsd",
        "region": region,
        "subject_idx": SUBJECT_IDX,
        "batchsize": 256,
        "num_workers": 16,
    })


def per_layer_rsa(acts, neural):
    """Per-layer Spearman RSA using all stimuli (computed on CPU)."""
    neural_rdm = compute_rdm(neural.float())
    scores = {}
    for layer, a in acts.items():
        flat = a.flatten(start_dim=1).float()
        rdm = compute_rdm(flat)
        scores[layer] = compute_rdm_correlation(rdm, neural_rdm, correlation="Spearman")
    return scores


def per_layer_rsa_split(acts, neural, split_seed=42):
    """Per-layer Spearman RSA on train and test halves (computed on CPU).

    Returns (train_scores, test_scores) dicts.
    """
    n = neural.size(0)
    rng = np.random.RandomState(split_seed)
    perm = rng.permutation(n)
    train_idx = torch.from_numpy(perm[: n // 2])
    test_idx = torch.from_numpy(perm[n // 2:])

    results = {}
    for label, idx in [("train", train_idx), ("test", test_idx)]:
        neural_rdm = compute_rdm(neural[idx].float())
        scores = {}
        for layer, a in acts.items():
            flat = a[idx].flatten(start_dim=1).float()
            rdm = compute_rdm(flat)
            scores[layer] = compute_rdm_correlation(rdm, neural_rdm, correlation="Spearman")
        results[label] = scores
    return results["train"], results["test"]


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_results = {}

    for region, models in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Region: {region}")
        print(f"{'='*60}")

        all_results[region] = {}

        # Load neural data + dataloader once per region
        base_cfg = make_cfg(models[0]["checkpoint_dir"], models[0]["cfg_id"], region)
        neural_data, dataloader = get_neural_loader(base_cfg)
        print(f"  NSD stimuli loaded: {len(neural_data)}")

        for m in models:
            name = m["name"]
            print(f"\n  [{name}] checkpoint={m['checkpoint_dir']}/cfg{m['cfg_id']}a")

            cfg = make_cfg(m["checkpoint_dir"], m["cfg_id"], region)

            # Load model + feature extractor
            model = load_model(cfg, device)
            extractor = configure_feature_extractor(cfg, model)

            # Extract activations
            acts, ids = get_activations(extractor, dataloader, device)
            print(f"    Layers: {list(acts.keys())}")
            print(f"    Stimuli: {len(ids)}")

            # Align activations with neural data
            acts_al, neural_al = prepare_data_for_alignment(cfg, acts, neural_data, ids)
            print(f"    Aligned: {neural_al.shape[0]} stimuli x {neural_al.shape[1]} voxels")

            # All stimuli RSA (on CPU to avoid GPU OOM with other processes)
            print("    Computing RSA (all stimuli)...")
            scores_all = per_layer_rsa(acts_al, neural_al)

            # 50/50 split RSA (train and test halves)
            print("    Computing RSA (50/50 train + test)...")
            scores_train, scores_test = per_layer_rsa_split(acts_al, neural_al)

            layer_order = list(acts_al.keys())
            all_results[region][name] = {
                "layer_order": layer_order,
                "all_stimuli": scores_all,
                "train_50": scores_train,
                "test_50": scores_test,
            }

            for layer in layer_order:
                print(f"      {layer:<15} all={scores_all[layer]:.4f}  "
                      f"train={scores_train[layer]:.4f}  test={scores_test[layer]:.4f}")

            del model, extractor, acts, acts_al
            gc.collect()
            torch.cuda.empty_cache()

    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
