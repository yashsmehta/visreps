"""
Extract concept-averaged activations for THINGS visualizations.

Models (AlexNet architecture, seed 1):
  - Untrained: random-init AlexNet (no checkpoint)
  - 2-class (AlexNet PCA): /data/ymehta3/alexnet_pca/cfg2a/checkpoint_epoch_20.pth
  - 1000-class:            /data/ymehta3/default/cfg1000a/checkpoint_epoch_20.pth
  - CLIP 4-class:          /data/ymehta3/clip_pca/cfg4a/checkpoint_epoch_20.pth

Outputs: experiments/things_visualizations/data/things_viz_data.npz
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

import numpy as np
import torch
from tqdm import tqdm

from visreps.dataloaders.neural import load_things_data, _make_loader
from visreps.dataloaders.obj_cls import get_transform
from visreps.models.utils import FeatureExtractor
from visreps.models.custom_model import CustomCNN
from visreps.analysis.rsa import compute_rdm, compute_rdm_correlation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
NUM_WORKERS = 8

CHECKPOINTS = {
    "2-class": "/data/ymehta3/alexnet_pca/cfg2a/checkpoint_epoch_20.pth",
    "1000-class": "/data/ymehta3/default/cfg1000a/checkpoint_epoch_20.pth",
    "clip-4": "/data/ymehta3/clip_pca/cfg4a/checkpoint_epoch_20.pth",
}

# Best layers from results.db (seed 1, spearman, things-behavior)
BEST_LAYERS = {
    "2-class": "conv5_post",
    "1000-class": "fc1_pre",
    "clip-4": "fc2_pre",
}

ALL_LAYERS = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2"]
LABELS_PATH = "/data/shared/datasets/hebart2023.things-data/behavior/variables/labels.txt"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "things_visualizations", "data")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "things_viz_data.npz")


def load_model_from_checkpoint(path):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    return checkpoint["model"].to(DEVICE).eval()


def make_extractor(model, layers):
    return_nodes = {l: l for l in layers}
    return FeatureExtractor(model, return_nodes, extract_pre_and_post=True).to(DEVICE).eval()


def extract_single_layer(extractor, dataloader, layer_name):
    """Extract flattened activations for a single layer. Returns (acts, ids)."""
    all_acts, all_ids = [], []
    with torch.no_grad():
        for imgs, keys in tqdm(dataloader, desc=f"  Extracting {layer_name}", leave=False):
            out = extractor(imgs.to(DEVICE))[layer_name]
            all_acts.append(out.view(out.size(0), -1).cpu().numpy())
            all_ids.extend(keys)
    return np.vstack(all_acts), all_ids


def extract_all_layers(extractor, dataloader, pool_size=3):
    """Extract all layers with adaptive pooling for conv layers. Returns (layer_acts, ids)."""
    adaptive_pool = torch.nn.AdaptiveAvgPool2d((pool_size, pool_size))
    layer_names = list(extractor.return_nodes.keys())
    layer_acts = {l: [] for l in layer_names}
    all_ids = []

    with torch.no_grad():
        for imgs, keys in tqdm(dataloader, desc="  Extracting all layers", leave=False):
            feats = extractor(imgs.to(DEVICE))
            all_ids.extend(keys)
            for l in layer_names:
                out = feats[l]
                if out.dim() == 4:
                    out = adaptive_pool(out)
                layer_acts[l].append(out.view(out.size(0), -1).cpu().numpy())

    return {l: np.vstack(v) for l, v in layer_acts.items()}, all_ids


def concept_average(acts, ids, concept_image_ids):
    """Average per-image activations to per-concept. Returns (concept_acts, concept_names)."""
    id_to_idx = {str(sid): i for i, sid in enumerate(ids)}
    concept_names = sorted(concept_image_ids.keys())
    concept_acts = []
    for concept in concept_names:
        indices = [id_to_idx[sid] for sid in concept_image_ids[concept] if sid in id_to_idx]
        concept_acts.append(acts[indices].mean(axis=0) if indices else np.zeros(acts.shape[1]))
    return np.vstack(concept_acts), concept_names


def select_best_layer(layer_acts, concept_names, embeddings):
    """Run layer selection using 20% of concepts (same split as eval pipeline)."""
    rng = np.random.RandomState(42)
    n = len(concept_names)
    sel_idx = rng.permutation(n)[:int(n * 0.2)]

    emb_matrix = np.array([embeddings[c] for c in concept_names])
    neural_rdm = compute_rdm(torch.tensor(emb_matrix[sel_idx], dtype=torch.float32))

    best_layer, best_score, scores = None, -float("inf"), {}
    for layer_name, acts in layer_acts.items():
        model_rdm = compute_rdm(torch.tensor(acts[sel_idx], dtype=torch.float32))
        score = compute_rdm_correlation(model_rdm, neural_rdm, correlation="Spearman")
        scores[layer_name] = score
        if score > best_score:
            best_score, best_layer = score, layer_name
    return best_layer, scores


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load THINGS data
    print("Loading THINGS data...")
    targets, img_paths = load_things_data()
    embeddings = targets["embeddings"]
    concept_image_ids = targets["image_ids"]
    concept_names = sorted(embeddings.keys())
    print(f"  {len(concept_names)} concepts, {len(img_paths)} images")

    emb_matrix = np.array([embeddings[c] for c in concept_names])
    with open(LABELS_PATH) as f:
        dimension_labels = [line.strip() for line in f]

    transform = get_transform(ds_stats="imgnet")
    dataloader = _make_loader(img_paths, transform, BATCH_SIZE, NUM_WORKERS)

    # Extract features from checkpoint models
    results = {}
    for model_key, ckpt_path in CHECKPOINTS.items():
        best_layer = BEST_LAYERS[model_key]
        base_layer = best_layer.replace("_pre", "").replace("_post", "")
        print(f"\n  {model_key} (best layer: {best_layer}, checkpoint: {ckpt_path})")

        model = load_model_from_checkpoint(ckpt_path)
        extractor = make_extractor(model, [base_layer])
        acts, ids = extract_single_layer(extractor, dataloader, best_layer)
        concept_acts, cnames = concept_average(acts, ids, concept_image_ids)
        assert cnames == concept_names
        results[model_key] = concept_acts
        print(f"  Concept-averaged: {concept_acts.shape}")
        del model, extractor, acts; torch.cuda.empty_cache()

    # Untrained model: extract all layers, run selection
    print("\n  Untrained model (layer selection required)")
    model = CustomCNN(num_classes=1000).to(DEVICE).eval()
    extractor = make_extractor(model, ALL_LAYERS)
    layer_acts_raw, ids = extract_all_layers(extractor, dataloader)

    layer_concept_acts = {}
    for lname, acts in layer_acts_raw.items():
        ca, cnames = concept_average(acts, ids, concept_image_ids)
        assert cnames == concept_names
        layer_concept_acts[lname] = ca

    best_layer, sel_scores = select_best_layer(layer_concept_acts, concept_names, embeddings)
    print(f"  Best layer: {best_layer}")
    for l, s in sorted(sel_scores.items(), key=lambda x: -x[1]):
        print(f"    {l:<15} {s:.4f}{' <-- best' if l == best_layer else ''}")

    results["untrained"] = layer_concept_acts[best_layer]
    del model, extractor, layer_acts_raw, layer_concept_acts; torch.cuda.empty_cache()

    # Collect representative image paths (first image per concept)
    rep_image_paths = [img_paths[concept_image_ids[c][0]] for c in concept_names]

    # Save
    print(f"\nSaving to {OUTPUT_PATH}...")
    np.savez_compressed(
        OUTPUT_PATH,
        untrained_acts=results["untrained"],
        twoclass_acts=results["2-class"],
        thousand_acts=results["1000-class"],
        clip4_acts=results["clip-4"],
        embeddings=emb_matrix,
        dimension_labels=np.array(dimension_labels),
        concept_names=np.array(concept_names),
        rep_image_paths=np.array(rep_image_paths),
        best_layers=np.array([
            f"untrained:{best_layer}",
            f"2-class:{BEST_LAYERS['2-class']}",
            f"1000-class:{BEST_LAYERS['1000-class']}",
            f"clip-4:{BEST_LAYERS['clip-4']}",
        ]),
    )

    data = np.load(OUTPUT_PATH, allow_pickle=True)
    print("\nVerification:")
    for key in data.files:
        print(f"  {key}: {data[key].shape}")
    print("Done!")


if __name__ == "__main__":
    main()
