"""Extract concept-averaged FC1 and FC2 activations for CLIP 8-class and 1000-class.

Saves post-ReLU activations for both layers, concept-averaged over THINGS images.

Usage (from project root):
    python manuscript/figures/fig4/extract_activations.py
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

import numpy as np
import torch
from tqdm import tqdm

from visreps.dataloaders.neural import load_things_data, _make_loader
from visreps.dataloaders.obj_cls import get_transform
from visreps.models.utils import FeatureExtractor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
NUM_WORKERS = 8
LAYERS = ["fc1", "fc2"]

CHECKPOINTS = {
    "clip8":    "/data/ymehta3/clip_pca/cfg8c/checkpoint_epoch_20.pth",
    "thousand": "/data/ymehta3/default/cfg1000c/checkpoint_epoch_20.pth",
}

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_model(path):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    return checkpoint["model"].to(DEVICE).eval()


def extract_layers(model, dataloader, layers):
    """Extract post-ReLU activations for specified layers."""
    extractor = FeatureExtractor(
        model, {l: l for l in layers},
        extract_pre_and_post=True, post_relu=True,
    ).to(DEVICE).eval()

    layer_acts = {l: [] for l in layers}
    all_ids = []
    with torch.no_grad():
        for imgs, keys in tqdm(dataloader, desc="  Extracting"):
            feats = extractor(imgs.to(DEVICE))
            all_ids.extend(keys)
            for l in layers:
                # Use post-ReLU key if available
                key = f"{l}_post" if f"{l}_post" in feats else l
                out = feats[key]
                layer_acts[l].append(out.view(out.size(0), -1).cpu().numpy())

    return {l: np.vstack(v) for l, v in layer_acts.items()}, all_ids


def concept_average(acts, ids, concept_image_ids):
    """Average per-image activations to per-concept."""
    id_to_idx = {str(sid): i for i, sid in enumerate(ids)}
    concept_names = sorted(concept_image_ids.keys())
    concept_acts = []
    for concept in concept_names:
        indices = [id_to_idx[sid] for sid in concept_image_ids[concept]
                   if sid in id_to_idx]
        if indices:
            concept_acts.append(acts[indices].mean(axis=0))
        else:
            concept_acts.append(np.zeros(acts.shape[1]))
    return np.vstack(concept_acts), concept_names


def main():
    # Load THINGS dataset
    print("Loading THINGS dataset...")
    targets, img_paths = load_things_data()
    concept_image_ids = targets["image_ids"]
    transform = get_transform(ds_stats="imgnet")
    loader = _make_loader(img_paths, transform, BATCH_SIZE, NUM_WORKERS)

    save_dict = {}

    for model_name, ckpt_path in CHECKPOINTS.items():
        print(f"\n--- {model_name} ---")
        model = load_model(ckpt_path)
        layer_acts, ids = extract_layers(model, loader, LAYERS)
        del model
        torch.cuda.empty_cache()

        for layer in LAYERS:
            concept_acts, concept_names = concept_average(
                layer_acts[layer], ids, concept_image_ids)
            key = f"{model_name}_{layer}"
            save_dict[key] = concept_acts.astype(np.float32)
            print(f"  {key}: {concept_acts.shape}")

    save_dict["concept_names"] = np.array(concept_names)

    out = os.path.join(OUTPUT_DIR, "activations.npz")
    np.savez_compressed(out, **save_dict)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
