"""Evaluate fine-category structure using only the held-out ImageNet split.

Creates a fixed 100-category x 50-image panel from the deterministic 20% split,
extracts FC2 representations, and reports fine-category SNR plus linear decoding
within the supplied coarse-label buckets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from experiments.manifold_analysis.manifold_snr import manifold_snr
from experiments.manifold_analysis.run_coarsegrain_manifolds import (
    load_model,
    make_projection,
)
from visreps.dataloaders.obj_cls import get_transform


OUTPUT = Path("experiments/manifold_analysis/heldout_sweep_results")


def build_panel(
    manifest_path: Path, n_categories: int, seed: int
) -> tuple[list[str], list[list[str]]]:
    """Sample categories from the fixed 1,000-category held-out manifest."""
    manifest = json.loads(manifest_path.read_text())
    available = sorted(manifest["categories"])
    rng = np.random.default_rng(seed)
    categories = sorted(rng.choice(available, n_categories, replace=False).tolist())
    paths = [manifest["categories"][category] for category in categories]
    return categories, paths


@torch.inference_mode()
def extract_fc2(
    model, paths: list[list[str]], projection: torch.Tensor, device: torch.device,
    batch_size: int,
) -> np.ndarray:
    transform = get_transform(data_augment=False)
    flat = [path for category in paths for path in category]
    values = []
    for start in tqdm(range(0, len(flat), batch_size), desc="FC2 extraction"):
        images = torch.stack([
            transform(Image.open(path).convert("RGB"))
            for path in flat[start : start + batch_size]
        ]).to(device)
        x = model.features(images)
        x = model.adaptive_pool(x).flatten(1)
        x = model.classifier[:8](x)  # FC2 batch norm + ReLU
        values.append((x @ projection).cpu())
    n_categories, n_images = len(paths), len(paths[0])
    return torch.cat(values).reshape(n_categories, n_images, 2048).permute(0, 2, 1).numpy()


def panel_labels(
    categories: list[str], paths: list[list[str]], n_coarse: int
) -> tuple[np.ndarray, np.ndarray]:
    table = pd.read_csv(f"pca_labels/pca_labels_clip/n_classes_{n_coarse}.csv")
    label_by_name = dict(zip(table["image"], table["pca_label"], strict=True))
    fine = np.repeat(np.asarray(categories)[:, None], len(paths[0]), axis=1).ravel()
    coarse = np.asarray([
        [label_by_name[Path(path).name] for path in row] for row in paths
    ]).ravel()
    return fine, coarse


def decode(
    manifolds: np.ndarray, fine: np.ndarray, coarse: np.ndarray, seed: int,
    min_images: int,
) -> dict[str, float | int]:
    features = manifolds.transpose(0, 2, 1).reshape(len(fine), -1)
    correct = total = 0
    chance_correct = 0.0
    bucket_balanced = []
    eligible_cells = 0
    for coarse_label in np.unique(coarse):
        bucket = coarse == coarse_label
        labels, counts = np.unique(fine[bucket], return_counts=True)
        eligible = labels[counts >= min_images]
        use = bucket & np.isin(fine, eligible)
        if len(eligible) < 2:
            continue
        eligible_cells += len(eligible)
        indices = np.flatnonzero(use)
        train_idx, test_idx = train_test_split(
            indices, test_size=0.4, random_state=seed + int(coarse_label),
            stratify=fine[indices],
        )
        scaler = StandardScaler().fit(features[train_idx])
        # Euclidean nearest-centroid classification has linear pairwise decision
        # boundaries and directly tests the prototype geometry summarized by SNR.
        classifier = NearestCentroid().fit(
            scaler.transform(features[train_idx]), fine[train_idx]
        )
        prediction = classifier.predict(scaler.transform(features[test_idx]))
        correct += int(np.sum(prediction == fine[test_idx]))
        total += len(test_idx)
        chance_correct += len(test_idx) / len(eligible)
        bucket_balanced.append(balanced_accuracy_score(fine[test_idx], prediction))
    return {
        "accuracy": correct / total if total else None,
        "balanced_accuracy_bucket_mean": float(np.mean(bucket_balanced)) if total else None,
        "chance_accuracy": chance_correct / total if total else None,
        "n_test_images": total,
        "n_eligible_category_bucket_cells": eligible_cells,
        "n_usable_coarse_buckets": len(bucket_balanced),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path(
        "experiments/manifold_analysis/heldout_dataset/manifest.json"
    ))
    parser.add_argument("--models", nargs="+", default=[
        *[f"cfg{n}a" for n in (2, 4, 8, 16, 32, 64)],
        "cfg1000a",
    ])
    parser.add_argument("--coarseness", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    parser.add_argument("--n-categories", type=int, default=100)
    parser.add_argument("--images-per-category", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-images", type=int, default=5)
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel_path = OUTPUT / "image_panel.json"
    if panel_path.exists():
        panel = json.loads(panel_path.read_text())
        categories, paths = panel["categories"], panel["image_paths"]
    else:
        categories, paths = build_panel(args.manifest, args.n_categories, args.seed)
        panel_path.write_text(json.dumps({"categories": categories, "image_paths": paths}, indent=2))

    labels_by_coarseness = {
        n: panel_labels(categories, paths, n) for n in args.coarseness
    }
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    projection = make_projection(4096, seed=102, device=device)
    results_path = OUTPUT / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for model_name in args.models:
        cache = OUTPUT / f"{model_name}_fc2.npz"
        if cache.exists():
            with np.load(cache) as saved:
                manifolds = saved["fc2_post"]
        else:
            n_classes = int(model_name[3:-1])
            seed_letter = model_name[-1]
            model, _ = load_model(n_classes, seed_letter, device)
            manifolds = extract_fc2(model, paths, projection, device, args.batch_size)
            np.savez_compressed(cache, fc2_post=manifolds)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        snr = manifold_snr(manifolds, n_shots=5)
        n_classes = int(model_name[3:-1])
        grouping_levels = args.coarseness if n_classes == 1000 else [n_classes]
        results[model_name] = {
            "snr": float(snr["mean"]),
            "decoding_by_grouping": {
                str(n): decode(
                    manifolds, *labels_by_coarseness[n], args.seed, args.min_images
                )
                for n in grouping_levels
            },
        }
        results_path.write_text(json.dumps(results, indent=2))
        print(model_name, json.dumps(results[model_name], indent=2))


if __name__ == "__main__":
    main()
