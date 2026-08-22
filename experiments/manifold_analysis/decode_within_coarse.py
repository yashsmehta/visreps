"""Quick diagnostic: decode ImageNet classes within a coarse-label bucket.

This intentionally reuses the cached manifold panel. It is an exploratory
train/test split of those images, not the final paper evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler


RESULTS = Path("experiments/manifold_analysis/coarsegrain_results")


def load_panel_labels(n_coarse: int) -> tuple[np.ndarray, np.ndarray]:
    panel = json.loads((RESULTS / "image_panel.json").read_text())
    categories = np.asarray(panel["categories"])
    paths = np.asarray(panel["image_paths"])

    table = pd.read_csv(f"pca_labels/pca_labels_clip/n_classes_{n_coarse}.csv")
    label_by_name = dict(zip(table["image"], table["pca_label"], strict=True))
    coarse = np.asarray(
        [[label_by_name[Path(path).name] for path in row] for row in paths]
    )
    fine = np.repeat(categories[:, None], paths.shape[1], axis=1)
    return fine.ravel(), coarse.ravel()


def decode_model(
    model: str,
    fine: np.ndarray,
    coarse: np.ndarray,
    *,
    test_size: float,
    split_seed: int,
    min_images: int,
) -> dict[str, float | int | str]:
    with np.load(RESULTS / f"{model}_manifolds.npz") as saved:
        # Stored as categories x features x images; panel labels are category x image.
        features = saved["fc2_post"].transpose(0, 2, 1).reshape(len(fine), -1)

    correct = total = 0
    balanced_scores: list[float] = []
    chance_correct = 0.0
    n_buckets = 0

    for coarse_label in np.unique(coarse):
        bucket = coarse == coarse_label
        labels, counts = np.unique(fine[bucket], return_counts=True)
        eligible = labels[counts >= min_images]
        use = bucket & np.isin(fine, eligible)
        if len(eligible) < 2:
            continue

        indices = np.flatnonzero(use)
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=split_seed + int(coarse_label),
            stratify=fine[indices],
        )
        scaler = StandardScaler().fit(features[train_idx])
        x_train = scaler.transform(features[train_idx])
        x_test = scaler.transform(features[test_idx])
        classifier = NearestCentroid().fit(x_train, fine[train_idx])
        prediction = classifier.predict(x_test)

        correct += int(np.sum(prediction == fine[test_idx]))
        total += len(test_idx)
        balanced_scores.append(balanced_accuracy_score(fine[test_idx], prediction))
        chance_correct += len(test_idx) / len(eligible)
        n_buckets += 1

    return {
        "model": model,
        "accuracy": correct / total,
        "balanced_accuracy_bucket_mean": float(np.mean(balanced_scores)),
        "chance_accuracy": chance_correct / total,
        "n_test_images": total,
        "n_coarse_buckets": n_buckets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-coarse", type=int, default=8)
    parser.add_argument("--models", nargs="+", default=["cfg8a", "cfg1000a"])
    parser.add_argument("--test-size", type=float, default=0.4)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--min-images", type=int, default=5)
    args = parser.parse_args()

    fine, coarse = load_panel_labels(args.n_coarse)
    results = [
        decode_model(
            model,
            fine,
            coarse,
            test_size=args.test_size,
            split_seed=args.split_seed,
            min_images=args.min_images,
        )
        for model in args.models
    ]
    print(json.dumps({"n_coarse": args.n_coarse, "results": results}, indent=2))


if __name__ == "__main__":
    main()
