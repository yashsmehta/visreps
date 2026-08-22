"""Compute fine-manifold SNR using only distinctions within a coarse class."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("experiments/manifold_analysis/heldout_sweep_results")
LEVELS = (2, 4, 8, 16, 32, 64)


def coarse_labels(paths: list[list[str]], n_classes: int) -> np.ndarray:
    table = pd.read_csv(f"pca_labels/pca_labels_clip/n_classes_{n_classes}.csv")
    lookup = dict(zip(table["image"], table["pca_label"], strict=True))
    return np.asarray([[lookup[Path(path).name] for path in row] for row in paths])


def variable_manifold_snr(manifolds: list[np.ndarray], groups: np.ndarray, n_shots: int = 5) -> dict:
    """Sorscher SNR for variable-sized manifolds, restricted within groups."""
    centers, radii, axes, n_images = [], [], [], []
    for x in manifolds:
        centers.append(x.mean(axis=1))
        _, singular_values, vh = np.linalg.svd(
            (x - x.mean(axis=1, keepdims=True)).T, full_matrices=False
        )
        radii.append(singular_values)
        axes.append(vh)
        n_images.append(x.shape[1])

    radius_sq = np.asarray([np.sum(r**2) for r in radii])
    dimensions = np.asarray([s**2 / np.sum(r**4) for s, r in zip(radius_sq, radii)])
    directed = []
    for a in range(len(manifolds)):
        for b in range(len(manifolds)):
            if a == b or groups[a] != groups[b]:
                continue
            delta = centers[a] - centers[b]
            distance = np.linalg.norm(delta)
            if distance == 0 or radius_sq[a] <= 0 or radius_sq[b] <= 0:
                continue
            direction = delta / distance
            normalized_sq = distance**2 / (radius_sq[a] / n_images[a])
            center_self = np.sum((axes[a] @ direction) ** 2 * radii[a] ** 2) / radius_sq[a]
            center_other = np.sum((axes[b] @ direction) ** 2 * radii[b] ** 2) / radius_sq[a]
            cosines = axes[a] @ axes[b].T
            overlap = np.sum(
                cosines**2 * radii[a][:, None] ** 2 * radii[b][None, :] ** 2
            ) / radius_sq[a] ** 2
            bias = radius_sq[b] / radius_sq[a] - 1
            value = 0.5 * (normalized_sq + bias / n_shots) / np.sqrt(
                1 / dimensions[a] / n_shots
                + normalized_sq * (center_self + center_other / n_shots)
                + overlap / n_shots
            )
            directed.append(value)
    return {
        "mean": float(np.mean(directed)),
        "n_directed_pairs": len(directed),
        "n_submanifolds": len(manifolds),
    }


def compute(manifold_array: np.ndarray, labels: np.ndarray, min_images: int = 5) -> dict:
    manifolds, groups = [], []
    for group in np.unique(labels):
        for fine_idx in range(labels.shape[0]):
            keep = labels[fine_idx] == group
            if keep.sum() >= min_images:
                manifolds.append(manifold_array[fine_idx][:, keep])
                groups.append(group)
    return variable_manifold_snr(manifolds, np.asarray(groups))


def main() -> None:
    panel = json.loads((ROOT / "image_panel.json").read_text())
    labels = {n: coarse_labels(panel["image_paths"], n) for n in LEVELS}
    results = {}
    with np.load(ROOT / "cfg1000a_fc2.npz") as saved:
        fine_model = saved["fc2_post"]
    for n in LEVELS:
        with np.load(ROOT / f"cfg{n}a_fc2.npz") as saved:
            coarse_model = saved["fc2_post"]
        results[str(n)] = {
            "coarse_model": compute(coarse_model, labels[n]),
            "fine_supervised_model": compute(fine_model, labels[n]),
        }
        print(n, results[str(n)])
    (ROOT / "within_coarse_snr_seed_a.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
