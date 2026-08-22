"""Few-shot manifold signal-to-noise ratio from Sorscher et al. (2022)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def manifold_snr(manifolds: ArrayLike, n_shots: int = 5) -> dict[str, object]:
    """Compute directed pairwise SNRs and their mean.

    Args:
        manifolds: Array shaped (n_categories, n_features, n_images).
        n_shots: Number of examples per category in the hypothetical prototype
            classifier. The defining DNN analyses use five.

    Returns:
        ``mean`` is the mean over all ordered category pairs. ``pairwise`` is
        the directed (n_categories, n_categories) matrix with a NaN diagonal.
    """
    x = np.asarray(manifolds, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("manifolds must have shape (categories, features, images)")
    if x.shape[0] < 2 or x.shape[2] < 2:
        raise ValueError("at least two categories and two images are required")
    if not isinstance(n_shots, (int, np.integer)) or n_shots < 1:
        raise ValueError("n_shots must be a positive integer")
    if not np.isfinite(x).all():
        raise ValueError("manifolds contains NaN or infinite values")

    n_categories, _, n_images = x.shape
    centers = x.mean(axis=2)

    # The reference code applies SVD to (images, features). Vh therefore holds
    # feature-space principal axes and s holds the within-manifold radii.
    radii = []
    axes = []
    for category in x:
        _, singular_values, vh = np.linalg.svd(
            (category - category.mean(axis=1, keepdims=True)).T,
            full_matrices=False,
        )
        radii.append(singular_values)
        axes.append(vh)
    radii = np.stack(radii)
    axes = np.stack(axes)

    radius_sq_sum = np.sum(radii**2, axis=1)
    if np.any(radius_sq_sum <= 0):
        raise ValueError("every category must have nonzero within-category variance")
    dimensions = radius_sq_sum**2 / np.sum(radii**4, axis=1)

    center_delta = centers[:, None, :] - centers[None, :, :]
    center_distance = np.linalg.norm(center_delta, axis=2)
    normalized_distance = center_distance / np.sqrt(
        radius_sq_sum[:, None] / n_images
    )

    center_self = np.full((n_categories, n_categories), np.nan)
    center_other = np.full_like(center_self, np.nan)
    subspace_overlap = np.full_like(center_self, np.nan)

    for a in range(n_categories):
        for b in range(n_categories):
            if a == b:
                continue
            if center_distance[a, b] == 0:
                raise ValueError(f"categories {a} and {b} have identical centers")

            direction = center_delta[a, b] / center_distance[a, b]
            center_self[a, b] = np.sum(
                (axes[a] @ direction) ** 2 * radii[a] ** 2
            ) / radius_sq_sum[a]
            center_other[a, b] = np.sum(
                (axes[b] @ direction) ** 2 * radii[b] ** 2
            ) / radius_sq_sum[a]

            axis_cosines = axes[a] @ axes[b].T
            subspace_overlap[a, b] = np.sum(
                axis_cosines**2
                * radii[a, :, None] ** 2
                * radii[b, None, :] ** 2
            ) / radius_sq_sum[a] ** 2

    signal_noise_overlap = normalized_distance**2 * (
        center_self + center_other / n_shots
    )
    bias = radius_sq_sum[None, :] / radius_sq_sum[:, None] - 1
    pairwise = 0.5 * (normalized_distance**2 + bias / n_shots) / np.sqrt(
        1 / dimensions[:, None] / n_shots
        + signal_noise_overlap
        + subspace_overlap / n_shots
    )
    np.fill_diagonal(pairwise, np.nan)

    return {
        "mean": float(np.nanmean(pairwise)),
        "pairwise": pairwise,
        "dimension": dimensions,
        "radius": np.sqrt(radius_sq_sum / n_images),
    }
