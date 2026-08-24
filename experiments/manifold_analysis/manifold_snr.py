"""Few-shot manifold geometry from Sorscher, Ganguli & Sompolinsky (2022)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ManifoldGeometry:
    """PCA geometry shared by all shot counts for a set of manifolds."""

    centers: FloatArray
    radii: FloatArray
    axes: FloatArray
    radius_sq_sum: FloatArray
    dimension: FloatArray
    n_images: int


def fit_manifold_geometry(manifolds: ArrayLike) -> ManifoldGeometry:
    """Estimate centroids, PCA axes/radii, RMS radius, and PR dimension."""
    x = np.asarray(manifolds, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("manifolds must have shape (categories, features, images)")
    if x.shape[0] < 2 or x.shape[2] < 2:
        raise ValueError("at least two categories and two images are required")
    if not np.isfinite(x).all():
        raise ValueError("manifolds contains NaN or infinite values")

    centers = x.mean(axis=2)
    radii, axes = [], []
    for category in x:
        _, singular_values, vh = np.linalg.svd(
            (category - category.mean(axis=1, keepdims=True)).T,
            full_matrices=False,
        )
        radii.append(singular_values)
        axes.append(vh)
    radii_array = np.stack(radii)
    axes_array = np.stack(axes)
    radius_sq_sum = np.sum(radii_array**2, axis=1)
    if np.any(radius_sq_sum <= 0):
        raise ValueError("every category must have nonzero within-category variance")
    dimension = radius_sq_sum**2 / np.sum(radii_array**4, axis=1)
    return ManifoldGeometry(
        centers=centers,
        radii=radii_array,
        axes=axes_array,
        radius_sq_sum=radius_sq_sum,
        dimension=dimension,
        n_images=x.shape[2],
    )


def fit_pairwise_geometry(geometry: ManifoldGeometry) -> dict[str, FloatArray]:
    """Compute directed, shot-independent pair geometry once."""
    centers, radii, axes = geometry.centers, geometry.radii, geometry.axes
    radius_sq_sum = geometry.radius_sq_sum
    n_categories = len(centers)
    delta = centers[:, None, :] - centers[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    normalized_distance_sq = distance**2 / (
        radius_sq_sum[:, None] / geometry.n_images
    )

    signal_self = np.full((n_categories, n_categories), np.nan)
    signal_other = np.full_like(signal_self, np.nan)
    noise_noise = np.full_like(signal_self, np.nan)
    for a in range(n_categories):
        for b in range(n_categories):
            if a == b:
                continue
            if distance[a, b] == 0:
                raise ValueError(f"categories {a} and {b} have identical centers")
            direction = delta[a, b] / distance[a, b]
            signal_self[a, b] = np.sum(
                (axes[a] @ direction) ** 2 * radii[a] ** 2
            ) / radius_sq_sum[a]
            signal_other[a, b] = np.sum(
                (axes[b] @ direction) ** 2 * radii[b] ** 2
            ) / radius_sq_sum[a]
            cosines = axes[a] @ axes[b].T
            noise_noise[a, b] = np.sum(
                cosines**2
                * radii[a, :, None] ** 2
                * radii[b, None, :] ** 2
            ) / radius_sq_sum[a] ** 2

    return {
        "signal": normalized_distance_sq,
        "bias": radius_sq_sum[None, :] / radius_sq_sum[:, None] - 1,
        "signal_overlap_self": signal_self,
        "signal_overlap_other": signal_other,
        "noise_noise_overlap": noise_noise,
    }


def snr_from_geometry(
    geometry: ManifoldGeometry,
    n_shots: int = 5,
    *,
    pair_geometry: dict[str, FloatArray] | None = None,
) -> dict[str, object]:
    """Compute directed SNR, optionally reusing shot-independent pair terms."""
    if not isinstance(n_shots, (int, np.integer)) or n_shots < 1:
        raise ValueError("n_shots must be a positive integer")
    terms = fit_pairwise_geometry(geometry) if pair_geometry is None else pair_geometry
    signal = terms["signal"].copy()
    bias = terms["bias"].copy()
    signal_self = terms["signal_overlap_self"].copy()
    signal_other = terms["signal_overlap_other"].copy()
    noise_noise = terms["noise_noise_overlap"].copy()
    n_categories = len(geometry.centers)
    dimension_noise = np.broadcast_to(
        1 / geometry.dimension[:, None] / n_shots,
        (n_categories, n_categories),
    ).copy()
    signal_noise_self = signal * signal_self
    signal_noise_other = signal * signal_other / n_shots
    noise_noise_scaled = noise_noise / n_shots
    numerator = 0.5 * (signal + bias / n_shots)
    denominator = np.sqrt(
        dimension_noise + signal_noise_self + signal_noise_other + noise_noise_scaled
    )
    # Keep the operation ordering of the released reference implementation;
    # besides reproducibility, the regression test checks its exact mean.
    pairwise = 0.5 * (signal + bias / n_shots) / np.sqrt(
        1 / geometry.dimension[:, None] / n_shots
        + signal * (signal_self + signal_other / n_shots)
        + noise_noise / n_shots
    )
    for value in (
        signal,
        bias,
        dimension_noise,
        signal_self,
        signal_other,
        signal_noise_self,
        signal_noise_other,
        noise_noise,
        noise_noise_scaled,
        pairwise,
    ):
        np.fill_diagonal(value, np.nan)

    return {
        "mean": float(np.nanmean(pairwise)),
        "pairwise": pairwise,
        "predicted_error": ndtr(-pairwise),
        "signal": signal,
        "bias": bias,
        "dimension": geometry.dimension.copy(),
        "dimension_noise": dimension_noise,
        "signal_overlap_self": signal_self,
        "signal_overlap_other": signal_other,
        "signal_noise_self": signal_noise_self,
        "signal_noise_other": signal_noise_other,
        "noise_noise_overlap": noise_noise,
        "noise_noise": noise_noise_scaled,
        "radius": np.sqrt(geometry.radius_sq_sum / geometry.n_images),
        "numerator": numerator,
        "denominator": denominator,
    }


def manifold_snr(manifolds: ArrayLike, n_shots: int = 5) -> dict[str, object]:
    """Fit manifold geometry and compute all ordered-pair few-shot SNRs."""
    return snr_from_geometry(fit_manifold_geometry(manifolds), n_shots=n_shots)


def empirical_nearest_prototype_error(
    manifolds: ArrayLike,
    pairs: ArrayLike,
    *,
    n_shots: int = 5,
    n_trials: int = 100,
    seed: int = 0,
) -> FloatArray:
    """Estimate directed two-class nearest-prototype error for selected pairs."""
    x = np.asarray(manifolds, dtype=np.float64)
    pair_array = np.asarray(pairs, dtype=int)
    if x.ndim != 3 or pair_array.ndim != 2 or pair_array.shape[1] != 2:
        raise ValueError("expected manifolds (classes, features, images) and pairs (N, 2)")
    if n_shots < 1 or n_shots >= x.shape[2]:
        raise ValueError("n_shots must be positive and smaller than images per class")
    if n_trials < 1:
        raise ValueError("n_trials must be positive")
    if np.any(pair_array < 0) or np.any(pair_array >= x.shape[0]):
        raise ValueError("pair index is outside the manifold array")

    rng = np.random.default_rng(seed)
    errors = np.empty(len(pair_array), dtype=np.float64)
    all_indices = np.arange(x.shape[2])
    for pair_index, (a, b) in enumerate(pair_array):
        if a == b:
            raise ValueError("empirical validation requires distinct classes")
        mistakes = total = 0
        for _ in range(n_trials):
            train_a = rng.choice(all_indices, n_shots, replace=False)
            train_b = rng.choice(all_indices, n_shots, replace=False)
            prototype_a = x[a][:, train_a].mean(axis=1)
            prototype_b = x[b][:, train_b].mean(axis=1)
            test = x[a][:, np.setdiff1d(all_indices, train_a, assume_unique=True)].T
            distance_a = np.sum((test - prototype_a) ** 2, axis=1)
            distance_b = np.sum((test - prototype_b) ** 2, axis=1)
            mistakes += int(np.count_nonzero(distance_b <= distance_a))
            total += len(test)
        errors[pair_index] = mistakes / total
    return errors
