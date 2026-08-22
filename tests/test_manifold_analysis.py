"""Regression tests for the paper-faithful manifold metrics."""

from __future__ import annotations

import sys
import types

import numpy as np

from experiments.manifold_analysis.manifold_capacity import manifold_capacity
from experiments.manifold_analysis.manifold_snr import manifold_snr


def _reference_snr(manifolds: np.ndarray, n_shots: int) -> np.ndarray:
    """Literal transcription of bsorsch/geometry-fewshot-learning geometry()."""
    radii, centers, axes = [], [], []
    for manifold in manifolds.transpose(0, 2, 1):
        centers.append(manifold.mean(0))
        _, singular_values, vh = np.linalg.svd(
            manifold - manifold.mean(0), full_matrices=False
        )
        radii.append(singular_values)
        axes.append(vh)
    radii, centers, axes = map(np.stack, (radii, centers, axes))

    n_categories, n_images = len(centers), radii.shape[1]
    distances = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    normalized = distances / np.sqrt(
        np.sum(radii**2, axis=-1)[:, None] / n_images
    )
    dimensions = np.sum(radii**2, axis=-1) ** 2 / np.sum(radii**4, axis=-1)

    center_self, center_other, subspace = [], [], []
    for a in range(n_categories):
        for b in range(n_categories):
            if a == b:
                center_self.append(np.nan)
                center_other.append(np.nan)
                subspace.append(np.nan)
                continue
            direction = centers[a] - centers[b]
            direction /= np.linalg.norm(direction)
            center_self.append(
                np.sum((axes[a] @ direction) ** 2 * radii[a] ** 2)
                / np.sum(radii[a] ** 2)
            )
            center_other.append(
                np.sum((axes[b] @ direction) ** 2 * radii[b] ** 2)
                / np.sum(radii[a] ** 2)
            )
            cosines = axes[a] @ axes[b].T
            subspace.append(
                np.sum(
                    cosines**2
                    * radii[a, :, None] ** 2
                    * radii[b, None, :] ** 2
                )
                / np.sum(radii[a] ** 2) ** 2
            )

    center_self, center_other, subspace = [
        np.asarray(value).reshape(n_categories, n_categories)
        for value in (center_self, center_other, subspace)
    ]
    overlap = (center_self + center_other / n_shots) * normalized**2
    bias = (
        np.sum(radii**2, axis=-1)[None, :]
        / np.sum(radii**2, axis=-1)[:, None]
        - 1
    )
    return 0.5 * (normalized**2 + bias / n_shots) / np.sqrt(
        1 / dimensions[:, None] / n_shots + overlap + subspace / n_shots
    )


def test_snr_matches_authors_reference_code() -> None:
    manifolds = np.random.default_rng(4).normal(size=(6, 23, 8))
    actual = manifold_snr(manifolds, n_shots=5)
    expected = _reference_snr(manifolds, n_shots=5)

    np.testing.assert_allclose(
        actual["pairwise"], expected, rtol=2e-13, atol=2e-13, equal_nan=True
    )
    assert actual["mean"] == np.nanmean(expected)


def test_capacity_uses_reference_harmonic_mean(monkeypatch) -> None:
    expected = np.array([0.5, 1.0, 2.0])

    def fake_reference(x, kappa, n_t):
        assert len(x) == 3
        assert all(manifold.shape == (7, 4) for manifold in x)
        assert kappa == 0
        assert n_t == 11
        return expected, np.ones(3), np.ones(3), 0.2, 1

    package = types.ModuleType("mftma")
    module = types.ModuleType("mftma.manifold_analysis_correlation")
    module.manifold_analysis_corr = fake_reference
    monkeypatch.setitem(sys.modules, "mftma", package)
    monkeypatch.setitem(sys.modules, "mftma.manifold_analysis_correlation", module)

    result = manifold_capacity(np.ones((3, 7, 4)), n_probes=11)
    assert result["mean"] == 1 / np.mean(1 / expected)
    np.testing.assert_array_equal(result["per_manifold"], expected)
