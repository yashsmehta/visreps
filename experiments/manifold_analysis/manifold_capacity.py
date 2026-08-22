"""Replica mean-field manifold capacity from Chung et al. (2018)."""

from __future__ import annotations

import inspect

import numpy as np
from numpy.typing import ArrayLike


def manifold_capacity(
    manifolds: ArrayLike,
    *,
    margin: float = 0.0,
    n_probes: int = 200,
    seed: int = 0,
) -> dict[str, object]:
    """Compute correlated-manifold classification capacity.

    This calls the exact implementation cited by the CCN paper rather than a
    numerically different reimplementation. From the repository root, install
    the experiment dependencies with::

        uv sync --extra manifold

    Args:
        manifolds: Array shaped (n_categories, n_features, n_images).
        margin: Classification margin. The paper uses zero.
        n_probes: Gaussian probes per manifold. The reference default is 200.
        seed: Seed for factor analysis and Gaussian probes.

    Returns:
        ``mean`` is the required harmonic mean of per-manifold capacities.
    """
    x = np.asarray(manifolds, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("manifolds must have shape (categories, features, images)")
    if x.shape[0] < 2 or x.shape[2] < 2:
        raise ValueError("at least two categories and two images are required")
    if x.shape[1] <= x.shape[2]:
        raise ValueError("capacity requires n_features > n_images")
    if not np.isfinite(x).all():
        raise ValueError("manifolds contains NaN or infinite values")
    if margin < 0:
        raise ValueError("margin must be nonnegative")
    if not isinstance(n_probes, (int, np.integer)) or n_probes < 1:
        raise ValueError("n_probes must be a positive integer")

    # The reference package targets older Python/SciPy APIs. These aliases only
    # restore removed names; they do not alter the numerical implementation.
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    import scipy.misc
    import scipy.special

    if not hasattr(scipy.misc, "comb"):
        scipy.misc.comb = scipy.special.comb  # type: ignore[attr-defined]

    try:
        from mftma.manifold_analysis_correlation import manifold_analysis_corr
    except ImportError as error:
        raise ImportError(
            "manifold capacity requires the authors' neural_manifolds_replicaMFT "
            "package; see this function's docstring for the install command"
        ) from error

    random_state = np.random.get_state()
    try:
        np.random.seed(seed)
        capacity, radius, dimension, center_correlation, center_rank = (
            manifold_analysis_corr(
                [category for category in x],
                kappa=float(margin),
                n_t=int(n_probes),
            )
        )
    finally:
        np.random.set_state(random_state)
    capacity = np.asarray(capacity, dtype=np.float64)
    if np.any(~np.isfinite(capacity)) or np.any(capacity <= 0):
        raise RuntimeError("reference implementation returned invalid capacities")

    return {
        "mean": float(1 / np.mean(1 / capacity)),
        "per_manifold": capacity,
        "radius": np.asarray(radius),
        "dimension": np.asarray(dimension),
        "center_correlation": float(center_correlation),
        "center_rank": int(center_rank),
    }
