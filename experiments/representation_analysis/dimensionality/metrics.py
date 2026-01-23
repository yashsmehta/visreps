"""
Pure metric computation functions for dimensionality analysis.

All functions take numpy arrays and return numeric results.
No side effects, no printing, no plotting.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def eigenspectrum(X):
    """Compute eigenvalues of the covariance matrix.

    Uses Gram matrix trick when n_features > n_samples for efficiency.

    Args:
        X: Feature matrix (n_samples, n_features)

    Returns:
        Eigenvalues in descending order
    """
    X = X - X.mean(axis=0)
    n_samples, n_features = X.shape

    if n_features > n_samples:
        # Gram matrix: same non-zero eigenvalues, smaller matrix
        M = X @ X.T / (n_samples - 1)
    else:
        M = np.cov(X, rowvar=False)

    eigenvalues = np.linalg.eigvalsh(M)[::-1]
    return np.maximum(eigenvalues, 0)  # Numerical stability


def participation_ratio(X):
    """Compute participation ratio (effective dimensionality).

    PR = (sum(lambda))^2 / sum(lambda^2)

    Intuition: If variance is spread across d dimensions equally,
    PR = d. If concentrated in one dimension, PR = 1.

    Args:
        X: Feature matrix (n_samples, n_features)

    Returns:
        Participation ratio (float)
    """
    eigs = eigenspectrum(X)
    total = eigs.sum()
    if total == 0:
        return 0.0
    return (total ** 2) / (eigs ** 2).sum()


def cumulative_variance(X):
    """Compute cumulative variance explained by principal components.

    Args:
        X: Feature matrix (n_samples, n_features)

    Returns:
        Array of cumulative variance explained (0 to 1)
    """
    eigs = eigenspectrum(X)
    total = eigs.sum()
    if total == 0:
        return np.zeros_like(eigs)
    return np.cumsum(eigs / total)


def n_components_for_variance(X, threshold=0.9):
    """Number of components needed to explain threshold variance.

    Args:
        X: Feature matrix (n_samples, n_features)
        threshold: Variance threshold (default 0.9 = 90%)

    Returns:
        Number of components (int)
    """
    cumvar = cumulative_variance(X)
    return int(np.searchsorted(cumvar, threshold) + 1)


def two_nn_dimension(X, n_samples=None, seed=42):
    """Estimate intrinsic dimension using Two-NN method (Facco et al., 2017).

    Uses ratio of distances to 1st and 2nd nearest neighbors.
    MLE estimator: d = n / sum(log(mu)) where mu = r2/r1.

    Args:
        X: Feature matrix (n_samples, n_features)
        n_samples: Subsample size for speed (None = use all)
        seed: Random seed for subsampling

    Returns:
        (dimension, std_error) tuple
    """
    rng = np.random.default_rng(seed)

    if n_samples is not None and len(X) > n_samples:
        idx = rng.choice(len(X), n_samples, replace=False)
        X = X[idx]

    # Center (no normalization - preserves manifold geometry)
    X = X - X.mean(axis=0)

    # Find 2 nearest neighbors (excluding self)
    nn = NearestNeighbors(n_neighbors=3, algorithm='auto', n_jobs=-1)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)

    r1, r2 = distances[:, 1], distances[:, 2]

    # mu = r2/r1 where r1 > 0 and mu >= 1
    valid = r1 > 1e-10
    mu = r2[valid] / r1[valid]
    mu = mu[mu >= 1.0]

    if len(mu) < 10:
        return np.nan, np.nan

    # MLE estimate
    log_mu = np.log(mu)
    n = len(mu)
    dimension = n / log_mu.sum()

    # Bootstrap standard error
    boot_idx = rng.choice(n, (100, n), replace=True)
    boot_dims = n / log_mu[boot_idx].sum(axis=1)
    std_error = np.std(boot_dims)

    return dimension, std_error


def hoyer_sparsity(X):
    """Compute Hoyer sparsity for each sample.

    S = (sqrt(n) - L1/L2) / (sqrt(n) - 1)

    Range: 0 (uniform) to 1 (maximally sparse).

    Args:
        X: Feature matrix (n_samples, n_features)

    Returns:
        Array of sparsity values per sample
    """
    n_features = X.shape[1]
    sqrt_n = np.sqrt(n_features)

    X_abs = np.abs(X)
    l1 = X_abs.sum(axis=1)
    l2 = np.linalg.norm(X_abs, axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        sparsity = (sqrt_n - l1 / l2) / (sqrt_n - 1)

    return np.where(l2 < 1e-10, 1.0, sparsity)


def fraction_active(X, threshold=0):
    """Fraction of neurons active (> threshold) per sample.

    Args:
        X: Feature matrix (n_samples, n_features)
        threshold: Activation threshold (default 0)

    Returns:
        Array of fraction active per sample
    """
    return np.mean(np.abs(X) > threshold, axis=1)
