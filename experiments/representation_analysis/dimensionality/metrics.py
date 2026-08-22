"""
Pure metric computation functions for dimensionality analysis.

All functions take numpy arrays and return numeric results.
No side effects, no printing, no plotting.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _l2_normalize_rows(X, eps=1e-12):
    """L2-normalize each row of X to unit norm."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + eps)


def eigenspectrum(X, sample_normalize=True):
    """Compute eigenvalues of the covariance matrix.

    Uses Gram matrix trick when n_features > n_samples for efficiency.

    Args:
        X: Feature matrix (n_samples, n_features)
        sample_normalize: If True, L2-normalize each row to unit norm before
            centering. Removes the per-sample magnitude scale so the
            eigenspectrum reflects the *shape* of representations on the
            unit sphere, independent of activation magnitude. This is the
            right thing for cross-model comparisons when models differ in
            saturation behavior (e.g. extreme post-BN activations on a
            small fraction of inputs).

    Returns:
        Eigenvalues in descending order (non-negative).
    """
    if sample_normalize:
        X = _l2_normalize_rows(X)
    X = X - X.mean(axis=0)
    n_samples, n_features = X.shape

    if n_features > n_samples:
        M = X @ X.T / (n_samples - 1)
    else:
        M = np.cov(X, rowvar=False)

    eigenvalues = np.linalg.eigvalsh(M)[::-1]
    return np.maximum(eigenvalues, 0)


def participation_ratio(X, sample_normalize=True):
    """Compute participation ratio (effective dimensionality).

    PR = (sum(lambda))^2 / sum(lambda^2)

    Intuition: If variance is spread across d dimensions equally,
    PR = d. If concentrated in one dimension, PR = 1.

    Args:
        X: Feature matrix (n_samples, n_features)
        sample_normalize: see eigenspectrum().

    Returns:
        Participation ratio (float)
    """
    eigs = eigenspectrum(X, sample_normalize=sample_normalize)
    total = eigs.sum()
    if total == 0:
        return 0.0
    return (total ** 2) / (eigs ** 2).sum()


def cumulative_variance(X, sample_normalize=True):
    """Cumulative variance explained by principal components."""
    eigs = eigenspectrum(X, sample_normalize=sample_normalize)
    total = eigs.sum()
    if total == 0:
        return np.zeros_like(eigs)
    return np.cumsum(eigs / total)


def n_components_for_variance(X, threshold=0.9, sample_normalize=True):
    """Number of components needed to explain `threshold` variance."""
    cumvar = cumulative_variance(X, sample_normalize=sample_normalize)
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


def power_law_exponent(eigs, rank_min=5, rank_max_frac=0.5, rank_max_abs=500,
                       min_ratio=1e-6):
    """Fit power-law decay to eigenspectrum: lambda_n ~ n^(-alpha).

    Fits a line to log(lambda_n) vs log(n) over a mid-range of ranks, skipping
    the noisy head (first few PCs) and the floating-point noise floor in the tail.
    Ranks where lambda_n < min_ratio * lambda_1 are discarded as noise.

    Args:
        eigs: Eigenvalues in descending order (1D array)
        rank_min: Smallest rank to include in fit (1-indexed). Skips noisy top PCs.
        rank_max_frac: Use at most this fraction of total ranks.
        rank_max_abs: Hard cap on largest rank used.
        min_ratio: Drop ranks where lambda_n / lambda_1 < min_ratio (noise floor).

    Returns:
        Dict with: 'alpha' (slope magnitude), 'intercept', 'r2',
        'rank_min', 'rank_max', 'n_fit' (#points actually used).
    """
    eigs = np.asarray(eigs)
    n = len(eigs)
    rank_max = min(int(n * rank_max_frac), rank_max_abs, n)
    bad = {'alpha': np.nan, 'intercept': np.nan, 'r2': np.nan,
           'rank_min': rank_min, 'rank_max': rank_max, 'n_fit': 0}

    if rank_max <= rank_min + 2 or eigs[0] <= 0:
        return bad

    ranks = np.arange(rank_min, rank_max + 1)
    vals = eigs[rank_min - 1:rank_max]

    # Drop ranks below the noise floor (cliffs to ~0 after a few modes are
    # common for over-trained classifiers; fitting there is meaningless).
    keep = vals > (min_ratio * eigs[0])
    if keep.sum() < 5:
        return bad

    log_n = np.log(ranks[keep])
    log_l = np.log(vals[keep])

    slope, intercept = np.polyfit(log_n, log_l, 1)
    pred = slope * log_n + intercept
    ss_res = np.sum((log_l - pred) ** 2)
    ss_tot = np.sum((log_l - log_l.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {'alpha': -slope, 'intercept': intercept, 'r2': r2,
            'rank_min': rank_min, 'rank_max': int(ranks[keep].max()),
            'n_fit': int(keep.sum())}
