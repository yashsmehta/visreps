"""Encoding score: Ridge regression for voxelwise neural prediction (himalaya).

For voxelwise datasets (NSD fMRI, TVSD electrophysiology) only — not applicable
to behavioral embeddings (THINGS). The metric is always Pearson r between
predicted and actual voxel responses (not configurable like RSA's compare_method).

Pipeline: select best layer on train, evaluate on test, optional bootstrap CIs.
"""
from __future__ import annotations

import logging
from typing import Dict, List, TYPE_CHECKING

import numpy as np
import torch
from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV
from himalaya.scoring import correlation_score
from visreps.utils import rprint

if TYPE_CHECKING:
    from visreps.analysis.alignment import AlignmentData

logger = logging.getLogger(__name__)


def _znorm(X, mean, std):
    """Z-normalize using precomputed statistics."""
    return (X - mean) / std


def _znorm_fit(X):
    """Z-normalize X using its own stats. Returns (normalized, mean, std)."""
    mean = X.mean(dim=0)
    std = X.std(dim=0) + 1e-8
    return _znorm(X, mean, std), mean, std


def _flatten_to_cpu(acts):
    """Flatten 4D→2D and ensure CPU float32. Returns a new dict (no mutation)."""
    return {
        layer: (a.flatten(start_dim=1) if a.ndim > 2 else a).cpu().float()
        for layer, a in acts.items()
    }


def _fit_and_score(X_tr, Y_tr, X_te, Y_te, alphas, backend):
    """Fit RidgeCV on train, predict on test, return (predictions, mean Pearson r).

    Accepts X_te as a CPU tensor when needed to reduce GPU peak memory.
    Moves X_te to GPU only after fit completes and X_tr can be freed.
    """
    # fit_intercept=False because data is already z-normalized (zero mean).
    # Avoids himalaya's internal X_offset copy which doubles GPU memory.
    model = RidgeCV(alphas=alphas, cv=5, fit_intercept=False)
    model.fit(X_tr, Y_tr)
    # Move X_te to GPU for prediction (X_tr may still be held by caller)
    if not hasattr(X_te, 'device') or X_te.device.type == 'cpu':
        X_te = backend.asarray(X_te)
    pred = model.predict(X_te)
    score = float(correlation_score(Y_te, pred).mean())
    return pred, score


def compute_encoding_score(
    selection: "AlignmentData",
    evaluation: "AlignmentData",
    bootstrap: bool = True,
    n_bootstrap: int = 1000,
    seed: int = 42,
    verbose: bool = False,
    reconstruct_pca_k: int | None = None,
) -> List[Dict]:
    """Train/test encoding score: select best layer on train, evaluate on test.

    For voxelwise datasets (NSD, TVSD) only. The metric is always Pearson r
    between predicted and actual voxel responses — ``compare_method`` is
    hardcoded to ``"pearson"`` (unlike RSA where it's configurable).

    1. Split train 80/20 into fit/val (seeded). For each layer, fit RidgeCV on
       fit portion, score on val portion (mean Pearson r). Pick best layer.
       Y is normalized with fit-only stats (no leakage into val).
    2. Refit RidgeCV for best layer on FULL train data.
    3. Predict on test, compute mean Pearson r as point estimate.
    4. If bootstrap, subsample 90% of test predictions/targets n_bootstrap
       times and recompute correlation per subsample for 95% CIs.

    Does NOT mutate the input AlignmentData objects.

    Args:
        selection: AlignmentData with train activations and neural data.
        evaluation: AlignmentData with test activations and neural data.
        bootstrap: Whether to compute bootstrap CIs on the test set.
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed for train split and bootstrap.
        verbose: Print progress details.
        reconstruct_pca_k: If not None, reconstruct the best layer's activations
            from this many PCs after layer selection (PCA fit on train only).

    Returns:
        Single-element list with result dict: layer, compare_method ("pearson"),
        score, ci_low, ci_high, analysis, layer_selection_scores, and
        bootstrap_scores (when bootstrap=True).
    """
    backend = set_backend("torch_cuda", on_error="warn")
    compare_method = "pearson"  # Encoding score is always Pearson r (not configurable)
    rng = np.random.RandomState(seed)
    alphas = np.logspace(-10, 10, 20)

    # Flatten & copy to CPU (no mutation of inputs)
    train_acts = _flatten_to_cpu(selection.activations)
    test_acts = _flatten_to_cpu(evaluation.activations)
    Y_train_raw = selection.neural.cpu().float()
    Y_test_raw = evaluation.neural.cpu().float()

    n_train = Y_train_raw.size(0)
    n_test = Y_test_raw.size(0)
    n_voxels = Y_train_raw.size(1)

    if verbose:
        rprint(
            f"Train/test encoding: {n_train} train, {n_test} test, "
            f"{n_voxels} voxels (backend={backend.name})",
            style="info",
        )

    torch.cuda.empty_cache()

    # ── 1. Layer selection on train (80/20 fit/val split) ─────
    split = int(0.8 * n_train)
    perm = rng.permutation(n_train)
    fit_idx, val_idx = perm[:split], perm[split:]

    # Z-normalize Y using FIT-ONLY stats (no leakage into val)
    Y_fit_normed, Y_fit_mean, Y_fit_std = _znorm_fit(Y_train_raw[fit_idx])
    Y_fit_gpu = backend.asarray(Y_fit_normed)
    Y_val_gpu = backend.asarray(_znorm(Y_train_raw[val_idx], Y_fit_mean, Y_fit_std))

    selection_scores = []
    best_layer, best_score = None, -float("inf")

    for layer, acts_cpu in train_acts.items():
        n_features = acts_cpu.size(1)
        X_fit_normed, fit_mean, fit_std = _znorm_fit(acts_cpu[fit_idx])
        X_val_normed = _znorm(acts_cpu[val_idx], fit_mean, fit_std)

        X_fit_gpu = backend.asarray(X_fit_normed)
        del X_fit_normed

        _, score = _fit_and_score(X_fit_gpu, Y_fit_gpu, X_val_normed, Y_val_gpu, alphas, backend)
        selection_scores.append({"layer": layer, "score": score})

        if verbose:
            rprint(f"  [select] {layer:<15} r={score:.4f}  ({n_features} features)", style="info")
        if score > best_score:
            best_score = score
            best_layer = layer

        del X_fit_gpu, X_val_normed
        torch.cuda.empty_cache()

    del Y_fit_gpu, Y_val_gpu

    best_n_features = train_acts[best_layer].size(1)
    if verbose:
        rprint(
            f"  Best layer: {best_layer} (val r={best_score:.4f}, "
            f"{best_n_features} features, {n_voxels} voxels, "
            f"alphas=[{alphas[0]:.0e}, {alphas[-1]:.0e}])",
            style="highlight",
        )

    # ── 1b. Optional: reconstruct best layer from k PCs (train-fitted) ─
    if reconstruct_pca_k is not None:
        from sklearn.decomposition import PCA as _PCA

        rprint(
            f"  Reconstructing {best_layer} from {reconstruct_pca_k} PCs (train-fitted)",
            style="info",
        )
        train_np = train_acts[best_layer].numpy()
        test_np = test_acts[best_layer].numpy()
        _pca = _PCA(n_components=min(reconstruct_pca_k, train_np.shape[1]))
        _pca.fit(train_np)
        train_acts[best_layer] = torch.from_numpy(
            _pca.inverse_transform(_pca.transform(train_np)).astype(np.float32)
        )
        test_acts[best_layer] = torch.from_numpy(
            _pca.inverse_transform(_pca.transform(test_np)).astype(np.float32)
        )
        del train_np, test_np

    # ── 2. Refit best layer on full train, evaluate on test ───
    # Z-normalize X with full-train stats
    X_train_normed, train_mean, train_std = _znorm_fit(train_acts[best_layer])
    X_train_gpu = backend.asarray(X_train_normed)
    del X_train_normed

    X_test_normed_cpu = _znorm(test_acts[best_layer], train_mean, train_std)
    del train_mean, train_std

    # Z-normalize Y with full-train stats for final evaluation
    Y_train_normed, Y_mean, Y_std = _znorm_fit(Y_train_raw)
    Y_train_gpu = backend.asarray(Y_train_normed)
    Y_test_gpu = backend.asarray(_znorm(Y_test_raw, Y_mean, Y_std))

    pred_test, point_estimate = _fit_and_score(
        X_train_gpu, Y_train_gpu, X_test_normed_cpu, Y_test_gpu, alphas, backend,
    )
    del X_train_gpu, X_test_normed_cpu, Y_train_gpu
    torch.cuda.empty_cache()

    # Per-voxel Pearson r for detailed reporting
    voxel_scores = correlation_score(Y_test_gpu, pred_test)
    median_r = float(voxel_scores.median())

    if verbose:
        rprint(
            f"  Test encoding: mean r={point_estimate:.4f}, median r={median_r:.4f} "
            f"({n_voxels} voxels)",
            style="highlight",
        )

    # ── 3. Bootstrap on test predictions (optional) ───────────
    ci_low, ci_high = None, None
    bootstrap_scores_list = None

    if bootstrap:
        n_subsample = int(n_test * 0.9)
        bootstrap_scores = np.empty(n_bootstrap, dtype=np.float64)

        for i in range(n_bootstrap):
            boot_idx = rng.choice(n_test, size=n_subsample, replace=False)
            bootstrap_scores[i] = float(
                correlation_score(Y_test_gpu[boot_idx], pred_test[boot_idx]).mean()
            )

        ci_low = float(np.percentile(bootstrap_scores, 2.5))
        ci_high = float(np.percentile(bootstrap_scores, 97.5))
        bootstrap_scores_list = bootstrap_scores.tolist()

    rprint("")
    msg = f"  Encoding  | {best_layer} = {point_estimate:.4f}"
    if bootstrap:
        msg += f"  [95% CI: {ci_low:.4f}, {ci_high:.4f}]"
    rprint(msg, style="highlight")

    result = {
        "layer": best_layer,
        "compare_method": compare_method,
        "score": point_estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "analysis": "encoding_score",
        "layer_selection_scores": selection_scores,
    }
    if bootstrap_scores_list is not None:
        result["bootstrap_scores"] = bootstrap_scores_list

    return [result]
