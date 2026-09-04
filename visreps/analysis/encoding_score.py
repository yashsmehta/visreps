"""Encoding score: Ridge regression for voxelwise neural prediction (himalaya).

For voxelwise datasets (NSD fMRI, TVSD electrophysiology) only — not applicable
to behavioral embeddings (THINGS). The metric is always Pearson r between
predicted and actual voxel responses (not configurable like RSA's compare_method).

Two steps, exposed separately so the caller can pick one layer per ROI
across subjects:
    select_layer_scores(...)  -> per-layer validation r on an 80/20 split of train
    evaluate_layer(...)       -> refit chosen layer on full train, score on test
compute_encoding_score(...) chains them for the single-subject case.
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

ALPHAS = np.logspace(-10, 10, 20)


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

    X_te may be a CPU tensor; it is moved to GPU only after the fit completes
    to keep peak GPU memory low.
    """
    # fit_intercept=False because data is already z-normalized (zero mean).
    # Avoids himalaya's internal X_offset copy which doubles GPU memory.
    model = RidgeCV(alphas=alphas, cv=5, fit_intercept=False)
    model.fit(X_tr, Y_tr)
    if not hasattr(X_te, 'device') or X_te.device.type == 'cpu':
        X_te = backend.asarray(X_te)
    pred = model.predict(X_te)
    score = float(correlation_score(Y_te, pred).mean())
    return pred, score


def select_layer_scores(
    selection: "AlignmentData", seed: int = 42, verbose: bool = False,
) -> List[Dict]:
    """Per-layer validation score on a seeded 80/20 fit/val split of the train data.

    Y and X are z-normalized with fit-only stats (no leakage into val).
    Returns [{"layer": name, "score": mean Pearson r on val}, ...].
    """
    backend = set_backend("torch_cuda", on_error="warn")
    train_acts = _flatten_to_cpu(selection.activations)
    Y_train = selection.neural.cpu().float()

    n_train = Y_train.size(0)
    split = int(0.8 * n_train)
    perm = np.random.RandomState(seed).permutation(n_train)
    fit_idx, val_idx = perm[:split], perm[split:]

    Y_fit_normed, Y_fit_mean, Y_fit_std = _znorm_fit(Y_train[fit_idx])
    Y_fit_gpu = backend.asarray(Y_fit_normed)
    Y_val_gpu = backend.asarray(_znorm(Y_train[val_idx], Y_fit_mean, Y_fit_std))

    scores = []
    for layer, acts in train_acts.items():
        X_fit_normed, fit_mean, fit_std = _znorm_fit(acts[fit_idx])
        X_val_normed = _znorm(acts[val_idx], fit_mean, fit_std)
        X_fit_gpu = backend.asarray(X_fit_normed)
        del X_fit_normed

        _, score = _fit_and_score(X_fit_gpu, Y_fit_gpu, X_val_normed, Y_val_gpu, ALPHAS, backend)
        scores.append({"layer": layer, "score": score})
        if verbose:
            rprint(f"  [select] {layer:<15} r={score:.4f}  ({acts.size(1)} features)", style="info")

        del X_fit_gpu, X_val_normed
        torch.cuda.empty_cache()

    del Y_fit_gpu, Y_val_gpu
    return scores


def evaluate_layer(
    layer: str,
    selection: "AlignmentData",
    evaluation: "AlignmentData",
    bootstrap: bool = True,
    n_bootstrap: int = 1000,
    seed: int = 42,
    verbose: bool = False,
    reconstruct_pca_k: int | None = None,
) -> Dict:
    """Refit RidgeCV for ``layer`` on full train, score on test (mean Pearson r).

    If ``bootstrap``, subsample 90% of the test predictions/targets
    ``n_bootstrap`` times and recompute the score for 95% CIs.
    ``reconstruct_pca_k`` reconstructs the activations from that many
    train-fitted PCs before fitting.
    """
    backend = set_backend("torch_cuda", on_error="warn")
    rng = np.random.RandomState(seed)

    X_train = _flatten_to_cpu({layer: selection.activations[layer]})[layer]
    X_test = _flatten_to_cpu({layer: evaluation.activations[layer]})[layer]
    Y_train = selection.neural.cpu().float()
    Y_test = evaluation.neural.cpu().float()
    n_test = Y_test.size(0)

    if reconstruct_pca_k is not None:
        from sklearn.decomposition import PCA

        rprint(f"  Reconstructing {layer} from {reconstruct_pca_k} PCs (train-fitted)", style="info")
        pca = PCA(n_components=min(reconstruct_pca_k, X_train.size(1))).fit(X_train.numpy())
        X_train = torch.from_numpy(pca.inverse_transform(pca.transform(X_train.numpy())).astype(np.float32))
        X_test = torch.from_numpy(pca.inverse_transform(pca.transform(X_test.numpy())).astype(np.float32))

    # Z-normalize X and Y with full-train stats
    X_train_normed, train_mean, train_std = _znorm_fit(X_train)
    X_train_gpu = backend.asarray(X_train_normed)
    X_test_normed = _znorm(X_test, train_mean, train_std)
    del X_train_normed, X_train

    Y_train_normed, Y_mean, Y_std = _znorm_fit(Y_train)
    Y_train_gpu = backend.asarray(Y_train_normed)
    Y_test_gpu = backend.asarray(_znorm(Y_test, Y_mean, Y_std))

    pred_test, point_estimate = _fit_and_score(
        X_train_gpu, Y_train_gpu, X_test_normed, Y_test_gpu, ALPHAS, backend,
    )
    del X_train_gpu, X_test_normed, Y_train_gpu
    torch.cuda.empty_cache()

    if verbose:
        median_r = float(correlation_score(Y_test_gpu, pred_test).median())
        rprint(
            f"  Test encoding: mean r={point_estimate:.4f}, median r={median_r:.4f} "
            f"({Y_test.size(1)} voxels)",
            style="highlight",
        )

    ci_low, ci_high, bootstrap_scores_list = None, None, None
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

    result = {
        "layer": layer,
        "compare_method": "pearson",
        "score": point_estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "analysis": "encoding_score",
    }
    if bootstrap_scores_list is not None:
        result["bootstrap_scores"] = bootstrap_scores_list
    return result


def compute_encoding_score(
    selection: "AlignmentData",
    evaluation: "AlignmentData",
    bootstrap: bool = True,
    n_bootstrap: int = 1000,
    seed: int = 42,
    verbose: bool = False,
    reconstruct_pca_k: int | None = None,
    quiet: bool = False,
) -> List[Dict]:
    """Single-subject encoding score: select best layer on train, evaluate on test.

    Does NOT mutate the input AlignmentData objects. Returns a single-element
    list with the result dict (see ``evaluate_layer``) plus
    ``layer_selection_scores``.
    """
    if verbose:
        rprint(
            f"Train/test encoding: {selection.neural.size(0)} train, "
            f"{evaluation.neural.size(0)} test, {selection.neural.size(1)} voxels",
            style="info",
        )

    selection_scores = select_layer_scores(selection, seed=seed, verbose=verbose)
    best = max(selection_scores, key=lambda s: s["score"])
    if verbose:
        rprint(f"  Best layer: {best['layer']} (val r={best['score']:.4f})", style="highlight")

    result = evaluate_layer(
        best["layer"], selection, evaluation,
        bootstrap=bootstrap, n_bootstrap=n_bootstrap, seed=seed,
        verbose=verbose, reconstruct_pca_k=reconstruct_pca_k,
    )
    result["layer_selection_scores"] = selection_scores

    if not quiet:
        msg = f"\n  {result['layer']:<7} {result['score']:.4f}"
        if bootstrap:
            msg += f"  [{result['ci_low']:.4f}, {result['ci_high']:.4f}]"
        rprint(msg, style="highlight")

    return [result]
