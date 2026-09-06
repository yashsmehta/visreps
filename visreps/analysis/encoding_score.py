"""Encoding score: Ridge regression for voxelwise neural prediction (himalaya).

For voxelwise datasets (NSD fMRI, TVSD electrophysiology) only — not applicable
to behavioral embeddings (THINGS). The metric is always Pearson r between
predicted and actual voxel responses (not configurable like RSA's compare_method).

Two steps, exposed separately so the caller can pick one layer per ROI
across subjects:
    select_layer_scores(...)  -> per-layer validation r on an 80/20 split of train
    evaluate_layer(...)       -> refit chosen layer on full train, score on test
compute_encoding_score(...) chains them for the single-subject case.

Both steps accept ``target_groups``: {name: column slice of the voxel axis}.
Ridge with per-target alpha selection is independent across targets, so voxels
from several ROIs that share stimuli can be fit together and scored per group.
This gives identical numbers to fitting each ROI on its own, with one SVD
instead of one per ROI.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import torch
from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV
from himalaya.scoring import correlation_score
from visreps.utils import rprint

if TYPE_CHECKING:
    from visreps.analysis.alignment import AlignmentData

logger = logging.getLogger(__name__)

# Ridge penalties searched per voxel. The range is narrow on purpose: across
# NSD and TVSD, fine- and coarse-grained models, and every layer, the selected
# alpha sits in log10 alpha ~ [3.7, 5.0], and nothing clips at these bounds.
# Spending the same 20 points on 6 decades instead of 20 gives 0.32-decade
# steps (2.1x apart) rather than 1.05 (11.3x), at no extra cost.
ALPHAS = np.logspace(2, 8, 20)


def _znorm(X, mean, std):
    """Z-normalize using precomputed statistics."""
    return (X - mean) / std


def _znorm_fit(X):
    """Z-normalize X using its own stats. Returns (normalized, mean, std)."""
    mean = X.mean(dim=0)
    std = X.std(dim=0) + 1e-8
    return _znorm(X, mean, std), mean, std


def _flat(a):
    """Flatten 4D→2D and ensure CPU float32."""
    return (a.flatten(start_dim=1) if a.ndim > 2 else a).cpu().float()


def _flatten_to_cpu(acts):
    """``_flat`` over a dict of activations. Returns a new dict (no mutation)."""
    return {layer: _flat(a) for layer, a in acts.items()}


def _fit_and_score(X_tr, Y_tr, X_te, Y_te, alphas, backend):
    """Fit RidgeCV on train, predict on test, return (predictions, per-target Pearson r).

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
    return pred, correlation_score(Y_te, pred)


def _group_means(r, groups):
    """Mean of per-target scores ``r`` within each column slice of ``groups``."""
    return {name: float(r[sl].mean()) for name, sl in groups.items()}


def _bootstrap_scores(Y, pred, groups, n_bootstrap, rng):
    """Per-group mean r over ``n_bootstrap`` stimulus resamples (with replacement).

    Draws the resample indices in the same order as a plain loop would, then
    scores them in chunks: each chunk lays its resamples side by side along the
    target axis so one ``correlation_score`` call scores all of them at once.
    """
    n = Y.size(0)
    pred = torch.as_tensor(pred, device=Y.device)  # himalaya returns predictions on CPU
    idx = np.stack([rng.choice(n, size=n, replace=True) for _ in range(n_bootstrap)])
    idx = torch.as_tensor(idx, device=Y.device)
    chunk = max(1, 50_000_000 // (n * Y.size(1)))
    out = {name: np.empty(n_bootstrap, dtype=np.float64) for name in groups}
    for start in range(0, n_bootstrap, chunk):
        b = idx[start:start + chunk]
        Y_b = Y[b].permute(1, 0, 2).reshape(n, -1)
        pred_b = pred[b].permute(1, 0, 2).reshape(n, -1)
        r = correlation_score(Y_b, pred_b).reshape(len(b), -1)  # (chunk, n_targets)
        for name, sl in groups.items():
            out[name][start:start + len(b)] = r[:, sl].mean(dim=1).cpu().numpy()
    return out


def select_layer_scores(
    selection: "AlignmentData", seed: int = 42, verbose: bool = False,
    target_groups: Optional[Dict[str, slice]] = None,
) -> List[Dict] | Dict[str, List[Dict]]:
    """Per-layer validation score on a seeded 80/20 fit/val split of the train data.

    Y and X are z-normalized with fit-only stats (no leakage into val).
    Returns [{"layer": name, "score": mean Pearson r on val}, ...], or, when
    ``target_groups`` is given, {group: that list scored on the group's voxels}.
    """
    backend = set_backend("torch_cuda", on_error="warn")
    groups = target_groups or {None: slice(None)}
    train_acts = _flatten_to_cpu(selection.activations)
    Y_train = selection.neural.cpu().float()

    n_train = Y_train.size(0)
    split = int(0.8 * n_train)
    perm = np.random.RandomState(seed).permutation(n_train)
    fit_idx, val_idx = perm[:split], perm[split:]

    Y_fit_normed, Y_fit_mean, Y_fit_std = _znorm_fit(Y_train[fit_idx])
    Y_fit_gpu = backend.asarray(Y_fit_normed)
    Y_val_gpu = backend.asarray(_znorm(Y_train[val_idx], Y_fit_mean, Y_fit_std))

    scores = {name: [] for name in groups}
    for layer, acts in train_acts.items():
        X_fit_normed, fit_mean, fit_std = _znorm_fit(acts[fit_idx])
        X_val_normed = _znorm(acts[val_idx], fit_mean, fit_std)
        X_fit_gpu = backend.asarray(X_fit_normed)
        del X_fit_normed

        _, r = _fit_and_score(X_fit_gpu, Y_fit_gpu, X_val_normed, Y_val_gpu, ALPHAS, backend)
        for name, score in _group_means(r, groups).items():
            scores[name].append({"layer": layer, "score": score})
            if verbose:
                tag = f" [{name}]" if name is not None else ""
                rprint(f"  [select] {layer:<15} r={score:.4f}  ({acts.size(1)} features){tag}", style="info")

        del X_fit_gpu, X_val_normed
        torch.cuda.empty_cache()

    del Y_fit_gpu, Y_val_gpu
    return scores if target_groups else scores[None]


def evaluate_layer(
    layer: str,
    selection: "AlignmentData",
    evaluation: "AlignmentData",
    bootstrap: bool = True,
    n_bootstrap: int = 1000,
    seed: int = 42,
    verbose: bool = False,
    reconstruct_pca_k: int | None = None,
    target_groups: Optional[Dict[str, slice]] = None,
) -> Dict:
    """Refit RidgeCV for ``layer`` on full train, score on test (mean Pearson r).

    If ``bootstrap``, resample the test predictions/targets with replacement
    ``n_bootstrap`` times and recompute the score for percentile 95% CIs.
    ``reconstruct_pca_k`` reconstructs the activations from that many
    train-fitted PCs before fitting. With ``target_groups``, returns
    {group: result dict} scored on each group's voxels.
    """
    backend = set_backend("torch_cuda", on_error="warn")
    rng = np.random.RandomState(seed)
    groups = target_groups or {None: slice(None)}

    X_train = _flat(selection.activations[layer])
    X_test = _flat(evaluation.activations[layer])
    Y_train = selection.neural.cpu().float()
    Y_test = evaluation.neural.cpu().float()

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

    pred_test, r_test = _fit_and_score(
        X_train_gpu, Y_train_gpu, X_test_normed, Y_test_gpu, ALPHAS, backend,
    )
    del X_train_gpu, X_test_normed, Y_train_gpu
    torch.cuda.empty_cache()

    point_estimates = _group_means(r_test, groups)
    if verbose:
        for name, sl in groups.items():
            tag = f" [{name}]" if name is not None else ""
            rprint(
                f"  Test encoding: mean r={point_estimates[name]:.4f}, "
                f"median r={float(r_test[sl].median()):.4f} ({r_test[sl].numel()} voxels){tag}",
                style="highlight",
            )

    boot = _bootstrap_scores(Y_test_gpu, pred_test, groups, n_bootstrap, rng) if bootstrap else {}

    results = {}
    for name in groups:
        result = {
            "layer": layer,
            "compare_method": "pearson",
            "score": point_estimates[name],
            "ci_low": None,
            "ci_high": None,
            "analysis": "encoding_score",
        }
        if bootstrap:
            result["ci_low"] = float(np.percentile(boot[name], 2.5))
            result["ci_high"] = float(np.percentile(boot[name], 97.5))
            result["bootstrap_scores"] = boot[name].tolist()
        results[name] = result
    return results if target_groups else results[None]


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
