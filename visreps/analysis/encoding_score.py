"""Encoding score: Ridge regression with per-voxel alpha selection (himalaya)."""

import numpy as np
import torch
from typing import Dict, List
from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV
from visreps.utils import rprint


def _pearson_r(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Pearson correlation per column."""
    yp = y_pred - y_pred.mean(dim=0)
    yt = y_true - y_true.mean(dim=0)
    num = (yp * yt).sum(dim=0)
    denom = torch.sqrt((yp**2).sum(dim=0) * (yt**2).sum(dim=0)) + 1e-10
    return num / denom


def compute_encoding_alignment(
    cfg: Dict,
    activations_dict: Dict[str, torch.Tensor],
    neural_data: torch.Tensor,
) -> List[Dict]:
    """Compute encoding score per layer using ridge regression."""
    backend = set_backend("torch_cuda", on_error="warn")
    rprint(f"Encoding scores (backend={backend.name})", style="info")

    alphas = np.logspace(-10, 10, 20)
    seed = cfg.get("seed", 42)
    n_folds = cfg.get("cv_folds", 5)

    # Train/test split
    n = len(neural_data)
    idx = np.random.default_rng(seed).permutation(n)
    split = int(cfg.get("train_fraction", 0.8) * n)
    tr_idx, te_idx = idx[:split], idx[split:]

    Y = backend.asarray(neural_data.float())
    Y_tr, Y_te = Y[tr_idx], Y[te_idx]

    results = []
    for layer, acts in activations_dict.items():
        # Flatten activations
        X = acts.flatten(start_dim=1) if acts.ndim > 2 else acts
        X = backend.asarray(X.float())

        # Split BEFORE normalization to avoid data leakage
        X_tr, X_te = X[tr_idx], X[te_idx]

        # Z-normalize using training statistics only
        train_mean = X_tr.mean(dim=0)
        train_std = X_tr.std(dim=0) + 1e-8
        X_tr = (X_tr - train_mean) / train_std
        X_te = (X_te - train_mean) / train_std

        # Fit ridge with per-voxel alpha selection (fit_intercept=True for non-centered Y)
        model = RidgeCV(alphas=alphas, cv=n_folds, fit_intercept=True)
        model.fit(X_tr, Y_tr)

        pred = backend.asarray(model.predict(X_te))
        score = float(_pearson_r(pred, Y_te).mean().cpu())
        all_alphas = backend.to_numpy(model.best_alphas_)
        alpha_median = float(np.median(all_alphas))

        rprint(f"{layer:<15} r={score:.4f} (α_median={alpha_median:.2e})", style="highlight")
        results.append({
            "layer": layer, "score": score, "analysis": "encoding_score",
            "alpha_median": alpha_median, "n_train": len(tr_idx), "n_test": len(te_idx),
        })

    return results
