from __future__ import annotations

import logging
from typing import Dict, List, TYPE_CHECKING

import numpy as np
import scipy.stats
import torch
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from visreps.utils import console, rprint

if TYPE_CHECKING:
    from visreps.analysis.alignment import AlignmentData

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _kendall_tau_a(x: np.ndarray, y: np.ndarray) -> tuple:
    """Kendall tau-a: (C - D) / n_pairs. No tie adjustment."""
    n = len(x)
    if n < 2:
        return (float("nan"), float("nan"))

    tau_b = scipy.stats.kendalltau(x, y).statistic
    if np.isnan(tau_b):
        return (float("nan"), float("nan"))

    # Convert tau-b to tau-a: tau_a = (C-D)/n0, tau_b = (C-D)/sqrt((n0-t_x)*(n0-t_y))
    n0 = n * (n - 1) // 2
    t_x = sum(c * (c - 1) // 2 for c in np.unique(x, return_counts=True)[1])
    t_y = sum(c * (c - 1) // 2 for c in np.unique(y, return_counts=True)[1])

    # Cast to float64 to avoid integer overflow for large n
    denom = np.sqrt(np.float64(n0 - t_x) * np.float64(n0 - t_y))
    tau_a = float("nan") if denom == 0 else float(tau_b * denom / n0)
    return (tau_a, float("nan"))


_CORR_FUNCS = {
    "pearson": scipy.stats.pearsonr,
    "spearman": scipy.stats.spearmanr,
    "kendall": _kendall_tau_a,
}


def _rank(x: torch.Tensor) -> torch.Tensor:
    """Row-wise dense ranking via double argsort (ties get consecutive ranks)."""
    return torch.argsort(torch.argsort(x, dim=1), dim=1).float()


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def compute_rdm(
    representations: torch.Tensor, *, correlation: str = "Pearson", correction: float = 1e-12
) -> torch.Tensor:
    """Return an (n_samples x n_samples) RDM (1 - correlation) built with Pearson or Spearman.

    Diagonal entries are 0 (zero self-dissimilarity). Off-diagonal values range
    from 0 (identical representations) to 2 (perfectly anti-correlated).

    Args:
        representations: Row-major activations (n_samples, n_features).
        correlation: "Pearson" or "Spearman" (case-insensitive).
        correction: Numerical stabiliser added to variances.
    """
    corr = correlation.lower()
    if corr not in {"pearson", "spearman"}:
        raise ValueError("correlation must be 'Pearson' or 'Spearman'")

    x = representations.float().clone()
    if corr == "spearman":
        x = _rank(x)

    x -= x.mean(dim=1, keepdim=True)
    std = x.pow(2).mean(dim=1).add(correction).sqrt()

    # Guard against zero variance rows
    zero_mask = std < correction * 10
    if zero_mask.any():
        logger.warning("%d/%d rows have ~zero variance for %s RDM", zero_mask.sum(), len(std), correlation)
        std[zero_mask] = 1.0

    cov = (x @ x.T) / x.size(1)
    corr_mat = cov / (std[:, None] * std[None, :] + correction)
    corr_mat.clamp_(-1, 1).fill_diagonal_(1.0)
    rdm = 1.0 - corr_mat
    return rdm


def compute_rdm_correlation(
    rdm1: torch.Tensor, rdm2: torch.Tensor, *, correlation: str = "Kendall"
) -> float:
    """Correlation between two RDMs using Pearson / Spearman / Kendall.

    Returns NaN if correlation cannot be computed (e.g., zero variance).
    """
    if rdm1.shape != rdm2.shape or rdm1.ndim != 2:
        raise ValueError("RDMs must share the same 2-D shape")

    n = rdm1.size(0)
    if n <= 1:
        logger.warning("RDM dimension <= 1; correlation undefined")
        return float("nan")

    idx = torch.triu_indices(n, n, offset=1, device=rdm1.device)
    v1 = rdm1[idx[0], idx[1]].cpu().numpy()
    v2 = rdm2[idx[0], idx[1]].cpu().numpy()
    if v1.size == 0:
        return float("nan")

    corr = correlation.lower()
    if corr not in _CORR_FUNCS:
        raise ValueError("correlation must be 'Pearson', 'Spearman', or 'Kendall'")

    try:
        val, _ = _CORR_FUNCS[corr](v1, v2)
        if np.isnan(val):
            logger.warning("NaN returned for %s correlation", correlation)
            return float("nan")
        return float(val)
    except Exception as e:  # pragma: no cover
        logger.error("Error computing %s correlation: %s", correlation, e)
        return float("nan")


def compute_rsa(
    cfg: Dict,
    selection: "AlignmentData",
    evaluation: "AlignmentData",
    n_select: int | None = None,
    bootstrap: bool = True,
    n_bootstrap: int = 1000,
    seed: int = 42,
    verbose: bool = False,
    re_extract_fn=None,
) -> List[Dict]:
    """Train/test RSA: select best layer on train data, evaluate on the test set.

    1. Use all train stimuli (or subsample *n_select* if specified) for layer selection.
    2. Build RDMs (Pearson) and pick the best-aligning layer.
    3. If *re_extract_fn* is provided, re-extract the best layer without SRP
       for exact test RDMs; otherwise use SRP'd activations.
    4. If *bootstrap*, subsample 90% of the eval set *n_bootstrap* times for 95% CIs.

    Args:
        cfg: Must contain ``compare_method`` ("spearman" or "kendall").
        selection: AlignmentData with train activations and neural data.
        evaluation: AlignmentData with test activations and neural data.
        n_select: Max train stimuli for layer selection. None = use all (default).
        bootstrap: Whether to compute bootstrap CIs on the test set.
        n_bootstrap: Number of bootstrap iterations for CIs.
        seed: Random seed for subsampling and bootstrap.
        verbose: Print progress details.
        re_extract_fn: Optional callback ``(layer_name, stimulus_ids=None) -> (tensor, ids)``
            for re-extracting a single layer without SRP.

    Returns:
        Single-element list with result dict containing: layer, compare_method,
        score, ci_low, ci_high, analysis, layer_selection_scores, and
        bootstrap_scores (when bootstrap=True).
    """
    method = cfg.get("compare_method", "spearman").lower()
    rng = np.random.RandomState(seed)

    n_train = selection.neural.size(0)
    n_test = evaluation.neural.size(0)

    if n_select is not None and n_select < n_train:
        n_sel = n_select
        sel_idx = rng.choice(n_train, size=n_sel, replace=False)
        sel_label = f"subsampling {n_sel}"
    else:
        n_sel = n_train
        sel_idx = np.arange(n_train)
        sel_label = f"using all {n_sel}"

    if verbose:
        rprint(
            f"Train/test RSA: {n_train} train, {n_test} test, "
            f"{sel_label} for layer selection",
            style="info",
        )
        rprint(f"Building RDMs with Pearson, comparing with {method.capitalize()}", style="info")

    # ── 1. Select from train for layer selection ─────────────────
    neural_rdm_sel = compute_rdm(selection.neural[sel_idx])

    selection_scores = []
    best_layer, best_score = None, -float("inf")
    for layer, acts in selection.activations.items():
        flat = acts[sel_idx].flatten(start_dim=1) if acts.ndim > 2 else acts[sel_idx]
        layer_rdm = compute_rdm(flat)
        score = compute_rdm_correlation(layer_rdm, neural_rdm_sel, correlation=method.capitalize())
        selection_scores.append({"layer": layer, "score": score})
        if verbose:
            rprint(f"  [select] {layer:<15} RSA = {score:.4f}", style="info")
        if score > best_score:
            best_score = score
            best_layer = layer

    if verbose:
        rprint(f"  Best layer: {best_layer} (score={best_score:.4f})", style="highlight")

    # ── 2. Evaluate on the full test set ─────────────────────────
    # Re-extract best layer without SRP for exact test RDMs if callback available
    if re_extract_fn is not None:
        rprint(f"  Re-extracting {best_layer} without SRP for exact test RDMs...", style="info")
        exact_acts, _ = re_extract_fn(best_layer, evaluation.stimulus_ids)
        test_acts_flat = exact_acts.flatten(start_dim=1) if exact_acts.ndim > 2 else exact_acts
    else:
        test_acts = evaluation.activations[best_layer]
        test_acts_flat = test_acts.flatten(start_dim=1) if test_acts.ndim > 2 else test_acts

    test_neural_rdm = compute_rdm(evaluation.neural)
    test_model_rdm = compute_rdm(test_acts_flat)

    point_estimate = compute_rdm_correlation(
        test_model_rdm, test_neural_rdm, correlation=method.capitalize()
    )
    if verbose:
        rprint(f"  Test RSA = {point_estimate:.4f}", style="highlight")

    # ── 3. Bootstrap on the test set (optional) ──────────────────
    ci_low, ci_high = None, None
    bootstrap_scores_list = None

    if bootstrap:
        n_subsample = int(n_test * 0.9)
        bootstrap_scores = np.empty(n_bootstrap, dtype=np.float64)

        progress = Progress(
            TextColumn("  Bootstrap             "),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("iters"),
            TimeElapsedColumn(),
            console=console,
        )
        progress.start()
        boot_task = progress.add_task("bootstrap", total=n_bootstrap)
        for i in range(n_bootstrap):
            boot_idx = torch.from_numpy(
                rng.choice(n_test, size=n_subsample, replace=False)
            ).to(test_neural_rdm.device)
            rdm_brain_boot = test_neural_rdm[boot_idx][:, boot_idx]
            rdm_model_boot = test_model_rdm[boot_idx][:, boot_idx]
            bootstrap_scores[i] = compute_rdm_correlation(
                rdm_model_boot, rdm_brain_boot, correlation=method.capitalize()
            )
            progress.advance(boot_task)
        progress.stop()

        ci_low = float(np.percentile(bootstrap_scores, 2.5))
        ci_high = float(np.percentile(bootstrap_scores, 97.5))
        bootstrap_scores_list = bootstrap_scores.tolist()

    rprint("")
    msg = f"  {method.capitalize():<10}| {best_layer} = {point_estimate:.4f}"
    if bootstrap:
        msg += f"  [95% CI: {ci_low:.4f}, {ci_high:.4f}]"
    rprint(msg, style="highlight")

    result = {
        "layer": best_layer,
        "compare_method": method,
        "score": point_estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "analysis": "rsa",
        "layer_selection_scores": selection_scores,
    }
    if bootstrap_scores_list is not None:
        result["bootstrap_scores"] = bootstrap_scores_list

    return [result]


def _concept_average_exact(raw_acts, raw_ids, data):
    """Concept-average exact per-image activations using AlignmentData's concept mapping.

    Args:
        raw_acts: (n_images, features) tensor from extract_single_layer.
        raw_ids: List of image IDs matching rows of raw_acts.
        data: AlignmentData with concept_image_ids and stimulus_ids (concept order).

    Returns:
        (n_concepts, features) tensor in the same concept order as data.stimulus_ids.
    """
    id_to_idx = {str(k): i for i, k in enumerate(raw_ids)}
    concept_avgs = []
    for concept in data.stimulus_ids:
        img_ids = data.concept_image_ids[concept]
        indices = [id_to_idx[sid] for sid in img_ids if sid in id_to_idx]
        if indices:
            concept_avgs.append(raw_acts[indices].float().mean(0))
        else:
            # Fallback: zero vector (shouldn't happen if data is consistent)
            concept_avgs.append(torch.zeros(raw_acts.size(1)))
    return torch.stack(concept_avgs).to(raw_acts.dtype)
