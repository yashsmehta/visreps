import logging
import os
from typing import Dict, List

import numpy as np
import scipy.stats
import torch
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from visreps.utils import console, rprint

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

    x = representations.float()
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


def compute_rsa_alignment(
    cfg: Dict, activations_dict: Dict[str, torch.Tensor], neural_data: torch.Tensor,
    verbose: bool = False,
) -> List[Dict]:
    """Compute RSA alignment for each layer in *activations_dict*.

    Always uses Pearson correlation for building RDMs and computes both
    Spearman and Kendall-tau correlations for comparing RDMs.

    Args:
        cfg: Configuration dictionary.
        activations_dict: Dictionary mapping layer names to activation tensors.
        neural_data: Neural data tensor (e.g., fMRI responses).
        verbose: If True, print per-layer details.

    Returns:
        List of dictionaries, each containing results for a layer.
    """
    if verbose:
        rprint("Building RDMs with Pearson correlation", style="info")
        rprint("Comparing RDMs with Spearman + Kendall correlation", style="info")

    neural_rdm = compute_rdm(neural_data)
    results = []
    layer_rdms = {}

    for layer, acts in activations_dict.items():
        flat_acts = acts.flatten(start_dim=1) if acts.ndim > 2 else acts
        layer_rdm = compute_rdm(flat_acts)
        layer_rdms[layer] = layer_rdm
        score_sp = compute_rdm_correlation(layer_rdm, neural_rdm, correlation="Spearman")
        score_kt = compute_rdm_correlation(layer_rdm, neural_rdm, correlation="Kendall")

        if verbose:
            rprint(
                f"Layer {layer:<15} RSA: Spearman={score_sp:.4f}, Kendall={score_kt:.4f}",
                style="highlight",
            )
        results.append(
            {
                "layer": layer,
                "score_spearman": score_sp,
                "score_kendall": score_kt,
                "analysis": "rsa",
            }
        )

    # --- Internal Flag for Saving RDMs --- (Manually change to True to enable saving)
    save_rdms = False
    if save_rdms:
        try:
            save_dir = os.path.join("model_checkpoints", "RDMs", f"{cfg.neural_dataset}", f"pca4cls")
            os.makedirs(save_dir, exist_ok=True)

            filename_parts = [
                f"pca_labels_{cfg.pca_labels}",
                f"cfgid_{cfg.cfg_id}",
                f"seed_{cfg.seed}"
            ]

            if cfg.reconstruct_from_pcs:
                filename_parts.insert(2, f"pca_k_{cfg.pca_k}")

            save_path = os.path.join(save_dir, "_".join(filename_parts) + ".npz")

            rdms_to_save_np = {layer: rdm.cpu().numpy() for layer, rdm in layer_rdms.items()}
            rdms_to_save_np["neural"] = neural_rdm.cpu().numpy()

            np.savez(save_path, **rdms_to_save_np)
            rprint(f"Saved RDMs (NumPy) to: {save_path}", style="success")
        except KeyError as e:
            logger.error(f"Missing key in cfg for saving RDMs: {e}")
            rprint(f"Error saving RDMs: Missing required key {e} in cfg", style="error")
        except Exception as e:
            logger.error(f"Failed to save RDMs to {save_path}: {e}")
            rprint(f"Error saving RDMs: {e}", style="error")

    return results


def compute_rsa_kfold(
    cfg: Dict,
    activations_dict: Dict[str, torch.Tensor],
    neural_data: torch.Tensor,
    n_folds: int = 5,
    bootstrap: bool = False,
    n_bootstrap: int = 1000,
    seed: int = 42,
    verbose: bool = False,
) -> List[Dict]:
    """5-fold cross-validated RSA with unbiased layer selection.

    Always uses Pearson correlation for building RDMs and computes both
    Spearman and Kendall-tau correlations for comparing RDMs. Layer selection
    is performed independently for each comparison metric.

    Procedure:
        1. Shuffle stimuli into *n_folds* folds (fixed seed, same for all subjects).
        2. For each fold k:
           a. Build selection-set RDMs once (Pearson).
           b. Compute per-layer RSA with both Spearman and Kendall → pick best
              layer independently per metric.
           c. Build evaluation-set RDMs once (Pearson). Compute eval RSA with
              each metric's best layer.
        3. Point estimate = mean of the *n_folds* evaluation scores (per metric).
        4. If *bootstrap* is True, run bootstrap resampling on the last fold's
           evaluation set to obtain 95% CI bounds for both metrics.

    Args:
        cfg: Config dict.
        activations_dict: {layer_name: activations} tensors.
        neural_data: Neural response tensor (n_stimuli, n_voxels).
        n_folds: Number of CV folds (default 5).
        bootstrap: Whether to compute bootstrap CIs on the last fold.
        n_bootstrap: Number of bootstrap iterations (only used if bootstrap=True).
        seed: Random seed for shuffling stimuli (fixed across subjects).
        verbose: If True, print per-fold details. Default False (summary only).

    Returns:
        Single-element list with a dict containing: layer, score_spearman,
        score_kendall, ci_low_spearman, ci_high_spearman, ci_low_kendall,
        ci_high_kendall, analysis, layer_selection_scores_spearman,
        layer_selection_scores_kendall, fold_results_spearman,
        fold_results_kendall, and bootstrap_scores_spearman/kendall
        (when bootstrap=True).
    """
    from collections import Counter, defaultdict

    _COMPARE_METHODS = ("spearman", "kendall")

    n_stimuli = neural_data.size(0)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_stimuli)

    fold_indices = np.array_split(perm, n_folds)

    if verbose:
        rprint(
            f"K-fold CV: {n_folds} folds, ~{len(fold_indices[0])} stimuli per fold "
            f"(total {n_stimuli})",
            style="info",
        )
        rprint("Building RDMs with Pearson, comparing with Spearman + Kendall", style="info")

    # Per-metric tracking
    fold_scores = {m: [] for m in _COMPARE_METHODS}
    fold_best_layers = {m: [] for m in _COMPARE_METHODS}
    all_selection_scores = {m: [] for m in _COMPARE_METHODS}
    bootstrap_scores = {}

    progress = Progress(
        TextColumn("  K-fold RSA            "),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("folds"),
        TimeElapsedColumn(),
        console=console,
    )
    progress.start()
    kfold_task = progress.add_task("kfold", total=n_folds)

    for k in range(n_folds):
        sel_idx = fold_indices[k]
        eval_idx = np.concatenate([fold_indices[j] for j in range(n_folds) if j != k])

        if verbose:
            rprint(f"\n  Fold {k+1}/{n_folds}: {len(sel_idx)} select, {len(eval_idx)} eval", style="info")

        # ── Build selection RDMs once (Pearson) ──────────────────
        neural_rdm_sel = compute_rdm(neural_data[sel_idx])
        layer_rdms_sel = {}
        for layer, acts in activations_dict.items():
            flat = acts[sel_idx].flatten(start_dim=1) if acts.ndim > 2 else acts[sel_idx]
            layer_rdms_sel[layer] = compute_rdm(flat)

        # ── Layer selection independently per metric ─────────────
        for method in _COMPARE_METHODS:
            best_layer, best_score = None, -float("inf")
            for layer, layer_rdm in layer_rdms_sel.items():
                score = compute_rdm_correlation(
                    layer_rdm, neural_rdm_sel, correlation=method.capitalize()
                )
                all_selection_scores[method].append({"layer": layer, "score": score})
                if verbose:
                    rprint(f"    [select/{method}] {layer:<15} RSA = {score:.4f}", style="info")
                if score > best_score:
                    best_score = score
                    best_layer = layer
            fold_best_layers[method].append(best_layer)
            if verbose:
                rprint(f"    Best layer ({method}): {best_layer} (score={best_score:.4f})", style="highlight")

        # ── Evaluate on remaining folds ──────────────────────────
        rdm_brain_eval = compute_rdm(neural_data[eval_idx])
        last_fold_rdm_models = {}

        for method in _COMPARE_METHODS:
            best_layer = fold_best_layers[method][-1]
            eval_acts = activations_dict[best_layer][eval_idx]
            eval_acts = eval_acts.flatten(start_dim=1) if eval_acts.ndim > 2 else eval_acts
            rdm_model_eval = compute_rdm(eval_acts)

            # Cache for bootstrap on last fold
            if bootstrap and k == n_folds - 1:
                last_fold_rdm_models[method] = rdm_model_eval

            eval_score = compute_rdm_correlation(
                rdm_model_eval, rdm_brain_eval, correlation=method.capitalize()
            )
            fold_scores[method].append(eval_score)
            if verbose:
                rprint(f"    Eval RSA ({method}) = {eval_score:.4f}", style="highlight")

        # ── Bootstrap on last fold's eval set ────────────────────
        if bootstrap and k == n_folds - 1:
            if verbose:
                rprint(f"\n  Running bootstrap ({n_bootstrap} iters) on fold {k+1} eval set...", style="info")
            n_eval = len(eval_idx)
            bootstrap_scores = {m: np.empty(n_bootstrap, dtype=np.float64) for m in _COMPARE_METHODS}

            boot_progress = Progress(
                TextColumn("  Bootstrap             "),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("iters"),
                TimeElapsedColumn(),
                console=console,
                disable=not verbose,
            )
            boot_progress.start()
            boot_task = boot_progress.add_task("bootstrap", total=n_bootstrap)
            for i in range(n_bootstrap):
                boot_idx = torch.from_numpy(
                    rng.choice(n_eval, size=n_eval, replace=True)
                ).to(rdm_brain_eval.device)
                rdm_brain_boot = rdm_brain_eval[boot_idx][:, boot_idx]

                for method in _COMPARE_METHODS:
                    rdm_model_boot = last_fold_rdm_models[method][boot_idx][:, boot_idx]
                    bootstrap_scores[method][i] = compute_rdm_correlation(
                        rdm_model_boot, rdm_brain_boot, correlation=method.capitalize()
                    )
                boot_progress.advance(boot_task)
            boot_progress.stop()

        progress.advance(kfold_task)

    progress.stop()

    # ── Per-layer mean selection scores ──────────────────────────
    mean_selection_scores = {}
    for method in _COMPARE_METHODS:
        layer_score_accum = defaultdict(list)
        for entry in all_selection_scores[method]:
            layer_score_accum[entry["layer"]].append(entry["score"])
        mean_selection_scores[method] = [
            {"layer": layer, "score": float(np.mean(scores))}
            for layer, scores in layer_score_accum.items()
        ]

    # ── Per-fold diagnostic info ─────────────────────────────────
    fold_results = {}
    for method in _COMPARE_METHODS:
        fold_results[method] = [
            {"fold": k, "layer": fold_best_layers[method][k], "eval_score": fold_scores[method][k]}
            for k in range(n_folds)
        ]

    # ── Aggregate across folds ───────────────────────────────────
    point_estimates = {m: float(np.mean(fold_scores[m])) for m in _COMPARE_METHODS}
    reported_layers = {
        m: Counter(fold_best_layers[m]).most_common(1)[0][0] for m in _COMPARE_METHODS
    }

    # Spearman-selected layer is the primary reported layer
    reported_layer = reported_layers["spearman"]

    ci = {m: (None, None) for m in _COMPARE_METHODS}
    if bootstrap:
        for method in _COMPARE_METHODS:
            ci[method] = (
                float(np.percentile(bootstrap_scores[method], 2.5)),
                float(np.percentile(bootstrap_scores[method], 97.5)),
            )

    rprint("")  # blank line before results
    for method in _COMPARE_METHODS:
        layer_m = reported_layers[method]
        count = Counter(fold_best_layers[method])[layer_m]
        msg = f"  {method.capitalize():<10}| {layer_m} = {point_estimates[method]:.4f}  ({count}/{n_folds} folds)"
        if bootstrap:
            msg += f"  [95% CI: {ci[method][0]:.4f}, {ci[method][1]:.4f}]"
        rprint(msg, style="highlight")

    result = {
        "layer": reported_layer,
        "score_spearman": point_estimates["spearman"],
        "score_kendall": point_estimates["kendall"],
        "ci_low_spearman": ci["spearman"][0],
        "ci_high_spearman": ci["spearman"][1],
        "ci_low_kendall": ci["kendall"][0],
        "ci_high_kendall": ci["kendall"][1],
        "analysis": "rsa",
        "layer_selection_scores_spearman": mean_selection_scores["spearman"],
        "layer_selection_scores_kendall": mean_selection_scores["kendall"],
        "fold_results_spearman": fold_results["spearman"],
        "fold_results_kendall": fold_results["kendall"],
    }
    if bootstrap:
        result["bootstrap_scores_spearman"] = bootstrap_scores["spearman"].tolist()
        result["bootstrap_scores_kendall"] = bootstrap_scores["kendall"].tolist()

    return [result]
