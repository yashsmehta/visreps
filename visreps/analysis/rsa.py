import logging
import os
from typing import Dict, List

import numpy as np
import scipy.stats
import torch
from tqdm import tqdm
from visreps.utils import rprint

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

def compute_rsm(
    representations: torch.Tensor, *, correlation: str = "Pearson", correction: float = 1e-12
) -> torch.Tensor:
    """Return an (n_samples x n_samples) RSM built with Pearson or Spearman.

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
        logger.warning("%d/%d rows have ~zero variance for %s RSM", zero_mask.sum(), len(std), correlation)
        std[zero_mask] = 1.0

    cov = (x @ x.T) / x.size(1)
    rsm = cov / (std[:, None] * std[None, :] + correction)
    rsm.clamp_(-1, 1).fill_diagonal_(1.0)
    return rsm


def compute_rsm_correlation(
    rsm1: torch.Tensor, rsm2: torch.Tensor, *, correlation: str = "Kendall"
) -> float:
    """Correlation between two RSMs using Pearson / Spearman / Kendall.

    Returns NaN if correlation cannot be computed (e.g., zero variance).
    """
    if rsm1.shape != rsm2.shape or rsm1.ndim != 2:
        raise ValueError("RSMs must share the same 2-D shape")

    n = rsm1.size(0)
    if n <= 1:
        logger.warning("RSM dimension <= 1; correlation undefined")
        return float("nan")

    idx = torch.triu_indices(n, n, offset=1, device=rsm1.device)
    v1 = rsm1[idx[0], idx[1]].cpu().numpy()
    v2 = rsm2[idx[0], idx[1]].cpu().numpy()
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
    cfg: Dict, activations_dict: Dict[str, torch.Tensor], neural_data: torch.Tensor
) -> List[Dict]:
    """Compute RSA alignment for each layer in *activations_dict*.

    Args:
        cfg: Configuration dictionary. May contain:
            make_rsm_correlation: method for building RSMs (default "Pearson").
            compare_rsm_correlation: method for comparing RSMs (default "Kendall").
        activations_dict: Dictionary mapping layer names to activation tensors.
        neural_data: Neural data tensor (e.g., fMRI responses).

    Returns:
        List of dictionaries, each containing results for a layer.
    """
    make_rsm_corr = cfg.get("make_rsm_correlation", "Pearson")
    cmp_rsm_corr = cfg.get("compare_rsm_correlation", "Kendall")

    rprint(f"Building RSMs with {make_rsm_corr} correlation", style="info")
    rprint(f"Comparing RSMs with {cmp_rsm_corr} correlation", style="info")

    neural_rsm = compute_rsm(neural_data, correlation=make_rsm_corr)
    results = []
    layer_rsms = {}

    for layer, acts in activations_dict.items():
        flat_acts = acts.flatten(start_dim=1) if acts.ndim > 2 else acts
        layer_rsm = compute_rsm(flat_acts, correlation=make_rsm_corr)
        layer_rsms[layer] = layer_rsm
        score = compute_rsm_correlation(layer_rsm, neural_rsm, correlation=cmp_rsm_corr)

        rprint(
            f"Layer {layer:<15} RSA ({make_rsm_corr} / {cmp_rsm_corr}): {score:.4f}",
            style="highlight",
        )
        results.append(
            {
                "layer": layer,
                "score": score,
                "analysis": "rsa",
                "make_rsm_correlation": make_rsm_corr,
                "compare_rsm_correlation": cmp_rsm_corr,
            }
        )

    # --- Internal Flag for Saving RSMs --- (Manually change to True to enable saving)
    save_rsms = False
    if save_rsms:
        try:
            save_dir = os.path.join("model_checkpoints", "RSMs", f"{cfg.neural_dataset}", f"pca4cls")
            os.makedirs(save_dir, exist_ok=True)

            # Construct filename based on whether reconstruct_from_pcs is True
            filename_parts = [
                f"pca_labels_{cfg.pca_labels}",
                f"cfgid_{cfg.cfg_id}",
                f"seed_{cfg.seed}"
            ]

            if cfg.reconstruct_from_pcs:
                filename_parts.insert(2, f"pca_k_{cfg.pca_k}")

            save_path = os.path.join(save_dir, "_".join(filename_parts) + ".npz")

            # Prepare data to save (convert to numpy)
            rsms_to_save_np = {layer: rsm.cpu().numpy() for layer, rsm in layer_rsms.items()}
            rsms_to_save_np["neural"] = neural_rsm.cpu().numpy()

            np.savez(save_path, **rsms_to_save_np)
            rprint(f"Saved RSMs (NumPy) to: {save_path}", style="success")
        except KeyError as e:
            logger.error(f"Missing key in cfg for saving RSMs: {e}")
            rprint(f"Error saving RSMs: Missing required key {e} in cfg", style="error")
        except Exception as e:
            logger.error(f"Failed to save RSMs to {save_path}: {e}")
            rprint(f"Error saving RSMs: {e}", style="error")

    return results


def compute_rsa_split_half_bootstrap(
    cfg: Dict,
    activations_dict: Dict[str, torch.Tensor],
    neural_data: torch.Tensor,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> List[Dict]:
    """Split-half layer selection + bootstrap evaluation for unbiased model-level RSA.

    Procedure:
        1. Split stimuli 50/50 (selection half vs evaluation half).
        2. On the selection half, compute per-layer RSA → pick the best layer.
        3. Precompute the best-layer model RSM and the neural RSM on the evaluation half.
        4. Compute the point estimate as the direct RSA on the full evaluation half
           (no resampling — avoids upward bias from bootstrap duplicates).
        5. Bootstrap: resample stimulus indices with replacement, reindex the
           precomputed RSMs (no recomputation), and compare upper triangles.
           The bootstrap distribution provides CI bounds only.

    Why reindexing works:
        RSM[i,j] = corr(row_i, row_j) depends only on those two rows, not on
        the rest of the dataset.  A bootstrap sample with indices ``boot_idx``
        produces RSM_boot[p,q] = RSM_orig[boot_idx[p], boot_idx[q]].
        When boot_idx maps two positions to the same stimulus, the off-diagonal
        entry is 1.0 (self-correlation) — identical to rebuilding from scratch.

    Why the point estimate is NOT the bootstrap mean:
        Bootstrap resampling with replacement creates duplicate stimuli, producing
        trivially concordant (1.0, 1.0) pairs in both RSM upper triangles. This
        systematically inflates the bootstrap mean. The unbiased point estimate
        is the direct RSA on the full (unresampled) evaluation half.

    Returns:
        Single-element list with a dict containing: layer, score, ci_low, ci_high,
        analysis, make_rsm_correlation, compare_rsm_correlation,
        layer_selection_scores (List[Dict] of per-layer scores on the selection half).
    """
    make_rsm_corr = cfg.get("make_rsm_correlation", "Pearson")
    cmp_rsm_corr = cfg.get("compare_rsm_correlation", "Kendall")

    n_stimuli = neural_data.size(0)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_stimuli)
    half = n_stimuli // 2
    sel_idx, eval_idx = perm[:half], perm[half:]

    rprint(f"Split-half: {half} stimuli for selection, {len(eval_idx)} for evaluation", style="info")

    # ── Step 1: Layer selection on first half ────────────────────────────
    neural_rsm_sel = compute_rsm(neural_data[sel_idx], correlation=make_rsm_corr)

    selection_scores = []
    best_layer, best_score = None, -float("inf")
    for layer, acts in activations_dict.items():
        flat = acts[sel_idx].flatten(start_dim=1) if acts.ndim > 2 else acts[sel_idx]
        layer_rsm = compute_rsm(flat, correlation=make_rsm_corr)
        score = compute_rsm_correlation(layer_rsm, neural_rsm_sel, correlation=cmp_rsm_corr)
        selection_scores.append({"layer": layer, "score": score})
        rprint(f"  [selection] {layer:<10} RSA = {score:.4f}", style="info")
        if score > best_score:
            best_score = score
            best_layer = layer

    rprint(f"  Best layer: {best_layer} (score={best_score:.4f})", style="highlight")

    # ── Step 2: Precompute RSMs on evaluation half (done once) ───────────
    eval_acts = activations_dict[best_layer][eval_idx]
    eval_acts = eval_acts.flatten(start_dim=1) if eval_acts.ndim > 2 else eval_acts

    rsm_model_full = compute_rsm(eval_acts, correlation=make_rsm_corr)
    rsm_brain_full = compute_rsm(neural_data[eval_idx], correlation=make_rsm_corr)

    # ── Step 3: Point estimate on the full (unresampled) evaluation half ─
    point_estimate = compute_rsm_correlation(
        rsm_model_full, rsm_brain_full, correlation=cmp_rsm_corr
    )
    rprint(f"  Eval-half point estimate: {best_layer} RSA = {point_estimate:.4f}", style="info")

    # ── Step 4: Bootstrap via RSM reindexing (for CI only) ───────────────
    n_eval = len(eval_idx)
    bootstrap_scores = np.empty(n_bootstrap, dtype=np.float64)

    for i in tqdm(range(n_bootstrap), desc="  Bootstrap", unit="iter"):
        boot_idx = torch.from_numpy(
            rng.choice(n_eval, size=n_eval, replace=True)
        ).to(rsm_model_full.device)

        rsm_model_boot = rsm_model_full[boot_idx][:, boot_idx]
        rsm_brain_boot = rsm_brain_full[boot_idx][:, boot_idx]

        bootstrap_scores[i] = compute_rsm_correlation(
            rsm_model_boot, rsm_brain_boot, correlation=cmp_rsm_corr
        )

    # 95% CI → tails at 2.5% and 97.5%
    ci_low, ci_high = np.percentile(bootstrap_scores, [2.5, 97.5])

    rprint(
        f"  Result: {best_layer} RSA = {point_estimate:.4f} "
        f"[95% CI: {ci_low:.4f}, {ci_high:.4f}]",
        style="highlight",
    )

    return [{
        "layer": best_layer,
        "score": float(point_estimate),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "analysis": "rsa",
        "make_rsm_correlation": make_rsm_corr,
        "compare_rsm_correlation": cmp_rsm_corr,
        "layer_selection_scores": selection_scores,
    }]
