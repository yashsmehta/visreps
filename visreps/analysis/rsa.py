import logging
import os
from typing import Dict, List

import numpy as np
import scipy.stats
import torch
from visreps.utils import rprint

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

_CORR_FUNCS = {
    "pearson": scipy.stats.pearsonr,
    "spearman": scipy.stats.spearmanr,
    "kendall": scipy.stats.kendalltau,
}


def _rank(x: torch.Tensor) -> torch.Tensor:
    """Row‑wise dense ranking via double argsort (ties get consecutive ranks)."""
    return torch.argsort(torch.argsort(x, dim=1), dim=1).float()


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def compute_rsm(
    representations: torch.Tensor, *, correlation: str = "Pearson", correction: float = 1e-12
) -> torch.Tensor:
    """Return an (n_samples × n_samples) RSM built with Pearson or Spearman.

    Args:
        representations: Row‑major activations (n_samples, n_features).
        correlation: "Pearson" or "Spearman" (case‑insensitive).
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
    rsm1: torch.Tensor, rsm2: torch.Tensor, *, correlation: str = "Spearman"
) -> float:
    """Correlation between two RSMs using Pearson / Spearman / Kendall.

    Returns NaN if correlation cannot be computed (e.g., zero variance).
    """
    if rsm1.shape != rsm2.shape or rsm1.ndim != 2:
        raise ValueError("RSMs must share the same 2‑D shape")

    n = rsm1.size(0)
    if n <= 1:
        logger.warning("RSM dimension ≤ 1; correlation undefined")
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
        cfg: Configuration dictionary. Must contain 'exp_name' and 'cfg_id'
             if saving RSMs. May also contain:
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
    layer_rsms = {}  # Dictionary to store layer RSMs

    for layer, acts in activations_dict.items():
        flat_acts = acts.flatten(start_dim=1) if acts.ndim > 2 else acts
        layer_rsm = compute_rsm(flat_acts, correlation=make_rsm_corr)
        layer_rsms[layer] = layer_rsm  # Store layer RSM
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
    save_rsms = True
    if save_rsms:
        try:
            save_dir = os.path.join("model_checkpoints", cfg.exp_name, f"cfg{cfg.cfg_id}", "RSMs")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"rsms_{cfg.neural_dataset}_epoch_{cfg.epoch}.npz")

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

