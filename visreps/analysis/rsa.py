import numpy as np
import torch
from tqdm.auto import tqdm
from visreps.analysis.metrics import pearson_r, spearman_r
from visreps.utils import rprint
import logging

# Setup logger
logger = logging.getLogger(__name__)


def compute_rsm(activations: torch.Tensor, correction: float = 1e-12) -> torch.Tensor:
    """
    Pearson-R correlation matrix across samples (rows).
    Input : (n_samples, n_features) tensor
    Output: (n_samples, n_samples) RSM
    """
    n_samples, n_features = activations.shape
    x   = activations.float()                                 # ensure fp32
    x  -= x.mean(dim=1, keepdim=True)                         # row-center
    std = x.pow(2).mean(dim=1).add(correction).sqrt()         # row σ

    mask = std < correction * 10                              # near-zero σ
    if mask.any():
        logger.warning(f"{mask.sum().item()} / {n_samples} samples are constant.")
        std[mask] = 1.0                                       # avoid div-by-0

    cov = (x @ x.T) / n_features                              # covariance
    rsm = cov / (std[:, None] * std[None, :] + correction)    # correlation
    rsm.clamp_(-1, 1).fill_diagonal_(1.0)
    return rsm

def compute_rsm_correlation(rsm1: torch.Tensor, rsm2: torch.Tensor, correlation: str = "Pearson") -> float:
    """Compute correlation between two RSMs using specified method (Pearson/Spearman)"""
    # Get upper triangular values (excluding diagonal)
    triu_indices = torch.triu_indices(rsm1.shape[-1], rsm1.shape[-1], offset=1)
    vec1 = rsm1[..., triu_indices[0], triu_indices[1]]
    vec2 = rsm2[..., triu_indices[0], triu_indices[1]]
    
    # Compute correlation
    corr_func = pearson_r if correlation == "Pearson" else spearman_r
    return float(corr_func(vec1, vec2))

def bootstrap_correlation(
    rsm1: torch.Tensor,
    rsm2: torch.Tensor,
    n_bootstraps: int = 500,
    subsample_fraction: float = 0.9,
    correlation: str = "Pearson",
    seed: int = 0,
    batch_size: int = 500,
) -> torch.Tensor:
    """Bootstrap correlation between RSMs with subsampling"""
    n_samples = rsm1.shape[-1]
    n_subset = int(n_samples * subsample_fraction)
    rng = np.random.default_rng(seed=seed)
    corr_func = pearson_r if correlation == "Pearson" else spearman_r
    bootstrap_scores = []
    
    for batch_idx in tqdm(range(0, n_bootstraps, batch_size), desc="Bootstrap", leave=False):
        batch_size = min(batch_size, n_bootstraps - batch_idx)
        batch_scores = []
        
        for _ in range(batch_size):
            # Sample indices
            idx = rng.permutation(n_samples)[:n_subset]
            # Get submatrices
            sub_rsm1 = rsm1[idx][:, idx]
            sub_rsm2 = rsm2[idx][:, idx]
            # Get upper triangular values
            triu_indices = torch.triu_indices(n_subset, n_subset, offset=1)
            vec1 = sub_rsm1[triu_indices[0], triu_indices[1]]
            vec2 = sub_rsm2[triu_indices[0], triu_indices[1]]
            # Compute correlation
            batch_scores.append(corr_func(vec1, vec2, return_diagonal=True))
            
        bootstrap_scores.append(torch.stack(batch_scores))
        
    return torch.cat(bootstrap_scores)

def compute_rsa_alignment(
    cfg,
    activations_dict,
    neural_data
):
    """Return list of dicts with RSA alignment scores for each layer."""
    results = []
    correlation = cfg.get('correlation', 'Pearson')
    n_bootstraps = cfg.get('n_bootstraps', 500)
    subsample_fraction = cfg.get('subsample_fraction', 0.9)
    do_bootstrap = cfg.get('do_bootstrap', False)
    rprint(f"Computing RSA scores using {correlation} correlation...", style="info")
    neural_rsm = compute_rsm(neural_data)
    for layer, activations in activations_dict.items():
        if activations.ndim > 2:
            activations = activations.flatten(start_dim=1)
        layer_rsm = compute_rsm(activations)
        score = compute_rsm_correlation(layer_rsm, neural_rsm, correlation)
        rprint(f"Layer {layer:<20} RSA Score: {score:.4f}", style="highlight")
        result = {
            "layer": layer,
            "score": score,
            "analysis": "rsa",
            "correlation": correlation,
        }
        if do_bootstrap:
            bootstrap_scores = bootstrap_correlation(
                layer_rsm, neural_rsm,
                n_bootstraps=n_bootstraps,
                subsample_fraction=subsample_fraction,
                correlation=correlation
            )
            bootstrap_mean = float(bootstrap_scores.mean())
            bootstrap_std = float(bootstrap_scores.std())
            bootstrap_ci = bootstrap_scores.quantile(torch.tensor([0.025, 0.975]))
            result.update({
                "bootstrap_mean": bootstrap_mean,
                "bootstrap_std": bootstrap_std,
                "bootstrap_ci_lower": float(bootstrap_ci[0]),
                "bootstrap_ci_upper": float(bootstrap_ci[1])
            })
        results.append(result)
    return results
