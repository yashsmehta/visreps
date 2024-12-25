import itertools
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
from tqdm.auto import tqdm
from visreps.analysis.metrics import pearson_r, spearman_r

def compute_rsm(activations: torch.Tensor) -> torch.Tensor:
    """Compute RSM from activations tensor (n_samples, n_features) -> (n_samples, n_samples)"""
    return pearson_r(
        activations.transpose(-2, -1),
        correction=0,
        return_diagonal=False,
    )

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
    """Compute RSA alignment between model layer activations and neural data"""
    results = []
    messages = []
    
    # Get RSA parameters from config
    correlation = cfg.get('correlation', 'Pearson')
    n_bootstraps = cfg.get('n_bootstraps', 500)
    subsample_fraction = cfg.get('subsample_fraction', 0.9)
    
    messages.append(f"Computing RSA scores using {correlation} correlation...")
    
    # Compute neural RSM once
    neural_rsm = compute_rsm(neural_data)
    
    # Compute RSA for each layer
    for layer, activations in activations_dict.items():
        # Flatten activations if needed
        print(f"Layer: {layer}, Activations: {activations.shape}")
        if activations.ndim > 2:
            activations = activations.flatten(start_dim=1)
            
        # Compute layer RSM and correlation
        layer_rsm = compute_rsm(activations)
        score = compute_rsm_correlation(layer_rsm, neural_rsm, correlation)
        
        # Compute bootstrap scores
        bootstrap_scores = bootstrap_correlation(
            layer_rsm, neural_rsm,
            n_bootstraps=n_bootstraps,
            subsample_fraction=subsample_fraction,
            correlation=correlation
        )
        
        # Compute bootstrap statistics
        bootstrap_mean = float(bootstrap_scores.mean())
        bootstrap_std = float(bootstrap_scores.std())
        bootstrap_ci = bootstrap_scores.quantile(torch.tensor([0.025, 0.975]))
        
        messages.append(f"Layer {layer:<20} RSA Score: {score:.4f} (Bootstrap Mean: {bootstrap_mean:.4f} ± {bootstrap_std:.4f})")
        
        # Store results with summary statistics instead of raw bootstrap scores
        results.append({
            "layer": layer,
            "score": score,
            "analysis": "rsa",
            "correlation": correlation,
            "bootstrap_mean": bootstrap_mean,
            "bootstrap_std": bootstrap_std,
            "bootstrap_ci_lower": float(bootstrap_ci[0]),
            "bootstrap_ci_upper": float(bootstrap_ci[1])
        })
        
    return results, messages
