import numpy as np
import torch
from itertools import batched
from tqdm.auto import tqdm
from typing import Callable, Tuple, Union, Iterator, Dict, List
import pandas as pd


def pearson_r(x: torch.Tensor, y: torch.Tensor = None, return_diagonal: bool = False) -> torch.Tensor:
    """
    Compute Pearson correlation between matrices x and y.
    If y is None, compute correlation between x and itself.
    """
    if y is None:
        y = x
        
    # Center the data
    x_centered = x - x.mean(dim=1, keepdim=True)
    y_centered = y - y.mean(dim=1, keepdim=True)
    
    # Compute correlation matrix
    x_std = torch.sqrt((x_centered ** 2).sum(dim=1, keepdim=True))
    y_std = torch.sqrt((y_centered ** 2).sum(dim=1, keepdim=True))
    
    corr = torch.matmul(x_centered, y_centered.t()) / (torch.matmul(x_std, y_std.t()) + 1e-8)
    
    if return_diagonal:
        return torch.diagonal(corr)
    return corr


def spearman_r(x: torch.Tensor, y: torch.Tensor = None, return_diagonal: bool = False) -> torch.Tensor:
    """Compute Spearman correlation between matrices x and y."""
    if y is None:
        y = x
    x_rank = x.argsort(dim=1).argsort(dim=1).float()
    y_rank = y.argsort(dim=1).argsort(dim=1).float()
    return pearson_r(x_rank, y_rank, return_diagonal=return_diagonal)


def compute_rsm(x: torch.Tensor) -> torch.Tensor:
    """Compute representational similarity matrix using Pearson correlation."""
    # Flatten any spatial dimensions
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    
    # Normalize the features
    x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-8)
    
    # Compute correlation matrix
    rsm = pearson_r(x)
    return rsm


def extract_upper_triangle(rsm: torch.Tensor) -> torch.Tensor:
    """Extract upper triangle of the RSM, excluding diagonal."""
    n = rsm.shape[0]  # Should be a square matrix
    x_indices, y_indices = torch.triu_indices(n, n, offset=1)
    return rsm[x_indices, y_indices]


def _get_correlation_function(correlation: str) -> Callable:
    """Get the correlation function based on method name."""
    if correlation == "Pearson":
        return pearson_r
    elif correlation == "Spearman":
        return spearman_r
    else:
        raise ValueError(f"Unsupported correlation method: {correlation}")


def compute_rsa_correlation(
    rsm_x: torch.Tensor,
    rsm_y: torch.Tensor,
    correlation: str = "Pearson",
    n_bootstraps: int = 5_000,
    subsample_fraction: float = 0.9,
    seed: int = 0,
    batch_size: int = 500,
) -> Tuple[float, torch.Tensor]:
    """
    Compute RSA correlation between two RSMs with bootstrapping.
    """
    # Extract upper triangles first
    triu_x = extract_upper_triangle(rsm_x)
    triu_y = extract_upper_triangle(rsm_y)
    
    # Compute base correlation
    func = _get_correlation_function(correlation)
    r = float(func(triu_x.unsqueeze(0), triu_y.unsqueeze(0), return_diagonal=True))
    
    # Bootstrap
    n_pairs = triu_x.shape[0]
    rng = np.random.default_rng(seed=seed)
    r_bootstrapped = []
    
    for bootstrap_batch in tqdm(
        batched(range(n_bootstraps), batch_size),
        desc="bootstrap",
        total=(n_bootstraps + batch_size - 1) // batch_size,
        leave=False,
    ):
        batch_correlations = []
        for _ in bootstrap_batch:
            # Sample pair indices
            sample_indices = torch.tensor(
                rng.choice(n_pairs, size=int(n_pairs * subsample_fraction), replace=True)
            )
            
            # Compute correlation for this sample
            batch_r = func(
                triu_x[sample_indices].unsqueeze(0),
                triu_y[sample_indices].unsqueeze(0),
                return_diagonal=True,
            )
            batch_correlations.append(batch_r)
            
        r_bootstrapped.append(torch.stack(batch_correlations))
    
    bootstrap_scores = torch.cat(r_bootstrapped)
    
    return r, bootstrap_scores


def calculate_rsa_score(
    neural_responses: Union[np.ndarray, torch.Tensor],
    activations: Union[np.ndarray, torch.Tensor],
    correlation: str = "Pearson",
    n_bootstraps: int = 5_000,
) -> float:
    """
    Calculate RSA score between neural responses and model activations.
    """
    # Convert inputs to torch tensors if they're numpy arrays
    if isinstance(neural_responses, np.ndarray):
        neural_responses = torch.from_numpy(neural_responses)
    if isinstance(activations, np.ndarray):
        activations = torch.from_numpy(activations)
    
    # Move to float32 for numerical stability
    neural_responses = neural_responses.float()
    activations = activations.float()
    
    # Compute RSMs
    neural_rsm = compute_rsm(neural_responses)
    activation_rsm = compute_rsm(activations)
    
    # Compute RSA correlation with bootstrapping
    rsa_score, bootstrap_scores = compute_rsa_correlation(
        neural_rsm,
        activation_rsm,
        correlation=correlation,
        n_bootstraps=n_bootstraps,
    )
    
    # Compute confidence intervals
    ci_lower = float(torch.quantile(bootstrap_scores, 0.025))
    ci_upper = float(torch.quantile(bootstrap_scores, 0.975))
    
    return rsa_score


def calculate_cls_accuracy(data_loader, model, device):
    """Calculate classification accuracy."""
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            batch_correct = (predicted == labels).sum().item()
            correct += batch_correct
    
    final_accuracy = 100 * correct / total
    return final_accuracy


def compute_neural_alignment(
    activations_dict: Dict[str, torch.Tensor], 
    neural_data: Dict[str, np.ndarray],
    keys: List[str],
    cfg: Dict
) -> pd.DataFrame:
    """Compute neural alignment scores for each layer using specified metric
    
    Args:
        activations_dict: Dict mapping layer names to activation tensors
        neural_data: Dict mapping stimulus IDs to neural responses
        keys: List of stimulus IDs in the order they were processed
        cfg: Configuration dictionary with metric settings
        
    Returns:
        DataFrame containing alignment scores and metadata for each layer
    """
    results = []
    neural_responses = np.array([neural_data[str(key)] for key in keys])
    metric_name = cfg.get('metric', 'rsa')
    
    # Get metric function based on config
    metric_fn = {
        'rsa': calculate_rsa_score,
        # Add more metrics here as needed
    }.get(metric_name)
    
    if not metric_fn:
        raise ValueError(f"Unknown metric: {metric_name}. Available metrics: ['rsa']")
    
    print(f"Computing {metric_name.upper()} scores...")
    for layer, activations in activations_dict.items():
        # Flatten activations if needed
        activations = activations.flatten(start_dim=1) if activations.ndim > 2 else activations
        score = metric_fn(neural_responses, activations.cpu().numpy())
        print(f"Layer {layer:<20} {metric_name.upper()} Score: {score:.4f}")
        
        # Store results
        result = {
            "layer": layer,
            f"{metric_name}_score": score,
            "metric": metric_name
        }
        result.update(cfg if isinstance(cfg, dict) else vars(cfg))
        results.append(result)
    
    return pd.DataFrame(results)