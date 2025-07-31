"""GPU-based encoding score calculation for neuroscience experiments."""

import torch
import numpy as np
from typing import Dict, List
from visreps.utils import rprint


def pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Pearson correlation between predictions and targets."""
    x_centered = x - x.mean(dim=0)
    y_centered = y - y.mean(dim=0)
    
    # Compute variances
    x_var = (x_centered ** 2).sum(dim=0)
    y_var = (y_centered ** 2).sum(dim=0)
    
    # Handle zero variance cases
    valid_mask = (x_var > 1e-10) & (y_var > 1e-10)
    
    # Initialize correlation with zeros
    corr = torch.zeros(x.shape[1], device=x.device)
    
    # Compute correlation only for valid features
    if valid_mask.any():
        numerator = (x_centered[:, valid_mask] * y_centered[:, valid_mask]).sum(dim=0)
        denominator = torch.sqrt(x_var[valid_mask] * y_var[valid_mask])
        corr[valid_mask] = numerator / denominator
    
    return corr


def ridge_regression_gpu(X_train, y_train, X_test, y_test, alphas=None, device="cuda"):
    """Ridge regression with LOO-CV for alpha selection."""
    alphas = alphas or np.logspace(-4, 4, 10).tolist()
    
    # Ensure we have enough samples for LOO-CV
    if X_train.shape[0] < 3:
        raise ValueError(f"Need at least 3 training samples for LOO-CV, got {X_train.shape[0]}")
    
    # Move to device and center data
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)
    X_mean, y_mean = X_train.mean(0, keepdim=True), y_train.mean(0, keepdim=True)
    X_train_c, y_train_c = X_train - X_mean, y_train - y_mean
    
    # Precompute matrices
    XTX = X_train_c.T @ X_train_c
    XTy = X_train_c.T @ y_train_c
    I = torch.eye(X_train.shape[1], device=device)
    
    # Find best alpha via LOO-CV
    best_alpha, best_score = alphas[0], -float('inf')
    
    for alpha in alphas:
        # Solve system once per alpha
        XTX_reg = XTX + alpha * I
        beta = torch.linalg.solve(XTX_reg, XTy)
        
        # Compute diagonal of hat matrix efficiently
        # H = X @ (X'X + αI)^(-1) @ X'
        # We use the fact that diag(ABC) = sum(A ⊙ (CB)') where ⊙ is element-wise product
        XTX_reg_inv_XT = torch.linalg.solve(XTX_reg, X_train_c.T)
        H_diag = (X_train_c * XTX_reg_inv_XT.T).sum(dim=1)
        
        # Compute LOO predictions
        y_hat = X_train_c @ beta + y_mean
        residuals = y_train - y_hat
        loo_factor = 1 - H_diag
        
        # Avoid division by very small numbers
        valid_loo = loo_factor.abs() > 1e-6
        loo_preds = y_train.clone()
        if valid_loo.any():
            loo_preds[valid_loo] = y_train[valid_loo] - residuals[valid_loo] / loo_factor[valid_loo].unsqueeze(1)
        
        # Compute CV score
        score = pearson_correlation(loo_preds, y_train).mean()
        
        if score > best_score and not torch.isnan(score):
            best_alpha, best_score = alpha, score
    
    # Final model and predictions
    beta_final = torch.linalg.solve(XTX + best_alpha * I, XTy)
    y_pred = (X_test - X_mean) @ beta_final + y_mean
    
    return y_pred, pearson_correlation(y_pred, y_test)


def calculate_encoding_scores(model_features, neural_data, train_fraction=0.8, seed=42, device=None):
    """Calculate encoding scores using ridge regression (80/20 split, LOO-CV for alpha)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Validate inputs
    if len(model_features) != len(neural_data):
        raise ValueError(f"model_features and neural_data must have same number of samples, "
                        f"got {len(model_features)} and {len(neural_data)}")
    
    if len(model_features) < 10:
        raise ValueError(f"Need at least 10 samples for reliable train/test split, got {len(model_features)}")
    
    # Train-test split
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(model_features))
    n_train = int(train_fraction * len(model_features))
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    
    # Run ridge regression
    try:
        predictions, scores = ridge_regression_gpu(
            model_features[train_idx], neural_data[train_idx],
            model_features[test_idx], neural_data[test_idx], 
            device=device
        )
    except torch.linalg.LinAlgError as e:
        raise RuntimeError(f"Ridge regression failed due to numerical issues: {e}")
    
    return {
        'scores': scores.cpu(),
        'mean_score': scores.mean().cpu(),
        'predictions': predictions.cpu(),
        'r_squared': (scores ** 2).cpu(),
        'n_train': n_train,
        'n_test': len(test_idx),
    }


def compute_encoding_alignment(
    cfg: Dict, activations_dict: Dict[str, torch.Tensor], neural_data: torch.Tensor
) -> List[Dict]:
    """Compute encoding alignment (R²) for each layer using ridge regression.
    
    This function follows the same interface as compute_rsa_alignment to allow
    seamless switching between RSA and encoding score analyses.
    
    Args:
        cfg: Configuration dictionary containing at least 'seed'
        activations_dict: Dictionary mapping layer names to activation tensors
        neural_data: Neural data tensor (e.g., fMRI responses)
    
    Returns:
        List of dictionaries, each containing results for a layer with keys:
            - layer: layer name
            - score: mean R² value
            - analysis: "encoding_score"
            - n_train: number of training samples
            - n_test: number of test samples
    """
    rprint("Computing encoding scores with ridge regression", style="info")
    rprint("Using 80/20 train-test split with LOO-CV for alpha selection", style="info")
    
    if not activations_dict:
        raise ValueError("activations_dict is empty")
    
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = cfg.get("seed", 42)
    
    # Store n_train/n_test from first layer (all layers have same split)
    n_train, n_test = None, None
    
    for layer, acts in activations_dict.items():
        # Flatten activations if needed (handle conv layers)
        flat_acts = acts.flatten(start_dim=1) if acts.ndim > 2 else acts
        
        try:
            # Calculate encoding scores
            encoding_results = calculate_encoding_scores(
                model_features=flat_acts,
                neural_data=neural_data,
                train_fraction=0.8,
                seed=seed,
                device=device
            )
            
            # Store n_train/n_test from first successful computation
            if n_train is None:
                n_train = encoding_results['n_train']
                n_test = encoding_results['n_test']
            
            # Use mean R² as the score
            score = float(encoding_results['r_squared'].mean())
            
            rprint(
                f"Layer {layer:<15} Encoding (R²): {score:.4f}",
                style="highlight",
            )
            
        except (ValueError, RuntimeError) as e:
            # If encoding fails for a layer, report it and use NaN
            rprint(f"Layer {layer:<15} Encoding failed: {e}", style="error")
            score = float('nan')
        
        results.append({
            "layer": layer,
            "score": score,
            "analysis": "encoding_score",
            "make_rsm_correlation": "N/A",
            "compare_rsm_correlation": "N/A",
        })
    
    return results


def main():
    """Example usage with synthetic data."""
    # Generate synthetic data
    model_features = torch.randn(1000, 100)
    weights = torch.randn(100, 50)
    neural_data = model_features @ weights + 0.5 * torch.randn(1000, 50)
    
    # Calculate encoding scores
    results = calculate_encoding_scores(model_features, neural_data)
    
    print(f"Mean encoding score (r): {results['mean_score']:.3f}")
    print(f"Mean R²: {results['r_squared'].mean():.3f}")
    print(f"Training samples: {results['n_train']}")
    print(f"Test samples: {results['n_test']}")
    print(f"Score range: [{results['scores'].min():.3f}, {results['scores'].max():.3f}]")


if __name__ == "__main__":
    main()