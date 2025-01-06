import torch


def r2_score(y: torch.Tensor, y_predicted: torch.Tensor) -> torch.Tensor:
    """Compute R² score between true and predicted values.
    
    R² = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²
    where ȳ is the mean of the true values
    """
    y = y.unsqueeze(1) if y.ndim == 1 else y
    y_predicted = y_predicted.unsqueeze(1) if y_predicted.ndim == 1 else y_predicted
    
    # Compute mean of true values
    y_mean = y.mean(dim=-2, keepdim=True)
    
    # Sum of squared errors (numerator)
    sse = ((y - y_predicted) ** 2).sum(dim=-2)
    
    # Total sum of squares (denominator)
    ss = ((y - y_mean) ** 2).sum(dim=-2)
    
    # Handle edge case where variance is 0
    ss = torch.where(ss == 0, torch.ones_like(ss), ss)
    
    return 1 - sse / ss
