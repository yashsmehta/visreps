from typing import Self, Dict, List

import torch
from tqdm.auto import tqdm
from visreps.analysis.metrics import r2_score
from visreps.utils import rprint

from visreps.analysis.regression._definition import Regression

EPSILON = 1e-15


class LinearRegression(Regression):
    def __init__(
        self: Self,
        *,
        fit_intercept: bool = True,
        l2_penalty: float | torch.Tensor | None = None,
        rcond: float | None = None,
        driver: str | None = None,
        allow_ols_on_cuda: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.coefficients: torch.Tensor | None = None
        self.intercept: torch.Tensor | None = None

        self.fit_intercept = fit_intercept
        self.l2_penalty = l2_penalty
        self.rcond = rcond
        self.driver = driver
        self.allow_ols_on_cuda = allow_ols_on_cuda
        self.device = device

    def to(self: Self, device: torch.device | str) -> None:
        if self.coefficients is not None:
            self.coefficients = self.coefficients.to(device)
        if self.intercept is not None:
            self.intercept = self.intercept.to(device)

    def fit(
        self: Self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        x = torch.clone(x).to(self.device)
        # TODO check why x and y are not necessarily on same device
        y = torch.clone(y).to(x.device)

        x = x.unsqueeze(dim=-1) if x.ndim == 1 else x
        y = y.unsqueeze(dim=-1) if y.ndim == 1 else y

        # many sets of predictors, only 1 set of targets
        if x.ndim == 3 and y.ndim == 2:
            y = y.unsqueeze(0)

        n_samples, n_features = x.shape[-2], x.shape[-1]

        # TODO: underdetermined systems on CUDA use a different driver
        if (
            (not self.allow_ols_on_cuda)
            and (self.l2_penalty is None)
            and (n_samples < n_features)
        ):
            x = x.to(torch.device("cpu"))
            y = y.to(torch.device("cpu"))

        if y.shape[-2] != n_samples:
            error = (
                f"number of samples in x and y must be equal (x={n_samples},"
                f" y={y.shape[-2]})",
            )
            raise ValueError(error)

        if self.fit_intercept:
            x_mean = x.mean(dim=-2, keepdim=True)
            x -= x_mean
            y_mean = y.mean(dim=-2, keepdim=True)
            y -= y_mean

        if self.l2_penalty is None:
            self.coefficients, _, _, _ = torch.linalg.lstsq(
                x,
                y,
                rcond=self.rcond,
                driver=self.driver,
            )
        else:
            if isinstance(self.l2_penalty, float | int) or (
                isinstance(self.l2_penalty, torch.Tensor)
                and self.l2_penalty.numel() == 1
            ):
                l2_penalty = self.l2_penalty * torch.ones(y.shape[-1], device=x.device)
            elif isinstance(self.l2_penalty, torch.Tensor):
                l2_penalty = self.l2_penalty.to(x.device)

            u, s, vt = torch.linalg.svd(x, full_matrices=False)
            idx = s > EPSILON
            s_nnz = s[idx].unsqueeze(-1)
            d = torch.zeros(
                size=(len(s), l2_penalty.numel()),
                dtype=x.dtype,
                device=x.device,
            )
            d[idx] = s_nnz / (s_nnz**2 + l2_penalty)
            self.coefficients = vt.transpose(-2, -1) @ (d * (u.transpose(-2, -1) @ y))

        if self.fit_intercept:
            self.intercept = y_mean - x_mean @ self.coefficients
        else:
            self.intercept = torch.zeros(1)

    def predict(self: Self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.coefficients.device) @ self.coefficients + self.intercept

    def weights(self: Self) -> torch.Tensor:
        return self.coefficients


def compute_linear_regression_alignment(
    cfg: Dict,
    activations_dict: Dict[str, torch.Tensor],
    neural_data: torch.Tensor,
) -> List[Dict]:
    """Compute linear regression alignment between model layer activations and neural data
    
    Args:
        cfg: Configuration dictionary with analysis settings
        activations_dict: Dict mapping layer names to activation tensors
        neural_data: Neural response tensor (n_samples, n_voxels)
        
    Returns:
        List of dicts containing alignment scores and metadata for each layer
    """
    results = []
    
    # Get regression parameters from config
    fit_intercept = cfg.get('fit_intercept', True)
    l2_penalty = cfg.get('l2_penalty', None)
    n_bootstraps = cfg.get('n_bootstraps', 500)
    subsample_fraction = cfg.get('subsample_fraction', 0.9)
    
    rprint(f"Computing linear regression scores...", style="info")
    
    # Compute regression for each layer
    for layer, activations in activations_dict.items():
        print(f"\nProcessing Layer: {layer}")
        print(f"Activations shape: {activations.shape}")
        
        # Flatten activations if needed
        if activations.ndim > 2:
            activations = activations.flatten(start_dim=1)
            
        # Initialize regression model
        model = LinearRegression(
            fit_intercept=fit_intercept,
            l2_penalty=l2_penalty,
            device=activations.device
        )
        
        # Fit model and compute score
        model.fit(activations, neural_data)
        predictions = model.predict(activations)
        # Compute R² score for each voxel and take mean
        scores = r2_score(neural_data, predictions)
        score = float(scores.mean())  # Average R² across all voxels
        rprint(f"Layer {layer:<20} Mean R² Score: {score:.4f}", style="highlight")
        
        # Compute bootstrap scores if requested
        bootstrap_scores = []
        if n_bootstraps > 0:
            n_samples = len(activations)
            n_subset = int(n_samples * subsample_fraction)
            rng = torch.Generator(device=activations.device).manual_seed(0)
            
            for _ in tqdm(range(n_bootstraps), desc="Bootstrap", leave=False):
                # Sample indices
                idx = torch.randperm(n_samples, generator=rng, device=activations.device)[:n_subset]
                # Get subsets
                X_subset = activations[idx]
                y_subset = neural_data[idx]
                # Fit and score
                model.fit(X_subset, y_subset)
                pred_subset = model.predict(X_subset)
                # Compute mean R² across voxels for this bootstrap sample
                bootstrap_scores.append(float(r2_score(y_subset, pred_subset).mean()))
                
            bootstrap_scores = torch.tensor(bootstrap_scores)
            bootstrap_mean = float(bootstrap_scores.mean())
            bootstrap_std = float(bootstrap_scores.std())
            bootstrap_ci = bootstrap_scores.quantile(torch.tensor([0.025, 0.975]))
            print(f"Bootstrap results - Mean: {bootstrap_mean:.4f} ± {bootstrap_std:.4f}")
        else:
            bootstrap_mean = score
            bootstrap_std = 0.0
            bootstrap_ci = torch.tensor([score, score])
            
        # Store results
        results.append({
            "layer": layer,
            "score": score,
            "analysis": "linear_regression",
            "bootstrap_mean": bootstrap_mean,
            "bootstrap_std": bootstrap_std,
            "bootstrap_ci_lower": float(bootstrap_ci[0]),
            "bootstrap_ci_upper": float(bootstrap_ci[1])
        })
        
    return results
