from typing import Self, Dict, List, Union

import torch
from tqdm.auto import tqdm
from visreps.analysis.metrics import r2_score
from visreps.analysis.regression._utilities import regression_cv
from visreps.analysis.regression._definition import Regression
from visreps.utils import rprint


EPSILON = 1e-15


class LinearRegression(Regression):
    def __init__(
        self: Self,
        *,
        fit_intercept: bool = True,
        l2_penalty: Union[float, torch.Tensor, None] = None,
        rcond: float | None = None,
        driver: str | None = None,
        allow_ols_on_cuda: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        # Model parameters
        self.coefficients: torch.Tensor | None = None
        self.intercept:   torch.Tensor | None = None

        # Hyperparameters
        self.fit_intercept = fit_intercept
        self.l2_penalty = l2_penalty
        self.rcond = rcond
        self.driver = driver
        self.allow_ols_on_cuda = allow_ols_on_cuda
        self.device = device

    def to(self: Self, device: Union[torch.device, str]) -> None:
        if self.coefficients is not None:
            self.coefficients = self.coefficients.to(device)
        if self.intercept is not None:
            self.intercept = self.intercept.to(device)

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        # 1) Move inputs to device
        x = x.to(self.device)
        y = y.to(self.device)

        # 2) Flatten so x has shape (N, features), y has shape (N, outputs)
        #    (If x, y are already 2D or bigger, adjust as needed.)
        n_samples = x.shape[0]
        x = x.reshape(n_samples, -1)
        y = y.reshape(n_samples, -1)

        # 3) Center data if requested
        if self.fit_intercept:
            x_mean = x.mean(dim=0, keepdim=True)
            y_mean = y.mean(dim=0, keepdim=True)

            # Shift / center
            x_centered = x - x_mean
            y_centered = y - y_mean
        else:
            x_centered = x
            y_centered = y

        # 4) Solve the linear system
        if self.l2_penalty is None:
            # Ordinary Least Squares via torch.linalg.lstsq
            # returns (solution, residuals, rank, singular_values)
            self.coefficients, _, _, _ = torch.linalg.lstsq(
                x_centered, y_centered,
                rcond=self.rcond,
                driver=self.driver
            )
        else:
            # Ridge regression
            l2_penalty = float(self.l2_penalty)
            xtx = x_centered.T @ x_centered  # (features, features)
            xty = x_centered.T @ y_centered  # (features, outputs)
            identity = torch.eye(xtx.shape[0], device=x.device)
            regularized_xtx = xtx + l2_penalty * identity

            # Try Cholesky first
            try:
                L = torch.linalg.cholesky(regularized_xtx)
                self.coefficients = torch.cholesky_solve(xty, L)
            except RuntimeError:
                # Fallback to standard solver
                self.coefficients = torch.linalg.solve(regularized_xtx, xty)

        # 5) Compute intercept
        if self.fit_intercept:
            # shape = (1, outputs)
            self.intercept = y_mean - x_mean @ self.coefficients
        else:
            # No intercept
            self.intercept = torch.zeros((1, y.shape[1]), device=x.device)

    def predict(self: Self, x: torch.Tensor) -> torch.Tensor:
        # Move to device & flatten similarly
        device = self.coefficients.device
        x = x.to(device)
        n_samples = x.shape[0]
        x = x.reshape(n_samples, -1)

        return x @ self.coefficients + self.intercept

    def weights(self: Self) -> torch.Tensor:
        return self.coefficients


##############################################################################
# Cross-Validated Alignment Function
##############################################################################

def compute_linear_regression_alignment(
    cfg: Dict,
    activations_dict: Dict[str, torch.Tensor],
    neural_data: torch.Tensor,
) -> List[Dict]:
    """
    Compute cross-validated linear regression alignment between
    model layer activations and neural data.
    """
    results = []
    
    fit_intercept = cfg.get('fit_intercept', True)
    l2_penalty = cfg.get('l2_penalty', 1.0)
    n_folds = cfg.get('n_folds', 5)
    shuffle = cfg.get('shuffle', True)
    seed = cfg.get('seed', 42)
    
    rprint(f"Computing cross-validated linear regression scores...", style="info")
    
    for layer, activations in activations_dict.items():
        print(f"\nProcessing Layer: {layer}")
        print(f"Activations shape: {activations.shape}")

        # Prepare the model arguments *but do NOT instantiate the model yet*
        model_kwargs = {
            'fit_intercept': fit_intercept,
            'l2_penalty': l2_penalty,
            'device': activations.device
        }
        
        # Perform K-fold cross-validation on this layer’s data
        # using our updated regression_cv, which re‐instantiates
        # the model each fold.
        y_true_folds, y_pred_folds = regression_cv(
            x=activations,
            y=neural_data,
            model_class=LinearRegression,  # Pass in the class, not an instance
            model_kwargs=model_kwargs,     # Additional kwargs
            n_folds=n_folds,
            shuffle=shuffle,
            seed=seed
        )
        
        # Compute R² scores for each fold
        fold_scores = []
        for y_true, y_pred in zip(y_true_folds, y_pred_folds):
            scores = r2_score(y_true, y_pred)  # shape (D,) if multiple outputs
            fold_scores.append(float(scores.mean()))  # average R² across outputs

        # Aggregate folds
        fold_scores_tensor = torch.tensor(fold_scores, dtype=torch.float32)
        cv_mean = float(fold_scores_tensor.mean())
        cv_std = float(fold_scores_tensor.std())
        cv_ci = fold_scores_tensor.quantile(torch.tensor([0.025, 0.975]))

        rprint(f"Layer {layer:<20} Mean CV R² Score: {cv_mean:.4f} ± {cv_std:.4f}",
               style="highlight")
        
        # Add result
        results.append({
            "layer": layer,
            "analysis": "linear_regression_cv",
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "cv_ci_lower": float(cv_ci[0]),
            "cv_ci_upper": float(cv_ci[1]),
            "n_folds": n_folds
        })
        
    return results