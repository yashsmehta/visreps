import pytest
import torch
from omegaconf import OmegaConf
from unittest.mock import patch, MagicMock, call
import pandas as pd

from visreps.analysis.regression.linear_regression import (
    LinearRegression,
    compute_linear_regression_alignment
)
# No need to mock regression_cv from here if patching below
# from visreps.analysis.regression._utilities import regression_cv
from visreps.analysis.metrics import r2_score # Keep for direct use in class tests

# --- Fixtures ---

@pytest.fixture
def sample_data():
    """Provides simple X, y data for regression tests."""
    torch.manual_seed(0)
    X = torch.randn(20, 5)
    # Simple linear relationship + noise
    true_coef = torch.tensor([[1.0], [-0.5], [2.0], [0.0], [-1.5]]) # (5, 1)
    true_intercept = torch.tensor([[0.8]]) # (1, 1)
    y = X @ true_coef + true_intercept + torch.randn(20, 1) * 0.1
    return X, y, true_coef, true_intercept

@pytest.fixture
def sample_multi_output_data():
    """Provides X, y data with multiple outputs."""
    torch.manual_seed(42)
    X = torch.randn(30, 4)
    # Output 1
    true_coef1 = torch.tensor([[1.0], [-1.0], [0.5], [2.0]]) # (4, 1)
    true_intercept1 = torch.tensor([[0.2]]) # (1, 1)
    y1 = X @ true_coef1 + true_intercept1 + torch.randn(30, 1) * 0.2
    # Output 2
    true_coef2 = torch.tensor([[-0.2], [1.5], [-1.8], [0.0]]) # (4, 1)
    true_intercept2 = torch.tensor([[-1.1]]) # (1, 1)
    y2 = X @ true_coef2 + true_intercept2 + torch.randn(30, 1) * 0.3
    y = torch.cat([y1, y2], dim=1) # (30, 2)
    true_coef = torch.cat([true_coef1, true_coef2], dim=1) # (4, 2)
    true_intercept = torch.cat([true_intercept1, true_intercept2], dim=1) # (1, 2)
    return X, y, true_coef, true_intercept

# --- Tests for LinearRegression Class ---

@pytest.mark.parametrize("fit_intercept", [True, False])
def test_linear_regression_ols_fit_predict(sample_data, fit_intercept):
    """Test OLS fit and predict with and without intercept."""
    X, y, true_coef, true_intercept = sample_data
    # Explicitly create model on CPU for testing consistency
    device = torch.device("cpu")
    model = LinearRegression(fit_intercept=fit_intercept, l2_penalty=None, device=device)
    
    # Ensure data is on the same device as the model expects (CPU)
    X, y, true_coef, true_intercept = X.to(device), y.to(device), true_coef.to(device), true_intercept.to(device)
    
    model.fit(X, y)

    # Check shapes
    assert model.coefficients is not None
    assert model.intercept is not None
    assert model.coefficients.shape == (X.shape[1], y.shape[1]) # (features, outputs)
    assert model.intercept.shape == (1, y.shape[1])

    # Check fitted values (approximate due to noise)
    if fit_intercept:
        # Compare centered coefficients
        assert torch.allclose(model.coefficients, true_coef, atol=0.1)
        assert torch.allclose(model.intercept, true_intercept, atol=0.1)
    else:
        # Refit with sklearn for comparison when no intercept (pytorch lstsq assumes centering)
        # Only import sklearn if needed for this specific test case
        pytest.importorskip("sklearn.linear_model")
        from sklearn.linear_model import LinearRegression as SklearnLR
        sk_model = SklearnLR(fit_intercept=False).fit(X.numpy(), y.numpy())
        # Ensure sklearn results are tensors on the correct device
        sk_coef_tensor = torch.from_numpy(sk_model.coef_.T).to(device).to(model.coefficients.dtype)
        assert torch.allclose(model.coefficients, sk_coef_tensor, atol=1e-5)
        assert torch.allclose(model.intercept, torch.zeros_like(model.intercept))

    # Check prediction
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    # High R2 score expected only if intercept is allowed for this data
    if fit_intercept:
        r2 = r2_score(y, y_pred)
        assert r2.ndim == 1 and r2.shape[0] == y.shape[1]
        assert torch.all(r2 > 0.95)

    # Check weights() method
    assert torch.equal(model.weights(), model.coefficients)

@pytest.mark.parametrize("fit_intercept", [True, False])
def test_linear_regression_ridge_fit_predict(sample_data, fit_intercept):
    """Test Ridge regression fit and predict."""
    X, y, _, _ = sample_data
    l2_penalty = 10.0
    device = torch.device("cpu")
    model = LinearRegression(fit_intercept=fit_intercept, l2_penalty=l2_penalty, device=device)
    X, y = X.to(device), y.to(device)
    model.fit(X, y)

    # Check shapes
    assert model.coefficients is not None
    assert model.intercept is not None
    assert model.coefficients.shape == (X.shape[1], y.shape[1])
    assert model.intercept.shape == (1, y.shape[1])

    pytest.importorskip("sklearn.linear_model")
    from sklearn.linear_model import Ridge as SklearnRidge
    sk_model = SklearnRidge(alpha=l2_penalty, fit_intercept=fit_intercept).fit(X.numpy(), y.numpy())
    
    # Ensure expected_coef has the same shape as model.coefficients
    expected_coef_np = sk_model.coef_.T # Shape (features,) or (features, outputs)
    if expected_coef_np.ndim == 1 and model.coefficients.ndim == 2:
         expected_coef_np = expected_coef_np.reshape(-1, 1) # Reshape (features,) -> (features, 1)
    expected_coef = torch.from_numpy(expected_coef_np).to(device).to(model.coefficients.dtype)

    if fit_intercept:
        expected_intercept = torch.from_numpy(sk_model.intercept_.reshape(1, -1)).to(device).to(model.intercept.dtype)
        assert torch.allclose(model.intercept, expected_intercept, atol=1e-4)
    else:
        assert isinstance(sk_model.intercept_, float)
        assert sk_model.intercept_ == pytest.approx(0.0)
        assert torch.allclose(model.intercept, torch.zeros_like(model.intercept))

    # Compare coefficients after ensuring shapes match
    assert model.coefficients.shape == expected_coef.shape
    assert torch.allclose(model.coefficients, expected_coef, atol=1e-4)

    # Check prediction
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape

def test_linear_regression_multi_output(sample_multi_output_data):
    """Test LinearRegression with multiple outputs."""
    X, y, true_coef, true_intercept = sample_multi_output_data
    # Explicitly create models on CPU
    device = torch.device("cpu")
    model_ols = LinearRegression(fit_intercept=True, l2_penalty=None, device=device)
    model_ridge = LinearRegression(fit_intercept=True, l2_penalty=1.0, device=device)
    
    # Ensure data is on CPU
    X, y, true_coef, true_intercept = X.to(device), y.to(device), true_coef.to(device), true_intercept.to(device)

    # Fit OLS
    model_ols.fit(X, y)
    assert model_ols.coefficients.shape == (X.shape[1], y.shape[1]) # (4, 2)
    assert model_ols.intercept.shape == (1, y.shape[1]) # (1, 2)
    assert torch.allclose(model_ols.coefficients, true_coef, atol=0.2)
    assert torch.allclose(model_ols.intercept, true_intercept, atol=0.2)
    y_pred_ols = model_ols.predict(X)
    assert y_pred_ols.shape == y.shape
    r2_ols = r2_score(y, y_pred_ols)
    assert r2_ols.shape == (y.shape[1],) # Check shape is (num_outputs,)
    assert torch.all(r2_ols > 0.9) # Check both outputs

    # Fit Ridge
    model_ridge.fit(X, y)
    assert model_ridge.coefficients.shape == (X.shape[1], y.shape[1])
    assert model_ridge.intercept.shape == (1, y.shape[1])
    y_pred_ridge = model_ridge.predict(X)
    assert y_pred_ridge.shape == y.shape
    r2_ridge = r2_score(y, y_pred_ridge)
    assert r2_ridge.shape == (y.shape[1],)
    assert torch.all(r2_ridge > 0.8) # Ridge might slightly lower R2

# --- Tests for compute_linear_regression_alignment --- 

# Apply patches directly to the test function
@patch("visreps.analysis.regression.linear_regression.r2_score")
@patch("visreps.analysis.regression.linear_regression.regression_cv")
def test_compute_linear_regression_alignment(mock_regression_cv, mock_r2_score):
    """Test the cross-validated regression alignment function with mocks."""
    # Mock config
    cfg = OmegaConf.create({
        "fit_intercept": True,
        "l2_penalty": 5.0,
        "n_folds": 3,
        "shuffle": False,
        "seed": 123
    })

    # Mock data
    mock_activations = {
        "layer1": torch.randn(30, 10),
        "layer2": torch.randn(30, 20)
    }
    mock_neural_data = torch.randn(30, 5)
    device = torch.device("cpu")
    mock_activations["layer1"] = mock_activations["layer1"] .to(device)
    mock_activations["layer2"] = mock_activations["layer2"] .to(device)
    mock_neural_data = mock_neural_data.to(device)

    # Mock return values for dependencies
    # regression_cv returns lists of y_true, y_pred tensors for each fold
    mock_regression_cv.side_effect = [
        # Layer 1 returns (3 folds)
        ([torch.randn(10, 5, device=device) for _ in range(3)], [torch.randn(10, 5, device=device) for _ in range(3)]),
        # Layer 2 returns (3 folds)
        ([torch.randn(10, 5, device=device) for _ in range(3)], [torch.randn(10, 5, device=device) for _ in range(3)])
    ]
    # r2_score returns a tensor of scores (one per output dim) for each fold call
    mock_r2_score.side_effect = [
        # Layer 1, Fold 1-3 calls -> shape (n_outputs,)
        torch.tensor([0.8, 0.7, 0.75, 0.8, 0.7]), torch.tensor([0.85, 0.75, 0.8, 0.9, 0.6]), torch.tensor([0.75, 0.65, 0.7, 0.6, 0.7]),
        # Layer 2, Fold 1-3 calls -> shape (n_outputs,)
        torch.tensor([0.6, 0.5, 0.55, 0.6, 0.4]), torch.tensor([0.65, 0.55, 0.6, 0.7, 0.5]), torch.tensor([0.55, 0.45, 0.5, 0.4, 0.5])
    ]

    # --- Run the function ---
    results = compute_linear_regression_alignment(cfg, mock_activations, mock_neural_data)
    # --- End Run ---

    # Assertions
    assert isinstance(results, list)
    assert len(results) == 2 # One result per layer

    # Check calls to regression_cv
    expected_model_kwargs = {
        'fit_intercept': cfg.fit_intercept,
        'l2_penalty': cfg.l2_penalty,
        'device': device
    }
    expected_calls_cv = [
        call(
            x=mock_activations["layer1"], 
            y=mock_neural_data, 
            model_class=LinearRegression, 
            model_kwargs=expected_model_kwargs, 
            n_folds=cfg.n_folds, 
            shuffle=cfg.shuffle, 
            seed=cfg.seed
        ),
        call(
            x=mock_activations["layer2"], 
            y=mock_neural_data, 
            model_class=LinearRegression, 
            model_kwargs=expected_model_kwargs, 
            n_folds=cfg.n_folds, 
            shuffle=cfg.shuffle, 
            seed=cfg.seed
        )
    ]
    mock_regression_cv.assert_has_calls(expected_calls_cv)

    # Check calls to r2_score (based on the side effect structure)
    assert mock_r2_score.call_count == 6 # 2 layers * 3 folds

    # Check results content (based on mocked r2 scores)
    # Layer 1: mean R2s across outputs per fold = [0.75, 0.78, 0.68]. Mean = 0.7367. Std = ~0.05
    l1_fold_means = [torch.tensor([0.8, 0.7, 0.75, 0.8, 0.7]).mean().item(), 
                       torch.tensor([0.85, 0.75, 0.8, 0.9, 0.6]).mean().item(), 
                       torch.tensor([0.75, 0.65, 0.7, 0.6, 0.7]).mean().item()]
    l1_mean = torch.tensor(l1_fold_means).mean().item()
    l1_std = torch.tensor(l1_fold_means).std().item()
    assert results[0]["layer"] == "layer1"
    assert results[0]["analysis"] == "linear_regression_cv"
    assert results[0]["cv_mean"] == pytest.approx(l1_mean)
    assert results[0]["cv_std"] == pytest.approx(l1_std)
    assert "cv_ci_lower" in results[0]
    assert "cv_ci_upper" in results[0]
    assert results[0]["n_folds"] == cfg.n_folds

    # Layer 2: mean R2s across outputs per fold = [0.53, 0.6, 0.46]. Mean = 0.53. Std = ~0.07
    l2_fold_means = [torch.tensor([0.6, 0.5, 0.55, 0.6, 0.4]).mean().item(), 
                       torch.tensor([0.65, 0.55, 0.6, 0.7, 0.5]).mean().item(), 
                       torch.tensor([0.55, 0.45, 0.5, 0.4, 0.5]).mean().item()]
    l2_mean = torch.tensor(l2_fold_means).mean().item()
    l2_std = torch.tensor(l2_fold_means).std().item()
    assert results[1]["layer"] == "layer2"
    assert results[1]["analysis"] == "linear_regression_cv"
    assert results[1]["cv_mean"] == pytest.approx(l2_mean)
    assert results[1]["cv_std"] == pytest.approx(l2_std)
    assert "cv_ci_lower" in results[1]
    assert "cv_ci_upper" in results[1]
    assert results[1]["n_folds"] == cfg.n_folds 