import pytest
import torch
from omegaconf import OmegaConf
from unittest.mock import patch, MagicMock, call
import pandas as pd
import numpy as np

from visreps.analysis.rsa import (
    compute_rsm,
    compute_rsm_correlation,
    bootstrap_correlation,
    compute_rsa_alignment
)
from visreps.analysis.metrics import pearson_r, spearman_r

# --- Fixtures ---

@pytest.fixture
def sample_activations():
    """Provides sample activation tensors for RSA tests."""
    torch.manual_seed(123)
    # Simple case: (samples, features)
    acts1 = torch.randn(10, 5)
    # Case with more features: (samples, features)
    acts2 = torch.randn(10, 20)
    # Case needing flattening: (samples, channels, H, W)
    acts3 = torch.randn(10, 3, 4, 4)
    return acts1, acts2, acts3

@pytest.fixture
def sample_rsms(sample_activations):
    """Provides pre-computed RSMs from sample_activations."""
    acts1, acts2, _ = sample_activations
    rsm1 = compute_rsm(acts1)
    rsm2 = compute_rsm(acts2)
    # RSM from a slightly correlated version of acts1
    acts1_corr = acts1 + torch.randn(10, 5) * 0.5
    rsm3 = compute_rsm(acts1_corr)
    return rsm1, rsm2, rsm3 # (10x10 tensors)

# --- Tests for RSM computation ---

def test_compute_rsm(sample_activations):
    """Test RSM computation produces correct shape and properties."""
    acts1, _, _ = sample_activations
    n_samples = acts1.shape[0]
    rsm = compute_rsm(acts1)

    assert isinstance(rsm, torch.Tensor)
    assert rsm.shape == (n_samples, n_samples)
    # Diagonal should be constant (may not be exactly 1.0 due to implementation)
    diag_vals = torch.diag(rsm)
    assert torch.allclose(diag_vals, diag_vals[0], atol=1e-6)
    # RSM should be symmetric
    assert torch.allclose(rsm, rsm.T, atol=1e-6)

# --- Tests for RSM correlation ---

@pytest.mark.parametrize("correlation_metric, corr_func", [
    ("Pearson", pearson_r),
    ("Spearman", spearman_r)
])
def test_compute_rsm_correlation(sample_rsms, correlation_metric, corr_func):
    """Test correlation between RSMs using different metrics."""
    rsm1, rsm2, rsm3 = sample_rsms

    # Correlation of RSM with itself should be ~1.0
    corr_self = compute_rsm_correlation(rsm1, rsm1, correlation=correlation_metric)
    assert corr_self == pytest.approx(1.0)

    # Correlation between RSMs from different activations
    corr_diff = compute_rsm_correlation(rsm1, rsm2, correlation=correlation_metric)
    assert isinstance(corr_diff, float)
    assert -1.0 <= corr_diff <= 1.0

    # Correlation between RSMs from similar activations (rsm1, rsm3)
    corr_sim = compute_rsm_correlation(rsm1, rsm3, correlation=correlation_metric)
    assert isinstance(corr_sim, float)
    assert corr_sim > corr_diff # Expect higher correlation for similar RSMs
    assert -1.0 <= corr_sim <= 1.0

# --- Tests for Bootstrap Correlation ---

def test_bootstrap_correlation(sample_rsms):
    """Test bootstrap correlation estimation."""
    rsm1, _, rsm3 = sample_rsms # Use rsm1 and rsm3 (correlated)
    n_bootstraps = 100
    subsample_fraction = 0.8
    seed = 42

    bootstrap_scores = bootstrap_correlation(
        rsm1, rsm3,
        n_bootstraps=n_bootstraps,
        subsample_fraction=subsample_fraction,
        correlation="Pearson",
        seed=seed
    )

    assert isinstance(bootstrap_scores, torch.Tensor)
    assert bootstrap_scores.shape == (n_bootstraps,)
    # Scores should be centered around the true correlation
    true_corr = compute_rsm_correlation(rsm1, rsm3, correlation="Pearson")
    assert bootstrap_scores.mean().item() == pytest.approx(true_corr, abs=0.1)
    assert bootstrap_scores.std().item() > 0 # Expect some variance

# --- Tests for compute_rsa_alignment --- 

# Apply patches directly to the test function
@patch("visreps.analysis.rsa.bootstrap_correlation")
@patch("visreps.analysis.rsa.compute_rsm_correlation")
@patch("visreps.analysis.rsa.compute_rsm")
def test_compute_rsa_alignment(mock_compute_rsm, mock_rsm_corr, mock_bootstrap):
    """Test the main RSA alignment function with mocks."""
    # Mock config
    cfg = OmegaConf.create({
        "correlation": "Spearman",
        "n_bootstraps": 50,
        "subsample_fraction": 0.7,
        "do_bootstrap": True, # Test with bootstrap enabled
        "seed": 99 # For bootstrap mocking if needed
    })

    # Mock data
    mock_activations_dict = {
        "layer_flat": torch.randn(20, 10), # Already flat
        "layer_conv": torch.randn(20, 5, 3, 3) # Needs flattening
    }
    mock_neural_data = torch.randn(20, 15)

    # Mock return values
    mock_neural_rsm = torch.randn(20, 20)
    mock_layer1_rsm = torch.randn(20, 20)
    mock_layer2_rsm = torch.randn(20, 20)
    mock_compute_rsm.side_effect = [mock_neural_rsm, mock_layer1_rsm, mock_layer2_rsm]

    mock_rsm_corr.side_effect = [0.75, 0.55] # Layer1 score, Layer2 score

    mock_bootstrap_scores1 = torch.linspace(0.7, 0.8, cfg.n_bootstraps)
    mock_bootstrap_scores2 = torch.linspace(0.5, 0.6, cfg.n_bootstraps)
    mock_bootstrap.side_effect = [mock_bootstrap_scores1, mock_bootstrap_scores2]

    # --- Run the function ---
    results = compute_rsa_alignment(cfg, mock_activations_dict, mock_neural_data)
    # --- End Run ---

    # Assertions
    assert isinstance(results, list)
    assert len(results) == 2 # One per layer

    # Check calls to compute_rsm
    # We cannot directly compare tensors in mock calls easily.
    # Check the number of calls and maybe types/shapes if needed.
    assert mock_compute_rsm.call_count == 3 
    # First call is neural data
    assert torch.equal(mock_compute_rsm.call_args_list[0][0][0], mock_neural_data)
    # Second call is flat layer
    assert torch.equal(mock_compute_rsm.call_args_list[1][0][0], mock_activations_dict["layer_flat"])
    # Third call is flattened conv layer
    assert torch.equal(mock_compute_rsm.call_args_list[2][0][0], mock_activations_dict["layer_conv"].flatten(start_dim=1))

    # Check calls to compute_rsm_correlation
    expected_rsm_corr_calls = [
        call(mock_layer1_rsm, mock_neural_rsm, cfg.correlation),
        call(mock_layer2_rsm, mock_neural_rsm, cfg.correlation)
    ]
    # Use call_args_list for potentially more robust comparison if call() fails
    assert mock_rsm_corr.call_count == 2
    assert mock_rsm_corr.call_args_list[0] == expected_rsm_corr_calls[0]
    assert mock_rsm_corr.call_args_list[1] == expected_rsm_corr_calls[1]
    # mock_rsm_corr.assert_has_calls(expected_rsm_corr_calls) # This might still fail due to tensor comparison

    # Check calls to bootstrap_correlation
    if cfg.do_bootstrap:
        expected_bootstrap_calls = [
            call(mock_layer1_rsm, mock_neural_rsm,
                 n_bootstraps=cfg.n_bootstraps, subsample_fraction=cfg.subsample_fraction,
                 correlation=cfg.correlation),
            call(mock_layer2_rsm, mock_neural_rsm,
                 n_bootstraps=cfg.n_bootstraps, subsample_fraction=cfg.subsample_fraction,
                 correlation=cfg.correlation)
        ]
        assert mock_bootstrap.call_count == 2
        assert mock_bootstrap.call_args_list[0] == expected_bootstrap_calls[0]
        assert mock_bootstrap.call_args_list[1] == expected_bootstrap_calls[1]
        # mock_bootstrap.assert_has_calls(expected_bootstrap_calls)
    else:
        mock_bootstrap.assert_not_called()

    # Check results content for Layer 1
    res1 = results[0]
    assert res1["layer"] == "layer_flat"
    assert res1["analysis"] == "rsa"
    assert res1["correlation"] == cfg.correlation
    assert res1["score"] == pytest.approx(0.75)
    if cfg.do_bootstrap:
        assert res1["bootstrap_mean"] == pytest.approx(mock_bootstrap_scores1.mean().item())
        assert res1["bootstrap_std"] == pytest.approx(mock_bootstrap_scores1.std().item())
        assert "bootstrap_ci_lower" in res1
        assert "bootstrap_ci_upper" in res1

    # Check results content for Layer 2
    res2 = results[1]
    assert res2["layer"] == "layer_conv"
    assert res2["analysis"] == "rsa"
    assert res2["correlation"] == cfg.correlation
    assert res2["score"] == pytest.approx(0.55)
    if cfg.do_bootstrap:
        assert res2["bootstrap_mean"] == pytest.approx(mock_bootstrap_scores2.mean().item())
        assert res2["bootstrap_std"] == pytest.approx(mock_bootstrap_scores2.std().item())
        assert "bootstrap_ci_lower" in res2
        assert "bootstrap_ci_upper" in res2 