import pytest
import torch
from visreps.analysis.metrics import r2_score

# --- Test Cases for r2_score ---

def test_r2_score_perfect_fit():
    """Test R2 score when prediction perfectly matches the target."""
    y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
    # Expected R2 = 1.0
    assert r2_score(y_true, y_pred).item() == pytest.approx(1.0)

def test_r2_score_mean_prediction():
    """Test R2 score when prediction is always the mean of the target."""
    y_true = torch.tensor([1.0, 2.0, 3.0, 4.0]) # Mean is 2.5
    y_pred = torch.tensor([2.5, 2.5, 2.5, 2.5])
    # Expected R2 = 0.0
    assert r2_score(y_true, y_pred).item() == pytest.approx(0.0)

def test_r2_score_imperfect_fit():
    """Test R2 score for an imperfect fit."""
    y_true = torch.tensor([3.0, -0.5, 2.0, 7.0])
    y_pred = torch.tensor([2.5, 0.0, 2.0, 8.0])
    # Manually calculated R2 = 1 - SSE/SST
    # SSE = (3-2.5)**2 + (-0.5-0)**2 + (2-2)**2 + (7-8)**2 = 0.25 + 0.25 + 0 + 1 = 1.5
    # Mean(y_true) = (3 - 0.5 + 2 + 7) / 4 = 11.5 / 4 = 2.875
    # SST = (3-2.875)**2 + (-0.5-2.875)**2 + (2-2.875)**2 + (7-2.875)**2 
    #     = 0.125**2 + (-3.375)**2 + (-0.875)**2 + 4.125**2
    #     = 0.015625 + 11.390625 + 0.765625 + 17.015625 = 29.1875
    # R2 = 1 - 1.5 / 29.1875 = 1 - 0.05139... = 0.9486
    expected_r2 = 0.94860869565
    assert r2_score(y_true, y_pred).item() == pytest.approx(expected_r2)

def test_r2_score_negative_fit():
    """Test R2 score when the fit is worse than predicting the mean."""
    y_true = torch.tensor([1.0, 2.0, 3.0])
    y_pred = torch.tensor([3.0, 2.0, 1.0]) # Worse than predicting mean (2.0)
    # SSE = (1-3)**2 + (2-2)**2 + (3-1)**2 = 4 + 0 + 4 = 8
    # Mean(y_true) = 2.0
    # SST = (1-2)**2 + (2-2)**2 + (3-2)**2 = 1 + 0 + 1 = 2
    # R2 = 1 - 8 / 2 = -3.0
    expected_r2 = -3.0
    assert r2_score(y_true, y_pred).item() == pytest.approx(expected_r2)

def test_r2_score_zero_variance():
    """Test R2 score when the true values have zero variance."""
    y_true = torch.tensor([2.0, 2.0, 2.0, 2.0])
    y_pred_perfect = torch.tensor([2.0, 2.0, 2.0, 2.0])
    y_pred_imperfect = torch.tensor([1.0, 3.0, 2.0, 2.0])
    # R2 should be 1.0 for perfect prediction even with zero variance
    assert r2_score(y_true, y_pred_perfect).item() == pytest.approx(1.0)
    # R2 should be < 1 (specifically -1.0) for imperfect prediction when SST=0
    # R2 = 1 - SSE / max(SST, eps) -> 1 - SSE / 1.0 = 1 - SSE
    # SSE = (2-1)^2 + (2-3)^2 + (2-2)^2 + (2-2)^2 = 1 + 1 + 0 + 0 = 2
    # Expected R2 = 1 - 2 = -1.0
    assert r2_score(y_true, y_pred_imperfect).item() == pytest.approx(-1.0)

def test_r2_score_multi_output():
    """Test R2 score with multiple output dimensions."""
    y_true = torch.tensor([[0.5, 1.0], [-1.0, 1.0], [7.0, -6.0]]) # (N, D)
    y_pred = torch.tensor([[0.0, 2.0], [-1.0, 2.0], [8.0, -5.0]])
    # Output 1:
    # y_true1 = [0.5, -1, 7], mean = 6.5/3 = 2.1667
    # y_pred1 = [0, -1, 8]
    # SSE1 = (0.5-0)**2 + (-1-(-1))**2 + (7-8)**2 = 0.25 + 0 + 1 = 1.25
    # SST1 = (0.5-2.1667)**2 + (-1-2.1667)**2 + (7-2.1667)**2 
    #      = (-1.6667)**2 + (-3.1667)**2 + (4.8333)**2 
    #      = 2.7778 + 10.0278 + 23.3611 = 36.1667
    # R2_1 = 1 - 1.25 / 36.1667 = 1 - 0.03456 = 0.9654
    # Output 2:
    # y_true2 = [1, 1, -6], mean = -4/3 = -1.3333
    # y_pred2 = [2, 2, -5]
    # SSE2 = (1-2)**2 + (1-2)**2 + (-6-(-5))**2 = 1 + 1 + 1 = 3
    # SST2 = (1-(-1.3333))**2 + (1-(-1.3333))**2 + (-6-(-1.3333))**2
    #      = (2.3333)**2 + (2.3333)**2 + (-4.6667)**2
    #      = 5.4444 + 5.4444 + 21.7778 = 32.6667
    # R2_2 = 1 - 3 / 32.6667 = 1 - 0.0918 = 0.9082
    expected_r2 = torch.tensor([0.96543209876, 0.9081632653])
    result = r2_score(y_true, y_pred)
    assert torch.allclose(result, expected_r2, atol=1e-4) 