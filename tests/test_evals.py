import pytest
import torch
from omegaconf import OmegaConf
from unittest.mock import patch, MagicMock
import pandas as pd

# Import the function to test
from visreps.evals import eval as run_evaluation

# --- Fixtures ---

@pytest.fixture
def base_eval_cfg():
    """Provides a base OmegaConf object for evaluation tests."""
    return OmegaConf.create({
        "load_model_from": "torchvision", # Default: don't load from checkpoint
        "model_name": "mock_model",
        "exp_name": "test_eval_exp",
        "cfg_id": "cfg0", # For checkpoint loading case
        # Add necessary keys for model_utils.configure_feature_extractor
        "layers_to_extract": ["layer1"], 
        "neural_dataset": "mock_neural_data",
        "subject": "S1",
        "data_dir": "/tmp/mock_neural",
        "batch_size": 2,
        # Add necessary keys for alignment
        "analysis": "rsa",
        # Add necessary keys for saving
        "log_expdata": False, # Default: don't save results
        # Add any other keys expected by the eval function or its dependencies
        # Add dummy keys that might be loaded from train_cfg if load_model_from='checkpoint'
        "dummy_train_key": "value",
        "device": "cpu", # Explicitly set device for test environment
    })

# --- Test Cases ---

@pytest.mark.parametrize(
    "load_from_ckpt, log_results",
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ]
)
def test_eval_pipeline_calls(base_eval_cfg, load_from_ckpt, log_results, tmp_path):
    """Test the main evaluation pipeline, mocking external dependencies."""
    cfg = base_eval_cfg.copy()
    cfg.load_model_from = "checkpoint" if load_from_ckpt else "torchvision"
    cfg.log_expdata = log_results
    # Ensure analysis allows saving if log_results is True
    if log_results:
        cfg.analysis = "rsa" # Example analysis type that allows saving

    # Determine expected device *before* patching
    expected_device_str = "cuda" if torch.cuda.is_available() else "cpu"
    expected_device = torch.device(expected_device_str)

    # Mock data/objects returned by dependencies
    mock_model_instance = MagicMock(spec=torch.nn.Module)
    mock_neural_data = {"some_key": "some_value"}
    mock_dataloader = [torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)] # Dummy loader
    mock_activations = {"layer1": torch.randn(4, 10)}
    mock_keys = ["key1", "key2"]
    mock_results_df = pd.DataFrame({"metric": [0.5, 0.6]})
    mock_train_cfg = OmegaConf.create({"original_lr": 0.01, "mode": "train"}) # Mock loaded train cfg

    # Patch all external calls within visreps.evals
    with (
        patch("visreps.evals.model_utils.load_model", return_value=mock_model_instance) as mock_load_model,
        patch("visreps.evals.model_utils.configure_feature_extractor", return_value=mock_model_instance) as mock_config_extractor,
        patch("visreps.evals.model_utils.get_activations", return_value=(mock_activations, mock_keys)) as mock_get_activations,
        patch("visreps.evals.get_neural_loader", return_value=(mock_neural_data, mock_dataloader)) as mock_get_loader,
        patch("visreps.evals.compute_neural_alignment", return_value=mock_results_df) as mock_compute_align,
        patch("visreps.evals.save_results") as mock_save_results,
        patch("visreps.evals.OmegaConf.load", return_value=mock_train_cfg) as mock_omega_load,
        patch("visreps.evals.torch.device") as mock_torch_device # Patch device to ensure consistency
    ):
        # Let the mock return the expected device determined above
        mock_torch_device.return_value = expected_device

        # --- Run the evaluation function ---
        results_df = run_evaluation(cfg)
        # --- End Run ---

        # Assertions
        # Verify the essential device determination call happened
        mock_torch_device.assert_any_call(expected_device_str)

        if load_from_ckpt:
            mock_omega_load.assert_called_once_with(f"model_checkpoints/{cfg.exp_name}/cfg{cfg.cfg_id}/config.json")
            # Check if keys from loaded config were merged (excluding 'mode')
            assert hasattr(cfg, "original_lr") 
        else:
            mock_omega_load.assert_not_called()
            assert not hasattr(cfg, "original_lr") # Should not be present if not loaded

        # Use the actual device determined *before* patching for assertions
        mock_load_model.assert_called_once_with(cfg, expected_device)
        mock_config_extractor.assert_called_once_with(cfg, mock_model_instance)
        mock_get_loader.assert_called_once_with(cfg)
        mock_get_activations.assert_called_once_with(mock_model_instance, mock_dataloader, expected_device)
        mock_compute_align.assert_called_once_with(cfg, mock_activations, mock_neural_data, mock_keys)

        if log_results and cfg.analysis != "cross_decomposition":
            mock_save_results.assert_called_once_with(mock_results_df, cfg)
        else:
            mock_save_results.assert_not_called()

        pd.testing.assert_frame_equal(results_df, mock_results_df)

# Add more tests if specific internal logic needs verification,
# e.g., handling different analysis types if they significantly change the flow. 