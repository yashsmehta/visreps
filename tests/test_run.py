import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from omegaconf import OmegaConf

# Import the main function to test
from visreps.run import main as run_main

# --- Test Cases ---

@pytest.mark.parametrize("mode", ["train", "eval"])
def test_run_main_dispatch(mode):
    """Test that run.main() correctly dispatches to Trainer or evals based on mode."""

    # Mock command line arguments
    mock_args = MagicMock()
    mock_args.mode = mode
    mock_args.config = None # Use default config path logic
    mock_args.override = []

    # Mock config object returned by load_config and validate_config
    mock_cfg = OmegaConf.create({
        "mode": mode,
        # Add any other minimal required keys by Trainer or eval
        "some_key": "some_value"
    })

    # Mock the Trainer class and its train method
    mock_trainer_instance = MagicMock()
    mock_trainer_instance.train = MagicMock()

    # Patch all dependencies
    with patch("visreps.run.argparse.ArgumentParser") as mock_ArgumentParser, \
         patch("visreps.run.utils.load_config", return_value=mock_cfg) as mock_load_config, \
         patch("visreps.run.utils.validate_config", return_value=mock_cfg) as mock_validate_config, \
         patch("visreps.run.Trainer", return_value=mock_trainer_instance) as mock_Trainer_class, \
         patch("visreps.run.evals.eval") as mock_evals_eval:

        # Configure the mock parser to return our mock args
        mock_parser = mock_ArgumentParser.return_value
        mock_parser.parse_args.return_value = mock_args

        # --- Run the main function ---
        run_main()
        # --- End Run ---

        # Assertions
        mock_ArgumentParser.assert_called_once()
        mock_parser.parse_args.assert_called_once()

        # Determine expected config path based on mode
        expected_config_path = f"configs/{mode}/base.json"
        mock_load_config.assert_called_once_with(expected_config_path, mock_args.override)
        mock_validate_config.assert_called_once_with(mock_cfg)

        if mode == "train":
            mock_Trainer_class.assert_called_once_with(mock_cfg)
            mock_trainer_instance.train.assert_called_once()
            mock_evals_eval.assert_not_called()
        elif mode == "eval":
            mock_Trainer_class.assert_not_called()
            mock_trainer_instance.train.assert_not_called()
            mock_evals_eval.assert_called_once_with(mock_cfg)
        else:
            pytest.fail(f"Unexpected mode: {mode}")

# Test edge cases like providing a specific config path or overrides if needed 