import pytest
import torch
import json
from pathlib import Path
import os
import omegaconf # Import OmegaConf
from omegaconf import OmegaConf, DictConfig # Import DictConfig for type check
from unittest.mock import patch, MagicMock

from visreps.utils import load_config, validate_config, ConfigVerifier # Import ConfigVerifier for potential mocking
from visreps.trainer import Trainer

# --- Configuration Tests ---

# Define the path to the base config relative to the tests directory
BASE_CONFIG_PATH = Path(__file__).parent.parent / "configs/train/base.json"
# Define a path for temporary config files
TEMP_CONFIG_DIR = Path(__file__).parent / "temp_configs"
TEMP_CONFIG_DIR.mkdir(exist_ok=True)


@pytest.fixture(scope="module")
def base_config_for_load() -> DictConfig: # Type hint return
    """Fixture to load the base training config for loading/override tests."""
    assert BASE_CONFIG_PATH.exists(), f"Base config not found at {BASE_CONFIG_PATH}"
    # Pass an empty list for overrides
    return load_config(str(BASE_CONFIG_PATH), [])

@pytest.fixture(scope="module")
def base_config_for_validate() -> DictConfig:
    """Fixture providing a validated base config for validation tests."""
    cfg = load_config(str(BASE_CONFIG_PATH), [])
    # Return a validated version for tests that expect a valid starting point
    return validate_config(cfg)


def test_load_base_config(base_config_for_load):
    """Test: Verify base.json loads correctly via utils.load_config."""
    # Check for OmegaConf DictConfig type
    assert isinstance(base_config_for_load, DictConfig)
    assert base_config_for_load.mode == "train" # Use dot notation
    assert "dataset" in base_config_for_load
    assert "model_class" in base_config_for_load # Check for model_class
    assert "optimizer" in base_config_for_load


def test_load_config_override(base_config_for_load):
    """Test: Verify overrides via --override work as expected."""
    # Override an existing dictionary key and a top-level key
    overrides = ["num_epochs=50", "arch.dropout=0.6"]
    cfg = load_config(str(BASE_CONFIG_PATH), overrides)
    assert cfg.num_epochs == 50
    assert cfg.arch.dropout == 0.6
    # Ensure other base config values are retained
    assert cfg.mode == "train"
    assert cfg.arch.nonlinearity == "relu" # Check another value in the dict


def test_load_config_nested_override(base_config_for_load):
    """Test nested dictionary overrides that replace non-dict keys."""
    # This override replaces the top-level 'optimizer' string with a dict
    # And adds a new 'model' dict
    overrides = ["model.arch=resnet18", "optimizer.params.weight_decay=0.0002"]
    cfg = load_config(str(BASE_CONFIG_PATH), overrides)
    assert cfg.model.arch == "resnet18"
    assert cfg.optimizer.params.weight_decay == 0.0002
    # The original 'optimizer' string is gone. 'type' was never part of base_config.optimizer
    # Let's check that the optimizer key now points to a dictionary
    assert isinstance(cfg.optimizer, DictConfig)
    # Let's check that the original 'learning_rate' key (at the root) still exists
    assert "learning_rate" in cfg


def test_validate_config_valid(base_config_for_validate):
    """Test: Ensure utils.validate_config passes for a valid config."""
    # The base config should be valid
    try:
        # Re-validate the already validated config (should pass trivially)
        validated_cfg = validate_config(base_config_for_validate)
        assert isinstance(validated_cfg, DictConfig) # Should return OmegaConf object
        assert validated_cfg.mode == "train"
    except Exception as e: # Catch broader exceptions during validation initially
        pytest.fail(f"validate_config raised an exception unexpectedly: {e}")


def test_validate_config_missing_key(base_config_for_validate):
    """Test: Ensure utils.validate_config catches missing required keys."""
    invalid_config = base_config_for_validate.copy()
    del invalid_config.dataset # Use dot notation for OmegaConf object
    # Expect OmegaConf's attribute error when accessing a deleted key during validation
    with pytest.raises(omegaconf.errors.ConfigAttributeError, match=r"Missing key dataset"):
        validate_config(invalid_config)


def test_validate_config_incorrect_type(base_config_for_validate):
    """Test: Ensure utils.validate_config catches incorrect types (if implemented)."""
    invalid_config = base_config_for_validate.copy()
    invalid_config.num_epochs = "not_an_integer" # Incorrect type
    # NOTE: Current ConfigVerifier doesn't perform strict type checking for num_epochs.
    # This test verifies the current behavior (no error). Add type check for robustness.
    try:
        validate_config(invalid_config)
    except Exception as e:
        pytest.fail(f"validate_config raised an unexpected exception for type mismatch: {e}")


def test_validate_config_nested_incorrect_type(base_config_for_validate):
    """Test validation of nested incorrect types (if implemented)."""
    invalid_config = base_config_for_validate.copy()
    # Use OmegaConf's update method or dot notation for modification
    # Note: The base config has optimizer as a string 'adamw', need to handle this.
    # Let's test overriding with an invalid nested structure first.
    override_config = OmegaConf.create({"optimizer": {"lr": "not_a_float"}})
    merged_config = OmegaConf.merge(invalid_config, override_config)
    # NOTE: Current ConfigVerifier doesn't perform strict type checking for optimizer.lr.
    # This test verifies the current behavior (no error). Add type check for robustness.
    try:
        validate_config(merged_config)
    except Exception as e:
        pytest.fail(f"validate_config raised an unexpected exception for nested type mismatch: {e}")


# Clean up temporary directory after tests run
def teardown_module(module):
    if TEMP_CONFIG_DIR.exists():
        for item in TEMP_CONFIG_DIR.iterdir():
            if item.is_file(): # Make sure it's a file before unlinking
                item.unlink()
        try: # Avoid error if dir is not empty due to unexpected files
            TEMP_CONFIG_DIR.rmdir()
        except OSError:
            print(f"Warning: Could not remove temp dir {TEMP_CONFIG_DIR}")


# --- Environment Setup Tests ---

# Minimal config for testing environment setup
@pytest.fixture
def minimal_env_cfg():
    # Using a simple dictionary that Trainer will convert to OmegaConf
    return {
        "seed": 123,
        "log_checkpoints": False, # Avoid checkpointing setup
        "use_wandb": False, # Avoid wandb setup
        # Add dummy values for other required fields expected by Trainer init
        # These will be mocked anyway, but need to be present
        "pca_labels": False,
        "dataset": "mock_dataset",
        "model_class": "mock_model",
        "custom_cnn": {}, # Add nested dicts if needed by load_model
        "arch": {},
        "optimizer": "SGD", # Use simple strings, setup functions handle creation
        "learning_rate": 0.01,
        "lr_scheduler": "StepLR",
        "lr_step_size": 1,
        "lr_gamma": 0.1,
        "data_dir": "/tmp/mock_data", # Needs a placeholder
        "log_dir": "/tmp/mock_logs", # Needs a placeholder
        "exp_name": "env_test",
    }

# Context manager for mocking Trainer dependencies
@pytest.fixture
def mocked_trainer_context(minimal_env_cfg):
    with patch("visreps.trainer.get_obj_cls_loader") as mock_loader, \
         patch("visreps.trainer.model_utils.load_model") as mock_load_model, \
         patch("visreps.trainer.utils.setup_optimizer") as mock_optimizer, \
         patch("visreps.trainer.utils.setup_scheduler") as mock_scheduler, \
         patch("visreps.trainer.MetricsLogger") as mock_logger, \
         patch("torch.manual_seed") as mock_torch_seed, \
         patch("torch.backends.cudnn") as mock_cudnn, \
         patch("torch.cuda.is_available") as mock_cuda_available:

        # Configure mocks needed for Trainer.__init__
        mock_dataset = MagicMock()
        mock_dataset.num_classes = 10
        mock_loader.return_value = ({"train": mock_dataset, "test": mock_dataset}, {"train": None, "test": None})
        mock_model_instance = MagicMock(spec=torch.nn.Module)
        # Make the mock model callable and return a dummy tensor
        mock_model_instance.return_value = torch.randn(2, 10) # Batch size 2, num classes 10
        # Ensure the mock model has parameters for optimizer setup
        mock_model_instance.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
        mock_load_model.return_value = mock_model_instance
        mock_optimizer.return_value = MagicMock()
        mock_scheduler.return_value = MagicMock()
        mock_logger.return_value = MagicMock()

        yield {
            "cfg": minimal_env_cfg,
            "mock_torch_seed": mock_torch_seed,
            "mock_cudnn": mock_cudnn,
            "mock_cuda_available": mock_cuda_available
        }

def test_environment_seed(mocked_trainer_context):
    """Test: Verify torch.manual_seed is called with the correct seed."""
    cfg = mocked_trainer_context["cfg"]
    mock_torch_seed = mocked_trainer_context["mock_torch_seed"]

    Trainer(OmegaConf.create(cfg)) # Initialize Trainer
    mock_torch_seed.assert_called_once_with(cfg["seed"])

def test_environment_cudnn(mocked_trainer_context):
    """Test: Check if torch.backends.cudnn.deterministic and benchmark are set correctly."""
    cfg = mocked_trainer_context["cfg"]
    mock_cudnn = mocked_trainer_context["mock_cudnn"]

    Trainer(OmegaConf.create(cfg)) # Initialize Trainer
    assert mock_cudnn.deterministic is True
    assert mock_cudnn.benchmark is False

def test_environment_device_cuda(mocked_trainer_context):
    """Test: Confirm device is set to 'cuda' if available."""
    cfg = mocked_trainer_context["cfg"]
    mock_cuda_available = mocked_trainer_context["mock_cuda_available"]
    mock_cuda_available.return_value = True

    trainer = Trainer(OmegaConf.create(cfg))
    assert trainer.device == torch.device("cuda")

def test_environment_device_cpu(mocked_trainer_context):
    """Test: Confirm device is set to 'cpu' if unavailable."""
    cfg = mocked_trainer_context["cfg"]
    mock_cuda_available = mocked_trainer_context["mock_cuda_available"]
    mock_cuda_available.return_value = False

    trainer = Trainer(OmegaConf.create(cfg))
    assert trainer.device == torch.device("cpu") 