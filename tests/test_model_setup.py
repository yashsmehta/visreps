import pytest
import torch
from omegaconf import OmegaConf
from unittest.mock import patch
# Import the actual AlexNet class if torchvision is available
try:
    from torchvision.models import AlexNet
except ImportError:
    AlexNet = None # Define as None if not available

from visreps.models import utils as model_utils
from visreps.models.custom_cnn import TinyCustomCNN, CustomCNN # Import base CustomCNN too
from visreps.models import standard_cnn # Keep for referencing the function if needed

# --- Fixtures ---

@pytest.fixture(scope="module")
def base_model_cfg():
    """Provides a base configuration for model loading tests."""
    return OmegaConf.create({
        "seed": 42,
        "pca_labels": False,
        "model_class": "standard_cnn", # Default to standard
        "standard_cnn": {
            "model_name": "AlexNet",
            "pretrained_dataset": "none", # Load untrained for simplicity
        },
        "custom_cnn": {
            "model_name": "TinyCustomCNN", # Default custom model
        },
        "arch": {}, # Default empty arch dict
        "pca_n_classes": 10 # Default, can be overridden
    })

@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"))])
def device(request):
    """Fixture for testing on CPU and available GPU."""
    return torch.device(request.param)

# --- Model Loading and Initialization Tests ---

@pytest.mark.parametrize(
    "model_class, model_name, expected_type_str", # Use string representation
    [
        ("standard_cnn", "AlexNet", "AlexNet"),
        ("custom_cnn", "TinyCustomCNN", "CustomCNN"), # Expect base CustomCNN
        # Add more standard or custom models here if implemented
    ]
)
def test_load_correct_architecture(base_model_cfg, device, model_class, model_name, expected_type_str):
    """Test: Verify load_model loads the correct architecture based on config."""
    cfg = base_model_cfg.copy()
    cfg.model_class = model_class
    if model_class == "standard_cnn":
        cfg.standard_cnn.model_name = model_name
    elif model_class == "custom_cnn":
        cfg.custom_cnn.model_name = model_name

    num_classes = 100 # Arbitrary number for architecture check

    # Determine expected type based on string
    expected_type = None
    if expected_type_str == "AlexNet":
        if AlexNet is None:
             pytest.skip("torchvision not available, skipping AlexNet architecture test.")
        expected_type = AlexNet
    elif expected_type_str == "CustomCNN":
        expected_type = CustomCNN
    # Add other types here if needed

    if expected_type is None:
        pytest.fail(f"Test configuration error: Unknown expected_type_str '{expected_type_str}'")

    model = model_utils.load_model(cfg, device, num_classes=num_classes)
    assert isinstance(model, expected_type)
    assert next(model.parameters()).device.type == device.type # Compare device types

@pytest.mark.parametrize("num_classes", [10, 200, 1000])
def test_load_model_num_classes_standard(base_model_cfg, device, num_classes):
    """Test: Check num_classes is set correctly for standard AlexNet."""
    cfg = base_model_cfg.copy()
    cfg.model_class = "standard_cnn"
    cfg.standard_cnn.model_name = "AlexNet"
    cfg.pca_labels = False # Ensure standard mode

    if AlexNet is None: pytest.skip("torchvision needed for AlexNet class check.")

    model = model_utils.load_model(cfg, device, num_classes=num_classes)
    # Assuming AlexNet structure where the final layer is model.classifier[-1]
    final_layer = model.classifier[-1]
    assert isinstance(final_layer, torch.nn.Linear)
    assert final_layer.out_features == num_classes

@pytest.mark.parametrize("pca_n_classes", [2, 8, 32])
def test_load_model_num_classes_pca(base_model_cfg, device, pca_n_classes):
    """Test: Check num_classes is set correctly for PCA labels (AlexNet)."""
    cfg = base_model_cfg.copy()
    cfg.model_class = "standard_cnn"
    cfg.standard_cnn.model_name = "AlexNet"
    cfg.pca_labels = True
    cfg.pca_n_classes = pca_n_classes

    if AlexNet is None: pytest.skip("torchvision needed for AlexNet class check.")

    model = model_utils.load_model(cfg, device, num_classes=pca_n_classes)
    # Assuming AlexNet structure
    final_layer = model.classifier[-1]
    assert isinstance(final_layer, torch.nn.Linear)
    assert final_layer.out_features == pca_n_classes


def test_load_model_device(base_model_cfg, device):
    """Test: Ensure model parameters and buffers are moved to the correct device."""
    num_classes = 10
    cfg = base_model_cfg.copy()
    # Use TinyCustomCNN as it's self-contained and has buffers (BatchNorm)
    cfg.model_class = "custom_cnn"
    cfg.custom_cnn.model_name = "TinyCustomCNN"

    model = model_utils.load_model(cfg, device, num_classes=num_classes)

    # Check parameters
    for param in model.parameters():
        assert param.device.type == device.type # Compare device types

    # Check buffers (e.g., BatchNorm running mean/var)
    for buffer in model.buffers():
        assert buffer.device.type == device.type # Compare device types

# --- Weight Initialization Randomness Test ---

def test_model_weights_different_seeds(base_model_cfg, device):
    """Test: Verify that models initialized with different seeds have different weights."""
    num_classes = 10
    cfg1 = base_model_cfg.copy()
    cfg2 = base_model_cfg.copy()

    # Use TinyCustomCNN for consistency
    cfg1.model_class = "custom_cnn"
    cfg2.model_class = "custom_cnn"

    # Set different seeds
    cfg1.seed = 42
    cfg2.seed = 123

    # Load models (load_model internally sets the seed)
    model1 = model_utils.load_model(cfg1, device, num_classes=num_classes)
    model2 = model_utils.load_model(cfg2, device, num_classes=num_classes)

    # Compare weights
    all_equal = True
    param_diffs = []

    params1 = list(model1.parameters())
    params2 = list(model2.parameters())

    assert len(params1) == len(params2), "Models have different number of parameter tensors"

    for i, (p1, p2) in enumerate(zip(params1, params2)):
        assert p1.shape == p2.shape, f"Parameter shape mismatch at index {i}: {p1.shape} vs {p2.shape}"
        if torch.allclose(p1, p2):
            # This might happen for non-initialized layers or biases set to zero
            # print(f"Warning: Parameters at index {i} are identical.") # Optional warning
            pass
        else:
            all_equal = False
            diff = (p1 - p2).abs().mean().item()
            param_diffs.append((i, diff))

    # Assert that *some* weights are different
    assert not all_equal, "Model weights were identical despite different seeds!"
    print(f"\nWeight comparison (Seed {cfg1.seed} vs {cfg2.seed}): Weights differ as expected.")
    # Optional: Print some differences
    # print("Indices with different weights (mean abs diff):")
    # for idx, diff in param_diffs[:5]:
    #     print(f"  Index {idx}: {diff:.6f}") 