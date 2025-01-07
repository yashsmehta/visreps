import time
import random
from pathlib import Path
import os
import pickle
import warnings
import torch
import torch.optim as optim
from filelock import FileLock, Timeout
from rich.console import Console
from rich.theme import Theme
from omegaconf import OmegaConf

# Suppress specific torch.load FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, 
                       message="You are using `torch.load` with `weights_only=False`.*")

def setup_logging():
    """Initialize Rich with custom theme and return themed print function"""
    custom_theme = Theme({
        "info": "bold white",
        "success": "green",
        "warning": "bold yellow",
        "error": "bold red",
        "highlight": "bold magenta",
        "setup": "cyan"
    })
    console = Console(theme=custom_theme)
    return console.print

# Initialize Rich print globally as rprint
rprint = setup_logging()

def check_trainer_config(cfg):
    """
    Validates the trainer configuration for the number of elements and content in 'conv_trainable' and 'fc_trainable'.

    This function checks that the 'conv_trainable' list contains exactly 5 elements and that each element is either '0' or '1'.
    Similarly, it ensures the 'fc_trainable' list contains exactly 3 elements, each of which must also be '0' or '1'.
    These checks ensure that the configuration adheres to expected format and content, which is critical for the training process.

    Args:
        cfg (OmegaConf): The configuration object containing 'conv_trainable' and 'fc_trainable' attributes.

    Returns:
        OmegaConf: The validated configuration object.

    Raises:
        AssertionError: If any of the conditions on 'conv_trainable' or 'fc_trainable' are not met.
    """
    assert cfg.model_class in [
        "custom_cnn",
        "standard_cnn",
    ], "model_class must be one of 'custom_cnn', 'standard_cnn'!"
    
    if cfg.model_class == "custom_cnn":
        assert all(
            char in "01" for char in cfg.arch.conv_trainable
        ), "conv_trainable must only contain '0's and '1's!"
        assert all(
            char in "01" for char in cfg.arch.fc_trainable
        ), "fc_trainable must only contain '0's and '1's!"
    
    return cfg


def load_pickle(file_path):
    """Load data from pickle file"""
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Pickle file not found at path: {file_path}")
    except pickle.UnpicklingError:
        raise pickle.UnpicklingError(f"Error unpickling file at {file_path}. File may be corrupted.")
    except Exception as e:
        raise RuntimeError(f"Error loading pickle file at {file_path}: {str(e)}")


def save_results(df, cfg, timeout=60):
    """Save results to CSV with file locking in logs/mode/exp_name.csv format.
    Adds all config parameters from OmegaConf while avoiding metadata."""
    # Create a clean DataFrame without the metadata
    clean_df = df.copy()
    
    # Convert OmegaConf to primitive container and add all config params
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(config_dict, dict):
        for key, value in config_dict.items():
            if not key.startswith('_') and not isinstance(value, (dict, list)):
                clean_df[key] = value

    # Add random delay
    random.seed(os.urandom(10))
    time.sleep(random.uniform(0, 5))
    
    # Setup paths and lock
    save_dir = Path('logs') / cfg.mode / cfg.load_model_from
    save_dir.mkdir(parents=True, exist_ok=True)
    results_path = save_dir / f"{cfg.exp_name}.csv"
    lock = FileLock(str(results_path.with_suffix(".lock")), timeout=timeout)
    
    try:
        with lock:
            write_header = not results_path.exists()
            clean_df.to_csv(results_path, mode="a", header=write_header, index=False)
            rprint(f"Successfully saved results to {results_path}", style="success")
    except Timeout:
        rprint(f"ERROR: Could not acquire lock for {results_path} after {timeout}s", style="error")
        raise
    except Exception as e:
        rprint(f"ERROR: Failed to save results to {results_path}: {str(e)}", style="error")
        raise
        
    return str(results_path)


def get_optimizer_class(optimizer_name):
    """Get optimizer class by name with exact or fuzzy matching."""
    available_optimizers = {name.lower(): getattr(optim, name) 
                          for name in dir(optim) 
                          if name[0].isupper() and not name.startswith('_')}
    
    opt_name = optimizer_name.lower()
    if opt_name in available_optimizers:
        return available_optimizers[opt_name]
    
    matches = [name for name in available_optimizers.keys() 
              if name.startswith(opt_name) or opt_name.startswith(name)]
    if matches:
        return available_optimizers[matches[0]]
    
    raise ValueError(f"Could not find optimizer '{optimizer_name}'. Available optimizers: {list(available_optimizers.keys())}")


def calculate_cls_accuracy(data_loader, model, device):
    """Calculate classification accuracy with proper device handling and numerical stability.
    
    Args:
        data_loader: PyTorch DataLoader
        model: PyTorch model
        device: torch.device for computation
    
    Returns:
        float: Classification accuracy as percentage (0-100)
    """
    model.eval()  # Ensure model is in eval mode
    correct = 0
    total = 0
    
    # Use autocast based on device type
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    
    with torch.no_grad(), torch.autocast(device_type=autocast_device, dtype=autocast_dtype):
        for images, labels in data_loader:
            # Move data to device efficiently
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            
            # Accumulate statistics without keeping autograd history
            total += labels.size(0)
            correct += torch.eq(predicted, labels).sum().item()
    
    # Handle edge case and ensure floating point division
    if total == 0:
        return 0.0
        
    # Use float for stable division and percentage calculation
    return (100.0 * correct) / total