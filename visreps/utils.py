import time
import random
from pathlib import Path
import os
import pickle
from datetime import datetime
import pandas as pd
from typing import Dict
import torch.optim as optim
from rich.console import Console
from rich.theme import Theme

def setup_logging():
    """Initialize Rich with custom theme and return themed print function"""
    custom_theme = Theme({
        "info": "bold blue",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "highlight": "bold magenta",
        "setup": "bold cyan"
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


def save_logs(df, cfg):
    """
    Saves logs to a CSV file based on the experiment configuration.

    This function modifies the dataframe by dropping specific columns and adding configuration details.
    It then determines the appropriate logging directory based on the model type and creates the directory if it does not exist.
    The function handles concurrent write operations by using a lock file to ensure data integrity.
    Logs are appended to a CSV file named after the experiment name.

    Args:
        df (pandas.DataFrame): The dataframe containing the logs to be saved.
        cfg (OmegaConf): The configuration object containing experiment and model settings.

    Returns:
        pathlib.Path: The path to the directory where the log file is saved.
    """
    df = df.drop(columns=["model_layer_index"])
    df["exp_name"] = cfg.exp_name
    df["seed"] = cfg.seed
    if cfg.model.name != "custom":
        df["model"] = cfg.model.name
        df["pretrained"] = cfg.model.pretrained
    else:
        df["nonlin"] = cfg.model.nonlin
        df["weights_init"] = cfg.model.weights_init
        df["norm"] = cfg.model.norm

    logdata_path = Path(cfg.log_dir)
    if cfg.model.name == "custom":
        logdata_path = logdata_path / "custom_arch"
    else:
        logdata_path = logdata_path / "standard_arch"

    logdata_path.mkdir(parents=True, exist_ok=True)
    csv_file = logdata_path / f"{cfg.exp_name}.csv"
    write_header = not csv_file.exists()

    lock_file = csv_file.with_suffix(".lock")
    while lock_file.exists():
        rprint(f"Waiting for lock on {csv_file}...", style="warning")
        time.sleep(random.uniform(1, 5))

    try:
        lock_file.touch()
        df.to_csv(csv_file, mode="a", header=write_header, index=False)
        rprint(f"Saved logs to {csv_file}", style="success")
    finally:
        lock_file.unlink()

    return logdata_path




def log_results(results, folder_name, cfg_id):
    """
    Log the results to a CSV file within a dynamically named directory based on configuration ID.

    This function first introduces a random sleep delay to simulate asynchronous operations and avoid
    potential clashes in a multi-threaded environment. It then constructs a path for logging based on
    the provided folder name and configuration ID, ensuring the directory exists or is created. A CSV file
    is named according to the configuration ID, and results are written to this file. If the file does not
    exist, it creates a new one with headers; otherwise, it writes without headers. Finally, it logs the
    path to which the results were saved and the sleep time.

    Args:
        results (DataFrame): The DataFrame containing results to log.
        folder_name (str): The base name of the folder where logs will be stored.
        cfg_id (int): The configuration ID used to uniquely name the log file.

    """
    random.seed(os.urandom(10))
    sleep_time = random.uniform(0, 3)
    time.sleep(sleep_time)
    folder_name = f"logs/{folder_name}"
    logdata_path = Path(folder_name)
    logdata_path.mkdir(parents=True, exist_ok=True)
    csv_file = logdata_path / f"cfg{cfg_id}.csv"
    write_header = not csv_file.exists()

    results.to_csv(csv_file, mode="a", header=write_header, index=False)
    rprint(f"Saved logs to {csv_file} after sleeping for {sleep_time:.2f} seconds", style="success")


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



def save_results(results_df: pd.DataFrame, cfg: Dict, result_type: str = None) -> str:
    """Save results to CSV in a structured directory
    
    Args:
        results_df: DataFrame containing results
        cfg: Configuration dictionary
        result_type: Type of results (e.g., 'neural_alignment', 'training', etc.)
                    Used to organize results in subdirectories
    
    Returns:
        Path where results were saved
    """
    # Create base directory structure
    exp_name = getattr(cfg, 'exp_name', 'default')
    model_class = getattr(cfg, 'model_class', 'unknown')
    save_dir = os.path.join('logs', exp_name, model_class)
    
    # Add result type subdirectory if specified
    if result_type:
        save_dir = os.path.join(save_dir, result_type)
    
    os.makedirs(save_dir, exist_ok=True)
    
    results_path = os.path.join(save_dir, f"results.csv")
    results_df.to_csv(results_path, index=False)
    
    return results_path


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
