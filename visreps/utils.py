import colorlog
import logging
import time
import random
from pathlib import Path
import torch
import os


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
    # assert len(cfg.conv_trainable) == 5, "conv_trainable must have 5 elements!"
    # assert len(cfg.fc_trainable) == 3, "fc_trainable must have 3 elements!"
    assert cfg.model_class in [
        "base_cnn", 
        "wavelet_net", 
    ], "model_class must be one of 'base_cnn', 'wavelet_net'!"
    assert all(char in "01" for char in cfg.conv_trainable), "conv_trainable must only contain '0's and '1's!"
    assert all(char in "01" for char in cfg.fc_trainable), "fc_trainable must only contain '0's and '1's!"
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
        print(f"Waiting for lock on {csv_file}...")
        time.sleep(random.uniform(1, 5))

    try:
        lock_file.touch()
        df.to_csv(csv_file, mode="a", header=write_header, index=False)
        print(f"Saved logs to {csv_file}")
    finally:
        lock_file.unlink()

    return logdata_path


def calculate_accuracy(data_loader, model, device):
    """
    Calculate the accuracy of a model on a given dataset.

    This function iterates over all batches in the provided data loader, computes the model's predictions,
    and compares them to the actual labels to determine the number of correct predictions. It then calculates
    the percentage of correct predictions over the total number of samples.

    Args:
        data_loader (torch.utils.data.DataLoader): The data loader containing the dataset to evaluate.
        model (torch.nn.Module): The model to evaluate.
        device (str): The device to perform computations on ('cuda' or 'cpu').

    Returns:
        float: The accuracy of the model on the provided dataset, as a percentage.
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def make_checkpoint_dir(folder, parent_dir="model_checkpoints"):
    """
    Create a new subdirectory for a training checkpoint within a specified parent directory.

    This function constructs a directory path using the provided `folder` name under the `parent_dir`.
    It ensures that the directory exists and then creates a new subdirectory within it to store the
    checkpoint. The subdirectory is named by incrementing the count of existing directories.

    Args:
        folder (str): The name of the main folder under which the checkpoint directory will be created.
        parent_dir (str): The parent directory where the `folder` will be located. Defaults to 'model_checkpoints'.

    Returns:
        str: The path to the newly created checkpoint subdirectory.
    """
    checkpoint_dir = os.path.join(parent_dir, folder)
    os.makedirs(checkpoint_dir, exist_ok=True)
    ith_folder = len(os.listdir(checkpoint_dir)) + 1
    checkpoint_subdir = os.path.join(checkpoint_dir, "cfg" + str(ith_folder))
    os.makedirs(checkpoint_subdir, exist_ok=True)
    return checkpoint_subdir


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

    results.to_csv(csv_file, mode='a', header=write_header, index=False)
    print(f"Saved logs to {csv_file} after sleeping for {sleep_time:.2f} seconds")


def setup_logging():
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    ))

    logger = colorlog.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger