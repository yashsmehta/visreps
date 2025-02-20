import os
import json
import glob
import subprocess
from itertools import product

# Base configuration file and parameter grid
BASE_CONFIG_PATH = "configs/eval/base.json"

PARAM_GRID = {
    "exp_name": ["imagenet_cnn"],
    "config_folders": ["cfg1"],
    "region": [
        "early visual stream",
        "midventral visual stream",
        "ventral visual stream"
    ],
    "eval_epochs": [0, 10],
    "analysis": ["rsa"],
    "subject_idx": [0, 1, 2, 3, 4, 5, 6, 7],
    "return_nodes": [["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2", "fc3"]]
}

def load_json_config(filepath):
    """Load JSON configuration from a file."""
    with open(filepath, "r") as file:
        return json.load(file)

def get_epoch_number(filepath):
    """Extract the epoch number from a checkpoint filename."""
    try:
        # Expected format: checkpoint_epoch_X.pth
        return int(os.path.basename(filepath).split('_')[2].split('.')[0])
    except (IndexError, ValueError):
        return -1

def list_checkpoint_files(directory, pattern="checkpoint_epoch_*.pth", epochs=None):
    """
    List checkpoint files in a directory.
    If 'epochs' is provided, only return checkpoints matching the epoch numbers.
    """
    files = glob.glob(os.path.join(directory, pattern))
    if epochs is not None:
        # Allow single int input for epochs
        if isinstance(epochs, (int, float)):
            epochs = [epochs]
        files = [f for f in files if get_epoch_number(f) in epochs]
    return sorted(files, key=get_epoch_number)

def run_experiment():
    # Load the base configuration
    base_config = load_json_config(BASE_CONFIG_PATH)

    # Generate all combinations of parameters from the grid
    keys = list(PARAM_GRID.keys())
    combinations = product(*(PARAM_GRID[key] for key in keys))

    for combo in combinations:
        params = dict(zip(keys, combo))
        exp_name = params["exp_name"]
        config_folder = params["config_folders"]

        checkpoint_dir = os.path.join("model_checkpoints", exp_name, config_folder)
        checkpoint_files = list_checkpoint_files(checkpoint_dir, epochs=params["eval_epochs"])

        # Display current configuration details
        print("\nConfiguration:")
        print(f"  Experiment      : {exp_name}")
        print(f"  Config Folder   : {config_folder}")
        print(f"  Region          : {params['region']}")
        print(f"  Analysis        : {params['analysis']}")
        print(f"  Subject Index   : {params['subject_idx']}")

        for checkpoint in checkpoint_files:
            rel_checkpoint = os.path.relpath(checkpoint)

            # Create command-line overrides from parameters
            overrides = [f"{key}={json.dumps(value)}" for key, value in params.items()]
            overrides += [
                "mode=eval",
                "neural_dataset=nsd",
                "log_expdata=true",
                f"cfg_id={config_folder.replace('cfg', '')}",
                "load_model_from=checkpoint",
                f"checkpoint_model={os.path.basename(rel_checkpoint)}"
            ]

            cmd = ["python", "-m", "visreps.run",
                   "--config", BASE_CONFIG_PATH,
                   "--override"] + overrides

            print(f"\nExecuting command for checkpoint {rel_checkpoint}:")
            print(" ".join(cmd))
            subprocess.run(cmd)

if __name__ == "__main__":
    run_experiment()