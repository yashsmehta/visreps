import json
import subprocess
from itertools import product
import os
from pathlib import Path

# Configuration file paths
BASE_CONFIG = "configs/train/cluster_base.json"

# Define the parameter grid for training
PARAM_GRID = {
    "seed": [1],
    "model_name": ["CustomCNN"],
    "pca_labels": [True],
    "pca_n_classes": [8, 16, 32, 64],
    "pca_labels_folder": ["pca_labels_dreamsim"],
    "checkpoint_dir": ["dreamsim_pca"],
    "log_checkpoints": ["True"]
}

# Slurm configuration
SLURM_CONFIG = {
    "job-name": "visreps",
    "output": "scripts/slurm/slurm_logs/%j.out",
    "error": "scripts/slurm/slurm_logs/%j.err",
    "ntasks": "1",
    "cpus-per-task": "32",
    "gres": "gpu:1",
    "time": "8:00:00",
    "partition": "a100",
    "qos": "qos_gpu",
    "account": "mbonner5_gpu",
}

def flatten_config(config, parent_key="", sep="."):
    """Recursively flattens a nested configuration dictionary."""
    flat = {}
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            flat.update(flatten_config(v, new_key, sep))
        else:
            flat[new_key] = v
    return flat

def load_config(filepath):
    """Loads a JSON configuration from the given file."""
    with open(filepath, "r") as file:
        return json.load(file)

def generate_slurm_script(job_name, overrides):
    """Generates a Slurm script for the given parameters."""
    script = ["#!/bin/bash"]
    
    # Add Slurm directives
    for key, value in SLURM_CONFIG.items():
        script.append(f"#SBATCH --{key}={value}")
    
    # Add environment setup
    script.extend([
        "",
        "# Activate the virtual environment",
        "source .venv/bin/activate",
        "",
        "# Print debug information",
        "echo \"Running on node: $(hostname)\"",
        "echo \"GPU information:\"",
        "nvidia-smi",
        "",
        "# Run the training script"
    ])

    # Construct the training command
    cmd = [
        "python -m visreps.run",
        f"--config {BASE_CONFIG}",
        "--override"
    ] + [f"{override}" for override in overrides]
    
    script.append(" ".join(cmd))
    script.append("\ndeactivate")
    
    return "\n".join(script)

def main():
    # Create logs directory if it doesn't exist
    Path("scripts/slurm/slurm_logs").mkdir(parents=True, exist_ok=True)
    Path("scripts/slurm/tmp").mkdir(parents=True, exist_ok=True)
    
    # Generate all combinations of grid parameters
    param_names = list(PARAM_GRID.keys())
    param_combos = list(product(*PARAM_GRID.values()))

    total_runs = len(param_combos)
    print(f"Submitting {total_runs} Slurm jobs")

    # Submit each combination as a separate Slurm job
    for i, combo in enumerate(param_combos):
        # Create overrides from the parameter combination
        overrides = [
            f"{name}={json.dumps(value)}" for name, value in zip(param_names, combo)
        ]
        overrides.append("mode=train")

        # Create temporary script file
        script_path = f"scripts/slurm/tmp/train_job_{i+1}.sh"
        script_content = generate_slurm_script(f"job_{i+1}", overrides)
        
        with open(script_path, "w") as f:
            f.write(script_content)
        
        # Submit the job
        cmd = ["sbatch", script_path]
        print(f"\nSubmitting job_{i+1} with overrides:")
        for override in overrides:
            print(f"  {override}")
        subprocess.run(cmd)
        
        # Clean up temporary script
        os.remove(script_path)

if __name__ == "__main__":
    main()
