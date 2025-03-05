import json
import subprocess
from itertools import product
import os
from pathlib import Path

# Configuration
BASE_CONFIG_PATH = "configs/eval/base.json"
CHECKPOINTS_DIR = "model_checkpoints"

# Evaluation parameter grid
# Note: return_nodes needs to be configured in the base config file
PARAM_GRID = {
    "exp_name": ["imagenet_pca_untrained"],
    "config_folders": ["cfg5"],
    "eval_epochs": [10],
    "region": [
        "early visual stream",
        # "ventral visual stream",
    ],
    "analysis": ["rsa"],
    "subject_idx": [0],
}

# Slurm configuration
SLURM_CONFIG = {
    "job-name": "visreps_eval",
    "output": "slurm/slurm_logs/%j.out",
    "error": "slurm/slurm_logs/%j.err",
    "ntasks": "1",
    "cpus-per-task": "16",
    "gres": "gpu:1",
    "time": "1:00:00",
    "partition": "v100",
    "qos": "qos_gpu",
    "account": "mbonner5_gpu",
}

def get_checkpoint_path(exp_name: str, config_folder: str, epoch: int) -> str:
    """Get the path to a checkpoint file."""
    return os.path.join(
        CHECKPOINTS_DIR, exp_name, config_folder, f"checkpoint_epoch_{epoch}.pth"
    )

def generate_slurm_script(job_name: str, overrides: list, checkpoint_path: str) -> str:
    """Generates a Slurm script for evaluation."""
    script = ["#!/bin/bash"]
    
    # Add Slurm directives
    for key, value in SLURM_CONFIG.items():
        if key == "job-name":
            value = f"{value}_{job_name}"
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
        "# Run the evaluation script"
    ])

    # Construct the evaluation command
    cmd = [
        "python -m visreps.run",
        f"--config {BASE_CONFIG_PATH}",
        "--override"
    ] + [f"{override}" for override in overrides]
    
    script.append(" ".join(cmd))
    script.append("\ndeactivate")
    
    return "\n".join(script)

def main():
    # Create logs directory if it doesn't exist
    Path("slurm/slurm_logs").mkdir(parents=True, exist_ok=True)
    Path("slurm/tmp").mkdir(parents=True, exist_ok=True)
    
    # Generate all combinations of grid parameters
    param_names = list(PARAM_GRID.keys())
    param_combos = list(product(*PARAM_GRID.values()))

    total_runs = len(param_combos)
    print(f"Submitting {total_runs} evaluation jobs")

    # Submit each combination as a separate Slurm job
    for combo in param_combos:
        # Create parameter dictionary
        params = dict(zip(param_names, combo))
        
        # Get checkpoint path
        checkpoint_path = get_checkpoint_path(
            params["exp_name"], params["config_folders"], params["eval_epochs"]
        )
        
        # Create overrides from parameters
        overrides = []
        for k, v in params.items():
            if k == "return_nodes":
                # Use json.dumps to properly format the list
                overrides.append(f"{k}={json.dumps(v[0])}")
            else:
                overrides.append(f"{k}={json.dumps(v)}")
        
        overrides.extend([
            "mode=eval",
            "neural_dataset=nsd",
            "log_expdata=true",
            f"cfg_id={params['config_folders'].replace('cfg', '')}",
            "load_model_from=checkpoint",
            f"checkpoint_model={os.path.basename(checkpoint_path)}",
        ])

        # Generate a unique job name
        job_name = f"{params['exp_name']}_{params['config_folders']}_e{params['eval_epochs']}"
        
        # Create temporary script file
        script_path = f"slurm/tmp/eval_{job_name}.sh"
        script_content = generate_slurm_script(job_name, overrides, checkpoint_path)
        
        with open(script_path, "w") as f:
            f.write(script_content)
        
        # Submit the job
        cmd = ["sbatch", script_path]
        print(f"\nSubmitting job: {job_name}")
        subprocess.run(cmd)
        
        # Clean up temporary script
        os.remove(script_path)

if __name__ == "__main__":
    main()
