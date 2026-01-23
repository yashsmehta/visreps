import json
import subprocess
from itertools import product
import os
from pathlib import Path

# Configuration
BASE_CONFIG_PATH = "configs/eval/base.json"
SLURM_LOG_DIR = "scripts/slurm/slurm_logs"
SAVE_SLURM_LOGS = True

# Evaluation parameter grid
# Note: return_nodes needs to be configured in the base config file if used
PARAM_GRID = {
    "checkpoint_dir": ["model_checkpoints/imagenet_1k"],
    "cfg_id": [1], 
    "seed": [1], 
    "results_csv": ["cluster_eval_test.csv"],
    "notes": [""],
    "compare_rsm_correlation": ["Spearman"],
    "reconstruct_from_pcs": [False],
    "neural_dataset": ["nsd"],
    "eval_checkpoint_at_epoch": [20],
}

# Slurm configuration
SLURM_CONFIG = {
    "job-name": "visreps",
    "output": f"{SLURM_LOG_DIR}/%j.out",
    "error": f"{SLURM_LOG_DIR}/%j.err",
    "ntasks": "1",
    "cpus-per-task": "16",
    "gres": "gpu:1",
    "time": "0:10:00",
    "partition": "v100",
    "qos": "qos_gpu",
    "account": "mbonner5_gpu",
}

def get_checkpoint_path(exp_name: str, config_folder: str, epoch: int) -> str:
    """Get the path to a checkpoint file."""
    return os.path.join(
        CHECKPOINTS_DIR, exp_name, config_folder, f"checkpoint_epoch_{epoch}.pth"
    )

def generate_slurm_script(job_name: str, overrides: list) -> str:
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
    
    # Check if output should be logged to a file
    if SAVE_SLURM_LOGS:
        Path(SLURM_LOG_DIR).mkdir(parents=True, exist_ok=True)
        output_log_path = os.path.join(SLURM_LOG_DIR, f"{job_name}_output.log")
        script.append(f"echo 'Logging output to {output_log_path}'")
        script.append(f"({' '.join(cmd)}) > {output_log_path} 2>&1")
    else:
        script.append(" ".join(cmd))
    
    script.append("\ndeactivate")
    
    return "\n".join(script)

def main():
    # Create logs directory if it doesn't exist
    Path(SLURM_LOG_DIR).mkdir(parents=True, exist_ok=True)
    Path("scripts/slurm/tmp").mkdir(parents=True, exist_ok=True)
    
    # Generate all combinations of grid parameters
    param_names = list(PARAM_GRID.keys())
    param_combos = list(product(*PARAM_GRID.values()))

    total_runs = len(param_combos)
    print(f"Submitting {total_runs} evaluation jobs")

    # Submit each combination as a separate Slurm job
    for combo in param_combos:
        # Create parameter dictionary
        params = dict(zip(param_names, combo))
        
        # Construct checkpoint_model name
        checkpoint_model_name = f"checkpoint_epoch_{params['eval_checkpoint_at_epoch']}.pth"
        
        # Create overrides from parameters
        overrides = []
        for k, v in params.items():
            if k == "eval_checkpoint_at_epoch": # This key is used to form checkpoint_model_name, not passed directly as an override
                continue
            overrides.append(f"{k}={json.dumps(v)}")
        
        overrides.append(f"checkpoint_model={json.dumps(checkpoint_model_name)}")

        overrides.extend([
            "mode=eval",
            "log_expdata=true",
            "load_model_from=checkpoint",
        ])

        # Generate a unique job name suffix
        job_suffix_parts = [
            f"cfg{params.get('cfg_id', 'N')}",
            f"s{params.get('seed', 'N')}",
        ]
        if 'pca_k' in params:
            job_suffix_parts.append(f"k{params['pca_k']}")
        if 'neural_dataset' in params:
             job_suffix_parts.append(params['neural_dataset'])
        
        job_name_suffix = "_".join(job_suffix_parts)
        
        # Create temporary script file
        script_path = f"scripts/slurm/tmp/eval_{job_name_suffix}.sh"
        script_content = generate_slurm_script(job_name_suffix, overrides)
        
        with open(script_path, "w") as f:
            f.write(script_content)
        
        # Submit the job
        cmd = ["sbatch", script_path]
        print(f"\nSubmitting job: {SLURM_CONFIG.get('job-name', 'eval')}_{job_name_suffix}")
        subprocess.run(cmd)
        
        # Clean up temporary script
        os.remove(script_path)

if __name__ == "__main__":
    main()
