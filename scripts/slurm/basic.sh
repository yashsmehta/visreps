#!/bin/bash
#SBATCH --job-name=visreps
#SBATCH --output=slurm_logs/test_%j.out
#SBATCH --error=slurm_logs/test_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --partition=v100
#SBATCH --qos=qos_gpu
#SBATCH --account=mbonner5_gpu

# Activate the virtual environment
source .venv/bin/activate

# Print some debug information
echo "Running on node: $(hostname)"
echo "GPU information:"
nvidia-smi

# Run the training script with test config
python -m visreps.run --mode train --config configs/train/base.json

# Deactivate the virtual environment
deactivate 