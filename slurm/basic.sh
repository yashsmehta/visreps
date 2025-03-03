#!/bin/bash
#SBATCH --job-name=visreps
#SBATCH --output=slurm/slurm_logs/test_%j.out
#SBATCH --error=slurm/slurm_logs/test_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --account=mbonner5_gpu

# Activate the virtual environment
source .venv/bin/activate

# Print some debug information
echo "Running on node: $(hostname)"
echo "CPU information:"
lscpu | grep "Model name"
echo "Memory information:"
free -h
echo "Disk information:"
df -h /scratch4/mbonner5/shared/imagenet
echo "GPU information:"
nvidia-smi

# Run the training script with test config
python -m visreps.run --mode train --config configs/train/base.json

# Deactivate the virtual environment
deactivate 