#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --output=myjob.out
#SBATCH --error=myjob.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --partition=your_partition

# Load any necessary modules or dependencies
module load some_module

# Activate the virtual environment
source /path/to/venv/bin/activate

# Run your Python script
python myscript.py

# Deactivate the virtual environment
deactivate