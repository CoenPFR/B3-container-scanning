#!/bin/bash -e
#SBATCH --job-name=yolov12_val
#SBATCH --output=yolov12_val.out
#SBATCH --error=yolov12_val.err
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Load necessary modules
module load conda

# Activate conda environment
conda activate hrrcxh_yolov12

# Run the training script
srun python val_model.py
