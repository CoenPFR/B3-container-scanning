#!/bin/bash -e
#SBATCH --job-name=yolov12_train
#SBATCH --output=yolov12_train.out
#SBATCH --error=yolov12_train.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Load necessary modules
module load conda

# Activate conda environment
conda activate hrrcxh_yolov12

# Run the training script
srun python train_model.py
