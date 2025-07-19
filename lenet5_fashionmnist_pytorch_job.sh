#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mincpus=16
#SBATCH --job-name=python
#SBATCH --chdir=/data/home/ss231/computer-vision-python
#SBATCH --output=lenet5_fashionmnist_pytorch.out
#SBATCH --error=lenet5_fashionmnist_pytorch.err
#SBATCH --time=6-12:00:00

# Load necessary modules
# module load python/3.11.11-cuda
module load python/3.12.8

# Experimental model training
# python /data/home/ss231/computer-vision-python/lenet5_fashionmnist_pytorch.py \
#    --use_tensorboard \
#    --use_gpu \
#    --epochs 100 \
#    --optimizer sgd

# Final model training with hyperparameter tuning
python /data/home/ss231/computer-vision-python/lenet5_fashionmnist_pytorch.py \
    --use_gpu \
    --epochs 45 \
    --optimizer sgd \
    --save_model

# Job completion check
if [ $? -eq 0 ]; then
    echo "Job completed successfully."
else
    echo "Job failed."
fi