#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mincpus=16
#SBATCH --job-name=python
#SBATCH --chdir=/data/home/ss231/computer-vision-python
#SBATCH --output=googlenet_cifar10_pytorch.out
#SBATCH --error=googlenet_cifar10_pytorch.err
#SBATCH --time=6-12:00:00

# Load necessary modules
# module load python/3.11.11-cuda
module load python/3.12.8

# Experimental model training
python /data/home/ss231/computer-vision-python/googlenet_cifar10_pytorch.py \
    --use_tensorboard \
    --use_gpu \
    --epochs 100 \
    --optimizer sgd

# Final model training with hyperparameter tuning
# python /data/home/ss231/computer-vision-python/googlenet_cifar10_pytorch.py \
#    --use_gpu \
#    --epochs 15 \
#    --optimizer sgd \
#    --save_model

# Job completion check
if [ $? -eq 0 ]; then
    echo "Job completed successfully."
else
    echo "Job failed."
fi