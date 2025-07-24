#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mincpus=16
#SBATCH --job-name=python
#SBATCH --chdir=/data/home/ss231/computer-vision-python
#SBATCH --output=gan_cifar10_pytorch.out
#SBATCH --error=gan_cifar10_pytorch.err
#SBATCH --time=6-12:00:00

# Load necessary modules
# module load python/3.11.11-cuda
module load python/3.12.8

# Model training
python /data/home/ss231/computer-vision-python/gan_cifar10_pytorch.py

# Job completion check
if [ $? -eq 0 ]; then
    echo "Job completed successfully."
else
    echo "Job failed."
fi