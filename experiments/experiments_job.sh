#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mincpus=16
#SBATCH --job-name=python
#SBATCH --chdir=/data/home/ss231/computer-vision-python
#SBATCH --output=experiments.out
#SBATCH --error=experiments.err
#SBATCH --time=01:00:00

module load python/3.12.8
# module load python/3.11.11-cuda

python /data/home/ss231/computer-vision-python/fashionmnist_mlp_pytorch.py
if [ $? -eq 0 ]; then
    echo "Job completed successfully."
else
    echo "Job failed."
fi