#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mincpus=16
#SBATCH --job-name=python
#SBATCH --chdir=/data/home/ss231/computer-vision-python
#SBATCH --output=alexnet_fashionmnist_pytorch_final.out
#SBATCH --error=alexnet_fashionmnist_pytorch_final.err
#SBATCH --time=6-12:00:00

module load python/3.12.8
python /data/home/ss231/computer-vision-python/alexnet_fashionmnist_pytorch_final.py
if [ $? -eq 0 ]; then
    echo "Job completed successfully."
else
    echo "Job failed."
fi