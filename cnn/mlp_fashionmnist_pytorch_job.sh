#!/bin/bash
#SBATCH --job-name=cnn
#SBATCH --chdir=/data/home/ss231/computer-vision-python
#SBATCH --nodes=1
#SBATCH --mincpus=8
#SBATCH --output=mlp_fashionmnist_pytorch.out
#SBATCH --error=mlp_fashionmnist_pytorch.err
#SBATCH --time=5-10:00:00

module load python/3.12.8
python /data/home/ss231/computer-vision-python/mlp_fashionmnist_pytorch.py
if [ $? -eq 0 ]; then
    echo "Job completed successfully."
else
    echo "Job failed."
fi