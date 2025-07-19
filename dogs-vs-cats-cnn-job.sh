#!/bin/bash
#SBATCH --job-name=cnn
#SBATCH --chdir=/data/home/ss231/computer-vision-python
#SBATCH --nodes=1
#SBATCH --mincpus=8
#SBATCH --output=dogs_vs_cats_cnn.out
#SBATCH --error=dogs_vs_cats_cnn.err
#SBATCH --time=01:00:00

module load python/3.12.8
python /data/home/ss231/computer-vision-python/dogs-vs-cats-cnn.py
if [ $? -eq 0 ]; then
    echo "Job completed successfully."
else
    echo "Job failed."
fi