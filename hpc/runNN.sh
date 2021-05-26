#!/bin/bash

#SBATCH --job-name=runGRU
#SBATCH --output=logs/job.%j.out
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --gres=gpu
#SBATCH --partition=brown
#SBATCH --mem=8G

echo Executing...
python3 ../models/trainNN.py
echo Finished!