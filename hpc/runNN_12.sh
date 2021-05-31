#!/bin/bash

#SBATCH --job-name=RNN_12
#SBATCH --output=logs/job.%j.out
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --gres=gpu
#SBATCH --partition=brown
#SBATCH --mem=8G
#SBATCH --mail-type=BEGIN,END,FAIL

echo Executing...
echo RNN_12
python3 ../models/RNN_12.py
echo Finished!