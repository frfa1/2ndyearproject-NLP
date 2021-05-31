#!/bin/bash

#SBATCH --job-name=RNN_4
#SBATCH --output=logs/job.%j.out
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --gres=gpu
#SBATCH --partition=brown
#SBATCH --mem=8G
#SBATCH --mail-type=BEGIN,END,FAIL

echo Executing...
echo RNN_4
python3 ../models/RNN_4.py
echo Finished!