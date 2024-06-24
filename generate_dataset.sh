#!/bin/bash
#Define the resource requirements here using #SBATCH


#Max wallTime for the job
#SBATCH -t 48:00:00

#Resource requiremenmt commands end here
#SBATCH -p nvidia
#SBATCH --mem=480G
#SBATCH --gres=gpu:a100:1
#SBATCH -c 32
#SBATCH -w cn254

#Add the lines for running your code/application
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate cost_model_env

#Execute the code
python generate_dataset.py