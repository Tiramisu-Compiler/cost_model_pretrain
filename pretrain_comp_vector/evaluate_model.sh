#!/bin/bash
#Define the resource requirements here using #SBATCH


#Max wallTime for the job
#SBATCH -t 5:00:00

#Resource requiremenmt commands end here
#SBATCH -q c2
#SBATCH -p nvidia
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100:1
#SBATCH -w cn261

#Add the lines for running your code/application
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate cost_model_env

#Execute the code
python evaluate_model.py