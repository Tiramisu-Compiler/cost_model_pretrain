#!/bin/bash
#Define the resource requirements here using #SBATCH


#Max wallTime for the job
#SBATCH -t 30:00:00

#Resource requiremenmt commands end here
#SBATCH -q c2
#SBATCH -p nvidia
#SBATCH --mem=480G
#SBATCH --gres=gpu:a100:1
#SBATCH -c 32

#Add the lines for running your code/application
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate cost_model_env

#Execute the code
python generate_comp_tensors_mp.py