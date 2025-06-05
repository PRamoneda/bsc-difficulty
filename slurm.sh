#!/bin/bash
#SBATCH -J try
#SBATCH -N 1
#SBATCH --account=upf97
#SBATCH --partition=acc
#SBATCH --qos=acc_resa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --time=72:00:00
#SBATCH --output=debug_%j_output.txt
#SBATCH --mail-user=pedro.ramoneda@upf.edu

module load anaconda
source ../bsc-structure/optuna/bin/activate


python3 main.py