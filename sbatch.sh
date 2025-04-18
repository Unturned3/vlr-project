#!/bin/bash

###
### Modify this file as appropriate.
### e.g. slurm partition name, GPU types, python interpreter path, etc.
###

#SBATCH --job-name=vlr
#SBATCH --partition=general
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:L40S:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

echo 'sbatch script running...'

# NCCL support on the Babel cluster is unreliable
export NCCL_P2P_DISABLE=1

python='/data/user_data/yusenh/vlr/bin/python3'
srun $python train.py
