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

# This is for my experiments varing the number of patches.
# Remove if not needed.
#patches_per_side=(3 10 14 18 22 25 27 30 32)
#crop_px=$((18 * 16))
#echo "Image size: $crop_px"

# NCCL support on the Babel cluster is unreliable
export NCCL_P2P_DISABLE=1

# NOTE we need to escape the equal sign in the ckpt path!
# Or else Hydra will complain.
#ckpt='logs/jd9v8xm8-epoch\=43.ckpt'
ckpt='null'

python='/data/user_data/yusenh/vlr/bin/python3'
for patch_size in 8 16 32 64
do
    for num_layers in 4 6 8 10
    do
        srun $python train.py \
            exp_group=vary_patchsize_X_num_layers
    done
done
