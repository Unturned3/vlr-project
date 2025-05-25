#!/bin/bash -l

# SBATCH --array=0
#SBATCH --job-name=vlr
#SBATCH --partition=general
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:L40S:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G

export OMP_NUM_THREADS=4    # Set to n_cpu / n_processes.
export NCCL_P2P_DISABLE=1   # Don't use NV-Link (causes hangs on some systems).

MASTER_ADDR='localhost'
MASTER_PORT=$(shuf -i 10000-65535 -n 1)  # Random port

echo "MASTER_PORT: $MASTER_PORT"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

echo
nvidia-smi --query-gpu=index,gpu_uuid --format=csv
echo

patches_per_side=(3 10 14 18 22 25 27 30 32)
crop_px=$((27 * 16))

conda activate vlrh

srun --cpu-bind=none \
    torchrun \
    --nnodes=1 \
    --nproc_per_node=gpu \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    train.py \
    crop_image_to_px=$crop_px
