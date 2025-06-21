#!/bin/bash

set -euo pipefail

# TODO: Let W&B Sweeps handle the arguments like crop_image_to_px, etc.

torchrun \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --nproc_per_node=gpu \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    train.py crop_image_to_px=224
