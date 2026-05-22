#!/bin/bash
#SBATCH --job-name=rl
#SBATCH --account=cil
#SBATCH --time=00:30:00
#SBATCH --gpus=2080ti:1
#SBATCH --output=/home/dchileban/rl/logs/%j.out
#SBATCH --error=/home/dchileban/rl/logs/%j.err
set -e

cd /home/dchileban/so101-world-model/model

uv sync --extra cu126

uv run torchrun \
    --nproc_per_node=1 \
    -m scripts.train \
    --config=cosmos_predict2/configs/config.py \
    -- experiment=w2a_lerobot_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz1
