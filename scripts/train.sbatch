#!/bin/bash
#SBATCH --job-name=soar
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/soar-%j.out

# Launches soar.train under torchrun with one process per allocated GPU.
# WORLD_SIZE==1 is detected inside train.py and skips FSDP automatically, so this
# script also works for single-GPU runs (just set --gres=gpu:1).

set -euo pipefail
mkdir -p logs

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "[sbatch] host=$(hostname) detected ${NUM_GPUS} GPUs"

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    -m soar.train "$@"
