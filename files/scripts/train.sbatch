#!/bin/bash
#SBATCH --job-name=soar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=runs/soar-%j.out
#SBATCH --error=runs/soar-%j.err

set -euo pipefail
source ~/.bashrc
mamba activate /n/holylabs/dam_lab/Lab/nnayak1/envs/synthetic_persona

cd /n/holylabs/dam_lab/Users/nnayak1/self-improving

# Prevent HF from writing to home if quota-limited; adjust if needed.
# export HF_HOME=/tmp/hf

# For a smoke run, pass --smoke.
python -m soar.train "$@"
