#!/bin/bash
#SBATCH --job-name=reinforce-pp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
#SBATCH --mem=400G
#SBATCH --array=0-3
#SBATCH --output=files/logs/%x-%A_%a.out

# Solver-only REINFORCE++ baseline: grpo.sbatch with
# adv_estimator=reinforce_plus_plus.
# One array task per train config (see slurm_scripts/README.md).
#
# Usage (from the repo root):
#   sbatch -A <account> -p <partition> slurm_scripts/train/reinforce_pp.sbatch
# Env overrides and extra args: as in grpo.sbatch.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export ADV_ESTIMATOR=reinforce_plus_plus METHOD="${METHOD:-reinforce-pp}"
exec bash slurm_scripts/train/grpo.sbatch "$@"
