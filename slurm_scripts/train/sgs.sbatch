#!/bin/bash
#SBATCH --job-name=sgs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
#SBATCH --mem=400G
#SBATCH --array=0-3
#SBATCH --output=files/logs/%x-%A_%a.out

# SGS (Self-Guided Self-Play, arXiv 2604.20209; sgs.sgs_simple): Qwen3-4B Solver,
# Conjecturer and frozen Guide; REINFORCE-1/2 Solver and REINFORCE Conjecturer
# (paper Algorithm 1).
# One array task per train config (see slurm_scripts/README.md).
#
# Usage (from the repo root):
#   sbatch -A <account> -p <partition> slurm_scripts/train/sgs.sbatch
# Env overrides: MODEL NUM_ITERS (5) MINI_BATCH_SIZE (256) K (8) RESP_LEN (4096) OUT_ROOT (files/runs)
#   RESET_OPTIMIZER=1 (fresh optimizer state every iteration; adds -optim-reset)
# Extra CLI flags pass through to sgs.sgs_simple.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[[ -f env.sh ]] && source env.sh   # optional, user-provided (modules, HF_HOME, PYTHON, OUT_ROOT)
PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false RAY_DEDUP_LOGS=0
# Ray session dir on node-local /tmp: long shared-filesystem paths push Ray's unix
# sockets past the 107-byte AF_UNIX limit.
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID:-local}"
mkdir -p "${RAY_TMPDIR}"

CONFIGS=(L8-128 L8-16-128 L8-16-32-128 all)
FILES=(sorting_1000_L8-128.jsonl sorting_1000_L8-16-128.jsonl sorting_1000_L8-16-32-128.jsonl sorting_1000.jsonl)
TASK="${SLURM_ARRAY_TASK_ID:-0}"

MODEL="${MODEL:-Qwen/Qwen3-4B}"
NUM_ITERS="${NUM_ITERS:-5}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-256}"
K="${K:-8}"
RESP_LEN="${RESP_LEN:-4096}"
DATASET="data/sorting/${FILES[$TASK]}"
VAL_DATASET="data/sorting/sorting_val_100.jsonl"
OUT_ROOT="${OUT_ROOT:-files/runs}"
EXTRA_ARGS=(); OPTIM_TAG=""
if [[ "${RESET_OPTIMIZER:-0}" == "1" ]]; then EXTRA_ARGS+=(--reset-optimizer); OPTIM_TAG="-optim-reset"; fi
M="${MODEL##*/}"
RUN_NAME="sgs-${M,,}-sorting-${CONFIGS[$TASK]}-${NUM_ITERS}iters-mbs${MINI_BATCH_SIZE}-k${K}${OPTIM_TAG}"
# -j<array job>_<task> suffix lets eval jobs find this run from TRAIN_JOB=<array job>_<task>.
OUT_DIR="${OUT_ROOT}/${RUN_NAME}-$(date +%Y%m%d-%H%M%S)-j${SLURM_ARRAY_JOB_ID:-local}_${TASK}"
mkdir -p "${OUT_DIR}"
echo "host: $(hostname)  run: ${RUN_NAME}  out: ${OUT_DIR}  data: ${DATASET}"
nvidia-smi -L

"${PYTHON}" -m sgs.sgs_simple \
    --dataset "${DATASET}" \
    --val-dataset "${VAL_DATASET}" \
    --model "${MODEL}" \
    --out-dir "${OUT_DIR}" \
    --run-name "${RUN_NAME}" \
    --n-gpus 4 \
    --iters "${NUM_ITERS}" \
    --mini-batch-size "${MINI_BATCH_SIZE}" \
    --k "${K}" \
    --solver-response-length "${RESP_LEN}" \
    --conj-response-length "${RESP_LEN}" \
    --guide-response-length "${RESP_LEN}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
