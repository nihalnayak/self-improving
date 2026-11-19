#!/bin/bash
#SBATCH --job-name=sgs-onpolicy-grpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=1-12:00:00
#SBATCH --mem=400G
#SBATCH --array=0-3
#SBATCH --output=files/logs/%x-%A_%a.out

# SGS with an on-policy GRPO Solver (sgs.sgs_onpolicy_grpo): Qwen3-4B Solver,
# Conjecturer and frozen Guide (also the KL reference); one optimizer step per
# freshly sampled 32-prompt chunk; fresh optimizer state every iteration.
# One array task per train config (see slurm_scripts/README.md).
#
# Usage (from the repo root):
#   sbatch -A <account> -p <partition> slurm_scripts/train/sgs_onpolicy_grpo.sbatch
# Env overrides: TASK_NAME (sorting | multiplication) MODEL NUM_ITERS (5) MINI_BATCH_SIZE (32) K (8) RESP_LEN (4096) OUT_ROOT (files/runs)
# Extra CLI flags pass through to sgs.sgs_onpolicy_grpo (e.g. --all-targets).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[[ -f env.sh ]] && source env.sh   # optional, user-provided (modules, HF_HOME, PYTHON, OUT_ROOT)
PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false RAY_DEDUP_LOGS=0
# Ray session dir on node-local /tmp: long shared-filesystem paths push Ray's unix
# sockets past the 107-byte AF_UNIX limit.
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID:-local}"
mkdir -p "${RAY_TMPDIR}"

TASK_NAME="${TASK_NAME:-sorting}"   # sorting | multiplication (sgs/tasks.py)
case "${TASK_NAME}" in
    sorting)
        CONFIGS=(L8-128 L8-16-128 L8-16-32-128 all)
        FILES=(sorting_1000_L8-128.jsonl sorting_1000_L8-16-128.jsonl sorting_1000_L8-16-32-128.jsonl sorting_1000.jsonl) ;;
    multiplication)
        CONFIGS=(L2-10 L2-4-10 L2-4-6-10 all)
        FILES=(multiplication_1000_L2-10.jsonl multiplication_1000_L2-4-10.jsonl multiplication_1000_L2-4-6-10.jsonl multiplication_1000.jsonl) ;;
    *) echo "unknown TASK_NAME=${TASK_NAME}" >&2; exit 1 ;;
esac
TASK="${SLURM_ARRAY_TASK_ID:-0}"

MODEL="${MODEL:-Qwen/Qwen3-4B}"
NUM_ITERS="${NUM_ITERS:-5}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
K="${K:-8}"
RESP_LEN="${RESP_LEN:-4096}"
DATASET="data/${TASK_NAME}/${FILES[$TASK]}"
VAL_DATASET="data/${TASK_NAME}/${TASK_NAME}_val_100.jsonl"
OUT_ROOT="${OUT_ROOT:-files/runs}"
M="${MODEL##*/}"
RUN_NAME="sgs-onpolicy-grpo-${M,,}-${TASK_NAME}-${CONFIGS[$TASK]}-${NUM_ITERS}iters-mbs${MINI_BATCH_SIZE}-k${K}-optim-reset"
# -j<array job>_<task> suffix lets eval jobs find this run from TRAIN_JOB=<array job>_<task>.
OUT_DIR="${OUT_ROOT}/${RUN_NAME}-$(date +%Y%m%d-%H%M%S)-j${SLURM_ARRAY_JOB_ID:-local}_${TASK}"
mkdir -p "${OUT_DIR}"
echo "host: $(hostname)  run: ${RUN_NAME}  out: ${OUT_DIR}  data: ${DATASET}"
nvidia-smi -L

"${PYTHON}" -m sgs.sgs_onpolicy_grpo \
    --task "${TASK_NAME}" \
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
    --reset-optimizer \
    "$@"
