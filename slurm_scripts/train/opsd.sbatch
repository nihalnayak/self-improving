#!/bin/bash
#SBATCH --job-name=opsd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
#SBATCH --mem=400G
#SBATCH --array=0-3
#SBATCH --output=files/logs/%x-%A_%a.out

# On-policy self-distillation (sgs.opsd, OPSD/SDPO-style): a Qwen3-4B student
# (2 GPUs, FSDP + vLLM rollout) is trained only on forward KL(teacher || student)
# over the top-100 tokens of a frozen copy of the initial model (2 GPUs) that sees
# the gold sorted list in its prompt. Same data/batch/eval settings as grpo.sbatch.
# One array task per train config (see slurm_scripts/README.md).
#
# Usage (from the repo root):
#   sbatch -A <account> -p <partition> slurm_scripts/train/opsd.sbatch
# Env overrides: MODEL EPOCHS (5) TRAIN_BATCH_SIZE (32) MINI_BATCH_SIZE (32) K (8)
#   RESP_LEN (4096) OUT_ROOT (files/runs)
# Extra args pass through to sgs.opsd (CLI flags or Hydra overrides).

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
EPOCHS="${EPOCHS:-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
K="${K:-8}"
RESP_LEN="${RESP_LEN:-4096}"
DATASET="data/sorting/${FILES[$TASK]}"
VAL_DATASET="data/sorting/sorting_val_100.jsonl"
OUT_ROOT="${OUT_ROOT:-files/runs}"
M="${MODEL##*/}"
RUN_NAME="opsd-${M,,}-sorting-${CONFIGS[$TASK]}-${EPOCHS}ep-bs${TRAIN_BATCH_SIZE}-mbs${MINI_BATCH_SIZE}-k${K}-resp${RESP_LEN}"
# -j<array job>_<task> suffix lets eval jobs find this run from TRAIN_JOB=<array job>_<task>.
OUT_DIR="${OUT_ROOT}/${RUN_NAME}-$(date +%Y%m%d-%H%M%S)-j${SLURM_ARRAY_JOB_ID:-local}_${TASK}"
mkdir -p "${OUT_DIR}"
echo "host: $(hostname)  run: ${RUN_NAME}  out: ${OUT_DIR}  data: ${DATASET}"
nvidia-smi -L


"${PYTHON}" -m sgs.opsd \
    --dataset "${DATASET}" \
    --val-dataset "${VAL_DATASET}" \
    --model "${MODEL}" \
    --out-dir "${OUT_DIR}" \
    --run-name "${RUN_NAME}" \
    --n-gpus 2 \
    --teacher-gpus 2 \
    --epochs "${EPOCHS}" \
    --train-batch-size "${TRAIN_BATCH_SIZE}" \
    --mini-batch-size "${MINI_BATCH_SIZE}" \
    --k "${K}" \
    --response-length "${RESP_LEN}" \
    "$@"
