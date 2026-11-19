#!/bin/bash
#SBATCH --job-name=handrolled-grpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
#SBATCH --mem=400G
#SBATCH --array=0-3
#SBATCH --output=files/logs/%x-%A_%a.out

# Solver-only GRPO with the hand-rolled SGS trainer (sgs.handrolled_solver_grpo):
# sgs.sgs_grpo without Conjecturer/Guide, configured like grpo.sbatch (one on-policy
# step per 32-prompt batch, frozen KL ref with kl_coef 0.001, eval once per epoch).
# A parity check of the hand-rolled trainer against verl's. 2 GPUs.
# One array task per train config (see slurm_scripts/README.md).
#
# Usage (from the repo root):
#   sbatch -A <account> -p <partition> slurm_scripts/train/handrolled_grpo.sbatch
# Env overrides: MODEL EPOCHS (5) TRAIN_BATCH_SIZE (32) K (8) RESP_LEN (4096) KL_COEF (0.001)
#   OUT_ROOT (files/runs)
# Extra CLI flags pass through to sgs.handrolled_solver_grpo.

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
K="${K:-8}"
RESP_LEN="${RESP_LEN:-4096}"
KL_COEF="${KL_COEF:-0.001}"
DATASET="data/sorting/${FILES[$TASK]}"
VAL_DATASET="data/sorting/sorting_val_100.jsonl"
OUT_ROOT="${OUT_ROOT:-files/runs}"
# verl-style epochs: |D| // batch optimizer steps per epoch (last partial batch dropped).
STEPS_PER_EPOCH=$(( $(wc -l < "${DATASET}") / TRAIN_BATCH_SIZE ))
M="${MODEL##*/}"
RUN_NAME="handrolled-grpo-${M,,}-sorting-${CONFIGS[$TASK]}-${EPOCHS}ep-bs${TRAIN_BATCH_SIZE}-mbs${TRAIN_BATCH_SIZE}-k${K}-resp${RESP_LEN}"
# -j<array job>_<task> suffix lets eval jobs find this run from TRAIN_JOB=<array job>_<task>.
OUT_DIR="${OUT_ROOT}/${RUN_NAME}-$(date +%Y%m%d-%H%M%S)-j${SLURM_ARRAY_JOB_ID:-local}_${TASK}"
mkdir -p "${OUT_DIR}"
echo "host: $(hostname)  run: ${RUN_NAME}  out: ${OUT_DIR}  data: ${DATASET}"
nvidia-smi -L


"${PYTHON}" -m sgs.handrolled_solver_grpo \
    --dataset "${DATASET}" \
    --val-dataset "${VAL_DATASET}" \
    --model "${MODEL}" \
    --out-dir "${OUT_DIR}" \
    --run-name "${RUN_NAME}" \
    --n-gpus 2 \
    --iters $(( EPOCHS * STEPS_PER_EPOCH )) \
    --mini-batch-size "${TRAIN_BATCH_SIZE}" \
    --eval-every "${STEPS_PER_EPOCH}" \
    --save-every 0 \
    --k "${K}" \
    --solver-response-length "${RESP_LEN}" \
    --kl-coef "${KL_COEF}" \
    "$@"
