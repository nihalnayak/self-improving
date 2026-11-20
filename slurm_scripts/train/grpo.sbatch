#!/bin/bash
#SBATCH --job-name=grpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
#SBATCH --mem=400G
#SBATCH --array=0-3
#SBATCH --output=files/logs/%x-%A_%a.out

# Solver-only GRPO baseline (sgs.solver_simple, verl's PPO trainer with
# adv_estimator=grpo): same targets, prompt and verifier as SGS; train + val pass@5
# once per epoch. Also used by reinforce_pp.sbatch and grpo_sgs_synth.sbatch.
# One array task per train config (see slurm_scripts/README.md).
#
# Usage (from the repo root):
#   sbatch -A <account> -p <partition> slurm_scripts/train/grpo.sbatch
# Env overrides: MODEL EPOCHS (5) TRAIN_BATCH_SIZE (32) MINI_BATCH_SIZE (32) K (8)
#   RESP_LEN (4096) OUT_ROOT (files/runs)
#   ADV_ESTIMATOR (grpo | reinforce_plus_plus) METHOD (run-name prefix, default grpo)
#   TASK_NAME (sorting | multiplication | lcm | preorder, default sorting)
#   DATA_DIR (train-file directory, default data/${TASK_NAME})
# Extra args pass through to sgs.solver_simple: CLI flags or Hydra overrides
#   (e.g. trainer.total_training_steps=3).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[[ -f env.sh ]] && source env.sh   # optional, user-provided (modules, HF_HOME, PYTHON, OUT_ROOT)
PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false RAY_DEDUP_LOGS=0
# Ray session dir on node-local /tmp: long shared-filesystem paths push Ray's unix
# sockets past the 107-byte AF_UNIX limit.
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID:-local}"
mkdir -p "${RAY_TMPDIR}"

TASK_NAME="${TASK_NAME:-sorting}"   # sorting | multiplication | lcm | preorder (sgs/tasks.py)
case "${TASK_NAME}" in
    sorting)
        CONFIGS=(L8-128 L8-16-128 L8-16-32-128 all)
        FILES=(sorting_1000_L8-128.jsonl sorting_1000_L8-16-128.jsonl sorting_1000_L8-16-32-128.jsonl sorting_1000.jsonl) ;;
    multiplication)
        CONFIGS=(L2-10 L2-4-10 L2-4-6-10 all)
        FILES=(multiplication_1000_L2-10.jsonl multiplication_1000_L2-4-10.jsonl multiplication_1000_L2-4-6-10.jsonl multiplication_1000.jsonl) ;;
    lcm)
        CONFIGS=(L2-32 L2-4-32 L2-4-8-32 all)
        FILES=(lcm_1000_L2-32.jsonl lcm_1000_L2-4-32.jsonl lcm_1000_L2-4-8-32.jsonl lcm_1000.jsonl) ;;
    preorder)
        CONFIGS=(L8-128 L8-16-128 L8-16-32-128 all)
        FILES=(preorder_1000_L8-128.jsonl preorder_1000_L8-16-128.jsonl preorder_1000_L8-16-32-128.jsonl preorder_1000.jsonl) ;;
    *) echo "unknown TASK_NAME=${TASK_NAME}" >&2; exit 1 ;;
esac
TASK="${SLURM_ARRAY_TASK_ID:-0}"


MODEL="${MODEL:-Qwen/Qwen3-4B}"
EPOCHS="${EPOCHS:-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
K="${K:-8}"
RESP_LEN="${RESP_LEN:-4096}"
ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"
METHOD="${METHOD:-grpo}"
DATA_DIR="${DATA_DIR:-data/${TASK_NAME}}"
DATASET="${DATA_DIR}/${FILES[$TASK]}"
VAL_DATASET="data/${TASK_NAME}/${TASK_NAME}_val_100.jsonl"
OUT_ROOT="${OUT_ROOT:-files/runs}"
M="${MODEL##*/}"
RUN_NAME="${METHOD}-${M,,}-${TASK_NAME}-${CONFIGS[$TASK]}-${EPOCHS}ep-bs${TRAIN_BATCH_SIZE}-mbs${MINI_BATCH_SIZE}-k${K}-resp${RESP_LEN}"
# -j<array job>_<task> suffix lets eval jobs find this run from TRAIN_JOB=<array job>_<task>.
OUT_DIR="${OUT_ROOT}/${RUN_NAME}-$(date +%Y%m%d-%H%M%S)-j${SLURM_ARRAY_JOB_ID:-local}_${TASK}"
mkdir -p "${OUT_DIR}"
echo "host: $(hostname)  run: ${RUN_NAME}  out: ${OUT_DIR}  data: ${DATASET}"
nvidia-smi -L


"${PYTHON}" -m sgs.solver_simple \
    --task "${TASK_NAME}" \
    --dataset "${DATASET}" \
    --val-dataset "${VAL_DATASET}" \
    --model "${MODEL}" \
    --out-dir "${OUT_DIR}" \
    --run-name "${RUN_NAME}" \
    --n-gpus 4 \
    --adv-estimator "${ADV_ESTIMATOR}" \
    --epochs "${EPOCHS}" \
    --train-batch-size "${TRAIN_BATCH_SIZE}" \
    --mini-batch-size "${MINI_BATCH_SIZE}" \
    --k "${K}" \
    --response-length "${RESP_LEN}" \
    "$@"
