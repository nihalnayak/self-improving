#!/bin/bash
#SBATCH --job-name=sgs-onpolicy-grpo-mv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=1-12:00:00
#SBATCH --mem=400G
#SBATCH --array=0-3
#SBATCH --output=files/logs/%x-%A_%a.out

# On-policy SGS-GRPO with majority-vote rewards on synthetic problems
# (sgs.sgs_onpolicy_grpo_majority_vote): synthetic problems are scored without ground
# truth; the k Solver answers vote and groups with agreement >= MV_RATIO are
# pseudo-labelled by the mode. Otherwise identical to sgs_onpolicy_grpo.sbatch.
# One array task per train config (see slurm_scripts/README.md).
#
# Usage (from the repo root):
#   sbatch -A <account> -p <partition> slurm_scripts/train/sgs_onpolicy_grpo_mv.sbatch
#   sbatch -A <account> -p <partition> --export=ALL,MV_RATIO=0.25 slurm_scripts/train/sgs_onpolicy_grpo_mv.sbatch
# Env overrides: MV_RATIO (0.5) MODEL NUM_ITERS (5) MINI_BATCH_SIZE (32) K (8) RESP_LEN (4096) OUT_ROOT (files/runs)
# Extra CLI flags pass through to sgs.sgs_onpolicy_grpo_majority_vote.

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

MV_RATIO="${MV_RATIO:-0.5}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
NUM_ITERS="${NUM_ITERS:-5}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
K="${K:-8}"
RESP_LEN="${RESP_LEN:-4096}"
DATASET="data/sorting/${FILES[$TASK]}"
VAL_DATASET="data/sorting/sorting_val_100.jsonl"
OUT_ROOT="${OUT_ROOT:-files/runs}"
M="${MODEL##*/}"
RUN_NAME="sgs-onpolicy-grpo-mv${MV_RATIO}-${M,,}-sorting-${CONFIGS[$TASK]}-${NUM_ITERS}iters-mbs${MINI_BATCH_SIZE}-k${K}-optim-reset"
# -j<array job>_<task> suffix lets eval jobs find this run from TRAIN_JOB=<array job>_<task>.
OUT_DIR="${OUT_ROOT}/${RUN_NAME}-$(date +%Y%m%d-%H%M%S)-j${SLURM_ARRAY_JOB_ID:-local}_${TASK}"
mkdir -p "${OUT_DIR}"
echo "host: $(hostname)  run: ${RUN_NAME}  out: ${OUT_DIR}  data: ${DATASET}"
nvidia-smi -L

"${PYTHON}" -m sgs.sgs_onpolicy_grpo_majority_vote \
    --mv-ratio "${MV_RATIO}" \
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
