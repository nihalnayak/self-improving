#!/bin/bash
#SBATCH --job-name=eval-sort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00
#SBATCH --mem=128G
#SBATCH --output=files/logs/%x-%j.out

# Solver pass@5 on sorting (sgs.eval_sort) for ONE run's last checkpoint, or for any
# HF model dir / hub id. Writes ${RESULTS}/${LABEL}/summary.jsonl and per-sample
# completions. Plot afterwards with slurm_scripts/visualize/sort_eval.sbatch.
#
# Usage (from the repo root):
#   sbatch -A <account> -p <partition> \
#       --export=ALL,LABEL=grpo-all,TRAIN_JOB=12345_3,DATASETS="train=data/sorting/sorting_1000.jsonl val=data/sorting/sorting_val_100.jsonl" \
#       slurm_scripts/eval/eval_sort.sbatch
#   sbatch ... --export=ALL,LABEL=base-all,CKPT=Qwen/Qwen3-4B,DATASETS="..." slurm_scripts/eval/eval_sort.sbatch
# Required env: LABEL, and either TRAIN_JOB (<array job>_<task> of a training job; its run
#   dir is found as ${OUT_ROOT}/*-j${TRAIN_JOB}) or CKPT (HF model dir, run dir, or hub id).
# Optional env: DATASETS (space-separated split=path.jsonl; default: eval_sort's train + val)
#   RESULTS (default files/results/sgs/eval_sorting_qwen3-4b_5ep) OUT_ROOT (files/runs)
# Extra CLI flags pass through to sgs.eval_sort.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[[ -f env.sh ]] && source env.sh   # optional, user-provided (modules, HF_HOME, PYTHON, OUT_ROOT)
PYTHON="${PYTHON:-python}"
export TOKENIZERS_PARALLELISM=false

LABEL="${LABEL:?LABEL is required}"
OUT_ROOT="${OUT_ROOT:-files/runs}"
RESULTS="${RESULTS:-files/results/sgs/eval_sorting_qwen3-4b_5ep}"
if [[ -n "${TRAIN_JOB:-}" ]]; then
    matches=( "${OUT_ROOT}"/*-j"${TRAIN_JOB}" )
    if [[ ${#matches[@]} -ne 1 || ! -d "${matches[0]}" ]]; then
        echo "expected exactly one run dir ${OUT_ROOT}/*-j${TRAIN_JOB}, got: ${matches[*]}" >&2
        exit 1
    fi
    CKPT="${matches[0]}"
fi
CKPT="${CKPT:?TRAIN_JOB or CKPT is required}"
echo "host: $(hostname)  label: ${LABEL}  ckpt: ${CKPT}  results: ${RESULTS}/${LABEL}"
nvidia-smi -L

"${PYTHON}" -m sgs.eval_sort \
    --checkpoints "${CKPT}" \
    --last-only \
    ${DATASETS:+--datasets ${DATASETS}} \
    --out-dir "${RESULTS}/${LABEL}" \
    "$@"
