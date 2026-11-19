#!/bin/bash
#SBATCH --job-name=synthetic-difficulty
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00
#SBATCH --mem=64G
#SBATCH --output=files/logs/%x-%j.out

# Difficulty of SGS synthetic problems vs their original targets, per iteration
# (analysis.synthetic_difficulty): mean list length and base-model pass@5 on every
# distinct list in <run>/synthetic.jsonl. One output dir per run under
# files/results/sgs/synthetic_difficulty/<run name>/. Works on any SGS-family run
# (sgs, sgs_grpo, sgs_onpolicy_grpo[_mv]). Plot all of them with
# slurm_scripts/visualize/sgs_synth.sbatch.
#
# Usage (from the repo root):
#   sbatch -A <account> -p <partition> --export=ALL,TRAIN_JOBS="12345_0 12345_3" \
#       slurm_scripts/eval/synthetic_difficulty.sbatch
# Required env: TRAIN_JOBS (space-separated <array job>_<task> of SGS training jobs)
# Optional env: MODEL (base model, default Qwen/Qwen3-4B) OUT_ROOT (files/runs)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[[ -f env.sh ]] && source env.sh   # optional, user-provided (modules, HF_HOME, PYTHON, OUT_ROOT)
PYTHON="${PYTHON:-python}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn   # avoids "Cannot re-initialize CUDA in forked subprocess"

TRAIN_JOBS="${TRAIN_JOBS:?TRAIN_JOBS is required}"
OUT_ROOT="${OUT_ROOT:-files/runs}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"

for job in ${TRAIN_JOBS}; do
    matches=( "${OUT_ROOT}"/*-j"${job}" )
    if [[ ${#matches[@]} -ne 1 || ! -d "${matches[0]}" ]]; then
        echo "expected exactly one run dir ${OUT_ROOT}/*-j${job}, got: ${matches[*]}" >&2
        exit 1
    fi
    echo "run: ${matches[0]}"
    "${PYTHON}" -m analysis.synthetic_difficulty --run "${matches[0]}" --model "${MODEL}"
done
