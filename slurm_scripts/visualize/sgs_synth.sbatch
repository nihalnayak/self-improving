#!/bin/bash
#SBATCH --job-name=plot-sgs-synth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-00:30:00
#SBATCH --mem=16G
#SBATCH --output=files/logs/%x-%j.out

# Per-iteration difficulty and length distribution of SGS synthetic problems, one
# column per train config (visualize.sgs_synth). Reads the outputs of
# slurm_scripts/eval/synthetic_difficulty.sbatch for every 5-iter SGS run under
# RESULTS_DIR. CPU only. Writes sgs_synth.{pdf,png,csv} into RESULTS_DIR.
#
# NOTE: visualize/sgs_synth.py reads <run>/synthetic.jsonl from a hardcoded RUNS
# path at the top of the file; point it at your OUT_ROOT before running.
#
# Usage (from the repo root):
#   sbatch -A <account> -p <partition> slurm_scripts/visualize/sgs_synth.sbatch
# Optional env: RESULTS_DIR (default files/results/sgs/synthetic_difficulty) MODEL (axis label)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[[ -f env.sh ]] && source env.sh   # optional, user-provided (modules, HF_HOME, PYTHON, OUT_ROOT)
PYTHON="${PYTHON:-python}"

RESULTS_DIR="${RESULTS_DIR:-files/results/sgs/synthetic_difficulty}"
"${PYTHON}" -m visualize.sgs_synth --results-dir "${RESULTS_DIR}" --model "${MODEL:-Qwen/Qwen3-4B}"
