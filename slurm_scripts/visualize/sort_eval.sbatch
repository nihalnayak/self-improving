#!/bin/bash
#SBATCH --job-name=plot-sort-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-00:30:00
#SBATCH --mem=16G
#SBATCH --output=files/logs/%x-%j.out

# Re-plot every eval under RESULTS (one subdir per label, written by
# slurm_scripts/eval/eval_sort.sbatch). CPU only. Writes into RESULTS:
#   visualize.sort_eval   sort_eval.{pdf,png}, summary.csv, sort_eval_val_by_config.{pdf,png}
#   visualize.pass_at_k   pass@k curves per method
#
# Usage (from the repo root):
#   sbatch -A <account> -p <partition> slurm_scripts/visualize/sort_eval.sbatch
# Optional env: RESULTS (default files/results/sgs/eval_sorting_qwen3-4b_5ep)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[[ -f env.sh ]] && source env.sh   # optional, user-provided (modules, HF_HOME, PYTHON, OUT_ROOT)
PYTHON="${PYTHON:-python}"

RESULTS="${RESULTS:-files/results/sgs/eval_sorting_qwen3-4b_5ep}"
"${PYTHON}" -m visualize.sort_eval --results-dir "${RESULTS}"
"${PYTHON}" -m visualize.pass_at_k --sort-results-dir "${RESULTS}" --save-png
echo "figures in ${RESULTS}"
