#!/bin/bash
#SBATCH --job-name=soar_train
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=0-10:00:00
#SBATCH --mem=128G
#SBATCH --output=/n/netscratch/dam_lab/Lab/nnayak1/soar/logs/soar-%j.out
#SBATCH --error=/n/netscratch/dam_lab/Lab/nnayak1/soar/logs/soar-%j.err

set -euo pipefail

timestamp() { date +"%Y-%m-%d %H:%M:%S %Z"; }
log()  { echo "[$(timestamp)] $*"; }
warn() { echo "[$(timestamp)] [WARN] $*" >&2; }
die()  { echo "[$(timestamp)] [ERROR] $*" >&2; exit 1; }

env_directory_setup() {
    module purge
    module load cuda gcc/13.2.0-fasrc01
    export HF_HOME="/n/netscratch/dam_lab/Lab/nnayak1/.cache/huggingface"
    export PYTHON="/n/holylabs/dam_lab/Lab/nnayak1/envs/synthetic_persona/bin/python"
    export PATH="$(dirname "${PYTHON}"):${PATH}"
    export CUR_DIR="/n/holylabs/dam_lab/Users/nnayak1/self-improving"
    cd "${CUR_DIR}"
}

env_directory_setup

SUBJECT="${SUBJECT:-math}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
MODEL_SHORT="$(basename "${BASE_MODEL}" | tr '[:upper:]' '[:lower:]')"
OUT_ROOT="${OUT_ROOT:-/n/netscratch/dam_lab/Lab/nnayak1/soar/runs}"
RUN_NAME="${RUN_NAME:-soar-${SUBJECT}-${MODEL_SHORT}-$(date +%Y%m%d-%H%M%S)-j${SLURM_JOB_ID:-local}}"
OUT_DIR="${OUT_ROOT}/${RUN_NAME}"

mkdir -p "${OUT_DIR}"
mkdir -p "/n/netscratch/dam_lab/Lab/nnayak1/soar/logs"

log "run name:  ${RUN_NAME}"
log "out dir:   ${OUT_DIR}"
log "subject:   ${SUBJECT}"
log "base:      ${BASE_MODEL}"

"${PYTHON}" -m soar.train --out "${OUT_DIR}" "$@"

log "training completed"
