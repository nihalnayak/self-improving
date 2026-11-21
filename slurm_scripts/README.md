# slurm_scripts

Train, evaluate and plot GRPO vs SGS-OnPolicy-GRPO (Qwen3-4B) on three tasks: `lcm`,
`lcm_wo_coprime` (LCM pairs with GCD > 1) and `preorder`. Run every command from the repo root.

## Setup (once)

1. Build the Python env: see [Installation](../README.md#installation).
2. Create `env.sh` at the repo root (gitignored; every job sources it). Set `PYTHON` to your env:
   ```bash
   module purge
   module load gcc/15.2.0-fasrc01 cuda/13.3.1-fasrc01
   export PYTHON="${HOME}/.conda/envs/sgs/bin/python"
   export PATH="$(dirname "${PYTHON}"):${PATH}"
   export HF_HOME="/n/netscratch/dam_lab/Lab/${USER}/.cache/huggingface"
   export OUT_ROOT="${OUT_ROOT:-/n/netscratch/dam_lab/Lab/${USER}/sgs/runs}"   # runs + checkpoints
   export TOKENIZERS_PARALLELISM=false
   ```
3. Log in to wandb and create the log dir:
   ```bash
   wandb login
   mkdir -p files/logs
   ```
4. In every shell you submit from (or in `~/.bashrc`):
   ```bash
   export SBATCH_ACCOUNT=kempner_dam_lab SBATCH_PARTITION=kempner_h200
   ```
5. Check: `source env.sh && "$PYTHON" tests/test_lcm.py` should end with `N/N passed`.

## Run train → eval → viz

Do the three steps once per task: `T=lcm`, `T=lcm_wo_coprime`, `T=preorder`.

### 1. Train + eval

GRPO (5 epochs) and SGS (5 iterations) on 4 train configs (array tasks 0-3). Each eval starts when
its own training task succeeds; the base-model evals start right away.

```bash
T=lcm                                    # or: T=lcm_wo_coprime, T=preorder
RESULTS=files/results/sgs/eval_${T}_qwen3-4b_5ep
if [[ $T == lcm* ]]; then CONFIGS=(L2-32 L2-4-32 L2-4-8-32 all); else CONFIGS=(L8-128 L8-16-128 L8-16-32-128 all); fi

G=$(sbatch --parsable --export=ALL,TASK_NAME=$T slurm_scripts/train/grpo.sbatch)
S=$(sbatch --parsable --export=ALL,TASK_NAME=$T slurm_scripts/train/sgs_onpolicy_grpo.sbatch)
echo "$T: GRPO array $G, SGS array $S"          # write these down: step 2 needs S

for i in 0 1 2 3; do
  c=${CONFIGS[$i]}; f=${T}_1000; [[ $c != all ]] && f+=_$c
  D="train=data/$T/$f.jsonl val=data/$T/${T}_val_100.jsonl"
  # eval each run's last checkpoint once its training task succeeds, and the base model now
  sbatch -t 4:00:00 --dependency=afterok:${G}_$i \
    --export=ALL,LABEL=grpo-$c,TRAIN_JOB=${G}_$i,RESULTS=$RESULTS,DATASETS="$D" slurm_scripts/eval/eval_sort.sbatch --task $T
  sbatch -t 4:00:00 --dependency=afterok:${S}_$i \
    --export=ALL,LABEL=sgs-onpolicy-grpo-reset-$c,TRAIN_JOB=${S}_$i,RESULTS=$RESULTS,DATASETS="$D" slurm_scripts/eval/eval_sort.sbatch --task $T
  sbatch -t 4:00:00 \
    --export=ALL,LABEL=base-$c,CKPT=Qwen/Qwen3-4B,RESULTS=$RESULTS,DATASETS="$D" slurm_scripts/eval/eval_sort.sbatch --task $T
done
```

Training takes about a day. Watch it with `squeue -u $USER`; logs are in `files/logs/`.

### 2. Synthetic difficulty (after SGS training finishes)

```bash
T=lcm                                    # or: T=lcm_wo_coprime, T=preorder
S=<SGS array job id>
ROOT=/n/netscratch/dam_lab/Lab/$USER/sgs/runs
for run in "$ROOT"/sgs-onpolicy-grpo-*-${T}-*-5iters-*-j${S}_*; do
sbatch -J synth --gres=gpu:1 --cpus-per-task=12 --mem=64G -t 4:00:00 -o files/logs/%x-%j.out <<EOF
#!/bin/bash
source env.sh
export VLLM_WORKER_MULTIPROC_METHOD=spawn
"\$PYTHON" -m analysis.synthetic_difficulty --task $T --run "$run" \
    --out-dir files/results/sgs/synthetic_difficulty_$T/$(basename "$run")
EOF
done
```

### 3. Plots (after the evals finish; CPU is enough)

```bash
source env.sh
T=lcm                                    # or: T=lcm_wo_coprime, T=preorder
ROOT=/n/netscratch/dam_lab/Lab/$USER/sgs/runs
R=files/results/sgs/eval_${T}_qwen3-4b_5ep
"$PYTHON" -m visualize.sort_curves --task $T --results-dir $R --runs-root $ROOT
"$PYTHON" -m visualize.sort_curves --task $T --results-dir $R --runs-root $ROOT --metric pass1
"$PYTHON" -m visualize.sort_eval   --task $T --results-dir $R
"$PYTHON" -m visualize.pass_at_k   --task $T --sort-results-dir $R --save-png
"$PYTHON" -m visualize.sgs_synth   --task $T --results-dir files/results/sgs/synthetic_difficulty_$T --runs-root $ROOT
```

Main figures:
- `files/results/sgs/eval_${T}_qwen3-4b_5ep/sort_curves_{train,val}.png`: pass@5 per epoch or
  iteration, GRPO vs SGS vs base, one row per train config
- in the same dir: `sort_eval*.png` and `pass_at_k_*.png`
- `files/results/sgs/synthetic_difficulty_${T}/sgs_synth.png`
