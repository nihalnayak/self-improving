# slurm_scripts

Slurm batch scripts for every experiment in this repo, split by stage:

```
slurm_scripts/
  train/       one script per method; a 4-task array over the train configs
  eval/        GPU evaluation of trained runs (pass@5) and of SGS synthetic problems
  visualize/   CPU-only plotting of eval results
```

The scripts contain no cluster-specific settings: no account, partition, paths, modules, or
credentials. You pass those at submit time or put them in an optional `env.sh`.

To run the LCM, LCM-without-coprimes and preorder experiments, do the one-time
[setup](#first-time-setup-new-lab-member), then follow
[Running lcm, lcm_wo_coprime and preorder](#running-lcm-lcm_wo_coprime-and-preorder).

## First-time setup (new lab member)

Do this once on the FASRC cluster. Everything is per user: your own clone, Python env,
Hugging Face cache and wandb login. Runs and checkpoints go under `/n/netscratch/dam_lab/Lab/$USER/`.

1. **Clone the repo and build the environment.** Follow [Installation](../README.md#installation)
   in the top-level README (conda env with torch/vllm/verl at the pinned versions). Run every
   command in this README **from the repo root**.

2. **Write `env.sh` at the repo root.** It is gitignored, so you keep your own copy. Every script
   here sources it at the start of the job. Template for FASRC (replace the `PYTHON` path with
   your env):

   ```bash
   # env.sh -- sourced by every slurm script at the start of the job.
   module purge
   module load gcc/15.2.0-fasrc01 cuda/13.3.1-fasrc01
   export PYTHON="${HOME}/.conda/envs/sgs/bin/python"                       # your env's python
   export PATH="$(dirname "${PYTHON}"):${PATH}"
   export HF_HOME="/n/netscratch/dam_lab/Lab/${USER}/.cache/huggingface"    # model downloads (large)
   # run dirs + checkpoints (large). The ${OUT_ROOT:-...} form lets one submission override it
   # (sbatch --export=ALL,OUT_ROOT=...) without env.sh overwriting it.
   export OUT_ROOT="${OUT_ROOT:-/n/netscratch/dam_lab/Lab/${USER}/sgs/runs}"
   export TOKENIZERS_PARALLELISM=false
   ```

   Check it with `source env.sh && "$PYTHON" -c "import verl, vllm, datasets; print('ok')"`.
   netscratch is not backed up and old files are purged, so copy anything you need to keep.
   `files/results/` is small and is a good place for it.

3. **Log in to wandb** once with `wandb login`; it stores your key in `~/.netrc`. Metrics go to
   wandb project `sgs`. To run without wandb, `export WANDB_MODE=offline` before submitting.

4. **Set the Slurm account and partition** in every shell you submit from (or in `~/.bashrc`).
   The scripts deliberately contain none:

   ```bash
   export SBATCH_ACCOUNT=kempner_dam_lab SBATCH_PARTITION=kempner_h200
   ```

   Check that you are on the account with `sacctmgr -nP show assoc user=$USER format=account`.

5. **Create the log directory.** `mkdir -p files/logs`. It is gitignored, and Slurm won't create it,
   so a job without it dies before it starts.

6. **Sanity checks before launching anything big:**
   - `sbatch --test-only slurm_scripts/train/grpo.sbatch` validates the script without submitting.
   - `"$PYTHON" tests/test_lcm.py` and `"$PYTHON" tests/test_preorder.py` must each end with
     `N/N passed`.
   - On an interactive GPU node, `nvidia-smi -L` must show a full GPU. If it lists a `MIG`
     slice, don't run vLLM or training there; use `sbatch`.

## Prerequisites

1. Install the environment (see [Installation](../README.md#installation) in the top-level README) and log in to wandb once (`wandb login`). To skip wandb, `export WANDB_MODE=offline`.
2. **Submit every script from the repo root.** Scripts `cd "$SLURM_SUBMIT_DIR"` and use repo-relative paths.
3. `mkdir -p files/logs`. Slurm writes `files/logs/<job-name>-<jobid>[_<task>].out` there and will not create the directory itself.
4. The sorting data must exist in `data/sorting/` (see [Data](../README.md#data)).
5. Optional: `env.sh` at the repo root (gitignored). If present, every script `source`s it first. Use it for anything site-specific:

   ```bash
   # env.sh (example)
   module load gcc/<version> cuda/<version>
   export PYTHON=/path/to/env/bin/python        # default: python on PATH
   export HF_HOME=/path/to/large/disk/huggingface
   export OUT_ROOT=/path/to/large/disk/sgs/runs # default: files/runs (checkpoints are large)
   ```

## Submitting

Account and partition always come from the command line or the environment:

```bash
sbatch -A <account> -p <partition> slurm_scripts/train/sgs_onpolicy_grpo.sbatch
# or once per shell:
export SBATCH_ACCOUNT=<account> SBATCH_PARTITION=<partition>
sbatch slurm_scripts/train/sgs_onpolicy_grpo.sbatch
```

Here is how to change what runs:

- **Run one train config only.** Override the array, e.g. `sbatch --array=3 ...` for `all`.
- **Change hyperparameters.** Use the env overrides listed in each script header: `sbatch --export=ALL,NUM_ITERS=10,K=16 ...`.
- **Pass anything else to the Python module.** Append it after the script: `sbatch ... slurm_scripts/train/grpo.sbatch --lr 2e-6`. The `solver_simple`-based scripts (`grpo`, `reinforce_pp`, `grpo_sgs_synth`, `opsd`) also accept Hydra overrides, e.g. `trainer.total_training_steps=3`.

## Array index → train config

All train scripts use `--array=0-3`, and each task trains on one file in `data/sorting/`. Validation is always `sorting_val_100.jsonl`, which has 20 lists at each length 8, 16, 32, 64, 128.

| task | config | train file | list lengths (1000 lists) |
| --- | --- | --- | --- |
| 0 | `L8-128` | `sorting_1000_L8-128.jsonl` | 8, 128 |
| 1 | `L8-16-128` | `sorting_1000_L8-16-128.jsonl` | 8, 16, 128 |
| 2 | `L8-16-32-128` | `sorting_1000_L8-16-32-128.jsonl` | 8, 16, 32, 128 |
| 3 | `all` | `sorting_1000.jsonl` | 8, 16, 32, 64, 128 |

## Script reference

All train scripts use Qwen3-4B by default (`MODEL=...` to change), with k=8 rollouts per prompt and a 4096-token response length (`RESP_LEN`).

### train/

| script | Python module | GPUs / time / mem | what it trains | main env overrides (default) |
| --- | --- | --- | --- | --- |
| `sgs.sbatch` | `sgs.sgs_simple` | 4 / 24h / 400G | SGS (paper Alg. 1): Solver (REINFORCE½) + Conjecturer (REINFORCE) + frozen Guide | `NUM_ITERS` (5), `MINI_BATCH_SIZE` (256), `RESET_OPTIMIZER` (0) |
| `sgs_grpo.sbatch` | `sgs.sgs_grpo` | 4 / 24h / 400G | SGS with a GRPO Solver; rollouts sampled once per iteration (off-policy mini-batches) | `NUM_ITERS` (5), `MINI_BATCH_SIZE` (32), `RESET_OPTIMIZER` (0) |
| `sgs_onpolicy_grpo.sbatch` | `sgs.sgs_onpolicy_grpo` | 4 / 36h / 400G | SGS with an on-policy GRPO Solver (fresh rollouts every step), `--reset-optimizer` | `NUM_ITERS` (5), `MINI_BATCH_SIZE` (32) |
| `sgs_onpolicy_grpo_mv.sbatch` | `sgs.sgs_onpolicy_grpo_majority_vote` | 4 / 36h / 400G | As above, but synthetic problems are rewarded by majority vote, without ground truth | `MV_RATIO` (0.5), `NUM_ITERS` (5) |
| `grpo.sbatch` | `sgs.solver_simple` | 4 / 24h / 400G | Solver-only GRPO baseline (verl PPO trainer) | `EPOCHS` (5), `TRAIN_BATCH_SIZE` (32), `MINI_BATCH_SIZE` (32), `ADV_ESTIMATOR` (grpo), `DATA_DIR` |
| `reinforce_pp.sbatch` | `sgs.solver_simple` | 4 / 24h / 400G | Solver-only REINFORCE++ baseline (`grpo.sbatch` with `ADV_ESTIMATOR=reinforce_plus_plus`) | as `grpo.sbatch` |
| `grpo_sgs_synth.sbatch` | `sgs.solver_simple` | 4 / 24h / 400G | GRPO on train + SGS synthetic problems (`grpo.sbatch` with `DATA_DIR=files/data/sorting_sgs_onpolicy_5iters_synth`) | as `grpo.sbatch` |
| `handrolled_grpo.sbatch` | `sgs.handrolled_solver_grpo` | 2 / 24h / 400G | Solver-only GRPO with the hand-rolled SGS trainer (parity check vs. verl) | `EPOCHS` (5), `TRAIN_BATCH_SIZE` (32), `KL_COEF` (0.001) |
| `opsd.sbatch` | `sgs.opsd` | 4 (2 student + 2 teacher) / 24h / 400G | On-policy self-distillation: forward KL to a frozen teacher that sees the gold answer | `EPOCHS` (5), `TRAIN_BATCH_SIZE` (32), `MINI_BATCH_SIZE` (32) |

Every train script writes to `${OUT_ROOT}/<run-name>-<YYYYmmdd-HHMMSS>-j<array job id>_<task>/`. `OUT_ROOT` defaults to `files/runs`. The `-j<array job id>_<task>` suffix is how the eval scripts find a run from `TRAIN_JOB=<array job id>_<task>`. Each run dir holds the following:

- **Checkpoints.** The last step is always saved, and the eval uses it.
- **Per-epoch or per-iteration train/val pass@5.** SGS variants and `handrolled_grpo` write it to `eval_pass5.jsonl`, `opsd` writes it to `eval_metrics.jsonl`, and `grpo` / `reinforce_pp` log it only to wandb. Every method also writes `eval_samples/`.
- **`synthetic.jsonl`** (SGS variants only): every synthetic problem, per iteration.

All metrics are logged to wandb project `sgs`.

### eval/

| script | Python module | resources | inputs (env) | outputs |
| --- | --- | --- | --- | --- |
| `eval_sort.sbatch` | `sgs.eval_sort --last-only` | 1 GPU / 2h / 128G | `LABEL` (required); `TRAIN_JOB=<array>_<task>` **or** `CKPT=<HF dir, run dir or hub id>`; `DATASETS="train=<file> val=<file>"`; `RESULTS` (default `files/results/sgs/eval_sorting_qwen3-4b_5ep`) | `${RESULTS}/${LABEL}/summary.jsonl` (pass@5 per length, per split) + `<split>_samples.jsonl` |
| `synthetic_difficulty.sbatch` | `analysis.synthetic_difficulty` | 1 GPU / 2h / 64G | `TRAIN_JOBS="<array>_<task> ..."` (SGS-family runs only); `MODEL` | `files/results/sgs/synthetic_difficulty/<run name>/` (length + base pass@5 per iteration, csv/pdf/png) |

Use labels of the form `<method>-<config>` (e.g. `grpo-L8-128`, `base-all`). `visualize.sort_eval` parses them to draw per-config panels. To evaluate the base model, set `CKPT=Qwen/Qwen3-4B` and use a label like `base-<config>`.

### visualize/ (CPU only, use a CPU partition)

| script | Python module(s) | inputs (env) | outputs |
| --- | --- | --- | --- |
| `sort_eval.sbatch` | `visualize.sort_eval`, `visualize.pass_at_k` | `RESULTS` (default `files/results/sgs/eval_sorting_qwen3-4b_5ep`) | in `RESULTS`: `sort_eval.{pdf,png}`, `sort_eval_val_by_config.{pdf,png}`, `summary.csv`, pass@k plots |
| `sgs_synth.sbatch` | `visualize.sgs_synth` | `RESULTS_DIR` (default `files/results/sgs/synthetic_difficulty`) | in `RESULTS_DIR`: `sgs_synth.{pdf,png,csv}` |

`visualize/sgs_synth.py` reads each run's `synthetic.jsonl` from a `RUNS` path hardcoded at the top of the file. Point it at your `OUT_ROOT` first.

## Typical workflow: train → eval → plot

This chain submits training, then one eval per array task (each waits for its own task with `afterok`), then a plot job once all evals have ended (`afterany`):

```bash
METHOD=sgs_onpolicy_grpo          # file name in slurm_scripts/train/
LABEL=sgs-onpolicy-grpo-reset     # label prefix in the results dir
CPU_PARTITION="<cpu partition>"   # for the plotting job
CONFIGS=(L8-128 L8-16-128 L8-16-32-128 all)
FILES=(sorting_1000_L8-128.jsonl sorting_1000_L8-16-128.jsonl sorting_1000_L8-16-32-128.jsonl sorting_1000.jsonl)

A=$(sbatch --parsable slurm_scripts/train/${METHOD}.sbatch)
EVALS=()
for i in 0 1 2 3; do
  EVALS+=($(sbatch --parsable --dependency=afterok:${A}_${i} \
    --export=ALL,LABEL=${LABEL}-${CONFIGS[$i]},TRAIN_JOB=${A}_${i},DATASETS="train=data/sorting/${FILES[$i]} val=data/sorting/sorting_val_100.jsonl" \
    slurm_scripts/eval/eval_sort.sbatch))
done
sbatch -p "${CPU_PARTITION}" --dependency=afterany:$(IFS=:; echo "${EVALS[*]}") slurm_scripts/visualize/sort_eval.sbatch
```

(This assumes `SBATCH_ACCOUNT` / `SBATCH_PARTITION` are exported. `CPU_PARTITION` overrides the partition for the CPU-only plot job.)

## Ordering dependencies

- `grpo_sgs_synth.sbatch` needs finished `sgs_onpolicy_grpo.sbatch` runs first. Then run `python misc/make_sgs_synth_sorting.py`, after editing its `RUNS` path and per-config job ids, to build `files/data/sorting_sgs_onpolicy_5iters_synth/`.
- `visualize/sgs_synth.sbatch` needs `eval/synthetic_difficulty.sbatch` to have run on the SGS runs. It only picks up 5-iteration runs (`*-5iters-*`).
- `visualize/sort_eval.sbatch` plots whatever `eval_sort.sbatch` has written so far. Re-run it after each new eval.

## Running lcm, lcm_wo_coprime and preorder

These runs compare GRPO with SGS-OnPolicy-GRPO on three tasks (`sgs/tasks.py`; data described in
[data/README.md](../data/README.md)):

| experiment | task | data | level |
| --- | --- | --- | --- |
| `lcm` | least common multiple of two integers | `data/lcm/lcm_1000*.jsonl`, `lcm_val_100.jsonl` | operand digits: 2, 4, 8, 16, 32 |
| `lcm_wo_coprime` | same, only pairs with GCD > 1 (its own task; inherits everything from `lcm`) | `data/lcm_wo_coprime/lcm_wo_coprime_*.jsonl` | same |
| `preorder` | preorder traversal from inorder + postorder | `data/preorder/preorder_*.jsonl` | tree nodes: 8, 16, 32, 64, 128 |

Each experiment trains 4 configs (array tasks 0-3) × 2 methods with Qwen3-4B on 4 GPUs: GRPO for
5 epochs and SGS for 5 iterations. It then evaluates the last checkpoints and the base model on
the config's train file plus the task's `*_val_100.jsonl` (20 problems per level).

| array task | lcm / lcm_wo_coprime config | preorder config |
| --- | --- | --- |
| 0 | `L2-32` (digits 2, 32) | `L8-128` (nodes 8, 128) |
| 1 | `L2-4-32` | `L8-16-128` |
| 2 | `L2-4-8-32` | `L8-16-32-128` |
| 3 | `all` (every level) | `all` |

Do the [first-time setup](#first-time-setup-new-lab-member) first. Then, from the repo root
with `SBATCH_ACCOUNT` / `SBATCH_PARTITION` exported, there are three steps per experiment:
1. Submit training and evals.
2. After SGS training: synthetic difficulty.
3. After all evals: plots.

Runtimes from 2-iteration test runs on 4 H200s:
- SGS: about 2 h (lcm) to 3 h (preorder) per iteration, so a 5-iteration job takes 10-16 h.
- GRPO: about 1 h per epoch.
- One eval: under 4 h.

### Step 1: submit training and evals

Set `T` to the experiment and paste the block. Run it once for each of `lcm`, `lcm_wo_coprime` and
`preorder`. Each run submits 14 jobs: 2 training arrays of 4 tasks each, plus 12 evals.

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

Check progress with `squeue -u $USER`. Logs are in `files/logs/` (`grpo-<array>_<task>.out`,
`sgs-onpolicy-grpo-<array>_<task>.out`, `eval-sort-<job>.out`). Each eval writes
`$RESULTS/<label>/summary.jsonl`.

### Step 2: synthetic difficulty (after an SGS array finishes)

This measures, for each SGS run, how hard the synthetic problems were compared with the targets
they were generated for: mean level, and base-model pass@5 on every distinct problem. Set `T` and the
SGS array id `S` that step 1 printed.

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

### Step 3: plots (after the evals finish; CPU is enough)

Run these on any node, for example an interactive session, after `source env.sh`. Re-run them
whenever more evals land.

```bash
T=lcm                                    # or: T=lcm_wo_coprime, T=preorder
ROOT=/n/netscratch/dam_lab/Lab/$USER/sgs/runs
R=files/results/sgs/eval_${T}_qwen3-4b_5ep
"$PYTHON" -m visualize.sort_curves --task $T --results-dir $R --runs-root $ROOT
"$PYTHON" -m visualize.sort_curves --task $T --results-dir $R --runs-root $ROOT --metric pass1
"$PYTHON" -m visualize.sort_eval   --task $T --results-dir $R
"$PYTHON" -m visualize.pass_at_k   --task $T --sort-results-dir $R --save-png
"$PYTHON" -m visualize.sgs_synth   --task $T --results-dir files/results/sgs/synthetic_difficulty_$T --runs-root $ROOT
```

Always pass `--task`. Without it the scripts assume sorting and put every lcm problem at "level 2"
(the list `[a, b]` has length 2).

### What you get

- **`$R/sort_curves_train.png` and `sort_curves_val.png`** (the main figures). They show pass@5 per
  epoch/iteration, with one row per train config and one column per level. GRPO (orange) vs SGS
  (purple), with the base model as a dotted line. `sort_curves.csv` has the numbers, and
  `sort_curves_pass1_*` shows pass@1.
- **`$R/sort_eval.png`, `sort_eval_val_by_config.png` and `summary.csv`:** last-checkpoint pass@5
  bars. **`pass_at_k_{train,val}.png`:** pass@k for k = 1..5.
- **`files/results/sgs/synthetic_difficulty_$T/sgs_synth.png`:**
  - top row: base solve rate of synthetic vs original problems per iteration
  - bottom row: the level mix of the synthetic problems
- **`$ROOT/<run>/`:** checkpoints, `eval_pass5.jsonl` (SGS), `eval_samples/`, `synthetic.jsonl`
  (SGS). Everything is also in wandb project `sgs`.

### Common problems

- **The job log says `files/logs/...: No such file`, or no log appears.** Run `mkdir -p files/logs`.
- **An eval sits in `PENDING (DependencyNeverSatisfied)`.** Its training task failed. Read that
  task's log in `files/logs/`, then `scancel` the eval and resubmit both.
- **`expected exactly one run dir ...-j<id>`.** `TRAIN_JOB` must be `<array job id>_<task>`, and the
  eval must see the same `OUT_ROOT` as training.
- **`sort_curves` fails with `too many values to unpack`.** A label directory in `$R` holds two
  runs, because step 1 was submitted twice into the same results dir. Move the old `$R` away
  before resubmitting.
- **`sort_curves` prints `no config with all of ['grpo', 'sgs-onpolicy-grpo-reset']`.** It draws a
  config only after both its GRPO and SGS evals exist. Wait for them.
- **Scoring rules.** The answer is read from the first `\boxed{...}`, and `\boxed{5,312,856}` counts
  as 5312856. A response cut off at 4096 tokens has no box and counts as wrong.
- **lcm_wo_coprime** is its own task (`--task lcm_wo_coprime`, class `LCMWoCoprime` in
  `sgs/tasks.py`), but it inherits everything from `lcm`: prompts, loader, verifier and levels. Only
  its data (`data/lcm_wo_coprime/`) and its names in metrics, runs and results differ.
- **LCM loading.** `LCM.load` reads the jsonl with Python's `json`, not `load_dataset`, because
  pyarrow rounds 16- and 32-digit integers. Don't change that; `tests/test_lcm.py` checks it.

## Notes for AI agents

- Do not write `--account` / `--partition` or absolute paths into these scripts. Pass them at submit time, or put them in `env.sh`.
- Always run `sbatch` from the repo root. Check a script with `sbatch --test-only -A <acct> -p <part> <script>` before submitting.
- Monitor jobs with `squeue -u $USER` and `sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS`. Logs are in `files/logs/`.
- To debug on an interactive GPU node, run the train script directly (`bash slurm_scripts/train/<x>.sbatch`); with no `SLURM_ARRAY_TASK_ID` it runs task 0. Or call the module yourself with `python -m sgs.<module> --help`.
- Common failures:
  - **Ray `AF_UNIX path length` errors.** The scripts already set `RAY_TMPDIR=/tmp/ray_$SLURM_JOB_ID`; keep it on node-local disk.
  - **vLLM OOM / KV-cache errors.** Lower `--gpu-memory-utilization` (default 0.6) or `RESP_LEN`, or give the job more GPUs (keep `--n-gpus` in sync with `--gres`).
  - **`expected exactly one run dir ...-j<id>`** from eval. The `TRAIN_JOB` must be `<array job id>_<task>`, and `OUT_ROOT` must match the one used for training.
