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

## Notes for AI agents

- Do not write `--account` / `--partition` or absolute paths into these scripts. Pass them at submit time, or put them in `env.sh`.
- Always run `sbatch` from the repo root. Check a script with `sbatch --test-only -A <acct> -p <part> <script>` before submitting.
- Monitor jobs with `squeue -u $USER` and `sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS`. Logs are in `files/logs/`.
- To debug on an interactive GPU node, run the train script directly (`bash slurm_scripts/train/<x>.sbatch`); with no `SLURM_ARRAY_TASK_ID` it runs task 0. Or call the module yourself with `python -m sgs.<module> --help`.
- Common failures:
  - **Ray `AF_UNIX path length` errors.** The scripts already set `RAY_TMPDIR=/tmp/ray_$SLURM_JOB_ID`; keep it on node-local disk.
  - **vLLM OOM / KV-cache errors.** Lower `--gpu-memory-utilization` (default 0.6) or `RESP_LEN`, or give the job more GPUs (keep `--n-gpus` in sync with `--gres`).
  - **`expected exactly one run dir ...-j<id>`** from eval. The `TRAIN_JOB` must be `<array job id>_<task>`, and `OUT_ROOT` must match the one used for training.
