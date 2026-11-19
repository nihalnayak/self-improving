# Self-Guided Self-Play for Sorting

This repo implements **SGS (Self-Guided Self-Play, [arXiv 2604.20209](https://arxiv.org/abs/2604.20209))** and its variants on a controlled task: sorting lists of integers.
Three copies of an LLM (Qwen3-4B) play different roles:

- **Solver** (trained) sorts lists.
- **Conjecturer** (trained) proposes easier "stepping-stone" lists for the targets the Solver cannot solve yet.
- **Guide** (frozen) scores how useful each proposal is.

The Solver learns from both the real targets and the synthetic problems. The project asks whether self-generated curricula help a model solve problems (long lists) it initially fails on. It compares SGS against solver-only RL (GRPO, REINFORCE++) and on-policy self-distillation (OPSD).

All trainers are built on [verl](https://github.com/volcengine/verl) (FSDP actors + vLLM rollouts, orchestrated by Ray).

## Methods

| Method | Module | Slurm script | Description |
| --- | --- | --- | --- |
| SGS | `sgs/sgs_simple.py` | `train/sgs.sbatch` | Paper Algorithm 1. REINFORCE½ Solver, REINFORCE Conjecturer, frozen Guide. |
| SGS-GRPO | `sgs/sgs_grpo.py` | `train/sgs_grpo.sbatch` | GRPO Solver (PPO clip + KL to the Guide). Rollouts are sampled once per iteration. |
| SGS on-policy GRPO | `sgs/sgs_onpolicy_grpo.py` | `train/sgs_onpolicy_grpo.sbatch` | Like SGS-GRPO, but each optimizer step uses fresh rollouts. |
| SGS on-policy GRPO + majority vote | `sgs/sgs_onpolicy_grpo_majority_vote.py` | `train/sgs_onpolicy_grpo_mv.sbatch` | Synthetic problems are rewarded by Solver majority vote, without ground truth. |
| GRPO | `sgs/solver_simple.py` | `train/grpo.sbatch` | Solver-only baseline using verl's PPO trainer. |
| REINFORCE++ | `sgs/solver_simple.py` | `train/reinforce_pp.sbatch` | Same as GRPO with `--adv-estimator reinforce_plus_plus`. |
| GRPO (SGS synth.) | `sgs/solver_simple.py` | `train/grpo_sgs_synth.sbatch` | GRPO on the train targets plus SGS's synthetic problems. |
| Hand-rolled GRPO | `sgs/handrolled_solver_grpo.py` | `train/handrolled_grpo.sbatch` | Solver-only GRPO with the SGS trainer code (parity check vs. verl). |
| OPSD | `sgs/opsd.py` | `train/opsd.sbatch` | On-policy self-distillation: forward KL to a frozen teacher that sees the gold answer. |

## Repository layout

```
sgs/              trainers (one file per method) + common.py (prompt, verifier, verl worker patches)
  eval_sort.py    pass@n evaluation of checkpoints / HF models on sorting (vLLM)
misc/             data generation and standalone probes
  gen_sort_data.py          generate sorting datasets
  make_sgs_synth_sorting.py build the train + SGS-synthetic dataset
  sort_benchmark.py         Solver prompt / parser, base-model pass@5 benchmark
  sort_proposer.py          Conjecturer prompt probe
  sort_opsd_kl.py           OPSD teacher prompt + KL probe
analysis/         post-hoc analysis of runs (synthetic-problem difficulty, Guide re-rating)
visualize/        plots (pass@n bars, pass@k curves, synthetic-problem distributions)
slurm_scripts/    train/, eval/, visualize/ sbatch scripts (see slurm_scripts/README.md)
data/sorting/     train and validation sorting datasets
configs/          accelerate/FSDP config (legacy SOAR code)
```

Run outputs (checkpoints, logs, results) go under `files/`, which is gitignored.

## Installation

Requires Linux with NVIDIA GPUs. Experiments used H100/H200s: 4 GPUs per training run and 1 per eval.

```bash
git clone <repo-url> self-improving && cd self-improving

# On a cluster with environment modules, load a compiler and CUDA first.
# (These are the versions we used on Harvard FASRC; use your site's equivalents.)
module load gcc/15.2.0-fasrc01 cuda/13.3.1-fasrc01

conda create -n sgs python=3.12 -y
conda activate sgs

pip install torch==2.13.0 vllm==0.31.0           # pick the CUDA build matching your driver
pip install verl==0.9.1                           # required: every trainer in sgs/ is built on verl
pip install transformers==5.17.0 trl==1.16.0 peft==0.21.2 accelerate==1.15.0 \
    datasets==5.1.0 ray==2.60.0 wandb==0.31.0 math-verify==0.9.0 \
    numpy pandas matplotlib seaborn

wandb login                                       # or: export WANDB_MODE=offline
mkdir -p files/logs                               # slurm log directory
```

You don't need to install the repo as a package. Run everything from the repo root as modules (`python -m sgs.<module>`).
(`requirements.txt` is from the earlier SOAR code and is out of date; use the versions above.)

**Optional `env.sh`.** If an `env.sh` exists at the repo root (it is gitignored), every slurm script sources it first. Put site-specific setup there:

```bash
# env.sh
module load gcc/<version> cuda/<version>
export PYTHON=/path/to/envs/sgs/bin/python
export HF_HOME=/path/to/large/disk/huggingface
export OUT_ROOT=/path/to/large/disk/sgs/runs   # run dirs + checkpoints (default: files/runs)
```

## Data

The datasets are in `data/sorting/`. Each line is `{"list": [...], "sorted": [...]}`, with integers in [0, 10000].

| File | Contents |
| --- | --- |
| `sorting_1000.jsonl` | 1000 train lists, 200 at each length 8, 16, 32, 64, 128 (`all`) |
| `sorting_1000_L8-128.jsonl` | 1000 train lists, lengths 8 and 128 |
| `sorting_1000_L8-16-128.jsonl` | 1000 train lists, lengths 8, 16, 128 |
| `sorting_1000_L8-16-32-128.jsonl` | 1000 train lists, lengths 8, 16, 32, 128 |
| `sorting_val_100.jsonl` | 100 validation lists, 20 at each length 8 … 128 |

To regenerate them exactly:

```bash
python misc/gen_sort_data.py --out data/sorting/sorting_1000.jsonl
python misc/gen_sort_data.py --n 1000 --lengths 8,128        --out data/sorting/sorting_1000_L8-128.jsonl
python misc/gen_sort_data.py --n 1000 --lengths 8,16,128     --out data/sorting/sorting_1000_L8-16-128.jsonl
python misc/gen_sort_data.py --n 1000 --lengths 8,16,32,128  --out data/sorting/sorting_1000_L8-16-32-128.jsonl
python misc/gen_sort_data.py --per-length 20 --seed 1        --out data/sorting/sorting_val_100.jsonl
```

## Training

Each training script is a slurm array with 4 tasks, one per train file above (task 0 = `L8-128` … task 3 = `all`). Defaults: Qwen3-4B, 5 iterations (SGS) or 5 epochs (baselines), 8 rollouts per prompt, 4096-token responses.

```bash
# all four train configs
sbatch -A <account> -p <partition> slurm_scripts/train/sgs_onpolicy_grpo.sbatch
# one config, with overrides
sbatch -A <account> -p <partition> --array=3 --export=ALL,NUM_ITERS=10 slurm_scripts/train/sgs_onpolicy_grpo.sbatch
```

To run directly on a GPU node without slurm:

```bash
python -m sgs.sgs_onpolicy_grpo \
    --dataset data/sorting/sorting_1000.jsonl --val-dataset data/sorting/sorting_val_100.jsonl \
    --out-dir files/runs/sgs-onpolicy-grpo-debug --n-gpus 4 --iters 5 --reset-optimizer
```

Every module documents its arguments in its docstring and `--help`. Train/val pass@5 is logged to wandb (project `sgs`) and to the run directory after every iteration or epoch.

## Evaluation and visualization

```bash
# pass@5 per list length for a run's last checkpoint (TRAIN_JOB = <array job id>_<task>)
sbatch -A <account> -p <partition> \
    --export=ALL,LABEL=sgs-onpolicy-grpo-all,TRAIN_JOB=12345_3,DATASETS="train=data/sorting/sorting_1000.jsonl val=data/sorting/sorting_val_100.jsonl" \
    slurm_scripts/eval/eval_sort.sbatch
# the base model
sbatch ... --export=ALL,LABEL=base-all,CKPT=Qwen/Qwen3-4B,DATASETS="..." slurm_scripts/eval/eval_sort.sbatch

# plots over every eval in the results dir (CPU)
sbatch -A <account> -p <cpu-partition> slurm_scripts/visualize/sort_eval.sbatch
```

- Results go to `files/results/sgs/eval_sorting_qwen3-4b_5ep/<label>/`, and figures are written to the same directory.
- For SGS runs, `eval/synthetic_difficulty.sbatch` followed by `visualize/sgs_synth.sbatch` analyzes how hard the synthetic problems are across iterations.

See [`slurm_scripts/README.md`](slurm_scripts/README.md) for the full script reference, the train → eval → plot dependency chain, and troubleshooting.

## Legacy: SOAR

This repo started as a reproduction of SOAR ([arXiv 2601.18778](https://arxiv.org/abs/2601.18778), "Teaching Models to Teach Themselves") on MATH. That code is kept for reference but is not part of the SGS experiments:

- `soar/`: the SOAR trainers.
- `baselines/`: its baselines.
- `eval.py`: MATH eval.
- `misc/build_fail128.py`, `misc/stats_fail128.py`: building the fail@128 MATH subset.
- `analysis/judge_promotion_questions.py`: judging promotion questions.
