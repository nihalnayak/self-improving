# Self-Guided Self-Play

SGS ([arXiv 2604.20209](https://arxiv.org/abs/2604.20209)) and baselines (GRPO, REINFORCE++, OPSD)
on algorithmic tasks with verifiable answers: sorting, multiplication, LCM (`lcm`,
`lcm_wo_coprime`) and preorder traversal. Trainers are built on
[verl](https://github.com/volcengine/verl). To run experiments, see
[slurm_scripts/README.md](slurm_scripts/README.md).

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

## Repository layout

```
sgs/              trainers (one file per method) and shared code
  tasks.py        per-task prompts, verifiers and loaders (--task)
  eval_sort.py    pass@n evaluation of checkpoints / HF models on any task (vLLM)
data/             one directory per task (see data/README.md)
misc/             data generators (gen_*_data.py) and standalone probes
tests/            per-task pipeline tests (python tests/test_<task>.py)
analysis/         post-hoc analysis of runs (synthetic-problem difficulty)
visualize/        plots (training curves, pass@n bars, pass@k, synthetic-problem stats)
slurm_scripts/    sbatch scripts for train / eval / visualize (see slurm_scripts/README.md)
files/            run outputs, logs and results (gitignored)
soar/, baselines/, eval.py, configs/   legacy SOAR reproduction on MATH (not used by SGS)
```