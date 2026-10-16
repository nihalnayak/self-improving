# SOAR-mini

A minimal, hackable implementation of SOAR (arxiv 2601.18778 — "Teaching Models
to Teach Themselves"). A **teacher** adapter proposes problems, a **student**
adapter trains on each candidate, and the teacher is rewarded by the student's
measured improvement on a held-out eval set.

Scoped for one H100, Qwen2.5-0.5B-Instruct, one subject of MMLU-Pro, and ~60
lines in the main loop. Designed to be read top-to-bottom.

## Quickstart

```bash
mamba activate /n/holylabs/dam_lab/Lab/nnayak1/envs/synthetic_persona
cd /n/holylabs/dam_lab/Users/nnayak1/self-improving

# 1) sanity: baseline eval of Qwen2.5-0.5B on MMLU-Pro math
python eval.py --n 32

# 2) smoke test (1 outer step, tiny shapes)
python -m soar.train --smoke

# 3) full v1 run
python -m soar.train

# 4) on slurm (1 H100)
sbatch scripts/train.sbatch
```

## File map

Read in this order:

| File | What it does |
| --- | --- |
| `soar/config.py` | All hparams in one dataclass (`SoarConfig`). |
| `soar/data.py` | Load MMLU-Pro, filter `math`, split into Q_eval + Q_train exemplars. |
| `soar/verify.py` | Letter-match reward (extract `A`–`J` from model output). |
| `soar/teacher.py` | Few-shot prompt template + parser for teacher-generated problems. |
| `soar/models.py` | Load Qwen base + two LoRA adapters (`teacher`, `student`) + vLLM engine. |
| `soar/rollout.py` | `sync_adapter_to_disk` + `vllm_generate` (LoRA hot-swap pattern). |
| `soar/grpo.py` | Single `grpo_loss` reused by both teacher and student. |
| `soar/student.py` | `inner_train_student`: K GRPO steps on one candidate problem. |
| `soar/train.py` | **The main outer loop** (6 numbered blocks). |
| `eval.py` | Standalone student eval on MMLU-Pro math. |
| `scripts/train.sbatch` | 1x H100 slurm job. |

## How to extend

- Swap base model → `SoarConfig.base_model`.
- Different subject → `SoarConfig.subject` + change the data filter in `data.py`.
- Promotion mechanism → block 6 in `train.py` has a commented stub.
- Multi-GPU → wrap the per-candidate loop in `train.py` with `accelerate` and
  shard candidates across ranks.
- PPO clipping → add ratio clip to `grpo.grpo_loss`.
