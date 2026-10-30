"""SOAR training loop — full-parameter FSDP (via accelerate) + TRL RLOO/GRPO + TRL 1.16 colocate vLLM.

Launch:
    accelerate launch --config_file configs/fsdp.yaml --num_processes=$NUM_GPUS \
        -m soar.train_trl --out-dir ... --fail128-dir ...

Single GPU: `--num_processes=1` (accelerate skips FSDP, same code path).

Multi-GPU: `--num_processes=2..4` (or more). Both teacher and inner-loop student
are fully sharded via FSDP. Inner-loop trainer picked by `cfg.rl_algo` ∈
{"rloo","grpo","grpo_no_norm"} — defaults to RLOO, matching
`train_single_gpu_trl_lora.py`. "grpo" sets scale_rewards=True, "grpo_no_norm"
sets scale_rewards=False. Teacher outer update uses matching advantages
(leave-one-out, (R-mean)/std, or R-mean).

Six phases per outer step (SOAR Alg. 1):

    1. TEACHER GENERATES g*n parsed problems — vLLM on rank 0 only, read from
       `_scratch/teacher/`; destroy engine before returning so vLLM's KV cache
       is gone before any training backward.
    2. SAMPLE Q_R — baseline eval runs inside the first inner trainer's
       colocate engine (no separate ad-hoc engine).
    3. INNER TRAINING: for each (X_k, j in 1..r):
         - build a FRESH student_model from `_scratch/baseline_student/`
         - run RLOOTrainer or GRPOTrainer (TRL wraps with FSDP internally); its
           auto-created ref_model gets its weights overwritten with
           baseline_student_ckpt so KL tracks the live baseline (not the
           pretrained base).
         - save trained student (full-param, FSDP-gather) to
           `_scratch/trained_oS_kK_jJ/`
         - rank-sharded vLLM greedy-eval on Q_R via trainer.vllm_generation.llm
           → R[k][j] = acc - base_acc
         - thorough colocate teardown, del trainer, del student_model
    4. PROMOTION CHECK — rolling-m MA of R̄_t; if >τ, _replace_dir the chosen
       trained ckpt into `_scratch/baseline_student/` + copy to
       `promotions/promo_NNN/student/`. Then (always) dump X[k] + R_k rows to
       `teacher_data/outer_{step}/` for post-hoc analysis.
    5. TEACHER RLOO UPDATE — dataset-level LOO advantages over g rewards,
       broadcast to each X_k's n rollouts. teacher_model is pre-wrapped FSDP;
       teacher_ref is a second FSDP-wrapped model built fresh from
       `_scratch/teacher/`, used for KL, destroyed after. Save teacher_model
       back to `_scratch/teacher/` and mirror to `teacher_latest/`.
    6. LOGGING + CHECKPOINT — wandb.log; every cfg.ckpt_every outer steps,
       cp -r the two scratch dirs to ckpt_outer{step}/.

Model layout:

    teacher_model      — persistent FSDP-wrapped, teacher_opt prepared with it.
                         Weights saved to disk after each phase-5 update.
    student_model      — built fresh from disk per inner run, FSDP-wrapped by
                         TRL internally.
    teacher_ref        — built fresh from disk at phase-5 start, FSDP-wrapped,
                         frozen. Destroyed at phase-5 end.
    `_scratch/teacher/`           — HF-layout, vLLM-readable
    `_scratch/baseline_student/`  — HF-layout, vLLM-readable
    `_scratch/trained_oS_kK_jJ/`  — HF-layout, deleted at end of outer step
                                    (promoted copy survives via promotions/)
    `teacher_latest/`             — HF-layout mirror of current teacher
                                    (updated every outer step for external evals)

Memory discipline: vLLM engine is destroyed (`del engine; gc.collect();
empty_cache()`) at the end of every generation call. TRL colocate engine is
torn down via `engine_core.shutdown()` + `destroy_model_parallel()` + nulling
`vllm_generation.llm` before trainer/student teardown. teacher_ref is destroyed
at end of phase 5. Opt-in instrumentation via `SOAR_MEM_DEBUG=1`.
"""

import argparse
import gc
import json
import math
import os
import random
import shutil
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import List, Optional, Tuple

import datasets
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedType, broadcast_object_list
from safetensors.torch import load_file as _load_safetensors
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, RLOOConfig, RLOOTrainer

# Force vLLM CUDA graphs off for TRL's colocate init. TRL 1.16's
# VLLMGeneration._init_vllm calls LLM(...) without `enforce_eager`; on vLLM 0.32
# (above TRL's supported 0.31 ceiling) the captured-graph state outlives
# engine_core.shutdown() + destroy_model_parallel() and makes the next inner
# trainer's colocate re-init hang. Patch the LLM symbol inside TRL's module
# namespace (TRL did `from vllm import LLM` at import time, so patching
# `vllm.LLM` here would be a no-op for TRL). _vllm_generate_once below passes
# enforce_eager=True directly for its own rank-0 engine; this covers TRL's
# colocate path.
from trl.generation import vllm_generation as _trl_vllm_gen

_trl_vllm_LLM_orig = _trl_vllm_gen.LLM

def _trl_vllm_LLM_eager(*args, **kwargs):
    kwargs.setdefault("enforce_eager", True)
    return _trl_vllm_LLM_orig(*args, **kwargs)

_trl_vllm_gen.LLM = _trl_vllm_LLM_eager

from soar.config import SoarConfig, smoke_config
from soar.data import MathProblem, load_fail128_split
from soar.models import student_chat_prompt
from soar.rloo import _per_token_logp, rloo_advantages, tokenize_rollouts
from soar.teacher import parse_teacher_output
from soar.verify import score_math_answer, success_math_answer


# ------------- teacher + student prompts (verbatim from SOAR paper, Appendix B.1) -------------
# Inlined here so the full generation surface — prompt text + chat wrapping + vLLM
# call — reads top-to-bottom in this file.

TEACHER_PROMPT = (
    "You are generating a new math problem for a math assistant.\n"
    "\n"
    "Allowed topics: Algebra, Counting and Probability, Geometry, Intermediate Algebra, "
    "Number Theory, Prealgebra, or Precalculus.\n"
    "\n"
    "\n"
    "Output rules (follow EXACTLY):\n"
    "    - Provide the final formatted problem in this structure: "
    "<question>[full math question]</question><answer>\\boxed{[answer]}</answer>\n"
    "    - Any explanations, steps, or reasoning about the problem goes OUTSIDE the "
    "<question> and <answer> tags.\n"
    "\n"
    "    Constraints:\n"
    "        - The problem must be original, challenging, and require at least 2--3 steps of reasoning.\n"
    "        - Output exactly ONE problem. You MUST follow the specified format EXACTLY.\n"
    "Begin now:"
)

def _chat_prompt(tok, user_msg: str) -> str:
    """Wrap `user_msg` with the model's chat template + assistant-start token.
    Used for the teacher turn (teacher follows the base model's chat format)."""
    return tok.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )


# ------------- accelerator / logging helpers -------------

def _log(accel: Accelerator, *args, **kwargs) -> None:
    if accel.is_main_process:
        print(*args, **kwargs, flush=True)


def _is_fsdp(accel: Accelerator) -> bool:
    return accel.distributed_type == DistributedType.FSDP


def _broadcast_object(obj, accel: Accelerator, src: int = 0):
    """Broadcast a pickle-able Python object from `src` to all ranks."""
    if accel.num_processes == 1:
        return obj
    box = [obj if accel.process_index == src else None]
    broadcast_object_list(box, from_process=src)
    return box[0]


# ------------- model / tokenizer builders -------------

def _build_model(cfg: SoarConfig) -> AutoModelForCausalLM:
    """Plain HF model in bf16, no device_map (accelerate / FSDP handles placement)."""
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    return model


def _build_tokenizer(cfg: SoarConfig):
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


# ------------- disk helpers -------------
def _rmtree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _replace_dir(src: Path, dst: Path) -> None:
    _rmtree(dst)
    shutil.copytree(src, dst)


def _append_promotion_questions(out_dir: str, promotion_idx: int,
                                X_kstar: List[MathProblem]) -> None:
    path = Path(out_dir) / "promotion_questions.jsonl"
    with open(path, "a") as f:
        for p in X_kstar:
            d = asdict(p)
            d["promotion_idx"] = promotion_idx
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def _save_teacher_outer_data(
    out_dir: str, outer_step: int,
    X: List[List[MathProblem]], R_k: List[float],
    promoted_k_star: Optional[int],
) -> None:
    """One jsonl per group under teacher_data/outer_{step:04d}/.

    Filename encodes group reward and promotion status so `ls` is enough
    to see which splits were strongest. Overwrite mode so re-running an
    outer step produces a clean file.
    """
    step_dir = Path(out_dir) / "teacher_data" / f"outer_{outer_step:04d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    for k, problems in enumerate(X):
        suffix = "_PROMOTED" if promoted_k_star == k else ""
        fname = f"k{k}_R{R_k[k]:+.4f}{suffix}.jsonl"
        with open(step_dir / fname, "w") as f:
            for p in problems:
                d = asdict(p)
                d["outer_step"] = outer_step
                d["k"] = k
                d["R_k"] = float(R_k[k])
                f.write(json.dumps(d, ensure_ascii=False) + "\n")


# ------------- memory diagnostics (opt-in: SOAR_MEM_DEBUG=1) -------------
#
# Why: FSDP + colocate vLLM + round-trip model loads is a leaky stack; this
# instrumentation is opt-in so prod runs pay nothing. Setting SOAR_MEM_DEBUG=1
# records allocator history and logs per-phase deltas; SOAR_MEM_TOP_TENSORS=N
# dumps the N largest live tensors; SOAR_MEM_SNAPSHOT_EVERY=K writes a pickle
# allocator snapshot every K outer steps.
_MEM_DEBUG = os.environ.get("SOAR_MEM_DEBUG", "0") == "1"
_MEM_TOP_TENSORS = int(os.environ.get("SOAR_MEM_TOP_TENSORS", "0"))
_MEM_SNAPSHOT_EVERY = int(os.environ.get("SOAR_MEM_SNAPSHOT_EVERY", "0"))
_GIB = 1024 ** 3


def _mem_bytes() -> Tuple[int, int]:
    if not torch.cuda.is_available():
        return 0, 0
    return (torch.cuda.memory_allocated(), torch.cuda.memory_reserved())


def _mem_log(tag: str, *, baseline: Optional[Tuple[int, int]] = None,
             prev: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    a, r = _mem_bytes()
    parts = [f"[mem] {tag}: alloc={a/_GIB:.3f}GiB reserved={r/_GIB:.3f}GiB"]
    if prev is not None:
        parts.append(f"Δprev alloc={(a - prev[0])/_GIB:+.3f}GiB "
                     f"reserved={(r - prev[1])/_GIB:+.3f}GiB")
    if baseline is not None:
        parts.append(f"Δbase alloc={(a - baseline[0])/_GIB:+.3f}GiB "
                     f"reserved={(r - baseline[1])/_GIB:+.3f}GiB")
    print("  ".join(parts), flush=True)
    return a, r


def _mem_top_tensors(n: int) -> None:
    """Print the N largest cuda tensors in gc.get_objects (approximate; expensive)."""
    import weakref
    seen = set()
    rows = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                key = (obj.data_ptr(), tuple(obj.shape), str(obj.dtype))
                if key in seen:
                    continue
                seen.add(key)
                rows.append((obj.numel() * obj.element_size(),
                             tuple(obj.shape), str(obj.dtype),
                             weakref.ref(obj)))
        except Exception:
            pass
    rows.sort(key=lambda r: -r[0])
    print(f"  [mem] top-{n} cuda tensors:", flush=True)
    for sz, shape, dt, _ in rows[:n]:
        print(f"    {sz/1024**2:9.2f}MiB  {shape}  {dt}", flush=True)


def _mem_dump_snapshot(out_dir: Path, tag: str) -> None:
    try:
        snap_path = out_dir / f"cuda_mem_{tag}.pickle"
        torch.cuda.memory._dump_snapshot(str(snap_path))
        print(f"  [mem] snapshot → {snap_path}", flush=True)
    except Exception as e:
        print(f"  [mem] snapshot failed: {e}", flush=True)


# ------------- FSDP full-state-dict I/O (compatible with HF layout + vLLM) -------------
def _load_safetensors_merged(path: Path, device: str = "cpu") -> dict:
    shards = sorted(path.glob("model-*.safetensors"))
    if shards:
        sd: dict = {}
        for shard in shards:
            sd.update(_load_safetensors(str(shard), device=device))
        return sd
    return _load_safetensors(str(path / "model.safetensors"), device=device)


def _save_hf_from_fsdp(accel: Accelerator, model, tok, path: Path) -> None:
    """Save an FSDP-wrapped (or plain) model to `path` in HF layout (vLLM-readable).

    Under FSDP: enter FULL_STATE_DICT ctx on all ranks (collective); only rank 0
    sees the full gathered state_dict (offloaded to CPU to avoid rank-0 OOM) and
    writes files.
    """
    path.mkdir(parents=True, exist_ok=True)
    unwrapped = accel.unwrap_model(model)
    if _is_fsdp(accel):
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            FullStateDictConfig,
            StateDictType,
        )
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            sd = model.state_dict()
            if accel.is_main_process:
                unwrapped.save_pretrained(str(path), state_dict=sd,
                                          safe_serialization=True)
                tok.save_pretrained(str(path))
    else:
        unwrapped.save_pretrained(str(path), safe_serialization=True)
        tok.save_pretrained(str(path))
    accel.wait_for_everyone()


def _load_hf_into_fsdp(accel: Accelerator, model, path: Path) -> None:
    """Load a full-layout HF checkpoint from `path` into an already-wrapped model."""
    if _is_fsdp(accel):
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            FullStateDictConfig,
            StateDictType,
        )
        sd = _load_safetensors_merged(path, device="cpu")
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
        ):
            model.load_state_dict(sd)
        del sd
    else:
        unwrapped = accel.unwrap_model(model)
        sd = _load_safetensors_merged(path, device="cpu")
        unwrapped.load_state_dict(sd)
        del sd
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------- vLLM lifecycle (rank-0 only, strict teardown) -------------

_VLLM_STASH_KEYS = (
    "RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT",
    "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_NAME",
    "TORCHELASTIC_RUN_ID", "TORCHELASTIC_USE_AGENT_STORE",
    "TORCHELASTIC_MAX_RESTARTS", "TORCHELASTIC_RESTART_COUNT",
)


def _vllm_generate_once(
    cfg: SoarConfig, ckpt_dir: Path, prompts: List[str], *,
    max_new_tokens: int, temperature: float, top_p: float = 1.0,
) -> List[str]:
    """Spin up vLLM → generate → destroy → empty cache. Rank 0 only; caller gates."""
    from vllm import LLM, SamplingParams  # local import so other ranks don't pay import cost
    stash = {k: os.environ.pop(k) for k in _VLLM_STASH_KEYS if k in os.environ}
    # Let the prior TRL/vLLM teardown fully quiesce before vLLM measures free
    # memory — otherwise its V1 profiling asserts on a tiny positive delta.
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    try:
        engine = LLM(
            model=str(ckpt_dir),
            dtype="bfloat16",
            gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
            max_model_len=cfg.vllm_max_model_len,
            enforce_eager=True,
            disable_log_stats=True,
        )
    finally:
        os.environ.update(stash)
    sp = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)
    outputs = engine.generate(prompts, sp, use_tqdm=False)
    texts = [o.outputs[0].text for o in outputs]
    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return texts


def _eval_via_trainer_llm(cfg: SoarConfig, trainer, tok,
                          questions: List[MathProblem],
                          accel: Accelerator) -> float:
    """Greedy-score `questions` using the live colocate vLLM.

    Each rank generates its own shard (round-robin by index) and we all-reduce
    (correct, total) so the returned acc is identical on every rank. Avoids
    booting a fresh ad-hoc rank-0 vLLM between TRL colocate instances — that
    pattern leaves partial TP state and deadlocks the next trainer's init.

    TRL 1.16 colocate mode uses vllm_tensor_parallel_size=1: each rank's
    `trainer.vllm_generation.llm` is independent, so manual sharding is correct.
    """
    from vllm import SamplingParams

    local_qs = questions[accel.process_index::accel.num_processes]
    local_prompts = [student_chat_prompt(tok, q.problem) for q in local_qs]
    sp = SamplingParams(temperature=0.0, top_p=1.0,
                        max_tokens=cfg.student_max_new_tokens)
    if local_prompts:
        outputs = trainer.vllm_generation.llm.generate(
            local_prompts, sp, use_tqdm=False,
        )
        local_correct = sum(
            1.0 for o, q in zip(outputs, local_qs)
            if success_math_answer(o.outputs[0].text, q.answer)
        )
        local_total = float(len(local_qs))
    else:
        local_correct, local_total = 0.0, 0.0

    stats = torch.tensor([local_correct, local_total], device=accel.device)
    stats = accel.reduce(stats, reduction="sum")
    total = float(stats[1].item())
    return float(stats[0].item() / total) if total > 0 else 0.0


def _generate_teacher_batch_vllm(
    cfg: SoarConfig, ckpt_dir: Path, teacher_prompt_str: str, needed: int,
    accel: Accelerator,
) -> Tuple[List[str], List[MathProblem], int]:
    """Rank 0 loops teacher vLLM gen until exactly `needed` parseable problems are
    accepted; broadcasts the accepted tuple to all ranks.

    Each round requests `needed - len(accepted)` fresh completions, parses them,
    appends accepted ones. Unbounded loop — if the teacher's yield collapses the
    caller sees it in the per-round log line. `cfg.teacher_rejection_cap` is
    unused here (train.py's older PEFT loop still references it).

    `top_p=1.0` is not specified in the paper; chosen as the neutral default.
    """
    if accel.is_main_process:
        accepted_raw: List[str] = []
        accepted_problems: List[MathProblem] = []
        total_generated = 0
        round_idx = 0
        while len(accepted_problems) < needed:
            remaining = (needed - len(accepted_problems)) * 3
            # adding <question> because the yield for parsable questions was very low without it.
            raw = _vllm_generate_once(
                cfg, ckpt_dir, [teacher_prompt_str + "<question>"] * remaining,
                max_new_tokens=cfg.teacher_max_new_tokens,
                temperature=cfg.teacher_temperature, top_p=1.0,
            )
            total_generated += len(raw)
            for r in raw:
                p = parse_teacher_output("<question>" + r)
                if p is not None:
                    accepted_raw.append(r)
                    accepted_problems.append(p)
                    if len(accepted_problems) >= needed:
                        break
            round_idx += 1
            print(f"  [teacher-gen round {round_idx}] "
                  f"total_generated={total_generated} "
                  f"accepted={len(accepted_problems)}/{needed} "
                  f"cum_yield={len(accepted_problems) / max(total_generated, 1):.2%}",
                  flush=True)
        payload = (accepted_raw, accepted_problems, total_generated)
    else:
        payload = None
    accel.wait_for_everyone()
    return _broadcast_object(payload, accel, src=0)


# ------------- inner loop: TRL RLOO / GRPO (full-param, FSDP via TRL/accelerate) -------------

def _run_inner_rl(cfg: SoarConfig, tok, X_k: List[MathProblem], tag: str,
                  baseline_ckpt_dir: Path, out_dir: str,
                  accel: Accelerator,
                  *, inner_steps: Optional[int] = None,
                  eval_questions: Optional[List[MathProblem]] = None,
                  eval_baseline: bool = False,
                  ) -> Tuple[dict, Path, Optional[float], Optional[float]]:
    """One SOAR inner training run on X_k via TRL RLOOTrainer / GRPOTrainer.

    Trainer picked by `cfg.rl_algo` ∈ {"rloo","grpo","grpo_no_norm"}. Builds a fresh
    student_model from `baseline_ckpt_dir` so TRL can deepcopy it into its own
    ref_model, then overwrites ref_model weights with the same baseline (so KL
    tracks the live baseline, not the pretrained base).

    `inner_steps` overrides `cfg.inner_steps` when provided (caller passes the
    promotion-bumped count — paper B.7's "+5 per promotion").

    When `eval_questions` is given, scores Q_R using the trainer's colocate
    vLLM (via `_eval_via_trainer_llm`) — before training if `eval_baseline`,
    and always after training (post `vllm_generation.sync_weights()`). Returns
    (stats, trained_dir, base_acc_or_None, trained_acc_or_None).
    """
    assert cfg.rl_algo in ("rloo", "grpo", "grpo_no_norm"), (
        f"bad rl_algo={cfg.rl_algo!r}"
    )

    inner_mem_entry = _mem_bytes() if _MEM_DEBUG else None

    def log_mem(stage: str) -> None:
        if _MEM_DEBUG:
            _mem_log(f"{tag}/{stage}", baseline=inner_mem_entry)

    n_steps = inner_steps if inner_steps is not None else cfg.inner_steps
    ds = datasets.Dataset.from_list([
        {
            "prompt": student_chat_prompt(tok, p.problem),
            "answer": p.answer,
        }
        for p in X_k
    ])
    counters = {"n_rollouts": 0, "n_correct": 0, "reward_sum": 0.0}

    def reward_fn(completions, answer, **_):
        scores = [float(score_math_answer(c, a)) for c, a in zip(completions, answer)]
        counters["n_rollouts"] += len(scores)
        counters["n_correct"] += sum(1 for s in scores if s >= 120.0)
        counters["reward_sum"] += sum(scores)
        return scores

    student_model = AutoModelForCausalLM.from_pretrained(
        str(baseline_ckpt_dir), torch_dtype=torch.bfloat16,
    )
    student_model.config.use_cache = False

    # Paper (SOAR §4): batch=8 problems × group=32 completions per optimizer step.
    # TRL constraint: (per_device_train_batch_size * num_processes) must be
    # divisible by num_generations so each micro-batch holds whole groups.
    # Pick the smallest pdt that satisfies the constraint: pdt = G / gcd(G, N).
    # Everything else flows into gradient_accumulation_steps.
    G = cfg.student_group_size
    B = cfg.student_batch_problems
    N = accel.num_processes
    pdt = G // math.gcd(G, N)
    assert (pdt * N) % G == 0  # by construction
    total_sequences_per_step = B * G
    denom = pdt * N
    if total_sequences_per_step % denom != 0:
        raise ValueError(
            f"Cannot hit batch={B} x group={G} = {total_sequences_per_step} sequences/step "
            f"with per_device_train_batch_size={pdt} x num_processes={N} = {denom}. "
            f"Pick a GPU count where (B*G) is divisible by (pdt*N)."
        )
    grad_accum = total_sequences_per_step // denom
    _log(accel, f"  [{cfg.rl_algo}] B={B} G={G} N={N} pdt={pdt} grad_accum={grad_accum} "
                f"(prompts/step={B}, seqs/step={B*G})")

    # TRL 1.16 dropped `max_prompt_length` from both RLOOConfig and GRPOConfig;
    # prompt truncation is now handled implicitly by `vllm_max_model_length`
    # (renamed from `vllm_max_model_len`). RLOO's `normalize_advantages=False`
    # (TRL default) and GRPO's `scale_rewards=False` both keep the raw
    # leave-one-out / group-relative advantages the SOAR paper uses.
    # TRL (grpo/rloo _trainer.py) only forces device_map=None for distributed_type
    # in {MULTI_GPU, DEEPSPEED} when auto-building ref_model via
    # create_model_from_path; for FSDP it lets the helper default device_map="auto",
    # which shards ref_model across every visible GPU in the rank's process and
    # makes FSDP wrap fail with "Inconsistent compute device cuda:1 vs cuda:0".
    common_kwargs = dict(
        output_dir=str(Path(out_dir) / "_trl" / tag),
        num_generations=G,
        beta=cfg.kl_beta,
        model_init_kwargs={"device_map": None},
        learning_rate=cfg.student_lr,
        lr_scheduler_type="cosine",
        warmup_steps=0,
        per_device_train_batch_size=pdt,
        gradient_accumulation_steps=grad_accum,
        max_steps=n_steps,
        max_completion_length=cfg.student_max_new_tokens,
        temperature=cfg.student_temperature,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        vllm_max_model_length=cfg.max_prompt_len + cfg.student_max_new_tokens,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        seed=cfg.seed,
        disable_tqdm=True,
    )
    if cfg.rl_algo in ("grpo", "grpo_no_norm"):
        args = GRPOConfig(
            scale_rewards=(cfg.rl_algo == "grpo"),
            **common_kwargs,
        )
        TrainerCls = GRPOTrainer
    else:
        args = RLOOConfig(**common_kwargs)
        TrainerCls = RLOOTrainer

    log_mem("before trainer construction")
    trainer = TrainerCls(
        model=student_model,
        reward_funcs=reward_fn,
        args=args,
        train_dataset=ds,
        processing_class=tok,
    )
    # Full-param: TRL deepcopied student_model into trainer.ref_model and (if FSDP
    # configured) accelerate-prepared it. Overwrite its weights with the current
    # baseline so KL anchors to the live baseline (which may have been promoted
    # away from pretrained).
    if trainer.ref_model is not None:
        _load_hf_into_fsdp(accel, trainer.ref_model, baseline_ckpt_dir)

    # Baseline eval: colocate engine was initialised from baseline weights and
    # still holds them (no training step has run yet). Reuses the same engine
    # that will host the trained eval → no extra engine lifecycle.
    base_acc: Optional[float] = None
    if eval_baseline and eval_questions is not None:
        base_acc = _eval_via_trainer_llm(cfg, trainer, tok, eval_questions, accel)

    trainer.train()
    log_mem("after training")

    # Post-training eval: TRL only syncs model→vLLM before each rollout, so after
    # the final gradient step the colocate engine lags by one step. Force a sync
    # via `vllm_generation.sync_weights()` (TRL 1.16 replaces 0.19's
    # `_move_model_to_vllm`) so trained_acc reflects the saved weights.
    trained_acc: Optional[float] = None
    if eval_questions is not None:
        gen = getattr(trainer, "vllm_generation", None)
        if gen is not None and getattr(gen, "llm", None) is not None:
            gen.sync_weights()
        trained_acc = _eval_via_trainer_llm(cfg, trainer, tok, eval_questions, accel)

    n = max(counters["n_rollouts"], 1)
    stats = {
        "n_rollouts": counters["n_rollouts"],
        "n_correct": counters["n_correct"],
        "mean_reward_run": counters["reward_sum"] / n,
    }

    # Save the trained student (FSDP full-state-dict gather) to a tag-specific dir.
    trained_dir = Path(out_dir) / "_scratch" / f"trained_{tag}"
    model_for_save = (trainer.model_wrapped
                      if getattr(trainer, "model_wrapped", None) is not None
                      else trainer.model)
    _save_hf_from_fsdp(accel, model_for_save, tok, trained_dir)
    log_mem("after save")

    # TRL 1.16 colocate engine lives under `trainer.vllm_generation.llm`. Full
    # teardown is required: LLMEngine.__del__ only clears the DP group, leaving
    # _TP/_PP + captured cudagraphs pinned. Walk down to engine_core and shut
    # it down explicitly before destroy_model_parallel; otherwise the next
    # trainer's NCCL init deadlocks on stale group state.
    _gen = getattr(trainer, "vllm_generation", None)
    if _gen is not None and getattr(_gen, "llm", None) is not None:
        llm = _gen.llm
        engine = llm.llm_engine
        core_client = getattr(engine, "engine_core", None)
        if core_client is None or not callable(
            getattr(core_client, "shutdown", None)
        ):
            raise RuntimeError(
                "Unexpected vLLM engine layout; "
                "inspect your installed vLLM version before teardown."
            )
        core_client.shutdown()
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
        _gen.llm = None
        del core_client, engine, llm

    trainer.model.zero_grad(set_to_none=True)
    if trainer.accelerator is not accel:
        trainer.accelerator.free_memory()

    del trainer, student_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    log_mem("after cleanup")

    return stats, trained_dir, base_acc, trained_acc


# ------------- teacher outer update (manual accel-backed RLOO with explicit frozen ref) -------------

def _rloo_loss_with_ref(model, ref_model, batch, advantages, kl_beta) -> dict:
    """RLOO policy-gradient loss + KL to an explicit frozen `ref_model`.

    Non-PEFT twin of soar.rloo.rloo_loss (which gets its ref via
    peft_model.disable_adapter()).
    """
    assert advantages.ndim == 1 and advantages.numel() == batch.input_ids.shape[0]

    logp_new = _per_token_logp(model, batch.input_ids, batch.attention_mask)
    with torch.no_grad():
        logp_ref = _per_token_logp(ref_model, batch.input_ids, batch.attention_mask)

    m = batch.completion_mask
    denom = m.sum(dim=-1).clamp(min=1.0)
    mean_logp_new = (logp_new * m).sum(dim=-1) / denom
    mean_logp_ref = (logp_ref * m).sum(dim=-1) / denom
    kl = mean_logp_new - mean_logp_ref

    pg_loss = -(advantages * mean_logp_new).mean()
    kl_term = kl.mean()
    loss = pg_loss + kl_beta * kl_term
    return {
        "loss": loss,
        "pg_loss": pg_loss.detach(),
        "kl": kl_term.detach(),
        "adv_abs_mean": advantages.abs().mean().detach(),
    }


def _shard_indices(total: int, accel: Accelerator) -> List[int]:
    """Strided row indices for this rank, [rank::world_size]. Correctness fallback:
    if total < world_size, every rank runs all rows."""
    ws = accel.num_processes
    if total < ws:
        return list(range(total))
    r = accel.process_index
    return list(range(r, total, ws))


def _teacher_rloo_step(
    teacher_model, teacher_ref, tok, cfg: SoarConfig,
    teacher_prompt_str: str, accepted_raw: List[str],
    R_k: List[float], teacher_opt: torch.optim.Optimizer, n_per: int,
    accel: Accelerator,
) -> Tuple[dict, float]:
    """One teacher RLOO step. Returns (metrics, R_spread). Skips if g<2 or spread ≤ 1e-6.

    NOTE on Table 3 (paper) teacher batch=2, group=4: `group=4` matches
    `cfg.outer_group_size` (datasets per LOO group). `batch=2` is likely outer
    gradient accumulation — the teacher updates every 2 outer steps instead of
    every one — but this is NOT implemented here; the current code does one
    teacher gradient step per outer iteration using all g*n completions.
    """
    g = len(R_k)
    needed = g * n_per
    dev = accel.device

    R_k_t = torch.tensor(R_k, dtype=torch.float32, device=dev)
    R_spread = (R_k_t.max() - R_k_t.min()).item() if g >= 2 else 0.0
    if g < 2 or R_spread <= 1e-6:
        return {"teacher/skipped": 1.0}, R_spread

    if cfg.rl_algo == "grpo":
        std = R_k_t.std(unbiased=False)
        dataset_adv = (R_k_t - R_k_t.mean()) / (std + 1e-8)    # [g]
    elif cfg.rl_algo == "grpo_no_norm":
        dataset_adv = R_k_t - R_k_t.mean()                     # [g]
    else:  # rloo
        dataset_adv = rloo_advantages(R_k_t, group_size=g)     # [g]
    per_rollout_adv = dataset_adv.repeat_interleave(n_per)     # [g*n]
    batch = tokenize_rollouts(
        tok, [teacher_prompt_str + "<question>"] * needed, accepted_raw, device=dev.type,
        max_prompt_len=cfg.max_prompt_len,
        max_total_len=cfg.max_prompt_len + cfg.teacher_max_new_tokens,
    )

    local_idx = _shard_indices(needed, accel)
    teacher_model.train()
    agg = {"loss": 0.0, "pg_loss": 0.0, "kl": 0.0, "adv_abs_mean": 0.0}
    for i in local_idx:
        micro = type(batch)(
            input_ids=batch.input_ids[i:i + 1],
            attention_mask=batch.attention_mask[i:i + 1],
            completion_mask=batch.completion_mask[i:i + 1],
            rewards=batch.rewards,
        )
        out = _rloo_loss_with_ref(teacher_model, teacher_ref, micro,
                                  advantages=per_rollout_adv[i:i + 1],
                                  kl_beta=cfg.kl_beta)
        accel.backward(out["loss"] / needed)
        for k in agg:
            v = out[k]
            agg[k] += float(v.detach() if isinstance(v, torch.Tensor) else v)
    denom = max(len(local_idx), 1)
    for k in agg:
        agg[k] /= denom

    grad_norm = accel.clip_grad_norm_(teacher_model.parameters(), 1.0)
    teacher_opt.step()
    teacher_opt.zero_grad(set_to_none=True)

    return {
        "teacher/loss": agg["loss"],
        "teacher/pg_loss": agg["pg_loss"],
        "teacher/kl": agg["kl"],
        "teacher/adv_abs_mean": agg["adv_abs_mean"],
        "teacher/grad_norm": float(grad_norm),
        "teacher/adv_min": float(dataset_adv.min()),
        "teacher/adv_max": float(dataset_adv.max()),
    }, R_spread


# ------------- main outer loop -------------

def main(cfg: SoarConfig) -> None:
    accel = Accelerator()
    random.seed(cfg.seed + accel.process_index)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    if accel.is_main_process:
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(cfg.out_dir) / "config.json", "w") as f:
            json.dump({k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in cfg.__dict__.items()}, f, indent=2)
    accel.wait_for_everyone()

    _log(accel, f"[setup] world_size={accel.num_processes} "
                f"distributed_type={accel.distributed_type.name}  "
                f"base_model={cfg.base_model}")

    # Persistent teacher: FSDP-wrapped with its optimizer.
    teacher_model = _build_model(cfg)
    # use_reentrant=False is required for FSDP + gradient checkpointing.
    teacher_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    teacher_opt = torch.optim.AdamW(teacher_model.parameters(), lr=cfg.teacher_lr)
    teacher_model, teacher_opt = accel.prepare(teacher_model, teacher_opt)
    tok = _build_tokenizer(cfg)

    _log(accel, f"[setup] loading fail@128 split from {cfg.fail128_dir}")
    train_set, _ = load_fail128_split(cfg.fail128_dir)
    _log(accel, f"[setup] |train|={len(train_set)}")
    assert len(train_set) >= cfg.n_reward_questions, "not enough train problems for Q_R"

    # Seed both role-dirs on disk from the pretrained teacher weights (vLLM reads these).
    scratch_dir = Path(cfg.out_dir) / "_scratch"
    teacher_ckpt = scratch_dir / "teacher"
    baseline_student_ckpt = scratch_dir / "baseline_student"
    teacher_latest_dir = Path(cfg.out_dir) / "teacher_latest"
    if accel.is_main_process:
        scratch_dir.mkdir(parents=True, exist_ok=True)
    accel.wait_for_everyone()
    _save_hf_from_fsdp(accel, teacher_model, tok, teacher_ckpt)
    if accel.is_main_process:
        _replace_dir(teacher_ckpt, baseline_student_ckpt)
    accel.wait_for_everyone()

    if accel.is_main_process:
        wandb.init(project=cfg.wandb_project, mode=cfg.wandb_mode, config=cfg.__dict__)

    promotions_window: deque = deque(maxlen=cfg.promotion_window)
    n_promotions = 0
    g, n_per = cfg.outer_group_size, cfg.inner_dataset_size
    r_parallel = cfg.r_parallel_students
    needed_teacher = g * n_per

    if _MEM_DEBUG and torch.cuda.is_available():
        try:
            torch.cuda.memory._record_memory_history(max_entries=200_000)
            _log(accel, "[mem] recording cuda memory history (SOAR_MEM_DEBUG=1)")
        except Exception as e:
            _log(accel, f"[mem] could not enable history: {e}")
    mem_baseline: Optional[Tuple[int, int]] = None
    mem_prev_step: Optional[Tuple[int, int]] = None

    for outer_step in range(cfg.n_outer_steps):
        t_start = time.time()
        _log(accel, f"\n========== OUTER STEP {outer_step}/{cfg.n_outer_steps} ==========")
        mem_step_start = None
        if _MEM_DEBUG:
            mem_step_start = _mem_log(
                f"outer {outer_step} START",
                baseline=mem_baseline, prev=mem_prev_step,
            )
            if mem_baseline is None:
                mem_baseline = mem_step_start
        trained_ckpts: List[List[Optional[Path]]] = [
            [None] * r_parallel for _ in range(g)
        ]

        try:
            # ----- 1. TEACHER GENERATES g*n parsed problems (vLLM, rank 0 only) -----
            t_phase = time.time()
            teacher_prompt_str = _chat_prompt(tok, TEACHER_PROMPT)
            accepted_raw, accepted_problems, total_generated = _generate_teacher_batch_vllm(
                cfg, teacher_ckpt, teacher_prompt_str, needed_teacher, accel,
            )
            teacher_gen_sec = time.time() - t_phase
            teacher_yield = len(accepted_problems) / max(total_generated, 1)
            if len(accepted_problems) < needed_teacher:
                _log(accel, f"[step {outer_step}] teacher yield too low "
                            f"({len(accepted_problems)}/{needed_teacher}) — skipping")
                if accel.is_main_process:
                    wandb.log({
                        "outer_step": outer_step, "skipped": 1,
                        "teacher/yield": teacher_yield,
                        "teacher/total_generated": total_generated,
                        "timing/teacher_gen_sec": teacher_gen_sec,
                    }, step=outer_step)
                continue
            X = [accepted_problems[k * n_per:(k + 1) * n_per] for k in range(g)]

            # ----- 2. SAMPLE Q_R (baseline eval runs inside first inner trainer) -----
            if accel.is_main_process:
                Q_R = rng.sample(train_set, cfg.n_reward_questions)
            else:
                Q_R = None
            Q_R = _broadcast_object(Q_R, accel, src=0)
            baseline_eval_sec = 0.0  # kept for log schema; baseline eval cost now folded into first inner run

            # ----- 3. INNER TRAINING: g datasets × r parallel runs -----
            t_phase = time.time()
            total_inner = g * r_parallel
            R_matrix: List[List[float]] = [[0.0] * r_parallel for _ in range(g)]
            outer_n_rollouts = 0
            outer_n_correct = 0
            outer_reward_sum = 0.0
            run_idx = 0
            base_acc: Optional[float] = None
            for k in range(g):
                for j in range(r_parallel):
                    run_idx += 1
                    t_run = time.time()
                    tag = f"o{outer_step}_k{k}_j{j}"
                    effective_inner_steps = min(
                        cfg.inner_steps + cfg.inner_steps_per_promotion * n_promotions,
                        30,
                    )
                    inner_stats, trained_dir, base_acc_run, trained_acc = _run_inner_rl(
                        cfg, tok, X[k], tag, baseline_student_ckpt, cfg.out_dir, accel,
                        inner_steps=effective_inner_steps,
                        eval_questions=Q_R,
                        eval_baseline=(run_idx == 1),
                    )
                    if base_acc_run is not None:
                        base_acc = base_acc_run
                        _log(accel, f"[step {outer_step}] baseline acc={base_acc:.3f}")
                    trained_ckpts[k][j] = trained_dir
                    outer_n_rollouts += int(inner_stats["n_rollouts"])
                    outer_n_correct += int(inner_stats["n_correct"])
                    outer_reward_sum += (float(inner_stats["mean_reward_run"])
                                         * int(inner_stats["n_rollouts"]))
                    R_matrix[k][j] = trained_acc - base_acc
                    _log(accel, f"[step {outer_step}] ({run_idx}/{total_inner}) "
                                f"X_{k} r{j}: acc={trained_acc:.3f} "
                                f"Δ={trained_acc - base_acc:+.3f} "
                                f"({time.time() - t_run:.1f}s)")

            inner_train_sec = time.time() - t_phase
            R_k = [float(mean(row)) for row in R_matrix]
            R_bar_t = float(mean(R_k))
            R_flat = [r for row in R_matrix for r in row]
            R_std = float(torch.tensor(R_flat).std().item()) if len(R_flat) > 1 else 0.0

            # ----- 4. PROMOTION CHECK -----
            promoted_this_step = 0
            k_star: Optional[int] = None
            promotions_window.append(R_bar_t)
            ma = (float(mean(promotions_window))
                  if len(promotions_window) == cfg.promotion_window
                  else -float("inf"))
            if ma > cfg.promotion_threshold:
                k_star = int(max(range(g), key=lambda k: R_k[k]))
                rewards_kstar = R_matrix[k_star]
                sorted_j = sorted(range(r_parallel), key=lambda j: rewards_kstar[j])
                j_star = sorted_j[r_parallel // 2]
                n_promotions += 1
                if accel.is_main_process:
                    _replace_dir(trained_ckpts[k_star][j_star], baseline_student_ckpt)
                    _append_promotion_questions(cfg.out_dir, n_promotions, X[k_star])
                    promo_dir = Path(cfg.out_dir) / "promotions" / f"promo_{n_promotions:03d}"
                    promo_dir.mkdir(parents=True, exist_ok=True)
                    _replace_dir(baseline_student_ckpt, promo_dir / "student")
                    with open(Path(cfg.out_dir) / "promotions" / "index.jsonl", "a") as f:
                        f.write(json.dumps({
                            "promotion_idx": n_promotions, "outer_step": outer_step,
                            "k_star": k_star, "j_star": j_star,
                            "R_bar_t": R_bar_t, "moving_avg": ma,
                        }) + "\n")
                accel.wait_for_everyone()
                promotions_window.clear()
                promoted_this_step = 1
                _log(accel, f"[step {outer_step}] PROMOTED "
                            f"(k*={k_star}, j*={j_star}, MA={ma:+.3f})")

            if accel.is_main_process:
                _save_teacher_outer_data(
                    cfg.out_dir, outer_step, X, R_k,
                    promoted_k_star=(k_star if promoted_this_step else None),
                )

            # ----- 5. TEACHER RLOO UPDATE -----
            t_phase = time.time()
            teacher_ref = _build_model(cfg)
            teacher_ref = accel.prepare(teacher_ref)
            _load_hf_into_fsdp(accel, teacher_ref, teacher_ckpt)
            teacher_ref.eval()
            for p in teacher_ref.parameters():
                p.requires_grad_(False)
            teacher_metrics, R_spread = _teacher_rloo_step(
                teacher_model, teacher_ref, tok, cfg,
                teacher_prompt_str, accepted_raw, R_k, teacher_opt, n_per, accel,
            )
            del teacher_ref
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Persist updated teacher weights for next outer step's vLLM gen.
            _save_hf_from_fsdp(accel, teacher_model, tok, teacher_ckpt)
            # Mirror to a stable top-level path so external evals always have a
            # current teacher to point at (ckpt_outer{N}/ only fires every ckpt_every).
            if accel.is_main_process:
                _replace_dir(teacher_ckpt, teacher_latest_dir)
            accel.wait_for_everyone()
            teacher_update_sec = time.time() - t_phase

            # ----- 6. LOGGING + CHECKPOINT -----
            dt = time.time() - t_start
            gpu_mem_gb = (torch.cuda.max_memory_allocated() / 2 ** 30
                          if torch.cuda.is_available() else 0.0)
            log = {
                "outer_step": outer_step,
                "baseline_acc": base_acc,
                "teacher/dataset_reward_mean": R_bar_t,
                "teacher/dataset_reward_max": float(max(R_k)),
                "teacher/dataset_reward_min": float(min(R_k)),
                "teacher/dataset_reward_std": R_std,
                "teacher/dataset_reward_spread": R_spread,
                "teacher/yield": teacher_yield,
                "teacher/total_generated": total_generated,
                "student/n_correct": outer_n_correct,
                "student/n_rollouts": outer_n_rollouts,
                "student/correct_rate": outer_n_correct / max(outer_n_rollouts, 1),
                "student/mean_reward": outer_reward_sum / max(outer_n_rollouts, 1),
                "promotion/count_total": n_promotions,
                "promotion/triggered_this_step": promoted_this_step,
                "promotion/moving_avg": ma if ma != -float("inf") else 0.0,
                "promotion/window_fill": len(promotions_window) / cfg.promotion_window,
                "timing/teacher_gen_sec": teacher_gen_sec,
                "timing/baseline_eval_sec": baseline_eval_sec,
                "timing/inner_train_sec": inner_train_sec,
                "timing/teacher_update_sec": teacher_update_sec,
                "timing/step_sec": dt,
                "sys/gpu_max_mem_gb": gpu_mem_gb,
                **{f"R_k/dataset_{k}": R_k[k] for k in range(g)},
                **teacher_metrics,
            }
            if _MEM_DEBUG:
                mem_prev_step = _mem_log(
                    f"outer {outer_step} END",
                    baseline=mem_baseline, prev=mem_step_start,
                )
                a_now, r_now = mem_prev_step
                log["sys/mem_alloc_gib"] = a_now / _GIB
                log["sys/mem_reserved_gib"] = r_now / _GIB
                if mem_step_start is not None:
                    log["sys/mem_alloc_delta_step_gib"] = (
                        (a_now - mem_step_start[0]) / _GIB
                    )
                    log["sys/mem_reserved_delta_step_gib"] = (
                        (r_now - mem_step_start[1]) / _GIB
                    )
                if mem_baseline is not None:
                    log["sys/mem_alloc_delta_baseline_gib"] = (
                        (a_now - mem_baseline[0]) / _GIB
                    )
                    log["sys/mem_reserved_delta_baseline_gib"] = (
                        (r_now - mem_baseline[1]) / _GIB
                    )
                if _MEM_TOP_TENSORS:
                    _mem_top_tensors(_MEM_TOP_TENSORS)
                if (_MEM_SNAPSHOT_EVERY
                        and (outer_step + 1) % _MEM_SNAPSHOT_EVERY == 0
                        and accel.is_main_process):
                    _mem_dump_snapshot(Path(cfg.out_dir), f"outer_{outer_step:04d}")
            if accel.is_main_process:
                wandb.log(log, step=outer_step)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            _log(accel, f"[step {outer_step}] R̄_t={R_bar_t:+.3f} "
                        f"promo={promoted_this_step} dt={dt:.1f}s "
                        f"peak_mem={gpu_mem_gb:.1f}GB")

            if outer_step % cfg.ckpt_every == 0:
                ckpt_dir = Path(cfg.out_dir) / f"ckpt_outer{outer_step}"
                if accel.is_main_process:
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    _replace_dir(teacher_ckpt, ckpt_dir / "teacher")
                    _replace_dir(baseline_student_ckpt, ckpt_dir / "baseline_student")
                accel.wait_for_everyone()

        finally:
            # Delete all per-(k,j) trained dirs so disk doesn't grow unbounded.
            # Promoted copies survive via promotions/promo_* and baseline_student/.
            if accel.is_main_process:
                for row in trained_ckpts:
                    for p in row:
                        if p is not None:
                            _rmtree(p)
            accel.wait_for_everyone()

    if accel.is_main_process:
        wandb.finish()
    _log(accel, f"[done] total promotions = {n_promotions}")


# ------------- CLI (identical flag set to train.py) -------------

def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end sanity run")

    ap.add_argument("--base-model", type=str, default=None)

    ap.add_argument("--fail128-dir", type=str, default=None)
    ap.add_argument("--n-reward-questions", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)

    ap.add_argument("--n-outer-steps", type=int, default=None)
    ap.add_argument("--outer-group-size", type=int, default=None)
    ap.add_argument("--inner-dataset-size", type=int, default=None)
    ap.add_argument("--r-parallel-students", type=int, default=None)
    ap.add_argument("--teacher-lr", type=float, default=None)
    ap.add_argument("--teacher-max-new-tokens", type=int, default=None)
    ap.add_argument("--teacher-temperature", type=float, default=None)
    ap.add_argument("--teacher-rejection-cap", type=int, default=None)

    ap.add_argument("--inner-steps", type=int, default=None)
    ap.add_argument("--student-group-size", type=int, default=None)
    ap.add_argument("--student-batch-problems", type=int, default=None)
    ap.add_argument("--student-lr", type=float, default=None)
    ap.add_argument("--student-max-new-tokens", type=int, default=None)
    ap.add_argument("--student-temperature", type=float, default=None)

    ap.add_argument("--kl-beta", type=float, default=None)
    ap.add_argument("--max-prompt-len", type=int, default=None)
    ap.add_argument("--rl-algo", type=str,
                    choices=("rloo", "grpo", "grpo_no_norm"), default=None)

    ap.add_argument("--promotion-threshold", type=float, default=None)
    ap.add_argument("--promotion-window", type=int, default=None)

    ap.add_argument("--vllm-gpu-memory-utilization", type=float, default=None)
    ap.add_argument("--vllm-max-model-len", type=int, default=None)
    ap.add_argument("--vllm-max-loras", type=int, default=None)

    ap.add_argument("--wandb-project", type=str, default=None)
    ap.add_argument("--wandb-mode", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--ckpt-every", type=int, default=None)
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = smoke_config() if args.smoke else SoarConfig()
    for name, value in vars(args).items():
        if name == "smoke" or value is None:
            continue
        setattr(cfg, name, value)
    main(cfg)
