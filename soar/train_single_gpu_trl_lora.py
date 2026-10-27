"""SOAR training loop — single-GPU + dual-LoRA (teacher + student) + TRL GRPO + colocate vLLM.

Launch:
    accelerate launch --num_processes=1 -m soar.train_single_gpu_trl_lora \
        --out-dir ... --fail128-dir ...

Dual-LoRA variant of train_trl.py:
    - Teacher is a PEFT-wrapped base model. KL reference uses `disable_adapter()`
      (the frozen base), so there is no second teacher_ref model in memory.
    - Student is LoRA-wrapped by TRL via `peft_config=` (same path as
      train_trl.py's `--use-lora`); `cfg.use_lora` is forced True at entry.
    - FSDP code paths are stripped (single-GPU only). Multi-GPU runs should keep
      using train_trl.py.

Six phases per outer step (SOAR Alg. 1), identical schema to train_trl.py:

    1. TEACHER GENERATES g*n parsed problems via a fresh vLLM engine loaded
       from `_scratch/teacher/` (merged teacher weights, see phase 5).
    2. SAMPLE Q_R (baseline eval runs inside the first inner trainer).
    3. INNER TRAINING: for each (X_k, j in 1..r), build a fresh student_model
       from `_scratch/baseline_student/`, wrap in LoRA via TRL, run GRPOTrainer.
       KL ref = base model with LoRA disabled (TRL handles automatically when
       `peft_config` is passed and `ref_model=None`). After training, merge the
       student adapter and save merged HF layout to `_scratch/trained_oS_kK_jJ/`.
    4. PROMOTION CHECK — same rolling-m MA logic as train_trl.py.
    5. TEACHER RLOO UPDATE — manual AdamW step on the teacher LoRA params with
       KL anchored to `teacher_model.disable_adapter()`. After the step, merge
       the teacher adapter into the base, save to `_scratch/teacher/`, and
       unmerge so the LoRA remains for the next step.
    6. LOGGING + CHECKPOINT — same wandb schema as train_trl.py.

Memory layout (single H100, Llama-3.2-3B bf16):
    base model          ~6 GB
    teacher LoRA        ~50 MB
    teacher AdamW state ~200 MB
    TRL colocate vLLM   `cfg.vllm_gpu_memory_utilization` fraction
    phase-1 teacher-gen vLLM: separate ephemeral engine (destroyed after use)
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
from accelerate.utils import broadcast_object_list
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, PeftModel, get_peft_model

from soar.config import SoarConfig, smoke_config
from soar.data import MathProblem, load_fail128_split
from soar.rloo import _per_token_logp, rloo_advantages, tokenize_rollouts
from soar.teacher import parse_teacher_output
from soar.verify import score_math_answer, success_math_answer


# ------------- teacher + student prompts (verbatim from SOAR paper, Appendix B.1) -------------

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

STUDENT_PROMPT_TEMPLATE = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant "
    "solves it. The assistant first shows the complete reasoning process step by step, then "
    "provides the final answer in \\boxed{} format: 'User: [question] Assistant: [detailed "
    "reasoning] The final answer is: \\boxed{[answer]}.\n"
    "User: <QUESTION> Assistant: "
)


def _chat_prompt(tok, user_msg: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _student_prompt(question: str) -> str:
    return STUDENT_PROMPT_TEMPLATE.replace("<QUESTION>", question)


# ------------- accelerator / logging helpers -------------

def _log(accel: Accelerator, *args, **kwargs) -> None:
    if accel.is_main_process:
        print(*args, **kwargs, flush=True)


def _broadcast_object(obj, accel: Accelerator, src: int = 0):
    if accel.num_processes == 1:
        return obj
    box = [obj if accel.process_index == src else None]
    broadcast_object_list(box, from_process=src)
    return box[0]


# ------------- model / tokenizer builders -------------

def _build_teacher_peft(cfg: SoarConfig):
    """Base model wrapped with a single LoRA adapter. Trainable params = LoRA only."""
    base = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.bfloat16)
    base.config.use_cache = False
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=list(cfg.lora_target_modules),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    # PEFT inits LoRA in fp32; cast trainable params to bf16 for consistent dtype.
    for p in model.parameters():
        if p.requires_grad and p.dtype == torch.float32:
            p.data = p.data.to(torch.bfloat16)
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


def _save_merged_teacher(peft_model, tok, path: Path) -> None:
    """Merge teacher LoRA → save clean HF layout → unmerge.

    `merge_adapter()` folds deltas into `.base_layer.weight` in place but keeps
    the LoRA wrapper structure, so `state_dict` still has `.base_layer.*` and
    `.lora_A/B.*` keys that vLLM can't load. Strip those from the dict before
    `save_pretrained`, then unmerge so the teacher LoRA stays trainable.
    """
    path.mkdir(parents=True, exist_ok=True)
    peft_model.merge_adapter()
    try:
        base = peft_model.get_base_model()
        clean_sd = {}
        for k, v in base.state_dict().items():
            if (".lora_A." in k or ".lora_B." in k
                    or ".lora_embedding_A." in k or ".lora_embedding_B." in k
                    or ".lora_magnitude_vector" in k):
                continue
            clean_sd[k.replace(".base_layer.", ".")] = v
        base.save_pretrained(
            str(path), state_dict=clean_sd, safe_serialization=True,
        )
        tok.save_pretrained(str(path))
    finally:
        peft_model.unmerge_adapter()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _save_peft_student_adapter(trainer, trained_dir: Path) -> None:
    """Save the TRL student LoRA adapter to `trained_dir` (no merge)."""
    trained_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(trained_dir))


# ------------- memory diagnostics (opt-in: SOAR_MEM_DEBUG=1) -------------
#
# Why: suspected GPU leak after inner GRPO runs. Env-var gated so prod runs are
# untouched. Enable with SOAR_MEM_DEBUG=1; dump a snapshot every N outer steps
# with SOAR_MEM_SNAPSHOT_EVERY=N (opens in https://pytorch.org/memory_viz).
_MEM_DEBUG = os.environ.get("SOAR_MEM_DEBUG", "0") == "1"
_MEM_SNAPSHOT_EVERY = int(os.environ.get("SOAR_MEM_SNAPSHOT_EVERY", "0") or 0)
_MEM_TOP_TENSORS = int(os.environ.get("SOAR_MEM_TOP_TENSORS", "0") or 0)
_GIB = 1024 ** 3


def _mem_bytes() -> Tuple[int, int]:
    if not torch.cuda.is_available():
        return 0, 0
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()


def _mem_log(stage: str, *,
             baseline: Optional[Tuple[int, int]] = None,
             prev: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    a, r = _mem_bytes()
    parts = [f"[mem] {stage}: alloc={a/_GIB:.3f}GiB reserved={r/_GIB:.3f}GiB"]
    if prev is not None:
        parts.append(f"Δprev=(a{(a-prev[0])/_GIB:+.3f},r{(r-prev[1])/_GIB:+.3f})")
    if baseline is not None:
        parts.append(
            f"Δbase=(a{(a-baseline[0])/_GIB:+.3f},r{(r-baseline[1])/_GIB:+.3f})"
        )
    print(" ".join(parts), flush=True)
    return a, r


def _mem_top_tensors(n: int = 15) -> None:
    seen = {}
    for obj in gc.get_objects():
        try:
            if not torch.is_tensor(obj) or not obj.is_cuda:
                continue
            stor = obj.untyped_storage()
            key = (stor.data_ptr(), stor.nbytes())
            if key[0] == 0 or key in seen:
                continue
            seen[key] = (stor.nbytes(), tuple(obj.shape), str(obj.dtype), str(obj.device))
        except Exception:
            continue
    top = sorted(seen.values(), key=lambda v: v[0], reverse=True)[:n]
    print(f"[mem] top {len(top)} live cuda storages:", flush=True)
    for nbytes, shape, dtype, device in top:
        print(f"  {nbytes/1024**2:9.2f} MiB  shape={shape}  {dtype}  {device}",
              flush=True)


def _mem_dump_snapshot(out_dir: Path, tag: str) -> None:
    snap_dir = out_dir / "mem_snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_path = snap_dir / f"{tag}.pickle"
    try:
        torch.cuda.memory._dump_snapshot(str(snap_path))
        print(f"[mem] dumped snapshot → {snap_path}", flush=True)
    except Exception as e:
        print(f"[mem] snapshot dump failed: {e}", flush=True)


# ------------- vLLM lifecycle (ephemeral engines, strict teardown) -------------

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
    """Spin up vLLM → generate → destroy → empty cache."""
    from vllm import LLM, SamplingParams
    stash = {k: os.environ.pop(k) for k in _VLLM_STASH_KEYS if k in os.environ}
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


def _eval_via_trainer_hf(cfg: SoarConfig, trainer, tok,
                         questions: List[MathProblem]) -> float:
    """Greedy-score `questions` via HF generate on trainer.model (vLLM disabled)."""
    model = trainer.accelerator.unwrap_model(trainer.model)
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    prev_side = tok.padding_side
    tok.padding_side = "left"
    correct = 0.0
    batch_size = max(1, cfg.student_batch_problems)
    try:
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            prompts = [_student_prompt(q.problem) for q in batch]
            inputs = tok(prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=cfg.student_max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_id,
                )
            new_tokens = out[:, inputs["input_ids"].shape[1]:]
            texts = tok.batch_decode(new_tokens, skip_special_tokens=True)
            for text, q in zip(texts, batch):
                if success_math_answer(text, q.answer):
                    correct += 1.0
    finally:
        tok.padding_side = prev_side
        if was_training:
            model.train()
    total = float(len(questions))
    return float(correct / total) if total > 0 else 0.0


def _eval_via_trainer_llm(cfg: SoarConfig, trainer, tok,
                          questions: List[MathProblem]) -> float:
    """Greedy-score `questions`. Uses the colocate vLLM when present, else HF generate."""
    # TRL 1.16: colocate engine lives at `trainer.vllm_generation.llm`
    # (was `trainer.llm` in 0.19.x).
    gen = getattr(trainer, "vllm_generation", None)
    if gen is None or getattr(gen, "llm", None) is None:
        return _eval_via_trainer_hf(cfg, trainer, tok, questions)
    from vllm import SamplingParams

    prompts = [_student_prompt(q.problem) for q in questions]
    sp = SamplingParams(temperature=0.0, top_p=1.0,
                        max_tokens=cfg.student_max_new_tokens)
    outputs = gen.llm.generate(prompts, sp, use_tqdm=False)
    correct = sum(
        1.0 for o, q in zip(outputs, questions)
        if success_math_answer(o.outputs[0].text, q.answer)
    )
    total = float(len(questions))
    return float(correct / total) if total > 0 else 0.0


def _generate_teacher_batch_vllm(
    cfg: SoarConfig, ckpt_dir: Path, teacher_prompt_str: str, needed: int,
    accel: Accelerator,
) -> Tuple[List[str], List[MathProblem], int]:
    """Loop teacher vLLM gen until exactly `needed` parseable problems are accepted."""
    accepted_raw: List[str] = []
    accepted_problems: List[MathProblem] = []
    total_generated = 0
    round_idx = 0
    while len(accepted_problems) < needed:
        remaining = (needed - len(accepted_problems)) * 3
        # <question> prefix massively boosts parsable yield.
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
        _log(accel, f"  [teacher-gen round {round_idx}] "
                    f"total_generated={total_generated} "
                    f"accepted={len(accepted_problems)}/{needed} "
                    f"cum_yield={len(accepted_problems) / max(total_generated, 1):.2%}")
    return accepted_raw, accepted_problems, total_generated


# ------------- inner loop: TRL GRPO with student LoRA via peft_config -------------

from unittest.mock import patch
from vllm import LLM

original_llm_init = LLM.__init__

def eager_llm_init(self, *args, **kwargs):
    kwargs["enforce_eager"] = True
    return original_llm_init(self, *args, **kwargs)


def _run_inner_grpo(cfg: SoarConfig, tok, X_k: List[MathProblem], tag: str,
                    baseline_ckpt_dir: Path, out_dir: str,
                    accel: Accelerator,
                    *, inner_steps: Optional[int] = None,
                    eval_questions: Optional[List[MathProblem]] = None,
                    eval_baseline: bool = False,
                    ) -> Tuple[dict, Path, Optional[float], Optional[float]]:
    """One SOAR inner training run on X_k via TRL GRPOTrainer with LoRA student.

    `cfg.use_lora` is forced True at main() entry, so TRL always receives a
    `peft_config`, sets ref_model=None, and uses `disable_adapter()` for KL —
    the base weights (loaded from baseline_ckpt_dir) are the KL anchor.
    """
    inner_mem_entry = _mem_bytes() if _MEM_DEBUG else None

    def log_mem(stage, tag_="") -> None:
        if _MEM_DEBUG:
            _mem_log(f"{tag}/{stage}", baseline=inner_mem_entry)
        else:
            torch.cuda.synchronize()
            print(
                f"[{tag_}] {stage}: "
                f"allocated={torch.cuda.memory_allocated() / 1024**3:.2f} GiB",
                flush=True,
            )

    n_steps = inner_steps if inner_steps is not None else cfg.inner_steps
    ds = datasets.Dataset.from_list([
        {
            "prompt": tok.apply_chat_template(
                [{"role": "user", "content": _student_prompt(p.problem)}],
                tokenize=False,
                add_generation_prompt=True,
            ),
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
        cfg.base_model, torch_dtype=torch.bfloat16,
    )
    student_model.config.use_cache = False
    if (baseline_ckpt_dir / "adapter_config.json").exists():
        pm = PeftModel.from_pretrained(student_model, str(baseline_ckpt_dir))
        student_model = pm.merge_and_unload()
        del pm
        gc.collect()

    # TRL 0.19.1 constraint: (per_device_train_batch_size * num_processes) must be
    # divisible by num_generations so each micro-batch holds whole groups.
    G = cfg.student_group_size
    B = cfg.student_batch_problems
    N = accel.num_processes
    pdt = G // math.gcd(G, N)
    assert (pdt * N) % G == 0
    total_sequences_per_step = B * G
    denom = pdt * N
    if total_sequences_per_step % denom != 0:
        raise ValueError(
            f"Cannot hit batch={B} x group={G} = {total_sequences_per_step} sequences/step "
            f"with per_device_train_batch_size={pdt} x num_processes={N} = {denom}."
        )
    grad_accum = total_sequences_per_step // denom
    _log(accel, f"  [grpo] B={B} G={G} N={N} pdt={pdt} grad_accum={grad_accum} "
                f"(prompts/step={B}, seqs/step={B*G})")

    # TRL 1.16 dropped `max_prompt_length` from GRPOConfig; prompt truncation is
    # now handled implicitly by `vllm_max_model_length` (renamed from
    # `vllm_max_model_len`). `scale_rewards` moved from bool → str.
    args = GRPOConfig(
        output_dir=str(Path(out_dir) / "_trl" / tag),
        num_generations=G,
        scale_rewards="none",
        beta=cfg.kl_beta,
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
    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=list(cfg.lora_target_modules),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # trainer = GRPOTrainer(
    #     model=student_model,
    #     reward_funcs=reward_fn,
    #     args=args,
    #     train_dataset=ds,
    #     processing_class=tok,
    #     peft_config=peft_config,
    # )
    log_mem("before trainer construction")
    # Patch only while TRL constructs its colocated engine.
    with patch.object(LLM, "__init__", eager_llm_init):
        trainer = GRPOTrainer(
            model=student_model,
            reward_funcs=reward_fn,
            args=args,
            train_dataset=ds,
            processing_class=tok,
            peft_config=peft_config,
        )

    # Baseline eval reuses the colocate engine (initialised from baseline_ckpt_dir,
    # no training step has run yet).
    base_acc: Optional[float] = None
    if eval_baseline and eval_questions is not None:
        base_acc = _eval_via_trainer_llm(cfg, trainer, tok, eval_questions)

    trainer.train()
    log_mem("after training")

    # Post-training eval: TRL only syncs model→vLLM before each rollout, so after
    # the final gradient step the colocate engine lags by one step. Force a sync.
    # TRL 1.16: `_move_model_to_vllm()` → `vllm_generation.sync_weights()`.
    trained_acc: Optional[float] = None
    if eval_questions is not None:
        gen = getattr(trainer, "vllm_generation", None)
        if gen is not None and getattr(gen, "llm", None) is not None:
            gen.sync_weights()
        trained_acc = _eval_via_trainer_llm(cfg, trainer, tok, eval_questions)

    n = max(counters["n_rollouts"], 1)
    stats = {
        "n_rollouts": counters["n_rollouts"],
        "n_correct": counters["n_correct"],
        "mean_reward_run": counters["reward_sum"] / n,
    }
    trained_dir = Path(out_dir) / "_scratch" / f"trained_{tag}"

    log_mem("after evaluation")

    _save_peft_student_adapter(trainer, trained_dir)
    log_mem("after saving")
    import weakref

    refs = {
        "trainer": weakref.ref(trainer),
        "student_model": weakref.ref(student_model),
        "training_model": weakref.ref(trainer.model),
        "accelerator": weakref.ref(trainer.accelerator),
    }

    # TRL 1.16: colocate engine lives under `trainer.vllm_generation.llm`.
    _gen = getattr(trainer, "vllm_generation", None)
    if _gen is not None and getattr(_gen, "llm", None) is not None:
        refs["vllm"] = weakref.ref(_gen.llm)
        refs["vllm_engine"] = weakref.ref(_gen.llm.llm_engine)

    # After saving the trained model, on every training rank.
    torch.cuda.synchronize()

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

        # Shut down the engine BEFORE destroying its communication groups.
        core_client.shutdown()

        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()

        _gen.llm = None
        del core_client, engine, llm

    trainer.model.zero_grad(set_to_none=True)

    # Clear only the disposable trainer's Accelerator.
    if trainer.accelerator is not accel:
        trainer.accelerator.free_memory()

    del trainer, student_model
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    for name, ref in refs.items():
        print(f"{name}: {'STILL ALIVE' if ref() is not None else 'released'}")
    log_mem("after cleanup")

    print(
        f"allocated={torch.cuda.memory_allocated() / 1024**3:.2f} GiB, "
        f"reserved={torch.cuda.memory_reserved() / 1024**3:.2f} GiB",
        flush=True,
    )

    if _MEM_DEBUG and inner_mem_entry is not None:
        a, r = _mem_bytes()
        da, dr = a - inner_mem_entry[0], r - inner_mem_entry[1]
        print(
            f"[mem] inner-run {tag} net Δ: alloc={da/_GIB:+.3f}GiB "
            f"reserved={dr/_GIB:+.3f}GiB",
            flush=True,
        )
        if _MEM_TOP_TENSORS:
            _mem_top_tensors(_MEM_TOP_TENSORS)

    # # TRL's colocate vLLM (self.llm) must be torn down explicitly: LLMEngine.__del__
    # # only clears the DP group, leaving _TP/_PP + captured cudagraphs pinned.
    # if getattr(trainer, "llm", None) is not None:
    #     from vllm.distributed.parallel_state import destroy_model_parallel
    #     destroy_model_parallel()
    #     trainer.llm = None

    # del trainer, student_model
    # gc.collect()
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    # from IPython import embed; embed()  # noqa: F401


    return stats, trained_dir, base_acc, trained_acc


# ------------- teacher outer update (manual RLOO with disable_adapter KL ref) -------------

def _rloo_loss_disable_adapter(peft_model, batch, advantages, kl_beta) -> dict:
    """RLOO policy-gradient loss + KL to the base model (adapter disabled).

    Same shape/semantics as soar.rloo.rloo_loss but spelled inline here to keep
    this file self-contained (parallels _rloo_loss_with_ref in train_trl.py).
    """
    assert advantages.ndim == 1 and advantages.numel() == batch.input_ids.shape[0]

    logp_new = _per_token_logp(peft_model, batch.input_ids, batch.attention_mask)
    with torch.no_grad(), peft_model.disable_adapter():
        logp_ref = _per_token_logp(peft_model, batch.input_ids, batch.attention_mask)

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


def _teacher_rloo_step(
    teacher_model, tok, cfg: SoarConfig,
    teacher_prompt_str: str, accepted_raw: List[str],
    R_k: List[float], teacher_opt: torch.optim.Optimizer, n_per: int,
    accel: Accelerator,
) -> Tuple[dict, float]:
    """One teacher RLOO step. Returns (metrics, R_spread). Skips if g<2 or spread ≤ 1e-6."""
    g = len(R_k)
    needed = g * n_per
    dev = accel.device

    R_k_t = torch.tensor(R_k, dtype=torch.float32, device=dev)
    R_spread = (R_k_t.max() - R_k_t.min()).item() if g >= 2 else 0.0
    if g < 2 or R_spread <= 1e-6:
        return {"teacher/skipped": 1.0}, R_spread

    dataset_adv = rloo_advantages(R_k_t, group_size=g)         # [g]
    per_rollout_adv = dataset_adv.repeat_interleave(n_per)     # [g*n]
    batch = tokenize_rollouts(
        tok, [teacher_prompt_str + "<question>"] * needed, accepted_raw, device=dev.type,
        max_prompt_len=cfg.max_prompt_len,
        max_total_len=cfg.max_prompt_len + cfg.teacher_max_new_tokens,
    )

    teacher_model.train()
    agg = {"loss": 0.0, "pg_loss": 0.0, "kl": 0.0, "adv_abs_mean": 0.0}
    for i in range(needed):
        micro = type(batch)(
            input_ids=batch.input_ids[i:i + 1],
            attention_mask=batch.attention_mask[i:i + 1],
            completion_mask=batch.completion_mask[i:i + 1],
            rewards=batch.rewards,
        )
        out = _rloo_loss_disable_adapter(
            teacher_model, micro,
            advantages=per_rollout_adv[i:i + 1],
            kl_beta=cfg.kl_beta,
        )
        accel.backward(out["loss"] / needed)
        for k in agg:
            v = out[k]
            agg[k] += float(v.detach() if isinstance(v, torch.Tensor) else v)
    for k in agg:
        agg[k] /= needed

    grad_norm = accel.clip_grad_norm_(
        [p for p in teacher_model.parameters() if p.requires_grad], 1.0,
    )
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
    # File advertises dual-LoRA; force on so the student path can't silently
    # fall back to full-parameter training.
    cfg.use_lora = True

    # TRL colocate vLLM uses `distributed_executor_backend='external_launcher'`,
    # which reads torchrun-style env vars in uniproc_executor._distributed_args.
    # `accelerate launch --num_processes 1` goes through simple_launcher and
    # does NOT set them, so GRPOTrainer init crashes with KeyError: 'RANK'.
    # Fill in single-process defaults (setdefault leaves torchrun values alone).
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    accel = Accelerator()
    random.seed(cfg.seed + accel.process_index)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    if accel.is_main_process:
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(cfg.out_dir) / "config.json", "w") as f:
            json.dump({k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in cfg.__dict__.items()}, f, indent=2)

    _log(accel, f"[setup] world_size={accel.num_processes} "
                f"distributed_type={accel.distributed_type.name}  "
                f"base_model={cfg.base_model}")

    teacher_model = _build_teacher_peft(cfg)
    # PEFT + gradient checkpointing: enable on the base and route grads through
    # inputs so LoRA params actually receive gradients.
    teacher_model.base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    teacher_model.enable_input_require_grads()

    teacher_trainable = [p for p in teacher_model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in teacher_trainable)
    n_total = sum(p.numel() for p in teacher_model.parameters())
    _log(accel, f"[setup] teacher LoRA trainable params: {n_trainable:,} / {n_total:,} "
                f"({100 * n_trainable / n_total:.3f}%)")

    teacher_opt = torch.optim.AdamW(teacher_trainable, lr=cfg.teacher_lr)
    teacher_model, teacher_opt = accel.prepare(teacher_model, teacher_opt)
    tok = _build_tokenizer(cfg)

    _log(accel, f"[setup] loading fail@128 split from {cfg.fail128_dir}")
    train_set, _ = load_fail128_split(cfg.fail128_dir)
    _log(accel, f"[setup] |train|={len(train_set)}")
    assert len(train_set) >= cfg.n_reward_questions, "not enough train problems for Q_R"

    # Seed both role-dirs on disk from the pretrained teacher weights (LoRA is
    # zero-initialised so merged == base; vLLM reads these).
    scratch_dir = Path(cfg.out_dir) / "_scratch"
    teacher_ckpt = scratch_dir / "teacher"
    baseline_student_ckpt = scratch_dir / "baseline_student"
    teacher_latest_dir = Path(cfg.out_dir) / "teacher_latest"
    if accel.is_main_process:
        scratch_dir.mkdir(parents=True, exist_ok=True)
    _save_merged_teacher(accel.unwrap_model(teacher_model), tok, teacher_ckpt)

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

        # ----- 1. TEACHER GENERATES g*n parsed problems (ephemeral vLLM) -----
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
        Q_R = rng.sample(train_set, cfg.n_reward_questions)
        baseline_eval_sec = 0.0  # kept for log schema; cost folded into first inner run

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
                effective_inner_steps = (
                    cfg.inner_steps + cfg.inner_steps_per_promotion * n_promotions
                )
                inner_stats, trained_dir, base_acc_run, trained_acc = _run_inner_grpo(
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
            promotions_window.clear()
            promoted_this_step = 1
            _log(accel, f"[step {outer_step}] PROMOTED "
                        f"(k*={k_star}, j*={j_star}, MA={ma:+.3f})")

        # ----- 5. TEACHER RLOO UPDATE (KL ref via disable_adapter) -----
        t_phase = time.time()
        teacher_metrics, R_spread = _teacher_rloo_step(
            teacher_model, tok, cfg,
            teacher_prompt_str, accepted_raw, R_k, teacher_opt, n_per, accel,
        )
        # Persist updated teacher weights (merged) for next outer step's vLLM gen.
        _save_merged_teacher(accel.unwrap_model(teacher_model), tok, teacher_ckpt)
        # Mirror to a stable top-level path so external evals always have a
        # current teacher to point at (ckpt_outer{N}/ only fires every ckpt_every).
        _replace_dir(teacher_ckpt, teacher_latest_dir)
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
            log["sys/mem_alloc_delta_step_gib"] = (a_now - mem_step_start[0]) / _GIB
            log["sys/mem_reserved_delta_step_gib"] = (r_now - mem_step_start[1]) / _GIB
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
                if baseline_student_ckpt.exists():
                    _replace_dir(baseline_student_ckpt, ckpt_dir / "baseline_student")

        # Promoted winner was already copied into baseline_student_ckpt and
        # promotions/promo_NNN/; every other trained_* adapter is dead weight.
        for row in trained_ckpts:
            for p in row:
                if p is not None:
                    _rmtree(p)

    if accel.is_main_process:
        wandb.finish()
    _log(accel, f"[done] total promotions = {n_promotions}")


# ------------- CLI (identical flag set to train_trl.py) -------------

def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end sanity run")

    ap.add_argument("--base-model", type=str, default=None)
    ap.add_argument("--lora-r", type=int, default=None)
    ap.add_argument("--lora-alpha", type=int, default=None)
    ap.add_argument("--use-lora", action="store_true", default=None)

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
