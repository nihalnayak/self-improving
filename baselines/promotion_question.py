"""PQ baseline (SOAR paper §4.3): Promotion Questions, mixed-corpus GRPO.

Train a fresh base student with LoRA via TRL GRPO on
`fail@128 train ∪ D_best` (D_best = `promotion_questions.jsonl` written by a
completed SOAR run). After training, evaluate the final LoRA on the fail@128
test split via `eval.py` (pass@k with k ∈ {1,4,8,16,32}, 32 samples @ T=1.0 —
Table 3), and emit a result JSON that mirrors `baselines/promoted_student.py`
so `visualize/pass_at_k.py` picks it up with a PQ legend entry.

Training follows the same TRL + LoRA path as `soar.train_single_gpu_trl_lora`'s
`_run_inner_grpo` (dual-adapter-free: TRL wraps the base with a single student
LoRA via `peft_config`, uses `disable_adapter()` as the KL ref). Hyperparams:
`pq_group_size=32`, `pq_batch_problems=8`, `pq_kl_beta=0.001`,
`pq_lr_warmup_steps=20`, cosine schedule, `pq_max_new_tokens=1024`, T=1.0,
`pq_max_train_steps=200` (reduced from paper's 1500 for sweep tractability).

CLI:
    accelerate launch --num_processes 1 -m baselines.promotion_question \\
        --soar-run <soar-run> --out-dir <pq-training-dir> \\
        --fail128-dir <fail128> --base-model <hf-id> --lr 1e-05
"""

import argparse
import gc
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional
from unittest.mock import patch

import datasets
import torch
import wandb
from accelerate import Accelerator
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from vllm import LLM, SamplingParams

from soar.config import SoarConfig
from soar.data import MathProblem, _read_jsonl, load_fail128_split
from soar.verify import pass_at_k, score_math_answer, success_math_answer


# ------------- paper §B.1 prompt (identical to soar.train_single_gpu_trl_lora) -------------

STUDENT_PROMPT_TEMPLATE = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant "
    "solves it. The assistant first shows the complete reasoning process step by step, then "
    "provides the final answer in \\boxed{} format: 'User: [question] Assistant: [detailed "
    "reasoning] The final answer is: \\boxed{[answer]}.\n"
    "User: <QUESTION> Assistant: "
)


def _student_prompt(question: str) -> str:
    return STUDENT_PROMPT_TEMPLATE.replace("<QUESTION>", question)


def _load_promotion_questions(soar_run: Path) -> List[MathProblem]:
    """Read `{soar_run}/promotion_questions.jsonl`. Extra `promotion_idx` field
    is ignored by `MathProblem.from_dict`."""
    path = soar_run / "promotion_questions.jsonl"
    if not path.exists():
        print(f"[pq] WARN: {path} not found — training on fail@128 train only")
        return []
    return _read_jsonl(path)


# vLLM LLM.__init__ patch so TRL's colocate engine uses enforce_eager
# (mirrors soar.train_single_gpu_trl_lora).
_original_llm_init = LLM.__init__


def _eager_llm_init(self, *args, **kwargs):
    kwargs["enforce_eager"] = True
    return _original_llm_init(self, *args, **kwargs)


class PeriodicValCallback(TrainerCallback):
    """Every N trainer global_steps, greedy-eval the student (via colocate
    vLLM) on each named question set and log `pq/val/{name}_acc` to wandb.
    Scoring matches `eval.py` (boxed-equiv via `success_math_answer`), so the
    final `fail128_test` curve lines up with the post-training `pq/test_pass_at_1`.

    Independently, every `pass_at_k_every` steps, draw `pass_at_k_samples`
    completions on `pass_at_k_set_name` at T=`pass_at_k_temperature` and log
    `pq/val/{name}_pass_at_{k}` for each k in `pass_at_k_ks` — matches the
    paper's Table 3 metric so the in-training curve lines up with the final
    post-training pass@k written by eval.py.
    """

    def __init__(self, tok, questions_by_name, every, max_new_tokens,
                 pass_at_k_every, pass_at_k_samples, pass_at_k_temperature,
                 out_dir, pass_at_k_set_name="fail128_test", pass_at_k_ks=(32,)):
        self.tok = tok
        self.questions = questions_by_name
        self.every = every
        self.max_new_tokens = max_new_tokens
        self.pass_at_k_every = pass_at_k_every
        self.pass_at_k_samples = pass_at_k_samples
        self.pass_at_k_temperature = pass_at_k_temperature
        self.pass_at_k_set_name = pass_at_k_set_name
        self.pass_at_k_ks = tuple(pass_at_k_ks)
        self.out_dir = Path(out_dir)
        self.val_log_path = self.out_dir / "val_metrics.jsonl"
        self.trainer = None  # late-bound after GRPOTrainer construction

    def on_step_end(self, args, state, control, **kwargs):
        if self.trainer is None or state.global_step == 0:
            return
        do_greedy = state.global_step % self.every == 0
        do_pass_at_k = state.global_step % self.pass_at_k_every == 0
        if not (do_greedy or do_pass_at_k):
            return
        gen = getattr(self.trainer, "vllm_generation", None)
        if gen is None or getattr(gen, "llm", None) is None:
            return
        # TRL only syncs weights before each rollout; force latest LoRA.
        gen.sync_weights()
        if do_greedy:
            sp = SamplingParams(temperature=0.0, top_p=1.0,
                                max_tokens=self.max_new_tokens)
            metrics = {}
            for name, qs in self.questions.items():
                if not qs:
                    continue
                prompts = [_student_prompt(q.problem) for q in qs]
                outs = gen.llm.generate(prompts, sp, use_tqdm=False)
                correct = sum(1.0 for o, q in zip(outs, qs)
                              if success_math_answer(o.outputs[0].text, q.answer))
                metrics[f"pq/val/{name}_acc"] = correct / len(qs)
            if metrics:
                print(f"[pq] step={state.global_step} val: "
                      + "  ".join(f"{k.split('/')[-1]}={v:.3f}"
                                  for k, v in metrics.items()))
                # Use train/global_step as the x-axis (step_sync=True is set
                # by HF's WandbCallback) — passing step= here is dropped as
                # non-monotonic since TRL has advanced wandb's internal _step.
                wandb.log({**metrics, "train/global_step": state.global_step})
                with self.val_log_path.open("a") as f:
                    f.write(json.dumps({"step": state.global_step, **metrics}) + "\n")
        if do_pass_at_k:
            qs = self.questions.get(self.pass_at_k_set_name, [])
            if qs:
                sp = SamplingParams(
                    n=self.pass_at_k_samples,
                    temperature=self.pass_at_k_temperature,
                    top_p=1.0,
                    max_tokens=self.max_new_tokens,
                )
                prompts = [_student_prompt(q.problem) for q in qs]
                outs = gen.llm.generate(prompts, sp, use_tqdm=False)
                completions_by_q = [[c.text for c in o.outputs] for o in outs]
                golds = [q.answer for q in qs]
                pk = pass_at_k(completions_by_q, golds, ks=self.pass_at_k_ks)
                pk_metrics = {
                    f"pq/val/{self.pass_at_k_set_name}_pass_at_{k}": v
                    for k, v in pk.items()
                }
                print(f"[pq] step={state.global_step} val pass@k: "
                      + "  ".join(f"{k.split('/')[-1]}={v:.3f}"
                                  for k, v in pk_metrics.items()))
                wandb.log({**pk_metrics, "train/global_step": state.global_step})
                with self.val_log_path.open("a") as f:
                    f.write(json.dumps({"step": state.global_step, **pk_metrics}) + "\n")


def _teardown_colocate_vllm(trainer) -> None:
    """Mirror the TRL-1.16 teardown dance from `_run_inner_grpo` so a
    subsequent vLLM engine (eval.py) can allocate GPU memory cleanly."""
    gen = getattr(trainer, "vllm_generation", None)
    if gen is None or getattr(gen, "llm", None) is None:
        return
    llm = gen.llm
    engine = llm.llm_engine
    core_client = getattr(engine, "engine_core", None)
    if core_client is None or not callable(getattr(core_client, "shutdown", None)):
        raise RuntimeError("Unexpected vLLM engine layout; inspect installed vLLM version.")
    core_client.shutdown()
    from vllm.distributed.parallel_state import destroy_model_parallel
    destroy_model_parallel()
    gen.llm = None
    del core_client, engine, llm


def _build_grpo_args(cfg: SoarConfig, out_dir: Path, run_name: str, lr: float,
                     max_steps: int, vllm_mem: float) -> GRPOConfig:
    # Same gcd formula as _run_inner_grpo: (per_device_train_batch_size * N)
    # must be divisible by num_generations, and total seqs/step
    # (pq_batch_problems * pq_group_size) must be an integer multiple of denom.
    G = cfg.pq_group_size
    B = cfg.pq_batch_problems
    N = 1  # num_processes
    pdt = G // math.gcd(G, N)
    total = B * G
    denom = pdt * N
    assert total % denom == 0, f"B={B} G={G} N={N} pdt={pdt} not divisible"
    grad_accum = total // denom
    print(f"[pq] B={B} G={G} N={N} pdt={pdt} grad_accum={grad_accum} "
          f"(prompts/step={B}, seqs/step={B*G})")

    return GRPOConfig(
        output_dir=str(out_dir / "_trl"),
        run_name=run_name,
        num_generations=G,
        scale_rewards="none",
        beta=cfg.pq_kl_beta,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=cfg.pq_lr_warmup_steps,
        per_device_train_batch_size=pdt,
        gradient_accumulation_steps=grad_accum,
        max_steps=max_steps,
        max_completion_length=cfg.pq_max_new_tokens,
        temperature=cfg.pq_temperature,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=vllm_mem,
        vllm_max_model_length=cfg.max_prompt_len + cfg.pq_max_new_tokens,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to=["wandb"],
        remove_unused_columns=False,
        seed=cfg.seed,
        disable_tqdm=True,
    )


def _train_pq_student(cfg: SoarConfig, args, combined: List[MathProblem],
                       d_best: List[MathProblem],
                       fail128_train: List[MathProblem],
                       fail128_test: List[MathProblem],
                       out_dir: Path) -> Path:
    """TRL GRPO training on `combined` problems. Returns the saved LoRA path."""
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    ds = datasets.Dataset.from_list([
        {
            "prompt": tok.apply_chat_template(
                [{"role": "user", "content": _student_prompt(p.problem)}],
                tokenize=False,
                add_generation_prompt=True,
            ),
            "answer": p.answer,
        }
        for p in combined
    ])

    counters = {"n_rollouts": 0, "n_correct": 0, "reward_sum": 0.0}

    def reward_fn(completions, answer, **_):
        scores = [float(score_math_answer(c, a)) for c, a in zip(completions, answer)]
        counters["n_rollouts"] += len(scores)
        counters["n_correct"] += sum(1 for s in scores if s >= 120.0)
        counters["reward_sum"] += sum(scores)
        return scores

    print(f"[pq] loading base model: {cfg.base_model}")
    student_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=torch.bfloat16,
    )
    student_model.config.use_cache = False

    grpo_args = _build_grpo_args(
        cfg, out_dir, run_name=out_dir.name, lr=args.lr,
        max_steps=args.max_steps, vllm_mem=args.vllm_gpu_memory_utilization,
    )
    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=list(cfg.lora_target_modules),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    val_cb = PeriodicValCallback(
        tok=tok,
        questions_by_name={
            "d_best": list(d_best),
            "fail128_train": list(fail128_train),
            "fail128_test": list(fail128_test),
        },
        every=args.val_every,
        max_new_tokens=cfg.pq_max_new_tokens,
        pass_at_k_every=args.pass_at_k_every,
        pass_at_k_samples=cfg.pq_pass_at_k_samples,
        pass_at_k_temperature=cfg.pq_temperature,
        out_dir=out_dir,
    )

    print("[pq] constructing GRPOTrainer (enforce_eager patched)")
    with patch.object(LLM, "__init__", _eager_llm_init):
        trainer = GRPOTrainer(
            model=student_model,
            reward_funcs=reward_fn,
            args=grpo_args,
            train_dataset=ds,
            processing_class=tok,
            peft_config=peft_config,
            callbacks=[val_cb],
        )
    val_cb.trainer = trainer

    print(f"[pq] trainer.train(max_steps={args.max_steps}, val_every={args.val_every})")
    trainer.train()

    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    print(f"[pq] saved LoRA to {final_dir}")

    n = max(counters["n_rollouts"], 1)
    print(f"[pq] train rollouts={counters['n_rollouts']} "
          f"correct={counters['n_correct']} "
          f"mean_reward={counters['reward_sum']/n:+.3f}")

    _teardown_colocate_vllm(trainer)
    trainer.model.zero_grad(set_to_none=True)
    if trainer.accelerator is not None:
        trainer.accelerator.free_memory()
    del trainer, student_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return final_dir


def _run_eval_subprocess(cfg: SoarConfig, lora_path: Path, args) -> dict:
    tmp_dir = Path(tempfile.mkdtemp(prefix="pq_eval_"))
    tmp_json = tmp_dir / "eval.json"
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable, str(repo_root / "eval.py"),
        "--student_lora", str(lora_path),
        "--fail128-dir", args.fail128_dir,
        "--n-samples", str(cfg.pq_pass_at_k_samples),
        "--temperature", str(cfg.pq_temperature),
        "--max-tokens", str(cfg.pq_max_new_tokens),
        "--ks", ",".join(str(k) for k in cfg.pq_pass_at_k_ks),
        "--output-json", str(tmp_json),
    ]
    print(f"[pq] running final eval: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(repo_root))
    return json.loads(tmp_json.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--soar-run", type=str, required=True,
                    help="path to a completed SOAR run (reads promotion_questions.jsonl + config.json)")
    ap.add_argument("--out-dir", type=str, required=True,
                    help="PQ training output dir (final LoRA + _trl logs land here)")
    ap.add_argument("--fail128-dir", type=str, required=True)
    ap.add_argument("--base-model", type=str, default=None,
                    help="override; default = base_model recorded in <soar-run>/config.json")
    ap.add_argument("--lr", type=float, default=None,
                    help="student LR; default = cfg.pq_lr (1e-05)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override cfg.pq_max_train_steps (default 200)")
    ap.add_argument("--val-every", type=int, default=5,
                    help="greedy-eval d_best/fail128_train/fail128_test every N "
                         "trainer global_steps and log to wandb")
    ap.add_argument("--pass-at-k-every", type=int, default=25,
                    help="sample cfg.pq_pass_at_k_samples completions on fail128_test "
                         "every N trainer global_steps and log "
                         "pq/val/fail128_test_pass_at_32 to wandb")
    ap.add_argument("--results-root", type=str, default="files/results/baselines")
    ap.add_argument("--wandb-project", type=str, default="soar-mini-pq")
    ap.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=None,
                    help="override cfg.seed; seeds random/torch and GRPOConfig")
    args = ap.parse_args()

    soar_run = Path(args.soar_run)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(soar_run / "config.json") as f:
        run_cfg = json.load(f)

    cfg = SoarConfig()
    cfg.fail128_dir = args.fail128_dir
    cfg.base_model = args.base_model or run_cfg["base_model"]
    if args.lr is not None:
        cfg.pq_lr = args.lr
    else:
        args.lr = cfg.pq_lr
    if args.max_steps is None:
        args.max_steps = cfg.pq_max_train_steps
    if args.seed is not None:
        cfg.seed = args.seed

    # torchrun env defaults so TRL's colocate vLLM can bring up its distributed
    # groups under `accelerate launch --num_processes 1`.
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    accel = Accelerator()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    model_short = Path(cfg.base_model).name.lower()
    run_name = out_dir.name

    print(f"[pq] soar_run = {soar_run}")
    print(f"[pq] out_dir  = {out_dir}  (run_name={run_name})")
    print(f"[pq] base     = {cfg.base_model}  (model_short={model_short})")
    print(f"[pq] lr       = {args.lr}  max_steps = {args.max_steps}")

    with open(out_dir / "config.json", "w") as f:
        json.dump({k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in cfg.__dict__.items()}, f, indent=2)

    print(f"[pq] loading fail@128 from {cfg.fail128_dir}")
    train_set, test_set = load_fail128_split(cfg.fail128_dir)
    print(f"[pq] |train|={len(train_set)}  |test|={len(test_set)}")

    d_best = _load_promotion_questions(soar_run)
    assert train_set or d_best, "[pq] no training data at all"
    combined = list(train_set) + list(d_best)
    rng.shuffle(combined)
    print(f"[pq] |D_best|={len(d_best)}  |combined|={len(combined)}")

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        mode=cfg.wandb_mode,
        config={**cfg.__dict__, "baseline": "pq", "soar_run": str(soar_run),
                "lr": args.lr, "max_steps": args.max_steps,
                "n_d_best": len(d_best), "n_fail128_train": len(train_set)},
    )

    final_lora = _train_pq_student(
        cfg, args, combined, d_best, train_set, test_set, out_dir,
    )

    eval_report = _run_eval_subprocess(cfg, final_lora, args)

    result = {
        "baseline": "promotion_question",
        "baseline_tag": "PQ",
        "run_name": run_name,
        "out_dir": str(out_dir),
        "soar_run": str(soar_run),
        "base_model": cfg.base_model,
        "model_short": model_short,
        "lr": args.lr,
        "max_steps": args.max_steps,
        "seed": cfg.seed,
        "n_d_best": len(d_best),
        "n_fail128_train": len(train_set),
        "lora_path": str(final_lora),
        "test_pass_at_k": eval_report["pass_at_k"],
        "n_test": eval_report["n_test"],
        "n_samples": eval_report["n_samples"],
        "temperature": eval_report["temperature"],
        "max_tokens": eval_report["max_tokens"],
        "ks": eval_report["ks"],
    }
    out_path = Path(args.results_root) / model_short / run_name / "promotion_question.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[pq] wrote {out_path}")
    print("[pq] test pass@k = "
          + ", ".join(f"{k}:{v}" for k, v in result["test_pass_at_k"].items()))

    for k_str, v in result["test_pass_at_k"].items():
        wandb.log({f"pq/test_pass_at_{k_str}": float(v)})
    wandb.finish()


if __name__ == "__main__":
    main()
