"""Load Qwen base + 2 LoRA adapters (teacher, student) and a vLLM engine on the same GPU."""

import os
from typing import Tuple

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from soar.config import SoarConfig


def load_base_and_adapters(cfg: SoarConfig) -> Tuple[PeftModel, "AutoTokenizer"]:
    """Return a PEFT-wrapped model with two adapters: 'teacher' (active) and 'student'."""
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    base.config.use_cache = False  # safer during training

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=list(cfg.lora_target_modules),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg, adapter_name="teacher")
    model.add_adapter("student", lora_cfg)
    # start with teacher active; main loop swaps as needed
    model.set_adapter("teacher")
    return model, tok


def load_vllm_engine(cfg: SoarConfig):
    """Load a vLLM engine on the same GPU, with LoRA enabled.

    The gpu_memory_utilization is set low so the HF model + optimizer state
    (which already live on this GPU) have room.
    """
    from vllm import LLM
    return LLM(
        model=cfg.base_model,
        enable_lora=True,
        max_loras=cfg.vllm_max_loras,
        max_lora_rank=cfg.lora_r,
        dtype="bfloat16",
        gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        max_model_len=cfg.vllm_max_model_len,
        enforce_eager=True,       # skip torch.compile for simpler launch
        disable_log_stats=True,
    )


def chat_prompt(tok, user_msg: str, system_msg: str | None = None) -> str:
    """Apply Qwen chat template and return a string prompt (for vLLM) with assistant start."""
    msgs = []
    if system_msg:
        msgs.append({"role": "system", "content": system_msg})
    msgs.append({"role": "user", "content": user_msg})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
