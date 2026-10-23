"""Load base model + 2 LoRA adapters (teacher, student) and a vLLM engine on the same GPU."""

import functools
from typing import Tuple

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from soar import dist_utils
from soar.config import SoarConfig


# Exact student prompt from paper page 20, Appendix B.1. Used for fail@128 filtering,
# inner-loop student training, and evaluation (same prompt in all three places).
# `<QUESTION>` is substituted with the actual problem text via str.replace (avoids clashes
# with the literal `{}` braces in `\boxed{}`).
STUDENT_PROMPT_TEMPLATE = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant "
    "solves it. The assistant first shows the complete reasoning process step by step, then "
    "provides the final answer in \\boxed{} format: 'User: [question] Assistant: [detailed "
    "reasoning] The final answer is: \\boxed{[answer]}.\n"
    "User: <QUESTION> Assistant: "
)


def _decoder_layer_cls(base_model) -> type:
    """Find the transformer-block class for FSDP's `transformer_auto_wrap_policy`.

    Walks the model and picks the first module class whose name ends in "DecoderLayer"
    (Llama, Qwen, Mistral, ...). Falls back to the top-level base model class.
    """
    for module in base_model.modules():
        name = type(module).__name__
        if name.endswith("DecoderLayer"):
            return type(module)
    return type(base_model)


def load_base_and_adapters(cfg: SoarConfig) -> Tuple[PeftModel, "AutoTokenizer"]:
    """Return a PEFT-wrapped model with two adapters: 'teacher' (active) and 'student'.

    When launched under torchrun (WORLD_SIZE > 1), the returned model is additionally
    wrapped with FSDP (FULL_SHARD, bf16 compute, use_orig_params=True for PEFT compat).
    """
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Pin this rank's shard to its local GPU; falls back to cuda:0 single-GPU.
    device = f"cuda:{dist_utils.local_rank()}"
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        dtype=torch.bfloat16,
        device_map={"": device},
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

    if dist_utils.is_distributed():
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={_decoder_layer_cls(base)},
        )
        model = FSDP(
            model,
            auto_wrap_policy=wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16,
            ),
            device_id=dist_utils.local_rank(),
            use_orig_params=True,  # REQUIRED for PEFT set_adapter / disable_adapter
            limit_all_gathers=True,
        )
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
    """Apply the model's chat template and return a string prompt (for vLLM) with assistant
    start. Used for the teacher turn.
    """
    msgs = []
    if system_msg:
        msgs.append({"role": "system", "content": system_msg})
    msgs.append({"role": "user", "content": user_msg})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def student_chat_prompt(tok, question: str) -> str:
    """Build the student input exactly as specified in SOAR paper page 20.

    The paper's student prompt is a self-contained conversation template (not the model's
    chat template), so we substitute `{question}` directly and feed the result as a plain
    text prompt to vLLM. The trailing "Assistant: " primes the model to continue.
    """
    return STUDENT_PROMPT_TEMPLATE.replace("<QUESTION>", question)
