"""vLLM rollout helpers with a simple LoRA hot-swap: save adapter -> unique path+id -> generate."""

import itertools
import os
import shutil
from pathlib import Path
from typing import List, Optional

from vllm import SamplingParams
from vllm.lora.request import LoRARequest

from soar import dist_utils

# Globally-unique integer ID for every LoRA registration vLLM has seen.
# vLLM caches by (name, int_id); using a fresh id each time forces a reload.
_LORA_ID_COUNTER = itertools.count(1)


def _fresh_id() -> int:
    return next(_LORA_ID_COUNTER)


def sync_adapter_to_disk(peft_model, adapter_name: str, out_dir: str, tag: str) -> str:
    """Save the named adapter to a unique directory and return its path.

    PEFT writes the active adapter when you call save_pretrained with no `selected_adapters`
    arg (newer PEFT versions). We switch active adapter -> save -> switch back.

    Under FSDP: all ranks must enter `summon_full_params` together (collective); only
    rank 0 writes the files. Returns the same path on all ranks (shared filesystem).
    """
    prev_active = peft_model.active_adapter
    peft_model.set_adapter(adapter_name)
    path = Path(out_dir) / "adapters" / f"{adapter_name}_{tag}"
    with dist_utils.summon_full_params(peft_model, writeback=False, rank0_only=True):
        if dist_utils.is_main():
            path.mkdir(parents=True, exist_ok=True)
            # PEFT ≥0.14: save_pretrained accepts selected_adapters list; both work.
            peft_model.save_pretrained(str(path), selected_adapters=[adapter_name])
    peft_model.set_adapter(prev_active)
    dist_utils.barrier()
    # When selected_adapters is used, PEFT saves to <path>/<adapter_name>/ in some versions.
    # Normalize: if nested, use nested path; else the given path.
    nested = path / adapter_name
    final = str(nested) if nested.exists() and (nested / "adapter_config.json").exists() else str(path)
    return final


def vllm_generate(
    engine,
    prompts: List[str],
    lora_path: str,
    lora_name: str,
    *,
    max_new_tokens: int,
    temperature: float,
    stop: Optional[List[str]] = None,
    top_p: float = 1.0,
) -> List[str]:
    """Run a batched vLLM generate with a specific LoRA adapter path."""
    req = LoRARequest(lora_name=lora_name, lora_int_id=_fresh_id(), lora_path=lora_path)
    sp = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        stop=stop or [],
    )
    outputs = engine.generate(prompts, sp, lora_request=req, use_tqdm=False)
    return [o.outputs[0].text for o in outputs]


def cleanup_old_adapters(out_dir: str, keep_last: int = 4) -> None:
    """Keep adapter dir from growing without bound (optional housekeeping).

    Rank 0 only — avoid filesystem races.
    """
    if not dist_utils.is_main():
        return
    adir = Path(out_dir) / "adapters"
    if not adir.exists():
        return
    dirs = sorted(adir.iterdir(), key=lambda p: p.stat().st_mtime)
    for p in dirs[:-keep_last]:
        shutil.rmtree(p, ignore_errors=True)
