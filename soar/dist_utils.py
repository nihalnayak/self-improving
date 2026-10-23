"""Distributed training helpers for torchrun + FSDP.

Safe no-ops when WORLD_SIZE=1 (plain `python -m soar.train`).
"""

import contextlib
import os
from typing import Any, Optional

import torch
import torch.distributed as dist


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_distributed() -> bool:
    return world_size() > 1


def is_main() -> bool:
    return rank() == 0


def init() -> None:
    if is_distributed() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank())


def shutdown() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast any pickle-able object from `src` to all ranks."""
    if not dist.is_initialized():
        return obj
    box = [obj if rank() == src else None]
    dist.broadcast_object_list(box, src=src)
    return box[0]


def summon_full_params(model, writeback: bool = False, rank0_only: bool = True):
    """Context: unsharded params on rank 0 (default) or all ranks.

    No-op when not distributed. Use `writeback=True` + `rank0_only=False` when all
    ranks need to mutate the same full tensor (e.g. restore from snapshot).
    """
    if not is_distributed():
        return contextlib.nullcontext()
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    return FSDP.summon_full_params(model, writeback=writeback, rank0_only=rank0_only)


def shard_indices(total: int) -> list[int]:
    """Indices this rank is responsible for, striped as [rank::world_size].

    If `total < world_size`, every rank gets all indices (correctness over throughput
    for tiny batches like the smoke config).
    """
    ws = world_size()
    if total < ws:
        return list(range(total))
    r = rank()
    return list(range(r, total, ws))
