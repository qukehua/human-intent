"""Distributed training helpers (torchrun / DDP)."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist


def is_dist_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_dist_available_and_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    if not is_dist_available_and_initialized():
        return 0
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    if not is_dist_available_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed(backend: str = "nccl") -> Tuple[bool, int, int, int]:
    """
    Initialize process group when launched with torchrun.

    Returns:
        (distributed, rank, local_rank, world_size)
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size <= 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return False, 0, 0, 1

    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    return True, rank, local_rank, world_size


def cleanup_distributed() -> None:
    if is_dist_available_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier() -> None:
    if is_dist_available_and_initialized():
        dist.barrier()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def seed_everything(seed: int, rank: int = 0) -> None:
    import numpy as np
    import random

    full_seed = int(seed) + int(rank)
    random.seed(full_seed)
    np.random.seed(full_seed)
    torch.manual_seed(full_seed)
    torch.cuda.manual_seed_all(full_seed)


def print0(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs)
