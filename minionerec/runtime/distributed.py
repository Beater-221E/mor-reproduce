"""Single-GPU / multi-GPU helpers (torchrun + optional plain python)."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist


def env_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def env_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def env_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_distributed() -> bool:
    return env_world_size() > 1


def is_main_process() -> bool:
    return env_rank() == 0


def _ensure_nccl_defaults() -> None:
    """
    Match scripts/env_mor.sh when the user launches bare torchrun.
    Without these, multi-V100 often busy-waits at ~100% util / ~500MiB with no logs.
    """
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_SHM_DISABLE", "0")
    os.environ.setdefault("NCCL_NET", "Socket")
    # Skip Tailscale/docker/loopback; otherwise NCCL can hang on the wrong iface.
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "^lo,docker,tailscale,tun,veth")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", os.environ["NCCL_SOCKET_IFNAME"])


def setup_distributed() -> tuple[int, int, int, torch.device, bool]:
    """
    Initialize process group when launched by torchrun.

    Returns: (rank, local_rank, world_size, device, distributed)
    Compatible with NPROC=1 (torchrun or plain python).
    """
    world_size = env_world_size()
    rank = env_rank()
    local_rank = env_local_rank()
    distributed = world_size > 1

    if distributed:
        _ensure_nccl_defaults()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if distributed and not dist.is_initialized():
        if rank == 0:
            print(
                f"[dist] init_process_group NCCL world_size={world_size} "
                f"MASTER_ADDR={os.environ.get('MASTER_ADDR')} "
                f"MASTER_PORT={os.environ.get('MASTER_PORT')} "
                f"NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE')} "
                f"NCCL_SOCKET_IFNAME={os.environ.get('NCCL_SOCKET_IFNAME', '<default>')}",
                flush=True,
            )
        # Pass device_id to avoid NCCL "device used by this process is currently unknown"
        kwargs = {"backend": "nccl"}
        if device.type == "cuda":
            try:
                kwargs["device_id"] = device
            except TypeError:
                pass
        dist.init_process_group(**kwargs)
        if rank == 0:
            print("[dist] process group ready", flush=True)
    return rank, local_rank, world_size, device, distributed


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def print_rank0(msg: str, **kwargs: Any) -> None:
    if is_main_process():
        print(msg, flush=True, **kwargs)


def resolve_effective_batch(micro: int, accum: int, world_size: int) -> int:
    """B_eff = micro * world_size * accum (official SFT/RL meaning)."""
    return int(micro) * int(world_size) * int(accum)


def resolve_sft_accum(global_batch: int, micro: int, world_size: int) -> int:
    """
    Official sft.py:
      gradient_accumulation_steps = batch_size // micro_batch_size // world_size
    Keep global_batch fixed; adapt accum to hardware.
    """
    denom = max(1, int(micro) * int(world_size))
    accum = max(1, int(global_batch) // denom)
    return accum


def all_reduce_mean_scalar(value: float, device: torch.device) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    t = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())


def all_reduce_sum_scalar(value: float, device: torch.device) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    t = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())
