"""Small helpers shared by the training and inference code."""

import os
import random

import numpy as np
import torch


def exists(x) -> bool:
    return x is not None


def _distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_world_size() -> int:
    return torch.distributed.get_world_size() if _distributed() else 1


def print_once(*args) -> None:
    """Print only on the main process."""
    if not _distributed() or torch.distributed.get_rank() == 0:
        print(*args)


def set_seed(seed: int = 0) -> None:
    """Seed torch (CPU and CUDA), numpy and `random`."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def sort_dict(d: dict) -> dict:
    return dict(sorted(d.items()))
