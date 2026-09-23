import torch.distributed as dist

# to avoid duplications while using DDP
def is_main_process() -> bool:
    """True on rank-0 or in single-GPU/debug runs."""
    return (not dist.is_initialized()) or dist.get_rank() == 0