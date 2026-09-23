import torch
import os

# Make sure DataLoader workers don't spawn big BLAS pools
def dl_worker_init_fn(_):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass