import os
import numpy as np
import h5py
import random
import time
import torch
from torch.utils.data import IterableDataset, get_worker_info
from src.utils.data_preparation_fast import FastARDataPreparer
import torch.distributed as dist
from src.utils.main_process_ddp import is_main_process

def inflate_array(arr, axes):
    """
    Utility to insert singleton dims at specified axes.
    """
    for ax in sorted(axes):
        arr = np.expand_dims(arr, axis=ax)
    return arr

class SWChunkedIterableDataset(IterableDataset):
    def __init__(self, data_root: str, split: str, ar_order: int,
                 chunk_size: int = 10, set_name: str = 'SW', seed: int = 1234):
        # shuffling the chunk within itself
        self.base_seed = seed
        self.epoch = 0
        
        self.split = split
        self.ar_order = ar_order
        self.chunk_size = chunk_size
        self.set_name = set_name
        self.h5_path = os.path.join(data_root, split, f"2D_rdb_NA_NA_{split}.h5")
        if is_main_process() and not os.path.isfile(self.h5_path):
            raise FileNotFoundError(f"Missing DR split file: {self.h5_path}")
        self._length = None
        
        # Log discovery once per split and only for the first AR to avoid duplication
        worker = get_worker_info()
        if is_main_process() and worker is None:
            print(f"[{self.set_name}-{self.split}] Found 1 file in {os.path.join(data_root, split)}")
   
    def __len__(self):
        # Lazily compute total number of samples: sims × (T - ar_order)
        if self._length is None:
            with h5py.File(self.h5_path, 'r') as f5:
                keys = list(f5.keys())
                T = f5[keys[0]]['data'].shape[0]
                self._length = len(keys) * max(0, T - self.ar_order)
        return self._length
    
    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        
    def __iter__(self):
        # --- set randomness with epoch seed ---
        g = torch.Generator()
        g.manual_seed(self.base_seed + self.epoch)
        
        # 1) DDP shard info
        if dist.is_available() and dist.is_initialized():
            world_size, rank = dist.get_world_size(), dist.get_rank()
        else:
            world_size, rank = 1, 0

        # 2) DataLoader‐worker shard info
        worker = get_worker_info()
        if worker is not None:
            n_workers = worker.num_workers
            worker_id = worker.id
        else:
            n_workers = 1
            worker_id = 0

        # 3) Compute global parameters
        total    = len(self)
        G        = world_size * n_workers
        max_valid    = total - (total % G)           # drop remainder
        per_subworker = max_valid // G
        my_id    = rank * n_workers + worker_id
        
        # print statement to confirm parallelization
        # print(f'[{self.set_name}](world_size-rank-worker_id-total_id)'
        #       f'->{world_size}-{rank}-{worker.id}-{my_id}')
        
        # 4) Open file and shuffle keys once
        f5   = h5py.File(self.h5_path, 'r')
        keys = sorted(f5.keys())
        #random.seed(42)
        #random.shuffle(keys)

        preparer   = FastARDataPreparer(self.ar_order, set_name=self.set_name)
        global_idx = 0               # counts every (xi, yi)
        yielded    = 0               # counts how many *this* sub-worker has yielded
    
        # 5) Stream in chunks, prepare AR, and yield only your share
        for idx in range(0, len(keys), self.chunk_size):
            batch_keys = keys[idx : idx + self.chunk_size]
            batch = np.stack([f5[k]['data'][...] for k in batch_keys], axis=0)
            batch = inflate_array(batch, axes=[2,5])

            # random shuffling (epoch seed) batch
            perm = torch.randperm(batch.shape[0], generator=g).numpy()
            batch_shuff = batch[perm]
            X, y = preparer.prepare(batch_shuff)
            for xi, yi in zip(X, y):
                # once we've walked past the common pool, we're done
                if global_idx >= max_valid:
                    return

                # if it's our turn in the global round‐robin
                if (global_idx % G) == my_id:
                    # if global_idx < 10:   # debug
                    #     print(f"[{self.set_name}] [first10] gidx={global_idx}->rank={rank}"
                    #           f" worker={worker_id} my_id={my_id}",flush=True)
                    yield xi, yi
                    yielded += 1
                    # stop once we've hit our quota
                    if yielded >= per_subworker:
                        return

                global_idx += 1
