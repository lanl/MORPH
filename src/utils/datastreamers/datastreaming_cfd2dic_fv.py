import os
import glob
import random
import numpy as np
import h5py
import time
from torch.utils.data import IterableDataset, get_worker_info
from src.utils.data_preparation_fast import FastARDataPreparer
import torch.distributed as dist
from src.utils.main_process_ddp import is_main_process


def inflate_array(arr, axes):
    for ax in sorted(axes):
        arr = np.expand_dims(arr, axis=ax)
    return arr

class CFD2DICChunkedIterableDataset(IterableDataset):
    '''
    The spatial force field (no time variation) present in the dataset
    is modified to be spatiotemporal
    '''
    def __init__(self, data_path, split, ar_order, chunk_size=5, set_name='CFD2D-IC',
                 num_loadfiles: int = None):
        # Split: 'train' or 'val', ar_order: autoregressive window length
        self.split = split
        self.data_path = data_path
        
        # Gather file paths for both .h5 and .hdf5 extensions
        pattern1 = os.path.join(data_path, split, '*.h5')
        pattern2 = os.path.join(data_path, split, '*.hdf5')
        self.file_paths = sorted(glob.glob(pattern1) + glob.glob(pattern2))
        
        self.ar_order = ar_order
        self.set_name = set_name
        self.chunk_size = chunk_size
        self.num_loadfiles = num_loadfiles
        
        # Pre-compute total samples across all files for __len__
        self._total_samples = self._compute_total_samples()
        # Log discovery once per split and only for the first AR to avoid duplication
        worker = get_worker_info()
        if is_main_process() and worker is None and self.ar_order == 1:
            print(f"[{self.set_name}-{self.split}] Found {len(self.file_paths)} files "
                  f" in {os.path.join(data_path, split)}")
            if num_loadfiles: 
                print(f"[{self.set_name}-{self.split}] Loading {num_loadfiles} files …")

    def _compute_total_samples(self):
        """
        Compute total AR samples across all files by summing number of sims * (T - ar_order)
        """
        paths = (self.file_paths
             if self.num_loadfiles is None
             else self.file_paths[:self.num_loadfiles])
        
        total = 0
        for p in paths:
            with h5py.File(p, 'r') as f:
                # Read dataset shape: N_sims × T × ...
                N, T = f["force"].shape[:2]
            total += N * max(0, T - self.ar_order)
        return total
    
    def __len__(self):
        # Return total number of (input, target) samples across all files
        if dist.is_available() and dist.is_initialized():
            return self._total_samples // dist.get_world_size()
        else:
            return self._total_samples
        
    def __iter__(self):
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

        # 3) Global parameters
        total = self._total_samples
        G     = world_size * n_workers
        max_valid    = total - (total % G)           # drop the remainder
        per_subworker = max_valid // G
    
        # 4) Shuffle files once
        paths = self.file_paths[:self.num_loadfiles] if self.num_loadfiles else self.file_paths
        random.shuffle(paths)

        preparer   = FastARDataPreparer(self.ar_order, set_name=self.set_name)
        global_idx = 0                # counts *every* sample
        yielded    = 0                # counts only this sub‐worker’s yields
        my_id      = rank * n_workers + worker_id

        # Iterate through assigned files
        for path in paths:
            # Log file opening once
            if is_main_process() and worker is None and self.ar_order == 1:
                print(f"[{self.set_name}-{self.split}] Opening file: {path}")
            
            with h5py.File(path, 'r') as f5:
                # load all the fields
                # Directly read your three main fields
                force    = f5['force']
                vel      = f5['velocity'] 
                
                n_sims = force.shape[0]
                for start in range(0, n_sims, self.chunk_size):
                    end = min(start + self.chunk_size, n_sims)
                    force_chunk = force[start:end]
                    vel_chunk  = vel[start:end]
                    
                    # expand the fields
                    force_chunk =  np.expand_dims(force_chunk, axis = (1,5))  # time and field
                    force_chunk = np.repeat(force_chunk, repeats = vel_chunk.shape[1], axis = 1)  # repeat time
                    vel_chunk   =  np.expand_dims(vel_chunk, axis = 5)
                    # print(f"{force_chunk.shape},{vel_chunk.shape}")
                    
                    # concat along the field dimensions
                    batch = np.concatenate((force_chunk, vel_chunk), axis = 5).astype(np.float32)
                    # print(f"Concat size: {batch.shape}")
                    
                    # Inflate depth and channel dims → (b, T, 1, H, W, C, F)
                    batch = inflate_array(batch, axes=[2])
                    # print(f"Inflated size: {batch.shape}")
                    
                    # Log each sim load on main thread
                    if is_main_process() and worker is None and self.ar_order == 1:
                        print(f"[{self.set_name}-{self.split}] AR={self.ar_order}"
                              f" Loaded file from {os.path.basename(path)}, raw shape: {batch.shape}")
                    
                    X, y = preparer.prepare(batch)
                    for xi, yi in zip(X, y):
                        if global_idx >= max_valid:
                            return   # we’ve exhausted the common pool
    
                        if (global_idx % G) == my_id:
                            yield xi, yi
                            yielded += 1
                            if yielded >= per_subworker:
                                return
    
                        global_idx += 1