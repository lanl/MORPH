import os
import glob
import random
import numpy as np
import h5py
import time
import torch
from torch.utils.data import IterableDataset, get_worker_info
from src.utils.data_preparation_fast import FastARDataPreparer
import torch.distributed as dist
from src.utils.main_process_ddp import is_main_process

def inflate_array(arr, axes):
    for ax in sorted(axes):
        arr = np.expand_dims(arr, axis=ax)
    return arr

class CFD2DChunkedIterableDataset(IterableDataset):
    def __init__(self, data_path, split, ar_order, chunk_size=5, set_name='CFD2D',
                 num_loadfiles: int = None, seed: int = 1234):
        
        # shuffling the chunk within itself
        self.base_seed = seed
        self.epoch = 0
        
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
        if is_main_process() and worker is None:
            print(f"[{self.set_name}-{self.split}] Found {len(self.file_paths)} files "
                  f" in {os.path.join(data_path, split)}")
            if num_loadfiles: 
                print(f"[{self.set_name}-{self.split}] Loading {num_loadfiles} files …")

    def _compute_total_samples(self):
        """
        Compute total AR samples across all files by summing number of sims * (T - ar_order)
        """
        paths = (self.file_paths if self.num_loadfiles is None
             else self.file_paths[:self.num_loadfiles])
        
        total = 0
        for p in paths:
            with h5py.File(p, 'r') as f:
                # Read dataset shape: N_sims × T × ...
                N, T = f["Vx"].shape[:2]
            total += N * max(0, T - self.ar_order)
        return total
    
    def __len__(self):
        # Return total number of (input, target) samples across all files
        return self._total_samples
    
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

        # 3) Global parameters
        total = self._total_samples
        G     = world_size * n_workers
        max_valid    = total - (total % G)           # drop the remainder
        per_subworker = max_valid // G
    
        # 4) Selective files to load
        paths = self.file_paths[:self.num_loadfiles] if self.num_loadfiles else self.file_paths

        preparer   = FastARDataPreparer(self.ar_order, set_name=self.set_name)
        global_idx = 0                # counts *every* sample
        yielded    = 0                # counts only this sub‐worker’s yields
        my_id      = rank * n_workers + worker_id

        # Iterate through assigned files
        for path in paths:
            with h5py.File(path, 'r') as f5:
                # load all the fields
                vx       = f5['Vx']        
                vy       = f5['Vy']       
                density  = f5['density']  
                pressure = f5['pressure']
                                
                n_sims = vx.shape[0]
                for start in range(0, n_sims, self.chunk_size):
                    end = min(start + self.chunk_size, n_sims)
                    vx_chunk       = vx[start:end]      
                    vy_chunk       = vy[start:end]
                    density_chunk  = density[start:end]  
                    pressure_chunk = pressure[start:end] 
                    
                    # expand dims
                    vx_chunk = np.expand_dims(vx_chunk, axis = (4,5))                # (100,21,128,128,1,1)
                    vy_chunk = np.expand_dims(vy_chunk, axis = (4,5))                # (100,21,128,128,1,1)
                    density_chunk = np.expand_dims(density_chunk, axis = (4,5))      # (100,21,128,128,1,1)
                    pressure_chunk = np.expand_dims(pressure_chunk, axis = (4,5))    # (100,21,128,128,1,1)
                    
                    v = np.concatenate((vx_chunk, vy_chunk), axis=4)                 # (100,21,128,128,2,1)
                    dens = np.repeat(density_chunk, repeats=2, axis=4)               # (100,21,128,128,2,1)
                    press = np.repeat(pressure_chunk, repeats=2, axis=4)             # (100,21,128,128,2,1)
                    
                    # Stack features along last dim
                    batch = np.concatenate((v, dens, press), axis = 5)               # (100,21,128,128,2,3)
                    
                    # Inflate
                    batch = inflate_array(batch, axes=[2])                           # (100,21,1,128,128,2,3)
                    
                    # random shuffling (epoch seed) batch
                    perm = torch.randperm(batch.shape[0], generator=g).numpy()
                    batch_shuff = batch[perm]
                    
                    # prepare into inputs and targets
                    #print(f'{self.set_name} Preparing the dataset (X(t),X(t+1))...')
                    X, y = preparer.prepare(batch_shuff)
                    for xi, yi in zip(X, y):
                        if global_idx >= max_valid:
                            return   # we’ve exhausted the common pool
    
                        if (global_idx % G) == my_id:
                            yield xi, yi
                            yielded += 1
                            if yielded >= per_subworker:
                                return
    
                        global_idx += 1
