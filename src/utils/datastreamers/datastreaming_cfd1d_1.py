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

class CFD1DChunkedIterableDataset(IterableDataset):
    def __init__(self, data_path, split, ar_order, chunk_size=5, set_name='CFD1D',
                 num_loadfiles: int = None, num_sims: int = None, seed: int = 1234):
        
        # shuffling the batch
        self.base_seed = seed
        self.epoch = 0
        
        # Split: 'train' or 'val', ar_order: autoregressive window length
        self.split = split
        self.data_path = data_path
        self.ar_order = ar_order
        self.set_name = set_name
        self.chunk_size = chunk_size
        self.num_loadfiles = num_loadfiles
        self.num_sims = num_sims
        
        # Gather file paths for both .h5 and .hdf5 extensions
        pattern1 = os.path.join(data_path, split, '*.h5')
        pattern2 = os.path.join(data_path, split, '*.hdf5')
        self.file_paths = sorted(glob.glob(pattern1) + glob.glob(pattern2))
        
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
                data = f["Vx"]
                n_sims = data.shape[0] if self.num_sims == None else self.num_sims
                t_steps = data.shape[1]
            total += n_sims * max(0, t_steps - self.ar_order)
        
        return total
    
    def __len__(self):
        return self._total_samples
    
    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        
    def __iter__(self):
        # --- set randomness with epoch seed ---
        g = torch.Generator()
        g.manual_seed(self.base_seed + self.epoch)
    
        # --- DDP info --
        if dist.is_available() and dist.is_initialized():
            world_size, rank = dist.get_world_size(), dist.get_rank()
        else:
            world_size, rank = 1, 0

        # --- DataLoader‐worker shard info ---
        worker = get_worker_info()
        if worker is not None:
            n_workers = worker.num_workers
            worker_id = worker.id
        else:
            n_workers = 1
            worker_id = 0

        # --- Global parameters for sharding ---
        total = self._total_samples
        G     = world_size * n_workers
        max_valid    = total - (total % G)           # drop the remainder
        per_subworker = max_valid // G
        
        # Selective files to load
        paths = self.file_paths[:self.num_loadfiles] if self.num_loadfiles else self.file_paths
        
        # Instantiate the data preparer
        preparer   = FastARDataPreparer(self.ar_order, set_name=self.set_name)
        global_idx = 0                # counts *every* sample
        yielded    = 0                # counts only this sub‐worker’s yields
        my_id      = rank * n_workers + worker_id
        
        # print statement to confirm parallelization
        # print(f'[{self.set_name}](world_size-rank-worker_id-total_id)'
        #       f'->{world_size}-{rank}-{worker.id}-{my_id}')
        
        # fixed, local seed across ranks
        seed = self.base_seed if self.split == 'train' else 2345
        rng = np.random.default_rng(seed) 

        # Iterate through assigned files
        for path in paths:
            with h5py.File(path, 'r') as f:
                # load all the fields
                vx       = f["Vx"]       # (S, T, W)
                density  = f["density"]  # (S, T, W)
                pressure = f["pressure"] # (S, T, W) 
                
                # Decide how many sims to use
                n_total = vx.shape[0]
                n_sims = n_total if self.num_sims is None else self.num_sims
                
                # Randomly select indices (every rank will hold random n_sims)
                selected_indices = rng.choice(n_total, n_sims, replace=False)
                
                for start in range(0, n_sims, self.chunk_size):
                    end = min(start + self.chunk_size, n_sims)
                    idx = selected_indices[start:end]
                    idx = np.sort(idx)
                    
                    # chunks
                    vx_chunk       = vx[idx]       
                    density_chunk  = density[idx] 
                    pressure_chunk = pressure[idx] 
                
                    # Stack features along last dim: raw shape (n_sims, T, W, 3_fields)
                    batch = np.stack([vx_chunk, density_chunk, pressure_chunk], axis = -1)
                    
                    # Inflate depth and channel dims → (b, T, 1, 1, W, 1, F)
                    batch = inflate_array(batch, axes=[2, 3, 5])
                    
                    # random shuffling (epoch seed) batch
                    perm = torch.randperm(batch.shape[0], generator=g).numpy()
                    
                    # shuffled batch
                    batch_shuff = batch[perm]
                    
                    # prepare into inputs and targets
                    #print(f'{self.set_name} Preparing the dataset (X(t),X(t+1))...')
                    X, y = preparer.prepare(batch_shuff)

                    for xi, yi in zip(X, y):
                        if global_idx >= max_valid:
                            return   # we’ve exhausted the common pool
    
                        if (global_idx % G) == my_id:
                            # if global_idx < 10:   # debug
                            #     print(f"[{self.set_name}] [first10] gidx={global_idx}->rank={rank}"
                            #           f" worker={worker_id} my_id={my_id}",flush=True)
                            yield xi, yi
                            yielded += 1
                            if yielded >= per_subworker:
                                return
    
                        global_idx += 1
