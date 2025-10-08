import itertools
from torch.utils.data import IterableDataset, get_worker_info
import random
import torch.distributed as dist                      # add dist
from itertools import islice
from src.utils.main_process_ddp import is_main_process

class MultiSourceIterableDataset:
    @staticmethod
    def ConcatStreams(*streams):
        """
        Chains each stream end-to-end: A then B then C...
        Usage: DataLoader(MultiSourceIterableDataset.ConcatStreams(ds1, ds2), ...)
        """
        class _Concat(IterableDataset):
            def __init__(self, streams):
                self.streams = streams
            def __iter__(self):
                return itertools.chain(*(iter(s) for s in self.streams))
            def __len__(self):
                return sum(len(s) for s in self.streams)

        return _Concat(streams)

    @staticmethod
    def RoundRobinStreams(*streams):
        """
        Interleaves samples round-robin: A, B, A, B...
        Usage: DataLoader(MultiSourceIterableDataset.RoundRobinStreams(ds1, ds2), ...)
        """
        class _RoundRobin(IterableDataset):
            def __init__(self, streams):
                self.streams = streams
            def __iter__(self):
                its = [iter(s) for s in self.streams]
                while its:
                    for it in its.copy():
                        try:
                            yield next(it)
                        except StopIteration:
                            its.remove(it)
            def __len__(self):
                return sum(len(s) for s in self.streams)

        return _RoundRobin(streams)
    
class WeightedMultiSourceIterableDataset_1(IterableDataset):
    """
    Mix N IterableDatasets by sampling according to weights.
    Usage: WeightedMultiSourceIterableDataset(ds1, ds2, ..., weights=[w1, w2, ...])
    """
    def __init__(self, *streams, weights=None):
        assert streams, "Need at least one stream"
        self.streams = list(streams)
        n = len(self.streams)
        if weights is None:
            # uniform mixing
            self.weights = [1.0/n] * n
        else:
            assert len(weights) == n, "Must give one weight per stream"
            total = sum(weights)
            self.weights = [w/total for w in weights]
        
    def __iter__(self):
        iters = [iter(s) for s in self.streams] # list of __iter__ of datasets
        weights = list(self.weights) # list of all sampling probs
        while iters:
            # choose one dataloader
            idx = random.choices(range(len(iters)), weights, k=1)[0]
            try:
                yield next(iters[idx])
            except StopIteration:
                iters.pop(idx)
                weights.pop(idx)
                if not iters:
                    break
                # re-normalize remaining weights
                total = sum(weights)
                weights = [w/total for w in weights]

class WeightedMultiSourceIterableDataset_2(IterableDataset):
    def __init__(self, *streams, weights=None, seed: int = 1234):
        assert streams, "Need at least one stream"
        self.streams = list(streams)
        self.base_seed = seed
        self.epoch = 0
        n = len(self.streams)
        if weights is None:
            self.weights = [1.0/n] * n
        else:
            assert len(weights) == n
            total = sum(weights)
            self.weights = [w/total for w in weights]
    
    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        
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
                                                               
        # Global worker id across (rank × num_workers)
        global_rank   = rank * n_workers + worker_id
        total_workers = world_size * n_workers
        # print(f"global rank {global_rank}, total workers {total_workers}")
        
        # ——— 3) Build the infinite mixer ———
        iters   = [iter(s) for s in self.streams]
        weights = list(self.weights)
        
        # 3) mixer with deterministic RNG per (epoch, rank) so all ranks
        #     *choose the same stream indices in the same order*
        rng = random.Random(self.base_seed + 1000*self.epoch)  # do NOT add rank here (we want same stream-idx sequence on all ranks)
        
        def mixer():
            while iters:
                idx = rng.choices(range(len(iters)), weights, k=1)[0]
                try:
                    yield next(iters[idx])
                except StopIteration:
                    iters.pop(idx)
                    weights.pop(idx)
                    if iters:
                        tot = sum(weights)
                        for i in range(len(weights)):
                            weights[i] /= tot

        # ——— 4) How many total_batches can we evenly shard? ———
        total_batches = sum(len(s) for s in self.streams)
        valid         = total_batches - (total_batches % total_workers)
        per_worker    = valid // total_workers
        
        # if rank == 0 and worker_id == 0:
        #     print(f"Total_batches={total_batches}, valid={valid}, per_worker={per_worker}")

        # ——— 5) Round-robin slice so each (rank,worker) sees exactly per_worker ———
        mixed = mixer()
        seen = 0
        for idx, batch in enumerate(itertools.islice(mixed, valid)):
            if idx % total_workers == global_rank:
                yield batch
                seen += 1
                if seen >= per_worker:
                    break


                
            
