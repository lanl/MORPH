# src/utils/batched_stream.py
import math
from torch.utils.data import IterableDataset
import torch
import numpy as np

def _to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x

def default_collate_pairs(pairs):
    xs, ys = zip(*pairs)
    xs = [_to_tensor(x) for x in xs]
    ys = [_to_tensor(y) for y in ys]
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

class BatchedStream(IterableDataset):
    """
    Wrap an IterableDataset that yields (x,y) samples and emit minibatches of size `batch_size`.
    No additional sharding here; sharding happens inside the child dataset.
    """
    def __init__(self, dataset, batch_size: int, drop_last: bool = False, collate_fn=default_collate_pairs):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.collate_fn = collate_fn

    def __len__(self):
        n = len(self.dataset)  # global sample count
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)

    def __iter__(self):
        buf = []
        for item in self.dataset:
            buf.append(item)
            if len(buf) == self.batch_size:
                yield self.collate_fn(buf)
                buf.clear()
        if buf and not self.drop_last:
            yield self.collate_fn(buf)
