import random
from torch.utils.data import IterableDataset
           
class WeightedMultiSourceIterableDataset(IterableDataset):
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
        iters = [iter(s) for s in self.streams]
        weights = list(self.weights)
        while iters:
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

