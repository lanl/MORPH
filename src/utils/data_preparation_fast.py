import torch
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class FastARDataPreparer:
    """
    Prepare 3D autoregressive data:
      - inputs:  sequences of length ar_order
      - targets: the next frame
    """

    def __init__(self, ar_order: int, set_name: str = ''):
        self.ar_order = ar_order
        self.set_name = set_name
    
    def prepare(self, all_data: np.ndarray):
        """
        Vectorized version of sliding‐window AR dataset creation.
        all_data: np.array of shape (n_s, T, D, H, W, C, F)
        returns: (inputs, targets) as torch.Tensors
          inputs shape (N, ar_order, F, C, D, H, W)
          targets shape (N,       , F, C, D, H, W)
        """
        all_data = np.ascontiguousarray(all_data, dtype=np.float32)
        n_s, T, D, H, W, C, F = all_data.shape
        k = self.ar_order
        
        # 1) use numpy’s sliding_window_view to get windows along time
        #    shape → (n_s, T-k+1, k, D, H, W, C, F)
        windows = sliding_window_view(all_data, window_shape=k, axis=1)
        
        # 2) inputs are the first k frames of each window
        #    flatten (n_s, T-k+1) → N = n_s*(T-k)
        N = n_s * (T - k)
        inp_np = windows[:, :T-k, ...].reshape(N, k, D, H, W, C, F)

        # 3) targets are frame t+k
        tgt_np = all_data[:, k:, ...].reshape(N, D, H, W, C, F)

        # 4) reorder axes seq→[N, k, F, C, D, H, W], tgt→[N, F, C, D, H, W]
        inp_np = inp_np.transpose(0,1,6,5,2,3,4)
        tgt_np = tgt_np.transpose(0,5,4,1,2,3)

        # 5) single zero‐copy conversion into torch
        inp_np = inp_np.copy()      # make writable
        tgt_np = tgt_np.copy()

        inp  = torch.from_numpy(inp_np)
        tgt  = torch.from_numpy(tgt_np)
        
        return inp, tgt

