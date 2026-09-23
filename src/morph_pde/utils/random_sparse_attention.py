import math
import torch
import torch.nn as nn

class RandomKNeighborsMHA(nn.Module):
    """
    True sparse MHA: for each token, attend to K randomly chosen keys (plus self).
    Complexity ~ O(B * H * L * K * Dh), like axial when K ≈ D+H+W.

    Args:
        dim:   embedding size C
        heads: number of heads
        k_default: fallback K if not provided at forward()
        attn_dropout, proj_dropout: dropouts
        bias:  add bias to projections
        share_index_across_batch: if True, same neighbors for all items in batch; else per-item
    """
    def __init__(self, dim, heads=8, k_default=64,
                 attn_dropout=0.0, proj_dropout=0.0,
                 bias=False, share_index_across_batch=True):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.k_default = k_default
        self.share_index_across_batch = share_index_across_batch

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    @staticmethod
    def _make_indices(L, K, device, per_batch=None):
        """
        Build random neighbor indices with self included.
        Returns:
            idx: (1, L, K) if per_batch is None, else (B, L, K)
        """
        K = int(min(max(1, K), L))
        # self index in col 0
        self_idx = torch.arange(L, device=device).view(1, L, 1)
        # random others (with replacement; simple & fast)
        if per_batch is None:
            rand_idx = torch.randint(0, L, (1, L, max(K - 1, 0)), device=device)
        else:
            B = per_batch
            self_idx = self_idx.expand(B, -1, -1)
            rand_idx = torch.randint(0, L, (B, L, max(K - 1, 0)), device=device)
        idx = torch.cat([self_idx, rand_idx], dim=-1)  # (..., L, K)
        return idx  # (1 or B, L, K)

    def forward(self, x, k_override=None):
        """
        x: (B, L, C)
        returns: (B, L, C)
        """
        B, L, C = x.shape
        K = int(self.k_default if k_override is None else k_override)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        H, Dh = self.heads, self.head_dim
        q = q.view(B, L, H, Dh).transpose(1, 2)  # (B,H,L,Dh)
        k = k.view(B, L, H, Dh).transpose(1, 2)  # (B,H,L,Dh)
        v = v.view(B, L, H, Dh).transpose(1, 2)  # (B,H,L,Dh)

        # neighbor indices
        if self.share_index_across_batch:
            idx = self._make_indices(L, K, x.device)             # (1,L,K)
            idx_bh = idx.view(1, 1, L, K).expand(B, H, -1, -1)   # (B,H,L,K)
        else:
            idx_b = self._make_indices(L, K, x.device, per_batch=B)  # (B,L,K)
            idx_bh = idx_b.unsqueeze(1).expand(-1, H, -1, -1)        # (B,H,L,K)

        # gather K,V neighbors: (B,H,L,K,Dh)
        # Flatten B and H so we can index in one shot
        k_flat = k.reshape(B * H, L, Dh)            # (BH, L, Dh)
        v_flat = v.reshape(B * H, L, Dh)            # (BH, L, Dh)
        idx_flat = idx_bh.reshape(B * H, L, K)      # (BH, L, K)

        # Build batch indices for advanced indexing
        bh = torch.arange(B * H, device=x.device).view(B * H, 1, 1)  # (BH,1,1)
        bh = bh.expand(B * H, L, K)                                  # (BH, L, K)

        # Gather K neighbors per (b,h,i) along the sequence dimension
        k_n = k_flat[bh, idx_flat, :]               # (BH, L, K, Dh)
        v_n = v_flat[bh, idx_flat, :]               # (BH, L, K, Dh)

        # Restore (B,H,...) shape
        k_n = k_n.view(B, H, L, K, Dh)              # (B,H,L,K,Dh)
        v_n = v_n.view(B, H, L, K, Dh)              # (B,H,L,K,Dh)

        # scaled dot products over only K neighbors
        q = q / math.sqrt(Dh)                                      # (B,H,L,Dh)
        attn = torch.matmul(q.unsqueeze(-2),                        # (B,H,L,1,Dh)
                            k_n.transpose(-1, -2)).squeeze(-2)      # -> (B,H,L,K)

        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn.unsqueeze(-2), v_n).squeeze(-2)     # (B,H,L,Dh)

        out = out.transpose(1, 2).reshape(B, L, C)                  # (B,L,C)
        out = self.proj_drop(self.out_proj(out))
        return out

class RandomSparse3DSpaceTime(nn.Module):
    """
    Random sparse attention over the whole D×H×W volume.
    K is chosen ~ D+H+W-3 to match axial per-token FLOPs.

    Args mirror your axial class where relevant.
    """
    def __init__(self, dim, heads, dropout=0.0, k_default=64, share_index_across_batch=True):
        super().__init__()
        self.mha = RandomKNeighborsMHA(dim, heads, k_default,
                                       attn_dropout=dropout, proj_dropout=dropout,
                                       share_index_across_batch=share_index_across_batch)

    def forward(self, x, grid_size):
        # x: (B, t, N, C), grid_size=(D,H,W), N=D*H*W
        B, t, N, C = x.shape
        D, H, W = grid_size
        assert N == D * H * W, "N must equal D*H*W"

        # Choose K ~ axial's neighbor count per token
        K = min(D + H + W - 3, N)

        x_flat = x.reshape(B * t, N, C)       # (B*t, N, C)
        y_flat = self.mha(x_flat, k_override=K)
        y = y_flat.view(B, t, N, C)
        return y
