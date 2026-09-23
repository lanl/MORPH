import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = float(dropout)

    def forward(self, q, k, v, attn_mask=None):
        """
        q,k,v: (B, h, L, d)
        returns: y (B, h, L, d), attn (B, h, L, L) or None
        """
        B, H, L, d = q.shape
        if L == 1:
            return v, None  # attention over one token is identity

        q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
        scale = 1.0 / math.sqrt(d)

        # (B,h,L,L)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask, torch.finfo(scores.dtype).min)
            else:
                scores = scores + attn_mask.to(dtype=scores.dtype, device=scores.device)

        scores = scores - scores.max(dim=-1, keepdim=True).values  # stable softmax
        attn = torch.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout if self.training else 0.0, training=self.training)

        y = torch.matmul(attn, v)  # (B,h,L,d)
        return y, attn