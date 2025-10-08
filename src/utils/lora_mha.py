# in src/utils/lora_mha.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.lora_linear import LoRALinear  # adjust import to your layout
from src.utils.sdpa import ScaledDotProductAttention

class LoRAMHA(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, rank=0, alpha=None, p=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.dropout    = dropout

        # Projections; LoRA is dormant if rank=0
        self.q = LoRALinear(embed_dim, embed_dim, rank=rank, alpha=alpha, p=p)
        self.k = LoRALinear(embed_dim, embed_dim, rank=rank, alpha=alpha, p=p)
        self.v = LoRALinear(embed_dim, embed_dim, rank=rank, alpha=alpha, p=p)
        self.o = LoRALinear(embed_dim, embed_dim, rank=rank, alpha=alpha, p=p)

        # manual attention module (replaces SDPA)
        self.sdpa = ScaledDotProductAttention(dropout=dropout)
        
    def forward(self, *inputs):
        """
        Drop-in signatures:
          - y, _ = attn(x)                 # self-attention
          - y, _ = attn(q, k, v)           # explicit qkv
        Shapes: (B, L, C)
        """
        if len(inputs) == 1:
            q_in = k_in = v_in = inputs[0]
        elif len(inputs) == 3:
            q_in, k_in, v_in = inputs
        else:
            raise TypeError(f"LoRAMHA.forward expected 1 or 3 inputs, got {len(inputs)}")

        B, L, C = q_in.shape

        # project
        q = self.q(q_in).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B,h,L,d)
        k = self.k(k_in).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(v_in).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # SDPA (masks optional; add if you actually use them)
        y, _ = self.sdpa(q, k, v)  # (B,h,L,d)

        # merge heads
        y = y.transpose(1, 2).contiguous().view(B, L, C)
        y = self.o(y)

        # match nn.MultiheadAttention return type: (output, attn_weights)
        return y, None
