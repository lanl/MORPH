import math, torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    """
    y = x @ W0^T  +  (alpha/r) * [ (x @ A^T) @ B^T ]  + b
    - If rank=0 → behaves like plain Linear (no LoRA params).
    """
    def __init__(self, in_features, out_features, bias=True, rank=0, alpha=None, p=0.0):
        super().__init__()
        # 1) the usual linear (we can freeze its params during finetuning)
        self.base = nn.Linear(in_features, out_features, bias=bias)

        # 2) LoRA config
        self.rank = int(rank)
        if self.rank > 0:
            self.alpha = (alpha if alpha is not None else 2*self.rank)
            self.scaling = self.alpha / self.rank

            # A: compress to r; B: expand back
            self.A = nn.Parameter(torch.zeros(self.rank, in_features))
            self.B = nn.Parameter(torch.zeros(out_features, self.rank))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)

            self.drop = nn.Dropout(p) if p > 0 else nn.Identity()
        else:
            # no LoRA path
            self.alpha = 1.0
            self.scaling = 1.0
            self.register_parameter('A', None)
            self.register_parameter('B', None)
            self.drop = nn.Identity()

    def forward(self, x):
        # base path
        y = self.base(x)

        # low-rank path (only if enabled)
        if self.rank > 0:
            # functional operation: (x @ A^T): [..., r]
            upd_r = F.linear(self.drop(x), self.A)
            # functional operation: (upd_r @ B^T): [..., out_features]
            upd   = F.linear(upd_r, self.B)
            y = y + self.scaling * upd
        return y
