import torch
import torch.nn as nn

class SimpleDecoder(nn.Module):
    def __init__(self, dim, max_out_ch):
        super().__init__()
        self.max_out_ch  = max_out_ch
        self.norm        = nn.LayerNorm(dim)
        self.linear      = nn.Linear(dim, max_out_ch)

    def forward(self, x, fields, components, patch_vol):
        out_ch = fields * patch_vol * components
        x = self.norm(x)
        x = self.linear(x)                     # → (B,t,n,max_out_ch)
        if out_ch < self.max_out_ch:
            x = x[..., :out_ch]                # → (B,t,n,out_ch)
        return x
