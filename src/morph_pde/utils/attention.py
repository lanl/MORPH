import torch
import torch.nn as nn
import torch.nn.functional as F

# Attention mechanism
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        
        # Get query, key, value from input
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).permute(0, 2, 1, 3), qkv)
        
        # Dot product attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        
        # Multiply attention with values
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, n, -1)
        
        return self.to_out(out)