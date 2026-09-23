import torch
import torch.nn as nn

class FieldCrossAttention(nn.Module):
    def __init__(self, patch_dim, heads=4, dropout=0.):
        super().__init__()
        # we attend over sequence-length = C, embed-dim = patch_dim
        self.attn = nn.MultiheadAttention(
            embed_dim=patch_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )
        # learnable “query” vector per patch
        self.q = nn.Parameter(torch.randn(1, 1, patch_dim))

    def forward(self, x):
        # x: (BatchSeq, F, patch_dim)
        Bn, F, E = x.shape
        #print(f"[FieldCrossAttention] input x shape: {x.shape}")  # (BatchSeq, F, E)
        # expand q to (BatchSeq, 1, E)
        q = self.q.expand(Bn, -1, -1)
        #print(f"[FieldCrossAttention] expanded query q shape: {q.shape}")  # (BatchSeq, 1, E)

        # cross-attn: query=(Bn,1,E), key/value=(Bn,F,E)
        out, attn_weights = self.attn(q, x, x)
        #print(f"[FieldCrossAttention] attn output shape: {out.shape}")  # (BatchSeq, 1, E)
        #print(f"[FieldCrossAttention] attn weights shape: {attn_weights.shape}")  # (BatchSeq, 1, F)

        fused = out.squeeze(1)
        #print(f"[FieldCrossAttention] fused output shape: {fused.shape}")  # (BatchSeq, E)
        return fused