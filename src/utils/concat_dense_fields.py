import torch
import torch.nn as nn

class FieldConcatProjection(nn.Module):
    """
    Concat fields along channel, zero-pad to max_fields, then project:
      (B, F, E) --pad/truncate--> (B, max_fields, E) --reshape--> (B, max_fields*E)
      --linear--> (B, out_dim)  where out_dim defaults to E.

    Args:
        patch_dim   (int): E, the per-field embedding size.
        max_fields  (int): maximum F you’ll ever see (default 3).
        out_dim     (int): output size; defaults to patch_dim to match your cross-attn.
        dropout     (float): dropout before the projection.
        trim_if_excess (bool): if F>max_fields, trim extra fields (True) or raise (False).
    """
    def __init__(self, patch_dim, dropout=0.0, max_fields=3):
        super().__init__()
        self.patch_dim = patch_dim
        self.max_fields = max_fields
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(patch_dim * max_fields, 1 * patch_dim) 

    def forward(self, x):
        # x: (B, F, E)
        B, F, E = x.shape

        # handle F relative to max_fields
        if F < self.max_fields:
            pad = x.new_zeros(B, self.max_fields - F, E)  # dtype/device-safe
            x = torch.cat([x, pad], dim=1)  # (B, max_fields, E)

        # flatten fields and project
        x = x.reshape(B, self.max_fields * E)   # (B, max_fields*E)
        x = self.dropout(x)
        fused = self.projection(x)              # (B, E)
        return fused