import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding_SLin_TSlice(nn.Module):
    """
    Learned absolute pos-emb table of shape (1, max_ar, max_patches, dim).
    At forward time, we:
      - slice the first t time-steps,
      - linearly interpolate from max_patches → n_patches (no time interp),
      - broadcast to batch.
    """

    def __init__(self, max_ar: int, max_patches: int, dim: int, emb_dropout: float = 0.0):
        super().__init__()
        # (1, max_ar, max_patches, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_ar, max_patches, dim))
        self.dropout       = nn.Dropout(emb_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: Tensor of shape (B, t, n_patches, dim)
        Returns:
          pe: Tensor of shape (B, t, n_patches, dim)
        """
        B, t, n, D = x.shape

        # slice time (depends on the AR)
        pe = self.pos_embedding[:, :t, :, :]       # (1, t, max_patches, dim)

        # move dim→channel and merge time into channels so we can do 1D interp
        # over the patch axis only
        # shape → (1, dim*t, max_patches)
        pe = pe.permute(0, 3, 1, 2).reshape(1, D*t, -1)

        # linear interpolate (1D) to n_patches. (1, D*t, n_patches)
        pe = F.interpolate(pe, size = n, mode='linear', align_corners=False)   

        # reshape & permute back to (1, t, n_patches, dim)
        pe = pe.reshape(1, D, t, n).permute(0, 2, 3, 1)
        # print(f"reshape and permute shape:{pe.shape}")
        
        # broadcast to batch
        pe = pe.expand(B, -1, -1, -1)
        # print(f"broadcast shape:{pe.shape}")
        
        # add dropout
        pe = self.dropout(pe)
        
        return pe