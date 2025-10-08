import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding_STBilinear(nn.Module):
    """
    Learned absolute pos-emb table of shape (1, max_ar, max_patches, dim).
    At forward time, we bilinear-interpolate from (max_ar, max_patches) → (t, n_patches),
    then broadcast to batch.
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

        # DO NOT slice time; we will resample both time and patches together
        # pe: (1, max_ar, max_patches, dim) → (1, D, max_ar, max_patches)
        pe = self.pos_embedding.permute(0, 3, 1, 2)

        # Bilinear interpolate over (time, patches) from (max_ar, max_patches) → (current_ar, n_patches)
        # shape stays (1, D, current_ar, n)
        pe = F.interpolate(pe, size=(t, n), mode='bilinear', align_corners=False, antialias=True)

        # Back to (1, current_ar, n_patches, dim)
        pe = pe.permute(0, 2, 3, 1)
        # print(f"reshape and permute shape:{pe.shape}")
        
        # broadcast to batch
        pe = pe.expand(B, -1, -1, -1)
        # print(f"broadcast shape:{pe.shape}")
        
        # Broadcast to batch and apply dropout
        pe = self.dropout(pe)
        return pe
