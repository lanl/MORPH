import torch
import torch.nn as nn
import math

class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, max_ar: int, max_patches: int, dim: int, emb_dropout: float = 0.0):
        super().__init__()
        self.max_ar = max_ar
        self.max_patches = max_patches
        self.dim = dim
        self.dropout = nn.Dropout(emb_dropout)

    @staticmethod
    def _sinusoidal_table(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / dim)
        )

        table = torch.zeros(length, dim, device=device, dtype=torch.float32)
        table[:, 0::2] = torch.sin(positions * div_term)

        cos_dim = table[:, 1::2].shape[1]
        if cos_dim > 0:
            table[:, 1::2] = torch.cos(positions * div_term[:cos_dim])

        return table.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, num_patches, dim = x.shape

        if dim != self.dim:
            raise ValueError(f"Expected embedding dim {self.dim}, got {dim}")
        if time_steps > self.max_ar:
            raise ValueError(f"Input AR {time_steps} exceeds configured max_ar {self.max_ar}")
        if num_patches > self.max_patches:
            raise ValueError(
                f"Input patch count {num_patches} exceeds configured max_patches {self.max_patches}"
            )

        time_pe = self._sinusoidal_table(time_steps, dim, x.device, x.dtype).unsqueeze(1)
        patch_pe = self._sinusoidal_table(num_patches, dim, x.device, x.dtype).unsqueeze(0)
        pe = (time_pe + patch_pe).unsqueeze(0).expand(batch_size, -1, -1, -1)
        return self.dropout(pe)