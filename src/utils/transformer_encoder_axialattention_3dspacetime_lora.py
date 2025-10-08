# src/utils/transformer_encoder_axialattention_3dspacetime.py
import torch.nn as nn
from src.utils.axial_attention_3dspacetime_2_lora import AxialAttention3DSpaceTime
from src.utils.lora_linear import LoRALinear

class EncoderBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.,
                 lora_r_attn: int = 0, lora_r_mlp: int = 0, 
                 lora_alpha: int = None, lora_p: float = 0.0,
                 activated_ar1k: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.axial_attn = AxialAttention3DSpaceTime(dim, heads, dropout,
                          lora_r_attn = lora_r_attn, lora_alpha = lora_alpha, 
                          lora_p = lora_p, activated_ar1k = activated_ar1k)
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP with LoRA linears (dormant if lora_r_mlp=0)
        self.mlp = nn.Sequential(
            LoRALinear(dim, mlp_dim, rank=lora_r_mlp, alpha=lora_alpha, p=lora_p),
            nn.GELU(),
            nn.Dropout(dropout),
            LoRALinear(mlp_dim, dim, rank=lora_r_mlp, alpha=lora_alpha, p=lora_p),
            nn.Dropout(dropout)
        )

    def forward(self, x, grid_size):
        # Axial attention block
        residual = x
        #print('Shape before norm and attentions', x.shape)
        x = self.norm1(x)
        #print('Shape after layer norm', x.shape)
        x_attn = self.axial_attn(x, grid_size)
        #print('Shape after axial attention', x_attn.shape)
        x = residual + x_attn
        #print('Shape after residual + x_attn', x.shape)

        # MLP block
        residual = x
        x = self.norm2(x)
        #print('Shape after second layer norm', x.shape)
        x = self.mlp(x)
        #print('Shape after MLP', x.shape)
        x = residual + x
        return x
