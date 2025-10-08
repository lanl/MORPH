import torch
import torch.nn as nn
from src.utils.lora_mha import LoRAMHA

class AxialAttention3DSpaceTime(nn.Module):
    def __init__(self, dim, heads, dropout=0., lora_r_attn: int = 0, lora_alpha: int = None, 
                 lora_p: float = 0.0, activated_ar1k: bool = False):
        super().__init__()
        self.activated_ar1k = activated_ar1k
        # time-axis
        self.attn_t = LoRAMHA(dim, heads, dropout=dropout, rank=lora_r_attn, alpha=lora_alpha, p=lora_p)
        # spatial axes
        self.attn_d = LoRAMHA(dim, heads, dropout=dropout, rank=lora_r_attn, alpha=lora_alpha, p=lora_p)
        self.attn_h = LoRAMHA(dim, heads, dropout=dropout, rank=lora_r_attn, alpha=lora_alpha, p=lora_p)
        self.attn_w = LoRAMHA(dim, heads, dropout=dropout, rank=lora_r_attn, alpha=lora_alpha, p=lora_p)

    def forward(self, x, grid_size):
        B, t, N, features = x.shape
        D, H, W = grid_size
        # assert N == D * H * W, f"token count N={N} != D*H*W={D*H*W}"
        
        # now reconstruct the 3D grid of tokens
        x = x.view(B, t, D, H, W, features) 
        
        if t > 1: # update weights only for AR(p) > AR(1)
            # —— 1) time-axis attention ——
            xt = x.permute(0,2,3,4,1,5)                             # (B, D, H, W, t, features)
            xt = xt.reshape(B*D*H*W, t, features)                   # (B·D·H·W, t, features)
            xt_attn, _ = self.attn_t(xt, xt, xt)                    # (B·D·H·W, t, features)
            xt = xt_attn.reshape(B, D, H, W, t, features).permute(0,4,1,2,3,5) # (B, t, D, H, W, features)
    
            # add residual
            x = x + xt

        # —— 2) spatial axes (with time folded into batch) ——
        # depth-axis
        xd = x.permute(0,1,3,4,2,5)                             # (B, t, H, W, D, features)
        xd = xd.reshape(B*t*H*W, D, features)                   # (B·t·H·W, D, features)
        xd_attn, _ = self.attn_d(xd, xd, xd)                    # (B·t·H·W, D, features)
        xd = xd_attn.reshape(B, t, H, W, D, features).permute(0,1,4,2,3,5) # (B, t, D, H, W, features)

        # height-axis
        xh = x.permute(0,1,2,4,3,5)                  # (B, t, D, W, H, features)
        xh = xh.reshape(B*t*D*W, H, features)        # (B·t·D·W, H, features)
        xh_attn, _ = self.attn_h(xh, xh, xh)         # (B·t·D·W, H, features)
        xh = xh_attn.reshape(B, t, D, W, H, features).permute(0,1,2,4,3,5) # (B, t, D, H, W, features)

        # width-axis
        xw = x.reshape(B*t*D*H, W, features)         # (B·t·D·H, W, features)
        xw_attn, _ = self.attn_w(xw, xw, xw)         # (B·t·D·H, W, features)
        xw = xw_attn.reshape(B, t, D, H, W, features)# (B, t, D, H, W, features)

        # —— 3) sum and flatten back to tokens ——
        x_comb = x + xd + xh + xw                    # (B, t, D, H, W, features)
        out = x_comb.view(B, t, D*H*W, features)     # (B, t*D*H*W, features)
        return out
    
    def _spatial_only(self, x, grid_size):
        return self.forward(x, grid_size)
