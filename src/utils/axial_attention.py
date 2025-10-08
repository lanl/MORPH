import torch
import torch.nn as nn

class AxialAttention3D(nn.Module):
    """
    Axial attention for 3D volumes along depth, height, and width axes.
    Expects input of shape (B, N, C) where N = D*H*W and D=H=W.
    """
    def __init__(self, dim, heads, dropout=0.):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.attn_d = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.attn_h = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.attn_w = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        # x: (B, N, C)
        B, N, C = x.shape
        # assume cubic grid
        D = round(N ** (1/3))
        assert D**3 == N, f"Number of patches ({N}) is not a perfect cube"
        H = W = D
        # reshape to (B, D, H, W, C)
        x_grid = x.view(B, D, H, W, C)
        #print("Reshape data", x_grid.shape)

        # depth-axis attention (along D)
        x_d = x_grid.permute(0,2,3,1,4)
        #print("Permute for depth attention", x_d.shape)
        x_d = x_d.reshape(B*H*W, D, C)
        #print("Reshape for depth attention", x_d.shape)
        x_d_attn, _ = self.attn_d(x_d, x_d, x_d)
        #print("depth attention", x_d_attn.shape)
        x_d_out = x_d_attn.view(B, H, W, D, C).permute(0,3,1,2,4)
        #print("depth attention permute", x_d_out.shape)

        # height-axis attention (along H)
        x_h = x_grid.permute(0,1,3,2,4)
        #print("Permute for height attention", x_h.shape)
        x_h = x_h.reshape(B*D*W, H, C)
        #print("Reshape for height attention", x_h.shape)
        x_h_attn, _ = self.attn_h(x_h, x_h, x_h)
        #print("height attention", x_h_attn.shape)
        x_h_out = x_h_attn.view(B, D, W, H, C).permute(0,1,3,2,4)
        #print("height attention permute", x_h_out.shape)

        # width-axis attention (along W)
        x_w = x_grid.reshape(B*D*H, W, C)
        #print("Reshape for width attention", x_w.shape)
        x_w_attn, _ = self.attn_w(x_w, x_w, x_w)
        #print("width attention", x_w_attn.shape)
        x_w_out = x_w_attn.view(B, D, H, W, C)
        #print("width attention", x_w_out.shape)

        # combine axial outputs
        x_comb = x_grid + x_d_out + x_h_out + x_w_out
        #print("Combined attentions", x_comb.shape)
        x_comb = x_comb.view(B, N, C)
        #print("Combined attentions reshaped", x_comb.shape)
        
        return x_comb