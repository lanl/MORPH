import torch
import torch.nn as nn
from src.utils.embedding_conv_patch_xatt_project import HybridPatchEmbedding3D
from src.utils.positional_encoding_spatiotemporal_li_slice import PositionalEncoding_SLin_TSlice
from src.utils.positional_encoding_spatiotemporal_bilinear import PositionalEncoding_STBilinear
from src.utils.transformer_encoder_axialattention_3dspacetime_lora import EncoderBlock
from src.utils.simple_decoder import SimpleDecoder

class ViT3DRegression(nn.Module):
    def __init__(self,
                 patch_size,
                 dim, depth,
                 heads, heads_xa, mlp_dim,
                 max_components = 3,
                 conv_filter = 32,
                 max_ar = 5,
                 max_patches = 512,
                 max_fields = 3,
                 dropout = 0.1,
                 emb_dropout = 0.1,
                 lora_r_attn = 0,
                 lora_r_mlp = 0,
                 lora_alpha = None,
                 lora_p = 0.0,
                 model_size = 'Ti',
                 activated_ar1k = False):
        super().__init__()
        
        # patch_size to be int or tuple
        if isinstance(patch_size, (tuple, list)):
            pD, pH, pW = patch_size
        else:
            pD = pH = pW = patch_size
               
        self.patch_size = (pD, pH, pW)
        self.max_ar      = max_ar
        self.max_components  = max_components
        self.max_patches = max_patches
        self.max_fields = max_fields
        max_patch_vol = pW**3
        max_decoder_out_ch = max_fields * max_components * max_patch_vol
        
        # Patch embedding on conv layers
        self.patch_embedding = HybridPatchEmbedding3D(patch_size=self.patch_size,
                               max_components = max_components, conv_filter=conv_filter,
                               embed_dim=dim, heads_xa=heads_xa)

        # Positional embedding (time slices, spatial linear interpolation)
        if model_size == 'L' and max_ar > 1:
            self.pos_encoding = PositionalEncoding_STBilinear(max_ar = self.max_ar,
                                max_patches = self.max_patches, dim = dim, 
                                emb_dropout = emb_dropout)
        
        else:
            self.pos_encoding = PositionalEncoding_SLin_TSlice(max_ar = self.max_ar,
                                max_patches = self.max_patches, dim = dim, 
                                emb_dropout = emb_dropout)
            
        self.dropout   = nn.Dropout(emb_dropout)

        # Transformer backbone with correct grid
        self.transformer_blocks = nn.ModuleList([
                                  EncoderBlock(dim, heads, mlp_dim, dropout,
                                               lora_r_attn = lora_r_attn, # PRETRAIN: keep 0 (dormant)
                                               lora_r_mlp = lora_r_mlp, # PRETRAIN: keep 0 (dormant)
                                               lora_alpha = lora_alpha, # PRETRAIN: keep None
                                               lora_p = lora_p,         # PRETRAIN: dropout
                                               activated_ar1k = activated_ar1k)         
                                  for _ in range(depth)
                                  ])
        
        # Decoder back to fields*components*patch_vol
        self.decoder = SimpleDecoder(dim, max_decoder_out_ch)
    
    def get_patch_info(self, volume):
        D, H, W = volume
        (pD, pH, pW) = self.patch_size
        # if any axis size == 1, use patch size 1 on that axis;
        # otherwise keep the original hyperparam
        pD = 1 if D == 1 else pD
        pH = 1 if H == 1 else pH
        pW = 1 if W == 1 else pW
        patch_sizes = (pD, pH, pW)
        
        # ensure divisibility per axis
        assert D % pD == 0 and H % pH == 0 and W % pW == 0, \
               "Each axis must be divisible by its patch_size"
               
        D_patches = D // pD
        H_patches = H // pH
        W_patches = W // pW
        n_patches = (D_patches, H_patches, W_patches)
        return patch_sizes, n_patches
        
    def forward(self, vol):
        # vol: (B, t, F, C, D, H, W)
        B, t, F, C, D, H, W = vol.shape
        #print(f"[ViT] input vol shape: {vol.shape}")
        
        #assert C == self.components, "expected C equal to your model’s components"
        
        # 1) patch-embed → (B, t, n, dim)
        x = self.patch_embedding(vol)
        #print(f"[ViT] patch embedding: {x.shape}")

        # 2) add time-slice pos-emb
        x = x + self.pos_encoding(x)  # (1, t, n, dim)
        #print(f"[ViT] after pos embedding: {x.shape}")
        x = self.dropout(x)

        # 3) transformer → (B, t, n, dim)
        ''' Adapt transformer to new grid (axial attentions) '''
        (pD, pH, pW), (D_patches, H_patches, W_patches) = self.get_patch_info((D, H, W))
        grid_size = (D_patches, H_patches, W_patches)
        patch_vol = pD * pH * pW
        for blk in self.transformer_blocks:
            x = blk(x, grid_size)
        # print(f"[ViT] after transformer: {x.shape}")

        # 4) decode → (B, t, n, out_ch*patch³)
        x = self.decoder(x, F, C, patch_vol)
        # print(f"[ViT] after decoder: {x.shape}")

        # 5) Reshape into component + field volume
        b, t, n, cpd = x.shape
        assert cpd == F * C * patch_vol, \
                f"expected {F*C*patch_vol}, got {cpd}"

        x_last = x[:, -1, :, :]
        
        # (b, n, fields, components, pD, pH, pW)
        x_last = x_last.view(b, n, F, C, pD, pH, pW)
        
        # reshape to patch grid
        x_last = x_last.reshape(b, D_patches, H_patches, W_patches, F, C, pD, pH, pW)
        #print(f"[ViT] Reshape to patch grid: {x_last.shape}")
        
        # reorder to (B, fields, components, D_p, p, H_p, p, W_p, p)
        x_last = x_last.permute(0, 4, 5, 1, 6, 2, 7, 3, 8).contiguous()
        
        # final reshape to (B, fields, components, D, H, W)
        x_last = x_last.view(b, F, C, D, H, W)
        
        # print(f"[ViT] Final shape: {x_last.shape}")
        return x_last
