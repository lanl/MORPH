import torch
import torch.nn as nn
from src.utils.patchify_3d import custom_patchify_3d
from src.utils.crossattention_fields import FieldCrossAttention
from src.utils.convolutional_operator import ConvOperator

class HybridPatchEmbedding3D(nn.Module):
    def __init__(self, patch_size, max_components, conv_filter, embed_dim, heads_xa):
        super().__init__()
        self.patch_size  = patch_size
        self.embed_dim   = embed_dim
        self.conv_filter = conv_filter
        self.heads_xa = heads_xa
        self.max_components = max_components
        
        pD, pH, pW = patch_size
        self.patch_size = (pD, pH, pW)
        
        # calculate max features for zero padding the features
        self.max_patch_vol = pW ** 3  # pW is choosen (in 1d datasets, alteast we will have pW) (8**3 * filters)
        self.max_features = self.max_patch_vol * self.conv_filter
        
        # conv_features takes C channels in, outputs 'filters' channels
        assert conv_filter % 8 == 0, "conv_filter must be divisible by 8"
        
        self.conv_features = ConvOperator(max_components, conv_filter)

        # project each patch-feature into embed_dim
        self.projection  = nn.Linear(self.max_features, embed_dim)
         
        # cross-attend in the smaller embed_dim space
        self.field_attn  = FieldCrossAttention(embed_dim, heads = heads_xa)

    def forward(self, x):
        B, t, F, C, D, H, W = x.shape
        # print(f"[Input]      x shape: {x.shape}")  # (B,t,F,C,D,H,W)

        # 1) merge batch, time, & fields C into one batch axis for conv
        x = x.view(B * t * F, C, D, H, W)
        # print(f"[Step1] merge B,t,F → batch: {x.shape}")  # (B*t*F, C,D,H,W)
        x = self.conv_features(x)
        #print("[Step1] Performed convolutional operations on components...")
        #print(f"[Step1] after conv_features: {x.shape}")  # (B*t*F, filters,D,H,W)

        # 2) patchify → (B*t*F, n, filters*patch_vol)
        x = custom_patchify_3d(x, self.patch_size)
        #print(f"[Step2] after patchify: {x.shape}")  # (B*t*F, n, filters*p^3)
        n_patches, features = x.shape[1], x.shape[2]
        
        # reshape to separate combined batch and n
        x = x.view(B * t, F, n_patches, features)
        #print(f"[Step2] separate F and patches: {x.shape}")  # (B*t, F, n, feat)
        
        # bring patches to front
        x = x.permute(0, 2, 1, 3)
        #print(f"[Step2] permute→ (B*t, n, F, feat): {x.shape}")
        
        # 3a) pad the features to max features
        x = x.reshape(-1, F, features)  # -> (B*t*n, F, feat)
        #print(f"[Step3] reshape (B*t*n, F, feat): {x.shape}")
        
        # if this modality has fewer than max_features, pad with zeros on the last dim
        if features < self.max_features:
            pad_amt = self.max_features - features
            # pad tensor of shape (B*t*n, F, pad_amt)
            pad = x.new_zeros(x.size(0), x.size(1), pad_amt)
            x = torch.cat([x, pad], dim=-1)
            #print(f"[Step3a] after zero padding: {x.shape}")
        
        # 3b) project each field's patch-feature to embed_dim
        x = self.projection(x)                                # -> (Bt*n, F, embed_dim)
        #print(f"[Step3b] after projection: {x.shape}")  # (B*t, n, embed_dim)
        
        # 4) collapse fields via cross-attn in embed space
        x = self.field_attn(x)
        #print(f"[Step4] after field_attn: {x.shape}")  # (B*t*n, embed_dim)
        
        # restore patch and batch dims
        x = x.view(B * t, n_patches, self.embed_dim)   # -> (Bt, n, embed_dim)
        #print(f"[Step4] restore dims: {x.shape}")  # (B*t, n, embed_dim)

        # 5) reshape back to (B, t, n, embed_dim)
        x = x.view(B, t, n_patches, self.embed_dim)
        # print(f"[Output]     final shape: {x.shape}")  # (B, t, n, embed_dim)
        return x