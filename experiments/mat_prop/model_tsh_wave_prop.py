import torch
import torch.nn as nn

class TaskSpecificHead_FC(nn.Module):
    def __init__(
        self,
        n_patches=64,
        feat_dim=256,
        output_dim=5,
        conv_channels=256,
        dropout_p=0.05
    ):
        super().__init__()
        self.n_patches = n_patches
        self.feat_dim = feat_dim
        self.c1 = conv_channels
        self.c2 = conv_channels // 4

        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout_p)

        # Normalize each patch embedding
        self.norm_z_patch = nn.LayerNorm(feat_dim)

        # Conv stack over patches
        self.conv1 = nn.Conv1d(
            in_channels=feat_dim,
            out_channels=self.c1,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=self.c1,
            out_channels=self.c2,
            kernel_size=3,
            padding=1
        )

        # Norm after conv stack (must match conv2 out_channels)
        self.norm_z_conv = nn.LayerNorm(self.c2)

        # MLP after conv+flatten (input must match n_patches * c2)
        self.fc1 = nn.Linear(n_patches * self.c2, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, z):
        B = z.shape[0]

        # z branch
        z = self.norm_z_patch(z)         # (B, n_patches, feat_dim)
        z = z.transpose(1, 2)            # (B, feat_dim, n_patches)

        z = self.drop(self.act(self.conv1(z)))  # (B, c1, n_patches)
        z = self.drop(self.act(self.conv2(z)))  # (B, c2, n_patches)

        z = z.transpose(1, 2)            # (B, n_patches, c2)
        z = self.norm_z_conv(z)          # (B, n_patches, c2)
        z = z.reshape(B, self.n_patches * self.c2) # (B, n_patches * c2)

        z = self.drop(self.act(self.fc1(z)))
        out = self.fc2(z)    # (B, output_dim)

        return out
