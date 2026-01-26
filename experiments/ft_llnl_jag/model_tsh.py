import torch
import torch.nn as nn

class TaskSpecificHead_FC(nn.Module):
    def __init__(self, n_patches=64, feat_dim=256, output_dim=5, scalar_dim=15, dropout_p=0.1):
        super().__init__()
        self.n_patches = n_patches
        self.feat_dim = feat_dim

        self.fc1 = nn.Linear(n_patches * feat_dim, feat_dim * 8)
        self.fc2 = nn.Linear(feat_dim * 8, feat_dim * 4)
        self.fc3 = nn.Linear(feat_dim * 4, feat_dim * 2)
        self.fc4 = nn.Linear(feat_dim * 2, feat_dim)

        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout_p)

        # separate norms for z and s
        self.norm_z = nn.LayerNorm(n_patches * feat_dim)
        self.norm_s = nn.LayerNorm(scalar_dim)

        self.fc_branch   = nn.Linear(scalar_dim, feat_dim)
        self.fc_combined = nn.Linear(feat_dim * 2, output_dim)

    def forward(self, z, s):
        B = z.shape[0]

        # safer than view+contiguous
        z = z.reshape(B, self.n_patches * self.feat_dim)
        z = self.norm_z(z)
        z = self.drop(self.act(self.fc1(z)))
        z = self.drop(self.act(self.fc2(z)))
        z = self.drop(self.act(self.fc3(z)))
        z = self.drop(self.act(self.fc4(z)))

        s = s.reshape(B, -1)          # expect (B, scalar_dim)
        s = self.norm_s(s)
        s = self.drop(self.act(self.fc_branch(s)))

        combined = torch.cat((z, s), dim=1)   # (B, feat_dim*2)
        out = self.fc_combined(combined)
        return out