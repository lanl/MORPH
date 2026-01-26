# Standalone model
import torch
import torch.nn as nn

class CNN2D(nn.Module):
    def __init__(self, in_channels = 4, final_out_channels = 256, scalar_dim = 15, output_dim=5):
        super(CNN2D, self).__init__()
        # 64x64 input with 4 channels
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)   # -> 32 x 64 x 64
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)  # -> 64 x 32 x 32
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1) # -> 128 x 16 x 16
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=final_out_channels, kernel_size=3, stride=2, padding=1) # -> 256 x 8 x 8

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.flatten = nn.Flatten()

        self.fc_cnn = nn.Linear(final_out_channels * 8 * 8, 256)

        # separate norms for z and s
        self.norm_I = nn.LayerNorm(final_out_channels)
        self.norm_s = nn.LayerNorm(scalar_dim)

        self.fc_branch   = nn.Linear(scalar_dim, final_out_channels)
        self.fc_combined = nn.Linear(final_out_channels * 2, output_dim)

    def forward(self, I, s):
        # cnn encoder
        I = self.activation(self.conv1(I))  # 32 x 64 x 64
        I = self.activation(self.conv2(I))  # 64 x 32 x 32
        I = self.activation(self.conv3(I))  # 128 x 16 x 16
        I = self.activation(self.conv4(I))  # 256 x 8 x 8
        I = self.norm_I(I)
        I = self.flatten(I)                    # flatten
        I = self.activation(self.fc_cnn(I))    # 256
        
        # branch
        s = self.norm_s(s)
        s = self.fc_branch(s)

        # combine
        x = torch.cat((I,s), dim = 1)
        x = self.fc_combined(x)
    
        return x                             