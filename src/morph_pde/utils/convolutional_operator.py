import torch
import torch.nn as nn

class ConvOperator(nn.Module):
    def __init__(self, max_in_ch, conv_filter, hidden_dim=8):
        """
        A conv‐operator that can take 1..max_in_ch channels dynamically,
        by padding to max_in_ch and projecting down via a 1×1 conv.

        Args:
            max_in_ch  (int): maximum channels you'll ever see (e.g. 3).
            conv_filter(int): final number of output feature maps.
            hidden_dim (int): intermediate channel count after 1×1 conv
                              (defaults to 8).
        """
        super().__init__()
        self.max_in_ch = max_in_ch
        self.hidden_dim = hidden_dim

        # 1×1×1 projection from max_in_ch → hidden_dim
        self.input_proj = nn.Conv3d(max_in_ch, hidden_dim,
                                    kernel_size=1, bias=False)
        

        # now build the “doubling” stack from hidden_dim → conv_filter
        layers = []
        prev = hidden_dim
        # double up until we hit conv_filter
        while prev < conv_filter:
            nxt = min(prev * 2, conv_filter)
            layers += [
                nn.Conv3d(prev, nxt, kernel_size=3, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            prev = nxt
            
        # (if hidden_dim == conv_filter, this just does one final conv below)
        layers += [
            nn.Conv3d(prev, conv_filter, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        self.conv_stack = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B*t*F, in_ch, D, H, W)
        in_ch = x.shape[1]         # in_ch: actual number of input channels (1 … max_in_ch)
        # pad to max_in_ch if needed
        if in_ch < self.max_in_ch:
            pad_sz = self.max_in_ch - in_ch            # how many channels we need to add
            # create a zero‐tensor of shape (batch, pad_sz, D, H, W)
            pad = x.new_zeros(x.size(0), pad_sz, *x.shape[2:])  # → (B*t*F, pad_sz, D, H, W)
            # concatenate along the channel dimension →
            # resulting shape is (B*t*F, max_in_ch, D, H, W)
            x = torch.cat([x, pad], dim=1)

        # project down to hidden_dim with 1×1×1 conv
        x = self.input_proj(x)

        # run through the doubling conv stack
        x = self.conv_stack(x)
        return x
