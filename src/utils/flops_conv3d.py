import math
from typing import Tuple, Union
import argparse

class Conv3dFLOPs:
    """
    Super small FLOP counter for one Conv3d-like op.
    - Counts multiply+add as 2 FLOPs by default.
    - Optionally counts bias adds (1 add per output element).
    """
    def __init__(self, macs_to_flops: int = 2, count_bias: bool = False):
        self.macs_to_flops = macs_to_flops
        self.count_bias = count_bias

    @staticmethod
    def _triple(x: Union[int, Tuple[int, int, int]]):
        if isinstance(x, (tuple, list)):
            assert len(x) == 3
            return int(x[0]), int(x[1]), int(x[2])
        x = int(x)
        return x, x, x

    def __call__(self,
                 x_shape: Tuple[int, int, int, int, int],
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int, int]] = 3,
                 stride: Union[int, Tuple[int, int, int]] = 1,
                 padding: Union[int, Tuple[int, int, int]] = 1,
                 dilation: Union[int, Tuple[int, int, int]] = 1,
                 groups: int = 1):
        B, C, D, H, W = x_shape
        Kd, Kh, Kw = self._triple(kernel_size)
        Sd, Sh, Sw = self._triple(stride)
        Pd, Ph, Pw = self._triple(padding)
        Dd, Dh, Dw = self._triple(dilation)

        # PyTorch Conv3d output size formula
        D_out = math.floor((D + 2*Pd - Dd*(Kd - 1) - 1) / Sd + 1)
        H_out = math.floor((H + 2*Ph - Dh*(Kh - 1) - 1) / Sh + 1)
        W_out = math.floor((W + 2*Pw - Dw*(Kw - 1) - 1) / Sw + 1)

        # MACs per single output location
        macs_per_out = (C // groups) * Kd * Kh * Kw

        total_out_elems = B * out_channels * D_out * H_out * W_out
        macs = total_out_elems * macs_per_out

        flops = macs * self.macs_to_flops
        if self.count_bias:
            flops += total_out_elems  # bias add per output element

        return {
            "out_shape": (B, out_channels, D_out, H_out, W_out),
            "per_output_position_macs": int(macs_per_out),
            "total_output_elements": int(total_out_elems),
            "macs": int(macs),
            "flops": int(flops),
            "gflops": int(flops) / 1e9,
        }
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Conv3dFLOPs")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--x_shape", type=int, nargs=5, default=(1, 3, 64, 64, 64), help="Input shape (B,C,D,H,W)")
    parser.add_argument("--out_channels", type=int, default=8, help="Number of output channels")
    parser.add_argument("--kernel_size", type=int, default=1, help="Kernel size (int or tuple)")
    parser.add_argument("--padding", type=int, default=0, help="Padding (int or tuple)")
    parser.add_argument("--stride", type=int, default=1, help="Stride (int or tuple)")
    parser.add_argument("--dilation", type=int, default=1, help="Dilation (int or tuple)")
    parser.add_argument("--groups", type=int, default=1, help="Number of groups)")
    flops_counter = Conv3dFLOPs()
    args = parser.parse_args()
    result = flops_counter(
        x_shape=tuple(args.x_shape),
        out_channels=args.out_channels,
        kernel_size=args.kernel_size,
        stride=args.stride,
        padding=args.padding,
        dilation=args.dilation,
        groups=args.groups
    )
    print(f'For dataset {args.dataset_name if args.dataset_name else "unknown"}, result: {result}')