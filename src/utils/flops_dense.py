import math
from os import name
from typing import Tuple
import argparse

class SimpleDenseFLOPs:
    """
    FLOPs for a last-dim Linear: y = x @ W (+ b)
    Input:  (B, C, Fin)
    Output: (B, C, Fout)
    MACs = B * C * Fin * Fout
    FLOPs = 2 * MACs (multiply + add), plus optional bias adds.
    """
    def __init__(self, macs_to_flops: int = 2, count_bias: bool = False):
        self.macs_to_flops = macs_to_flops
        self.count_bias = count_bias

    def __call__(self,
                 x_shape: Tuple[int, int, int],
                 out_features: int):
        B, C, Fin = map(int, x_shape)
        Fout = int(out_features)

        # Output shape
        out_shape = (B, C, Fout)

        # Work per single output element
        macs_per_out = Fin

        # Totals
        total_out_elems = B * C * Fout
        macs = total_out_elems * macs_per_out
        flops = macs * self.macs_to_flops
        if self.count_bias:
            flops += total_out_elems  # one add per output element

        return {
            "out_shape": out_shape,
            "per_output_position_macs": int(macs_per_out),
            "total_output_elements": int(total_out_elems),
            "macs": int(macs),
            "flops": int(flops),
            "gflops": int(flops) / 1e9,
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate FLOPs for a simple dense layer.")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--x_shape", type=int, nargs=3, default=(32, 128, 512), help="Input shape (B,C,Fin)")
    parser.add_argument("--embed_dims", type=int, default=256, help="Number of output channels")
    args = parser.parse_args()
    flops_calculator = SimpleDenseFLOPs()
    result = flops_calculator(x_shape = tuple(args.x_shape), out_features=args.embed_dims)
    print(f'For {args.dataset_name} - {result}')