from typing import Tuple, Optional, Dict
import argparse

class SimpleCrossCollapseFLOPs:
    """
    Cross-attention that reduces across F tokens:
        (B, F, E) --> (B, 1, E)   (default q_len=1)
    Counts ONLY: Q, K, V projections and the two attention matmuls.
    No output projection, no MLP, no softmax.

    FLOPs = 2 * MACs by default (1 mul + 1 add).
    """

    def __init__(self, macs_to_flops: int = 2, count_bias: bool = False):
        self.macs_to_flops = macs_to_flops
        self.count_bias = count_bias

    @staticmethod
    def _linear_macs(tokens: int, in_dim: int, out_dim: int) -> int:
        return tokens * in_dim * out_dim

    def __call__(self,
                 x_shape: Tuple[int, int, int],   # (B, F, E)
                 heads: int,
                 head_dim: Optional[int] = None,
                 q_len: int = 1) -> Dict:
        """
        q_len=1 does F -> 1 collapse.
        If you ever want no reduction, set q_len=F.
        """
        B, F, E = map(int, x_shape)
        d = head_dim
        Hd = heads * d

        macs = {}
        # Projections
        macs["q_proj"] = B * self._linear_macs(q_len, E, Hd)
        macs["k_proj"] = B * self._linear_macs(F,     E, Hd)
        macs["v_proj"] = B * self._linear_macs(F,     E, Hd)

        # Attention core
        macs["attn_qk"] = B * heads * q_len * F * d
        macs["attn_av"] = B * heads * q_len * F * d

        macs_total = sum(macs.values())
        bias_adds = B * (q_len + 2 * F) * Hd  # if you want to count bias adds
        flops_total = macs_total * self.macs_to_flops + (bias_adds if self.count_bias else 0)

        return {
            "out_shape": (B, q_len, E),  # (B, 1, E) when q_len=1
            "macs": int(macs_total),
            "breakdown_macs": {k: int(v) for k, v in macs.items()},
            "flops": int(flops_total),
            "gflops": int(flops_total) / 1e9,
        }
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate FLOPs for a simple cross-attention collapse.")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--x_shape", type=int, nargs=3, default=(32, 128, 512), help="Input shape (B,F,E)")
    parser.add_argument("--heads", type=int, default=32, help="Number of attention heads")
    args = parser.parse_args()

    B, F, E = args.x_shape
    head_dim = E // args.heads
    counter = SimpleCrossCollapseFLOPs()
    result = counter(tuple(args.x_shape), heads=args.heads, head_dim=head_dim)
    print(result)