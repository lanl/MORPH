from typing import Tuple, List, Optional, Dict

class SimpleAttentionFLOPs:
    """
    Counts only attention math:
      - Q, K, V linear projections
      - QK^T and Attn*V matmuls
    No output projection, no feed-forward MLP.

    Conventions:
      Input is (B, L, C) unless noted.
      FLOPs = 2 * MACs by default (1 mul + 1 add).
    """
    def __init__(self, macs_to_flops: int = 2, count_bias: bool = False):
        self.macs_to_flops = macs_to_flops
        self.count_bias = count_bias

    # ---------- helpers ----------
    @staticmethod
    def _linear_macs(tokens: int, in_dim: int, out_dim: int) -> int:
        return tokens * in_dim * out_dim

    def _finish(self, macs_dict: Dict[str, int], out_shape: Tuple[int, ...], bias_adds: int = 0):
        macs = sum(macs_dict.values())
        flops = macs * self.macs_to_flops + (bias_adds if self.count_bias else 0)
        return {
            "out_shape": out_shape,
            "macs": int(macs),
            "breakdown_macs": {k: int(v) for k, v in macs_dict.items()},
            "flops": int(flops),
            "gflops": int(flops) / 1e9,
        }

    # ---------- (1) Full self-attention over L ----------
    def full(self,
             x_shape: Tuple[int, int, int],  # (B, L, C)
             heads: int,
             head_dim: Optional[int] = None):
        B, L, C = map(int, x_shape)
        d = head_dim
        Hd = heads * d

        macs = {}
        macs["q_proj"] = B * self._linear_macs(L, C, Hd)
        macs["k_proj"] = B * self._linear_macs(L, C, Hd)
        macs["v_proj"] = B * self._linear_macs(L, C, Hd)
        macs["attn_qk"] = B * heads * L * L * d
        macs["attn_av"] = B * heads * L * L * d

        bias_adds = B * L * Hd * 3  # if counting bias
        return self._finish(macs, out_shape=(B, L, C), bias_adds=bias_adds)

    # ---------- (2) Axial attention over factors whose product == L ----------
    def axial(self,
              x_shape: Tuple[int, int, int],  # (B, L, C)
              heads: int,
              axis_lengths: List[int],        # e.g., [D, H, W] with D*H*W == L
              head_dim: Optional[int] = None):
        from math import prod
        B, L, C = map(int, x_shape)
        assert prod(axis_lengths) == L, "Product of axis_lengths must equal L"
        d = head_dim
        Hd = heads * d

        macs = {}
        # Shared Q/K/V once (simple & common)
        macs["q_proj"] = B * self._linear_macs(L, C, Hd)
        macs["k_proj"] = B * self._linear_macs(L, C, Hd)
        macs["v_proj"] = B * self._linear_macs(L, C, Hd)

        # Attention along each axis independently
        attn_qk = 0
        attn_av = 0
        for s in axis_lengths:
            groups = L // s            # number of independent lines for this axis
            attn_qk += B * heads * groups * s * s * d
            attn_av += B * heads * groups * s * s * d
        macs["attn_qk"] = attn_qk
        macs["attn_av"] = attn_av

        bias_adds = B * L * Hd * 3
        return self._finish(macs, out_shape=(B, L, C), bias_adds=bias_adds)

    # ---------- (3) Sparse "random" with per-query keys = D+H+W ----------
    def sparse_sum_axes(self,
                        x_shape: Tuple[int, int, int],  # (B, L, C)
                        heads: int,
                        axis_lengths: List[int],        # use sum(axis_lengths) keys per query
                        head_dim: Optional[int] = None):
        B, L, C = map(int, x_shape)
        d = head_dim
        Hd = heads * d
        K = int(sum(axis_lengths))    # your rule: same token count as axial => D + H + W

        macs = {}
        macs["q_proj"] = B * self._linear_macs(L, C, Hd)
        macs["k_proj"] = B * self._linear_macs(L, C, Hd)
        macs["v_proj"] = B * self._linear_macs(L, C, Hd)
        macs["attn_qk"] = B * heads * L * K * d
        macs["attn_av"] = B * heads * L * K * d

        bias_adds = B * L * Hd * 3
        return self._finish(macs, out_shape=(B, L, C), bias_adds=bias_adds)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate FLOPs for different attention types.")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--attn", type =str, choices=["full", "axial", "sparse"], 
                        default="full", help="Type of attention to calculate FLOPs for")
    parser.add_argument("--x_shape", type=int, nargs=3, default=(1, 512, 256), help="Input shape (B,L,C)")
    parser.add_argument("--axis_tokens", type=int, nargs='+', default=[16, 16, 16], 
                        help="Axis lengths for axial attention (if applicable)")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    args = parser.parse_args()

    counter = SimpleAttentionFLOPs()

    B, L, C = args.x_shape
    head_dim = C // args.heads

    if args.attn == "full":
        result = counter.full(tuple(args.x_shape), heads=args.heads, head_dim=head_dim)

    elif args.attn == "axial":
        assert args.axis_tokens[0] * args.axis_tokens[1] * args.axis_tokens[2] == args.x_shape[1], "Product of axis_tokens must equal sequence length L"
        result = counter.axial(tuple(args.x_shape), heads=args.heads, axis_lengths=args.axis_tokens, head_dim=head_dim)

    elif args.attn == "sparse":
        assert args.axis_tokens[0] * args.axis_tokens[1] * args.axis_tokens[2] == args.x_shape[1], "Sum of axis_tokens must equal sequence length L"
        result = counter.sparse_sum_axes(tuple(args.x_shape), heads=args.heads, axis_lengths=args.axis_tokens, head_dim=head_dim)

    else:
        raise ValueError(f"Unknown attention type: {args.attn}")
    
    print(f'For dataset [{args.dataset_name}] and attention [{args.attn}] -->> {result}')
