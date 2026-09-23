import torch
import torch.nn.functional as F

def to_poseidon_input(x1: torch.Tensor, out_size: int = 128, pad_mode: str = "replicate"):
    """
    (B,1,H,W) -> (B,4,out_size,out_size)
    Preserves aspect ratio by resizing so the LONGER side becomes out_size, then pads the shorter side.
    """
    assert x1.ndim == 4 and x1.shape[1] == 1, f"expected (B,1,H,W), got {tuple(x1.shape)}"
    B, _, H, W = x1.shape

    # scale so max(H,W) -> out_size
    scale = out_size / max(H, W)
    newH = int(round(H * scale))
    newW = int(round(W * scale))

    x_rs = F.interpolate(x1, size=(newH, newW), mode="bilinear", align_corners=False)

    pad_h = out_size - newH
    pad_w = out_size - newW
    assert pad_h >= 0 and pad_w >= 0, "Internal error: resized larger than out_size."

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    x_sq = F.pad(x_rs, (pad_left, pad_right, pad_top, pad_bottom), mode=pad_mode)

    x4 = x_sq.repeat(1, 4, 1, 1)

    meta = {
        "orig_hw": (H, W),
        "resized_hw": (newH, newW),
        "pad": (pad_left, pad_right, pad_top, pad_bottom),
        "out_size": out_size,
    }
    return x4, meta


def undo_poseidon_output(y4: torch.Tensor, meta: dict, take_channel: int = 0):
    """
    (B,4,out_size,out_size) -> (B,1,H,W):
      - select one channel
      - remove padding
      - resize back to original
    """
    H, W = meta["orig_hw"]
    newH, newW = meta["resized_hw"]
    pl, pr, pt, pb = meta["pad"]

    y = y4[:, take_channel:take_channel+1]      # (B,1,S,S)
    y = y[:, :, pt:pt+newH, pl:pl+newW]         # (B,1,newH,newW)
    y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
    return y
