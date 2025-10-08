# patch_pos_embed.py
import os
import torch
from typing import Optional

def _repeat_time_axis(x: torch.Tensor, target_ar: int) -> torch.Tensor:
    """
    x shape: (1, AR, P, D). Returns shape: (1, target_ar, P, D)
    Repeats along time (AR) and, if target_ar is not a multiple of AR,
    repeats the last slice to fill the remainder.
    """
    assert x.dim() == 4 and x.shape[0] == 1, f"Unexpected shape: {tuple(x.shape)}"
    B, AR, P, D = x.shape
    if AR == target_ar:
        return x

    reps = target_ar // AR
    rem  = target_ar - AR * reps

    out = x.repeat(1, reps, 1, 1)
    if rem > 0:
        out = torch.cat([out, x[:, -1:].repeat(1, rem, 1, 1)], dim=1)
    return out.contiguous()

def patch_checkpoint_pos_embed(in_path: str, out_path: Optional[str] = None, target_ar: int = 16):
    # Load on CPU; weights_only=True is fine
    ckpt = torch.load(in_path, map_location="cpu", weights_only=True)
    sd = ckpt["model_state_dict"]

    # Find the key with and without DDP 'module.' prefix
    candidates = [k for k in sd.keys() if k.endswith("pos_encoding.pos_embedding")]
    if not candidates:
        # Fallback in case it's named slightly differently
        candidates = [k for k in sd.keys() if k.endswith("pos_embedding")]
    if not candidates:
        raise KeyError("Could not find 'pos_encoding.pos_embedding' in checkpoint state_dict.")

    key = candidates[0]
    old_pe = sd[key]  # (1, 1, 4096, 256) in your case
    new_pe = _repeat_time_axis(old_pe, target_ar)
    sd[key] = new_pe
    print(f"{key}: {tuple(old_pe.shape)} -> {tuple(new_pe.shape)}")

    # --- Patch optimizer states too (e.g., Adam exp_avg / exp_avg_sq) so loading doesn't error ---
    opt = ckpt.get("optimizer_state_dict", None)
    if isinstance(opt, dict) and "state" in opt:
        for pid, st in opt["state"].items():
            for tname in ("exp_avg", "exp_avg_sq"):
                t = st.get(tname, None)
                if isinstance(t, torch.Tensor) and t.shape == old_pe.shape:
                    st[tname] = _repeat_time_axis(t, target_ar)
                    print(f"optimizer.{tname} for param_id {pid}: {tuple(t.shape)} -> {tuple(st[tname].shape)}")

    # Save to a new file unless you want in-place overwrite
    if out_path is None:
        root, ext = os.path.splitext(in_path)
        out_path = f"{root}_ar{target_ar}{ext}"

    torch.save(ckpt, out_path)
    print(f"Wrote patched checkpoint to: {out_path}")

#%% main
base_path = 'F:/FM/codes/lisdi/pdefoundationalmodel_vit/models/'
model = 'FM'
old_checkpoint = '/morph-L-FM-max_ar1_ep36'
new_checkpoint = old_checkpoint + '_extd'
in_path = base_path + model + old_checkpoint + '.pth'
out_path = base_path + model + new_checkpoint + '.pth'
target_ar = 16
patch_checkpoint_pos_embed(in_path, out_path, target_ar)
