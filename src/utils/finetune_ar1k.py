import torch
import torch.nn as nn
from src.utils.main_process_ddp import is_main_process

class FineTuneAR:
    """
    Freeze all params, train only:
      - LayerNorms (optionally last K blocks)
      - Positional encodings (name-based: pos_embed/positional/pos_enc/pos_embedding/pos_encoding)
      - Temporal attention submodules (name-based: att_t/temporal/time_attn; optionally last K)
    Returns optimizer configured for those params.
    """
    def __init__(self, model, ln_last_k_blocks = None, att_last_k_blocks = None, pe = None):
        self.model = model
        self.ln_last_k_blocks = ln_last_k_blocks
        self.att_last_k_blocks = att_last_k_blocks 
        self.pe = pe

    def _add_param(self, p, bucket, seen):
        if id(p) not in seen:
            p.requires_grad_(True)
            bucket.append(p)
            seen.add(id(p))

    def configure(self):
        base = self.model.module if hasattr(self.model, "module") else self.model

        # 1) Freeze everything
        for p in base.parameters():
            p.requires_grad_(False)

        ln_params, pos_params, att_params = [], [], []
        ln_names, pos_names, att_names = [], [], []
        seen = set()

        # 2) Collect LayerNorms in topological order
        ln_modules = [(n, m) for n, m in base.named_modules() if isinstance(m, nn.LayerNorm)]
        if self.ln_last_k_blocks is None:
            kept_ln = [n for n, _ in ln_modules]
        else:
            # ~2 LayerNorms per block in pre-norm transformers
            keep_count = max(1, 2 * self.ln_last_k_blocks)
            kept_ln = [n for n, _ in ln_modules[-keep_count:]]

        for mod_name, mod in base.named_modules():
            if isinstance(mod, nn.LayerNorm) and mod_name in kept_ln:
                for _, p in mod.named_parameters(recurse=False):
                    self._add_param(p, ln_params, seen)
                ln_names.append(mod_name)
       
        # 3) Positional encodings by name, avoid relative/rotary
        if self.pe!=None:
            POS_KEYS = ("pos_encoding", "position_embedding", "pos_embedding","pos_embed", "pos_enc", "positional")
            for name, p in base.named_parameters():
                lname = name.lower()
                if any(k in lname for k in POS_KEYS):
                    self._add_param(p, pos_params, seen)
                    pos_names.append(name)

        # 4) Temporal attention modules by name — now with "last K" option
        #    First gather matching modules in order; then keep the tail.
        att_modules = []
        for name, mod in base.named_modules():
            n = name.lower()
            if n.endswith("attn_t") or "att_t" in n or ".att_t." in n or "temporal" in n or "time_attn" in n:
                att_modules.append((name, mod))

        if self.att_last_k_blocks is None:
            kept_att = [n for n, _ in att_modules]
        else:
            keep_count = max(1, self.att_last_k_blocks)
            kept_att = [n for n, _ in att_modules[-keep_count:]]

        for mod_name, mod in att_modules:
            if mod_name in kept_att:
                for p in mod.parameters(recurse=True):
                    self._add_param(p, att_params, seen)
                att_names.append(mod_name)

        # Build param groups
        param_groups = []
        if att_params: param_groups.append({"params": att_params, "lr": 1e-4, "weight_decay": 0.0})
        if ln_params:  param_groups.append({"params": ln_params,  "lr": 1e-4, "weight_decay": 0.0})
        if pos_params: param_groups.append({"params": pos_params, "lr": 1e-4, "weight_decay": 0.0})

        if not param_groups:
            raise RuntimeError("FineTuneAR: No parameters matched LN/pos/temporal-attn. Check names.")

        optimizer = torch.optim.AdamW(param_groups)

        if is_main_process():
            # counts by tensors
            print(f"[FineTuneAR] tensors → LN={len(ln_params)} | POS={len(pos_params)} | ATT={len(att_params)}")

            # counts by scalar params (in millions)
            ln_M  = sum(p.numel() for p in ln_params)  / 1e6
            pos_M = sum(p.numel() for p in pos_params) / 1e6
            att_M = sum(p.numel() for p in att_params) / 1e6
            print(f"[FineTuneAR] params (M) → LN={ln_M:.3f} | POS={pos_M:.3f} | ATT={att_M:.3f}")

            total_M     = sum(p.numel() for p in base.parameters()) / 1e6
            trainable_M = sum(p.numel() for p in base.parameters() if p.requires_grad) / 1e6
            print(f"→ TOTAL PARAMS (M): {total_M:.3f} | TRAINABLE (M): {trainable_M:.3f}")

            # Optional visibility: which modules were kept
            print(f"Kept LN modules: {len(ln_names)} (tail)")
            print(f"Kept temporal attention modules: {len(att_names)} (tail)")

        return optimizer

    # Optional convenience
    def __call__(self):
        return self.configure()
