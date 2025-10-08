#%% Fine-tuning setup
from src.utils.crossattention_fields import FieldCrossAttention
from src.utils.convolutional_operator import ConvOperator
from src.utils.simple_decoder import SimpleDecoder
from src.utils.lora_linear import LoRALinear
import torch.nn as nn
import torch

'''
Level-1 (latent space): MHA, MLP via LoRA, Norms, Pos-enc
Level-2 ( + Encoding): Conv, Proj (linear), Cross-attention
Level-3 ( + Decoder): Linear layer in decoder
Level-4 (Full model): Unfreeze everything
'''

class SelectFineTuningParameters:
    def __init__(self, ft_model, args):
        self.ft_model = ft_model
        self.args = args

    def configure_levels(self):
        # ---------- LEVEL 4: full-model fine-tuning (early return) ----------
        if self.args.ft_level4:
            print('→ Level-4 Finetuning (Full model) activated')
            for p in self.ft_model.parameters():
                p.requires_grad_(True)
            params_full = [p for p in self.ft_model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW([{"params": params_full, "lr": self.args.lr_level4, 
                                            "weight_decay": self.args.wd_level4}])
            print(f"[Level-4] unfrozen → full: {sum(p.numel() for p in params_full)/1e6:.3f}M")
            return optimizer

        # ---------- Common: start by freezing everything ----------
        for p in self.ft_model.parameters():
            p.requires_grad_(False)

        # Collect params per level
        lora_params, norm_params, pe_params = [], [], []
        conv_params, proj_params, xattn_params = [], [], []
        dec_params = []

        #######################################################################
        # --- Level - 1 (LoRA in latent, LayerNorms, Pos-encoding) -----------
        #######################################################################
        if self.args.ft_level1:
            print('→ Level-1 Finetuning (latent space) activated')
            TRAIN_NORMS  = True
            TRAIN_POSENC = True

            # LoRA + LayerNorms
            for m in self.ft_model.modules():
                # LoRA: enable only A/B
                if isinstance(m, LoRALinear) and getattr(m, "rank", 0) > 0:
                    if m.A is not None:
                        m.A.requires_grad_(True); lora_params.append(m.A)
                    if m.B is not None:
                        m.B.requires_grad_(True); lora_params.append(m.B)

                # LayerNorms
                if TRAIN_NORMS and isinstance(m, nn.LayerNorm):
                    for p in m.parameters(recurse=False):
                        p.requires_grad_(True); norm_params.append(p)

            # Positional encoding (handle DP)
            root = getattr(self.ft_model, "module", self.ft_model)
            if TRAIN_POSENC and hasattr(root, "pos_encoding"):
                for p in root.pos_encoding.parameters(recurse=False):
                    p.requires_grad_(True); pe_params.append(p)

        #######################################################################
        # --- Level - 2 (Conv, Linear projection, Cross-attention) ------------
        #######################################################################
        if self.args.ft_level2:
            print('→ Level-2 Finetuning (Encoding) activated')
            TRAIN_CONV, TRAIN_PROJ, TRAIN_XATTN = True, True, True

            root = getattr(self.ft_model, "module", self.ft_model)
            pe = getattr(root, "patch_embedding", None)
            if pe is None:
                raise RuntimeError("[Level-2] patch_embedding not found on model")

            # Conv encoder
            if TRAIN_CONV and hasattr(pe, "conv_features"):
                cf = pe.conv_features
                if isinstance(cf, ConvOperator):
                    for p in cf.parameters():
                        p.requires_grad_(True); conv_params.append(p)
                else:
                    print("[Level-2] conv_features exists but is not ConvOperator")

            # Projection layer
            if TRAIN_PROJ and hasattr(pe, "projection"):
                for p in pe.projection.parameters(recurse=False):
                    p.requires_grad_(True); proj_params.append(p)

            # Cross-attention over fields
            if TRAIN_XATTN and hasattr(pe, "field_attn"):
                xa = pe.field_attn
                if isinstance(xa, FieldCrossAttention):
                    for p in xa.parameters():
                        p.requires_grad_(True); xattn_params.append(p)
                else:
                    print("[Level-2] field_attn exists but is not FieldCrossAttention")

        #######################################################################
        # --- Level - 3 (Decoder linear) --------------------------------------
        #######################################################################
        if self.args.ft_level3:
            print('→ Level-3 Finetuning (Decoding) activated')
            TRAIN_DEC_LINEAR = True

            root = getattr(self.ft_model, "module", self.ft_model)
            if not hasattr(root, "decoder"):
                raise RuntimeError("[Level-3] model has no .decoder attribute")

            dec = root.decoder
            if not isinstance(dec, SimpleDecoder):
                print("[Level-3] Warning: decoder exists but is not SimpleDecoder (continuing)")

            if TRAIN_DEC_LINEAR and hasattr(dec, "linear"):
                for p in dec.linear.parameters():
                    p.requires_grad_(True); dec_params.append(p)
            if not dec_params:
                print("[Level-3] Warning: no decoder params selected to train.")

        # ---------- Build optimizer once from collected groups ----------
        param_groups = []
        if lora_params: param_groups.append({'params': lora_params, 'lr': 1e-3, 'weight_decay': 0.0})
        if norm_params: param_groups.append({'params': norm_params, 'lr': 1e-4, 'weight_decay': 0.0})
        if pe_params:   param_groups.append({'params': pe_params,   'lr': 1e-4, 'weight_decay': 0.0})
        if conv_params: param_groups.append({'params': conv_params, 'lr': 1e-4, 'weight_decay': 1e-5})
        if proj_params: param_groups.append({'params': proj_params, 'lr': 1e-4, 'weight_decay': 1e-5})
        if xattn_params:param_groups.append({'params': xattn_params,'lr': 1e-4, 'weight_decay': 1e-5})
        if dec_params:  param_groups.append({'params': dec_params,  'lr': 1e-4, 'weight_decay': 1e-5})

        if not param_groups:
            raise RuntimeError("No parameters selected for fine‑tuning. "
                               "Did you pass any of --ft_level1/2/3 or --ft_level4?")

        optimizer = torch.optim.AdamW(param_groups)

        # ---------- Summaries ----------
        if self.args.ft_level1:
            print(f"[Level-1] unfrozen → LoRA(A+B): {sum(p.numel() for p in lora_params)/1e6:.3f}M | "
                  f"norms: {sum(p.numel() for p in norm_params)/1e6:.3f}M | "
                  f"pos-enc: {sum(p.numel() for p in pe_params)/1e6:.3f}M")
        if self.args.ft_level2:
            print(f"[Level-2] unfrozen → conv: {sum(p.numel() for p in conv_params)/1e6:.3f}M | "
                  f"proj: {sum(p.numel() for p in proj_params)/1e6:.3f}M | "
                  f"xattn: {sum(p.numel() for p in xattn_params)/1e6:.3f}M")
        if self.args.ft_level3:
            print(f"[Level-3] unfrozen → dec: {sum(p.numel() for p in dec_params)/1e6:.3f}M")
        
        # print model parameters
        total_params = sum(p.numel() for p in self.ft_model.parameters()) / 1e6
        trainable_params = sum(p.numel() for p in self.ft_model.parameters() if p.requires_grad) / 1e6
        print(f"→ TOTAL PARAMS (M): {total_params:.3f} | TRAINABLE (M): {trainable_params:.3f}")
        
        return optimizer
