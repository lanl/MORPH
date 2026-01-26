import os
from src.utils.vit_conv_xatt_axialatt2 import ViT3DRegression 
from src.utils.optimizer_finetuning import SelectFineTuningParameters
from huggingface_hub import hf_hub_download
import torch.nn as nn
import torch

def morph_ft(model_variant='S', device='cpu',
            standalone = False,  
            rank_lora_attn = 0, rank_lora_mlp = 0, lora_p = 0.05,
            lr_morph=1e-4, 
            wd_morph=1e-5,
            l1 = False,
            l2 = False,
            l3 = False,
            l4 = False,
            model_dir=None):
    
    # --- MORPH CONFIGURATION ---
    MORPH_MODELS = {
        'Ti': [8, 256,  4,  4, 1024],
        'S' : [8, 512,  8,  4, 2048],
        'M' : [8, 768, 12,  8, 3072],
        'L' : [8, 1024,16, 16, 4096]
        }

    model_size = MORPH_MODELS[model_variant]  # choose from 'Ti', 'S', 'M', 'L'
    filters, dim, depth, heads, mlp_dim = model_size
    heads_xa = 32
    dropout = 0.1
    emb_dropout = 0.1
    max_ar_order = 1
    patch_size = 8
    max_patches = 4096
    max_fields = 3
    max_components = 3

    morph = ViT3DRegression(patch_size=patch_size, dim=dim, depth=depth,
        heads=heads, heads_xa=heads_xa, mlp_dim=mlp_dim,
        max_components=max_components, conv_filter=filters,
        max_ar=max_ar_order,
        max_patches=max_patches, max_fields=max_fields,
        dropout=dropout, emb_dropout=emb_dropout,
        lora_r_attn=rank_lora_attn,            # <— rank of A and B in the attention module
        lora_r_mlp=rank_lora_mlp,              # <— rank of A and B in the MLP module
        lora_alpha=None,                       # defaults to 2*rank inside LoRA
        lora_p=lora_p                          # dropout on LoRA path
    ).to(device)

    # print('Model architecture:', ft_model)
    num_params_model = sum(p.numel() for p in morph.parameters()) / 1e6
    print(f"→ NUMBER OF PARAMETERS OF THE MODEL (in M): {num_params_model:.3g}")

    if standalone == False:
        print("==== Fine-tuning MORPH from foundational model weights ====")
        # load the foundational model weights
        checkpoint_name = {'Ti':"morph-Ti-FM-max_ar1_ep225.pth", 
                        'S': "morph-S-FM-max_ar1_ep225.pth",
                        'M': "morph-M-FM-max_ar1_ep290_latestbatch.pth", 
                        'L': "morph-L-FM-max_ar16_ep189_latestbatch.pth"}

        if os.path.exists(os.path.join(model_dir, "FM", checkpoint_name[model_variant])):
            weights_path = os.path.join(model_dir, "FM", checkpoint_name[model_variant])
        else:
            # e.g., grab the "Ti" checkpoint (change filename as needed)
            weights_path = hf_hub_download(
                repo_id="mahindrautela/MORPH",
                filename=checkpoint_name[model_variant],
                subfolder="models/FM",
                repo_type="model",              # optional
                resume_download=True,           # continue if interrupted
                local_dir=".",         # where to place it
                local_dir_use_symlinks=False    # copy file instead of symlink
        )
        print(weights_path)

        # Load weights into the model
        # ---- load the pretrained weights ----
        start_epoch = 0
        print(f"→ Loading checkpoints from {weights_path}")
        # --- Load pretrained checkpoint from foundational model ---
        ckpt = torch.load(weights_path, map_location=device, weights_only=True)
        state_dict = ckpt["model_state_dict"]

        # pick the real model if wrapped   
        target = morph.module if isinstance(morph, nn.DataParallel) else morph 

        if state_dict and next(iter(state_dict)).startswith("module."):
            print("→ Stripping 'module.' from checkpoints")
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            
        # strict=False because ft_model has extra LoRA params (A/B) not in ckpt
        missing, unexpected = target.load_state_dict(state_dict, strict=False)

        # sanity print
        print("Missing keys (expected: LoRA A/B etc.):",
                [k for k in missing if k.endswith((".A", ".B")) or ".lora" in k])
        print("Unexpected keys:", unexpected)
        print(f"→ Resumed from {weights_path}, starting at epoch {start_epoch}")

        # optimizer with selected fine-tuning parameters
        selector = SelectFineTuningParameters(morph, lr=lr_morph, wd=wd_morph,
                                             ft_level1=l1, ft_level2=l2,
                                             ft_level3=l3, ft_level4=l4)
        optimizer = selector.configure_levels()
    
    else:
        print("==== Training MORPH from scratch (standalone=True) ====")
        optimizer = torch.optim.AdamW(morph.parameters(), lr=lr_morph,
                                       weight_decay=wd_morph)

    return morph, optimizer