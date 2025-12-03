import os
import sys
import gc
import torch
import torch.nn as nn
import platform
import time
import math
import socket
from torch.utils.data import DataLoader  # MOD: for non-DDP loaders
import matplotlib
matplotlib.use("Agg")   # set backend first
import matplotlib.pyplot as plt

# Add project root to path
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

savepath_model = os.path.join(project_root, 'models')
savepath_results = os.path.join(project_root, 'experiments', 'results')
    
# Main entry
# Import model, trainers, and data loader
from config.argument_parser_pdegym import ArgsConfig
from src.utils.device_manager import DeviceManager
from src.utils.vit_conv_xatt_axialatt2 import ViT3DRegression
from src.utils.main_process_ddp import is_main_process
from src.utils.dataloaders.dataloading_pdegym import pdegym_datasets, build_dataloaders
from src.utils.trainers_pdegym import train_one_epoch_multitask, validate_multitask

# ---- arguments ----
args = ArgsConfig()
devices = DeviceManager.list_devices()
choice = 'FM_pdegym'  
max_ar_order = args.max_ar_order
filters, dim, heads, depth, mlp_dim = args.tf_params
dropout, emb_dropout                = args.tf_reg
device = devices[args.device_idx] if devices else 'cpu'

# ---- Data paths (pretraining on 6 datasets) ----
if args.dataset_root is None:
    dataset_root = project_root
else:
    dataset_root = args.dataset_root

# ---- Batch size ----
bs_ns = args.bs_ns
bs_ce = args.bs_ce

##########################################################################
####################### --- Batch sizes --- ##############################
##########################################################################

# scale batch size accordingly for parallel method dp
if args.parallel == 'dp':
    ng = torch.cuda.device_count()
    if ng == 0:
        raise RuntimeError("args.parallel='dp' but no GPUs are available.")
    bs_ns = bs_ns * ng
    bs_ce = bs_ce * ng
    
# ---- determine data configuration based on {choice} and {surrogate_type} ----
max_resolution = 128 * 128
max_patches = max_resolution // (args.patch_size**2)
max_fields = 3
max_components = 3

# some prints
print(f"→ filters: {filters}, dim: {dim}, heads: {heads}, depth: {depth}, mlp_dim: {mlp_dim}")
print(f"→ Current dataset root: {dataset_root}")
print(f"→ Current patch_size: {args.patch_size}")
print(f'→ [Max.] num_tokens: {max_patches}, fields: {max_fields}, '
        f'components: {max_components}')

##########################################################################
###################### ---- Data Loading ---- ############################
##########################################################################

# load datasets
datasets = ["2dNS_Sines_pdegym","2dNS_Gauss_pdegym","2dCE_RP_pdegym","2dCE_CRP_pdegym",
            "2dCE_KH_pdegym","2dCE_Gauss_pdegym"]

# --- Pretraining sets ---
loadpath_ns_sines    = os.path.join(dataset_root,'datasets', 'normalized_revin', datasets[0])
datapath_ns_gaussians = os.path.join(dataset_root,'datasets', 'normalized_revin', datasets[1])
loadpath_ce_rp       = os.path.join(dataset_root,'datasets', 'normalized_revin', datasets[2])
loadpath_ce_crp      = os.path.join(dataset_root,'datasets', 'normalized_revin', datasets[3])
loadpath_ce_kh       = os.path.join(dataset_root,'datasets', 'normalized_revin', datasets[4])
loadpath_ce_gauss    = os.path.join(dataset_root,'datasets', 'normalized_revin', datasets[5])  

# --- build datasets ---
train_ns, train_ce, val_ns, val_ce = pdegym_datasets(
    loadpath_ns_sines, datapath_ns_gaussians, 
    loadpath_ce_rp, loadpath_ce_crp,
    loadpath_ce_kh, loadpath_ce_gauss, 
    use_small_dataset = args.use_small_dataset,
    ar_order = args.ar_order)

# --- build dataloaders ---
# MOD: simple non-DDP loaders so the rest of the code can stay the same
train_loader_ns, train_loader_ce, val_loader_ns, val_loader_ce = build_dataloaders(
    train_ns, train_ce, val_ns, val_ce,
    batch_size_ns = bs_ns,
    batch_size_ce = bs_ce,
    num_workers = args.num_workers)

# train loader and val loaders
train_loaders = [train_loader_ce, train_loader_ns]
val_loaders   = [val_loader_ce,   val_loader_ns]

print(f'→ Length of train_loader_ce: {len(train_loader_ce)}, train_loader_ns: {len(train_loader_ns)}')
print(f'→ Length of val_loader_ce: {len(val_loader_ce)}, val_loader_ns: {len(val_loader_ns)}')

##########################################################################    
####################### ---- Model setup ---- ############################
##########################################################################

model_name = (f"morph_pdegym-{args.model_size}-ps_{args.patch_size}-parallel_{args.parallel}")
    
model = ViT3DRegression(
    patch_size = args.patch_size, dim = dim, depth = depth,
    heads = heads, heads_xa = args.heads_xa, mlp_dim = mlp_dim,
    max_components = max_components, conv_filter = filters, 
    conv_layer1_filters = args.conv_layer1_filters, 
    scale_filter = args.scale_filter,
    max_ar = max_ar_order, max_patches = max_patches, max_fields = max_fields,
    dropout = dropout, emb_dropout = emb_dropout,
    lora_r_attn = 0, lora_r_mlp = 0,
    lora_alpha = None, lora_p = 0.0,
    model_size = args.model_size, 
    activated_ar1k = args.activated_ar1k,
    conv_off = args.conv_off,
    replace_xatt = args.replace_xatt,
    attn_type=args.attn_type,
    pos_enc=args.pos_enc
).to(device)

num_params_model = sum(p.numel() for p in model.parameters()) / 1e6

print(f"→ NUMBER OF PARAMETERS OF THE ViT (in millions): "
        f"{num_params_model}")

# parallel setup
if args.parallel == 'dp' and torch.cuda.device_count() > 1:
    model =  nn.DataParallel(model)

##########################################################################
################### ---- loss and optimizer ------- ######################
##########################################################################

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=args.lr_patience
)

##########################################################################
##################### ---- resume logic ---- #############################
##########################################################################

start_epoch = 0
if args.resume and args.ckpt_name is None:
    args.parser.error("--resume requires you to also pass --ckpt_name <CHECKPOINT>")

if args.ckpt_name:
    resume_path = os.path.join(savepath_model, f'{choice}', args.ckpt_name)
    ckpt = torch.load(resume_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict"))
    
    if state_dict is None:
        raise KeyError("Checkpoint missing model_state_dict/state_dict")
    
    # pick the real model if wrapped
    if isinstance(model, nn.DataParallel):
        target = model.module 
    else:
        target = model
    
    # strip 'module.' prefix if present in checkpoint
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        
    # always load strictly (no finetune_ar1k logic)
    incompatible = target.load_state_dict(state_dict, strict=True)
    if (incompatible.missing_keys or incompatible.unexpected_keys):
        print(f"load_state_dict: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}")
    
    # always restore optimizer / scheduler
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    
    # Optionally override LR
    if getattr(args, "new_lr_ckpt", None) is not None:
        for g in optimizer.param_groups:
            g["lr"] = float(args.new_lr_ckpt)
                
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = ckpt.get("epoch", 0)
    print(f"Resumed from {resume_path}, starting at epoch {start_epoch}")
    
##########################################################################
###################### ---- training ---- ################################
##########################################################################
        
gc.collect()
torch.cuda.empty_cache()
ep_st = time.time()

savepath_model_folder = os.path.join(savepath_model, f'{choice}')
os.makedirs(savepath_model_folder, exist_ok=True)
model_path = os.path.join(savepath_model_folder, model_name)
train_losses, val_losses = [], []
best_val_loss = float('inf')
epochs_no_improve = 0

# MOD: task sampling probabilities CE vs NS (balanced per-task)
n_ce = len(train_ce)
n_ns = len(train_ns)
w_ce = 1.0 / n_ce
w_ns = 1.0 / n_ns
task_probs = torch.tensor([w_ce, w_ns], dtype=torch.float32)
task_probs = task_probs / task_probs.sum()
print(f"→ Task sampling probabilities: CE: {task_probs[0]:.4f}, NS: {task_probs[1]:.4f}")

for epoch in range(start_epoch, args.num_epochs):
    # MOD: multi-task training & validation
    tr_loss = train_one_epoch_multitask(
        model=model,
        train_loaders=train_loaders,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epoch=epoch,
        task_probs=task_probs)
        
    vl_loss, val_detail = validate_multitask(
        model=model,
        val_loaders=val_loaders,
        criterion=criterion,
        device=device)

    train_losses.append(tr_loss)
    val_losses.append(vl_loss)
    
    scheduler.step(vl_loss)
    
    current_lr = optimizer.param_groups[0]['lr']

    ce_loss = val_detail.get("ce", float('nan'))
    ns_loss = val_detail.get("ns", float('nan'))
    print(
        f"Time = {(time.time()-ep_st)/60:.2f} min., LR:{current_lr:.6f}, "
        f"Epoch {epoch+1}/{args.num_epochs} |"
        f"Train:{tr_loss:.5f}, Val:{vl_loss:.5f} "
        f"(ValCE:{ce_loss:.5f}, ValNS:{ns_loss:.5f})"
    )
    
    # --- warm-up epochs ---
    if (epoch + 1) > args.warm_epochs: 
        # --early stopping logic ---
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            epochs_no_improve = 0
            
            # --- save checkpoint (save_every vs noverwrite) ---
            if args.overwrite_weights:
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict()
                }
                ckpt_path = (f"{model_path}_ftAR1-{args.ar_order}_best.pth")
                torch.save(checkpoint, ckpt_path)
                print(f" Saved (Over-write) previous checkpoint: {ckpt_path}")
            
            else:
                if (epoch + 1) % args.save_every == 0:
                    checkpoint = {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict()
                    }
                    ckpt_path = (f"{model_path}_ftAR1-{args.ar_order}_ep{epoch+1}.pth")
                    torch.save(checkpoint, ckpt_path)
                    print(f" Saved checkpoint: {ckpt_path}")
                    
        else:
            epochs_no_improve += 1
            print(f"Not improved for {epochs_no_improve}/{args.patience} epochs")

    if epochs_no_improve >= args.patience:
        print(
            f"Early stopping triggered. Validation loss did not "
                f"improve for {args.patience} epochs."
            )
        break
    
    gc.collect()
    torch.cuda.empty_cache()

# --- Plot losses ---
os.makedirs(savepath_results, exist_ok=True)
fig, ax = plt.subplots()
ax.plot(train_losses, label='Train')
ax.plot(val_losses, label='Val')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend()
fig.savefig(os.path.join(
    savepath_results, f'loss_{choice}_max_ar_{args.max_ar_order}.png'
))

