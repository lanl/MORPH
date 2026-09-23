import glob
import json
import os
import sys
import argparse
from turtle import Shape
import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MPLBACKEND"] = "Agg"      # safest: force non-GUI backend
matplotlib.use("Agg")                 # belt-and-suspenders

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

# load the classes
from src.utils.vit_conv_xatt_axialatt2 import ViT3DRegression
from src.utils.data_preparation_fast import FastARDataPreparer
from src.utils.optimizer_finetuning import SelectFineTuningParameters
from src.utils.lr_schedulars import LRSched
from src.utils.trainers import Trainer

############################################
### -- instantiate argument parsers --- ###
###########################################

MORPH_MODELS = {
    'Ti': [8, 256,  4,  4, 1024],
    'S' : [8, 512,  8,  4, 2048],
    'M' : [8, 768, 12,  8, 3072],
    'L' : [8, 1024,16, 16, 4096]
    }

# ---- set arguments ----
parser = argparse.ArgumentParser(description="Run inference on trained ViT3D model")
parser.add_argument('--dataset_dir', type=str, help="Path to the input tensor", required=True)
parser.add_argument('--num_traj', type=int, default=5300, help="Number of trajectories to load (default: 5300)")
parser.add_argument('--model_size', type=str, choices = list(MORPH_MODELS.keys()),
                    default='S', help='choose from Ti, S, M, L')
parser.add_argument('--ckpt_from', type=str, choices=['FM', 'FT'], default='FM',
                    help="Whether to load checkpoints from foundational model (FM), previous finetuning (FT), or not load any weights (None)")
parser.add_argument('--checkpoint', type=str, help="Path to saved .pth state dict", required=True)
parser.add_argument('--min_lr', type=float, default=1e-7, help="Minimum LR for schedular")
parser.add_argument('--batch_size', type=int, default=1, help="Batch size for inference")
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--wd', type=float, default=1e-2, help = 'weight decay')
parser.add_argument('--warm_epochs', type = int, default = 5)
parser.add_argument('--num_epochs', type=int, default = 100, help="Fine-tuning epochs")
parser.add_argument('--patience', type=int, default=10, help="Early stopping criteria")

# --- Finetune levels ---                  
parser.add_argument('--ft_level1', action='store_true', help = "Level-1 finetuning (LoRA, PE, LN)")
parser.add_argument('--ft_level2', action='store_true', help = "Level-2 finetuning (Encoder)")
parser.add_argument('--ft_level3', action='store_true', help = "Level-3 finetuning (Decoder)")
parser.add_argument('--ft_level4', action='store_true', help = "All model parameters")
# --- save related ----
parser.add_argument('--save_every', type=int, default=1, help = "Save epochs at intervals")
parser.add_argument('--save_batch_ckpt', action='store_true', help = "Save batch checkpoints")
parser.add_argument('--save_batch_freq', type=int, default=1000, help = "Batch checkpoints frequency")
args = parser.parse_args()

# --- set the parser for defining parameters ---
args = parser.parse_args()
# print args
print("===== Arguments Used =====")
for k, v in vars(args).items():
    print(f"{k}: {v}")
print("==========================")

# device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# folder to load dataset and norm constants
savepath_model = os.path.join(project_root, "models")
os.makedirs(savepath_model, exist_ok=True)

# save results
savepath_results = os.path.join(project_root, "experiments")
os.makedirs(savepath_results, exist_ok=True)
norm_dir = os.path.join(project_root, "data")
os.makedirs(norm_dir, exist_ok=True)

# Dataset for Dataloader
class DatasetforDataloader(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- collect first/last frame per trajectory ---
p0_list  = []
p99_list = []

for traj_no in range(1, args.num_traj + 1):
    print(f"Loading trajectory {traj_no}/{args.num_traj}...")

    p0  = os.path.join(args.dataset_dir, f"lsc240420_id{traj_no:05d}_pvi_idx{0:05d}.npz")
    p99 = os.path.join(args.dataset_dir, f"lsc240420_id{traj_no:05d}_pvi_idx{99:05d}.npz")

    # skip unless BOTH files exist
    if not (os.path.exists(p0) and os.path.exists(p99)):
        print(f"Warning: Missing p0 or p99 for trajectory {traj_no}. Skipping.")
        continue

    with np.load(p0) as z:
        p0_list.append(z["av_density"].astype(np.float32))

    with np.load(p99) as z:
        p99_list.append(z["av_density"].astype(np.float32))
    
# guard against no valid trajectories found
assert len(p0_list) > 0, "No valid trajectories found"
assert len(p0_list) == len(p99_list), "Mismatch in number of p0 and p99 files loaded"

data_p0  = np.stack(p0_list, axis=0)[:,np.newaxis,...]  # (N,1,H,W)
data_p99 = np.stack(p99_list, axis=0)[:,np.newaxis,...]  # (N,1,H,W)
data = np.concatenate([data_p0, data_p99], axis=1)  # (N,2,H,W)

print(f"→ Loaded {len(p0_list)} first frames and {len(p99_list)} last frames.")
print(f"First shape: {data_p0.shape}, Last shape: {data_p99.shape}, Combined: {data.shape}")

# replace nans with zeros
nan_count = np.isnan(data).sum()
print(f"Found {nan_count} NaN values in the data. Replacing with zeros.")
test_data = np.nan_to_num(data, nan=0.0)

# split the data into train/val/test (80/10/10)
num_samples = data.shape[0]
num_trainval = int(0.9 * num_samples)
num_test = int(0.1 * num_samples)

trainval_data = data[:num_trainval]
test_data = data[num_trainval:]
print(f"Train/Val shape: {trainval_data.shape}, Test shape: {test_data.shape}")

del data
#############################################
############## --- normalization--- #########
#############################################

# perform normalization only on train and val set
mu_trainval = np.mean(trainval_data)
var_trainval = np.var(trainval_data)
norm_stats = [mu_trainval, var_trainval]
print(f"Train/Val mean: {mu_trainval:.6f}, Train/Val var: {var_trainval:.6f}")
np.save(os.path.join(norm_dir, "normstats_heat_avd.npy"), np.array(norm_stats))
print(f"→ Train/Val normalization stats saved to {norm_dir}")

# split into train and val set
num_train = int(0.9 * trainval_data.shape[0])
train_data = trainval_data[:num_train]
val_data = trainval_data[num_train:]
print(f"Train shape: {train_data.shape}, Val shape: {val_data.shape}") 

# normalize the train, val and test data using trainval stats
train_data_norm = (train_data - mu_trainval) / var_trainval
val_data_norm = (val_data - mu_trainval) / var_trainval
test_data_norm = (test_data - mu_trainval) / var_trainval
print(f"→ Train/Val/Test data normalized with shape: {train_data_norm.shape}, "
      f" {val_data_norm.shape}, {test_data_norm.shape}")

del trainval_data
#############################################
############## --- UPTF7 --- ################
#############################################

train_data_expand = np.expand_dims(train_data_norm, axis=[2,5,6])  # (N,T,H,W)->(N,T,D,H,W,C,F)
val_data_expand = np.expand_dims(val_data_norm, axis=[2,5,6])  # (N,T,H,W)->(N,T,D,H,W,C,F)
test_data_expand = np.expand_dims(test_data_norm, axis=[2,5,6])  # (N,T,H,W)->(N,T,D,H,W,C,F)

print(f"→ Train data shape after expanding dims: {train_data_expand.shape}")
print(f"→ Val data shape after expanding dims: {val_data_expand.shape}")
print(f"→ Test data shape after expanding dims: {test_data_expand.shape}")

##################################################
### ---- Define model and parallelization ---- ###
##################################################

# model configuration 
patch_size_pt = 8
max_patches, max_fields, max_components = 4096, 3, 3
filters, dim, heads, depth, mlp_dim = MORPH_MODELS[args.model_size]
dropout, emb_dropout = 0.1, 0.1

# set the finetuning level (lev) parameter
max_ar_order = 1 if args.model_size in ['Ti', 'S', 'M'] else 16
args.rank_lora_attn = 0
args.rank_lora_mlp = 0
args.lora_p = 0

# model
model_name = f"MORPH_FT_PLI-{args.model_size}"
ft_model = ViT3DRegression(patch_size = patch_size_pt, dim = dim, depth = depth,
        heads = heads, heads_xa = 32, mlp_dim = mlp_dim,
        max_components = max_components, conv_filter = filters, 
        max_ar = max_ar_order, max_patches = max_patches, max_fields = max_fields,
        dropout = dropout, emb_dropout = emb_dropout,
        lora_r_attn = args.rank_lora_attn,              # rank of A and B in the attention module
        lora_r_mlp = args.rank_lora_mlp,                # rank of A and B in the MLP module
        lora_alpha = None,                              # defaults to 2*rank inside LoRA
        lora_p = args.lora_p,                           # dropout on LoRA path
        model_size = args.model_size, 
        ).to(device)
# print('Model architecture:', ft_model)
num_params_model = sum(p.numel() for p in ft_model.parameters()) / 1e6
print(f"→ NUMBER OF PARAMETERS OF THE MODEL (in M): {num_params_model:.3g}")

##############################################
############ --- Data Parallel --- ###########
##############################################
n_gpus = torch.cuda.device_count()
print(f'→ Finetuning on {n_gpus} GPUs')
if n_gpus > 1:
    ft_model =  nn.DataParallel(ft_model)
    batch_size = n_gpus * args.batch_size
print(f'→ Overall Batch size for {batch_size}')
print(f'→ Parallelization on n GPUs: {n_gpus}')

##############################################
############ --- Dataloader --- ##############
##############################################
# prepare data into (inputs, targets)) format
print(f'→ Dataset preparation...')

preparer = FastARDataPreparer(ar_order = 1)
X_tr, y_tr = preparer.prepare(train_data_expand) # also converts to UPTF7 format (N,T,F,C,D,H,W)
X_va, y_va = preparer.prepare(val_data_expand) # also converts to UPTF7 format (N,T,F,C,D,H,W)
X_te, y_te = preparer.prepare(test_data_expand) # also converts to UPTF7 format (N,T,F,C,D,H,W)
print(f'→ Training Inputs: {X_tr.shape} and Targets: {y_tr.shape}')

ft_tr = DatasetforDataloader(X_tr, y_tr)
ft_va = DatasetforDataloader(X_va, y_va)
ft_te = DatasetforDataloader(X_te, y_te)

ft_tr_loader = DataLoader(ft_tr, batch_size=batch_size, shuffle=True)
ft_va_loader = DataLoader(ft_va, batch_size=batch_size, shuffle=False)
ft_te_loader = DataLoader(ft_te, batch_size=batch_size, shuffle=False)
print(f'→ Length dataloader: Tr {len(ft_tr_loader)}, Val {len(ft_va_loader)}')

###############################################
########### --- FINETUNING setup --- ##########
##############################################

# loss function
criterion = nn.MSELoss()

# optimizer and finetuning parameters
selector = SelectFineTuningParameters(ft_model, args)
optimizer = selector.configure_levels()
ft_model.train().to(device)
start_lr = optimizer.param_groups[0]['lr']
start_wd = optimizer.param_groups[0]['weight_decay']

# schedular
scheduler = LRSched.warmup_cosine(optimizer, args)
print(f'→ Min LR: {args.min_lr} | Warm epochs: {args.warm_epochs} '
    f'| LR: {start_lr} | Weight Decay: {start_wd}')

###############################################
########### --- Load weights --- ##############
###############################################

# ---- load the pretrained weights ----
start_epoch = 0
if args.ckpt_from == 'FM':
    print(f"→ Loading checkpoints from {args.ckpt_from}")
    # --- Load pretrained checkpoint from foundational model ---
    checkpoint_path = os.path.join(savepath_model, 'FM', args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"]

    target = ft_model.module if isinstance(ft_model, nn.DataParallel) else ft_model

    ckpt_has_module = state_dict and next(iter(state_dict)).startswith("module.")
    tgt_has_module  = any(k.startswith("module.") for k in target.state_dict().keys())

    if ckpt_has_module and not tgt_has_module:
        print("→ Stripping 'module.' from checkpoints")
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    elif (not ckpt_has_module) and tgt_has_module:
        print("→ Adding 'module.' to checkpoints")
        state_dict = {"module." + k: v for k, v in state_dict.items()}

    missing, unexpected = target.load_state_dict(state_dict, strict=False)

    # sanity print
    print("Missing keys (expected: LoRA A/B etc.):",
          [k for k in missing if k.endswith((".A", ".B")) or ".lora" in k])
    print("Unexpected keys:", unexpected)
    print(f"→ Resumed from {checkpoint_path}, starting at epoch {start_epoch}")
    
elif args.ckpt_from == 'FT':
    print(f"→ Loading checkpoints from {args.ckpt_from}")
    # ---- resume checkpoint from previous finetuned epochs ----
    resume_path = os.path.join(savepath_model, f'{args.ft_dataset}', args.checkpoint)
    ckpt = torch.load(resume_path, map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"]
    
    # pick the real model if wrapped
    target = ft_model.module if isinstance(ft_model, nn.DataParallel) else ft_model 
    
    if any(k.startswith("module.") for k in state_dict.keys()):
        print("→ Stripping 'module.' from checkpoints")
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        
    target.load_state_dict(state_dict, strict=True)
    
    # set optimizer from previous checkpoint
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt["epoch"]
    print(f"→ Resumed from {resume_path}, starting at epoch {start_epoch}")

else:
    print('→ No model is loaded. Running standalone with random init weights.')

##############################################
########### --- training --- #################
##############################################

model_path = os.path.join(savepath_model, "pli_new", model_name)
os.makedirs(model_path, exist_ok=True)
best_val_loss = float('inf')
epochs_no_improve = 0
ep_st = time.time()
history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "lr": [],
    "wall_time_s": [],
    }
for epoch in range(start_epoch, args.num_epochs):            
    tr_loss = Trainer.train_singlestep(ft_model, ft_tr_loader, criterion, optimizer, device,
                                       epoch, scheduler, model_path, 
                                       args.save_batch_ckpt, args.save_batch_freq)
    vl_loss = Trainer.validate_singlestep(ft_model, ft_va_loader, criterion, device)

    # --- step the schedular ---
    scheduler.step()     
    
    # Get current LR (from first param group)
    current_lr = optimizer.param_groups[0]['lr']

    # epoch time
    epoch_time = (time.time()-ep_st)/60
    
    print(f"Time = {epoch_time:.2f} min., LR:{current_lr:.6f}, "
        f"Epoch {epoch+1}/{args.num_epochs} |"
        f"Train:{tr_loss:.5f}, Val:{vl_loss:.5f}")
    
    # --- store training logging info ---
    history["epoch"].append(epoch)
    history["train_loss"].append(float(tr_loss))
    history["val_loss"].append(float(vl_loss))
    history["lr"].append(float(current_lr))
    history["wall_time_s"].append(float(epoch_time))

    if (epoch + 1) > args.warm_epochs: 
        # --early stopping logic ---
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            epochs_no_improve = 0
            
            # --- save checkpoint ---
            if (epoch + 1) % args.save_every == 0:
                checkpoint = {"epoch": epoch + 1,
                            "model_state_dict": ft_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "args": args}
                ckpt_path = f"{model_path}_ep{epoch+1}.pth"
                torch.save(checkpoint, ckpt_path)
                print(f" Saved checkpoint: {ckpt_path}")
    
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{args.patience} epochs")

        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered Validation loss did not improve for {args.patience} epochs.")
            break
    
# save history as json
hist_path = os.path.join(savepath_results, f"zz_history__{model_name}.json")
with open(hist_path, "w") as f:
    json.dump(history, f, indent=2)
print(f" Saved history: {hist_path}")

# save the learning curve                
fig, ax = plt.subplots()
ax.plot(history["train_loss"], label='Train')
ax.plot(history["val_loss"], label='Val')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend()
fig.savefig(os.path.join(savepath_results, (f'learningcurve_{model_name}.png')))

###############################################
############ --- Evaluations --- ##############
###############################################

print(f"→ Evalutions...")
input_tensor_all = []
out_tensor_all = []
for test_sample in range(test_data_expand.shape[0]):
    with torch.no_grad():
        input_batch = test_data_norm[test_sample,0,...]  # (H,W)
        test_data_uptf7 = np.expand_dims(input_batch, axis=[0,1,2,3,4])  # (N,T,F,C,D,H,W)
        input_tensor = torch.from_numpy(test_data_uptf7).to(device)  # (N,T,F,C,D,H,W)
        _,_,out_tensor = ft_model(input_tensor)
        out_tensor_all.append(out_tensor.cpu().squeeze(dim=(1,2,3)))

out_tensor_all = torch.cat(out_tensor_all, dim=0) # (N,H,W)
print(f"→ Predictions obtained with shape: {out_tensor_all.shape}")

input_tensor_all = torch.from_numpy(test_data_norm[:,0,...]) # (N,H,W)
target_tensor_all = torch.from_numpy(test_data_norm[:,1,...])  # (N,H,W)
print(f"→ Input: {input_tensor_all.shape}, Target tensor shape: {target_tensor_all.shape}")

###############################################
############ --- Visualization --- ############
###############################################

# squeeze the dims
mse = F.mse_loss(out_tensor_all, input_tensor_all)
print(f"→ Overall MSE between input and output: {mse.item():.6f}")

# unnormalize the data for visualization
input_tensor_unnorm = input_tensor_all.cpu() * var_trainval + mu_trainval
out_tensor_unnorm = out_tensor_all.cpu() * var_trainval + mu_trainval

# plot input and output side by side
test_sample = torch.randint(0, input_tensor_unnorm.shape[0], (1,)).item() 
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
im0 = axs[0].imshow(input_tensor_unnorm[test_sample], cmap='viridis')
axs[0].set_title("Input (Unnormalized)")
fig.colorbar(im0, ax=axs[0])
im1 = axs[1].imshow(out_tensor_unnorm[test_sample], cmap='viridis')
axs[1].set_title("Output (Unnormalized)")
fig.colorbar(im1, ax=axs[1])
plt.tight_layout()
plt.savefig(os.path.join(savepath_results, f"{model_name}_idx{test_sample}_pred.png"))