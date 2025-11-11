from fileinput import filename
from json import load
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MPLBACKEND"] = "Agg"      # safest: force non-GUI backend
matplotlib.use("Agg")                 # belt-and-suspenders

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

# load the classes
from src.utils.device_manager import DeviceManager
from src.utils.vit_conv_xatt_axialatt2 import ViT3DRegression
from src.utils.metrics_3d import Metrics3DCalculator
from src.utils.visualize_predictions_3d_full import Visualize3DPredictions
from src.utils.visualize_rollouts_3d_full import Visualize3DRolloutPredictions
from src.utils.data_preparation_fast import FastARDataPreparer
from config.data_config import DataConfig
from src.utils.normalization import RevIN
from src.utils.uptf7 import UPTF7
from pathlib import Path
from sklearn.model_selection import train_test_split

#%% Argument parser
MORPH_MODELS = {
    'Ti': [8, 256,  4,  4, 1024],
    'S' : [8, 512,  8,  4, 2048],
    'M' : [8, 768, 12,  8, 3072],
    'L' : [8, 1024,16, 16, 4096]
    }

# +++ instantiate argument parsers +++
patch_size = 8
DATA_CONFIG = DataConfig(project_root, patch_size)

# --- set arguments ---
parser = argparse.ArgumentParser(description="Run inference on trained ViT3D model")
parser.add_argument('--dataset', type=str, help="Path to saved .npy/.pth", required=True)
parser.add_argument('--dataset_name', type=str, help="Name of the dataset", default=None)
parser.add_argument('--dataset_specs', nargs=5, type=int, metavar=('F','C','D','H','W'),
                    default=[2,1,1,128,128], help="Dataset specs")
parser.add_argument('--model_choice', type=str, default = 'FM', help = "Model to finetune")
parser.add_argument('--download_model', action='store_true', help="Download model weights")
parser.add_argument('--model_size', type=str, choices = list(MORPH_MODELS.keys()),
                    default='Ti', help='choose from Ti, S, M, L')
parser.add_argument('--ckpt_from', type=str, choices = ['FM','FT'], default = 'FM',
                    help="Checkpoint information from FM or previous FT", required = True)
parser.add_argument('--checkpoint', type=str, default=None,
                    help="Path to saved .pth state dict if download_model is False")

# --- Finetune levels ---                     
parser.add_argument('--ft_level1', action='store_true', help = "Level-1 finetuning (LoRA, PE, LN)")
parser.add_argument('--ft_level2', action='store_true', help = "Level-2 finetuning (Encoder)")
parser.add_argument('--ft_level3', action='store_true', help = "Level-3 finetuning (Decoder)")
parser.add_argument('--ft_level4', action='store_true', help = "All model parameters")
parser.add_argument('--lr_level4', type = float, default = 1e-4, 
                    help = "Learning rate for level-4 finetuning")
parser.add_argument('--wd_level4', type = float, default = 0.0, 
                    help = "Weight decay for level-4 finetuning")

# --- parallelization ---
parser.add_argument('--parallel', type=str, choices=['dp','no'], default='dp', 
                    help="DataParallel vs No parallelization")

# ---lora parameters ---
parser.add_argument('--rank_lora_attn', type = int, default = 16, 
                    help = "Rank of attention layers in transformer module")
parser.add_argument('--rank_lora_mlp', type = int, default = 12, 
                    help = "Rank of MLP layers in transformer module")
parser.add_argument('--lora_p', type = float, default = 0.05, 
                    help = "Dropout inside LoRA layers")

# --- data and compute ---
parser.add_argument('--n_epochs', type=int, default = 150, help="Fine-tuning epochs")
parser.add_argument('--n_traj', type=int, help="Fine-tuning trajectories")
parser.add_argument('--rollout_horizon', type = int, default = 10, 
                    help = "Visualization: single step & rollouts")

# -- model default hyperparameters ---
parser.add_argument('--tf_reg', nargs=2, type=float, metavar=('dropout','emb_dropout'),
                    default=[0.1,0.1], help="Transformer regularization: dropouts")
parser.add_argument('--heads_xa', type=int, default=32, help = "Number of heads of cross attention")
parser.add_argument('--ar_order', type=int, default=1, help = "Autoregressive order of the data")
parser.add_argument('--max_ar_order', type=int, default=1, help="Max autoregressive order for the model")
parser.add_argument('--test_sample', type=int, default=0, help="Sample to plot from the test set")
parser.add_argument('--device_idx', type=int, default=0, help="CUDA device index to run on")
parser.add_argument('--patience', type=int, default=10, help="Early stopping criteria")

# --- optimizer hyperparameters ---
parser.add_argument('--batch_size', type=int, default=8, help="Batch size for finetuning")
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay")
parser.add_argument('--lr_scheduler', action='store_true', help="Use LR scheduler")

# --- save related ---
parser.add_argument('--overwrite_weights', action='store_true', 
                    help = "Over-ride previous checkpoints (saves storage)")
parser.add_argument('--save_every', type=int, default=1, help = "Save epochs at intervals")
parser.add_argument('--save_batch_ckpt', action='store_true', help = "Save batch checkpoints")
parser.add_argument('--save_batch_freq', type=int, default=1000, help = "Batch checkpoints frequency")

# --- set the parser for defining parameters ---
args = parser.parse_args()
devices = DeviceManager.list_devices()
filters, dim, heads, depth, mlp_dim = MORPH_MODELS[args.model_size]
dropout, emb_dropout = args.tf_reg
device = devices[args.device_idx] if devices else 'cpu'

# +++ default paths +++
savepath_results = os.path.join(project_root, "experiments", "results", "test")
os.makedirs(savepath_results, exist_ok=True)
savepath_model = os.path.join(project_root, "models")
os.makedirs(savepath_model, exist_ok=True)
loadpath_muvar = os.path.join(project_root, 'data')
os.makedirs(loadpath_muvar, exist_ok=True)
loadpath_datasets = os.path.join(project_root, 'datasets')
os.makedirs(loadpath_datasets, exist_ok=True)

# +++ Load dataset ++++
print(f'→ Loading dataset ...')
ext = Path(args.dataset).suffix.lower()
print(f'→ Dataset format: {ext}')

if ext == ".npy":
    print(f'→ Loading .npy dataset from {args.dataset}')
    dataset = np.load(os.path.join(loadpath_datasets, args.dataset))     # -> np.ndarray
    print("Original dataset shape", dataset.shape)
else:
    raise ValueError("Unsupported dataset format. Need a .npy file.")

# +++ download model if needed +++
if args.download_model and args.ckpt_from == 'FM' and args.checkpoint == None:
    print(f'→ Downloading model weights from hugging face...')
    if args.model_size == 'Ti':
        fname = 'morph-Ti-FM-max_ar1_ep225.pth'
    elif args.model_size == 'S':
        fname = 'morph-S-FM-max_ar1_ep225.pth'
    elif args.model_size == 'M':
        fname = 'morph-M-FM-max_ar1_ep290_latestbatch.pth'
    elif args.model_size == 'L':
        fname = 'morph-L-FM-max_ar16_ep189_latestbatch.pth'

    # e.g., grab the "Ti" checkpoint (change filename as needed)
    weights_path = hf_hub_download(
        repo_id="mahindrautela/MORPH",
        filename=fname,
        subfolder="models/FM",          # <local_dir>/<subfolder>/<filename>
        repo_type="model",              # optional
        resume_download=True,           # continue if interrupted
        local_dir=".",                  # download to current directory 
       local_dir_use_symlinks=False    # copy file instead of symlink
    )
    print(f'MORPH-FM-{args.model_size} weights saved to {weights_path}')

# number of samples and trajectory length
num_samples = dataset.shape[0]
traj_len = dataset.shape[1]

# +++ convert to UTPF7 format (N,T,F,C,D,H,W) +++
dataset_uptf7 = UPTF7(
    dataset=dataset, # numpy array
    num_samples=num_samples,
    traj_len=traj_len,
    fields=args.dataset_specs[0],
    components=args.dataset_specs[1],
    image_depth=args.dataset_specs[2],
    image_height=args.dataset_specs[3],
    image_width=args.dataset_specs[4]
).transform()
print("UPTF7 dataset shape", dataset_uptf7.shape)

# +++ REVIN normalization +++
dataset_name = 'name' if args.dataset_name is None else args.dataset_name
revin = RevIN(loadpath_muvar)
norm_prefix = f'norm_{dataset_name}'

# normalize data
revin.compute_stats(dataset_uptf7, prefix = norm_prefix) 
dataset_norm = revin.normalize(dataset_uptf7, prefix = norm_prefix)
print("Normalize dataset shape", dataset_norm.shape)

# Check round‐trip via denormalize
recovered = revin.denormalize(dataset_norm, prefix=norm_prefix)
tol_6 = 1e-2
max_error = 0.0
for i in range(recovered.shape[0]):
    maxerror_i = np.max(np.abs(recovered[i] - dataset_uptf7[i]))  # saving some memory
    max_error = max(maxerror_i, max_error)
assert max_error < tol_6, "Denormalization did not perfectly recover original!"
print("CFD3D RevIN round-trip OK")
del recovered

# +++ set the finetuning level (lev) parameter +++
if args.ft_level4:
    lev = 4
    args.ft_level1 = args.ft_level2 = args.ft_level3 = None
    args.rank_lora_attn = 0
    args.rank_lora_mlp = 0
    args.lora_p = 0
elif args.ft_level1 and args.ft_level2 and args.ft_level3:
    lev = 3
elif args.ft_level1 and args.ft_level2:
    lev = 2
elif args.ft_level1:
    lev = 1
else:
    raise ValueError("Select a fine-tuning level: --ft_level1/2/3 or --ft_level4")
print(f"→ Set Level-{lev} fine-tuning")

#%% Create dataloaders
# +++ Dataset for Dataloader +++
class DatasetforDataloader(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# +++ Load data via dataloaders +++
print("→ Data splitting...")
dataset_norm_rs = dataset_norm.transpose(0,1,4,5,6,3,2)    # already in (N,T,D,H,W,F,C)
train_data, tmp = train_test_split(dataset_norm_rs, test_size=0.2, random_state=42, shuffle=True)
val_data, test_data = train_test_split(tmp, test_size=0.5, random_state=42, shuffle=True)

# +++ select trajectories +++
n_traj = train_data.shape[0] if args.n_traj is None else args.n_traj
print(f'→ [{dataset_name}] Number of finetuning trajectories: {n_traj}')

# +++ Prepare data into (X,y) +++
print(f'→ [{dataset_name}] Dataset preparation...')
preparer = FastARDataPreparer(ar_order = args.ar_order)
X_tr, y_tr = preparer.prepare(train_data[0:n_traj])  # also converts (N,T,D,H,W,C,F) -> (N,T,F,C,D,H,W)
X_va, y_va = preparer.prepare(val_data) 
X_te, y_te = preparer.prepare(test_data)             
print(f'→ Training Inputs: {X_tr.shape} and Targets: {y_tr.shape}')
assert X_tr.shape[0] == n_traj * (train_data.shape[1] - 1), "Shape mismatch !!"

# free some memory
del train_data, val_data

#%% Define model architecture
patch_size  = 8
max_patches = 4096
max_fields = 3
max_components = 3

model_name = (f'ft_morph-{args.model_size}-{dataset_name}-max_ar{args.max_ar_order}_'
f'rank-lora{args.rank_lora_attn}_ftlevel{lev}_lr{args.lr_level4}_wd{args.wd_level4}')

ft_model = ViT3DRegression(
    patch_size=patch_size, dim=dim, depth=depth,
    heads=heads, heads_xa=args.heads_xa, mlp_dim=mlp_dim,
    max_components=max_components, conv_filter=filters,
    max_ar=args.max_ar_order,
    max_patches=max_patches, max_fields=max_fields,
    dropout=dropout, emb_dropout=emb_dropout,
    lora_r_attn=args.rank_lora_attn,            # <— rank of A and B in the attention module
    lora_r_mlp=args.rank_lora_mlp,              # <— rank of A and B in the MLP module
    lora_alpha=None,                            # defaults to 2*rank inside LoRA
    lora_p=args.lora_p                          # dropout on LoRA path
).to(device)

# print('Model architecture:', ft_model)
num_params_model = sum(p.numel() for p in ft_model.parameters()) / 1e6
print(f"→ NUMBER OF PARAMETERS OF THE MODEL (in M): {num_params_model:.3g}")

# +++ Parallelization +++
n_gpus = torch.cuda.device_count()
print(f'→ Finetuning on {n_gpus} GPUs')
batch_size = args.batch_size
if args.parallel == 'dp' and n_gpus > 1:
    ft_model =  nn.DataParallel(ft_model)
    batch_size = n_gpus * args.batch_size
print(f'→ Selected (Overall) Batch size for {dataset_name} is {batch_size}')

# +++ Create dataloaders +++
ft_tr = DatasetforDataloader(X_tr, y_tr)
ft_va = DatasetforDataloader(X_va, y_va)
ft_te = DatasetforDataloader(X_te, y_te)

ft_tr_loader = DataLoader(ft_tr, batch_size=batch_size, shuffle=True)
ft_va_loader = DataLoader(ft_va, batch_size=batch_size, shuffle=False)
ft_te_loader = DataLoader(ft_te, batch_size=batch_size, shuffle=False)
print(f'→ Length dataloader: Tr {len(ft_tr_loader)}, Val {len(ft_va_loader)}, Te {len(ft_te_loader)}')

#%% Fine-tuning setup
from src.utils.select_fine_tuning_parameters import SelectFineTuningParameters
selector = SelectFineTuningParameters(ft_model, args)
optimizer = selector.configure_levels()
ft_model.train().to(device)
start_lr = optimizer.param_groups[0]['lr']
start_wd = optimizer.param_groups[0]['weight_decay']
print(f'→ Starting from LR: {start_lr} & Weight Decay: {start_wd}')

#%% Loss function and schedular
# ---- define the loss function ----
criterion = nn.MSELoss()
# ---- learning rate scheduler ----
scheduler = None
if args.lr_scheduler:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                        factor=0.5, patience=5)

#%% Load weights
# ---- load the pretrained weights ----
start_epoch = 0
if args.ckpt_from == 'FM':
    print(f"→ Loading checkpoints from {args.ckpt_from}")
    # --- Load pretrained checkpoint from foundational model ---
    if args.checkpoint is None:
        checkpoint = fname
    else:
        checkpoint = args.checkpoint
    checkpoint_path = os.path.join(savepath_model, f'{args.model_choice}', checkpoint)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"]
    
    # pick the real model if wrapped
    target = ft_model.module if isinstance(ft_model, nn.DataParallel) else ft_model 

    if state_dict and next(iter(state_dict)).startswith("module.") and args.parallel == 'no':
        print("→ Stripping 'module.' from checkpoints")
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        
    # strict=False because ft_model has extra LoRA params (A/B) not in ckpt
    flag = True if args.ft_level4 else False
    missing, unexpected = target.load_state_dict(state_dict, strict=flag)
    
    # sanity print
    print("Missing keys (expected: LoRA A/B etc.):",
          [k for k in missing if k.endswith((".A", ".B")) or ".lora" in k])
    print("Unexpected keys:", unexpected)
    print(f"→ Resumed from {checkpoint_path}, starting at epoch {start_epoch}")
    
elif args.ckpt_from == 'FT':
    print(f"→ Loading checkpoints from {args.ckpt_from}")
    # ---- resume checkpoint from previous finetuned epochs ----
    resume_path = os.path.join(savepath_model, f'{dataset_name}', args.checkpoint)
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
    print('No model is loaded')
    
#%% Fine tuning 
import time
from src.utils.trainers import Trainer
savepath_model_folder = os.path.join(savepath_model, f'{dataset_name}')
os.makedirs(savepath_model_folder, exist_ok=True)
model_path = os.path.join(savepath_model_folder, model_name)
train_losses, val_losses = [], []
best_val_loss = float('inf')
epochs_no_improve = 0
ep_st = time.time()
for epoch in range(start_epoch, args.n_epochs):            
    tr_loss = Trainer.train_singlestep(ft_model, ft_tr_loader, criterion, optimizer, device,
                                       epoch, scheduler, model_path, 
                                       args.save_batch_ckpt, args.save_batch_freq)
    vl_loss = Trainer.validate_singlestep(ft_model, ft_va_loader, criterion, device)

    train_losses.append(tr_loss)
    val_losses.append(vl_loss)
    
    if args.lr_scheduler:
        scheduler.step(vl_loss)
    
    # Get current LR (from first param group)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Time = {(time.time()-ep_st)/60:.2f} min., LR:{current_lr:.6f}, Epoch {epoch+1}/{args.n_epochs} |"
      f"Train:{tr_loss:.5f}, Val:{vl_loss:.5f}")
    
    # --early stopping logic ---
    if vl_loss < best_val_loss:
        best_val_loss = vl_loss
        epochs_no_improve = 0
        print(f" New best validation loss: {best_val_loss:.5f}")
        # --- save checkpoint ---
        if (epoch + 1) % args.save_every == 0:
            checkpoint = {"epoch": epoch + 1,
                          "model_state_dict": ft_model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict()}
            if args.overwrite_weights:
                ckpt_path = f"{model_path}_tot-trajs{n_traj}_tot-eps{args.n_epochs}.pth"
            else:
                ckpt_path = f"{model_path}_tot-trajs{n_traj}_tot-eps{args.n_epochs}_ep{epoch+1}.pth"
            torch.save(checkpoint, ckpt_path)
            print(f" Saved checkpoint: {ckpt_path}")
    
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve}/{args.patience} epochs")

    if epochs_no_improve >= args.patience:
        print(f"Early stopping triggered Validation loss did not improve for {args.patience} epochs.")
        break
                    
    fig, ax = plt.subplots()
    ax.plot(train_losses, label='Train')
    ax.plot(val_losses, label='Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend()
    fig.savefig(os.path.join(savepath_results, (f'loss_{args.model_choice}_{dataset_name}_'
               f'max_ar_{args.max_ar_order}_rank-lora{args.rank_lora_attn}_ftlevel{lev}'
               f'_lr{args.lr_level4}_wd{args.wd_level4}.png')))

# free some memory
del X_tr, y_tr, X_va, y_va

#%% Calculate metrics
print(f"→[{dataset_name}] Evalutions Metrics...")
mse_tot = mae_tot = 0.0
n_samp = 0
out_all, tar_all = [],[]
# predictions from the trained model
with torch.no_grad():
    for inp, tar in tqdm(ft_te_loader):
        inp = inp.to(device)
        _,_, out = ft_model(inp)
        out_all.append(out.detach().cpu())
        tar_all.append(tar)
out_all = torch.concat(out_all, dim = 0)
tar_all = torch.concat(tar_all, dim = 0) 
print(f'→ [N*(T-1),F,C,D,H,W] Predictions: Outputs: {out_all.shape} and Targets: {tar_all.shape}')

# calculate MSE and MAE (normalized scale with samples = N*T)
mse = F.mse_loss(out_all, tar_all, reduction='mean')
mae = F.l1_loss(out_all, tar_all, reduction='mean')
rmse = mse**0.5

# reshape outputs and targets to shape of the test set
td_out = torch.from_numpy(test_data[:,1:]) # after initial frame
td_out = td_out.permute(0, 1, 6, 5, 2, 3, 4) # (N,T,D,H,W,C,F) -> (N,T,F,C,D,H,W)
out_all_rs = out_all.reshape(td_out.shape)
tar_all_rs = tar_all.reshape(td_out.shape)
print(f'→ [N,T-1,F,C,D,H,W] Reshaped Outputs: {out_all_rs.shape} and Targets: {tar_all_rs.shape}')

# denormalize the outputs and targets
outputs_denorm = RevIN.denormalize_testeval(loadpath_muvar, norm_prefix, out_all_rs,
                                            dataset = dataset_name)
targets_denorm = RevIN.denormalize_testeval(loadpath_muvar, norm_prefix, tar_all_rs, 
                                            dataset = dataset_name)
print(f'→ [N,T-1,F,C,D,H,W] Denormalized outputs {outputs_denorm.shape} and targets shape {targets_denorm.shape}')

assert out_all_rs.shape == outputs_denorm.shape, 'Norm and Denorm shapes (Outputs) dont match'
assert tar_all_rs.shape == targets_denorm.shape, 'Norm and Denorm shapes (Targets) dont match'

# calculate VRMSE and NRMSE (denormalized scale with samples = N)
vrmse = Metrics3DCalculator.calculate_VRMSE(outputs_denorm, targets_denorm).mean()
nrmse = Metrics3DCalculator.calculate_NRMSE(outputs_denorm, targets_denorm).mean()
      
# average value across the test set
print(f'→ RMSE: {rmse:.5f}, MAE: {mae:.5f}, MSE: {mse:.5f}'
      f' VRMSE: {vrmse:.5f}, NRMSE: {nrmse:.5f}')

# Store the results
savepath_results_ = os.path.join(savepath_results, f'{dataset_name}')
os.makedirs(savepath_results_, exist_ok=True)
metrics_str = (f" MAE: {mae:.5f}, MSE: {mse:.5f}, RMSE: {rmse:.5f},"
               f" NRMSE: {nrmse:.5f}, VRMSE: {vrmse:.5f}")
metrics_name = os.path.join(savepath_results_, (f'metrics_MORPH-{args.model_size}_'
              f'{args.model_choice}_ar{args.max_ar_order}_tot-trajs{n_traj}'
              f'_tot-eps{args.n_epochs}_rank-lora{args.rank_lora_attn}_ftlevel{lev}'
              f'_lr{args.lr_level4}_wd{args.wd_level4}.txt'))
with open(metrics_name, "w") as f:
    f.write(metrics_str)
print(f"→ Metrics written to {metrics_name}")

#%% Single-step predictions
#test_sample = np.random.randint(0, test_data.shape[0], 1).item() 
sim = test_data[args.test_sample] #(T,F,C,D,H,W)
sim_rs = np.transpose(sim, (0,5,4,1,2,3)).astype(np.float32) #(T,F,C,D,H,W)
sim_tensor = torch.from_numpy(sim_rs).unsqueeze(0)  #(N=1,T,F,C,D,H,W)
field_names = [f'field-{fi}' for fi in range(1, sim_tensor.shape[2] + 1)]

#%% Single-step predictions
print(f'→[{dataset_name}] Next step predictions...')
viz = Visualize3DPredictions(ft_model, sim_tensor, device)

figurename = (f'ft_st_MORPH-{args.model_size}_{args.model_choice}_ar{args.max_ar_order}_chAll_'
f'samp{args.test_sample}_tot-trajs{n_traj}_tot-eps{args.n_epochs}_'
f'rank-lora{args.rank_lora_attn}_ftlevel{lev}_lr{args.lr_level4}_wd{args.wd_level4}_t')

for t in range(args.rollout_horizon):
    viz.visualize_predictions(time_step=t, component=0, slice_dim='d',
                              save_path=savepath_results_,
                              figname=f'{figurename}{t}.png')

#%% Rollout predictions
print(f'→[{dataset_name}] Rollout predictions...')
viz_roll = Visualize3DRolloutPredictions(model=ft_model, test_dataset=sim_tensor,
                                         device=device,
                                         field_names=field_names,
                                         component_names=["d","h","w"])

figurename = (f'ft_ro_MORPH-{args.model_size}_{args.model_choice}_ar{args.max_ar_order}_tAll_'
f'samp{args.test_sample}_tot-trajs{n_traj}_tot-eps{args.n_epochs}_'
f'rank-lora{args.rank_lora_attn}_ftlevel{lev}_lr{args.lr_level4}_wd{args.wd_level4}_field')

for f in range(sim_tensor.shape[2]):
    viz_roll.visualize_rollout(start_step=0, num_steps=args.rollout_horizon, field=f,
                               component=0, slice_dim='d',
                               save_path=savepath_results_,
                               figname=f'{figurename}{f}.png')
