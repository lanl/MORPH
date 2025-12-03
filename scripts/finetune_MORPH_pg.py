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
from src.utils.normalization import RevIN
from src.utils.dataloaders.dataloader_ce_2d import CE2dDataLoader
from src.utils.dataloaders.dataloader_ns_2d import NS2dDataLoader
from src.utils.dataloaders.dataloader_fns_kf_2d import FNSKF2dDataLoader

#%% Argument parser
MORPH_MODELS = {
    'Ti': [8, 256,  4,  4, 1024],
    'S' : [8, 512,  8,  4, 2048],
    'M' : [8, 768, 12,  8, 3072],
    'L' : [8, 1024, 16, 16, 4096]
    }

# ---- set arguments ----
parser = argparse.ArgumentParser(description="Run inference on trained ViT3D model")
parser.add_argument('--dataset_root', type = str, help = "Location of dataset")
parser.add_argument('--model_choice', type=str, default = 'FM_pdegym', help = "Model to finetune")
parser.add_argument('--model_size', type=str, choices = list(MORPH_MODELS.keys()),
                    default='Ti', help='choose from Ti, S, M, L')
parser.add_argument('--ckpt_from', type=str, choices = ['FM','FT'], default = 'FM',
                    help="Checkpoint information from FM or previous FT", required = True)
parser.add_argument('--checkpoint', type=str, help="Path to saved .pth state dict", required=True)
parser.add_argument('--data_stats_name', type=str, help = "Data statistics for normalization")
parser.add_argument('--ft_dataset', choices=['CE-RPUI', 'CE-RM', 'NS-PwC', 'NS-SL','FNS-KF'], 
                    type=str, default = 'CE-RPUI', 
                    help = "Choose the finetuning set")
parser.add_argument('--patch_size', type=int, default=4, help='patch size for the model')
parser.add_argument('--conv_layer1_filters', type=int, default=8)
parser.add_argument('--scale_filter', type=int, default=2)
parser.add_argument('--conv_off', action='store_true', 
                            help='Whether to use convolutional front-end')
parser.add_argument('--replace_xatt', action='store_true', 
                    help='Replace cross-attention with concat+projection')
parser.add_argument('--attn_type', type=str, choices=['axial', 'full', 'sparse_random'], 
                    default='axial', help='Type of attention to use in the transformer blocks')
parser.add_argument('--activated_ar1k', action='store_true',
                    help = 'train time-axial-attention everytime')
parser.add_argument('--pos_enc', type=str, choices=['axial-pe', 'full-pe'], default='full-pe', 
                    help='type of positional encoding')
# --- Finetune levels ---                     
parser.add_argument('--ft_level1', action='store_true', help = "Level-1 finetuning (LoRA, PE, LN)")
parser.add_argument('--ft_level2', action='store_true', help = "Level-2 finetuning (Encoder)")
parser.add_argument('--ft_level3', action='store_true', help = "Level-3 finetuning (Decoder)")
parser.add_argument('--ft_level4', action='store_true', help = "All model parameters")
parser.add_argument('--lr_level4', type = float, default = 1e-4, 
                    help = "Learning rate for level-4 finetuning")
parser.add_argument('--wd_level4', type = float, default = 1e-6, 
                    help = "Weight decay for level-4 finetuning")

# --- parallelization ---
parser.add_argument('--parallel', type=str, choices=['dp','no'], default='dp', 
                    help="DataParallel vs No parallelization")
parser.add_argument('--device_idx', type=int, default=0, help="CUDA device index to run on")

# ---lora parameters ---
parser.add_argument('--rank_lora_attn', type = int, default = 16, 
                    help = "Rank of attention layers in transformer module")
parser.add_argument('--rank_lora_mlp', type = int, default = 12, 
                    help = "Rank of MLP layers in transformer module")
parser.add_argument('--lora_p', type = float, default = 0.05, 
                    help = "Dropout inside LoRA layers")

# --- data and compute ---
parser.add_argument('--n_epochs', type=int, default = 200, help="Fine-tuning epochs")
parser.add_argument('--n_traj_train', type=int, default = 128, help="Fine-tuning trajectories")
parser.add_argument('--n_traj_test', type=int, default = 240, help="Testing trajectories")
parser.add_argument('--rollout_horizon', type = int, default = 20, 
                    help = "Visualization: single step & rollouts")
parser.add_argument('--batch_size', type=int, default = 40, help="Batch size for loaders")

# -- model default hyperparameters ---
parser.add_argument('--tf_reg', nargs=2, type=float, metavar=('dropout','emb_dropout'),
                    default=[0.1,0.1], help="Transformer regularization: dropouts")
parser.add_argument('--heads_xa', type=int, default=8, help = "Number of heads of cross attention")
parser.add_argument('--ar_order', type=int, default=1, help = "Autoregressive order of the data")
parser.add_argument('--max_ar_order', type=int, default=1, help="Max autoregressive order for the model")
parser.add_argument('--test_sample', type=int, default=0, help="Sample to plot from the test set")
parser.add_argument('--early_stopping', action='store_true', help = 'Enable early stopping during training')
parser.add_argument('--patience', type=int, default=50, help="Early stopping criteria")

# --- save related ---
parser.add_argument('--overwrite_weights', action='store_true', 
                    help = "Over-ride previous checkpoints (saves storage)")
parser.add_argument('--save_every', type=int, default=1, help = "Save epochs at intervals")
parser.add_argument('--save_batch_ckpt', action='store_true', help = "Save batch checkpoints")
parser.add_argument('--save_batch_freq', type=int, default=1000, help = "Batch checkpoints frequency")
parser.add_argument('--save_truepred_tensors', action='store_true', help = "Save true and predicted tensors for analysis")  

# --- set the parser for defining parameters ---
args = parser.parse_args()
devices = DeviceManager.list_devices()
filters, dim, heads, depth, mlp_dim = MORPH_MODELS[args.model_size]
dropout, emb_dropout = args.tf_reg
device = devices[args.device_idx] if devices else 'cpu'

# --- set the model configuration ---
model_choice = args.model_choice
model_size = args.model_size
checkpoint = args.checkpoint

# --- set the test data congfiguration ---
ft_dataset = args.ft_dataset
n_epochs = args.n_epochs

# norm_prefix - prefix of mu, var files
norm_prefix = args.data_stats_name
print(f'→ Selected Model MORPH-{model_choice}-{model_size}')

# --- set the finetuning level (lev) parameter ---
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

#%% Folder locations
if args.dataset_root is None:
    dataset_root = project_root
else:
    dataset_root = args.dataset_root
print(f"→ Current dataset root: {dataset_root}")

# --- dataset locations ---
# location of REVIN data
datasets = ["2dCE_RPUI_pdegym","2dCE_RM_pdegym", "2dNS_PwC_pdegym", "2dNS_SL_pdegym", "2dFNS_KF_pdegym"]

#--- finetune sets ---
datapath_ce_rpui = os.path.join(dataset_root,'datasets', 'normalized_revin', datasets[0])
datapath_ce_rm = os.path.join(dataset_root,'datasets', 'normalized_revin', datasets[1])
datapath_ns_pwc = os.path.join(dataset_root,'datasets', 'normalized_revin', datasets[2])
datapath_ns_sl = os.path.join(dataset_root,'datasets', 'normalized_revin', datasets[3])
datapath_fns_kf = os.path.join(dataset_root,'datasets', 'normalized_revin', datasets[4])

datapaths = {'CE-RPUI': datapath_ce_rpui,'CE-RM': datapath_ce_rm, 'NS-PwC': datapath_ns_pwc, 
             'NS-SL': datapath_ns_sl, 'FNS-KF': datapath_fns_kf}

# batch size
batch_size = args.batch_size if args.batch_size is not None else 40

# savepaths
savepath_results = os.path.join(project_root, "experiments", "results", "test")
os.makedirs(savepath_results, exist_ok=True)

# savepaths
savepath_model = os.path.join(project_root, "models")

# savepath of mu and var
loadpath_muvar = os.path.join(project_root, 'data')

# Dataset for Dataloader
class DatasetforDataloader(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
#%% Load Test dataset  
# --- Load data via dataloaders ----
print("→ Loading test dataset...")
if ft_dataset == 'CE-RPUI' or ft_dataset == 'CE-RM':
    dataset = CE2dDataLoader(datapaths[ft_dataset], var_name='data' if ft_dataset == 'CE-RPUI' else 'solution')
elif ft_dataset == 'NS-PwC' or ft_dataset == 'NS-SL':
    dataset = NS2dDataLoader(datapaths[ft_dataset])
elif ft_dataset == 'FNS-KF':
    dataset = FNSKF2dDataLoader(datapaths[ft_dataset])
train_data, val_data = dataset.split_train()
test_data = dataset.split_test()
dataset = np.concatenate([train_data, val_data, test_data], axis = 0)

del dataset
print(f"[{ft_dataset}] Shape of train: {train_data.shape}, Val: {val_data.shape}," 
      f"Test data: {test_data.shape}")

# ---- determine data configuration based on {choice} and {surrogate_type} ----
max_resolution = 128 * 128
max_patches = max_resolution // (args.patch_size**2)
max_fields = 3
max_components = 3
print(f'→ Max patches: {max_patches}, Max fields: {max_fields}, '
      f' Max components: {max_components}')

#%% Define model and parallelization
# ---- Model init and load weights ----
# --- Rebuild model with LoRA ranks > 0 ---
model_name = (f'ft_morph-{args.model_size}-{ft_dataset}-max_ar{args.max_ar_order}_'
f'rank-lora{args.rank_lora_attn}_ftlevel{lev}_lr{args.lr_level4}_wd{args.wd_level4}')

ft_model = ViT3DRegression(
    patch_size=args.patch_size, dim=dim, depth=depth,
    heads=heads, heads_xa=args.heads_xa, mlp_dim=mlp_dim,
    max_components=max_components, conv_filter=filters,
    conv_layer1_filters = args.conv_layer1_filters, 
    scale_filter = args.scale_filter,
    max_ar=args.max_ar_order,
    max_patches=max_patches, max_fields=max_fields,
    dropout=dropout, emb_dropout=emb_dropout,
    lora_r_attn=args.rank_lora_attn,            # <— rank of A and B in the attention module
    lora_r_mlp=args.rank_lora_mlp,              # <— rank of A and B in the MLP module
    lora_alpha=None,                            # defaults to 2*rank inside LoRA
    lora_p=args.lora_p,                         # dropout on LoRA path
    activated_ar1k = args.activated_ar1k,
    conv_off = args.conv_off,
    replace_xatt = args.replace_xatt,
    attn_type=args.attn_type,
    pos_enc=args.pos_enc
).to(device)

# print('Model architecture:', ft_model)
num_params_model = sum(p.numel() for p in ft_model.parameters()) / 1e6
print(f"→ NUMBER OF PARAMETERS OF THE MODEL (in M): {num_params_model:.3g}")

#%% Parallelization
n_gpus = torch.cuda.device_count()
print(f'→ Finetuning on {n_gpus} GPUs')

if args.parallel == 'dp' and n_gpus > 1:
    ft_model =  nn.DataParallel(ft_model)
    batch_size = n_gpus * batch_size
print(f'→ Selected (Overall) Batch size for {ft_dataset} is {batch_size}')

#%% Prepare dataloader
print(f'→ [{ft_dataset}] Dataset preparation...')
preparer = FastARDataPreparer(ar_order = args.ar_order)

# select trajectories
n_traj_train = args.n_traj_train if args.n_traj_train is not None else train_data.shape[0]
print(f'→ [{ft_dataset}] Number of finetuning trajectories: {n_traj_train}')

X_tr, y_tr = preparer.prepare(train_data[0:n_traj_train]) # also converts (N,T,D,H,W,C,F) -> (N,T,F,C,D,H,W)
X_va, y_va = preparer.prepare(val_data[0:120]) # val data is 12.5% of train data
X_te, y_te = preparer.prepare(test_data) # also converts (N,T,D,H,W,C,F) -> (N,T,F,C,D,H,W)
print(f'→ Training Inputs: {X_tr.shape} and Targets: {y_tr.shape}')
assert X_tr.shape[0] == n_traj_train * (train_data.shape[1] - 1), "Shape mismatch !!"

# free some memory
del train_data, val_data

ft_tr = DatasetforDataloader(X_tr, y_tr)
ft_va = DatasetforDataloader(X_va, y_va)
ft_te = DatasetforDataloader(X_te, y_te)

ft_tr_loader = DataLoader(ft_tr, batch_size=batch_size, shuffle=True)
ft_va_loader = DataLoader(ft_va, batch_size=batch_size, shuffle=False)
ft_te_loader = DataLoader(ft_te, batch_size=batch_size, shuffle=False)
print(f'→ Length dataloader: Tr {len(ft_tr_loader)}, Val {len(ft_va_loader)}')
    
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
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=5)

#%% Load weights
# ---- load the pretrained weights ----
start_epoch = 0
if args.ckpt_from == 'FM':
    print(f"→ Loading checkpoints from {args.ckpt_from}")
    # --- Load pretrained checkpoint from foundational model ---
    checkpoint_path = os.path.join(savepath_model, f'{model_choice}', checkpoint)
    print(f"→ Loading checkpoint from {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"]
    
    # Is the current model wrapped?
    is_wrapped = isinstance(ft_model, (nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
    print(f"→ Current model is wrapped: {is_wrapped}")

    # Does the checkpoint have 'module.'?
    ckpt_has_prefix = any(k.startswith("module.") for k in state_dict)
    print(f"→ Checkpoint has 'module.' prefix: {ckpt_has_prefix}")

    if is_wrapped and ckpt_has_prefix:
        # wrapper ← prefixed ckpt
        target = ft_model
    elif is_wrapped and not ckpt_has_prefix:
        # inner module ← unprefixed ckpt
        target = ft_model.module
    elif (not is_wrapped) and ckpt_has_prefix:
        # plain model ← strip prefix from ckpt
        print("→ Stripping 'module.' from checkpoint keys")
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        target = ft_model
    else:
        # plain model ← plain ckpt
        target = ft_model
        
    # Load the state dict with strict=False if not finetuning the whole model, else strict=True
    if args.ft_level4: # finetuning whole model (no LoRA params)
        strict_flag = True
    else:
        strict_flag = False
    missing, unexpected = target.load_state_dict(state_dict, strict=strict_flag)
    
    # sanity print
    print("Missing keys (expected: LoRA A/B etc.):",
          [k for k in missing if k.endswith((".A", ".B")) or ".lora" in k])
    print("Unexpected keys:", unexpected)
    print(f"→ Resumed from {checkpoint_path}, starting at epoch {start_epoch}")
    
elif args.ckpt_from == 'FT':
    print(f"→ Loading checkpoints from {args.ckpt_from}")
    # ---- resume checkpoint from previous finetuned epochs ----
    resume_path = os.path.join(savepath_model, f'{ft_dataset}', checkpoint)
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
savepath_model_folder = os.path.join(savepath_model, f'{ft_dataset}')
os.makedirs(savepath_model_folder, exist_ok=True)
model_path = os.path.join(savepath_model_folder, model_name)
train_losses, val_losses = [], []
best_val_loss = float('inf')
epochs_no_improve = 0
ep_st = time.time()
for epoch in range(start_epoch, n_epochs):            
    tr_loss = Trainer.train_singlestep(ft_model, ft_tr_loader, criterion, optimizer, device,
                                       epoch, scheduler, model_path, 
                                       args.save_batch_ckpt, args.save_batch_freq)
    vl_loss = Trainer.validate_singlestep(ft_model, ft_va_loader, criterion, device)

    train_losses.append(tr_loss)
    val_losses.append(vl_loss)
    
    scheduler.step(vl_loss)
    
    # Get current LR (from first param group)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Time = {(time.time()-ep_st)/60:.2f} min., LR:{current_lr:.6f}, Epoch {epoch+1}/{n_epochs} |"
      f"Train:{tr_loss:.5f}, Val:{vl_loss:.5f}")
    
    # --early stopping logic ---
    if args.early_stopping:
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            epochs_no_improve = 0
            
            # --- save checkpoint ---
            if (epoch + 1) % args.save_every == 0:
                checkpoint = {"epoch": epoch + 1,
                            "model_state_dict": ft_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict()}
                if args.overwrite_weights:
                    ckpt_path = f"{model_path}_tot-trajs{n_traj_train}_tot-eps{n_epochs}.pth"
                else:
                    ckpt_path = f"{model_path}_tot-trajs{n_traj_train}_tot-eps{n_epochs}_ep{epoch+1}.pth"
                torch.save(checkpoint, ckpt_path)
                print(f" Saved checkpoint: {ckpt_path}")
        
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{args.patience} epochs")

        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered Validation loss did not improve for {args.patience} epochs.")
            break
    
    else:
        # --- save checkpoint ---
        if (epoch + 1) % args.save_every == 0:
            checkpoint = {"epoch": epoch + 1,
                          "model_state_dict": ft_model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict()}
            if args.overwrite_weights:
                ckpt_path = f"{model_path}_tot-trajs{n_traj_train}_tot-eps{n_epochs}.pth"
            else:
                ckpt_path = f"{model_path}_tot-trajs{n_traj_train}_tot-eps{n_epochs}_ep{epoch+1}.pth"
            torch.save(checkpoint, ckpt_path)
            print(f" Saved checkpoint: {ckpt_path}")

                    
    fig, ax = plt.subplots()
    ax.plot(train_losses, label='Train')
    ax.plot(val_losses, label='Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend()
    fig.savefig(os.path.join(savepath_results, (f'loss_{model_choice}_{ft_dataset}_'
               f'max_ar_{args.max_ar_order}_rank-lora{args.rank_lora_attn}_ftlevel{lev}'
               f'_lr{args.lr_level4}_wd{args.wd_level4}.png')))

# free some memory
del X_tr, y_tr, X_va, y_va

#%% Calculate metrics
print(f"→[{ft_dataset}] Evalutions Metrics...")
mse_tot = mae_tot = 0.0
n_samp = 0
out_all, tar_all = [],[]
# predictions from the trained model
with torch.no_grad():
    for inp, tar in tqdm(ft_te_loader):
        inp = inp.to(device)
        out = ft_model(inp)
        out_all.append(out.detach().cpu())
        tar_all.append(tar)
out_all = torch.concat(out_all, dim = 0)
tar_all = torch.concat(tar_all, dim = 0)
print(f'→ [N*(T-1),F,C,D,H,W] Predictions: Outputs: {out_all.shape} and Targets: {tar_all.shape}')

if args.save_truepred_tensors:
    torch.save(out_all, os.path.join(savepath_results, f"pred_{ft_dataset}_morph-{model_choice}.pt"))
    torch.save(tar_all, os.path.join(savepath_results, f"true_{ft_dataset}_morph-{model_choice}.pt"))

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
                                            dataset = ft_dataset)
targets_denorm = RevIN.denormalize_testeval(loadpath_muvar, norm_prefix, tar_all_rs, 
                                            dataset = ft_dataset)
print(f'→ [N,T-1,F,C,D,H,W] Denormalized outputs {outputs_denorm.shape} and targets shape {targets_denorm.shape}')

assert out_all_rs.shape == outputs_denorm.shape, 'Norm and Denorm shapes (Outputs) dont match'
assert tar_all_rs.shape == targets_denorm.shape, 'Norm and Denorm shapes (Targets) dont match'

# calculate MSE and MAE (denormalized scale with samples)
outputs_denorm_240 = outputs_denorm[-args.n_traj_test:]
targets_denorm_240 = targets_denorm[-args.n_traj_test:]
print(f'→ Outputs shape: {outputs_denorm_240.shape}, Targets shape: {targets_denorm_240.shape}')
mse_240_traj = F.mse_loss(outputs_denorm_240, targets_denorm_240, reduction='mean')
mae_240_traj = F.l1_loss(outputs_denorm_240, targets_denorm_240, reduction='mean')
print(f'→ Using last {args.n_traj_test} trajectories (denormalized), '
      f' Avg MSE over trajectories = {mse_240_traj:.5f}')

# calculate MSE and MAE for last 240 trajectories' snapshots
out_all_240 = outputs_denorm_240.reshape(outputs_denorm_240.shape[0]*outputs_denorm_240.shape[1],
                                         outputs_denorm_240.shape[2], outputs_denorm_240.shape[3],
                                         outputs_denorm_240.shape[4], outputs_denorm_240.shape[5], 
                                         outputs_denorm_240.shape[6])
tar_all_240 = targets_denorm_240.reshape(targets_denorm_240.shape[0]*targets_denorm_240.shape[1],
                                         targets_denorm_240.shape[2], targets_denorm_240.shape[3],
                                         targets_denorm_240.shape[4], targets_denorm_240.shape[5], 
                                         targets_denorm_240.shape[6])
print(f'→ Outputs shape: {out_all_240.shape}, Targets shape: {tar_all_240.shape}')
mse_240_steps = F.mse_loss(out_all_240, tar_all_240, reduction='mean')
mae_240_steps = F.l1_loss(out_all_240, tar_all_240, reduction='mean')
print(f'→ Using last {args.n_traj_test} trajectories (denormalized) snapshots ,'
      f' Avg MSE over snapshots = {mse_240_steps:.5f}')

# calculate VRMSE and NRMSE (denormalized scale with samples = N)
vrmse = Metrics3DCalculator.calculate_VRMSE(outputs_denorm, targets_denorm).mean()
nrmse = Metrics3DCalculator.calculate_NRMSE(outputs_denorm, targets_denorm).mean()

# Store the results
savepath_results_ = os.path.join(savepath_results, f'{ft_dataset}')
os.makedirs(savepath_results_, exist_ok=True)
metrics_str = (f" MAE_240: {mae_240_steps:.5f}, MSE_240: {mse_240_steps:.5f}"
               f" NRMSE: {nrmse:.5f}, VRMSE: {vrmse:.5f}")
metrics_name = os.path.join(savepath_results_, (f'metrics_MORPH-{model_size}_'
              f'{model_choice}_ar{args.max_ar_order}_tot-trajs{n_traj_train}'
              f'_tot-eps{n_epochs}_rank-lora{args.rank_lora_attn}_ftlevel{lev}'
              f'_lr{args.lr_level4}_wd{args.wd_level4}.txt'))
with open(metrics_name, "w") as f:
    f.write(metrics_str)
print(f"→ Metrics written to {metrics_name}")

#%% Single-step predictions
#test_sample = np.random.randint(0, test_data.shape[0], 1).item() 
sim = test_data[args.test_sample] #(T,D,H,W,C,F)
sim_rs = np.transpose(sim, (0,5,4,1,2,3)).astype(np.float32) #(T,F,C,D,H,W)
sim_tensor = torch.from_numpy(sim_rs).unsqueeze(0)  #(N=1,T,F,C,D,H,W)
print(f'→ [{ft_dataset}] Single test sample shape: {sim_tensor.shape}')
field_names = [f'field-{fi}' for fi in range(1, sim_tensor.shape[2] + 1)]
slice_dim = 'd' if ft_dataset not in ['CFD1D', 'DR1D'] else '1d'

# print(f'→[{ft_dataset}] Next step predictions...')
# viz = Visualize3DPredictions(ft_model, sim_tensor, device)
# figurename = (f'ft_st_MORPH-{model_size}_{model_choice}_ar{args.max_ar_order}_chAll_'
# f'samp{args.test_sample}_tot-trajs{n_traj_train}_tot-eps{n_epochs}_'
# f'rank-lora{args.rank_lora_attn}_ftlevel{lev}_lr{args.lr_level4}_wd{args.wd_level4}_t')
# for t in range(args.rollout_horizon):
#     viz.visualize_predictions(time_step=t, component=0, slice_dim=slice_dim,
#                               save_path=savepath_results_,
#                               figname=f'{figurename}{t}.png')

#%% Rollout predictions
print(f'→[{ft_dataset}] Rollout predictions...')
test_data_tensor = torch.from_numpy(np.transpose(test_data[-args.n_traj_test:], (0,1,6,5,2,3,4))).float() #(N,T,F,C,D,H,W)
print(f'→ Test data tensor shape for rollouts MSE: {test_data_tensor.shape}')
viz_roll = Visualize3DRolloutPredictions(model=ft_model, 
                                        test_dataset = sim_tensor,  # single trajectory
                                        device=device,
                                        field_names=field_names,
                                        component_names=["d","h","w"])

# calculate rollout MSE over n_traj_test trajectories
mse_roll_240 = viz_roll.rollout_mse(
    test_dataset_full=test_data_tensor,
    start_step=0, 
    num_steps=args.rollout_horizon)
print(f'→ Rollout MSE over {args.n_traj_test} trajectories: {mse_roll_240:.5f}')

# visualize rollouts for single test sample
figurename = (f'ft_ro_MORPH-{model_size}_{model_choice}_ar{args.max_ar_order}_tAll_'
f'samp{args.test_sample}_tot-trajs{n_traj_train}_tot-eps{n_epochs}_'
f'rank-lora{args.rank_lora_attn}_ftlevel{lev}_lr{args.lr_level4}_wd{args.wd_level4}_field')
for f in range(sim_tensor.shape[2]):
    viz_roll.visualize_rollout(start_step=0, num_steps=args.rollout_horizon, field=f,
                            component=1, slice_dim=slice_dim,
                            save_path=savepath_results_,
                            figname=f'{figurename}{f}.png')
