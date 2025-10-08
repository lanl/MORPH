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
from config.data_config import DataConfig
from src.utils.dataloaders.dataloaderchaos import DataloaderChaos
from src.utils.normalization import RevIN

#%% Argument parser
MORPH_MODELS = {
    'Ti': [8, 256,  4,  4, 1024],
    'S' : [8, 512,  8,  4, 2048],
    'M' : [8, 768, 12,  8, 3072],
    'L' : [8, 1024,16, 16, 4096]
    }

# -- instantiate argument parsers ---
patch_size = 8
DATA_CONFIG = DataConfig(project_root, patch_size)

# ---- set arguments ----
parser = argparse.ArgumentParser(description="Run inference on trained ViT3D model")
parser.add_argument('--dataset_root', type = str, help = "Location of dataset")
parser.add_argument('--model_choice', type=str, default = 'FM', help = "Model to finetune")
parser.add_argument('--model_size', type=str, choices = list(MORPH_MODELS.keys()),
                    default='Ti', help='choose from Ti, S, M, L')
parser.add_argument('--ckpt_from', type=str, choices = ['FM','FT'], default = 'FM',
                    help="Checkpoint information from FM or previous FT", required = True)
parser.add_argument('--checkpoint', type=str, help="Path to saved .pth state dict", required=True)
parser.add_argument('--ft_dataset', choices=['DR1D','CFD2D','CFD3D-TURB', 'BE1D',
                     'GSDR2D', 'TGC3D','FNS_KF_2D'], type=str, default = 'DR1D', 
                    help = "Choose the finetuning set")

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
parser.add_argument('--batch_size', type=int, help="Batch size for loaders")

# -- model default hyperparameters ---
parser.add_argument('--tf_reg', nargs=2, type=float, metavar=('dropout','emb_dropout'),
                    default=[0.1,0.1], help="Transformer regularization: dropouts")
parser.add_argument('--heads_xa', type=int, default=32, help = "Number of heads of cross attention")
parser.add_argument('--ar_order', type=int, default=1, help = "Autoregressive order of the data")
parser.add_argument('--max_ar_order', type=int, default=1, help="Max autoregressive order for the model")
parser.add_argument('--test_sample', type=int, default=0, help="Sample to plot from the test set")
parser.add_argument('--device_idx', type=int, default=0, help="CUDA device index to run on")
parser.add_argument('--patience', type=int, default=10, help="Early stopping criteria")

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

# --- set the model configuration ---
model_choice = args.model_choice
model_size = args.model_size
checkpoint = args.checkpoint

# --- set the test data congfiguration ---
ft_dataset = args.ft_dataset
n_epochs = args.n_epochs

# --- set the batch sizes ---
# setting it to half of the standalone model (trained on 2 GPUs)
batch_sizes = {'DR1D': 384 // 2, 'CFD2D': 64 // 2, 'CFD3D-TURB': 16 // 2,
               'BE1D': 384 // 2, 'GSDR2D': 64 // 2, 'TGC3D': 16 // 2, 'FNS_KF_2D': 64 //2 }
batch_size = args.batch_size if args.batch_size is not None else batch_sizes[ft_dataset]
print(f'→ Selected Batch size for {ft_dataset} is {batch_size}')

# norm_prefix - prefix of mu, var files
norm_prefix = f"stats_{ft_dataset.lower()}"
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
datasets = ["DR2d_data_pdebench","MHD3d_data_thewell","1dcfd_pdebench","2dSW_pdebench",
            "2dcfd_ic_pdebench","3dcfd_pdebench","1ddr_pdebench","2dcfd_pdebench",
            "3dcfd_turb_pdebench","1dbe_pdebench","2dgrayscottdr_thewell","3dturbgravitycool_thewell",
            "2dFNS_KF_pdegym"]

# --- Pretuning sets ---
datapath_dr = os.path.join(dataset_root,'datasets', 'normalized_revin', datasets[0])
datapath_mhd = os.path.join(dataset_root,'datasets', 'normalized_revin', datasets[1])
datapath_cfd1d = os.path.join(dataset_root,'datasets', 'normalized_revin', datasets[2])
datapath_sw2d = os.path.join(dataset_root,'datasets', 'normalized_revin', datasets[3])
datapath_cfd2dic = os.path.join(dataset_root,'datasets', 'normalized_revin', datasets[4])
datapath_cfd3d = os.path.join(dataset_root,'datasets', 'normalized_revin', datasets[5])

#--- finetune sets ---
datapath_dr1d = os.path.join(dataset_root,'datasets', 'normalized_revin', datasets[6])
datapath_cfd2d = os.path.join(dataset_root,'datasets', 'normalized_revin', datasets[7])
datapath_cfd3d_turb = os.path.join(dataset_root,'datasets', 'normalized_revin', datasets[8])
datapath_be1d = os.path.join(dataset_root,'datasets', 'normalized_revin', datasets[9])
datapath_gsdr2d = os.path.join(dataset_root,'datasets', 'normalized_revin', datasets[10])
datapath_tgc3d = os.path.join(dataset_root,'datasets', 'normalized_revin', datasets[11])
datapath_fns_kf_2d = os.path.join(dataset_root,'datasets', 'normalized_revin', datasets[12])

datapaths = {'MHD': datapath_mhd, 'DR' : datapath_dr,'CFD1D' : datapath_cfd1d,
'CFD2D-IC': datapath_cfd2dic, 'CFD3D': datapath_cfd3d, 'SW': datapath_sw2d,
             'DR1D': datapath_dr1d ,'CFD2D':datapath_cfd2d, 
             'CFD3D-TURB': datapath_cfd3d_turb, 'BE1D': datapath_be1d,
             'GSDR2D': datapath_gsdr2d, 'TGC3D': datapath_tgc3d, 
             'FNS_KF_2D': datapath_fns_kf_2d}
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
data_module = DataloaderChaos()
# import the inflated test data (N,T,D,H,W,C,F)
print(f'Loading data from {datapaths[ft_dataset]}')
train_data, val_data = data_module.load_data(ft_dataset, datapaths[ft_dataset], split = 'train')
test_data = data_module.load_data(ft_dataset, datapaths[ft_dataset], split = 'test') 
print(f"[{ft_dataset}] Shape of train: {train_data.shape}, Val: {val_data.shape}," 
      f"Test data: {test_data.shape}")

# ---- determine data configuration based on ----
patch_size  = 8
max_patches = 4096
max_fields = 3
max_components = 3

#%% Define model and parallelization
# ---- Model init and load weights ----
# --- Rebuild model with LoRA ranks > 0 ---
model_name = (f'ft_morph-{args.model_size}-{ft_dataset}-max_ar{args.max_ar_order}_'
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
n_traj = args.n_traj if args.n_traj is not None else train_data.shape[0]
print(f'→ [{ft_dataset}] Number of finetuning trajectories: {n_traj}')

X_tr, y_tr = preparer.prepare(train_data[0:n_traj]) # also converts (N,T,D,H,W,C,F) -> (N,T,F,C,D,H,W)
X_va, y_va = preparer.prepare(val_data[0: int(n_traj * 0.125)]) # val data is 12.5% of train data
X_te, y_te = preparer.prepare(test_data) # also converts (N,T,D,H,W,C,F) -> (N,T,F,C,D,H,W)
print(f'→ Training Inputs: {X_tr.shape} and Targets: {y_tr.shape}')
assert X_tr.shape[0] == n_traj * (train_data.shape[1] - 1), "Shape mismatch !!"

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
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"]
    
    # pick the real model if wrapped
    target = ft_model.module if isinstance(ft_model, nn.DataParallel) else ft_model 

    if state_dict and next(iter(state_dict)).startswith("module.") and args.parallel == 'no':
        print("→ Stripping 'module.' from checkpoints")
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        
    # strict=False because ft_model has extra LoRA params (A/B) not in ckpt
    missing, unexpected = target.load_state_dict(state_dict, strict=True)
    
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
    if vl_loss < best_val_loss:
        best_val_loss = vl_loss
        epochs_no_improve = 0
        
        # --- save checkpoint ---
        if (epoch + 1) % args.save_every == 0:
            checkpoint = {"epoch": epoch + 1,
                          "model_state_dict": ft_model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict()}
            if args.overwrite_weights:
                ckpt_path = f"{model_path}_tot-trajs{n_traj}_tot-eps{n_epochs}.pth"
            else:
                ckpt_path = f"{model_path}_tot-trajs{n_traj}_tot-eps{n_epochs}_ep{epoch+1}.pth"
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

# calculate VRMSE and NRMSE (denormalized scale with samples = N)
vrmse = Metrics3DCalculator.calculate_VRMSE(outputs_denorm, targets_denorm).mean()
nrmse = Metrics3DCalculator.calculate_NRMSE(outputs_denorm, targets_denorm).mean()
      
# average value across the test set
print(f'→ RMSE: {rmse:.5f}, MAE: {mae:.5f}, MSE: {mse:.5f}'
      f' VRMSE: {vrmse:.5f}, NRMSE: {nrmse:.5f}')

# Store the results
savepath_results_ = os.path.join(savepath_results, f'{ft_dataset}')
os.makedirs(savepath_results_, exist_ok=True)
metrics_str = (f" MAE: {mae:.5f}, MSE: {mse:.5f}, RMSE: {rmse:.5f},"
               f" NRMSE: {nrmse:.5f}, VRMSE: {vrmse:.5f}")
metrics_name = os.path.join(savepath_results_, (f'metrics_MORPH-{model_size}_'
              f'{model_choice}_ar{args.max_ar_order}_tot-trajs{n_traj}'
              f'_tot-eps{n_epochs}_rank-lora{args.rank_lora_attn}_ftlevel{lev}'
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
slice_dim = 'd' if ft_dataset not in ['CFD1D', 'DR1D'] else '1d'

print(f'→[{ft_dataset}] Next step predictions...')
viz = Visualize3DPredictions(ft_model, sim_tensor, device)
figurename = (f'ft_st_MORPH-{model_size}_{model_choice}_ar{args.max_ar_order}_chAll_'
f'samp{args.test_sample}_tot-trajs{n_traj}_tot-eps{n_epochs}_'
f'rank-lora{args.rank_lora_attn}_ftlevel{lev}_lr{args.lr_level4}_wd{args.wd_level4}_t')
for t in range(args.rollout_horizon):
    viz.visualize_predictions(time_step=t, component=0, slice_dim=slice_dim,
                              save_path=savepath_results_,
                              figname=f'{figurename}{t}.png')

#%% Rollout predictions
print(f'→[{ft_dataset}] Rollout predictions...')
viz_roll = Visualize3DRolloutPredictions(model=ft_model, test_dataset=sim_tensor,
                                         device=device,
                                         field_names=field_names,
                                         component_names=["d","h","w"])

figurename = (f'ft_ro_MORPH-{model_size}_{model_choice}_ar{args.max_ar_order}_tAll_'
f'samp{args.test_sample}_tot-trajs{n_traj}_tot-eps{n_epochs}_'
f'rank-lora{args.rank_lora_attn}_ftlevel{lev}_lr{args.lr_level4}_wd{args.wd_level4}_field')
for f in range(sim_tensor.shape[2]):
    viz_roll.visualize_rollout(start_step=0, num_steps=args.rollout_horizon, field=f,
                               component=0, slice_dim=slice_dim,
                               save_path=savepath_results_,
                               figname=f'{figurename}{f}.png')
