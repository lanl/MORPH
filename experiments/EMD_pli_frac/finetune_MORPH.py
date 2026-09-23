#%%
import glob
import json
import os
import sys
import argparse
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
from src.utils.device_manager import DeviceManager
from src.utils.vit_conv_xatt_axialatt2 import ViT3DRegression
from src.utils.metrics_3d import Metrics3DCalculator
from src.utils.visualize_predictions_3d_full import Visualize3DPredictions
from src.utils.visualize_rollouts_3d_full import Visualize3DRolloutPredictions
from src.utils.data_preparation_fast import FastARDataPreparer
from config.data_config import DataConfig
from src.utils.dataloaders.dataloaderchaos import DataloaderChaos
from src.utils.normalization import RevIN
from src.utils.optimizer_finetuning import SelectFineTuningParameters
from src.utils.lr_schedulars import LRSched
from src.utils.trainers import Trainer
from src.utils.simple_plotting import plot_samples

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
parser.add_argument('--dataset_root', type = str, help = "Location of dataset")
parser.add_argument('--mode', type=str, choices=['ft', 'tfs'], default='ft',
                    help = "Run in standalone mode with random init weights (no pretraining)")
parser.add_argument('--model_choice', type=str, default = 'FM', help = "Model to finetune")
parser.add_argument('--model_size', type=str, choices = list(MORPH_MODELS.keys()),
                    default='S', help='choose from Ti, S, M, L')
parser.add_argument('--ckpt_from', type=str, choices = ['FM','FT'], default = 'FM',
                    help="Checkpoint information from FM or previous FT", required=True)
parser.add_argument('--checkpoint', type=str, help="Path to saved .pth state dict", required=True)
parser.add_argument('--ft_dataset', choices=['DR1D','CFD2D','CFD3D-TURB', 'BE1D',
                     'GSDR2D', 'TGC3D','FNS_KF_2D','HEAT2D','FRAC2D'], type=str, default = 'HEAT2D', 
                    help = "Choose the finetuning set")

parser.add_argument('--patch_size', type=int, default=8, help="Patch size for the model")
parser.add_argument('--fields_to_use_heat2d', nargs='+', type=int, default=None,
                    help='list of field indices to use for HEAT2D dataset')

# --- Finetune levels ---                  
parser.add_argument('--ft_level1', action='store_true', help = "Level-1 finetuning (LoRA, PE, LN)")
parser.add_argument('--ft_level2', action='store_true', help = "Level-2 finetuning (Encoder)")
parser.add_argument('--ft_level3', action='store_true', help = "Level-3 finetuning (Decoder)")
parser.add_argument('--ft_level4', action='store_true', help = "All model parameters")

# --- visualization on ----
parser.add_argument('--viz_on', action = 'store_true', 
                    help = "Visualize predictions and rollouts on test set")

# --- hyperparameter study ---
parser.add_argument('--hyperparameter_study', action='store_true', 
                    help = "Use small settings for quick hyperparameter study runs")

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
parser.add_argument('--num_files_to_load', type=int, default=None, 
                    help="Number of data files to load for finetuning; None = all files")
parser.add_argument('--frac_traj', type=float, default = 1.0, help="Fine-tuning trajectories")
parser.add_argument('--num_epochs', type=int, default = 100, help="Fine-tuning epochs")
parser.add_argument('--rollout_horizon', type = int, default = 10, 
                    help = "Visualization: single step & rollouts")
parser.add_argument('--batch_size', type=int, help="Batch size for loaders")
parser.add_argument('--min_lr', type=float, default=1e-7, help="Minimum LR for schedular")
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--wd', type=float, default=1e-2, help = 'weight decay')
parser.add_argument('--warm_epochs', type = int, default = 5)

# -- model default hyperparameters ---
parser.add_argument('--tf_reg', nargs=2, type=float, metavar=('dropout','emb_dropout'),
                    default=[0.1,0.1], help="Transformer regularization: dropouts")
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
# print args
print("===== Arguments Used =====")
for k, v in vars(args).items():
    print(f"{k}: {v}")
print("==========================")

# tags for figs save
run_tag = (
    f"mode-{args.mode}_"
    f"mc-{args.model_choice}_"
    f"ms-{args.model_size}_"
    f"cf-{args.ckpt_from}_"
    f"ftds-{args.ft_dataset}_"
    f"ps-{args.patch_size}_"
    f"r1_{args.rank_lora_attn}_"
    f"r2_{args.rank_lora_mlp}_"
    f"ftl1-{args.ft_level1}_"
    f"ftl2-{args.ft_level2}_"
    f"ftl3-{args.ft_level3}_"
    f"ftl4-{args.ft_level4}_"
    f"ep-{args.num_epochs}_"
    f"traj-{args.frac_traj}_"
    f"lr-{args.lr}_"
    f"wd-{args.wd}_"
    f"fuh2d-{args.fields_to_use_heat2d}_"
    f"ftl-{args.num_files_to_load}"

)

# --- hyperparameter study ---
if args.hyperparameter_study:
    print("→ Hyperparameter study mode ON: using small settings for quick runs")
    run = wandb.init(project="morph_heat_frac_finetuning", config=args)
    
# model configuration
DATA_CONFIG = DataConfig(project_root, args.patch_size)

# device selection
devices = DeviceManager.list_devices()
device = devices[args.device_idx] if devices else 'cpu'

#####################################
### --- set the batch sizes --- ####
#####################################

# setting it to half of the standalone model (trained on 2 GPUs)
batch_sizes = {'DR1D': 384 // 2, 'CFD2D': 64 // 2, 'CFD3D-TURB': 16 // 2,
               'BE1D': 384 // 2, 'GSDR2D': 64 // 2, 'TGC3D': 16 // 2, 'FNS_KF_2D': 64 //2,
                'HEAT2D': 1, 'FRAC2D': 128}
batch_size = args.batch_size if args.batch_size is not None else batch_sizes[args.ft_dataset]
print(f'→ Selected Batch size for {args.ft_dataset} is {batch_size}')

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
            "2dFNS_KF_pdegym", "2dHEAT","2dFRAC_tung"] # cl sets added

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

# --- CL sets ---
datapath_heat2d = os.path.join(dataset_root,'datasets', 'normalized_revin', datasets[13])
datapath_frac2d = os.path.join(dataset_root,'datasets', 'normalized_revin', datasets[14])

datapaths = {'MHD': datapath_mhd, 'DR' : datapath_dr,'CFD1D' : datapath_cfd1d,
'CFD2D-IC': datapath_cfd2dic, 'CFD3D': datapath_cfd3d, 'SW': datapath_sw2d,
             'DR1D': datapath_dr1d ,'CFD2D':datapath_cfd2d, 
             'CFD3D-TURB': datapath_cfd3d_turb, 'BE1D': datapath_be1d,
             'GSDR2D': datapath_gsdr2d, 'TGC3D': datapath_tgc3d, 
             'FNS_KF_2D': datapath_fns_kf_2d,
             'HEAT2D': datapath_heat2d, 
             'FRAC2D': datapath_frac2d}

# savepaths
savepath_results = os.path.join(project_root, "experiments", "results", "test")
os.makedirs(savepath_results, exist_ok=True)

# savepaths
savepath_model = os.path.join(project_root, "models")

# savepath of mu and var
loadpath_muvar = os.path.join(project_root, 'data', f'stats_{args.ft_dataset.lower()}')

# Dataset for Dataloader
class DatasetforDataloader(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
#################################
### --- Load Test dataset --- ###
#################################

# --- Load data via dataloaders ----
print(f'==== Loading data from {datapaths[args.ft_dataset]} ====')
data_module = DataloaderChaos()
if args.ft_dataset == 'HEAT2D':
    # load .txt file for channel names
    fields_name = np.loadtxt(os.path.join(datapaths[args.ft_dataset], 'channel_names.txt'), dtype=str)
    print(f'→ Loaded field names: {fields_name}')

    # files to use
    print(f'→ Number of files to load for HEAT2D: {args.num_files_to_load}')

    # select fields to use
    fields_to_use = args.fields_to_use_heat2d  # selecting fields for HEAT2D

    print(f'→ Selected fields for HEAT2D: {fields_name[fields_to_use]}')
    train_data, val_data = data_module.load_heat2d(datapaths[args.ft_dataset], 
                            split = 'train', num_files= args.num_files_to_load,
                            fields_to_use = fields_to_use)
    test_data = data_module.load_heat2d(datapaths[args.ft_dataset], 
                            split = 'test', num_files = args.num_files_to_load,
                            fields_to_use = fields_to_use)
    
    # patch-size
    patch_size = (1,args.patch_size,args.patch_size)
    
elif args.ft_dataset == 'FRAC2D':
    # files to use
    print(f'→ Number of files to load for FRAC2D: {args.num_files_to_load}')
    # load data
    train_data, val_data = data_module.load_frac2d(datapaths[args.ft_dataset], 
                            split = 'train', num_files= args.num_files_to_load)
    test_data = data_module.load_frac2d(datapaths[args.ft_dataset], 
                            split = 'test', num_files = args.num_files_to_load)
    # patch-size
    patch_size = (1,args.patch_size,args.patch_size)

else:
    train_data, val_data = data_module.load_data(args.ft_dataset, datapaths[args.ft_dataset], 
                            split = 'train', num_files= args.num_files_to_load)
    test_data = data_module.load_data(args.ft_dataset, datapaths[args.ft_dataset], 
                            split = 'test', num_files = args.num_files_to_load)
    
total_samples = train_data.shape[0] + val_data.shape[0] + test_data.shape[0]
print(f"[{args.ft_dataset}] Shape of train: {train_data.shape}, Val: {val_data.shape}, " 
      f"Test data: {test_data.shape}")
print(f'==== Total number of samples (train+val+test): {total_samples} ====')

#############################################
### --- load normalization statistics --- ###
#############################################

# find files
mu_files = sorted(glob.glob(os.path.join(loadpath_muvar, "*mu.npy")))
var_files = sorted(glob.glob(os.path.join(loadpath_muvar, "*var.npy")))
if not mu_files:
    raise FileNotFoundError(f"No *mu.npy files found in {loadpath_muvar}")
elif not var_files:
    raise FileNotFoundError(f"No *var.npy files found in {loadpath_muvar}")
else:
    print(f'→ Found {len(mu_files)} mu files in {loadpath_muvar}')
    print(f'→ Found {len(var_files)} var files in {loadpath_muvar}')

# load stats files
mu_all  = np.concatenate([np.load(f) for f in mu_files], axis=0)
var_all = np.concatenate([np.load(f) for f in var_files], axis=0)

# reduce stats according to num_files_to_load
mu_test = mu_all[:test_data.shape[0]]
var_test = var_all[:test_data.shape[0]]
print(f'→ Loaded mu and var for the test set: {mu_test.shape}, {var_test.shape}')

# sanity checks
assert mu_test.shape == var_test.shape, "Mismatch in number of mu and var files"
assert mu_test.shape[0] == test_data.shape[0], \
    "Mismatch in loaded mu/var files and the dataset"

# some reshape according to the dataset
fields_to_use = None
if args.ft_dataset == "HEAT2D":
    fields_to_use = args.fields_to_use_heat2d
if args.ft_dataset == "HEAT2D" and fields_to_use is not None:
    mu_test = mu_test[:, fields_to_use]
    var_test = var_test[:, fields_to_use]
print(f'→ After selecting fields, mu and var shapes: {mu_test.shape}, {var_test.shape}')

# --- transform the data if required ---
if args.ft_dataset in ['HEAT2D','FRAC2D']:
    # reshape to (N,T,D,H,W,C,F) for FastARDataPreparer
    print(f"→ Reshaping data for {args.ft_dataset} …")
    train_data = train_data.transpose(0,1,4,5,6,3,2)
    val_data = val_data.transpose(0,1,4,5,6,3,2)
    test_data = test_data.transpose(0,1,4,5,6,3,2)
    print(f"[{args.ft_dataset}] Shape of train: {train_data.shape}, Val: {val_data.shape}, " 
      f"Test data: {test_data.shape}")

##################################################
### ---- Define model and parallelization ---- ###
##################################################

model_name = f'MORPH__{run_tag}'

# model configuration 
max_patches, max_fields, max_components = 4096, 3, 3
filters, dim, heads, depth, mlp_dim = MORPH_MODELS[args.model_size]
dropout, emb_dropout = args.tf_reg

# model
ft_model = ViT3DRegression(patch_size = patch_size, dim = dim, depth = depth,
        heads = heads, heads_xa = 32, mlp_dim = mlp_dim,
        max_components = max_components, conv_filter = filters, 
        max_ar = args.max_ar_order, max_patches = max_patches, max_fields = max_fields,
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

# --- Parallelization ---
n_gpus = torch.cuda.device_count()
print(f'→ Finetuning on {n_gpus} GPUs')
if args.parallel == 'dp' and n_gpus > 1:
    ft_model =  nn.DataParallel(ft_model)
    batch_size = n_gpus * batch_size
print(f'→ Selected (Overall) Batch size for {args.ft_dataset} is {batch_size}')

# --- Prepare dataloaders ---
# select trajectories
frac_traj = args.frac_traj if args.frac_traj is not None else 1.0
n_traj = int(frac_traj * train_data.shape[0])
print(f'→ [{args.ft_dataset}] Number of finetuning trajectories: {n_traj}')

# prepare data into (inputs, targets)) format
print(f'→ [{args.ft_dataset}] Dataset preparation...')
preparer = FastARDataPreparer(ar_order = args.ar_order)
X_tr, y_tr = preparer.prepare(train_data[0 : n_traj]) # also converts (N,T,D,H,W,C,F) -> (N,T,F,C,D,H,W)
X_va, y_va = preparer.prepare(val_data[0: n_traj]) # val data is 12.5% of train data
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

#%%#############################################
########### --- Load weights --- ##############
###############################################

# ---- load the pretrained weights ----
start_epoch = 0
if args.mode == 'ft' and args.ckpt_from == 'FM':
    print(f"→ Loading checkpoints from {args.ckpt_from}")
    # --- Load pretrained checkpoint from foundational model ---
    checkpoint_path = os.path.join(savepath_model, f'{args.model_choice}', args.checkpoint)
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
    
elif args.mode == 'ft' and args.ckpt_from == 'FT':
    print(f"→ Loading checkpoints from {args.ckpt_from}")
    # ---- resume checkpoint from previous finetuned epochs ----
    resume_path = os.path.join(savepath_model, f'{args.ft_dataset}', args.checkpoint)
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
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

savepath_model_folder = os.path.join(savepath_model, f'{args.ft_dataset}')
os.makedirs(savepath_model_folder, exist_ok=True)
model_path = os.path.join(savepath_model_folder, model_name)
train_losses, val_losses = [], []
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

    train_losses.append(tr_loss)
    val_losses.append(vl_loss)
    
    # --- step the schedular ---
    scheduler.step()        # no val loss needed
    
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

    # --- log to wandb ---
    if  args.hyperparameter_study:
        run.log({"train/loss": tr_loss, "val/loss": vl_loss, "lr": current_lr, "epoch": epoch})
    
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
                if args.overwrite_weights:
                    ckpt_path = f"{model_path}.pth"
                else:
                    ckpt_path = f"{model_path}_ep{epoch+1}.pth"
                torch.save(checkpoint, ckpt_path)
                print(f" Saved checkpoint: {ckpt_path}")
    
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{args.patience} epochs")

        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered Validation loss did not improve for {args.patience} epochs.")
            break
    
# ---- SAVE HISTORY ----
hist_path = os.path.join(savepath_model_folder, f"zz_history__{model_name}.json")
with open(hist_path, "w") as f:
    json.dump(history, f, indent=2)
print(f" Saved history: {hist_path}")

# learning curve                
fig, ax = plt.subplots()
ax.plot(train_losses, label='Train')
ax.plot(val_losses, label='Val')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend()
fig.savefig(os.path.join(savepath_results, (f'learningcurve_{model_name}.png')))

#%% Evaluations 
##############################################
############ --- Evaluations --- ##############
###############################################

print(f"→[{args.ft_dataset}] Evalutions Metrics...")
mse_tot = mae_tot = 0.0
n_samp = 0
predictions_all, tar_all = [],[]

# predictions from the trained model
with torch.no_grad():  
    for s in range(0, X_te.shape[0], batch_size):  
        inputs = X_te[s:s+batch_size].to(device)   

# predictions from the trained model
with torch.no_grad():
    for s in range(0, X_te.shape[0], batch_size):  
        inputs = X_te[s:s+batch_size].to(device) 
        _,_,prediction = ft_model(inputs)
        # squeeze dims
        prediction_sq = prediction[:,0,0,0,:,:]
        predictions_all.append(prediction_sq.detach().cpu())
predictions_all = torch.cat(predictions_all, dim = 0)
print(f'→ Predictions: {predictions_all.shape}')
print(f'→ Inputs: {X_te.shape}, Targets: {y_te.shape}')

# get the original inputs and targets
inputs_all = X_te[:,0,0,0,0,:,:] # (N,H,W)
targets_all = y_te[:,0,0,0,:,:] # (N,H,W)
print(f'→ Original inputs shape: {inputs_all.shape}, targets shape: {targets_all.shape}')

# calculate MSE and MAE (normalized scale with samples = N*T)
mse = F.mse_loss(predictions_all, targets_all, reduction='mean')
mae = F.l1_loss(predictions_all, targets_all, reduction='mean')
rmse = mse**0.5
print(f"→ Test MSE: {mse:.5f}, MAE: {mae:.5f}, RMSE: {rmse:.5f}")

# --- log to wandb ---
if args.hyperparameter_study:
    run.log({"Test-MSE": mse, "epoch": epoch})

# Store the results
savepath_results_ = os.path.join(savepath_results, f'{args.ft_dataset}')
os.makedirs(savepath_results_, exist_ok=True)

##############################################
############ --- Visualization --- ############
###############################################

if args.viz_on:
    plot_samples(inputs_all, targets_all, predictions_all, savepath_results_, args, n=10,
                 filename = f'morph-{args.model_size}-{args.ft_dataset}')

# hyperparameter study finish
if args.hyperparameter_study:
    run.finish()
