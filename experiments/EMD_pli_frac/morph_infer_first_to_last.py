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
from src.utils.data_preparation_fast import FastARDataPreparer
from src.utils.normalization import RevIN
from src.utils.optimizer_finetuning import SelectFineTuningParameters
from src.utils.lr_schedulars import LRSched
from src.utils.dataloaders.dataloader_heat2d import HEAT2dDataLoader

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
parser.add_argument('--model_size', type=str, choices = list(MORPH_MODELS.keys()),
                    default='S', help='choose from Ti, S, M, L')
parser.add_argument('--checkpoint', type=str, help="Path to saved .pth state dict", required=True)
parser.add_argument('--patch_size', type=int, default=8, help="Patch size for the model")
parser.add_argument('--fields_to_use_heat2d', nargs='+', type=int, default=None,
                    help='list of field indices to use for HEAT2D dataset')

# --- visualization on ----
parser.add_argument('--viz_on', action = 'store_true', 
                    help = "Visualize predictions and rollouts on test set")

# --- parallelization ---
parser.add_argument('--parallel', type=str, choices=['dp','no'], default='dp', 
                    help="DataParallel vs No parallelization")

# --- Finetune levels ---                  
parser.add_argument('--ft_level1', action='store_true', help = "Level-1 finetuning (LoRA, PE, LN)")
parser.add_argument('--ft_level2', action='store_true', help = "Level-2 finetuning (Encoder)")
parser.add_argument('--ft_level3', action='store_true', help = "Level-3 finetuning (Decoder)")
parser.add_argument('--ft_level4', action='store_true', help = "All model parameters")

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
parser.add_argument('--ar_order', type=int, default=1, help = "Autoregressive order of the data")
parser.add_argument('--max_ar_order', type=int, default=1, help="Max autoregressive order for the model")
parser.add_argument('--test_sample', type=int, default=0, help="Sample to plot from the test set")
parser.add_argument('--device_idx', type=int, default=0, help="CUDA device index to run on")

# --- set the parser for defining parameters ---
args = parser.parse_args()
# print args
print("===== Arguments Used =====")
for k, v in vars(args).items():
    print(f"{k}: {v}")
print("==========================")

# device selection
devices = DeviceManager.list_devices()
device = devices[args.device_idx] if devices else 'cpu'

#####################################
### --- set the batch sizes --- ####
#####################################

# setting it to half of the standalone model (trained on 2 GPUs)
batch_size = 1 if args.batch_size is None else args.batch_size
print(f'→ Selected Batch size is {batch_size}')

# data folder locations
if args.dataset_root is None:
    dataset_root = project_root
else:
    dataset_root = args.dataset_root
print(f"→ Current dataset root: {dataset_root}")

# locations
datapath_heat2d = os.path.join(dataset_root,'datasets', 'normalized_revin', "2dHEAT")
savepath_results = os.path.join(project_root, "experiments", "results", "test")
os.makedirs(savepath_results, exist_ok=True)
savepath_model = os.path.join(project_root, "models", "HEAT2D")
loadpath_muvar = os.path.join(project_root, 'data', 'stats_heat2d')

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
fields_name = np.loadtxt(os.path.join(datapath_heat2d, 'channel_names.txt'), dtype=str)
print(f'→ Loaded field names: {fields_name}')
print(f'→ Number of files to load for HEAT2D: {args.num_files_to_load}')

# load the test data
dataset = HEAT2dDataLoader(datapath_heat2d)
test_data = dataset.split_test(num_files = args.num_files_to_load, 
                               fields_to_use = args.fields_to_use_heat2d)
print(f'→ Test data shape: {test_data.shape}')

# rearrange to (N,T,F,C,D,H,W) for the model (F=fields, C=components)
test_data = test_data.transpose(0,1,4,5,6,3,2)
print(f"→ Reshaped test data shape: {test_data.shape}") 

#############################################
### --- load normalization statistics --- ###
#############################################
mu_test = np.load(os.path.join(loadpath_muvar, "mu_combined.npy"))
var_test = np.load(os.path.join(loadpath_muvar, "var_combined.npy"))

# work with the selected fields only (if specified)
mu_use = mu_test[args.fields_to_use_heat2d] if args.fields_to_use_heat2d is not None else mu_test
var_use = var_test[args.fields_to_use_heat2d] if args.fields_to_use_heat2d is not None else var_test
print(f'mu: {mu_use}, var: {var_use}')
breakpoint()

##################################################
### ---- Define model and parallelization ---- ###
##################################################

# model configuration 
patch_size_pt = 8
max_patches, max_fields, max_components = 4096, 3, 3
filters, dim, heads, depth, mlp_dim = MORPH_MODELS[args.model_size]
dropout, emb_dropout = 0.1, 0.1

# set the finetuning level (lev) parameter
args.ft_level4 = True
args.ft_level1 = args.ft_level2 = args.ft_level3 = False
args.rank_lora_attn = 0
args.rank_lora_mlp = 0
args.lora_p = 0

# model
model_name = "MORPH_infer_PLI"
ft_model = ViT3DRegression(patch_size = patch_size_pt, dim = dim, depth = depth,
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
    
###############################################
########### --- Load weights --- ##############
###############################################

checkpoint_path = os.path.join(savepath_model, args.checkpoint)
ckpt = torch.load(checkpoint_path, map_location=device)  
state_dict = ckpt["model_state_dict"]

if next(iter(state_dict)).startswith("module."):
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

ft_model.load_state_dict(state_dict, strict=True)
ft_model.eval()

##########################################
#### --- Parallelization ---#############
#########################################

n_gpus = torch.cuda.device_count()
print(f'→ Finetuning on {n_gpus} GPUs')
if args.parallel == 'dp' and n_gpus > 1:
    ft_model =  nn.DataParallel(ft_model)
    batch_size = n_gpus * batch_size
print(f'→ Selected (Overall) Batch size for is {batch_size}')

#######################################
# --- data preparation for inference ---
########################################

# prepare data into (inputs, targets)) format
print(f'→ Dataset preparation...')
preparer = FastARDataPreparer(ar_order = args.ar_order)
X_te, y_te = preparer.prepare(test_data) # also converts (N,T,D,H,W,C,F) -> (N,T,F,C,D,H,W)

# dataset and dataloader
ft_te = DatasetforDataloader(X_te, y_te)
ft_te_loader = DataLoader(ft_te, batch_size=batch_size, shuffle=False)
print(f'→ Length dataloader: Te {len(ft_te_loader)}')

breakpoint()

###############################################
############ --- Evaluations --- ##############
###############################################

print(f"→ Evalutions Metrics...")
mse_tot = mae_tot = 0.0
n_samp = 0
out_all, tar_all = [],[]
# predictions from the trained model
with torch.no_grad():
    for inp, tar in tqdm(ft_te_loader):
        inp = inp.to(device)
        _,_,out = ft_model(inp)
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
outputs_denorm = out_all_rs * var_use + mu_use
targets_denorm = tar_all_rs * var_use + mu_use
print(f'→ Denormalized outputs {outputs_denorm.shape} and targets shape {targets_denorm.shape}')

# calculate VRMSE and NRMSE (denormalized scale with samples = N)
vrmse = Metrics3DCalculator.calculate_VRMSE(outputs_denorm, targets_denorm).mean()
nrmse = Metrics3DCalculator.calculate_NRMSE(outputs_denorm, targets_denorm).mean()
      
# average value across the test set
print(f'→ RMSE: {rmse:.5f}, MAE: {mae:.5f}, MSE: {mse:.5f}'
      f' VRMSE: {vrmse:.5f}, NRMSE: {nrmse:.5f}')

# Store the results
savepath_results_ = os.path.join(savepath_results, "HEAT2D")
os.makedirs(savepath_results_, exist_ok=True)
metrics_str = (f" MAE: {mae:.5f}, MSE: {mse:.5f}, RMSE: {rmse:.5f},"
               f" NRMSE: {nrmse:.5f}, VRMSE: {vrmse:.5f}")
metrics_name = os.path.join(savepath_results_, f'metrics__{model_name}.txt')
with open(metrics_name, "w") as f:
    f.write(metrics_str)
print(f"→ Metrics written to {metrics_name}")

###############################################
############ --- Visualization --- ############
###############################################

if args.viz_on:
    for test_sample in range(0, 10):
        # input frame t=0 -> (H,W)
        inp_sample = torch.from_numpy(test_data[test_sample, 0, 0, :, :, 0, 0])

        # target frame t=1 -> (H,W)
        tar_sample = torch.from_numpy(test_data[test_sample, 1, 0, :, :, 0, 0])

        # prediction for step 0 (t=0 -> t=1) -> (H,W)
        out_sample = out_all_rs[test_sample, 0, 0, 0, 0, :, :]

        print(f"Input sample shape: {inp_sample.shape}, "
                f"Target sample shape: {tar_sample.shape}, "
                f"Output sample shape: {out_sample.shape}")

        mse_tp = (tar_sample - out_sample) ** 2  # compare pred vs target (more meaningful)

        vmin = min(inp_sample.min(), tar_sample.min(), out_sample.min()).item()
        vmax = max(inp_sample.max(), tar_sample.max(), out_sample.max()).item()

        fig, axs = plt.subplots(1, 4, figsize=(15, 5))
        axs[0].imshow(inp_sample.cpu(), cmap='viridis', vmin=vmin, vmax=vmax); axs[0].set_title('Input t0')
        axs[1].imshow(tar_sample.cpu(), cmap='viridis', vmin=vmin, vmax=vmax); axs[1].set_title('Target t1')
        axs[2].imshow(out_sample.cpu(), cmap='viridis', vmin=vmin, vmax=vmax); axs[2].set_title('Pred t1')
        axs[3].imshow(mse_tp.cpu(), cmap='viridis'); axs[3].set_title('Sq. Error (t1)')

        for a in axs: a.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(savepath_results_, f'tsp_{test_sample}_{model_name}.png'))
        plt.close(fig)

    