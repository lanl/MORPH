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
import h5py
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
from src.utils.explore_hdf5 import ExploreHDF5Structure

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
parser.add_argument('--input_frame', type=str, help="Path to the input tensor", required=True)
parser.add_argument('--target_npz', type=str, help="Path to the target tensor")
parser.add_argument('--checkpoint', type=str, help="Path to saved .pth state dict", required=True)
parser.add_argument('--batch_size', type=int, default=1, help="Batch size for inference")
parser.add_argument('--model_size', type=str, choices = list(MORPH_MODELS.keys()),
                    default='S', help='choose from Ti, S, M, L')
args = parser.parse_args()

# device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# folder to load dataset and norm constants
loadpath_data = os.path.join(project_root, "data")
os.makedirs(loadpath_data, exist_ok=True)
loadpath_model = os.path.join(project_root, "models", "pli_new")
os.makedirs(loadpath_model, exist_ok=True)

# save results
savepath_results = os.path.join(project_root, "experiments")
os.makedirs(savepath_results, exist_ok=True)

# Dataset for Dataloader
class DatasetforDataloader(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# load the test data
if args.input_frame.endswith('.npz'):
    test_data_path = os.path.join(loadpath_data, args.input_frame)  # shape (N,T,F,H,W)
    with np.load(test_data_path) as z:
        test_data = z["av_density"].astype(np.float32)   # ndarray, shape (1120, 400)
    print(f"→ Test data loaded with shape: {test_data.shape}")

elif args.input_frame.endswith('.h5') or args.input_frame.endswith('.hdf5'):
    test_data_path = os.path.join(loadpath_data, args.input_frame)
    #explorer = ExploreHDF5Structure()
    #explorer.explore_hdf5(test_data_path)  
    with h5py.File(test_data_path, "r") as f:
        test_data = f["t0_fields/av_density"][0, 0, :, :].astype(np.float32)  # (1120,400)
    print(f"→ Test data loaded with shape: {test_data.shape}")

# replace nans with zeros
nan_count = np.isnan(test_data).sum()
print(f"Found {nan_count} NaN values in the data. Replacing with zeros.")
test_data = np.nan_to_num(test_data, nan=0.0)

if args.target_npz:
    # also include the target for comparison
    target_data_path = os.path.join(loadpath_data, args.target_npz)  # shape (N,T,F,H,W)
    with np.load(target_data_path) as z:
        target_data = z["av_density"].astype(np.float32)   # ndarray, shape (1120, 400)
    print(f"→ Target data loaded with shape: {target_data.shape}")

    # replace nans with zeros
    nan_count = np.isnan(target_data).sum()
    target_data = np.nan_to_num(target_data, nan=0.0)

breakpoint()
#############################################
############## --- normalization--- #########
#############################################

# load normalizations stats
stats_test = np.load(os.path.join(loadpath_data, "normstats_heat_avd.npy"))
print(f"→ Normalization stats loaded. {stats_test}")
mu_test, var_test = stats_test[0], stats_test[1]   
print(f"→ Normalization stats loaded. Mean: {mu_test:.5f}, Var: {var_test:.5f}")

# normalize the test data
test_data_norm = (test_data - mu_test) / var_test
print(f"→ Test data normalized with shape: {test_data_norm.shape}")

#############################################
############## --- UPTF7 --- ################
#############################################

test_data_expand = np.expand_dims(test_data_norm, axis=[0,1,2,3,4])  # (N,T,F,C,D,H,W)
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
max_ar_order = 1 if args.model_size in ['Ti','S','M'] else 16
args.rank_lora_attn = 0
args.rank_lora_mlp = 0
args.lora_p = 0

# model
model_name = "MORPH_infer_PLI"
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
    
###############################################
########### --- Load weights --- ##############
###############################################

checkpoint_path = os.path.join(loadpath_model, args.checkpoint)
ckpt = torch.load(checkpoint_path, map_location=device)  
state_dict = ckpt["model_state_dict"]

if next(iter(state_dict)).startswith("module."):
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

ft_model.load_state_dict(state_dict, strict=True)
ft_model.eval()

###############################################
############ --- Evaluations --- ##############
###############################################

print(f"→ Evalutions...")
with torch.no_grad():
    input_tensor = torch.from_numpy(test_data_expand).to(device)  # (N,T,F,C,D,H,W)
    _,_,out_tensor = ft_model(input_tensor)
print(f"→ Inputs shape: {input_tensor.shape}")
print(f"→ Predictions obtained with shape: {out_tensor.shape}")

###############################################
############ --- Visualization --- ############
###############################################

# squeeze the dims
input_tensor = input_tensor[0,0,0,0,0, :, :]  # (H,W)
out_tensor = out_tensor[0,0,0,0, :, :]  # (H,W)
print(f"→ Input tensor shape after squeezing: {input_tensor.shape}")
print(f"→ Output tensor shape after squeezing: {out_tensor.shape}")

# unnormalize the data for visualization
input_tensor_uunnorm = input_tensor.cpu() * var_test + mu_test
out_tensor_unnorm = out_tensor.cpu() * var_test + mu_test
im_min = input_tensor_uunnorm.min().item()
im_max = input_tensor_uunnorm.max().item()

if args.target_npz:
    # plot input and output side by side
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    im0 = axs[0].imshow(input_tensor_uunnorm, cmap='viridis', vmin=im_min, vmax=im_max)
    axs[0].set_title("Input (Unnormalized)")
    fig.colorbar(im0, ax=axs[0])
    im2 = axs[1].imshow(target_data, cmap='viridis', vmin=im_min, vmax=im_max)
    axs[1].set_title("Target (Unnormalized)")
    fig.colorbar(im2, ax=axs[1])
    im1 = axs[2].imshow(out_tensor_unnorm, cmap='viridis', vmin=im_min, vmax=im_max)
    axs[2].set_title("Output (Unnormalized)")
    fig.colorbar(im1, ax=axs[2])
    plt.tight_layout()
    plt.savefig(os.path.join(savepath_results, f"{model_name}_pred_vs_input.png"))

else:
    # plot input and output side by side
    fig, axs = plt.subplots(1, 2, figsize=(8, 5))
    im0 = axs[0].imshow(input_tensor_uunnorm, cmap='viridis', vmin=im_min, vmax=im_max)
    axs[0].set_title("Input (Unnormalized)")
    fig.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(out_tensor_unnorm, cmap='viridis', vmin=im_min, vmax=im_max)
    axs[1].set_title("Output (Unnormalized)")
    fig.colorbar(im1, ax=axs[1])
    plt.tight_layout()
    plt.savefig(os.path.join(savepath_results, f"{model_name}_pred_vs_input.png"))
    print(f'Fig. saved to: {savepath_results}')