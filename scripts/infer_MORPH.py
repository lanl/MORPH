import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib
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

# location of REVIN data
datasets = ["DR2d_data_pdebench","MHD3d_data_thewell","1dcfd_pdebench","2dSW_pdebench",
            "2dcfd_ic_pdebench","3dcfd_pdebench","1ddr_pdebench","2dcfd_pdebench",
            "3dcfd_turb_pdebench","1dbe_pdebench","2dgrayscottdr_thewell",
            "3dturbgravitycool_thewell","2dFNS_KF_pdegym"]

# --- Pretuning sets ---
datapath_dr = os.path.join(project_root,'datasets', 'normalized_revin', datasets[0])
datapath_mhd = os.path.join(project_root,'datasets', 'normalized_revin', datasets[1])
datapath_cfd1d = os.path.join(project_root,'datasets', 'normalized_revin', datasets[2])
datapath_sw2d = os.path.join(project_root,'datasets', 'normalized_revin', datasets[3])
datapath_cfd2dic = os.path.join(project_root,'datasets', 'normalized_revin', datasets[4])
datapath_cfd3d = os.path.join(project_root,'datasets', 'normalized_revin', datasets[5])

#--- finetune sets ---
datapath_dr1d = os.path.join(project_root,'datasets', 'normalized_revin', datasets[6])
datapath_cfd2d = os.path.join(project_root,'datasets', 'normalized_revin', datasets[7])
datapath_cfd3d_turb = os.path.join(project_root,'datasets', 'normalized_revin', datasets[8])
datapath_be1d = os.path.join(project_root,'datasets', 'normalized_revin', datasets[9])
datapath_gsdr2d = os.path.join(project_root,'datasets', 'normalized_revin', datasets[10])
datapath_tgc3d = os.path.join(project_root,'datasets', 'normalized_revin', datasets[11])
datapath_fns_kf_2d = os.path.join(project_root,'datasets', 'normalized_revin', datasets[12])

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
loadpath_models = os.path.join(project_root, "models")

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
    
#%% ---- Data paths (training on 6 datasets) ----
patch_size = 8
DATA_CONFIG = DataConfig(project_root, patch_size)

MORPH_MODELS = {
    'Ti': [8, 256,  4,  4, 1024],
    'S' : [8, 512,  8,  4, 2048],
    'M' : [8, 768, 12,  8, 3072],
    'L' : [8, 1024,16, 16, 4096]
    }

# ---- arguments ----
parser = argparse.ArgumentParser(description="Run inference on trained ViT3D model")
# main
parser.add_argument('--model_choice', choices=['MHD','DR','CFD1D','CFD2D-IC','CFD3D',
                    'SW','DR1D','CFD2D','CFD3D-TURB', 'BE1D', 'GSDR2D' ,'TGC3D','FNS_KF_2D', 
                    'FM'], type = str, default = 'MHD', help = 'Select the model to test')
parser.add_argument('--model_size', type=str, choices = list(MORPH_MODELS.keys()),
                    default='Ti', help='choose from Ti, S, M, L')
parser.add_argument('--checkpoint', type=str, help="Path to saved .pth state dict")
parser.add_argument('--test_dataset', choices=['MHD','DR','CFD1D','CFD2D-IC','CFD3D',
                    'SW','DR1D','CFD2D','CFD3D-TURB', 'BE1D', 'GSDR2D', 'TGC3D',
                    'FNS_KF_2D'], 
                    type=str, default = 'MHD', help = 'Select the test dataset')
parser.add_argument('--ar_order', type=int, default=1, help = 'Autoregressive order of the data')
parser.add_argument('--rollout_horizon', type = int, default = 10, help = 'Visualization: single step & rollouts')
parser.add_argument('--device_idx', type=int, default=0, help="CUDA device index to run on")
parser.add_argument('--batch_size', type=int, help="Batch size for test loader")
parser.add_argument('--test_sample', type=int, default=0, help='Sample to visualize')

# model related
parser.add_argument('--tf_reg', nargs=2, type=float, metavar=('dropout','emb_dropout'),
                    default=[0.1,0.1], help='Transformer regularization: dropouts')
parser.add_argument('--heads_xa', type=int, default=32, help='Number of heads of cross attention')
parser.add_argument('--max_ar_order', type=int, default=1, help="Max autoregressive order for the model")

args = parser.parse_args()
devices = DeviceManager.list_devices()
filters, dim, heads, depth, mlp_dim = MORPH_MODELS[args.model_size]
dropout, emb_dropout = args.tf_reg
device = devices[args.device_idx] if devices else 'cpu'

# --- set the model configuration ---
model_choice = args.model_choice
model_size = args.model_size
checkpoint = args.checkpoint
checkpoint_path = os.path.join(loadpath_models, f'{model_choice}', checkpoint)

# --- set the test data congfiguration ---
test_dataset = args.test_dataset
batch_size = args.batch_size

# --- set the batch sizes ---
# setting it to half of the standalone model (trained on 2 GPUs)
batch_sizes = {'MHD': 16//2, 'DR': 64//2 , 'CFD1D': 128//2,                # pretraining sets
               'CFD2D-IC': 16 //2, 'CFD3D': 4 //2 , 'SW': 64 //2,          # pretraining sets
               'DR1D': 384 // 2, 'CFD2D': 64 // 2, 'CFD3D-TURB': 16 // 2,  # finetraining sets
               'BE1D': 384 // 2, 'GSDR2D': 64 // 2, 'TGC3D': 16 // 2,      # finetraining sets
               'FNS_KF_2D': 64//2}                                         # finetraining sets
batch_size = args.batch_size if args.batch_size is not None else batch_sizes[test_dataset]
print(f'→ Selected Batch size for {test_dataset} is {batch_size}')

# --- norm_prefix - prefix of mu, var files ---
norm_prefix = f"stats_{test_dataset.lower()}"
print(f'→ Selected Model MORPH-{model_choice}-{model_size} for Dataset {test_dataset}')

# *--- Load data via dataloaders ----
print("→ Loading test dataset...")
data_module = DataloaderChaos()
# import the inflated test data (N,T,D,H,W,C,F)
test_data = data_module.load_data(test_dataset, datapaths[test_dataset], split = 'test') 
print(f"→ [{test_dataset}] Shape of test data: {test_data.shape}")

# ---- determine data configuration based on {model_choice} ----
if model_choice != 'FM': # for standalone models
    patch_size  = DATA_CONFIG[test_dataset]['patch_size']
    max_patches = DATA_CONFIG[test_dataset]['max_patches']
    max_fields = DATA_CONFIG[test_dataset]['fields']
    max_components = DATA_CONFIG[test_dataset]['components']
    
else: # for FM
    patch_size  = 8
    max_patches = 4096
    max_fields = 3
    max_components = 3

#%% ---- Prepare test loader ----
print(f'→[{test_dataset}] Dataset preparation...')
te_preparer = FastARDataPreparer(ar_order = args.ar_order, set_name='Test')
X_te, y_te = te_preparer.prepare(test_data) # also converts (N,T,D,H,W,C,F) -> (N,T,F,C,D,H,W)
print(f'Images(N,T,F,C,D,H,W): {X_te.shape}, Targets(N,F,C,D,H,W): {y_te.shape}')
test_ds = DatasetforDataloader(X_te, y_te)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
print(f'→ Length dataloader: {len(test_loader)}')
                         
#%% ---- Model init and load weights ----
model = ViT3DRegression(patch_size = patch_size, dim = dim, depth = depth,
        heads = heads, heads_xa = args.heads_xa, mlp_dim = mlp_dim,
        max_components = max_components, conv_filter = filters, 
        max_ar = args.max_ar_order, 
        max_patches = max_patches, max_fields = max_fields,
        dropout = dropout, emb_dropout = emb_dropout,
        lora_r_attn = 0, lora_r_mlp = 0,
        lora_alpha = None, lora_p = 0.0).to(device)

num_params_model = sum(p.numel() for p in model.parameters()) / 1e6
print(f"→ NUMBER OF PARAMETERS OF THE ViT (in millions): "
      f"{num_params_model:.3g}")

model_weights = torch.load(checkpoint_path, map_location=device, weights_only=True)
state_dict = model_weights["model_state_dict"]
if ("module." in next(iter(state_dict))):
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.to(device).eval()

#%% Calculate metrics
print(f"→[{test_dataset}] Evalutions Metrics...")
mse_tot = mae_tot = 0.0
n_samp = 0
out_all, tar_all = [],[]
# predictions from the trained model
with torch.no_grad():
    for inp, tar in tqdm(test_loader):
        inp = inp.to(device)
        _, _, out = model(inp)
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
                                            dataset = test_dataset)
targets_denorm = RevIN.denormalize_testeval(loadpath_muvar, norm_prefix, tar_all_rs, 
                                            dataset = test_dataset)
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
savepath_results_ = os.path.join(savepath_results, f'{test_dataset}')
os.makedirs(savepath_results_, exist_ok=True)
metrics_str = (f" MAE: {mae:.5f}, MSE: {mse:.5f}, RMSE: {rmse:.5f},"
               f" NRMSE: {nrmse:.5f}, VRMSE: {vrmse:.5f}")
metrics_name = os.path.join(savepath_results_, 
               f'metrics_MORPH-{model_size}_{model_choice}_ar{args.max_ar_order}.txt')
with open(metrics_name, "w") as f:
    f.write(metrics_str)
print(f"→ Metrics written to {metrics_name}")

#%% Single-step predictions
#test_sample = np.random.randint(0, test_data.shape[0], 1).item() 
sim = test_data[args.test_sample] #(T,F,C,D,H,W)
sim_rs = np.transpose(sim, (0,5,4,1,2,3)).astype(np.float32) #(T,F,C,D,H,W)
sim_tensor = torch.from_numpy(sim_rs).unsqueeze(0)  #(N=1,T,F,C,D,H,W)
field_names = [f'field-{fi}' for fi in range(1, sim_tensor.shape[2] + 1)]
slice_dim = 'd' if test_dataset not in ['CFD1D', 'DR1D', 'BE1D'] else '1d'

print(f'→[{test_dataset}] Next step predictions...')
viz = Visualize3DPredictions(model, sim_tensor, device)
figurename = f'pred_st_MORPH-{model_size}_{model_choice}_ar{args.max_ar_order}_chAll_samp{args.test_sample}'
for t in range(args.rollout_horizon):
    viz.visualize_predictions(time_step=t, component=0, slice_dim=slice_dim,
                              save_path=savepath_results_,
                              figname=f'{figurename}_t{t}.png')

#%% Rollout predictions
print(f'→[{test_dataset}] Rollout predictions...')
viz_roll = Visualize3DRolloutPredictions(model=model, test_dataset=sim_tensor,
                                         device=device,
                                         field_names=field_names,
                                         component_names=["d","h","w"])

figurename = f'pred_ro_MORPH-{model_size}_{model_choice}_ar{args.max_ar_order}_tAll_samp{args.test_sample}'
for f in range(sim_tensor.shape[2]):
    viz_roll.visualize_rollout(start_step=0, num_steps=args.rollout_horizon, field=f,
                               component=0, slice_dim=slice_dim,
                               save_path=savepath_results_,
                               figname=f'{figurename}_field{f}')
