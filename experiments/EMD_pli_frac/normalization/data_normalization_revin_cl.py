#%%
import os
import numpy as np
import h5py
import sys

# Add project root to path
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

# load the classes
from src.utils.normalization import RevIN
from config.data_config_vis import DataConfig

# raw data directory
dataset_dir = "D:/data"

# instantiate the class
cfg = DataConfig(dataset_dir=dataset_dir, project_root=project_root)

# load paths
load_root = "D:/data"

# load all the filepaths
# --- New CL sets ---
loadpath_heat_2d = cfg['2dHEAT']['file_path_2dheat']
loadpath_frac_2d = cfg['2dFRAC_tung']['file_path_2dfrac']

# create folders if they don't exist
for base in (loadpath_heat_2d, loadpath_frac_2d):
    for split in ('train','val','test'):
        os.makedirs(os.path.join(base, split), exist_ok=True)
        
# savepath of mu and var
savepath_muvar_heat = os.path.join(project_root, 'data', 'stats_heat_2d')
savepath_muvar_frac = os.path.join(project_root, 'data', 'stats_frac_2d')
os.makedirs(savepath_muvar_heat, exist_ok=True)
os.makedirs(savepath_muvar_frac, exist_ok=True)

# reversible instance normalization
rev_heat_2d = RevIN(savepath_muvar_heat)
rev_frac_2d = RevIN(savepath_muvar_frac)

# savepath of normalized data
# --- New CL sets ---
savepath_norm_data_heat_2d = cfg['2dHEAT']['file_path_2dheat_n']
savepath_norm_data_frac_2d = cfg['2dFRAC_tung']['file_path_2dfrac_n']
    
# ensure the parent trees exist:
for base in (savepath_norm_data_heat_2d, savepath_norm_data_frac_2d):
    os.makedirs(base, exist_ok=True) # Create the full base paths (and any parents) if needed
    
# create folders if they don't exist
for base in (savepath_norm_data_heat_2d, savepath_norm_data_frac_2d):
    for split in ('train','val','test'):
        os.makedirs(os.path.join(base, split), exist_ok=True)

#%% NORMALIZATION FOR HEAT2D DATA 
##################################################################
######################## HEAT2d data #############################
##################################################################

from src.utils.dataloaders.dataloader_heat2d import HEAT2dDataLoader

dataset_heat_2d = HEAT2dDataLoader(loadpath_heat_2d)
train_data, val_data = dataset_heat_2d.split_train()
test_data = dataset_heat_2d.split_test()
dataset_heat_2d = np.concatenate((train_data,val_data,test_data), axis = 0).astype(np.float16)  # (N,T,F,C,D,H,W)
print("Shape of HEAT2d data", dataset_heat_2d.shape)  # (N,T,F,C,D,H,W)

N_train = train_data.shape[0]
N_val   = val_data.shape[0]

del train_data, val_data, test_data

#%% calculate revin stats for HEAT2d data and store it
dataset_heat_2d_norm_all = np.empty_like(dataset_heat_2d, dtype=np.float16)
for sim in range(dataset_heat_2d.shape[0]):
    dataset_heat_2d_sim = dataset_heat_2d[sim:sim+1].astype(np.float32)
    print(f"==>> Computing stats for sample {sim+1}/{dataset_heat_2d.shape[0]}")
    rev_heat_2d.compute_stats(dataset_heat_2d_sim, prefix=f'stats_heat_2d_{sim}')         # heat_2d_data: np.ndarray of shape (N,T,F,C,D,H,W)

    # normalize the data
    dataset_heat_2d_norm = rev_heat_2d.normalize(dataset_heat_2d_sim, prefix=f'stats_heat_2d_{sim}')
    print("Normalize dataset shape", dataset_heat_2d_norm.shape)

    # --- Check round‐trip via denormalize ---
    tol_1 = 1e-2
    recovered = rev_heat_2d.denormalize(dataset_heat_2d_norm, prefix=f'stats_heat_2d_{sim}')
    max_error = 0.0
    #print(f'Current sample: {i}, Current max_error:{max_error:.7f}')
    maxerror_i = np.max(np.abs(recovered - dataset_heat_2d_sim))  # saving some memory
    max_error = max(maxerror_i, max_error)
    print(f'Current max_error:{maxerror_i:.7f}')
    assert max_error < tol_1, "Denormalization did not perfectly recover original!"
    print("RevIN round-trip OK")

    # append to list
    dataset_heat_2d_norm_all[sim] = dataset_heat_2d_norm[0].astype(np.float16)

    del recovered, dataset_heat_2d_norm, dataset_heat_2d_sim
    
# Split back into train/val/test normalized sets ---
train_norm = dataset_heat_2d_norm_all[:N_train]
val_norm   = dataset_heat_2d_norm_all[N_train:N_train + N_val]
test_norm  = dataset_heat_2d_norm_all[N_train + N_val:]

del dataset_heat_2d_norm_all

#%% Gather filenames and derive chunk sizes per file
def get_files_and_chunks(split):
    in_dir = os.path.join(loadpath_heat_2d, split)
    files = sorted(f for f in os.listdir(in_dir) if f.endswith('.h5') or f.endswith('.hdf5'))
    chunks = []
    for f in files:
        with h5py.File(os.path.join(in_dir, f), 'r') as h5f:
            # each HEAT2d file holds 100 simulations
            n = h5f["data"].shape[0]
        chunks.append(n)
    return files, chunks

train_files, train_chunks = get_files_and_chunks('train')
val_files,   val_chunks   = get_files_and_chunks('val')
test_files,  test_chunks  = get_files_and_chunks('test')

for split, norm_data, files, chunks in [
    ('train', train_norm, train_files, train_chunks),
    ('val',   val_norm,   val_files,   val_chunks),
    ('test',  test_norm,  test_files,  test_chunks)]:
    
    out_dir = os.path.join(savepath_norm_data_heat_2d, split)
    ptr = 0
    for fname, sz in zip(files, chunks):
        # grab exactly as many *simulations* as the original file had
        chunk = norm_data[ptr:ptr + sz]    # shape (sz, T, F, C, D, H, W)
        ptr += sz

        # Original files store: (N, 2, F, H, W)
        # Current chunk is:        (sz, 2, F, 1, 1, H, W)
        chunk_out = chunk.squeeze(axis=(3, 4))  # -> (sz, 2, F, H, W)

        out_path = os.path.join(out_dir, fname)
        with h5py.File(out_path, 'w') as f_out:
            f_out.create_dataset('data', data=chunk_out, compression='lzf', dtype=np.float16) 

        print(f"[HEAT2d] Saved file: {fname}, chunks={sz}, shape={chunk_out.shape}, dtype={chunk_out.dtype}")

#%% NoRMALIZATION FOR FRAC2D DATA
##################################################################
######################## FRAC2d data #############################
##################################################################
from src.utils.dataloaders.dataloader_frac2d import FRAC2dDataLoader    

dataset_frac_2d = FRAC2dDataLoader(loadpath_frac_2d)
train_data, val_data = dataset_frac_2d.split_train()
test_data = dataset_frac_2d.split_test()
dataset_frac_2d = np.concatenate((train_data,val_data,test_data), axis = 0)  # (N,T,F,C,D,H,W)
print("Shape of FRAC2d data", dataset_frac_2d.shape)  # (N,T,F,C,D,H,W)
N_train = train_data.shape[0]
N_val   = val_data.shape[0]

del train_data, val_data, test_data

#%% calculate revin stats for HEAT2d data and store it
rev_frac_2d.compute_stats(dataset_frac_2d, prefix='stats_frac_2d')         # heat_2d_data: np.ndarray of shape (N,T,F,C,D,H,W)

# normalize the data
dataset_frac_2d_norm = rev_frac_2d.normalize(dataset_frac_2d, prefix='stats_frac_2d')
print("Normalize dataset shape", dataset_frac_2d_norm.shape)

# Checks for MHD ReVIN
tol_1 = 1e-4
# --- Check round‐trip via denormalize ---
recovered = rev_frac_2d.denormalize(dataset_frac_2d_norm, prefix='stats_frac_2d')
max_error = 0.0
for i in range(recovered.shape[0]):
    #print(f'Current sample: {i}, Current max_error:{max_error:.7f}')
    maxerror_i = np.max(np.abs(recovered[i] - dataset_frac_2d[i]))  # saving some memory
    max_error = max(maxerror_i, max_error)
assert max_error < tol_1, "Denormalization did not perfectly recover original!"
print("FRAC2d RevIN round-trip OK")
del recovered, dataset_frac_2d
   
# Split back into train/val/test normalized sets ---
train_norm = dataset_frac_2d_norm[:N_train]
val_norm   = dataset_frac_2d_norm[N_train:N_train + N_val]
test_norm  = dataset_frac_2d_norm[N_train + N_val:]

del dataset_frac_2d_norm

#%% Gather filenames and derive chunk sizes per file
print(f"train_norm shape: {train_norm.shape}, val_norm shape: {val_norm.shape}, test_norm shape: {test_norm.shape}")

def get_files_and_chunks(split):
    in_dir = os.path.join(loadpath_frac_2d, split)
    files = sorted(f for f in os.listdir(in_dir) if f.endswith('.h5') or f.endswith('.hdf5'))
    chunks = []
    for f in files:
        with h5py.File(os.path.join(in_dir, f), 'r') as h5f:
            # each FRAC2d file holds 1000 simulations
            n = h5f["data"].shape[0]
        chunks.append(n)
    return files, chunks

train_files, train_chunks = get_files_and_chunks('train')
val_files,   val_chunks   = get_files_and_chunks('val')
test_files,  test_chunks  = get_files_and_chunks('test')

for split, norm_data, files, chunks in [
    ('train', train_norm, train_files, train_chunks),
    ('val',   val_norm,   val_files,   val_chunks),
    ('test',  test_norm,  test_files,  test_chunks)]:
    
    out_dir = os.path.join(savepath_norm_data_frac_2d, split)
    ptr = 0
    for fname, sz in zip(files, chunks):
        # grab exactly as many *simulations* as the original file had
        chunk = norm_data[ptr:ptr + sz]    # shape (sz, T, F, C, D, H, W)
        ptr += sz

        # Original files store: (N, 2, H, W)
        # Current chunk is:        (sz, 2, 1, 1, 1, H, W)
        chunk_out = chunk.squeeze(axis=(2, 3, 4))  # -> (sz, 2, H, W)

        out_path = os.path.join(out_dir, fname)
        with h5py.File(out_path, 'w') as f_out:
            f_out.create_dataset('data', data=chunk_out, compression='lzf', dtype=np.float32) 

        print(f"[FRAC2d] Saved file: {fname}, chunks={sz}, shape={chunk_out.shape}, dtype={chunk_out.dtype}")
# %%
