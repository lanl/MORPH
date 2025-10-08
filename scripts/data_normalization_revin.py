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
# --- Pretraining ---
loadpath_dr2d  = cfg['DR2d_data_pdebench']['file_path_dr2d']
loadpath_mhd3d  = cfg['MHD3d_data_thewell']['file_path_mhd3d']
loadpath_cfd1d  = cfg['1dcfd_pdebench']['file_path_cfd1d']
loadpath_cfd2dic  = cfg['2dcfd_ic_pdebench']['file_path_cfd2d_ic']
loadpath_cfd3d  = cfg['3dcfd_pdebench']['file_path_cfd3d']
loadpath_sw2d  = cfg['2dSW_pdebench']['file_path_sw2d']
# --- Finetuning ---
loadpath_dr1d  = cfg['1ddr_pdebench']['file_path_dr1d']
loadpath_cfd2d  = cfg['2dcfd_pdebench']['file_path_cfd2d']
loadpath_cfd3d_turb = cfg['3dcfd_turb_pdebench']['file_path_3dcfd_turb']
loadpath_be1d = cfg['1dbe_pdebench']['file_path_be1d']
loadpath_gsdr2d = cfg['2dgrayscottdr_thewell']['file_path_2dgsdr']
loadpath_tgc3d = cfg['3dturbgravitycool_thewell']['file_path_3dtgc']
loadpath_fns_kf_2d = cfg['2dFNS_KF_pdegym']['file_path_2dfns_kf']
# --- New pretraining sets ---
loadpath_ce_crp_2d = cfg['2dCE_CRP_pdegym']['file_path_2dce_crp']
loadpath_ce_kh_2d = cfg['2dCE_KH_pdegym']['file_path_2dce_kh']
loadpath_ce_rp_2d = cfg['2dCE_RP_pdegym']['file_path_2dce_rp']
loadpath_ce_gauss_2d = cfg['2dCE_Gauss_pdegym']['file_path_2dce_gauss']
loadpath_ns_sines_2d = cfg['2dNS_Sines_pdegym']['file_path_2dns_sines']
loadpath_ns_gauss_2d = cfg['2dNS_Gauss_pdegym']['file_path_2dns_gauss']

# create folders if they don't exist
for base in (loadpath_dr2d, loadpath_mhd3d, loadpath_cfd1d, 
             loadpath_cfd1d, loadpath_cfd3d, loadpath_cfd3d,
             loadpath_sw2d, loadpath_dr1d, loadpath_cfd2d,
             loadpath_be1d, loadpath_cfd3d_turb, loadpath_gsdr2d,
             loadpath_tgc3d, loadpath_fns_kf_2d,
             loadpath_ce_crp_2d, loadpath_ce_kh_2d, loadpath_ce_rp_2d,
             loadpath_ce_gauss_2d, loadpath_ns_sines_2d, loadpath_ns_gauss_2d):
    for split in ('train','val','test'):
        os.makedirs(os.path.join(base, split), exist_ok=True)
        
# savepath of mu and var
savepath_muvar = os.path.join(project_root, 'data')

# reversible instance normalization
(rev_mhd, rev_dr, rev_cfd1d, rev_sw2d, rev_cfd2dic, rev_cfd3d,
 rev_dr1d, rev_cfd2d, rev_be1d, rev_cfd3d_turb, rev_gsdr2d, rev_tgc3d, rev_fns_kf_2d,
 rev_ce_2d, rev_ns_2d) = (RevIN(savepath_muvar) for _ in range(15))

# savepath of normalized data
# --- Pretraining ---
savepath_norm_data_dr2d = cfg['DR2d_data_pdebench']['file_path_dr2d_n']
savepath_norm_data_mhd3d = cfg['MHD3d_data_thewell']['file_path_mhd3d_n']
savepath_norm_data_cfd1d = cfg['1dcfd_pdebench']['file_path_cfd1d_n']
savepath_norm_data_cfd2dic = cfg['2dcfd_ic_pdebench']['file_path_cfd2d_ic_n']
savepath_norm_data_cfd3d = cfg['3dcfd_pdebench']['file_path_cfd3d_n']
savepath_norm_data_sw2d = cfg['2dSW_pdebench']['file_path_sw2d_n']

# --- Finetuning ---
savepath_norm_data_dr1d = cfg['1ddr_pdebench']['file_path_dr1d_n']
savepath_norm_data_cfd2d = cfg['2dcfd_pdebench']['file_path_cfd2d_n']
savepath_norm_data_cfd3d_turb = cfg['3dcfd_turb_pdebench']['file_path_3dcfd_turb_n']
savepath_norm_data_be1d = cfg['1dbe_pdebench']['file_path_be1d_n']
savepath_norm_data_gsdr2d = cfg['2dgrayscottdr_thewell']['file_path_2dgsdr_n']
savepath_norm_data_tgc3d = cfg['3dturbgravitycool_thewell']['file_path_3dtgc_n']
savepath_norm_data_fns_kf_2d = cfg['2dFNS_KF_pdegym']['file_path_2dfns_kf_n']

# --- New Pretraining sets ---
savepath_norm_data_ce_crp_2d = cfg['2dCE_CRP_pdegym']['file_path_2dce_crp_n']
savepath_norm_data_ce_rp_2d = cfg['2dCE_RP_pdegym']['file_path_2dce_rp_n']
savepath_norm_data_ce_kh_2d = cfg['2dCE_KH_pdegym']['file_path_2dce_kh_n']
savepath_norm_data_ce_gauss_2d = cfg['2dCE_Gauss_pdegym']['file_path_2dce_gauss_n']
savepath_norm_data_ns_sines_2d = cfg['2dNS_Sines_pdegym']['file_path_2dns_sines_n']
savepath_norm_data_ns_gauss_2d = cfg['2dNS_Gauss_pdegym']['file_path_2dns_gauss_n']
    
# ensure the parent trees exist:
# - datasets/normalized_revin
# - datasets/normalized_revin/MHD3d_data
# - datasets/normalized_revin/DR_data
for base in (savepath_norm_data_dr2d, savepath_norm_data_mhd3d,  savepath_norm_data_cfd1d, 
             savepath_norm_data_cfd2dic, savepath_norm_data_cfd3d, savepath_norm_data_sw2d,
             savepath_norm_data_dr1d, savepath_norm_data_cfd2d, savepath_norm_data_cfd3d_turb, 
             savepath_norm_data_be1d, savepath_norm_data_gsdr2d, savepath_norm_data_tgc3d, 
             savepath_norm_data_fns_kf_2d,
             # new pretraining sets 
             savepath_norm_data_ce_crp_2d, savepath_norm_data_ce_rp_2d, 
             savepath_norm_data_ce_kh_2d, savepath_norm_data_ce_gauss_2d, 
             savepath_norm_data_ns_sines_2d, savepath_norm_data_ns_gauss_2d):
    os.makedirs(base, exist_ok=True) # Create the full base paths (and any parents) if needed
    
# create folders if they don't exist
for base in (savepath_norm_data_dr2d, savepath_norm_data_mhd3d, savepath_norm_data_cfd1d, 
             savepath_norm_data_cfd2dic, savepath_norm_data_cfd3d, savepath_norm_data_sw2d,
             savepath_norm_data_dr1d, savepath_norm_data_cfd2d, savepath_norm_data_cfd3d_turb,
             savepath_norm_data_be1d,savepath_norm_data_gsdr2d, savepath_norm_data_tgc3d, 
             savepath_norm_data_fns_kf_2d,
             savepath_norm_data_ce_crp_2d, savepath_norm_data_ce_rp_2d, savepath_norm_data_ce_kh_2d,
             savepath_norm_data_ce_gauss_2d, savepath_norm_data_ns_sines_2d, savepath_norm_data_ns_gauss_2d):
    for split in ('train','val','test'):
        os.makedirs(os.path.join(base, split), exist_ok=True)

#%% --->> PRETRAINING DATASETS
#%% MHD3D data
from src.utils.dataloaders.dataloader_mhd import MHDDataLoader

dataset_mhd = MHDDataLoader(loadpath_mhd3d)
train_data, val_data = dataset_mhd.split_train()
test_data = dataset_mhd.split_test()
dataset_mhd = np.concatenate((train_data,val_data,test_data), axis = 0)
print("Shape of MHD data", dataset_mhd.shape) # (N,T,D,H,W,C,F)

# Reshape MHD data into (N,T,D,H,W,C,F)->(N,T,F,C,D,H,W)
dataset_mhd = dataset_mhd.transpose(0, 1, 6, 5, 2, 3, 4)
print("Reshape of MHD data", dataset_mhd.shape)

# calculate revin stats for MHD data and store it
rev_mhd.compute_stats(dataset_mhd, prefix='stats_mhd')         # mhd_data: np.ndarray of shape (N,T,C,F,D,H,W)

# normalize the data
dataset_mhd_norm = rev_mhd.normalize(dataset_mhd, prefix='stats_mhd')
print("Normalize dataset shape", dataset_mhd_norm.shape)

# Checks for MHD ReVIN
tol_1 = 1e-4
# Check round‐trip via denormalize
recovered = rev_mhd.denormalize(dataset_mhd_norm, prefix='stats_mhd')
diff = np.abs(recovered - dataset_mhd)
print(f"Round-trip max abs error: {diff.max():.3e}")
assert diff.max() < tol_1, "Denormalization did not perfectly recover original!"
print(" GSDR RevIN round-trip OK")
   
# Split back into train/val/test normalized sets ---
N_train = train_data.shape[0]
N_val   = val_data.shape[0]
train_norm = dataset_mhd_norm[:N_train]
val_norm   = dataset_mhd_norm[N_train:N_train + N_val]
test_norm  = dataset_mhd_norm[N_train + N_val:]

del train_data, val_data, test_data

# Gather filenames and derive chunk sizes per file
def get_files_and_chunks(split):
    in_dir = os.path.join(loadpath_mhd3d, split)
    files = sorted(f for f in os.listdir(in_dir) if f.endswith('.h5') or f.endswith('.hdf5'))
    chunks = []
    for f in files:
        with h5py.File(os.path.join(in_dir, f), 'r') as h5f:
            # each MHD file holds one or more sims along axis=0 of magnetic_field
            n = h5f['t1_fields/magnetic_field'].shape[0]
        chunks.append(n)
    return files, chunks

train_files, train_chunks = get_files_and_chunks('train')
val_files,   val_chunks   = get_files_and_chunks('val')
test_files,  test_chunks  = get_files_and_chunks('test')

for split, norm_data, files, chunks in [
    ('train', train_norm, train_files, train_chunks),
    ('val',   val_norm,   val_files,   val_chunks),
    ('test',  test_norm,  test_files,  test_chunks)]:
    
    out_dir = os.path.join(savepath_norm_data_mhd3d, split)
    ptr = 0
    for fname, sz in zip(files, chunks):
        # grab exactly as many *simulations* as the original file had
        chunk = norm_data[ptr:ptr + sz]    # shape (sz, T, C, F, D, H, W)
        ptr += sz

        # transpose back to (sz, T, D, H, W, C, F)
        chunk_out = chunk.transpose(0,1,4,5,6,3,2)

        out_path = os.path.join(out_dir, fname)
        with h5py.File(out_path, 'w') as f_out:
            # --- t1_fields group ---
            g1 = f_out.create_group('t1_fields')
            # field 0 = magnetic_field, field 1 = velocity
            # keep any time‐axis if T>1
            magnetic = chunk_out[..., :, 0]
            velocity = chunk_out[..., :, 1]
            print(f"[Save] Shape of B: {magnetic.shape}, V: {velocity.shape}")
            
            g1.create_dataset('magnetic_field',data=magnetic,compression='lzf')
            g1.create_dataset('velocity',data=velocity,compression='lzf')

            # --- t0_fields group ---
            g0 = f_out.create_group('t0_fields')
            # field 2 = density; drop the redundant channels axis to match original (one channel)
            density = chunk_out[..., 0, 2]
            print(f"[Save] Shape of Density: {density.shape}")
            g0.create_dataset('density',data=density,compression='lzf')

        print(f"[MHD] Saved file: {fname}, chunks={sz}, shape(s)=mag{magnetic.shape}, vel{velocity.shape}, dens{density.shape}")

print("Normalized MHD data saved under:", savepath_norm_data_mhd3d)

#%% DR2D data
from src.utils.dataloaders.dataloader_dr import split_and_save_h5, DR2DDataLoader

# first split the raw DR data into train/test/val. raw_h5_loadpath and data_path are the load path and save path
split_and_save_h5(raw_h5_loadpath = loadpath_dr2d, 
                  savepath = loadpath_dr2d, 
                  dataset_name='DR', 
                  train_frac = 0.8,
                  rand = True)

# load the splited raw DR data
loader = DR2DDataLoader(loadpath_dr2d)
train, val = loader.split_train()          # data is already inflated
test = loader.split_test()
dataset = np.concatenate((train,val,test), axis = 0)
print("Shape of inflated DR data", dataset.shape)     # (N, T, D, H, W, C, F)  

# Reshape & expand dims for RevIN (N, T, D, H, W, C, F) -> (N, T, F, C, D, H, W)
dataset_tr = dataset.transpose(0, 1, 6, 5, 2, 3, 4)
print("Transposed DR data", dataset_tr.shape)

# compute & normalize
rev_dr.compute_stats(dataset_tr, prefix='stats_dr')
dataset_dr_norm = rev_dr.normalize(dataset_tr, prefix='stats_dr')
print("Normalize dataset shape", dataset_dr_norm.shape)

# Check for DR dataset
# Check round‐trip via denormalize
tol_2 = 1e-5
recovered = rev_dr.denormalize(dataset_dr_norm, prefix='stats_dr')
print("Denormalized dataset shape", recovered.shape)

max_error = 0.0
for i in range(recovered.shape[0]):
    maxerror_i = np.max(np.abs(recovered[i] - dataset_tr[i]))  # saving some memory
    max_error = max(maxerror_i, max_error)
assert max_error < tol_2, "Denormalization did not perfectly recover original!"
print("DR RevIN round-trip OK")
del recovered

# --- Save the data in the same format as raw (N,T,H,W,F) ---
dataset_sq   = dataset_dr_norm.transpose(0, 1, 4, 5, 6, 3, 2)   # (N, T, D, H, W, C, F)
dataset_sq = np.squeeze(dataset_sq, axis = 2)[:,:,:,:,0,:] # since C is repeated, taking only one
print("Normed dataset in raw shape", dataset_sq.shape)

# save single HDF5 with same filename
raw_files = [f for f in os.listdir(loadpath_dr2d) if f.endswith('.h5') or f.endswith('.hdf5')]  
filename = raw_files[0] # get the name of the file
out_path = os.path.join(savepath_norm_data_dr2d, filename)
with h5py.File(out_path, 'w') as f_out:
    for i in range(dataset_sq.shape[0]): # since test and val are repeated
        grp = f_out.create_group(f"{i:04d}")
        grp.create_dataset('data', data = dataset_sq[i], compression='lzf')
print("Saved normalized DR to", out_path)

# split the normed DR data into train/test/val
split_and_save_h5(savepath_norm_data_dr2d, savepath_norm_data_dr2d,
                  dataset_name = 'DR',
                  train_frac = 0.8,
                  rand = False)

#%% CFD1D data
from src.utils.dataloaders.dataloader_cfd1d import CFD1dDataLoader, split_and_save_h5

# first split the raw DR data into train/test/val. raw_h5_loadpath and data_path are the load path and save path
split_and_save_h5(loadpath_cfd1d, loadpath_cfd1d, train_frac = 0.8, rand = True)

# load the splited raw DR data
# data is already inflated to shape (N,T,D,H,W,C,F)
num_files = 4 # fifth file is shock
loader = CFD1dDataLoader(data_path = loadpath_cfd1d, dataset_name='CFD1d') 
train, val = loader.split_train(num_files=num_files) 
test = loader.split_test(num_files=num_files)
dataset = np.concatenate((train,val,test), axis = 0)
print("Shape of CFD1d data", dataset.shape)
del train, val, test

# Reshape & expand dims for RevIN
dataset_tr = dataset.transpose(0, 1, 6, 5, 2, 3, 4)           # (N, T, F, C, D, H, W)
print("Transposed CFD1d data", dataset_tr.shape)

# --- REVIN normalization ---
rev_cfd1d.compute_stats(dataset_tr, prefix='stats_cfd1d')
dataset_cfd1d_norm = rev_cfd1d.normalize(dataset_tr, prefix='stats_cfd1d')
print("Normalize dataset shape", dataset_cfd1d_norm.shape)

# --- Check round‐trip via denormalize ---
tol_3 = 2e-5
recovered = rev_cfd1d.denormalize(dataset_cfd1d_norm, prefix='stats_cfd1d')
max_error = 0.0
for i in range(recovered.shape[0]):
    maxerror_i = np.max(np.abs(recovered[i] - dataset_tr[i]))  # saving some memory
    max_error = max(maxerror_i, max_error)
assert max_error < tol_3, "Denormalization did not perfectly recover original!"
print("CFD1D RevIN round-trip OK")
del recovered

# --- Save the normed data in the same format as raw (N,T,W,F) ---
# reshape into the shape (N, T, D, H, W, C, F)
dataset_sq   = dataset_cfd1d_norm.transpose(0, 1, 4, 5, 6, 3, 2)   # (N, T, D, H, W, C, F)
dataset_sq = np.squeeze(dataset_sq, axis = (2,3))                  # squeeze D, H
dataset_sq = dataset_sq[:,:,:,0,:]                                 # squeeze C dim (if C is repeated, use 1)             
print("Normed Data in Raw shape", dataset_sq.shape)

# load raw files
raw_files = sorted([f for f in os.listdir(loadpath_cfd1d)
                    if f.endswith(".h5") or f.endswith(".hdf5")])
raw_files = raw_files[:num_files]
splits= len(raw_files)
N = dataset_sq.shape[0]
chunk_size = N // splits
# Loop over each chunk and save
for i, fname in enumerate(raw_files):
    start = i * chunk_size
    end   = (i + 1) * chunk_size
    chunk = dataset_sq[start:end]  # shape (chunk_size, H, W, F)
    out_path = os.path.join(savepath_norm_data_cfd1d, fname)
    with h5py.File(out_path, "w") as f5:
        # assuming F == 3 and channel order [Vx, density, pressure]
        vel_chunk = chunk[..., 0]
        dens_chunk = data=chunk[..., 1]
        pres_chunk = data=chunk[..., 2]
        print(f'Shapes of chunks: {vel_chunk.shape}, {dens_chunk.shape}, '
              f'{pres_chunk.shape}')
        f5.create_dataset("Vx",       data=vel_chunk, compression="gzip")
        f5.create_dataset("density",  data=dens_chunk, compression="gzip")
        f5.create_dataset("pressure", data=pres_chunk, compression="gzip")
        
    print(f"Saved {chunk.shape[0]} frames "
          f"to {os.path.basename(out_path)}")
    
# Now split the normed data at savepath_norm_data_cfd1d
split_and_save_h5(savepath_norm_data_cfd1d, savepath_norm_data_cfd1d, 
                  train_frac = 0.8, rand = False)

#%% SW2D data
'''
data loading and processing similar to DR dataset.
'''
from src.utils.dataloaders.dataloader_sw2d import split_and_save_h5, SW2dDataLoader

# first split the raw DR data into train/test/val. raw_h5_loadpath and data_path are the load path and save path
split_and_save_h5(raw_h5_loadpath = loadpath_sw2d, 
                  savepath = loadpath_sw2d, 
                  dataset_name='SW2d', 
                  train_frac = 0.8, rand = True)

# load the splited raw DR data
loader = SW2dDataLoader(loadpath_sw2d)
train, val = loader.split_train()          # data is already inflated
test = loader.split_test()
dataset = np.concatenate((train,val,test), axis = 0)
print("Shape of inflated SW data", dataset.shape)        

# Reshape & expand dims for RevIN
dataset_tr = dataset.transpose(0, 1, 6, 5, 2, 3, 4)           # (N, T, F, C, D, H, W)
print("Transposed SW data", dataset_tr.shape)

# compute & normalize
rev_sw2d.compute_stats(dataset_tr, prefix='stats_sw')
dataset_sw2d_norm = rev_sw2d.normalize(dataset_tr, prefix='stats_sw')
print("Normalize dataset shape", dataset_sw2d_norm.shape)

# Check for SW2d dataset
# Check round‐trip via denormalize
tol_4 = 1e-5
recovered = rev_sw2d.denormalize(dataset_sw2d_norm, prefix='stats_sw')
max_error = 0.0
for i in range(recovered.shape[0]):
    maxerror_i = np.max(np.abs(recovered[i] - dataset_tr[i]))  # saving some memory
    max_error = max(maxerror_i, max_error)
assert max_error < tol_4, "Denormalization did not perfectly recover original!"
print("DR RevIN round-trip OK")
del recovered

# --- Save the data in the same format as raw (N,T,H,W,F) ---
dataset_sq   = dataset_sw2d_norm.transpose(0, 1, 4, 5, 6, 3, 2)   # (N, T, D, H, W, C, F)
dataset_sq = np.squeeze(dataset_sq, axis = 2)[:,:,:,:,0,:] # squeeze C dim (if C is repeated, use 1)
print("Normed dataset in raw shape", dataset_sq.shape)

# save single HDF5 with same filename
raw_files = [f for f in os.listdir(loadpath_sw2d) if f.endswith('.h5') or f.endswith('.hdf5')]  
filename = raw_files[0] # get the name of the file
out_path = os.path.join(savepath_norm_data_sw2d, filename)
with h5py.File(out_path, 'w') as f_out:
    for i in range(dataset_sq.shape[0]): # since test and val are repeated
        grp = f_out.create_group(f"{i:04d}")
        grp.create_dataset('data', data = dataset_sq[i], compression='lzf')
print("Saved normalized DR to", out_path)

# split the normed DR data into train/test/val
split_and_save_h5(savepath_norm_data_sw2d, savepath_norm_data_sw2d,
                  dataset_name = 'SW2d',
                  train_frac = 0.8,
                  rand = False)

#%% CFD2d (IC)
####################### Load and process CFD2d-IC data ##############################
from src.utils.dataloaders.dataloader_cfd2dic import split_and_save_h5, CFD2dicDataLoader

# first split the raw DR data into train/test/val. raw_h5_loadpath and data_path are the load path and save path
split_and_save_h5(raw_dir = loadpath_cfd2dic, 
                  out_dir = loadpath_cfd2dic, 
                  dataset_name='cfd2dic', 
                  train_frac = 0.8, rand = True)

# load the splited raw DR data
loader = CFD2dicDataLoader(loadpath_cfd2dic, force = True)
train, val = loader.split_train()          # data is already inflated
test = loader.split_test()
dataset = np.concatenate((train,val,test), axis = 0)
del train, val, test
print("Shape of inflated CFD2d data", dataset.shape)        

# Reshape & expand dims for RevIN
dataset = dataset.transpose(0, 1, 6, 5, 2, 3, 4)           # (N, T, F, C, D, H, W)
print("Transposed CFD2d data", dataset.shape)

# compute & normalize
rev_cfd2dic.compute_stats(dataset, prefix='stats_cfd2d-ic')
dataset_cfd2dic_norm = rev_cfd2dic.normalize(dataset, prefix='stats_cfd2d-ic')
print("Normalize dataset shape", dataset_cfd2dic_norm.shape)

# Check round‐trip via denormalize
tol_5 = 1e-5
recovered = rev_cfd2dic.denormalize(dataset_cfd2dic_norm, prefix='stats_cfd2d-ic')
max_error = 0.0
for i in range(recovered.shape[0]):
    # print(f'Current sample: {i}, Current max_error:{max_error:.7f}')
    maxerror_i = np.max(np.abs(recovered[i] - dataset[i]))  # saving some memory
    max_error = max(maxerror_i, max_error)
assert max_error < tol_5, "Denormalization did not perfectly recover original!"
print("DR RevIN round-trip OK")
del recovered

# --- Save the data in the same format as raw (N,T,H,W,F) ---
dataset_sq   = dataset_cfd2dic_norm.transpose(0, 1, 4, 5, 6, 3, 2)   # (N, T, D, H, W, C, F)
dataset_sq = np.squeeze(dataset_sq, axis = 2) # Final shape (N,T,H,W,C,F)
print("Normed dataset in raw shape", dataset_sq.shape)

# save single HDF5 with same filename
# load raw files
raw_files = sorted([f for f in os.listdir(loadpath_cfd2dic)
                    if f.endswith(".h5") or f.endswith(".hdf5")])
splits= 4
N = dataset_sq.shape[0]
chunk_size = N // splits
# Loop over each chunk and save
for i, fname in enumerate(raw_files):
    print(f'Processing: {raw_files}...')
    start = i * chunk_size
    end   = (i + 1) * chunk_size
    chunk = dataset_sq[start:end]  # shape (chunk_size, T, H, W, C, F)
    
    # reconstruct force & velocity in raw form 
    # force: pick either the first timestep (all fx, fy are the same over T)
    force_chunk = chunk[:, 0, :, :, :, 0]    # → (m, H, W, 2)
    # velocity: take the full time‐series
    vel_chunk   = chunk[..., 1]   # → (m, T, H, W, 2)
    
    out_path = os.path.join(savepath_norm_data_cfd2dic, fname)
    with h5py.File(out_path, "w") as f5:
        # assuming F == 3 and channel order [Vx, density, pressure]
        f5.create_dataset("force",    data=force_chunk, compression="gzip")
        f5.create_dataset("velocity",  data=vel_chunk, compression="gzip")
    
    print(f"Saved force: {force_chunk.shape}, vel: {vel_chunk.shape} trajectories to {fname}")

# split the normed DR data into train/test/val
split_and_save_h5(savepath_norm_data_cfd2dic, savepath_norm_data_cfd2dic,
                  dataset_name = 'cfd2dic',
                  train_frac = 0.8,
                  rand = False)

#%% CFD3D
from src.utils.dataloaders.dataloader_cfd3d import CFD3dDataLoader, split_and_save_h5

# first split the raw DR data into train/test/val. raw_h5_loadpath and data_path are the load path and save path
split_and_save_h5(raw_dir = loadpath_cfd3d, 
                  out_dir = loadpath_cfd3d, 
                  select_nfiles = 2, 
                  train_frac = 0.8, rand = True)

# load the splited raw DR data
loader = CFD3dDataLoader(data_path = loadpath_cfd3d, dataset_name='CFD3d') 
train, val = loader.split_train()                                       
test = loader.split_test()
dataset = np.concatenate((train,val,test), axis = 0)
print("Shape of CFD3d data", dataset.shape)        
del train, val, test

# Reshape & expand dims for RevIN
dataset = dataset.transpose(0, 1, 6, 5, 2, 3, 4)           # (N, T, F, C, D, H, W)
print("Transposed CFD3d data", dataset.shape)

# --- REVIN normalization ---
rev_cfd3d.compute_stats(dataset, prefix='stats_cfd3d')
dataset_cfd3d_norm = rev_cfd3d.normalize(dataset, prefix='stats_cfd3d')
print("Normalize dataset shape", dataset_cfd3d_norm.shape)

# --- Check round‐trip via denormalize ---
recovered = rev_cfd3d.denormalize(dataset_cfd3d_norm, prefix='stats_cfd3d')
tol_6 = 7e-5
max_error = 0.0
for i in range(recovered.shape[0]):
    #print(f'Current sample: {i}, Current max_error:{max_error:.7f}')
    maxerror_i = np.max(np.abs(recovered[i] - dataset[i]))  # saving some memory
    max_error = max(maxerror_i, max_error)
assert max_error < tol_6, "Denormalization did not perfectly recover original!"
print("CFD3D RevIN round-trip OK")
del recovered

# --- Save the normed data in the same format as raw (N,T,W,F) ---
# reshape into the shape (N, T, D, H, W, C, F)
dataset_sq   = dataset_cfd3d_norm.transpose(0, 1, 4, 5, 6, 3, 2)   # (N, T, D, H, W, C, F)
print("Normed Data in Raw shape", dataset_sq.shape)

# load raw files
raw_files = sorted([f for f in os.listdir(loadpath_cfd3d)
                    if f.endswith(".h5") or f.endswith(".hdf5")])
splits= len(raw_files)
N = dataset_sq.shape[0]
chunk_size = N // splits
# Loop over each chunk and save
for i, fname in enumerate(raw_files):
    start = i * chunk_size
    end   = (i + 1) * chunk_size
    chunk = dataset_sq[start:end]

    out_path = os.path.join(savepath_norm_data_cfd3d, fname)
    with h5py.File(out_path, "w") as f5:
        # assuming F == 3 and channel order [Vx, density, pressure]
        vel_chunk = chunk[...,0]
        vx_chunk, vy_chunk, vz_chunk  = vel_chunk[...,0], vel_chunk[...,1], vel_chunk[...,2]
        den_chunk = chunk[...,0,1]  # since density is repeated in component dim
        pre_chunk = chunk[...,0,2]  # since pressure is repeated in component dim
        print(f'vx:{vx_chunk.shape},vy:{vy_chunk.shape},vz:{vz_chunk.shape},den:{den_chunk.shape},pre:{pre_chunk.shape}')
        f5.create_dataset("Vx",       data=vx_chunk, compression="gzip")
        f5.create_dataset("Vy",       data=vy_chunk, compression="gzip")
        f5.create_dataset("Vz",       data=vz_chunk, compression="gzip")
        f5.create_dataset("density",  data=den_chunk, compression="gzip")
        f5.create_dataset("pressure", data=pre_chunk, compression="gzip")

    print(f"Saved {chunk.shape[0]} frames "
          f"to {os.path.basename(out_path)}")
    
# Now split the normed data at savepath_norm_data_cfd1d
split_and_save_h5(savepath_norm_data_cfd3d, savepath_norm_data_cfd3d,
                  select_nfiles = 2,
                  dataset_name = 'cfd3d',
                  train_frac = 0.8,
                  rand = False)

#%% --->> FINETUNING DATASETS
#%% DR1d
from src.utils.dataloaders.dataloader_dr1d import split_and_save_h5, DR1DDataLoader

# first split the raw DR data into train/test/val. raw_h5_loadpath and data_path are the load path and save path
split_and_save_h5(raw_h5_loadpath = loadpath_dr1d, 
                  savepath = loadpath_dr1d,
                  selected_idx = 0,
                  dataset_name='dr1d', 
                  train_frac = 0.8, rand = True)

# load the splited raw data
loader = DR1DDataLoader(data_path = loadpath_dr1d, dataset_name='DR1d') # data_path is savepath for split files
train, val = loader.split_train(selected_idx = 0) # data is already inflated to shape (N,T,D,H,W,C,F)
test = loader.split_test(selected_idx = 0)
dataset = np.concatenate((train, val, test), axis = 0)
print("Shape of DR1d concat data", dataset.shape)        
del train, val, test

# Reshape & expand dims for RevIN
dataset = dataset.transpose(0, 1, 6, 5, 2, 3, 4)           # (N, T, F, C, D, H, W)
print("Transposed CFD1d data", dataset.shape)

# --- REVIN normalization ---
rev_dr1d.compute_stats(dataset, prefix='stats_dr1d')
dataset_dr1d_norm = rev_dr1d.normalize(dataset, prefix='stats_dr1d')
print("Normalize dataset shape", dataset_dr1d_norm.shape)

# --- Check round‐trip via denormalize ---
recovered = rev_dr1d.denormalize(dataset_dr1d_norm, prefix='stats_dr1d')
tol_6 = 7e-5
max_error = 0.0
for i in range(recovered.shape[0]):
    print(f'Current sample: {i}, Current max_error:{max_error:.7f}')
    maxerror_i = np.max(np.abs(recovered[i] - dataset[i]))  # saving some memory
    max_error = max(maxerror_i, max_error)
assert max_error < tol_6, "Denormalization did not perfectly recover original!"
print("CFD3D RevIN round-trip OK")
del recovered

# --- Save the normed data in the same format as raw (N,T,W,F) ---
# reshape into the shape (N, T, D, H, W, C, F)
dataset_sq   = dataset_dr1d_norm.transpose(0, 1, 4, 5, 6, 3, 2)   # (N, T, D, H, W, C, F)
print("Normed Data in Raw shape", dataset_sq.shape)
dataset_sq = np.squeeze(dataset_sq, axis = (2,3,5,6)) # (N,T,W)
print("Normed dataset in raw shape", dataset_sq.shape)

# save single HDF5 with same filename
raw_files = [f for f in os.listdir(loadpath_dr1d) if f.endswith('.h5') or f.endswith('.hdf5')]
filename = raw_files[0] # get the name of the file
out_path = os.path.join(savepath_norm_data_dr1d, filename)
with h5py.File(out_path, 'w') as f_out:
    # Save the whole array at once under the name 'tensor'
    f_out.create_dataset('tensor', data=dataset_sq, compression='lzf')

# Now split the normed data at savepath_norm_data_cfd1d
split_and_save_h5(savepath_norm_data_dr1d, savepath_norm_data_dr1d,
                  dataset_name = 'dr1d',
                  selected_idx = 0,
                  train_frac = 0.8,
                  rand = False)

#%% CFD2D
####################### Load and process CFD2D data ##############################
from src.utils.dataloaders.dataloader_cfd2d import CFD2DDataLoader, split_and_save_h5

# first split the raw DR data into train/test/val. raw_h5_loadpath and data_path are the load path and save path
split_and_save_h5(raw_dir = loadpath_cfd2d, 
                  out_dir = loadpath_cfd2d,
                  select_nfiles = 1, 
                  train_frac = 0.8, rand = True)

# load the splited raw DR data
loader = CFD2DDataLoader(data_path = loadpath_cfd2d, dataset_name='CFD2d') # data_path is savepath for split files
train, val = loader.split_train(selected_idx = 0) # data is already inflated to shape (N,T,D,H,W,C,F)
test = loader.split_test(selected_idx = 0)
dataset = np.concatenate((train,val,test), axis = 0)
print("Shape of CFD2d data", dataset.shape)
del train, val, test

# Reshape & expand dims for RevIN
dataset = dataset.transpose(0, 1, 6, 5, 2, 3, 4)   # (N, T, F, C, D, H, W)
print("Transposed CFD1d data", dataset.shape)

# --- REVIN normalization ---
rev_cfd2d.compute_stats(dataset, prefix='stats_cfd2d')
dataset_cfd2d_norm = rev_cfd2d.normalize(dataset, prefix='stats_cfd2d')
print("Normalize dataset shape", dataset_cfd2d_norm.shape)

# --- Check round‐trip via denormalize ---
recovered = rev_cfd2d.denormalize(dataset_cfd2d_norm, prefix='stats_cfd2d')
tol_6 = 7e-5
max_error = 0.0
for i in range(recovered.shape[0]):
    print(f'Current sample: {i}, Current max_error:{max_error:.7f}')
    maxerror_i = np.max(np.abs(recovered[i] - dataset[i]))  # saving some memory
    max_error = max(maxerror_i, max_error)
assert max_error < tol_6, "Denormalization did not perfectly recover original!"
print("CFD2D RevIN round-trip OK")
del recovered

# --- Save the normed data in the same format as raw (N,T,W,F) ---
# reshape into the shape (N, T, D, H, W, C, F)
dataset_sq   = dataset_cfd2d_norm.transpose(0, 1, 4, 5, 6, 3, 2)   # (N, T, D, H, W, C, F)
print("Normed Data in Raw shape", dataset_sq.shape)

dataset_sq = np.squeeze(dataset_sq, axis = 2) # Final shape (N,T,H,W,C,F)
print("Normed dataset in raw shape", dataset_sq.shape)

# load raw files (ONLY ONE FILE)
raw_files = sorted([f for f in os.listdir(loadpath_cfd2d)
                    if f.endswith(".h5") or f.endswith(".hdf5")])
fname = raw_files[0]
out_path = os.path.join(savepath_norm_data_cfd2d, fname)
with h5py.File(out_path, "w") as f5:
    # F == 3 and channel order [Vx, density, pressure]
    vel_chunk = dataset_sq[...,0]
    vx_chunk, vy_chunk  = vel_chunk[...,0], vel_chunk[...,1]
    den_chunk = dataset_sq[...,0,1]  # since density is repeated in component dim
    pre_chunk = dataset_sq[...,0,2]  # since pressure is repeated in component dim
    print(f'vx:{vx_chunk.shape},vy:{vy_chunk.shape},den:{den_chunk.shape},pre:{pre_chunk.shape}')
    f5.create_dataset("Vx",       data=vx_chunk, compression="gzip")
    f5.create_dataset("Vy",       data=vy_chunk, compression="gzip")
    f5.create_dataset("density",  data=den_chunk, compression="gzip")
    f5.create_dataset("pressure", data=pre_chunk, compression="gzip")

print(f"Saved {dataset_sq.shape[0]} frames "
      f"to {os.path.basename(out_path)}")
    
# Now split the normed data at savepath_norm_data_cfd1d
split_and_save_h5(savepath_norm_data_cfd2d, savepath_norm_data_cfd2d,
                  select_nfiles = 1,
                  dataset_name = 'cfd2d',
                  train_frac = 0.8,
                  rand = False)

#%% CFD3D-Turb
from src.utils.dataloaders.dataloader_cfd3d_turb import CFD3dTurbDataLoader, split_and_save_h5

# first split the raw DR data into train/test/val. raw_h5_loadpath and data_path are the load path and save path
split_and_save_h5(raw_dir = loadpath_cfd3d_turb, 
                  out_dir = loadpath_cfd3d_turb, 
                  select_nfiles = 1, 
                  train_frac = 0.8, rand = True)

# load the splited raw DR data
loader = CFD3dTurbDataLoader(data_path = loadpath_cfd3d_turb, 
                             dataset_name='CFD3dTurb') # data_path is savepath for split files
train, val = loader.split_train() # data is already inflated to shape (N,T,D,H,W,C,F)
test = loader.split_test()
dataset = np.concatenate((train,val,test), axis = 0)
print("Shape of CFD3d-Turb data", dataset.shape)        
del train, val, test

# Reshape & expand dims for RevIN
dataset = dataset.transpose(0, 1, 6, 5, 2, 3, 4)           # (N, T, F, C, D, H, W)
print("Transposed CFD3D-Turb data", dataset.shape)

# --- REVIN normalization ---
rev_cfd3d_turb.compute_stats(dataset, prefix='stats_cfd3d-turb')
dataset_cfd3d_turb_norm = rev_cfd3d_turb.normalize(dataset, prefix='stats_cfd3d-turb')
print("Normalize dataset shape", dataset_cfd3d_turb_norm.shape)

# --- Check round‐trip via denormalize ---
recovered = rev_cfd3d_turb.denormalize(dataset_cfd3d_turb_norm, prefix='stats_cfd3d-turb')
tol_6 = 7e-5
max_error = 0.0
for i in range(recovered.shape[0]):
    print(f'Current sample: {i}, Current max_error:{max_error:.7f}')
    maxerror_i = np.max(np.abs(recovered[i] - dataset[i]))  # saving some memory
    max_error = max(maxerror_i, max_error)
assert max_error < tol_6, "Denormalization did not perfectly recover original!"
print("CFD3D-Turb RevIN round-trip OK")
del recovered

# --- Save the normed data in the same format as raw (N,T,W,F) ---
# reshape into the shape (N, T, D, H, W, C, F)
dataset_sq   = dataset_cfd3d_turb_norm.transpose(0, 1, 4, 5, 6, 3, 2)   # (N, T, D, H, W, C, F)
print("Normed Data in Raw shape", dataset_sq.shape)

# load raw files
raw_files = sorted([f for f in os.listdir(loadpath_cfd3d_turb)
                    if f.endswith(".h5") or f.endswith(".hdf5")])
splits= len(raw_files)
N = dataset_sq.shape[0]
chunk_size = N // splits
# Loop over each chunk and save
for i, fname in enumerate(raw_files):
    start = i * chunk_size
    end   = (i + 1) * chunk_size
    chunk = dataset_sq[start:end]

    out_path = os.path.join(savepath_norm_data_cfd3d_turb, fname)
    with h5py.File(out_path, "w") as f5:
        # assuming F == 3 and channel order [Vx, density, pressure]
        vel_chunk = chunk[...,0]
        vx_chunk, vy_chunk, vz_chunk  = vel_chunk[...,0], vel_chunk[...,1], vel_chunk[...,2]
        den_chunk = chunk[...,0,1]  # since density is repeated in component dim
        pre_chunk = chunk[...,0,2]  # since pressure is repeated in component dim
        print(f'vx:{vx_chunk.shape},vy:{vy_chunk.shape},vz:{vz_chunk.shape},den:{den_chunk.shape},pre:{pre_chunk.shape}')
        f5.create_dataset("Vx",       data=vx_chunk, compression="gzip")
        f5.create_dataset("Vy",       data=vy_chunk, compression="gzip")
        f5.create_dataset("Vz",       data=vz_chunk, compression="gzip")
        f5.create_dataset("density",  data=den_chunk, compression="gzip")
        f5.create_dataset("pressure", data=pre_chunk, compression="gzip")

    print(f"Saved {chunk.shape[0]} frames "
          f"to {os.path.basename(out_path)}")
    
# Now split the normed data at savepath_norm_data_cfd1d
split_and_save_h5(savepath_norm_data_cfd3d_turb, savepath_norm_data_cfd3d_turb,
                  select_nfiles = 1,
                  dataset_name = 'cfd3d_turb',
                  train_frac = 0.8,
                  rand = False)

#%% BE1D
from src.utils.dataloaders.dataloader_be1d import split_and_save_h5, BE1DDataLoader

# first split the raw DR data into train/test/val. 
# raw_h5_loadpath and data_path are the load path and save path
split_and_save_h5(raw_h5_loadpath = loadpath_be1d, 
                  savepath = loadpath_be1d,
                  selected_idx = 0,
                  dataset_name='be1d', 
                  train_frac = 0.8, rand = True)

# load the splited raw DR data
loader = BE1DDataLoader(data_path = loadpath_be1d, dataset_name='BE1d') # data_path is savepath for split files
train, val = loader.split_train(selected_idx = 0) # data is already inflated to shape (N,T,D,H,W,C,F)
test = loader.split_test(selected_idx = 0)
dataset = np.concatenate((train, val, test), axis = 0)
print("Shape of BE1D concat data", dataset.shape)        
del train, val, test

# Reshape & expand dims for RevIN
dataset = dataset.transpose(0, 1, 6, 5, 2, 3, 4)           # (N, T, F, C, D, H, W)
print("Transposed BE1D data", dataset.shape)

# --- REVIN normalization ---
rev_be1d.compute_stats(dataset, prefix='stats_be1d')
dataset_be1d_norm = rev_be1d.normalize(dataset, prefix='stats_be1d')
print("Normalize dataset shape", dataset_be1d_norm.shape)

# --- Check round‐trip via denormalize ---
recovered = rev_be1d.denormalize(dataset_be1d_norm, prefix='stats_be1d')
tol_6 = 7e-5
max_error = 0.0
for i in range(recovered.shape[0]):
    # print(f'Current sample: {i}, Current max_error:{max_error:.7f}')
    maxerror_i = np.max(np.abs(recovered[i] - dataset[i]))  # saving some memory
    max_error = max(maxerror_i, max_error)
assert max_error < tol_6, "Denormalization did not perfectly recover original!"
print("-> BE1D RevIN round-trip OK")
del recovered

# --- Save the normed data in the same format as raw (N,T,W,F) ---
# reshape into the shape (N, T, D, H, W, C, F)
dataset_sq   = dataset_be1d_norm.transpose(0, 1, 4, 5, 6, 3, 2)   # (N, T, D, H, W, C, F)
print("Normed Data in Raw shape", dataset_sq.shape)
dataset_sq = np.squeeze(dataset_sq, axis = (2,3,5,6)) # (N,T,W)
print("Normed dataset in raw shape", dataset_sq.shape)

# save single HDF5 with same filename
raw_files = [f for f in os.listdir(loadpath_be1d) if f.endswith('.h5') or f.endswith('.hdf5')]
filename = raw_files[0] # get the name of the file
out_path = os.path.join(savepath_norm_data_be1d, filename)
with h5py.File(out_path, 'w') as f_out:
    # Save the whole array at once under the name 'tensor'
    f_out.create_dataset('tensor', data=dataset_sq, compression='lzf')

# Now split the normed data at savepath_norm_data_cfd1d
split_and_save_h5(savepath_norm_data_be1d, savepath_norm_data_be1d,
                  dataset_name = 'be1d',
                  selected_idx = 0,
                  train_frac = 0.8,
                  rand = False)

#%% GSDR-2D
from src.utils.dataloaders.dataloader_gsdr2d import GSDR2dDataLoader

dataset_gsdr = GSDR2dDataLoader(loadpath_gsdr2d)
train_data, val_data = dataset_gsdr.split_train()
test_data = dataset_gsdr.split_test()
dataset_gsdr = np.concatenate((train_data,val_data,test_data), axis = 0)
print("Shape of GSDR2D data", dataset_gsdr.shape)

# Reshape GSDR data into (N,T,F,C,D,H,W)
dataset_gsdr = dataset_gsdr.transpose(0, 1, 6, 5, 2, 3, 4)
print("Reshape of MHD data", dataset_gsdr.shape)

# calculate revin stats for MHD data and store it
rev_gsdr2d.compute_stats(dataset_gsdr, prefix='stats_gsdr2d')  

# normalize the data
dataset_gsdr_norm = rev_gsdr2d.normalize(dataset_gsdr, prefix='stats_gsdr2d')
print("Normalize dataset shape", dataset_gsdr_norm.shape)

# Checks for GSDR ReVIN
tol_1 = 1e-4
# Check round‐trip via denormalize
recovered = rev_gsdr2d.denormalize(dataset_gsdr_norm, prefix='stats_gsdr2d')
diff = np.abs(recovered - dataset_gsdr)
print(f"Round-trip max abs error: {diff.max():.3e}")
assert diff.max() < tol_1, "Denormalization did not perfectly recover original!"

# Bring the raw shape (uninflate)
dataset_sq   = dataset_gsdr_norm.transpose(0, 1, 4, 5, 6, 3, 2)   # (N, T, D, H, W, C, F)
print("Normed Data in Raw shape", dataset_sq.shape)

dataset_sq = np.squeeze(dataset_sq, axis = (2,5)) # (N,T,H,W,F)
print(f'Raw shape of GSDR2d: {dataset_sq.shape}')

# Split back into train/val/test normalized sets ---
N_train = train_data.shape[0]
N_val   = val_data.shape[0]
train_norm = dataset_sq[:N_train]
val_norm   = dataset_sq[N_train:N_train + N_val]
test_norm  = dataset_sq[N_train + N_val:]

del train_data, val_data, test_data

# Gather filenames and derive chunk sizes per file
def get_files_and_chunks(split):
    in_dir = os.path.join(loadpath_gsdr2d, split)
    files = sorted(f for f in os.listdir(in_dir) if f.endswith('.h5') or f.endswith('.hdf5'))
    chunks = []
    for f in files:
        with h5py.File(os.path.join(in_dir, f), 'r') as h5f:
            # each MHD file holds one or more sims along axis=0 of magnetic_field
            n = h5f['t0_fields/A'].shape[0]
        chunks.append(n)
    return files, chunks

train_files, train_chunks = get_files_and_chunks('train')
val_files,   val_chunks   = get_files_and_chunks('val')
test_files,  test_chunks  = get_files_and_chunks('test')

for split, norm_data, files, chunks in [
    ('train', train_norm, train_files, train_chunks),
    ('val',   val_norm,   val_files,   val_chunks),
    ('test',  test_norm,  test_files,  test_chunks)]:
    
    out_dir = os.path.join(savepath_norm_data_gsdr2d, split)
    ptr = 0
    for fname, sz in zip(files, chunks):
        # grab exactly as many *simulations* as the original file had
        chunk = norm_data[ptr:ptr + sz]    # shape (sz, T, H, W, F)
        ptr += sz

        out_path = os.path.join(out_dir, fname)
        with h5py.File(out_path, 'w') as f_out:
            g = f_out.create_group('t0_fields')
            act_A = chunk[..., 0] # (sz, T, H, W)
            act_B = chunk[..., 1] # (sz, T, H, W)
            g.create_dataset('A',data=act_A,compression='lzf')
            g.create_dataset('B',data=act_B,compression='lzf')

        print(f"[GSDR2D] Saved file: {fname}, chunks={sz}, "
              f"shape(s) A={act_A.shape}, B={act_B.shape}")

print("Normalized GSDR data saved under:", savepath_norm_data_gsdr2d)

#%% TGC3d
from src.utils.dataloaders.dataloader_tgc3d import TGC3dDataLoader

dataset_tgc3d = TGC3dDataLoader(loadpath_tgc3d)
train_data, val_data = dataset_tgc3d.split_train()
test_data = dataset_tgc3d.split_test()
dataset_tgc3d = np.concatenate((train_data,val_data,test_data), axis = 0)
print("Shape of tgc3d data", dataset_tgc3d.shape) # (N,T,D,H,W,C,F)

# Reshape tgc3d data into (N,T,D,H,W,C,F) -> (N,T,F,C,D,H,W)
dataset_tgc3d = dataset_tgc3d.transpose(0, 1, 6, 5, 2, 3, 4)
print("Reshape of tgc3d data", dataset_tgc3d.shape)

# calculate revin stats for tgc3d data and store it
rev_tgc3d.compute_stats(dataset_tgc3d, prefix='stats_tgc3d')

# normalize the data
dataset_tgc3d_norm = rev_tgc3d.normalize(dataset_tgc3d, prefix='stats_tgc3d')
print("Normalize dataset shape", dataset_tgc3d_norm.shape)

# Checks for tgc3d ReVIN
tol_1 = 1e-3
# Check round‐trip via denormalize
recovered = rev_tgc3d.denormalize(dataset_tgc3d_norm, prefix='stats_tgc3d')
diff = np.abs(recovered - dataset_tgc3d)
print(f"Round-trip max abs error: {diff.max():.3e}")
assert diff.max() < tol_1, "Denormalization did not perfectly recover original!"
print(" TGC3D RevIN round-trip OK")
   
# Split back into train/val/test normalized sets ---
N_train = train_data.shape[0]
N_val   = val_data.shape[0]
train_norm = dataset_tgc3d_norm[:N_train]
val_norm   = dataset_tgc3d_norm[N_train:N_train + N_val]
test_norm  = dataset_tgc3d_norm[N_train + N_val:]

del train_data, val_data, test_data

# Gather filenames and derive chunk sizes per file
def get_files_and_chunks(split):
    in_dir = os.path.join(loadpath_tgc3d, split)
    files = sorted(f for f in os.listdir(in_dir) if f.endswith('.h5') or f.endswith('.hdf5'))
    chunks = []
    for f in files:
        with h5py.File(os.path.join(in_dir, f), 'r') as h5f:
            # each tgc3d file holds one or more sims along axis=0 of magnetic_field
            n = h5f['t1_fields/velocity'].shape[0]
        chunks.append(n)
    return files, chunks

train_files, train_chunks = get_files_and_chunks('train')
val_files,   val_chunks   = get_files_and_chunks('val')
test_files,  test_chunks  = get_files_and_chunks('test')

for split, norm_data, files, chunks in [
    ('train', train_norm, train_files, train_chunks),
    ('val',   val_norm,   val_files,   val_chunks),
    ('test',  test_norm,  test_files,  test_chunks)]:
    
    out_dir = os.path.join(savepath_norm_data_tgc3d, split)
    ptr = 0
    for fname, sz in zip(files, chunks):
        # grab exactly as many *simulations* as the original file had
        chunk = norm_data[ptr:ptr + sz]    # shape (sz, T, C, F, D, H, W)
        ptr += sz

        # transpose back to (sz, T, D, H, W, C, F)
        chunk_out = chunk.transpose(0,1,4,5,6,3,2)

        out_path = os.path.join(out_dir, fname)
        with h5py.File(out_path, 'w') as f_out:
            # --- t1_fields group ---
            g1 = f_out.create_group('t1_fields')
            # field 0 = magnetic_field, field 1 = velocity
            # keep any time‐axis if T>1
            velocity = chunk_out[..., :, 0]  
            print(f"[Save] Shape of V: {velocity.shape}")
            
            g1.create_dataset('velocity',data=velocity,compression='lzf')

            # --- t0_fields group ---
            g0 = f_out.create_group('t0_fields')
            # field 2 = density; drop the redundant channels axis to match original (one channel)
            density = chunk_out[..., 0, 1]
            #pressure = chunk_out[..., 0, 2]
            temperature = chunk_out[..., 0, 2]
            print(f"[Save] Shape of Rho: {density.shape}, T: {temperature.shape}")
            
            g0.create_dataset('density',data=density,compression='lzf')
            g0.create_dataset('temperature',data=temperature,compression='lzf')

        print(f"[tgc3d] Saved file: {fname}, chunks={sz}, shape(s)=V:{velocity.shape}," 
              f" Rho: {density.shape}, T: {temperature.shape}")

print("Normalized tgc3d data saved under:", savepath_norm_data_tgc3d)

#%% FNS-KF (PDEGym)
from src.utils.dataloaders.dataloader_fns_kf_2d import split_and_save_h5, FNSKF2dDataLoader

# first split the raw DR data into train/test/val. raw_h5_loadpath and data_path are the load path and save path
split_and_save_h5(raw_h5_loadpath = loadpath_fns_kf_2d, 
                  savepath = loadpath_fns_kf_2d, 
                  dataset_name='FNS-KF', 
                  train_frac = 0.8,
                  rand = True)

# load the splited raw DR data
loader = FNSKF2dDataLoader(loadpath_fns_kf_2d)
train, val = loader.split_train()          # data is already inflated
test = loader.split_test()
dataset = np.concatenate((train,val,test), axis = 0)
print("Shape of inflated FNS-KF data", dataset.shape)     

# Reshape & expand dims for RevIN (N, T, D, H, W, C, F) -> (N, T, F, C, D, H, W)
dataset_tr = dataset.transpose(0, 1, 6, 5, 2, 3, 4)
print("Transposed FNS-KF data", dataset_tr.shape)

# compute & normalize
rev_fns_kf_2d.compute_stats(dataset_tr, prefix='stats_fns_kf_2d')
dataset_fns_norm = rev_fns_kf_2d.normalize(dataset_tr, prefix='stats_fns_kf_2d')
print("Normalize dataset shape", dataset_fns_norm.shape)

# Check for DR dataset
# Check round‐trip via denormalize
tol_2 = 1e-5
recovered = rev_fns_kf_2d.denormalize(dataset_fns_norm, prefix='stats_fns_kf_2d')
print("Denormalized dataset shape", recovered.shape)

max_error = 0.0
for i in range(recovered.shape[0]):
    maxerror_i = np.max(np.abs(recovered[i] - dataset_tr[i]))  # saving some memory
    max_error = max(maxerror_i, max_error)
assert max_error < tol_2, "Denormalization did not perfectly recover original!"
print("FNS-KF RevIN round-trip OK")
del recovered

# --- Save the data in the same format as raw (N,T,H,W,F) ---
dataset_sq   = dataset_fns_norm.transpose(0, 1, 4, 5, 6, 3, 2)   # (N, T, D, H, W, C, F)
dataset_sq = np.squeeze(dataset_sq, axis = 2)[:,:,:,:,:,0]       # (N, T, H, W, C)
dataset_sq = dataset_sq.transpose(0,1,4,2,3)                     # (N, C, T, H, W)
print("Normed dataset in raw shape", dataset_sq.shape)

# save single HDF5 with same filename
raw_files = [f for f in os.listdir(loadpath_fns_kf_2d) if f.endswith('.h5') or f.endswith('.hdf5')]  
filename = raw_files[0] # get the name of the file
out_path = os.path.join(savepath_norm_data_fns_kf_2d, filename)
with h5py.File(out_path, 'w') as f_out:
    f_out.create_dataset('solution', data=dataset_sq, compression='lzf')
print("Saved normalized FNS-KF to", out_path)

# split the normed DR data into train/test/val
split_and_save_h5(savepath_norm_data_fns_kf_2d, savepath_norm_data_fns_kf_2d,
                  dataset_name = 'FNS-KF',
                  train_frac = 0.8,
                  rand = False)

#%% CE-CRP, KH, RP, Gauss (PDEGym)
from src.utils.dataloaders.dataloader_ce_2d import split_and_save_h5, CE2dDataLoader

# collect all loadpaths and savepaths for CE
loadpaths_ce = [loadpath_ce_crp_2d, loadpath_ce_kh_2d, loadpath_ce_rp_2d, 
                loadpath_ce_gauss_2d]
savepaths_ce = [savepath_norm_data_ce_crp_2d, savepath_norm_data_ce_kh_2d, 
                savepath_norm_data_ce_rp_2d, savepath_norm_data_ce_gauss_2d]
datasetnames_ce = ['CE-CRP', 'CE-KH', 'CE-RP', 'CE-Gauss']
stats_ce = ['stats_ce_crp_2d', 'stats_ce_kh_2d', 'stats_ce_rp_2d', 'stats_ce_gauss_2d']

# loadpath and savepath for CE-CRP, CE-RP, CE-KH, CE-Gauss
select_dataset = 3
loadpath_ce = loadpaths_ce[select_dataset]
savepath_ce = savepaths_ce[select_dataset]
dsname_ce = datasetnames_ce[select_dataset]
stat_ce = stats_ce[select_dataset]
print(f'Working on {dsname_ce} dataset ...')

# first split the raw DR data into train/test/val. raw_h5_loadpath and data_path are the load path and save path
split_and_save_h5(raw_h5_loadpath = loadpath_ce, 
                  savepath = loadpath_ce, 
                  dataset_name = dsname_ce,
                  train_frac = 0.8,
                  rand = True)

# load the splited raw DR data
loader = CE2dDataLoader(loadpath_ce)
train, val = loader.split_train()          # data is already inflated
test = loader.split_test()
dataset = np.concatenate((train,val,test), axis = 0)
print("Shape of inflated data", dataset.shape)

# Reshape & expand dims for RevIN (N, T, D, H, W, C, F) -> (N, T, F, C, D, H, W)
dataset_tr = dataset.transpose(0, 1, 6, 5, 2, 3, 4)
print("Transposed FNS-KF data", dataset_tr.shape)

# compute & normalize
rev_ce_2d = RevIN(savepath_muvar)
rev_ce_2d.compute_stats(dataset_tr, prefix = stat_ce)
dataset_norm = rev_ce_2d.normalize(dataset_tr, prefix = stat_ce)
print("Normalize dataset shape", dataset_norm.shape)

# Check for DR dataset
# Check round‐trip via denormalize
tol_2 = 1e-5
recovered = rev_ce_2d.denormalize(dataset_norm, prefix=stat_ce)
print("Denormalized dataset shape", recovered.shape)

max_error = 0.0
for i in range(recovered.shape[0]):
    maxerror_i = np.max(np.abs(recovered[i] - dataset_tr[i]))  # saving some memory
    max_error = max(maxerror_i, max_error)
assert max_error < tol_2, "Denormalization did not perfectly recover original!"
print("RevIN round-trip OK")
del recovered

# --- Save the data in the same format as raw (N,T,H,W,F) ---
dataset_sq   = dataset_norm.transpose(0, 1, 4, 5, 6, 3, 2)   # (N, T, D, H, W, C, F)
dataset_sq = np.squeeze(dataset_sq, axis = 2)                # (N, T, H, W, C, F)
den, vel, pre = dataset_sq[...,0], dataset_sq[...,1], dataset_sq[...,2]
vx, vy = vel[...,0:1], vel[...,1:2]
den = np.expand_dims(den[:,:,:,:,0], axis = -1)
pre = np.expand_dims(pre[:,:,:,:,0], axis = -1)
print(f'den:{den.shape}, vx:{vx.shape}, vy:{vy.shape} pre:{pre.shape}')
data = np.concatenate((den, vx, vy, pre), axis = -1)
dataset_sq = data.transpose(0,1,4,2,3)                 # (N, C, T, H, W)
print("Normed dataset in raw shape", dataset_sq.shape)

# save single HDF5 with same filename
raw_files = [f for f in os.listdir(loadpath_ce) if f.endswith('.h5') or f.endswith('.hdf5')] 
splits= len(raw_files)
N = dataset_sq.shape[0]
chunk_size = N // splits
for i, fname in enumerate(raw_files):
    start = i * chunk_size
    end = (i + 1) * chunk_size if i != splits - 1 else N  # fill the rest in the last
    chunk = dataset_sq[start:end]
    print(f'Chunk shape: {chunk.shape}')
    out_path = os.path.join(savepath_ce, fname) 
    with h5py.File(out_path, 'w') as f_out:
        f_out.create_dataset('data', data=chunk, compression='lzf')
        print("Saved normalized data to", out_path)

# split the normed DR data into train/test/val
split_and_save_h5(savepath_ce, savepath_ce,
                  dataset_name = dsname_ce,
                  train_frac = 0.8,
                  rand = False)

#%% NS-Sines,Gauss (PDEGym)
from src.utils.dataloaders.dataloader_ns_2d import split_and_save_h5, NS2dDataLoader

# collect all loadpaths and savepaths for CE
loadpaths_ns = [loadpath_ns_sines_2d,  loadpath_ns_gauss_2d]
savepaths_ns = [savepath_norm_data_ns_sines_2d, savepath_norm_data_ns_gauss_2d]
datasetnames_ns = ['NS-Sines', 'NS-Gauss']
stats_ns = ['stats_ns_sines_2d', 'stats_ns_gauss_2d']

# loadpath and savepath for CE-CRP, CE-RP, CE-KH, CE-Gauss
select_dataset = 1
loadpath_ns = loadpaths_ns[select_dataset]
savepath_ns = savepaths_ns[select_dataset]
dsname_ns = datasetnames_ns[select_dataset]
stat_ns = stats_ns[select_dataset]
print(f'Working on {dsname_ns} dataset ...')

# first split the raw DR data into train/test/val. raw_h5_loadpath and data_path are the load path and save path
split_and_save_h5(raw_h5_loadpath = loadpath_ns, 
                  savepath = loadpath_ns, 
                  dataset_name = dsname_ns, 
                  train_frac = 0.8,
                  rand = True)

# load the splited raw DR data
loader = NS2dDataLoader(loadpath_ns)
train, val = loader.split_train()          # data is already inflated
test = loader.split_test()
dataset = np.concatenate((train,val,test), axis = 0)
print("Shape of inflated data", dataset.shape)     

# Reshape & expand dims for RevIN (N, T, D, H, W, C, F) -> (N, T, F, C, D, H, W)
dataset_tr = dataset.transpose(0, 1, 6, 5, 2, 3, 4)
print("Transposed data", dataset_tr.shape)

# compute & normalize
rev_ns_2d = RevIN(savepath_muvar)
rev_ns_2d.compute_stats(dataset_tr, prefix = stat_ns)
dataset_norm = rev_ns_2d.normalize(dataset_tr, prefix = stat_ns)
print("Normalize dataset shape", dataset_norm.shape)

# Check for DR dataset
# Check round‐trip via denormalize
tol_2 = 1e-5
recovered = rev_ns_2d.denormalize(dataset_norm, prefix=stat_ns)
print("Denormalized dataset shape", recovered.shape)
max_error = 0.0
for i in range(recovered.shape[0]):
    maxerror_i = np.max(np.abs(recovered[i] - dataset_tr[i]))  # saving some memory
    max_error = max(maxerror_i, max_error)
assert max_error < tol_2, "Denormalization did not perfectly recover original!"
print("RevIN round-trip OK")
del recovered

# --- Save the data in the same format as raw (N,T,H,W,F) ---
dataset_sq   = dataset_norm.transpose(0, 1, 4, 5, 6, 3, 2)   # (N, T, D, H, W, C, F)
dataset_sq = np.squeeze(dataset_sq, axis = (2,6))            # (N, T, H, W, C, F)
dataset_sq = dataset_sq.transpose(0,1,4,2,3)                 # (N, C, T, H, W)
print("Normed dataset in raw shape", dataset_sq.shape)

# save single HDF5 with same filename
raw_files = [f for f in os.listdir(loadpath_ns) if f.endswith('.h5') or f.endswith('.hdf5')] 
splits= len(raw_files)
N = dataset_sq.shape[0]
chunk_size = N // splits
for i, fname in enumerate(raw_files):
    start = i * chunk_size
    end = (i + 1) * chunk_size if i != splits - 1 else N  # fill the rest in the last
    chunk = dataset_sq[start:end]
    print(f'Chunk shape: {chunk.shape}')
    out_path = os.path.join(savepath_ns, fname) 
    with h5py.File(out_path, 'w') as f_out:
        f_out.create_dataset('velocity', data=chunk, compression='lzf')
        print("Saved normalized data to", out_path)

# split the normed DR data into train/test/val
split_and_save_h5(savepath_ns, savepath_ns,
                  dataset_name = dsname_ns,
                  train_frac = 0.8,
                  rand = False)