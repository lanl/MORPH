import os
import sys
import numpy as np
import h5py

# Add project root to path
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from src.utils.data_plotter import DataPlotter
from src.utils.explore_hdf5 import ExploreHDF5Structure
from config.data_config_vis import DataConfig

# raw data directory
dataset_dir = "D:/data"

# instantiate the class
cfg = DataConfig(dataset_dir=dataset_dir, project_root=project_root, split = 'train')

# load all the filepaths
# ---pretrained---
file_path_dr2d  = cfg['DR2d_data_pdebench']['file_path_dr2d']
file_path_mhd3d  = cfg['MHD3d_data_thewell']['file_path_mhd3d']
file_path_cfd1d  = cfg['1dcfd_pdebench']['file_path_cfd1d']
file_path_cfd2d_ic  = cfg['2dcfd_ic_pdebench']['file_path_cfd2d_ic']
file_path_cfd3d  = cfg['3dcfd_pdebench']['file_path_cfd3d']
file_path_sw2d  = cfg['2dSW_pdebench']['file_path_sw2d']
# ---finetuned---
file_path_dr1d  = cfg['1ddr_pdebench']['file_path_dr1d']
file_path_cfd2d  = cfg['2dcfd_pdebench']['file_path_cfd2d']
file_path_be1d = cfg['1dbe_pdebench']['file_path_be1d']
file_path_cfd3d_turb = cfg['3dcfd_turb_pdebench']['file_path_3dcfd_turb']
file_path_gsdr2d = cfg['2dgrayscottdr_thewell']['file_path_2dgsdr']
file_path_tgc3d = cfg['3dturbgravitycool_thewell']['file_path_3dtgc']
file_path_fns_kf_2d = cfg['2dFNS_KF_pdegym']['file_path_2dfns_kf']

# -- new pretraining sets ---
file_path_ce_crp_2d = cfg['2dCE_CRP_pdegym']['file_path_2dce_crp']
file_path_ce_kh_2d = cfg['2dCE_KH_pdegym']['file_path_2dce_kh']
file_path_ce_rp_2d = cfg['2dCE_RP_pdegym']['file_path_2dce_rp']
file_path_ce_gauss_2d = cfg['2dCE_Gauss_pdegym']['file_path_2dce_gauss']
file_path_ns_sines_2d = cfg['2dNS_Sines_pdegym']['file_path_2dns_sines']
file_path_ns_gauss_2d = cfg['2dNS_Gauss_pdegym']['file_path_2dns_gauss']

# instantiate the data plotter
datasetplotter = DataPlotter(os.path.join(project_root, 'data'))

#%% Import the DR2D dataset
files = [f for f in os.listdir(file_path_dr2d)
         if f.endswith('.h5') or f.endswith('.hdf5')]
if not files:
    raise FileNotFoundError(f"No HDF5 files found in {files}")

print(f'Number of files in {file_path_dr2d}: {len(files)}')

# explore the structure of file 
# explorer = ExploreHDF5Structure()
# explorer.explore_hdf5(os.path.join(file_path_dr2d, files[0]))  
    
# Open the HDF5 file
n_files = 1
print('Loading diffusion-reaction (2D) data from PDEBench...')
for fname in files[:n_files]:
    path = os.path.join(file_path_dr2d, fname)
    with h5py.File(path, 'r') as f5:
        
        # Get all the sample keys (sorted)
        sample_keys = sorted(list(f5.keys()))
        num_samples = len(sample_keys)
        
        # Get the shape from the first sample
        first_sample = f5[sample_keys[0]]['data']
        sample_shape = first_sample.shape
        
        data = np.zeros((num_samples, *sample_shape), dtype=np.float32)
        # Load data from each sample
        for i, key in enumerate(sample_keys):
            data[i] = f5[key]['data'][...]  
    print(f"Shape of the imported data: {data.shape}")
    
    # plot the data
    datasetplotter.plot_sample_dr2d(data, start_t_idx=0, num_timesteps=5, 
                                    dataset_name = 'PBD-DR2D')

#%% Import the MHD3D dataset
files = [f for f in os.listdir(file_path_mhd3d)
         if f.endswith('.h5') or f.endswith('.hdf5')]
if not files:
    raise FileNotFoundError(f"No HDF5 files found in {file_path_mhd3d}")

print(f'Number of files in {file_path_mhd3d}: {len(files)}')

print('Loading MHD (3D, 64x64x64) data from The Well...')
n_files = 1
for fname in files[:n_files]:
    arrays = []
    path = os.path.join(file_path_mhd3d, fname)
    with h5py.File(path, 'r') as f5:
        # read fields
        magnetic_field = f5['t1_fields/magnetic_field'][:]  # (..., D,H,W,3)
        velocity       = f5['t1_fields/velocity'][:]        # same shape
        density        = f5['t0_fields/density'][:]         # (..., D,H,W)
        # expand density to 3 channels
        density = np.expand_dims(density, axis=-1)  # (..., D,H,W,1)
        density = np.tile(density, (1,)* (density.ndim-1) + (3,))  # (..., D,H,W,3)
        # stack into [..., D,H,W, channels*3]
        arr = np.stack([magnetic_field, velocity, density], axis=-1)  
        # resulting shape: (N, ..., 3 channels, 3 fields) if needed adjust axes
    arrays.append(arr.astype('float32'))
    
    arrays = np.concatenate(arrays, axis=0)
    selected_data = arrays[:,:,0,:,:,0,:] # d-slice & x-component of all fields
    print("Shape of the plotted data", selected_data.shape)
    datasetplotter.plot_sample_mhd3d(selected_data, start_t_idx = 0, num_timesteps=5,
                                     dataset_name = 'TW-MHD3D')

#%% Import the CFD1d data
files = [f for f in os.listdir(file_path_cfd1d)
         if f.endswith('.h5') or f.endswith('.hdf5')]

if not files:
    raise FileNotFoundError(f"No HDF5 files found in {files}")

print(f'Number of files in {file_path_cfd1d}: {len(files)}')

# explore the structure of file 
explorer = ExploreHDF5Structure()
explorer.explore_hdf5(os.path.join(file_path_cfd1d, files[0]))   

# Open the HDF5 file
print('Loading CFD-1D (compressible) data from PDEBench...')
n_files = 1
for fname in files[:n_files]:
    path = os.path.join(file_path_cfd1d, fname)
    with h5py.File(path, 'r') as f5:
        # Directly read your three main fields
        vx       = f5['Vx'][...]        # shape (10000, 101, 1024)
        density  = f5['density'][...]   # shape (10000, 101, 1024)
        pressure = f5['pressure'][...]  # shape (10000, 101, 1024)

        # Stack the fields in the last dim: resulting shape: (10000, 101, 1024, 3)
        data = np.stack([vx, density, pressure], axis=-1).astype(np.float32)
    print(f"Shape of the data: {data.shape}")
    
    # plot the data
    datasetplotter.plot_sample_cfd1d(data, start_t_idx=0, num_timesteps=5,
                                     dataset_name = 'PBD-CFD1D')
    
#%% Import the CFD2d (incompressible) data
files = [f for f in os.listdir(file_path_cfd2d_ic)
         if f.endswith('.h5') or f.endswith('.hdf5')]
if not files:
    raise FileNotFoundError(f"No HDF5 files found in {files}")
    
print(f'Number of files in {file_path_cfd2d_ic}: {len(files)}')

# explore the structure of file 
# explorer = ExploreHDF5Structure()
# explorer.explore_hdf5(os.path.join(file_path_cfd2d_ic, files[0]))   

# Open the HDF5 file
print('Loading CFD-2D (incompressible) data from PDEBench...')
n_files = 1
for fname in files[:n_files]:
    arrays = []
    path = os.path.join(file_path_cfd2d_ic, fname)
    with h5py.File(path, 'r') as f5:
        # Directly read your three main fields
        force    = f5['force'][...]
        vel      = f5['velocity'][...]  
        print(f'Raw fields: {force.shape},{vel.shape}')
        
        # expand the fields
        force =  np.expand_dims(force, axis = (1,5))  # time and field
        force = np.repeat(force, repeats = 1000, axis = 1)  # repeat time
        vel   =  np.expand_dims(vel, axis = 5)
        print(f'Expanded force and vel: {force.shape}, {vel.shape}')
        
        # Stack the fields in the last dim
        data = np.concatenate((force, vel), axis = 5).astype(np.float32)
        print("Shape of the data", data.shape)
    arrays.append(data)

    # plot the data
    arrays = np.concatenate(arrays, axis=0)
    selected_data = arrays[:,:,:,:,0,:]                   # x-component
    print("Shape of the plotted data", selected_data.shape)
    
    datasetplotter.plot_sample_cfd2d_ic(selected_data, start_t_idx=0, num_timesteps=5,
                                     dataset_name = 'PBD-CFD2D(IC)')

#%% Import the CFD3d (compressible) data
files = [f for f in os.listdir(file_path_cfd3d)
         if f.endswith('.h5') or f.endswith('.hdf5')]

if not files:
    raise FileNotFoundError(f"No HDF5 files found in {files}")

print(f'Number of files in {file_path_cfd3d}: {len(files)}')

# explore the structure of file 
# explorer = ExploreHDF5Structure()
# explorer.explore_hdf5(os.path.join(file_path_cfd3d, files[0]))   

# Open the HDF5 file
print('Loading CFD-3D (compressible) data from PDEBench...')
# pick the file you want (0-based index)
select_file = 0
fname = files[select_file]

# full path
path = os.path.join(file_path_cfd3d, fname)
print(f"Loaded file: {path}")

with h5py.File(path, 'r') as f5:
    # Directly read your three main fields
    vx       = f5['Vx'][...]        
    vy       = f5['Vy'][...]        
    vz       = f5['Vz'][...]        
    density  = f5['density'][...]   
    pressure = f5['pressure'][...]  
    
    # expand all the fields
    vx =  np.expand_dims(vx, axis = (5,6))                # (100,21,128,128,128,1,1)
    vy =  np.expand_dims(vy, axis = (5,6))                # (100,21,128,128,128,1,1)
    vz =  np.expand_dims(vz, axis = (5,6))                # (100,21,128,128,128,1,1)
    density = np.expand_dims(density, axis = (5,6))       # (100,21,128,128,128,1,1)
    pressure = np.expand_dims(pressure, axis = (5,6))     # (100,21,128,128,128,1,1)
    
    # make vector fields
    v = np.concatenate((vx, vy, vz), axis = 5)            # (100,21,128,128,128,3,1)
    density = np.repeat(density, repeats = 3, axis = 5)   # (100,21,128,128,128,3,1)
    pressure = np.repeat(pressure, repeats = 3, axis = 5) # (100,21,128,128,128,3,1)
    
    # Stack the fields in the last dim
    data = np.concatenate((v, density, pressure), axis = 6).astype(np.float32) # (100,21,128,128,128,3,3)
    print("Shape of the data", data.shape)
    
selected_data = data[:,:,0,:,:,0,:]                   # x-component and z-slice
print("Shape of the plotted data", selected_data.shape)
    
# plot the data
datasetplotter.plot_sample_cfd3d(selected_data, start_t_idx=0, num_timesteps=5,
                                 dataset_name = 'PBD-CFD3D')

del v, density, pressure, vx, vy, vz, arrays

#%% Import the SW data
files = [f for f in os.listdir(file_path_sw2d)
         if f.endswith('.h5') or f.endswith('.hdf5')]

if not files:
    raise FileNotFoundError(f"No HDF5 files found in {files}")

print(f'Number of files in {file_path_sw2d}: {len(files)}')

# explore the structure of file 
# explorer = ExploreHDF5Structure()
# explorer.explore_hdf5(os.path.join(file_path_sw2d, files[0]))
print('Loading SW data from PDEBench...')
n_files = 1
for fname in files[:n_files]:
    path = os.path.join(file_path_sw2d, fname)
    with h5py.File(path, 'r') as f5:
        
        # Get all the sample keys (sorted)
        sample_keys = sorted(list(f5.keys()))
        num_samples = len(sample_keys)
        print(f"num_samples: {num_samples}")
        
        # Get the shape from the first sample
        first_sample = f5[sample_keys[0]]['data']
        sample_shape = first_sample.shape
        
        data = np.zeros((num_samples, *sample_shape), dtype=np.float32)
        # Load data from each sample
        for i, key in enumerate(sample_keys):
            data[i] = f5[key]['data'][...]
    
    print(f"Shape of the imported data: {data.shape}")
    
    # plot the data
    datasetplotter.plot_sample_sw2d(data, start_t_idx = 0, num_timesteps=10,
                                     dataset_name = 'PBD-SW2D')
    
#%% Import the DR1d data
files = [f for f in os.listdir(file_path_dr1d)
         if f.endswith('.h5') or f.endswith('.hdf5')]

if not files:
    raise FileNotFoundError(f"No HDF5 files found in {files}")

print(f'Number of files in {file_path_sw2d}: {len(files)}')

# explore the structure of file 
# explorer = ExploreHDF5Structure()
# explorer.explore_hdf5(os.path.join(file_path_dr1d, files[0]))

# Open the HDF5 file
print('Loading diffusion-reaction (1D) data from PDEBench...')
n_files = 1
all_tensors = []
for fname in files[:n_files]:
    path = os.path.join(file_path_dr1d, fname)
    print(f"Loaded file: {path}")
    with h5py.File(path, 'r') as f5:
        data = f5['tensor'][...]  # (10000, 101, 1024)
        all_tensors.append(data)
        
# plot the data
print(f"Shape of the imported data: {data.shape}")
datasetplotter.plot_sample_dr1d(data, start_t_idx=0, num_timesteps=5,
                                 dataset_name = 'PBD-DR1D')

#%% Import the CFD2d (compressible) data
files = [f for f in os.listdir(file_path_cfd2d)
         if f.endswith('.h5') or f.endswith('.hdf5')]

if not files:
    raise FileNotFoundError(f"No HDF5 files found in {file_path_cfd2d}")

print(f'Number of files in {file_path_cfd2d}: {len(files)}')

# explore the structure of file 
# explorer = ExploreHDF5Structure()
# explorer.explore_hdf5(os.path.join(file_path_cfd2d, files[0]))

# pick the file you want (0-based index)
select_file = 0
fname = files[select_file]
print(f"Selected file: {fname}")

# full path
path = os.path.join(file_path_cfd2d, fname)
print(f"Loaded file: {path}")

print('Loading CFD2D (compressible) data from PDEBench...')
with h5py.File(path, 'r') as f5:
    vx       = f5['Vx'][...]        
    vy       = f5['Vy'][...]        
    density  = f5['density'][...]   
    pressure = f5['pressure'][...]  
    
    print(f'{vx.shape},{vy.shape},{density.shape},{pressure.shape}')
    # expand all the fields
    vx =  np.expand_dims(vx, axis=(4,5))                
    vy =  np.expand_dims(vy, axis=(4,5))                
    density = np.expand_dims(density, axis=(4,5))       
    pressure = np.expand_dims(pressure, axis=(4,5))
    
    print(f'{vx.shape},{vy.shape},{density.shape},{pressure.shape}')
    # make vector fields
    v = np.concatenate((vx, vy), axis=4)                    # (10000,21,128,128,2,1)
    density = np.repeat(density, repeats=2, axis=4)         # (10000,21,128,128,2,1)
    pressure = np.repeat(pressure, repeats=2, axis=4)       # (10000,21,128,128,2,1)
    
    print(f'{v.shape},{density.shape},{pressure.shape}')
    # stack the fields
    data = np.concatenate((v, density, pressure), axis=5).astype(np.float32) 
    print("Shape of the data", data.shape)

selected_data = data[:,:,:,:,0,:]   # x-component
print("Shape of the plotted data", selected_data.shape)

# plot
datasetplotter.plot_sample_cfd2d(selected_data, start_t_idx=0, num_timesteps=5,
                                 dataset_name = 'PBD-CFD2D')

#%% Import the Burgers-1D data
files = [f for f in os.listdir(file_path_be1d)
         if f.endswith('.h5') or f.endswith('.hdf5')]

if not files:
    raise FileNotFoundError(f"No HDF5 files found in {files}")

print(f'Number of files in {file_path_be1d}: {len(files)}')

# explore the structure of file 
# explorer = ExploreHDF5Structure()
# explorer.explore_hdf5(os.path.join(file_path_be1d, files[0]))

# Open the HDF5 file
print('Loading Burgers (1D) data from PDEBench...')
n_files = 1
all_tensors = []
for fname in files[:n_files]:
    path = os.path.join(file_path_be1d, fname)
    print(f"Loaded file: {path}")
    with h5py.File(path, 'r') as f5:
        data = f5['tensor'][...]  # (10000, 201, 1024)
        all_tensors.append(data)
        
# plot the data (Use the DRplotter since shape is similar)
print(f"Shape of the imported data: {data.shape}")
datasetplotter.plot_sample_dr1d(data, start_t_idx=0, num_timesteps=5, 
                                dataset_name = 'PBD-BE1D')

#%% Import the CFD-3D(Turb) data
files = [f for f in os.listdir(file_path_cfd3d_turb)
         if f.endswith('.h5') or f.endswith('.hdf5')]

if not files:
    raise FileNotFoundError(f"No HDF5 files found in {files}")

print(f'Number of files in {file_path_cfd3d_turb}: {len(files)}')

# explore the structure of file 
# explorer = ExploreHDF5Structure()
# explorer.explore_hdf5(os.path.join(file_path_cfd3d_rand, files[0]))   

# Open the HDF5 file
print('Loading CFD-3D (compressible, turbulent) data from PDEBench...')
# pick the file you want (0-based index)
select_file = 0
fname = files[select_file]

# full path
path = os.path.join(file_path_cfd3d_turb, fname)
print(f"Loaded file: {path}")

with h5py.File(path, 'r') as f5:
    # Directly read your three main fields
    vx       = f5['Vx'][...]        
    vy       = f5['Vy'][...]        
    vz       = f5['Vz'][...]        
    density  = f5['density'][...]   
    pressure = f5['pressure'][...]
    
    # expand all the fields
    vx =  np.expand_dims(vx, axis = (5,6))                # (600,21,64,64,64,1,1)
    vy =  np.expand_dims(vy, axis = (5,6))                # (600,21,64,64,64,1,1)
    vz =  np.expand_dims(vz, axis = (5,6))                # (600,21,64,64,64,1,1)
    density = np.expand_dims(density, axis = (5,6))       # (600,21,64,64,64,1,1)
    pressure = np.expand_dims(pressure, axis = (5,6))     # (600,21,64,64,64,1,1)
    
    # make vector fields
    v = np.concatenate((vx, vy, vz), axis = 5)            # (600,21,64,64,64,3,1)
    del vx,vy,vz
    density = np.repeat(density, repeats = 3, axis = 5)   # (600,21,64,64,64,3,1)
    pressure = np.repeat(pressure, repeats = 3, axis = 5) # (600,21,64,64,64,3,1)
    
    # Stack the fields in the last dim
    data = np.concatenate((v, density, pressure), axis = 6).astype(np.float32) # (600,21,64,64,64,3,3)
    print("Shape of the data", data.shape)
    
selected_data = data[:,:,0,:,:,0,:]                   # x-component and z-slice
print("Shape of the plotted data", selected_data.shape)
    
# plot the data
datasetplotter.plot_sample_cfd3d(selected_data, start_t_idx=0, num_timesteps=5, 
                                dataset_name = 'PBD-CFD3D(Turb)')

#del v, density, pressure, vx, vy, vz

#%% Import the GrayScottDiffReact-2D data
files = [f for f in os.listdir(file_path_gsdr2d)
         if f.endswith('.h5') or f.endswith('.hdf5')]
if not files:
    raise FileNotFoundError(f"No HDF5 files found in {file_path_gsdr2d}")

print(f'Number of files in {file_path_gsdr2d}: {len(files)}')

# explore the structure of file 
# explorer = ExploreHDF5Structure()
# explorer.explore_hdf5(os.path.join(split_path, files[0]))   

print('Loading GSDR (2D) data from The Well...')
n_files = 1
for fname in files[:n_files]:
    arrays = []
    path = os.path.join(file_path_gsdr2d, fname)
    with h5py.File(path, 'r') as f5:
        # read fields
        act_A = f5['t0_fields/A'][:]  # (160,1001,128,128)
        act_B = f5['t0_fields/B'][:]  # (160,1001,128,128)
        
        print(f'A:{act_A.shape}, B:{act_B.shape}')
        
        # expand density to 3 channels
        act_A = np.expand_dims(act_A, axis=-1)  
        act_B = np.expand_dims(act_B, axis=-1)  
        
        print(f'A:{act_A.shape}, B:{act_B.shape}')
        
        # concatenate into (N,T,H,W,F)
        data = np.concatenate((act_A, act_B), axis=4)
        
print("Shape of the plotted data", data.shape)
datasetplotter.plot_sample_dr2d(data, start_t_idx=0, num_timesteps=5,
                                dataset_name = 'TW-GSDR2D')

#%% Import the TGC data
files = [f for f in os.listdir(file_path_tgc3d)
         if f.endswith('.h5') or f.endswith('.hdf5')]
if not files:
    raise FileNotFoundError(f"No HDF5 files found in {file_path_tgc3d}")

print(f'Number of files in {file_path_tgc3d}: {len(files)}')

# explore the structure of file 
# explorer = ExploreHDF5Structure()
# explorer.explore_hdf5(os.path.join(file_path_tgc3d, files[0]))   

print('Loading TGC (3D, 64x64x64) data from The Well...')
select_file = 1
for fname in files:
    arrays = []
    path = os.path.join(file_path_tgc3d, fname)
    with h5py.File(path, 'r') as f5:
        # read fields
        dens = f5['t0_fields/density'][:]        # (80, 50, 64, 64, 64)
        press = f5['t0_fields/pressure'][:]      # (80, 50, 64, 64, 64)
        temp = f5['t0_fields/temperature'][:]    # (80, 50, 64, 64, 64)
        vel = f5['t1_fields/velocity'][:]        # (80, 50, 64, 64, 64, 3)
        
        print(f'dens:{dens.shape}, press:{press.shape}'
              f'temp:{temp.shape}, vel:{vel.shape}')
        
        dens = np.expand_dims(dens, axis=(5,6))   # (80, 50, 64, 64, 64, 1, 1)
        press = np.expand_dims(press, axis=(5,6)) # (80, 50, 64, 64, 64, 1, 1)
        temp = np.expand_dims(temp, axis=(5,6))   # (80, 50, 64, 64, 64, 1, 1)
        vel = np.expand_dims(vel, axis=6)         # (80, 50, 64, 64, 64, 3, 1)
        
        print(f'dens:{dens.shape}, press:{press.shape}'
              f'temp:{temp.shape}, vel:{vel.shape}')
        
        dens = np.repeat(dens, repeats = 3, axis = 5)   # (80, 50, 64, 64, 64, 3, 1)
        press = np.repeat(press, repeats = 3, axis = 5) # (80, 50, 64, 64, 64, 3, 1)
        temp = np.repeat(temp, repeats = 3, axis = 5)   # (80, 50, 64, 64, 64, 3, 1)
        print(f'dens:{dens.shape}, press:{press.shape}'
              f'temp:{temp.shape}, vel:{vel.shape}')
        
        data = np.concatenate((vel, dens, press, temp), axis = 6).astype(np.float32) # (80,50,64,64,64,3,4)
        print("Shape of the data", data.shape)
        
selected_data = data[:,:,0,:,:,0,:] # x-component and z-slice
print("Shape of the plotted data", selected_data.shape)
datasetplotter.plot_sample_tgc3d(selected_data, start_t_idx = 0, num_timesteps=5,
                                dataset_name = 'TW-TGC3D')

#%% Import the FNS-KF data
files = [f for f in os.listdir(file_path_fns_kf_2d)
         if f.endswith('.h5') or f.endswith('.hdf5')]
if not files:
    raise FileNotFoundError(f"No HDF5 files found in {file_path_fns_kf_2d}")

print(f'Number of files in {file_path_fns_kf_2d}: {len(files)}')

# explore the structure of file 
explorer = ExploreHDF5Structure()
explorer.explore_hdf5(os.path.join(file_path_fns_kf_2d, files[0]))   

print('Loading FNS-KF (2D, 128x128) data from PDEGYM...')
select_file = 1
for fname in files:
    arrays = []
    path = os.path.join(file_path_fns_kf_2d, fname)
    with h5py.File(path, 'r') as f5:
        # read fields
        vel = f5["solution"][:]
        print("Shape of the data", vel.shape)
vel = vel.transpose(0,1,3,4,2)
datasetplotter.plot_sample_dr2d(vel, start_t_idx=0, num_timesteps=5, 
                                dataset_name = 'PG-FNS-KF2D')

#%% Import PDEGym datasets -> CE (***NEW***)
file_path_pt_pdegym = [file_path_ce_crp_2d, file_path_ce_kh_2d,
                       file_path_ce_rp_2d, file_path_ce_gauss_2d]
select_dataset = 2
file_pdegym = file_path_pt_pdegym[select_dataset]
ds_names = ['PG-CE-CRP2D','PG-CE-KH2D','PG-CE-RP2D','PG-CE-Gauss2D']
files = [f for f in os.listdir(file_pdegym)
         if f.endswith('.h5') or f.endswith('.hdf5')]
if not files:
    raise FileNotFoundError(f"No HDF5 files found in {file_pdegym}")

print(f'Number of files in {file_pdegym}: {len(files)}')

# explore the structure of file 
explorer = ExploreHDF5Structure()
explorer.explore_hdf5(os.path.join(file_pdegym, files[0]))

print('Loading PDEGYM (2D, 128x128) data from...')
select_file = 1
for fname in files[:select_file]:
    arrays = []
    path = os.path.join(file_pdegym, fname)
    with h5py.File(path, 'r') as f5:
        # read fields
        data = f5["data"][:]
        print("Shape of the data", data.shape)
data = data.transpose(0,1,3,4,2) # (N,T,H,W,F)
print("Shape of the transposed data", data.shape)
datasetplotter.plot_sample_ce2d_pdegym(data, start_t_idx=0, num_timesteps=10, 
                                       dataset_name = ds_names[select_dataset])

#%% Import PDEGym datasets -> NS (***NEW***)
file_path_pt_pdegym = [file_path_ns_sines_2d,file_path_ns_gauss_2d]
ds_names = ['PG-NS-Sines2D','PG-NS-Gauss2D']
select_dataset = 1
file_pdegym = file_path_pt_pdegym[select_dataset]
files = [f for f in os.listdir(file_pdegym)
         if f.endswith('.h5') or f.endswith('.hdf5')]
if not files:
    raise FileNotFoundError(f"No HDF5 files found in {file_pdegym}")

print(f'Number of files in {file_pdegym}: {len(files)}')

# explore the structure of file 
explorer = ExploreHDF5Structure()
explorer.explore_hdf5(os.path.join(file_pdegym, files[0]))

print('Loading PDEGYM (2D, 128x128) data from...')
select_file = 1
for fname in files[:select_file]:
    arrays = []
    path = os.path.join(file_pdegym, fname)
    with h5py.File(path, 'r') as f5:
        # read fields
        data = f5["velocity"][:]
        print("Shape of the data", data.shape)
data = data.transpose(0,1,3,4,2) # (N,T,H,W,F)
datasetplotter.plot_sample_ns2d_pdegym(data, start_t_idx=0, num_timesteps=10, 
                                       dataset_name = ds_names[select_dataset])
