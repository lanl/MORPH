#%% visualize fracture data
import glob
import os
import sys
import numpy as np
import tarfile
import h5py
from natsort import natsorted
import matplotlib.pyplot as plt
import math
import random
import imageio.v2 as imageio
from matplotlib import cm

#%% Add project root to path
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..','..'))
sys.path.append(project_root)
print(f"Project root added to path: {project_root}")

# Import custom visualization utilities
from src.utils.explore_hdf5 import ExploreHDF5Structure
explore = ExploreHDF5Structure()

# extract .tar.gz files (# location of downloaded HF datasets)
data_loc = r'D:/data/material_fracturing/PHASE-FIELD/tungsten/'
sub_folders = ['combined_bc', 'horizontal_bc']

# processed data directory
process_data_loc = r'F:/FM/codes/morph_github/morph-cl/MORPH/datasets/normalized_revin/2dFRAC_tung_unnorm' # split into train/test/val manually later
os.makedirs(process_data_loc, exist_ok=True)  

#%% process all .tar.gz files in the sub-folders
def strip_tar_gz(path: str) -> str:
    """D:/.../foo.tar.gz -> D:/.../foo"""
    if path.endswith(".tar.gz"):
        return path[:-7]
    return os.path.splitext(path)[0]


for sub_folder in sub_folders:  # loop over sub-folders
    print(f"Processing sub-folder: {sub_folder}")

    zipped_files = natsorted(glob.glob(os.path.join(data_loc, sub_folder, "*.tar.gz")))
    print(f"Found {len(zipped_files)} zipped files in {os.path.join(data_loc, sub_folder)}")

    for zipped_file in zipped_files:  # loop over zipped files
        print(f"Processing zipped file: {zipped_file}")

        archive_path = zipped_file

        # specify the extracted folder name (remove .tar.gz properly)
        out_dir = strip_tar_gz(archive_path)  
        os.makedirs(out_dir, exist_ok=True)

        # skip if already extracted
        if os.path.isdir(out_dir) and os.listdir(out_dir):
            print("Skipping (already extracted):", out_dir)
        else:
            with tarfile.open(archive_path, "r:*") as tar:  
                tar.extractall(path=out_dir)
            print("Extracted to:", os.path.abspath(out_dir))

        # explore the content of the h5 file
        h5_files_list = natsorted(glob.glob(os.path.join(out_dir, "**", "*.h5"), recursive=True))
        print(f"Found {len(h5_files_list)} HDF5 files in {out_dir}:\n {h5_files_list[:10]}{' ...' if len(h5_files_list) > 10 else ''}")

        if not h5_files_list:
            print(f"No .h5 files found in {out_dir}, skipping.")
            continue

        model_append = []

        # get the total number of models to process (up to 1000)
        max_models = len(h5_files_list)

        for model in range(max_models):  # loop over models
            print(f"Exploring model: {model + 1} / {max_models}")

            # h5_files_list contains full paths now (from glob), so no os.path.join(out_dir, ...)
            h5_file_path = h5_files_list[model]
            print(f"Exploring HDF5 file: {h5_file_path}")

            # explore.explore_hdf5(h5_file_path) for first model only
            if model == 0:
                explore.explore_hdf5(h5_file_path)

            with h5py.File(h5_file_path, "r") as f:
                # 1) build ordered list of (label, key, sort_time)
                items = []

                # Find the keys in the file
                keys = list(f.keys())

                # Initial and final keys
                init_key = next((k for k in keys if k.endswith("fracture initial")), None)
                final_key = next((k for k in keys if k.endswith("fracture final")), None)

                # Add initial and final
                if init_key is not None:
                    items.append(("initial", init_key, -1.0))
                else:
                    raise ValueError(f"Didn't find initial fracture state for model {model} in {h5_file_path}")
                
                if final_key is not None:
                    items.append(("final", final_key, float("inf")))
                else:
                    raise ValueError(f"Didn't find final fracture state for model {model} in {h5_file_path}")
                
                # --- build (1, 2, 128, 128) for this model (initial, final) ---
                items.sort(key=lambda x: x[2])  # initial (-1) first, final (inf) last
                print('Length of items (MUST be 2):', len(items))

                if len(items) != 2:
                    raise ValueError(f"Missing initial/final for model {model} in {h5_file_path}")

                frames = [f[key][()] for (_label, key, _t) in items]   # two arrays (128,128)
                data = np.stack(frames, axis=0)[None, ...]            # (1,2,128,128)

                # quick sanity check
                if data.shape != (1, 2, 128, 128):
                    raise ValueError(f"model {model} has unexpected shape {data.shape} in {h5_file_path}")

                # append models
                model_append.append(data)

        # concatenate all models and save as .h5
        if model_append:
            all_data = np.concatenate(model_append, axis=0).astype(np.float32)  # (num_models, 2, 128, 128)

            save_dir = process_data_loc
            os.makedirs(save_dir, exist_ok=True)

            base_name = os.path.basename(out_dir)  # e.g., "test_xz_batch_000"
            save_path = os.path.join(save_dir, f"{base_name}_first_last.h5")

            # save as .h5 file
            with h5py.File(save_path, "w") as hf:
                hf.create_dataset("data", data=all_data)

            print(f"Saved concatenated data to: {save_path} with dataset name 'data' and shape {all_data.shape}")
        else:
            print(f"No valid models found in {out_dir} to save.")

# %% visualize a specific processed .h5 file
data_loc = r'D:/data/material_fracturing/processed_phase_field/tungsten/train'
list_h5_files = list(sorted([f for f in os.listdir(data_loc) if f.endswith('.h5')]))
print(f"Found {len(list_h5_files)} .h5 files in {data_loc}")

idx_rand_h5_file = random.randint(0, len(list_h5_files)-1)
print(f"Loading random .h5 file index: {idx_rand_h5_file}")

fname = list_h5_files[idx_rand_h5_file]
fpath = os.path.join(data_loc, fname)

#%% plot random trajectory from the selected .h5 file
with h5py.File(fpath, "r") as hf:
    data = hf["data"]  # (num_trajectories, 2, C, H, W)
    print("Data shape:", data.shape, "dtype:", data.dtype)

    # plot random trajectory
    traj_idx = random.randint(0, data.shape[0]-1)
    channel_idx = random.randint(0, data.shape[2]-1)

    img_0 = data[traj_idx, 0, :, :]
    img_99 = data[traj_idx, 1, :, :]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    im0 = ax[0].imshow(img_0, cmap='viridis', aspect='auto')
    fig.colorbar(im0, ax=ax[0])
    ax[0].set_title(f"Time step 0", fontsize=10)

    im1 = ax[1].imshow(img_99, cmap='viridis', aspect='auto')
    fig.colorbar(im1, ax=ax[1])
    ax[1].set_title(f"Time step 99", fontsize=10)
    
    fig.suptitle(
        f"h5 file = {idx_rand_h5_file} | traj = {traj_idx}",
        fontsize=12
    )
    plt.tight_layout()
    plt.show()

# %% create a video from the trajectory
gif_dir = os.path.join(r"D:\data\material_fracturing","gifs_t0_t99")
os.makedirs(gif_dir, exist_ok=True)

cmap = cm.get_cmap("viridis")
eps = 1e-12
stride = 1          # set >1 to skip trajectories (e.g., 5) to keep GIF smaller

def to_rgb(a, vmin, vmax):
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip((a - vmin) / (vmax - vmin + eps), 0, 1)
    return (cmap(x)[..., :3] * 255).astype(np.uint8)

use_h5_files_idx = list(np.random.randint(0, len(list_h5_files), 20))  # from above random selection
use_h5_files = [list_h5_files[i] for i in use_h5_files_idx]
print("Using .h5 files:", use_h5_files)
out_mp4 = os.path.join(gif_dir, f"mp4_bothBC_t1and100frames.mp4")

n = 0
with imageio.get_writer(out_mp4, fps=5, codec="libx264", quality=8, pixelformat="yuv420p") as w:
    for fname in use_h5_files:
        print("Processing:", fname)
        with h5py.File(os.path.join(data_loc, fname), "r") as hf:
            d = hf["data"]  # (B,2,H,W)
            for t in range(0, d.shape[0], stride):
                img0  = d[t, 0]
                img99 = d[t, 1]
                vmin = float(min(img0.min(), img99.min()))
                vmax = float(max(img0.max(), img99.max()))
                rgb0  = to_rgb(img0,  vmin, vmax)
                rgb99 = to_rgb(img99, vmin, vmax)
                sep = np.full((rgb0.shape[0], 8, 3), 255, np.uint8)  # 8px white bar
                frame = np.concatenate([rgb0, sep, rgb99], axis=1)
                w.append_data(frame)
                n += 1

print("saved:", out_mp4, "| frames:", n)

# %%
