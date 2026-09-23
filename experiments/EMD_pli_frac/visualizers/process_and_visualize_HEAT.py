#%%
import random
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import imageio.v2 as imageio
from matplotlib import cm

#%% Add project root to path
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..','..'))
sys.path.append(project_root)
print(f"Project root added to path: {project_root}")

# load the .npz file from directory
npz_folder_path = r'D:/data/HEAT/lsc240420_fp16_full_1and100frames/'
list_npz_files = list(sorted([f for f in os.listdir(npz_folder_path) if f.endswith('.npz')]))
print(f"Found {len(list_npz_files)} .npz files in {npz_folder_path}")

# %% loading data
def load_traj_step(traj_id: int, idx: int, folder=npz_folder_path):
    fname = f"lsc240420_id{traj_id:05d}_pvi_idx{idx:05d}.npz"
    fpath = os.path.join(folder, fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Missing: {fpath}")
    return np.load(fpath), fpath

# load trajectories, timestep 0 and 99
valid_trajs = []
ch_0_99_all = []
common_channels = None
for traj_no in range(1,5300+1):
    print("Loading trajectory ===>>>", traj_no)
    try:
        data_0, p0   = load_traj_step(traj_no, idx=0)
        data_99, p99 = load_traj_step(traj_no, idx=99)
    except FileNotFoundError as e:
        print(f"[SKIP] traj {traj_no}: {e}")
        continue  # skip entire trajectory if either file is missing
    
    valid_trajs.append(traj_no) # record valid trajectory number

    try:
        if traj_no == 1:
            print("Keys at time step 0:", data_0.files)
            for k in data_0.files:
                print(k, data_0[k].shape, data_0[k].dtype)

            print("Keys at time step 99:", data_99.files)
            for k in data_99.files:
                print(k, data_99[k].shape, data_99[k].dtype)

        # keys in both time steps
        exclude = {"sim_time", "Rcoord", "Zcoord"}
        keys_0 = sorted([k for k in data_0.files if k not in exclude and data_0[k].ndim == 2])
        keys_99 = sorted([k for k in data_99.files if k not in exclude and data_99[k].ndim == 2])
        print("Num channels:", len(keys_0), len(keys_99))
        ch_0_99 = [len(keys_0), len(keys_99)]

        # common channels
        both = set(keys_0) & set(keys_99)
        common_channels = both if common_channels is None else (common_channels & both)

        # find differences
        only_in_0  = sorted(set(data_0.files) - set(data_99.files))
        only_in_99 = sorted(set(data_99.files) - set(data_0.files))
        print("Keys only in time 0:", only_in_0)
        print("Keys only in time 99:", only_in_99)

        #% Create data array
        X_0 = np.stack([data_0[k] for k in keys_0], axis=0)
        X_99 = np.stack([data_99[k] for k in keys_99], axis=0)
        print("X shape (Time step 0):", X_0.shape, "dtype:", X_0.dtype)
        print("X shape (Time step 99):", X_99.shape, "dtype:", X_99.dtype)

        if only_in_0 == [] and only_in_99 == []:
            X = np.concatenate([X_0[np.newaxis, ...], X_99[np.newaxis, ...]], axis=0)
            #print("X shape (Both time steps):", X.shape, "dtype:", X.dtype)

            # % Find NaNs
            nan_mask = np.isnan(X)
            n_nans = int(nan_mask.sum())
            print("Total NaNs in X_0:", n_nans)

            nan_locs = np.argwhere(nan_mask)   # rows are [z_idx, r_idx, ch_idx]
            #print("nan_locs shape:", nan_locs.shape)

        ch_0_99_all.append(np.array(ch_0_99))

    finally:
        data_0.close()
        data_99.close()

print("ch_0_99_all shape (time step = 0):", np.array(ch_0_99_all)[:,0])
print("ch_0_99_all shape (time step = 99):", np.array(ch_0_99_all)[:,1])
print(f"Total valid trajectories with both time steps: {len(valid_trajs)}")

common_channels = sorted(common_channels)
print(f'{len(common_channels)} common_channels: {common_channels}')

# save common channels to a text file
with open(os.path.join("D:/data/HEAT", "common_channels.txt"), "w") as f:
    for ch in common_channels:
        f.write(f"{ch}\n")

#%% Reduce data and use only common channels
# load common channels from text file
# with open(os.path.join("D:/data/HEAT", "common_channels.txt"), "r") as f:
#     common_channels = [line.strip() for line in f.readlines()]

H, W = 1120, 400
C = len(common_channels)

batch_size = 100
N = len(valid_trajs)
n_files = (N + batch_size - 1) // batch_size  # ceil(N/100)

out_dir = r'F:/FM/codes/morph_github/morph-cl/MORPH/datasets/normalized_revin/2dHEAT_unnorm' # split into train/test/val manually later
os.makedirs(out_dir, exist_ok=True)

for file_idx in range(n_files):
    start = file_idx * batch_size
    end   = min(start + batch_size, N)
    batch_trajs = valid_trajs[start:end]  # up to 100 trajs (leftover in last)

    out_path = os.path.join(out_dir, f"HEAT_t1and100_fp16_full_processed_{file_idx:03d}.h5")
    print(f"\nWriting {out_path} with {len(batch_trajs)} trajectories...")

    with h5py.File(out_path, "w") as hf:
        dset = hf.create_dataset(
            "data",
            shape=(len(batch_trajs), 2, C, H, W),
            dtype=np.float16,
            compression="lzf",)

        for j, traj_no in enumerate(batch_trajs):
            p0  = os.path.join(npz_folder_path, f"lsc240420_id{traj_no:05d}_pvi_idx{0:05d}.npz")
            p99 = os.path.join(npz_folder_path, f"lsc240420_id{traj_no:05d}_pvi_idx{99:05d}.npz")

            # load data (auto-closes files)
            with np.load(p0) as data_0, np.load(p99) as data_99:
                X_0  = np.stack([data_0[k]  for k in common_channels], axis=0)
                X_99 = np.stack([data_99[k] for k in common_channels], axis=0)

            # replace NaNs, +Inf, -Inf with 0.0
            X_0  = np.nan_to_num(X_0,  nan=0.0, posinf=0.0, neginf=0.0)
            X_99 = np.nan_to_num(X_99, nan=0.0, posinf=0.0, neginf=0.0)

            # sanity check
            if X_0.shape != (C, H, W):
                raise ValueError(f"traj {traj_no} time 0 has unexpected shape {X_0.shape}")
            if X_99.shape != (C, H, W):
                raise ValueError(f"traj {traj_no} time 99 has unexpected shape {X_99.shape}")

            # write directly into HDF5 dataset (no need to build X with concatenate)
            dset[j, 0] = X_0
            dset[j, 1] = X_99

            if (j + 1) % 10 == 0 or (j + 1) == len(batch_trajs):
                # print progress
                print(f"  file {file_idx+1}/{n_files}: wrote {j+1}/{len(batch_trajs)}")

print("\nDone.")

# %% import and plot
# load common channels from text file
with open(os.path.join("D:/data/HEAT", "common_channels.txt"), "r") as f:
    common_channels = [line.strip() for line in f.readlines()]
    
# load the .npz file from directory
h5_folder_path = r'D:/data/HEAT/processed_h5/'
list_h5_files = list(sorted([f for f in os.listdir(h5_folder_path) if f.endswith('.h5')]))
print(f"Found {len(list_h5_files)} .h5 files in {h5_folder_path}")

idx_rand_h5_file = random.randint(0, len(list_h5_files)-1)
print(f"Loading random .h5 file index: {idx_rand_h5_file}")

fname = list_h5_files[idx_rand_h5_file]
fpath = os.path.join(h5_folder_path, fname)

with h5py.File(fpath, "r") as hf:
    data = hf["data"]  # (num_trajectories, 2, C, H, W)
    print("Data shape:", data.shape, "dtype:", data.dtype)

    # plot random trajectory
    traj_idx = random.randint(0, data.shape[0]-1)
    channel_idx = random.randint(0, data.shape[2]-1)

    img_0 = data[traj_idx, 0, channel_idx, :, :]
    img_99 = data[traj_idx, 1, channel_idx, :, :]

    fig, ax = plt.subplots(1, 2, figsize=(8, 6))

    im0 = ax[0].imshow(img_0, cmap='viridis', aspect='auto')
    fig.colorbar(im0, ax=ax[0])
    ax[0].set_title(f"Time step 0", fontsize=10)

    im1 = ax[1].imshow(img_99, cmap='viridis', aspect='auto')
    fig.colorbar(im1, ax=ax[1])
    ax[1].set_title(f"Time step 99", fontsize=10)
    
    fig.suptitle(
        f"h5 file = {idx_rand_h5_file} | traj = {traj_idx} | channel = {common_channels[channel_idx]}",
        fontsize=12
    )
    plt.tight_layout()
    plt.show()

# %% make GIFs for first 38 channels: (t0 | t99) across all trajectories in all h5 files
# load common channels from text file
with open(os.path.join("D:/data/HEAT", "common_channels.txt"), "r") as f:
    common_channels = [line.strip() for line in f.readlines()]

gif_dir = os.path.join('D:/data/HEAT', "gifs_t0_t99")
os.makedirs(gif_dir, exist_ok=True)

cmap = cm.get_cmap("viridis")
eps = 1e-12
stride = 1          # set >1 to skip trajectories (e.g., 5) to keep GIF smaller

def to_rgb(a, vmin, vmax):
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip((a - vmin) / (vmax - vmin + eps), 0, 1)
    return (cmap(x)[..., :3] * 255).astype(np.uint8)

for ch_idx in range(min(38, len(common_channels))):
    ch_name = str(common_channels[ch_idx]).replace("/", "_").replace("\\", "_")
    out_gif = os.path.join(gif_dir, f"ch{ch_idx:02d}_{ch_name}.gif")
    print(f"\nCreating GIF for channel {ch_idx}: {common_channels[ch_idx]} --> {out_gif}")
    n = 0
    with imageio.get_writer(out_gif, mode="I", fps=10) as w:
        for fname in list_h5_files:
            print("Processing:", fname)
            with h5py.File(os.path.join(h5_folder_path, fname), "r") as hf:
                d = hf["data"]  # (B,2,C,H,W)
                for t in range(0, d.shape[0], stride):
                    img0  = d[t, 0, ch_idx]
                    img99 = d[t, 1, ch_idx]
                    vmin = float(min(img0.min(), img99.min()))
                    vmax = float(max(img0.max(), img99.max()))
                    rgb0  = to_rgb(img0,  vmin, vmax)
                    rgb99 = to_rgb(img99, vmin, vmax)
                    sep = np.full((rgb0.shape[0], 8, 3), 255, np.uint8)  # 8px white bar
                    frame = np.concatenate([rgb0, sep, rgb99], axis=1)
                    w.append_data(frame)
                    n += 1

    print("saved:", out_gif, "| frames:", n)

# %%
