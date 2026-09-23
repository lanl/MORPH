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

# Add project root to path
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..','..'))
sys.path.append(project_root)
print(f"Project root added to path: {project_root}")

main_path = os.path.join(project_root, "src", "visualizers")

data_loc = r'D:/data/HEAT/lsc240420_id00001_pvi/'
out_path = os.path.join(data_loc, "combined.npz")

files = sorted(glob.glob(os.path.join(data_loc, "*.npz")))
assert files, f"No .npz files found in {data_loc}"

# common keys
common_path = r"D:\data\HEAT\common_channels.txt"
with open(common_path, "r") as f:
    keys = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

buffers_1 = []  # will hold per-file arrays of shape (C,H,W)

for i, fp in enumerate(files):
    buffers_2 = []  # will hold per-channel arrays of shape (1,H,W)
    with np.load(fp) as z:
        for k in keys:
            arr = z[k]
            print(f"Shape of the array for key '{k}': {arr.shape}")
            #print(f"File {i+1}/{len(files)}: {os.path.basename(fp)}, key: {k}, shape: {arr.shape}")
            buffers_2.append(arr[np.newaxis, ...])   # (1,H,W)

    file_tensor = np.concatenate(buffers_2, axis=0)[np.newaxis,...]  # (1,C,H,W)
    print(f"Combined file tensor shape: {file_tensor.shape}")
    buffers_1.append(file_tensor)   # (1,C,H,W)

data = np.concatenate(buffers_1, axis=0)  # (N,C,H,W)
print("Final combined data shape:", data.shape)  # (N,C,H,W)

# replace nans with zeros
nan_count = np.isnan(data).sum()
print(f"Found {nan_count} NaN values in the data. Replacing with zeros.")
data = np.nan_to_num(data, nan=0.0)

# channel wise normalization
for c in range(data.shape[1]):
    channel_data = data[:, c, :, :]
    min_val = channel_data.min()
    max_val = channel_data.max()
    if max_val > min_val:
        data[:, c, :, :] = (channel_data - min_val) / (max_val - min_val)
    else:
        print(f"Warning: Channel {c} has constant value {min_val}. Skipping normalization.")

print("Shape of the data array after normalization:", data.shape)

# %% plots
import numpy as np
import matplotlib.pyplot as plt

# pick ANY channels you want (0-based indices)
# e.g. row_chans = [2, 3, 4] or row_chans = list(range(1, 39))
row_names = ["$avg \\rho$", "$\\rho_{case}$", "$\\rho_{cushion}$", 
             "$\\rho_{maincharge}$", "$\\rho_{striker}$", "$\\rho_{throw}$"]   # can be shorter than row_chans
row_chans = [2, 5, 6, 7, 9, 10]  # example channels to visualize

t_idxs = list(range(0, 101, 13))

fig, axes = plt.subplots(
    len(row_chans), len(t_idxs),
    figsize=(2.4 * len(t_idxs), 2.4 * len(row_chans)),
    squeeze=False,   
)

for r, c in enumerate(row_chans):
    name = row_names[r] if r < len(row_names) else f"chan_{c}"
    for j, t in enumerate(t_idxs):
        ax = axes[r, j]
        ax.imshow(data[t, c], cmap='viridis', aspect="auto")
        if r == 0:
            ax.set_title(f"t={t}", fontsize=22)
        if j == 0:
            ax.set_ylabel(name, fontsize=22)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.savefig(os.path.join(main_path, "heat_sim1_selected_channels.png"), dpi=300)
plt.show()

# %%
