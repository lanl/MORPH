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

# Import custom visualization utilities
from src.utils.explore_hdf5 import ExploreHDF5Structure
explore = ExploreHDF5Structure()

# extract .tar.gz files (# location of downloaded HF datasets)
data_loc = r'D:/data/material_fracturing/PHASE-FIELD/tungsten/' 
sub_folders = ['combined_bc', 'horizontal-bc']
zipped_files = natsorted(glob.glob(os.path.join(data_loc, sub_folders[0], "*.tar.gz")))
print(f"Found {len(zipped_files)} zipped files in {os.path.join(data_loc, sub_folders[0])}")

# specify single file
zipped_file = os.path.join(data_loc, sub_folders[0], zipped_files[0])

# specify the extracted folder name
out_dir = os.path.splitext(zipped_file)[0] 
os.makedirs(out_dir, exist_ok=True)

# skip if already extracted
if os.path.isdir(out_dir) and os.listdir(out_dir):
    print("Skipping (already extracted):", out_dir)
else:
    os.makedirs(out_dir, exist_ok=True)
    with tarfile.open(zipped_file, "r:*") as tar:
        tar.extractall(path=out_dir)
    print("Extracted to:", os.path.abspath(out_dir))

#%% select .h files to explore and visualize
# explore the content of the h5 file
h5_files_list = natsorted([f for f in os.listdir(out_dir) if f.endswith('.h5')])
print(f"Found {len(h5_files_list)} HDF5 files in {out_dir}: \n {h5_files_list}")

# get a specific model to visualize
model = np.random.randint(1, 1001)  # models 1 to 1000
h5_file_path = os.path.join(out_dir, h5_files_list[model-1])
print(f"Exploring HDF5 file: {h5_file_path}")

# explore the content of the h5 file
explore.explore_hdf5(h5_file_path)

#%% plots fracture states for the selected model
ncols = 10  # grid columns
with h5py.File(h5_file_path, "r") as f:
    # 1) build ordered list of (label, key, sort_time)
    items = []

    # initial (put first)
    init_key = f"model {model} fracture initial"
    if init_key in f:
        items.append(("initial", init_key, -1.0))

    # all "fracture at time = ..."
    prefix = f"model {model} fracture at time = "
    for k in f.keys():
        if k.startswith(prefix):
            t_str = k.split(prefix, 1)[1]
            items.append((f"t={t_str}", k, float(t_str)))

    # final (put last)
    final_key = f"model {model} fracture final"
    if final_key in f:
        items.append(("final", final_key, float("inf")))

    # sort: initial (-1), then times, then final (inf)
    items.sort(key=lambda x: x[2])

    # 2) plot grid
    n = len(items)
    print(f"Visualizing {n} fracture states for model {model}")
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 2.6))
    axes = np.array(axes).reshape(-1)

    for ax, (label, key, _) in zip(axes, items):
        img = f[key][()]  # (128,128)
        ax.imshow(img, origin="lower")
        
        if label not in ("initial", "final") and isinstance(label, str) and label.startswith("t="):
            t_val = float(label.split("=", 1)[1])   # handles scientific notation
            ax.set_title(f"t = {t_val:.2e}", fontsize=18)
        else:
            ax.set_title(label, fontsize=18)
        
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

plt.tight_layout()
plt.savefig(f"fracture_model_{model}_states.png", dpi=300)
plt.show()

# %%
