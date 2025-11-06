
import argparse
from src.utils.normalization import RevIN
import os
import sys
import numpy as np

# Add project root to path
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

# savepath of mu and var
savepath_muvar = os.path.join(project_root, 'data')

# --- Load dataset ---
parser = argparse.ArgumentParser(description="UPTF7 Transformation")
parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset (numpy file)')
args = parser.parse_args()

# Load dataset
dataset  = np.load(args.dataset_path)  # shape (N, T, F, C, D, H, W)
print("Loaded dataset shape", dataset.shape)

# --- REVIN normalization ---
revin = RevIN(savepath_muvar)
revin.compute_stats(dataset, prefix='stats_')
dataset_norm = revin.normalize(dataset, prefix='stats_')
print("Normalize dataset shape", dataset_norm.shape)

# --- Check round‐trip via denormalize ---
recovered = revin.denormalize(dataset_norm, prefix='stats_')
tol_6 = 1e-2
max_error = 0.0
for i in range(recovered.shape[0]):
    print(f'Current sample: {i}, Current max_error:{max_error:.7f}')
    maxerror_i = np.max(np.abs(recovered[i] - dataset[i]))  # saving some memory
    max_error = max(maxerror_i, max_error)
assert max_error < tol_6, "Denormalization did not perfectly recover original!"
print("CFD3D RevIN round-trip OK")
del recovered