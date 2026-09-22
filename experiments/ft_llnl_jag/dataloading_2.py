import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, Subset

from visualization import data_visualizer
from sensitivity_analysis import sensitivity_analyser
from normalization import normalizer

# some important notes 
'''
These are two ways to perform scaling studies on the ICF-JAG dataset:
1. Mehod-1 (dataloading.py):
2. In this method, we first select a fraction of the dataset (data_frac),
3. Split into train/val/test.
4. Test set varies with data_frac.
5. Method-2 (This code): 
6. In this method, we first split the full dataset into train/val/test (80/10/10).
7. The data_frac is applied only on the training and validation sets.
8. The test set remains constant across different data_frac.
    
Conclusion:
1. Method-1 and Method-2 yeild similar results in scaling studies.
2. Method-1 and Method-2 yeild similar results in comparison of fine-tuning vs training from scratch.
3. The paper uses method-1.
'''

# dataset class
class DatasetforDataloader(Dataset):
    def __init__(self, X, p, s):
        self.X = X
        self.s = s
        self.p = p

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.p[i], self.s[i]


def _set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Optional (more deterministic, sometimes slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dataloaders(
    current_dir,
    model_dir,
    results_dir,
    data_frac,
    batch_size=8,
    params_to_use=None,
    seed=123,
):
    _set_all_seeds(seed)

    # Load ICF-JAG-10K dataset
    path_images = os.path.join(current_dir, "icf-jag-10k", "jag10K_images.npy")
    path_params = os.path.join(current_dir, "icf-jag-10k", "jag10K_params.npy")
    path_scalars = os.path.join(current_dir, "icf-jag-10k", "jag10K_0_scalars.npy")

    images = np.load(path_images, allow_pickle=False).astype(np.float32)
    params = np.load(path_params, allow_pickle=False).astype(np.float32)
    scalars = np.load(path_scalars, allow_pickle=False).astype(np.float32)

    print("images.shape:", images.shape)
    print("params.shape:", params.shape)
    print("scalars.shape:", scalars.shape)

    # reshape images
    images_reshape = images.reshape(images.shape[0], 64, 64, 4).astype(np.float32)
    print(f"Reshaped images: {images_reshape.shape}")  # (N, 64, 64, 4)

    # (Optional) Visualization + sensitivity on full dataset
    print("=== Data Visualization ===")
    data_visualizer(images_reshape, scalars, params, save_dir=results_dir)
    print(f"Data visualization saved to {results_dir}")

    print("=== Sensitivity Analysis ===")
    sensitivity_analyser(images_reshape, scalars, params, save_dir=results_dir)
    print(f"Sensitivity analysis results saved to {results_dir}")

    # Normalize (note: see leakage note below)
    print("=== Data Normalization ===")
    images_norm, scalars_norm, params_norm = normalizer(
        images_reshape, scalars, params, stats_dir=model_dir
    )

    # select parameters
    print(f"=== Using parameters indices: {params_to_use} ===")
    if params_to_use is None:
        params_sel = params_norm
    else:
        # allow int or list/array
        if isinstance(params_to_use, int):
            params_to_use = [params_to_use]
        params_sel = params_norm[:, params_to_use]

    X = images_norm
    s = scalars_norm
    p = params_sel

    full_dataset = DatasetforDataloader(X, p, s)

    # Fixed split (80/10/10) with seeded generator => test is constant
    N = len(full_dataset)
    train_size = int(0.8 * N)
    val_size = int(0.1 * N)
    test_size = N - train_size - val_size

    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=g
    )

    # Subsample train/val only, deterministically
    # Interpreting data_frac as "fraction of train/val kept" (most common for learning curves)
    train_keep = max(1, int(len(train_ds) * data_frac))
    val_keep = max(1, int(len(val_ds) * data_frac))

    rng = np.random.default_rng(seed)
    train_perm = rng.permutation(len(train_ds))
    val_perm = rng.permutation(len(val_ds))

    train_small = Subset(train_ds, train_perm[:train_keep].tolist())
    val_small = Subset(val_ds, val_perm[:val_keep].tolist())
    test_fixed = test_ds  # unchanged

    print(f"=== data_frac={data_frac:.3f} (seed={seed}) ===")
    print(f"Train: {len(train_small)} / {len(train_ds)} samples")
    print(f"Val:   {len(val_small)} / {len(val_ds)} samples")
    print(f"Test:  {len(test_fixed)} / {len(test_ds)} samples (fixed)")

    # Dataloaders (optionally seed shuffle generator too)
    dl_train = DataLoader(train_small, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(val_small, batch_size=batch_size, shuffle=False)
    dl_test = DataLoader(test_fixed, batch_size=batch_size, shuffle=False)

    return images_norm, p, scalars_norm, dl_train, dl_val, dl_test
