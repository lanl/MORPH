import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
import torch
import h5py
import os
import pickle
from scipy.io import loadmat
from scipy.interpolate import griddata

class DatasetforDataloader(Dataset):
    def __init__(self,X,y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,i):
        # create a tuple
        return self.X[i], self.y[i]
    
class Dataloading():
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    def load_data(self):
        """
        Used help from repo: https://github.com/kfukami/Voronoi-CNN

        Returns
        -------
        sst : (T, H*W) float32
            Raw SST array exactly like the repo uses.
        lat : (H,) float32
        lon : (W,) float32
        time : (T,) array
        land_mask : (H, W) bool
            True on land, False on ocean.
        """
        with h5py.File(os.path.join(self.dataset_dir, "sst_weekly.mat"), "r") as f:
            lat = np.array(f["lat"]).reshape(-1).astype(np.float32)
            lon = np.array(f["lon"]).reshape(-1).astype(np.float32)
            sst = np.array(f["sst"]).astype(np.float32)   # repo uses raw [T, H*W]
            time = np.array(f["time"]).reshape(-1)

        H, W = len(lat), len(lon)
        if sst.ndim != 2 or sst.shape[1] != H * W:
            raise ValueError(f"Expected sst shape [T, {H*W}], got {sst.shape}")

        land_mask = np.isnan(sst[0].reshape(H, W, order="F"))

        # prints
        print(f"Loaded SST data: sst shape={sst.shape}, "
              f"lat shape={lat.shape}, lon shape={lon.shape}, "
              f"time shape={time.shape}",
              f"land_mask shape={land_mask.shape}")
        
        has_nan = np.isnan(sst).any()
        num_nan = np.isnan(sst).sum()
        print(f"Raw data Has NaNs:{has_nan}, Number of NaNs: {num_nan}")
        
        return sst, lat, lon, time, land_mask

    def _frame_from_raw(self, sst, H, W, t):
        return sst[t].reshape(H, W, order="F")

    def _repo_sensor_locations(self, land_mask, n_sensors, seed):
        """
        - sample row/col independently with replacement
        - allow duplicates
        - resample only if point lands on land
        """
        H, W = land_mask.shape
        rng = np.random.RandomState(seed)

        locs = np.column_stack([
            rng.randint(H, size=n_sensors),
            rng.randint(W, size=n_sensors),
        ]).astype(np.int64)

        for s in range(n_sensors):
            a, b = locs[s]
            while land_mask[a, b]:
                a = rng.randint(H)
                b = rng.randint(W)
                locs[s] = [a, b]

        return locs

    def compute_sst_minmax(self, sst, lat, lon, train_idx):
        """
        Compute min/max using only training ocean points.
        Avoids land NaNs and avoids test leakage.
        """
        H, W = len(lat), len(lon)
        land_mask = np.isnan(self._frame_from_raw(sst, H, W, 0))

        vals_all = []

        for t in train_idx:
            frame = self._frame_from_raw(sst, H, W, t)
            vals = frame[~land_mask]
            vals = vals[~np.isnan(vals)].astype(np.float32)
            vals_all.append(vals)

        vals_all = np.concatenate(vals_all)

        sst_min = vals_all.min()
        sst_max = vals_all.max()

        print(f"SST min-max stats: min={sst_min:.6f}, max={sst_max:.6f}")

        return np.float32(sst_min), np.float32(sst_max)
    
    def get_inputs_targets(self, sst, lat, lon, n_sensors=10, 
                    seed=300, t_indices=None,
                    sst_min=None, sst_max=None):
        """
        Per-snapshot sensor setup:

        Every snapshot gets a different but deterministic sensor layout.

        X -> (N, 2, H, W)
            X[:, 0] = normalized Voronoi SST
            X[:, 1] = sensor mask

        Y -> (N, 2, H, W)
            Y[:, 0] = normalized full SST
            Y[:, 1] = ocean-valid mask
        """

        H, W = len(lat), len(lon)
        N = sst.shape[0]

        if t_indices is None:
            t_indices = np.arange(N)
        else:
            t_indices = np.asarray(t_indices)

        N_out = len(t_indices)

        land_mask = np.isnan(self._frame_from_raw(sst, H, W, 0))

        lon_grid, lat_grid = np.meshgrid(lon, lat)

        X = np.zeros((N_out, H, W, 2), dtype=np.float32)
        Y = np.zeros((N_out, H, W, 1), dtype=np.float32)

        sensor_locs_all = []

        print(
            f"Using per-snapshot sensor seeds: "
            f"base_seed={seed}, n_sensors={n_sensors}, snapshots={N_out}"
        )

        for k, t in enumerate(t_indices):

            # Different deterministic sensor layout for every snapshot.
            # Including n_sensors avoids accidental reuse across sensor-count blocks.
            snapshot_seed = int(seed + 1000003 * n_sensors + int(t))

            sensor_locs = self._repo_sensor_locations(land_mask, n_sensors, snapshot_seed)
            sensor_locs_all.append(sensor_locs)

            sensor_points = np.column_stack([
                lat[sensor_locs[:, 0]],
                lon[sensor_locs[:, 1]],
            ])

            sensor_mask = np.zeros((H, W), dtype=np.float32)
            sensor_mask[sensor_locs[:, 0], sensor_locs[:, 1]] = 1.0

            raw = self._frame_from_raw(sst, H, W, t).astype(np.float32)

            # land NaNs become raw 0 first, repo-style
            raw_filled = np.nan_to_num(raw, nan=0.0).astype(np.float32)

            if sst_min is not None and sst_max is not None:
                denom = sst_max - sst_min + 1e-8
                full = (raw_filled - sst_min) / denom
                land_value = (0.0 - sst_min) / denom
            else:
                full = raw_filled.copy()
                land_value = 0.0

            full = full.astype(np.float32)
            full[land_mask] = land_value

            sparse_values = full[sensor_locs[:, 0], sensor_locs[:, 1]]

            vor = griddata(sensor_points, sparse_values, 
                           (lat_grid, lon_grid), method="nearest")

            vor = np.nan_to_num(vor, nan=land_value).astype(np.float32)
            vor[land_mask] = land_value

            X[k, :, :, 0] = vor
            X[k, :, :, 1] = sensor_mask
            Y[k, :, :, 0] = full

        # Convert X to MORPH format: (N, 2, H, W)
        X = np.transpose(X, (0, 3, 1, 2))

        # Output ocean mask: 1 ocean, 0 land
        target_mask = (~land_mask).astype(np.float32)
        target_mask = np.broadcast_to(target_mask[None, :, :, None], (N_out, H, W, 1))

        Y2 = np.concatenate([Y, target_mask], axis=-1)
        Y = np.transpose(Y2, (0, 3, 1, 2))

        sensor_locs_return = np.stack(sensor_locs_all, axis=0) # shape: (N_out, n_sensors, 2)

        sensors_on_land = land_mask[
            sensor_locs_return[:, :, 0],
            sensor_locs_return[:, :, 1],
        ].sum()

        print(f"Prepared inputs and targets: X shape={X.shape}, Y shape={Y.shape}")
        print(f"sensor_locs_return shape={sensor_locs_return.shape}")
        print("Sensors on land across all snapshots:", sensors_on_land)
        print("NaNs in X:", np.isnan(X).sum())
        print("NaNs in Y:", np.isnan(Y).sum())

        return X, Y, sensor_locs_return
    
    def build_multi_sensor_dataset(
        self,
        dataloading,
        sst,
        lat,
        lon,
        t_indices,
        sensor_counts,
        sensor_seeds,
        sst_min,
        sst_max,
        split_name="train",
    ):
        X_list = []
        Y_list = []
        sensor_locs_dict = {}

        for n_sensors in sensor_counts:
            for seed in sensor_seeds:
                print(f"=== Building {split_name}: n_sensors={n_sensors}, seed={seed} ===")

                X, Y, sensor_locs = dataloading.get_inputs_targets(
                    sst,
                    lat,
                    lon,
                    n_sensors=n_sensors,
                    seed=seed,
                    t_indices=t_indices,
                    sst_min=sst_min,
                    sst_max=sst_max,
                )

                X_list.append(X)
                Y_list.append(Y)
                sensor_locs_dict[(n_sensors, seed)] = sensor_locs

        X_all = np.concatenate(X_list, axis=0)
        Y_all = np.concatenate(Y_list, axis=0)

        print(f"{split_name} multi-sensor X shape: {X_all.shape}")
        print(f"{split_name} multi-sensor Y shape: {Y_all.shape}")
        print(f"{split_name} configs: {list(sensor_locs_dict.keys())}")

        return X_all, Y_all, sensor_locs_dict
    
    
    def _pad_height_to_multiple(self, arr, multiple=8, value_pad=0.0):
        """
        arr: (N, C, H, W)
        Pads only H dimension.
        Channel 0 is padded with value_pad.
        Other channels are padded with 0.
        """
        N, C, H, W = arr.shape

        pad_total = (multiple - H % multiple) % multiple
        if pad_total == 0:
            return arr

        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top

        arr_pad = np.pad(
            arr,
            pad_width=((0, 0), (0, 0), (pad_top, pad_bottom), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )

        # Pad SST/value channel with land/sentinel value.
        arr_pad[:, 0, :pad_top, :] = value_pad
        arr_pad[:, 0, -pad_bottom:, :] = value_pad

        print(f"Padded height: {H} -> {arr_pad.shape[2]} "
            f"(top={pad_top}, bottom={pad_bottom}, value_pad={value_pad:.6f})")

        return arr_pad
    
    
    def pad_data(self, X, Y, sst_min=None, sst_max=None):
        """
        Pad the input and target data to make the height a multiple of 8.
        """
        # Pad H=180 -> 184 for patch_size=8
        if sst_min is not None and sst_max is not None:
            value_pad = np.float32((0.0 - sst_min) / (sst_max - sst_min + 1e-8))
        else:
            value_pad = np.float32(0.0)

        X_padH = self._pad_height_to_multiple(X, multiple=8, value_pad=value_pad)
        Y_padH = self._pad_height_to_multiple(Y, multiple=8, value_pad=value_pad)

        print(f"After padding: X_padH shape={X_padH.shape}, Y_padH shape={Y_padH.shape}")

        return X_padH, Y_padH

    def uptf7(self, X, set = 'inputs'):
        if set == 'inputs':
            # (B, F, H, W) -> (B, T=1, F, C=1, D=1, H, W)
            X_uptf7 = X[:, None, :, None, None, :, :]
            assert X_uptf7.ndim == 7, f"Expected 7D output, got {X_uptf7.ndim}D"
        if set == 'targets':
            # (B, F, H, W) -> (B, F, C=1, D=1, H, W)
            X_uptf7 = X[:, :, None, None, :, :]
            assert X_uptf7.ndim == 6, f"Expected 6D output, got {X_uptf7.ndim}D"
        return X_uptf7  
    
    def datasets(self, X, Y, data_frac = 1.0, split = 'train', random_state=42):

        # define dataloader
        train_ds = DatasetforDataloader(X, Y)

        # Subsample train/val only, deterministically
        # Interpreting data_frac as "fraction of train/val kept" 
        train_keep = max(1, int(len(train_ds) * data_frac))

        rng = np.random.default_rng(random_state)
        train_perm = rng.permutation(len(train_ds))

        train_small = Subset(train_ds, train_perm[:train_keep].tolist())

        print(f"=== data_frac={data_frac:.3f} (seed={random_state}) ===")
        print(f"{split}: {len(train_small)} / {len(train_ds)} samples")

        return train_small

    def dataloaders(self, ds, batch_size=32, shuf =  True, split = 'train'):

        # Dataloaders (optionally seed shuffle generator too)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuf)

        print(f"{split} batches: {len(dl)}")

        return dl