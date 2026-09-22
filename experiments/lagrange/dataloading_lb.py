import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
import torch
import h5py
import os

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

    def load_data(self, split = 'train'):
        with h5py.File(os.path.join(self.dataset_dir, f'{split}.h5'), 'r') as f:
            num_trajs = len(list(f.keys()))
            data_list = []
            for traj_key in f.keys():
                traj_data = f[traj_key]['position'][:].astype(np.float32)
                data_list.append(traj_data)
            data = np.stack(data_list, axis=0)  # shape: (num_trajs, T, Particles, Coordinates)
            print(f"Loaded {split} data: {data.shape} ==> (num_trajs, time-steps, Particles, Coordinates)")
    
        return data
    
    def visualize_data(self, results_dir, data, plot_remarks = 'train'):
        # Visualize single trajectory
        traj = data[0]  # shape: (T, C, H, W)
        T, P, C = traj.shape
        print(f"Visualizing trajectory with shape: {traj.shape}")

        # plot time steps in interval of 20, which mean 401 / 20 = 21 plots
        t_viz = [1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 
                 260, 280, 300, 320, 340, 360, 380, 400]
        plt.figure(figsize=(42, 12))
        for i, t in enumerate(t_viz):
            plt.subplot(3, 7, i+1)
            plt.scatter(traj[t,:,0], traj[t,:,1], s=10)
            plt.title(f'Time step {t}')
        plt.savefig(os.path.join(results_dir, f'trajectory_visualization_{plot_remarks}.png'))
        plt.close()
    
    def BoxNormalization(self, data, box_x_size, box_y_size):
        data_norm = np.copy(data)
        data_norm[..., 0] = data[..., 0] / box_x_size
        data_norm[..., 1] = data[..., 1] / box_y_size
        return data_norm

    def BoxUnnormalization(self, data, box_x_size, box_y_size):
        data_unnorm = np.copy(data)
        data_unnorm[..., 0] = data[..., 0] * box_x_size
        data_unnorm[..., 1] = data[..., 1] * box_y_size
        return data_unnorm

    def nsp_f(self, data):
        # data shape: (N, T, P, C)
        data_prev = data[:, :-1]   # shape: (N, T-1, P, C)
        data_next = data[:, 1:]    # shape: (N, T-1, P, C)

        data_prev_rs = data_prev.reshape(data_prev.shape[0] * data_prev.shape[1], 
                                         data_prev.shape[2], data_prev.shape[3])  # (N * T-1, P, C)
        data_next_rs = data_next.reshape(data_next.shape[0] * data_next.shape[1],
                                         data_next.shape[2], data_next.shape[3])  # (N * T-1, P, C)
        
        # reshape to (N * (T-1), C, P) for both prev and next
        data_prev_rs = data_prev_rs.transpose(0, 2, 1)  # (N * T-1, C, P)
        data_next_rs = data_next_rs.transpose(0, 2, 1)  # (N * T-1, C, P)

        print(f'Reshaped data_prev: {data_prev_rs.shape}, '
              f'Reshaped data_next: {data_next_rs.shape}')
        
        return data_prev_rs, data_next_rs
    
    def uptf7_f(self, X):
        # Convert to PyTorch format (N,F,P) --> (N, T, F, C, D, H, W = P)
        X_uptf = X[:, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]  # add T, C, D, H dimensions
        return X_uptf

    def datasets(self, prev, next, data_frac = 1.0, split = 'train', random_state=42):

        # define dataloader
        full_dataset = DatasetforDataloader(prev, next)

        # Interpreting data_frac as "fraction of train/val kept" 
        train_keep = max(1, int(len(full_dataset) * data_frac))
        val_keep = max(1, int(len(full_dataset) * data_frac))

        rng = np.random.default_rng(random_state)
        train_perm = rng.permutation(len(full_dataset))

        data_small = Subset(full_dataset, train_perm[:train_keep].tolist())

        print(f"=== data_frac={data_frac:.3f} (seed={random_state}) ===")
        print(f"{split}: {len(data_small)} / {len(full_dataset)} samples")

        return data_small

    def dataloaders(self, ds, batch_size=32, split='train'):

        # Dataloaders (optionally seed shuffle generator too)
        dl_train = DataLoader(ds, batch_size=batch_size, shuffle=True)

        print(f"{split} batches: {len(dl_train)}")

        return dl_train