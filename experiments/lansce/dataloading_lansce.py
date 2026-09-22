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

    def load_data(self):
        psp = np.load(self.dataset_dir + '/run9ps.npy')
        print(f'Phase space projections: {psp.shape}')

        rf = np.load(self.dataset_dir + '/run9rf.npy')
        print(f'RF data: {rf.shape}')

        return psp, rf
    
    def visualize_data(self, results_dir, psp, projection = 11, plot_remarks = None):
        allprojnames = ['$x-p_x$','$x-y$','$x-p_y$','$x-z$','$x-p_z$',
            '$y-p_x$','$y-p_y$','$y-z$','$y-p_z$','$z-p_x$',
            '$z-p_y$','$z-p_z$','$p_z-p_y$','$p_x-p_z$','$p_z-p_y$']
                    
        plot_samples = 48
        plt.figure(figsize=(16,8))
        psp_log = np.log(1 + psp)  # log scale for better visualization
        for i in range(plot_samples):  
            cols = 12
            plt.subplot(int(plot_samples/cols) + 1, cols, i + 1)
            plt.suptitle(f'allprojnames[projection] projection',fontsize=20)
            plt.imshow(psp_log[0,i,projection,:,:], aspect='auto', origin='lower', cmap='plasma')
            #plt.tick_params(left = False, right = False , labelleft = False ,
                            #labelbottom = False, bottom = False)
            plt.title('Mod.-'+str(i+1), fontsize = 12)
            if (i == 0 or i == 12 or i == 24 or i == 36 or i == 48):
                plt.ylabel('% $E_{r}$ (MeV)',fontsize=12)
                yticks = [-1.3, 0, 1.3]
                yticklabels = [str(yticks[0]), str(yticks[1]), str(yticks[2])]
                plt.yticks([0,128,256], yticklabels, fontsize=10)
            else:
                plt.yticks([],fontsize=10)
            
            if (i>35 and i<48):
                plt.xlabel('$\Delta \phi (deg)$',fontsize=12)
                xticks = [-60, 0, 60]
                xticklabels = [str(xticks[0]), str(xticks[1]), str(xticks[2])]
                plt.xticks([0,128,256], xticklabels, fontsize=10)
            else:
                plt.xticks([],fontsize=10) 
            
            plt.subplots_adjust(wspace=0.23, hspace=0.32)
        
        plt.savefig(os.path.join(results_dir, f'dataset_modules_4x12_proj_{projection}_{plot_remarks}.png'), bbox_inches='tight',dpi=300)

    def MinMaxNormalizePSP(self,psp):
        # simplified to see the memory utilization at each step
        # dividing an array with a number takes more time and multiple by (1/number)
        minval = np.min(psp)
        maxval = np.max(psp)
        diff = maxval-minval
        diffinv = 1/diff
        data_nm_numer = psp - minval
        psp = data_nm_numer*diffinv
        return psp, minval, maxval

    def UnNormalizePSP(self,data, minval, maxval):
        diff = maxval-minval
        data_un = data*diff + minval
        return data_un
    
    def normalize_rfs(self,rf):
        minval_list, maxval_list = [], []
        for i in range(rf.shape[1]):
            minval = rf[:,i].min()
            maxval = rf[:,i].max()
            rf[:,i] = (rf[:,i] - minval) / (maxval - minval)
            minval_list.append(minval)
            maxval_list.append(maxval)
            print(f'Parameter {i+1}: min={minval}, max={maxval}')
        return rf, minval_list, maxval_list

    def splits(self, psp, rf, seed=42):
        N = psp.shape[0]
        assert rf.shape[0] == N, "psp and rf must have same number of samples"

        test_size = int(N * 0.05) # 5% for test, 95% for train/val

        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(N, generator=g)

        te_idx = perm[:test_size]
        tv_idx = perm[test_size:]

        trainval_psp = psp[tv_idx].astype(np.float32)
        trainval_rf = rf[tv_idx].astype(np.float32)
        test_psp = psp[te_idx].astype(np.float32)
        test_rf = rf[te_idx].astype(np.float32)

        return (trainval_psp, trainval_rf), (test_psp, test_rf)
    
    def nsp_f(self, psp):
        # psp shape: (N, T, C, H, W)
        psp_prev = psp[:, :-1]   # shape: (N, T-1, C, H, W)
        psp_next = psp[:, 1:]    # shape: (N, T-1, C, H, W)

        psp_prev_rs = psp_prev.reshape(psp_prev.shape[0] * psp_prev.shape[1], 
                                       psp_prev.shape[2], psp_next.shape[3], 
                                       psp_next.shape[4])  # (N * T-1, C, H, W)
        psp_next_rs = psp_next.reshape(psp_next.shape[0] * psp_next.shape[1],
                                       psp_next.shape[2], psp_next.shape[3],
                                       psp_next.shape[4])  # (N * T-1, C, H, W)
        print(f'Reshaped psp_prev: {psp_prev_rs.shape}, '
              f'Reshaped psp_next: {psp_next_rs.shape}')
        
        return psp_prev_rs, psp_next_rs
    
    def uptf7_f(self, X):
        # Convert to PyTorch format (N,F,H,W) --> (N, T, F, C, D, H, W)
        X_uptf = X[:, np.newaxis, :, np.newaxis, np.newaxis]  # add C and D
        return X_uptf
    
    def uptf7_c(self, X):
        N,C,H,W = X.shape
        if C == 1:
            # Convert to PyTorch format (N,C,H,W) --> (N, T, F, C, D, H, W)
            X_uptf = X[:, np.newaxis, np.newaxis, :, np.newaxis]  # add F and D
        else:
            raise ValueError(f"Expected C=1 for uptf7_c, got C={C}")
        return X_uptf

    def datasets(self, X, Y, data_frac, random_state=42):

        # define dataloader
        full_dataset = DatasetforDataloader(X, Y)
        
        # Fixed split (90/5) with seeded generator => test is constant
        N = len(full_dataset)
        val_size = int(N * 0.05) # 5% for validation, 95% for training out of 95% trainval
        train_size = N - val_size

        g = torch.Generator().manual_seed(random_state)
        
        # split the dataset into train/val/test
        train_ds, val_ds = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], generator=g)

        # Subsample train/val only, deterministically
        # Interpreting data_frac as "fraction of train/val kept" 
        train_keep = max(1, int(len(train_ds) * data_frac))
        val_keep = max(1, int(len(val_ds) * data_frac))

        rng = np.random.default_rng(random_state)
        train_perm = rng.permutation(len(train_ds))
        val_perm = rng.permutation(len(val_ds))

        train_small = Subset(train_ds, train_perm[:train_keep].tolist())
        val_small = Subset(val_ds, val_perm[:val_keep].tolist())

        print(f"=== data_frac={data_frac:.3f} (seed={random_state}) ===")
        print(f"Train: {len(train_small)} / {len(train_ds)} samples")
        print(f"Val:   {len(val_small)} / {len(val_ds)} samples")

        return train_small, val_small

    def dataloaders(self, train_ds, val_ds, batch_size=32):

        # Dataloaders (optionally seed shuffle generator too)
        dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        print(f"Training batches: {len(dl_train)}")
        print(f"Validation batches: {len(dl_val)}")

        return dl_train, dl_val