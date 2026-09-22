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
        df_base = pd.read_csv(self.dataset_dir + '/IISHM_BaselineExpData.txt',header=None)
        print(f'Raw Baseline shape: {df_base.shape}')

        df_dam = pd.read_csv(self.dataset_dir + '/IISHM_DamageExpData.txt',header=None)
        print(f'Raw Damage shape: {df_dam.shape}')

        time_vector = pd.read_csv(self.dataset_dir + '/IISHM_timevector.txt',header=None)
        print(f'Time vector shape: {time_vector.shape}')
        
        df_base = np.array(df_base).astype(np.float32)[:,np.newaxis,:]
        df_dam = np.array(df_dam).astype(np.float32)[:,np.newaxis,:]
        df = np.concatenate([df_base,df_dam], axis=0).astype(np.float32)
        print(f'Combined shape: {df.shape}')
        
        rmsd = self.RMSD(df_base, df_dam)
        print(f'RMSD shape: {rmsd.shape}')

        return df, df_base, df_dam, time_vector, rmsd
    
    def plot_rmsd(self, results_dir, rmsd):
        # plt rmsd
        plt.figure(figsize=(12,4))
        plt.plot(rmsd)
        plt.title('RMSD between Baseline and Damage',fontsize=25)
        plt.xlabel('Sample Index',fontsize=25)
        plt.ylabel('RMSD',fontsize=25)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.ylim(0, 1.1)
        plt.savefig(f'{results_dir}/rmsd.png', dpi=300)

    # Function for RMSD (Root Mean Squared Difference)
    def RMSD(self, base, dam):
        num = np.sum((base[:,0,:]-dam[:,0,:])**2,axis=1)
        den = np.sum(base[:,0,:]**2,axis=1)
        rmsd = np.sqrt(np.divide(num,den))
        return rmsd
    
    def freq_stamps(self, df):
        seqlen = df.shape[2]
        dT = 1e-7
        freq = np.fft.fftfreq(seqlen, dT)
        print(f'Frequency shape: {freq.shape}')
        return freq
    
    def fft_data(self, df_base, df_dam):
        # Apply FFT along the time dimension (axis=2)
        df_base_fft = np.fft.fft(df_base, axis=2)
        df_dam_fft = np.fft.fft(df_dam, axis=2)

        # Take the magnitude (absolute value) of the FFT results
        df_base_fft_mag = np.abs(df_base_fft)
        df_dam_fft_mag = np.abs(df_dam_fft)

        return df_base_fft_mag, df_dam_fft_mag
    
    # visualization function for 2D data (if needed)
    def visualize_data(self, results_dir, df, df_base, df_dam, time_vector):
        time = time_vector
        rmsd = self.RMSD(df_base, df_dam)
        print(f'RMSD shape: {rmsd.shape}')

        rand_samp = random.randint(1,df_base.shape[0])
        print(f'sample number: {rand_samp}')
        print(f'RMSD: {rmsd[rand_samp]}')

        plt.figure(figsize=(12,4))
        plt.plot(time, df_base[rand_samp,0,:], label='Baseline')
        plt.plot(time, df_dam[rand_samp,0,:], label='Damage')
        plt.title('Training example',fontsize=25)
        plt.legend(['base', 'damage'], loc='upper right',fontsize=18)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.xlabel('Time is seconds',fontsize=25)
        plt.ylabel('Norm Amplitude',fontsize=25)
        plt.savefig(f'{results_dir}/sample_data.png', dpi=300)

        # take fft
        df_base_fft, df_dam_fft = self.fft_data(df_base, df_dam)
        freq = self.freq_stamps(df)

        plt.figure(figsize=(12,4))
        plt.plot(freq, df_base_fft[rand_samp,0,:], label='Baseline FFT')
        plt.plot(freq, df_dam_fft[rand_samp,0,:], label='Damage FFT')
        plt.title('FFT of training example',fontsize=25)
        plt.legend(['base_fft', 'damage_fft'], loc='upper right',fontsize=18)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.xlabel('Frequency (Hz)',fontsize=25)
        plt.ylabel('Magnitude',fontsize=25)
        plt.savefig(f'{results_dir}/sample_fft.png', dpi=300)

    def normalize_data(self, df):
        # Standardize each row to mean 0 and std 1
        df_mu = np.mean(df)
        df_sigma = np.std(df)
        df_norm = (df - df_mu) / (df_sigma + 1e-8)
        print(f'Normalized shape: {df_norm.shape}, mu = {df_mu}, sigma = {df_sigma}')
        return df_norm, df_mu, df_sigma
    
    def denormalize_data(self, df_norm, mu, sigma):
        df_denorm = df_norm * (sigma + 1e-8) + mu
        return df_denorm

    def create_labels_detection(self, df_base, df_dam):
        # Input/Features and labels extraction
        # Baseline = 0
        # Damage = 1
        mclass_1 = df_base.shape[0]
        mclass_2 = df_dam.shape[0]

        y_1 = np.zeros((mclass_1,1), dtype=np.float32)
        y_2 = np.ones((mclass_2,1), dtype=np.float32)

        y = np.concatenate([y_1,y_2], axis=0)
        y = np.array(y).astype(np.float32)

        print(f"Shape of labels: {y.shape}")
        return y
    
    def noise_augmentation(self, df, noise_level=0.01):
        noise = np.random.normal(0, noise_level, df.shape).astype(np.float32)
        df_noisy = df + noise
        return df_noisy
    
    def uptf7(self, X):
        # Convert to PyTorch format (N,C,W) --> (N, T, F, C, D, H, W)
        X_uptf = X[:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]  # add channel and spatial dims
        print(f'UTPF7 format shape: {X_uptf.shape}')
        return X_uptf

    def datasets(self, X, Y, data_frac, random_state=42):

        # define dataloader
        full_dataset = DatasetforDataloader(X, Y)
        
        # Fixed split (90/10/0) with seeded generator => test is constant
        N = len(full_dataset)
        test_size = int(N * 0.10)
        val_size = int(N * 0.10)
        train_size = N - val_size - test_size

        g = torch.Generator().manual_seed(random_state)
        
        # split the dataset into train/val/test
        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size], generator=g)

        # Subsample train/val only, deterministically
        # Interpreting data_frac as "fraction of train/val kept" 
        train_keep = max(1, int(len(train_ds) * data_frac))
        val_keep = max(1, int(len(val_ds) * data_frac))

        rng = np.random.default_rng(random_state)
        train_perm = rng.permutation(len(train_ds))
        val_perm = rng.permutation(len(val_ds))

        train_small = Subset(train_ds, train_perm[:train_keep].tolist())
        val_small = Subset(val_ds, val_perm[:val_keep].tolist())
        test_fixed = test_ds  # unchanged

        print(f"=== data_frac={data_frac:.3f} (seed={random_state}) ===")
        print(f"Train: {len(train_small)} / {len(train_ds)} samples")
        print(f"Val:   {len(val_small)} / {len(val_ds)} samples")
        print(f"Test:  {len(test_fixed)} / {len(test_ds)} samples (fixed)")

        return train_small, val_small, test_fixed

    def dataloaders(self, train_ds, val_ds, test_ds, batch_size=32):

        # Dataloaders (optionally seed shuffle generator too)
        dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        dl_test = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        print(f"Training batches: {len(dl_train)}")
        print(f"Validation batches: {len(dl_val)}")
        print(f"Test batches: {len(dl_test)}")

        return dl_train, dl_val, dl_test