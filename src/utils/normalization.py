import os
import numpy as np
import torch

class RevIN:
    def __init__(self, stats_folder, eps=1e-6):
        """
        stats_folder: directory where mu/var .npy files will be saved and loaded.
        eps: small constant to avoid divide-by-zero.
        """
        self.stats_folder = stats_folder
        self.eps = eps
        os.makedirs(stats_folder, exist_ok=True)
        self.mu = None  # shape (N, F)
        self.var = None  # shape (N, F)

    def compute_stats(self, data, prefix='stats'):
        """
        data: numpy array of shape (N, T, F, C, D, H, W)
        prefix: filename prefix for saved stats (.npy)
        Saves mu and var of shape (N, F) to stats_folder.
        """
        print("Running ReVIN (Numpy array of shape must be (N, T, F, C, D, H, W))...")
        
        # reorder to (N, F, T, C, D, H, W)
        data_re = data.transpose(0, 2, 1, 3, 4, 5, 6)
        N, F, T, C, D, H, W = data_re.shape

        # flatten spatial-temporal dims
        flat = data_re.reshape(N, F, -1)
        mu = flat.mean(axis=2)       # (N, F)
        var = flat.var(axis=2)       # (N, F)

        # save
        mu_path = os.path.join(self.stats_folder, f'{prefix}_mu.npy')
        var_path = os.path.join(self.stats_folder, f'{prefix}_var.npy')
        np.save(mu_path, mu)
        np.save(var_path, var)

        self.mu = mu
        self.var = var
        

    def normalize(self, batch, prefix='stats'):
        """
        batch: numpy array of shape (N, T, F, C, D, H, W)
        Returns normalized batch of same shape.
        """
        if self.mu is None or self.var is None:
            mu_path = os.path.join(self.stats_folder, f'{prefix}_mu.npy')
            var_path = os.path.join(self.stats_folder, f'{prefix}_var.npy')
            self.mu = np.load(mu_path)
            self.var = np.load(var_path)

        # transpose to (N, F, T, C, D, H, W)
        batch = batch.transpose(0, 2, 1, 3, 4, 5, 6)

        # reshape stats for broadcasting
        mu_b = self.mu[..., None, None, None, None, None]
        var_b = self.var[..., None, None, None, None, None]

        # normalize
        normalized = (batch - mu_b) / np.sqrt(var_b + self.eps)
        # back to original shape
        return normalized.transpose(0, 2, 1, 3, 4, 5, 6)

    def denormalize(self, batch, prefix='stats'):
        """
        batch: numpy array of shape (N, T, F, C, D, H, W)
        Returns denormalized batch of same shape.
        """
        if self.mu is None or self.var is None:
            mu_path = os.path.join(self.stats_folder, f'{prefix}_mu.npy')
            var_path = os.path.join(self.stats_folder, f'{prefix}_var.npy')
            self.mu = np.load(mu_path)
            self.var = np.load(var_path)

        batch = batch.transpose(0, 2, 1, 3, 4, 5, 6)
        mu_b = self.mu[..., None, None, None, None, None]
        var_b = self.var[..., None, None, None, None, None]

        # denormalize
        denorm = batch * np.sqrt(var_b + self.eps) + mu_b
        return denorm.transpose(0, 2, 1, 3, 4, 5, 6)
    
    def denormalize_testeval(loadpath_muvar, norm_prefix, test_data, dataset,
                             muvar_portion = None):
                
        # mu and var paths
        mu_path = os.path.join(loadpath_muvar, f'{norm_prefix}_mu.npy')
        var_path = os.path.join(loadpath_muvar, f'{norm_prefix}_var.npy')
        
        # load mu and var 
        mu = np.load(mu_path) if dataset != 'CFD2D-IC' else np.load(mu_path)[:,1:] # force field was static, removed in the dataset
        var = np.load(var_path) if dataset != 'CFD2D-IC' else np.load(var_path)[:,1:]
        print(f'[{dataset}] Shape of imported norm constants (mu/var) : {mu.shape}')
        
        # mu and var for the test examples (lst 10% values)
        if muvar_portion == None:
            muvar_portion = test_data.shape[0]
            
        mu_test = mu[- muvar_portion:]
        var_test = var[- muvar_portion:]
        print(f'[{dataset}] Shape of (test) norm constants (mu/var): {mu_test.shape}')
        
        assert mu_test.shape[0] == test_data.shape[0], \
            'mu/var and data shapes dont match'
        
        # Tranpose (N,T,F,C,D,H,W) -> (N,F,T,C,D,H,W)
        test_data_tr = test_data.permute(0, 2, 1, 3, 4, 5, 6) 
        print(f'[N,T,F,C,D,H,W] → [N,T,F,C,D,H,W]  Permuted test data for denorm. {test_data_tr.shape}')
        
        # modify mu and var for the data shape
        mu_b = mu_test[..., None, None, None, None, None] # (N,F,1,1,1,1,1)
        var_b = var_test[..., None, None, None, None, None] # (N,F,1,1,1,1,1)
        print(f'[N,F,T,C,D,H,W] Expanded (mu/var) : {mu_b.shape}')
        
        # denormalize
        mu_b, var_b = torch.from_numpy(mu_b), torch.from_numpy(var_b)
        test_data_denorm = test_data_tr * torch.sqrt(var_b + 1e-8) + mu_b
        print(f'[N,F,T,C,D,H,W] Denormalized shape {test_data_denorm.shape}')
        
        # Tranpose back (N,F,T,C,D,H,W) -> (N,T,F,C,D,H,W)
        test_data_denorm_tr = test_data_denorm.permute(0, 2, 1, 3, 4, 5, 6)
        print(f'[N,T,F,C,D,H,W] → [N,T,F,C,D,H,W] Permuted denormalized shape {test_data_denorm.shape}')
        
        return test_data_denorm_tr
