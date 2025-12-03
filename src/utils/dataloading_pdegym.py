import numpy as np
import random
import torch
import gc
from torch.utils.data import Dataset, DataLoader
from src.utils.dataloaders.dataloader_fns_kf_2d import FNSKF2dDataLoader
from src.utils.dataloaders.dataloader_ce_2d import CE2dDataLoader
from src.utils.dataloaders.dataloader_ns_2d import NS2dDataLoader
from torch.utils.data import ConcatDataset
from src.utils.data_preparation_fast import FastARDataPreparer

# Dataset for Dataloader
class DatasetforDataloader(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def pdegym_datasets(loadpath_ns_sines, datapath_ns_gaussians, 
                    loadpath_ce_rp, loadpath_ce_crp,
                    loadpath_ce_kh, loadpath_ce_gauss, use_small_dataset = False,
                    ar_order = 1):
    
    preparer = FastARDataPreparer(ar_order = ar_order)

    if use_small_dataset:
        print("→ Using SMALL datasets for quick testing...")

        # ---- NS-Sines ----
        dataset_ns_sines = NS2dDataLoader(loadpath_ns_sines, dataset_name='NS-Sines') 
        train_data_ns_sines, val_data_ns_sines = dataset_ns_sines.split_train()
        test_data_ns_sines = dataset_ns_sines.split_test()

        # concat and split again
        dataset_ns_sines = np.concatenate([train_data_ns_sines, val_data_ns_sines, test_data_ns_sines], axis = 0)
        train_data_ns_sines = dataset_ns_sines[:19640]
        val_data_ns_sines = dataset_ns_sines[19640:19640+120]
        test_data_ns_sines = dataset_ns_sines[19640+120:]
        print(f"NS-Sines train shape: {train_data_ns_sines.shape}, val shape: {val_data_ns_sines.shape}, test shape: {test_data_ns_sines.shape}")

        # prepare data
        X_tr_ns_sines, y_tr_ns_sines = preparer.prepare(train_data_ns_sines) # also converts (N,T,D,H,W,C,F) -> (N,T,F,C,D,H,W)
        X_va_ns_sines, y_va_ns_sines = preparer.prepare(val_data_ns_sines)
        del train_data_ns_sines, val_data_ns_sines, test_data_ns_sines
        gc.collect()
        
        # ---- CE-RP ----
        dataset_ce_rp = CE2dDataLoader(loadpath_ce_rp, dataset_name='CE-RP')
        train_data_ce_rp, val_data_ce_rp = dataset_ce_rp.split_train()
        test_data_ce_rp = dataset_ce_rp.split_test()

        # concat and split again
        dataset_ce_rp = np.concatenate([train_data_ce_rp, val_data_ce_rp, test_data_ce_rp], axis = 0)
        train_data_ce_rp = dataset_ce_rp[:9640]
        val_data_ce_rp = dataset_ce_rp[9640:9640+120]
        test_data_ce_rp = dataset_ce_rp[9640+120:]
        print(f"CE-RP train shape: {train_data_ce_rp.shape}, val shape: {val_data_ce_rp.shape}, test shape: {test_data_ce_rp.shape}")

        # prepare data
        X_tr_ce_rp, y_tr_ce_rp = preparer.prepare(train_data_ce_rp)
        X_va_ce_rp, y_va_ce_rp = preparer.prepare(val_data_ce_rp)
        del train_data_ce_rp, val_data_ce_rp, test_data_ce_rp
        gc.collect()

        # Wrap into DatasetforDataloader
        train_data_ns_sines = DatasetforDataloader(X_tr_ns_sines, y_tr_ns_sines)
        val_data_ns_sines   = DatasetforDataloader(X_va_ns_sines, y_va_ns_sines)
        train_data_ce_rp    = DatasetforDataloader(X_tr_ce_rp, y_tr_ce_rp)
        val_data_ce_rp      = DatasetforDataloader(X_va_ce_rp, y_va_ce_rp)

        # Use smaller subsets for quick testing
        train_ns = train_data_ns_sines
        val_ns   = val_data_ns_sines

        train_ce = train_data_ce_rp
        val_ce   = val_data_ce_rp

    else:
        print("→ Using FULL datasets for training...")
        # ---- NS-Sines ----
        dataset_ns_sines = NS2dDataLoader(loadpath_ns_sines, dataset_name='NS-Sines') 
        train_data_ns_sines, val_data_ns_sines = dataset_ns_sines.split_train()
        test_data_ns_sines = dataset_ns_sines.split_test()

        # concat and split again
        dataset_ns_sines = np.concatenate([train_data_ns_sines, val_data_ns_sines, test_data_ns_sines], axis = 0)
        train_data_ns_sines = dataset_ns_sines[:19640]
        val_data_ns_sines = dataset_ns_sines[19640:19640+120]
        test_data_ns_sines = dataset_ns_sines[19640+120:]
        print(f"→ NS-Sines train shape: {train_data_ns_sines.shape}, val shape: {val_data_ns_sines.shape}, test shape: {test_data_ns_sines.shape}")

        # prepare data
        X_tr_ns_sines, y_tr_ns_sines = preparer.prepare(train_data_ns_sines) # also converts (N,T,D,H,W,C,F) -> (N,T,F,C,D,H,W)
        X_va_ns_sines, y_va_ns_sines = preparer.prepare(val_data_ns_sines)
        del train_data_ns_sines, val_data_ns_sines, test_data_ns_sines
        gc.collect()

        # ---- NS-Gauss ----
        dataset_ns_gaussians = NS2dDataLoader(datapath_ns_gaussians, dataset_name='NS-Gaussians')
        train_data_ns_gaussians, val_data_ns_gaussians = dataset_ns_gaussians.split_train()
        test_data_ns_gaussians = dataset_ns_gaussians.split_test()

        # concat and split again
        dataset_ns_gaussians = np.concatenate([train_data_ns_gaussians, val_data_ns_gaussians, test_data_ns_gaussians], axis = 0)
        train_data_ns_gaussians = dataset_ns_gaussians[:19640]
        val_data_ns_gaussians = dataset_ns_gaussians[19640:19640+120]
        test_data_ns_gaussians = dataset_ns_gaussians[19640+120:]
        print(f"→ NS-Gaussians train shape: {train_data_ns_gaussians.shape}, val shape: {val_data_ns_gaussians.shape}, test shape: {test_data_ns_gaussians.shape}")

        # prepare data
        X_tr_ns_gaussians, y_tr_ns_gaussians = preparer.prepare(train_data_ns_gaussians)
        X_va_ns_gaussians, y_va_ns_gaussians = preparer.prepare(val_data_ns_gaussians)
        del train_data_ns_gaussians, val_data_ns_gaussians, test_data_ns_gaussians
        gc.collect()

        # ---- CE-RP ----
        dataset_ce_rp = CE2dDataLoader(loadpath_ce_rp, dataset_name='CE-RP')
        train_data_ce_rp, val_data_ce_rp = dataset_ce_rp.split_train()
        test_data_ce_rp = dataset_ce_rp.split_test()

        # concat and split again
        dataset_ce_rp = np.concatenate([train_data_ce_rp, val_data_ce_rp, test_data_ce_rp], axis = 0)
        train_data_ce_rp = dataset_ce_rp[:9640]
        val_data_ce_rp = dataset_ce_rp[9640:9640+120]
        test_data_ce_rp = dataset_ce_rp[9640+120:]
        print(f"→ CE-RP train shape: {train_data_ce_rp.shape}, val shape: {val_data_ce_rp.shape}, test shape: {test_data_ce_rp.shape}")

        # prepare data
        X_tr_ce_rp, y_tr_ce_rp = preparer.prepare(train_data_ce_rp)
        X_va_ce_rp, y_va_ce_rp = preparer.prepare(val_data_ce_rp)
        del train_data_ce_rp, val_data_ce_rp, test_data_ce_rp
        gc.collect()

        # ---- CE-CRP ----
        dataset_ce_crp = CE2dDataLoader(loadpath_ce_crp, dataset_name='CE-CRP')
        train_data_ce_crp, val_data_ce_crp = dataset_ce_crp.split_train()
        test_data_ce_crp = dataset_ce_crp.split_test()

        # concat and split again
        dataset_ce_crp = np.concatenate([train_data_ce_crp, val_data_ce_crp, test_data_ce_crp], axis = 0)
        train_data_ce_crp = dataset_ce_crp[:9640]
        val_data_ce_crp = dataset_ce_crp[9640:9640+120]
        test_data_ce_crp = dataset_ce_crp[9640+120:]
        print(f"→ CE-CRP train shape: {train_data_ce_crp.shape}, val shape: {val_data_ce_crp.shape}, test shape: {test_data_ce_crp.shape}")

        # prepare data
        X_tr_ce_crp, y_tr_ce_crp = preparer.prepare(train_data_ce_crp)
        X_va_ce_crp, y_va_ce_crp = preparer.prepare(val_data_ce_crp)
        del train_data_ce_crp, val_data_ce_crp, test_data_ce_crp
        gc.collect()

        # ---- CE-KH ----
        dataset_ce_kh = CE2dDataLoader(loadpath_ce_kh, dataset_name='CE-KH')
        train_data_ce_kh, val_data_ce_kh = dataset_ce_kh.split_train()
        test_data_ce_kh = dataset_ce_kh.split_test()

        # concat and split again
        dataset_ce_kh = np.concatenate([train_data_ce_kh, val_data_ce_kh, test_data_ce_kh], axis = 0)
        train_data_ce_kh = dataset_ce_kh[:9640]
        val_data_ce_kh = dataset_ce_kh[9640:9640+120]
        test_data_ce_kh = dataset_ce_kh[9640+120:]
        print(f"→ CE-KH train shape: {train_data_ce_kh.shape}, val shape: {val_data_ce_kh.shape}, test shape: {test_data_ce_kh.shape}")

        # prepare data
        X_tr_ce_kh, y_tr_ce_kh = preparer.prepare(train_data_ce_kh)
        X_va_ce_kh, y_va_ce_kh = preparer.prepare(val_data_ce_kh)
        del train_data_ce_kh, val_data_ce_kh, test_data_ce_kh
        gc.collect()

        # ---- CE-Gauss ----
        dataset_ce_gauss = CE2dDataLoader(loadpath_ce_gauss, dataset_name='CE-Gauss')
        train_data_ce_gauss, val_data_ce_gauss = dataset_ce_gauss.split_train()
        test_data_ce_gauss = dataset_ce_gauss.split_test()

        # concat and split again
        dataset_ce_gauss = np.concatenate([train_data_ce_gauss, val_data_ce_gauss, test_data_ce_gauss], axis = 0)
        train_data_ce_gauss = dataset_ce_gauss[:9640]
        val_data_ce_gauss = dataset_ce_gauss[9640:9640+120]
        test_data_ce_gauss = dataset_ce_gauss[9640+120:]
        print(f"→ CE-Gauss train shape: {train_data_ce_gauss.shape}, val shape: {val_data_ce_gauss.shape}, test shape: {test_data_ce_gauss.shape}")

        # prepare data
        X_tr_ce_gauss, y_tr_ce_gauss = preparer.prepare(train_data_ce_gauss)
        X_va_ce_gauss, y_va_ce_gauss = preparer.prepare(val_data_ce_gauss)
        del train_data_ce_gauss, val_data_ce_gauss, test_data_ce_gauss
        gc.collect()

        # Wrap into DatasetforDataloader
        train_data_ns_sines = DatasetforDataloader(X_tr_ns_sines, y_tr_ns_sines)
        val_data_ns_sines   = DatasetforDataloader(X_va_ns_sines, y_va_ns_sines)
        train_data_ns_gaussians = DatasetforDataloader(X_tr_ns_gaussians, y_tr_ns_gaussians)
        val_data_ns_gaussians   = DatasetforDataloader(X_va_ns_gaussians, y_va_ns_gaussians)
        train_data_ce_rp    = DatasetforDataloader(X_tr_ce_rp, y_tr_ce_rp)
        val_data_ce_rp      = DatasetforDataloader(X_va_ce_rp, y_va_ce_rp)
        train_data_ce_crp   = DatasetforDataloader(X_tr_ce_crp, y_tr_ce_crp)
        val_data_ce_crp     = DatasetforDataloader(X_va_ce_crp, y_va_ce_crp)
        train_data_ce_kh    = DatasetforDataloader(X_tr_ce_kh, y_tr_ce_kh)
        val_data_ce_kh      = DatasetforDataloader(X_va_ce_kh, y_va_ce_kh)
        train_data_ce_gauss = DatasetforDataloader(X_tr_ce_gauss, y_tr_ce_gauss)
        val_data_ce_gauss   = DatasetforDataloader(X_va_ce_gauss, y_va_ce_gauss)

        # Combine NS datasets: dataloader-2 (2 fields)
        train_ns = ConcatDataset([train_data_ns_sines, train_data_ns_gaussians])
        val_ns   = ConcatDataset([val_data_ns_sines,   val_data_ns_gaussians])

        # Combine CE datasets: dataloader-1 (4 fields)
        train_ce = ConcatDataset([train_data_ce_rp, train_data_ce_crp,
                                train_data_ce_kh, train_data_ce_gauss])
        val_ce   = ConcatDataset([val_data_ce_rp, val_data_ce_crp,
                                val_data_ce_kh, val_data_ce_gauss])

    return train_ns, train_ce, val_ns, val_ce

def build_dataloaders(train_ns, train_ce, val_ns, val_ce,
                    batch_size_ns=8, batch_size_ce=8,
                    num_workers=2):

    # --- Train loaders ---
    train_loader_ce = DataLoader(
        train_ce,
        batch_size=batch_size_ce,
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    train_loader_ns = DataLoader(
        train_ns,
        batch_size=batch_size_ns,
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    # --- Val loaders ---
    val_loader_ce = DataLoader(
        val_ce,
        batch_size=batch_size_ce,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )

    val_loader_ns = DataLoader(
        val_ns,
        batch_size=batch_size_ns,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )

    return train_loader_ns, train_loader_ce, val_loader_ns, val_loader_ce