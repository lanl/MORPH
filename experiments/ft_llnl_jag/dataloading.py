import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from visualization import data_visualizer
from sensitivity_analysis import sensitivity_analyser
from normalization import normalizer

# dataset class
class DatasetforDataloader(Dataset):
    def __init__(self,X,p,s):
        self.X = X
        self.s = s
        self.p = p

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,i):
        # create a tuple
        return self.X[i], self.p[i], self.s[i]
    
def dataloaders(current_dir, model_dir, results_dir, data_frac, batch_size = 8):
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

    print("=== Data Visualization ===")
    images_reshape = images.reshape(images.shape[0], 64, 64, 4).astype(np.float32)
    print(f"Reshaped images: {images_reshape.shape}")  # (N, 64, 64, 4)

    data_visualizer(images_reshape, scalars, params, save_dir=results_dir)
    print(f"Data visualization saved to {results_dir}")

    # Sensitivity analysis
    print("=== Sensitivity Analysis ===")
    sensitivity_analyser(images_reshape, scalars, params, save_dir=results_dir)
    print(f"Sensitivity analysis results saved to {results_dir}")

    # data normalization
    print("=== Data Normalization ===")
    images_norm, scalars_norm, params_norm = normalizer(images_reshape, scalars, params, stats_dir=model_dir)
    
    #  some stats before normalization
    print(f"=== Stats before normalization ===")
    min_images, max_images = images.min(), images.max()
    mean_images, std_images = images.mean(), images.std()
    print(f"images: min {min_images}, max {max_images}, mean {mean_images}, std {std_images}")

    min_params, max_params = params.min(), params.max()
    mean_params, std_params = params.mean(), params.std()
    print(f"params: min {min_params}, max {max_params}, mean {mean_params}, std {std_params}")

    min_scalars, max_scalars = scalars.min(), scalars.max()
    mean_scalars, std_scalars = scalars.mean(), scalars.std()
    print(f"scalars: min {min_scalars}, max {max_scalars}, mean {mean_scalars}, std {std_scalars}")

    # stats after normalization
    print(f"=== Stats after normalization ===")
    min_images, max_images = images_norm.min(), images_norm.max()
    mean_images, std_images = images_norm.mean(), images_norm.std()
    print(f"images: min {min_images}, max {max_images}, mean {mean_images}, std {std_images}")

    min_params, max_params = params_norm.min(), params_norm.max()
    mean_params, std_params = params_norm.mean(), params_norm.std()
    print(f"params: min {min_params}, max {max_params}, mean {mean_params}, std {std_params}")

    min_scalars, max_scalars = scalars_norm.min(), scalars_norm.max()
    mean_scalars, std_scalars = scalars_norm.mean(), scalars_norm.std()
    print(f"scalars: min {min_scalars}, max {max_scalars}, mean {mean_scalars}, std {std_scalars}")

    # select fraction of data
    print(f'=== Preparing data loaders with {data_frac*100}% of data ===')
    dataset_size = int(images_norm.shape[0] * data_frac)  # adjust as needed
    data_idx = np.random.choice(images_norm.shape[0], dataset_size, replace=False)

    # Dataset of dataloader
    X = images_norm[data_idx]
    params = params_norm[data_idx]
    scalars = scalars_norm[data_idx]
    full_dataset = DatasetforDataloader(X, params, scalars)

    # define the splits (80/10/10)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Split the dataset 
    train, val, test = torch.utils.data.random_split(full_dataset,[train_size, val_size, test_size])

    # Dataloaders
    BATCH_SIZE = batch_size
    dataloader_train = DataLoader(train, batch_size = BATCH_SIZE, shuffle=True)
    dataloader_val = DataLoader(val, batch_size = BATCH_SIZE, shuffle=False)
    dataloader_test = DataLoader(test, batch_size = BATCH_SIZE, shuffle=False)
    print(f"Number of training samples: {len(dataloader_train)}")
    print(f"Number of validation samples: {len(dataloader_val)}")
    print(f"Number of test samples: {len(dataloader_test)}")

    return images_norm, params_norm, scalars_norm, dataloader_train, dataloader_val, dataloader_test