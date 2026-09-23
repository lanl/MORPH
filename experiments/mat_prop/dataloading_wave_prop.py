import os
import zipfile
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download
import pandas as pd
from morph_pde.utils.importdataset import ImportImgData
import matplotlib.pyplot as plt

# dataset class
class DatasetforDataloader(Dataset):
    def __init__(self,X,y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,i):
        # create a tuple
        return self.X[i], self.y[i]

# Normalize function
def normalize_params(params):
    for i in range(params.shape[1]):
        minval = params[:,i].min()
        maxval = params[:,i].max()
        params[:,i] = (params[:,i] - minval) / (maxval - minval)
        print(f'Parameter {i+1}: min={minval}, max={maxval}')
    return params

# Denormalize function
def denormalize_params(norm_params, min_params, max_params):
    return norm_params * (max_params - min_params) + min_params

def show_img(ax, img):
    # handle HxW, HxWx1 (grayscale), or HxWx3 (RGB)
    if img.ndim == 3 and img.shape[-1] == 1:
        ax.imshow(img[..., 0], cmap='gray')
    elif img.ndim == 2:
        ax.imshow(img, cmap='gray')
    else:
        ax.imshow(img)
    ax.axis('off')

def dataloaders(dataset_dir, results_dir, data_frac, batch_size = 8):

    dataset_path = os.path.join(dataset_dir, "PGVR", "Dataset-2_GEN")
    if os.path.exists(dataset_path):
        print(f"Dataset already exists in {dataset_path}. Skipping download.")

    else:
        REPO_ID = "mahindrautela/WavePropagation_Composites"
        sub_folder = "Dataset-2_GEN"   # or "Dataset-1_COMM"

        zip_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=f"PGVR/{sub_folder}/dataset.zip",
            local_dir=dataset_dir,
            local_dir_use_symlinks=False,
        )

        # unzip
        extract_dir = os.path.join(dataset_dir, "PGVR", sub_folder)
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

    # --- load images and parameters ---
    S0_loc = os.path.join(dataset_dir, "PGVR", "Dataset-2_GEN", "GWrepresentations", "S0")
    A0_loc = os.path.join(dataset_dir, "PGVR", "Dataset-2_GEN", "GWrepresentations", "A0")
    params_loc = os.path.join(dataset_dir, "PGVR", "Dataset-2_GEN", "Labels","GeneratedMaterialProperties_1.txt")

    df = pd.read_csv(params_loc, header=None)
    rho = df.iloc[:,0:1]     
    E1 = df.iloc[:,1:2]     
    E2 = df.iloc[:,2:3]     
    G12 = df.iloc[:,3:4]    
    v12 = df.iloc[:,4:5]
    v23 = df.iloc[:,5:6]    
    dfn = pd.concat([rho,E1,E2,G12,v12,v23],axis=1)
    params = dfn.to_numpy().astype("float32")
    print("Shape of the parameters", params.shape)

    imagesA0 = ImportImgData.load_A0_images(dfn, A0_loc, imsize=128)
    imagesS0 = ImportImgData.load_S0_images(dfn, S0_loc, imsize=128)

    # ---  data visualization ---
    # pick 5 random indices from the first 100 (or fewer if dataset smaller)
    N = min(imagesA0.shape[0], imagesS0.shape[0], 100)
    idx = np.random.choice(N, 5, replace=False)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for j, k in enumerate(idx):
        show_img(axes[0, j], imagesA0[k]); axes[0, j].set_title(f"A0 #{k}")
        show_img(axes[1, j], imagesS0[k]); axes[1, j].set_title(f"S0 #{k}")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "sample_images.png"), dpi=300)

    # use whichever you created earlier (dfn_1 from your snippet, or dfn)
    names = ["rho", "E1", "E2", "G12", "v12", "v23"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
    for i, ax in enumerate(axes.ravel()):
        ax.hist(params[:, i], bins=50)
        ax.set_title(names[i])
        ax.set_xlabel("Value")
        if i % 3 == 0:
            ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "parameter_histograms.png"), dpi=300)

    # --- data normalization ---
    # Normalize the images to [0,1]
    imagesA0_norm = imagesA0.astype("float32") / 255 # normalize to [0,1]
    imagesS0_norm = imagesS0.astype("float32") / 255 # normalize to [0,1]

    img_merged = np.concatenate([imagesS0_norm,imagesA0_norm],axis=3)
    img_merged_nchw = img_merged.transpose(0,3,1,2)  # to NCHW format
    img_merged_uptf = img_merged_nchw[:,np.newaxis,np.newaxis,:,np.newaxis,:,:]  # add channel for time step
    print("Shape of the dataset (N, T, F, C, D, H, W) ==>>", img_merged_uptf.shape)

    # Normalize the parameters 
    params_norm = normalize_params(params.copy())

    # --- select fraction of data ---
    print(f'=== Preparing data loaders with {data_frac*100}% of data ===')
    dataset_size = int(img_merged_uptf.shape[0] * data_frac)  # adjust as needed
    data_idx = np.random.choice(img_merged_uptf.shape[0], dataset_size, replace=False)

    X = img_merged_uptf[data_idx]
    y = params_norm[data_idx]
    full_dataset = DatasetforDataloader(X, y)

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

    return X, y, dataloader_train, dataloader_val, dataloader_test