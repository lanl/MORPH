import os
import random
import zipfile
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
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

def _set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Optional (more deterministic, sometimes slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Normalize function
def normalize_params(params):
    minval_list, maxval_list = [], []
    for i in range(params.shape[1]):
        minval = params[:,i].min()
        maxval = params[:,i].max()
        params[:,i] = (params[:,i] - minval) / (maxval - minval)
        minval_list.append(minval)
        maxval_list.append(maxval)
        print(f'Parameter {i+1}: min={minval}, max={maxval}')
    return params, minval_list, maxval_list

def show_img(ax, img):
    # handle HxW, HxWx1 (grayscale), or HxWx3 (RGB)
    if img.ndim == 3 and img.shape[-1] == 1:
        ax.imshow(img[..., 0], cmap='gray')
    elif img.ndim == 2:
        ax.imshow(img, cmap='gray')
    else:
        ax.imshow(img)
    ax.axis('off')

def dataloaders(dataset_dir, results_dir, data_frac, batch_size = 8, seed=123):
    _set_all_seeds(seed)

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
    print(f'imagesA0_norm shape: {imagesA0_norm.shape}, imagesS0_norm shape: {imagesS0_norm.shape}')

    img_merged = np.concatenate([imagesS0_norm,imagesA0_norm],axis=3) # along field dims
    img_merged_nfhw = img_merged.transpose(0,3,1,2)  # to NCHW format
    img_merged_uptf = img_merged_nfhw[:,np.newaxis,:,np.newaxis,np.newaxis,:,:]  # add channel for time step
    print("Shape of the dataset (N, T, F, C, D, H, W) ==>>", img_merged_uptf.shape)

    # Normalize the parameters 
    params_norm, minval_list, maxval_list = normalize_params(params.copy())

    # define dataloader
    full_dataset = DatasetforDataloader(img_merged_uptf, params_norm)

   # Fixed split (90/10/0) with seeded generator => test is constant
    N = len(full_dataset)
    train_size = int(0.85 * N)
    val_size = int(0.05 * N)
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

    # print length of dataloaders
    print(f"Number of training samples: {len(dl_train.dataset)}")
    print(f"Number of validation samples: {len(dl_val.dataset)}")
    print(f"Number of test samples: {len(dl_test.dataset)}")

    return img_merged_uptf, params_norm, minval_list, maxval_list, dl_train, dl_val, dl_test