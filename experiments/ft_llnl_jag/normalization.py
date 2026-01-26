import numpy as np
from src.utils.normalization import RevIN

# Normalize function
def normalize_params(params):
    for i in range(params.shape[1]):
        mu_params = params[:,i].mean()
        std_params = params[:,i].std()
        params[:,i] = (params[:,i] - mu_params) / std_params
    return params

# Denormalize function
def denormalize_params(norm_params, mu_params, std_params):
    return norm_params * std_params + mu_params

def normalizer(images, scalars, params, stats_dir=None):
    # ---- reorganize images ---

    # Data in (N, T, D, H, W, C, F) format
    E_1 = images[:, :, :, 0]
    E_2 = images[:, :, :, 1]
    E_3 = images[:, :, :, 2]
    E_4 = images[:, :, :, 3]

    # Organize data into components
    E_12 = np.stack((E_1, E_2), axis=-1)[..., np.newaxis]  # (N, H, W, C, F) -> (N, 64, 64, 2, 1)
    E_34 = np.stack((E_3, E_4), axis=-1)[..., np.newaxis]  # (N, H, W, C, F) -> (N, 64, 64, 2, 1)
    E_13 = np.stack((E_1, E_3), axis=-1)[..., np.newaxis]  # (N, 64, 64, 2, 1)
    E_24 = np.stack((E_2, E_4), axis=-1)[..., np.newaxis]  # (N, 64, 64, 2, 1)

    # Organize data into fields and components
    # type - 1 where E12 and E34 are groups as fields 
    data_1 = np.concatenate((E_12, E_34), axis=-1)  # (N, 64, 64, 2, 2)
    print(f"Organized data shape (type-1): {data_1.shape}")  # (N, 64, 64, 2, 2)

    # type - 2 where E13 and E24 are groups as fields
    data_2 = np.concatenate((E_13, E_24), axis=-1)  # (N, 64, 64, 2, 2)
    print(f"Organized data shape (type-2): {data_2.shape}")  # (N, 64, 64, 2, 2)

    data  = data_1  # choose one organization
    print(f"Final data shape (N,H,W,C,F): {data.shape}")  # (N, 64, 64, 2, 2)

    # ---- Prepare with RevIN ----

    # Bring data into (N, T, D, H, W, C, F) format
    data = data[:, np.newaxis, np.newaxis, :, :, :, :]  
    print(f"Data shape (N, T, D, H, W, F, C): {data.shape}")  

    # Reshape the data in UPTF-7 format
    data = data.transpose(0, 1, 6, 5, 2, 3, 4)  
    print(f"Data shape (N, T, F, C, D, H, W): {data.shape}")  

    # --- RevIN normalization ---

    # Call normalization function
    rev_icf = RevIN(stats_dir)

    # compute & normalize -> Needs data in UPTF-7 format
    rev_icf.compute_stats(data, prefix='stats_icf')
    dataset_icf_norm = rev_icf.normalize(data, prefix='stats_icf')
    print("Normalize dataset shape", dataset_icf_norm.shape)

    # Check round‐trip via denormalize
    tol_2 = 1e-4
    recovered = rev_icf.denormalize(dataset_icf_norm, prefix='stats_icf')
    print("Denormalized dataset shape", recovered.shape)

    max_error = 0.0
    for i in range(recovered.shape[0]):
        maxerror_i = np.max(np.abs(recovered[i] - data[i]))  # saving some memory
        max_error = max(maxerror_i, max_error)
    assert max_error < tol_2, "Denormalization did not perfectly recover original!"
    print("RevIN round-trip OK")
    del recovered

    # ---- Normalize parameters and scalars ----
    params_norm = normalize_params(params.copy())
    scalars_norm = normalize_params(scalars.copy())

    return dataset_icf_norm, scalars_norm, params_norm
