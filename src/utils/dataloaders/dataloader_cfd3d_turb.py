import os
import numpy as np
import h5py

def split_and_save_h5(raw_dir: str,
                      out_dir: str,
                      select_nfiles: int,
                      dataset_name = 'cfd3d_turb',
                      train_frac: float = 0.8,
                      rand = True):
    """
    For each .h5/.hdf5 file in raw_dir:
      1) Load its data (Vx, density, pressure) and stack → (T, H, W, 3)
      2) Shuffle & split into train/val (val==test)
      3) Save three files:
         {orig_basename}_train.h5,
         {orig_basename}_val.h5,
         {orig_basename}_test.h5
    """
    os.makedirs(out_dir, exist_ok=True)

    # find all HDF5 files
    files = [f for f in os.listdir(raw_dir)
             if f.endswith(".h5") or f.endswith(".hdf5")]
    if not files:
        raise FileNotFoundError(f"No HDF5 files in {raw_dir}")

    for fname in files[0:select_nfiles]:
        print(f'{files[0:select_nfiles]}')
        base, _ = os.path.splitext(fname)
        path = os.path.join(raw_dir, fname)
        print(f"\nProcessing {fname} → base='{base}'")

        # --- load & stack ---
        with h5py.File(path, "r") as f5:
            vx       = f5['Vx'][...]        
            vy       = f5['Vy'][...]        
            vz       = f5['Vz'][...]        
            density  = f5['density'][...]   
            pressure = f5['pressure'][...]  
            data = np.stack([vx, vy, vz, density, pressure], axis=-1).astype(np.float32)
        T = data.shape[0]
        print(f"  loaded {T} frames of shape {data.shape[1:]}")

        # --- shuffle & split ---
        if rand:
            np.random.seed(42)
            idx = np.random.permutation(T) 
        else:
            # --- fixed 80/10/10 split without randomness ---
            idx = np.arange(T)
            
        train_end = int(train_frac * T)
        val_end   = int((train_frac + 0.1) * T)
        train_data = data[idx[:train_end]]
        val_data   = data[idx[train_end:val_end]]
        test_data  = data[idx[val_end:]]

        print(f"  → {train_data.shape[0]} train / "
              f"{val_data.shape[0]} val / " 
              f"{test_data.shape[0]} test")

        # --- helper to write one split for this file ---
        def write_split(split_name, arr):
            out_folder = os.path.join(out_dir, split_name)
            out_path = os.path.join(out_folder, f"{base}_{split_name}.h5")
            with h5py.File(out_path, "w") as fo:
                fo.create_dataset("Vx",       data=arr[..., 0], compression="gzip")
                fo.create_dataset("Vy",       data=arr[..., 1], compression="gzip")
                fo.create_dataset("Vz",       data=arr[..., 2], compression="gzip")
                fo.create_dataset("density",  data=arr[..., 3], compression="gzip")
                fo.create_dataset("pressure", data=arr[..., 4], compression="gzip")
            print(f"    saved {split_name} → {os.path.basename(out_path)}")

        # --- write train, val, test for this file ---
        write_split("train", train_data)
        write_split("val",   val_data)
        write_split("test",  test_data)
            
class CFD3dTurbDataLoader:
    def __init__(self, data_path, dataset_name='CFD3d-Turb'):
        self.data_path = data_path
        self.dataset_name = dataset_name
    
    def load_split(self, split):
        split_path = os.path.join(self.data_path, split)
        files = [f for f in os.listdir(split_path)
                 if f.endswith('.h5') or f.endswith('.hdf5')]

        if not files:
            raise FileNotFoundError(f"No HDF5 files found in {split_path}")
        
        print(f'Number of files in {split_path}: {len(files)}')
        
        arrays = []
        for fname in files:
            path = os.path.join(split_path, fname)
            with h5py.File(path, 'r') as f5:
                # Directly read your three main fields
                vx       = f5['Vx'][...]        
                vy       = f5['Vy'][...]        
                vz       = f5['Vz'][...]        
                density  = f5['density'][...]   
                pressure = f5['pressure'][...]  
                
                # expand all the fields
                vx =  np.expand_dims(vx, axis = (5,6))                # (600,21,64,64,64,1,1)
                vy =  np.expand_dims(vy, axis = (5,6))                # (600,21,64,64,64,1,1)
                vz =  np.expand_dims(vz, axis = (5,6))                # (600,21,64,64,64,1,1)
                density = np.expand_dims(density, axis = (5,6))       # (600,21,64,64,64,1,1)
                pressure = np.expand_dims(pressure, axis = (5,6))     # (600,21,64,64,64,1,1)
                
                # make vector fields
                v = np.concatenate((vx, vy, vz), axis = 5)            # (600,21,64,64,64,3,1)
                density = np.repeat(density, repeats = 3, axis = 5)   # (600,21,64,64,64,3,1)
                pressure = np.repeat(pressure, repeats = 3, axis = 5) # (600,21,64,64,64,3,1)
                print(f'Expanded data: V:{v.shape}, density:{density.shape}, pressure:{pressure.shape}')
                
                # Stack the fields in the last dim
                data = np.concatenate((v, density, pressure), axis = 6).astype(np.float32) # (100,21,128,128,128,3,3)
                print("Shape of the data", data.shape)
            arrays.append(data)

        # if you have multiple HDF5s per split, stack them; otherwise take the one
        if len(arrays) > 1:
            arrays = np.concatenate(arrays, axis=0)
        else:
            arrays = arrays[0]

        return arrays

    def split_train(self):
        print(f"[{self.dataset_name}] Importing training data...")
        train = self.load_split('train')

        print(f"[{self.dataset_name}] Importing validation data...")
        val   = self.load_split('val')

        return train, val

    def split_test(self):
        print(f"[{self.dataset_name}] Importing test data...")
        test = self.load_split('test')
        return test