import os
import numpy as np
import h5py

def split_and_save_h5(raw_dir: str,
                      out_dir: str,
                      dataset_name = 'cfd1d',
                      train_frac: float = 0.9,
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

    for fname in files:
        base, _ = os.path.splitext(fname)
        path = os.path.join(raw_dir, fname)
        print(f"\nProcessing {fname} → base='{base}'")

        # --- load & stack ---
        with h5py.File(path, "r") as f5:
            vx       = f5["Vx"][...]       # (T, H, W)
            density  = f5["density"][...]  # (T, H, W)
            pressure = f5["pressure"][...] # (T, H, W)
            data = np.stack([vx, density, pressure], axis=-1).astype(np.float32)
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
                fo.create_dataset("density",  data=arr[..., 1], compression="gzip")
                fo.create_dataset("pressure", data=arr[..., 2], compression="gzip")
            print(f"    saved {split_name} → {os.path.basename(out_path)}")

        # --- write train, val, test for this file ---
        write_split("train", train_data)
        write_split("val",   val_data)
        write_split("test",  test_data)
    
class CFD1dDataLoader:
    def __init__(self, data_path, dataset_name='CFD1d'):
        self.data_path = data_path
        self.dataset_name = dataset_name
        
    def load_split(self, split, num_files, sims):
        split_path = os.path.join(self.data_path, split)
        files = sorted([f for f in os.listdir(split_path)
                 if f.endswith('.h5') or f.endswith('.hdf5')])
        if not files:
            raise FileNotFoundError(f"No HDF5 files found in {files}")
        
        # select the number of files
        selected_files = files[:num_files] if num_files!=None else files
        
        # select the number of simulations
        sims = slice(None) if sims is None else sims
        
        print('Loading CFD-1D (compressible) data from PDEBench...')
        arrays = []
        for fname in selected_files:
            path = os.path.join(split_path, fname)
            print(f'Importing: {path}...')
            with h5py.File(path, 'r') as f5:
                # Directly read your three main fields
                vx       = f5['Vx'][sims, ...]        # shape (10000, 101, 1024)
                density  = f5['density'][sims, ...]   # shape (10000, 101, 1024)
                pressure = f5['pressure'][sims, ...]  # shape (10000, 101, 1024)

                # Stack the fields in the last dim: resulting shape: (10000, 101, 1024, 3)
                data = np.stack([vx, density, pressure], axis=-1).astype(np.float32)
                print(f" Shape of the imported data: {data.shape}")
            arrays.append(data)
        
        # concatenate all files along the sample axis (axis=0)
        arrays = np.concatenate(arrays, axis=0)
        print(f"Shape of the concat data ({len(selected_files)} files): {arrays.shape}")
        
        return arrays
    
    def inflate_array(self, arr, axes):
        # arr must be 4-D ⇒ (S, T, W, F)
        for ax in sorted(axes):
            arr = np.expand_dims(arr, axis = ax)
        # now shape is (S, T, 1, 1, W, 1, F)
        # repeat channel-axis=axes[1] to size 3
        # arr = np.repeat(arr, repeats=3, axis = axes[-1])
        # final shape (S, T, 1, H, W, 3, F)
        return arr

    def split_train(self, num_files = None, sims = None):
        n_total_1 = 8000 * num_files
        n_total_2 = 1000 * num_files
        
        if sims != None:
            # select random sims
            rng1 = np.random.default_rng(1234)
            rng2 = np.random.default_rng(4236)
            sims1 = rng1.choice(n_total_1, sims, replace=False)
            sims2 = rng2.choice(n_total_2, sims, replace=False)
            sims1 = np.sort(sims1)
            sims2 = np.sort(sims2)
        
        else:
            sims1 = None
            sims2 = None
        
        print(f"[{self.dataset_name}] Importing training data...")
        train = self.load_split('train', num_files = num_files, sims = sims1)
        train = self.inflate_array(train, axes=[2, 3, 5])

        print(f"[{self.dataset_name}] Importing validation data...")
        val   = self.load_split('val', num_files = num_files, sims = sims1)
        val   = self.inflate_array(val,   axes=[2, 3, 5])

        return train, val

    def split_test(self, num_files = None, sims = None):
        n_total_3 = 1000 * num_files
        
        if num_files and sims != None:
            # select random sims
            rng3 = np.random.default_rng(2345) 
            sims3 = rng3.choice(n_total_3, sims, replace=False)
            sims3 = np.sort(sims3)
        
        else:
            sims3 = None
        
        print(f"[{self.dataset_name}] Importing test data...")
        test = self.load_split('test', num_files = num_files, sims = sims3)
        test = self.inflate_array(test, axes=[2, 3, 5])
        return test