import os
import numpy as np
import h5py

def split_and_save_h5(raw_h5_loadpath, savepath, selected_idx = 0, dataset_name='BE1d', 
                      train_frac=0.8, rand = True):
    """
    Split every .h5/.hdf5 in raw_h5_dir into train/val/test under self.data_path,
    naming each “<orig_base>_<split>.h5”
    """
    print("Spliting the file into train/val/test...")
    # 1) find all raw files
    raw_files = [f for f in os.listdir(raw_h5_loadpath)
                 if f.endswith('.h5') or f.endswith('.hdf5')]
    print(f"Found {len(raw_files)} HDF5 files in '{raw_h5_loadpath}'")
    
    if not raw_files:
        raise FileNotFoundError(f"No HDF5 files found in {raw_h5_loadpath}")

    # process each file independently
    print(f'Selecting only 1 file: {raw_files[selected_idx]}')
    fname = raw_files[selected_idx]
    raw_path = os.path.join(raw_h5_loadpath, fname)
    
    with h5py.File(raw_path, 'r') as f_in:
        data = f_in['tensor'][...] 
            
        N    = data.shape[0]
        base = os.path.splitext(os.path.basename(raw_path))[0]
        
        # --- shuffle & split ---
        if rand:
            np.random.seed(42)
            idx = np.random.permutation(N) 
        else:
            # --- fixed 80/10/10 split without randomness ---
            idx = np.arange(N)
            
        train_end = int(train_frac * N)
        val_end   = int((train_frac + 0.1) * N)
        
        splits  = {
            'train': idx[:train_end],
            'val':   idx[train_end:val_end],
            'test':  idx[val_end:],   # or adjust val/test split as you like
        }

        for split_name, idxs in splits.items():
            print(f"{split_name:>5}: {len(idxs)} samples")

        # 4) write each split
        for split_name, idxs in splits.items():
            out_dir = os.path.join(savepath, split_name)
            os.makedirs(out_dir, exist_ok=True)
    
            out_fname = f"{base}_{split_name}.h5"
            out_path  = os.path.join(out_dir, out_fname)
    
            with h5py.File(out_path, 'w') as f_out:
                f_out.create_dataset('tensor', data=data[idxs], dtype=data.dtype)
    
            print(f"[{dataset_name}] Saved {len(idxs)} → {out_path}")
            
class BE1DDataLoader:
    def __init__(self, data_path, dataset_name='BE1d'):
        self.data_path = data_path
        self.dataset_name = dataset_name
        
    def inflate_array(self, arr, axes):
        for ax in sorted(axes):
            arr = np.expand_dims(arr, axis=ax)
        return arr
    
    def load_split(self, selected_idx, split):
        split_path = os.path.join(self.data_path, split)
        files = [f for f in os.listdir(split_path)
                 if f.endswith('.h5') or f.endswith('.hdf5')]

        if not files:
            raise FileNotFoundError(f"No HDF5 files found in {split_path}")

        fname = files[selected_idx]
        print(f'Selecting only 1 file: {fname}')
        
        fullpath = os.path.join(split_path, fname)
        with h5py.File(fullpath, 'r') as f5:
            data = f5['tensor'][...] 

        return data

    def split_train(self, selected_idx):
        print(f"[{self.dataset_name}] Importing training data...")
        train = self.load_split(selected_idx, 'train')
        train = self.inflate_array(train, axes=[2,3,5,6])

        print(f"[{self.dataset_name}] Importing validation data...")
        val   = self.load_split(selected_idx, 'val')
        val   = self.inflate_array(val,   axes=[2,3,5,6])

        return train, val

    def split_test(self, selected_idx):
        print(f"[{self.dataset_name}] Importing test data...")
        test = self.load_split(selected_idx, 'test')
        test = self.inflate_array(test, axes=[2,3,5,6])
        return test
