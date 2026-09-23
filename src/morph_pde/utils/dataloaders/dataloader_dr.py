import os
import numpy as np
import h5py

def split_and_save_h5(raw_h5_loadpath, savepath, dataset_name='DR', train_frac=0.8, rand = True):
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

    # 2) process each file independently
    for fname in raw_files:
        raw_path = os.path.join(raw_h5_loadpath, fname)
        print(f"Processing file: {fname}...")
        with h5py.File(raw_path, 'r') as f_in:
            keys = sorted(f_in.keys())
            print(f"\n>>> Keys in {fname}:", list(f_in.keys()))

        N    = len(keys)
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

        # 4) write out each split-file
        for split, idxs in splits.items():
            out_dir = os.path.join(savepath, split)
            os.makedirs(out_dir, exist_ok=True)

            out_fname = f"{base}_{split}.h5"
            out_path  = os.path.join(out_dir, out_fname)

            with h5py.File(raw_path, 'r') as f_in, \
                 h5py.File(out_path,  'w') as f_out:
                for i in idxs:
                    key = keys[i]
                    f_in.copy(key, f_out)

            print(f"[{dataset_name}] Saved {len(idxs)} → {out_path}")
            
class DR2DDataLoader:
    def __init__(self, data_path, dataset_name='DR'):
        self.data_path = data_path
        self.dataset_name = dataset_name
        
    def inflate_array(self, arr, axes):
        # arr must be 5-D ⇒ (S, T, H, W, F)
        for ax in sorted(axes):
            arr = np.expand_dims(arr, axis=ax)
        # now shape is (S, T, 1, H, W, 1, F)
        # repeat channel-axis=axes[1] to size 3
        # arr = np.repeat(arr, repeats=3, axis=axes[1])
        # final shape (S, T, 1, H, W, 3, F)
        return arr
    
    def load_split(self, split):
        split_path = os.path.join(self.data_path, split)
        files = [f for f in os.listdir(split_path)
                 if f.endswith('.h5') or f.endswith('.hdf5')]

        if not files:
            raise FileNotFoundError(f"No HDF5 files found in {split_path}")

        arrays = []
        for fname in files:
            fullpath = os.path.join(split_path, fname)
            with h5py.File(fullpath, 'r') as f5:
                keys = sorted(f5.keys())
                # exactly as in your standalone code:
                sample_shape = f5[keys[0]]['data'].shape    # e.g. (T, H, W, F)
                S = len(keys)
                all_data = np.zeros((S, *sample_shape), dtype=np.float32)
                for i, k in enumerate(keys):
                    all_data[i] = f5[k]['data'][...]
            arrays.append(all_data)

        # if you have multiple HDF5s per split, stack them; otherwise take the one
        if len(arrays) > 1:
            data = np.concatenate(arrays, axis=0)
        else:
            data = arrays[0]

        return data

    def split_train(self):
        print(f"[{self.dataset_name}] Importing training data...")
        train = self.load_split('train')
        train = self.inflate_array(train, axes=[2, 5])

        print(f"[{self.dataset_name}] Importing validation data...")
        val   = self.load_split('val')
        val   = self.inflate_array(val,   axes=[2, 5])

        return train, val

    def split_test(self):
        print(f"[{self.dataset_name}] Importing test data...")
        test = self.load_split('test')
        test = self.inflate_array(test, axes=[2, 5])
        return test
