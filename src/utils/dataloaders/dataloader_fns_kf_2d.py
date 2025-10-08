import os
import numpy as np
import h5py

def split_and_save_h5(raw_h5_loadpath, savepath, dataset_name='FNS-KF', 
                      train_frac=0.8, rand=True):
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

        # --- read full dataset once (unchanged pattern) ---
        with h5py.File(raw_path, 'r') as f_in:
            vel = f_in['solution'][:]  # shape: (N, 21, 2, 128, 128)
            file_attrs = dict(f_in.attrs)
            ds_attrs   = dict(f_in['solution'].attrs)

        N = len(vel)
        print(f'Total traj: {N}')
        base = os.path.splitext(os.path.basename(raw_path))[0]
        
        # --- shuffle & split ---
        if rand:
            np.random.seed(42)
            idx = np.random.permutation(N) 
        else:
            idx = np.arange(N)
            
        train_end = int(train_frac * N)
        val_end   = int((train_frac + 0.1) * N)
        
        splits = {
            'train': idx[:train_end],
            'val':   idx[train_end:val_end],
            'test':  idx[val_end:],
        }

        # 4) write out each split-file
        for split, idxs in splits.items():
            out_dir = os.path.join(savepath, split)
            os.makedirs(out_dir, exist_ok=True)

            # --- keep the SAME filename as input ---
            out_fname = f"{base}_{split}.h5"
            out_path  = os.path.join(out_dir, out_fname)

            # --- write a single 'solution' dataset with the split slice ---
            with h5py.File(out_path, 'w') as f_out:
                dset = f_out.create_dataset('solution', data=vel[idxs])
                # (optional but useful) preserve attrs
                for k, v in file_attrs.items():
                    f_out.attrs[k] = v
                for k, v in ds_attrs.items():
                    dset.attrs[k] = v

            print(f"[{dataset_name}] Saved {len(idxs)} → {out_path}")

            
class FNSKF2dDataLoader:
    def __init__(self, data_path, dataset_name='FNS-KF'):
        self.data_path = data_path
        self.dataset_name = dataset_name
        
    def inflate_array(self, arr, axes):
        for ax in sorted(axes):
            arr = np.expand_dims(arr, axis=ax)
        return arr
    
    def load_split(self, split):
        split_path = os.path.join(self.data_path, split)
        # only .h5 and .hdf5 files now
        files = [f for f in os.listdir(split_path)
                 if f.endswith('.h5') or f.endswith('.hdf5')]
        arrays = []

        for fname in files:
            path = os.path.join(split_path, fname)
            with h5py.File(path, 'r') as f5:
                # read fields
                vel = f5['solution'][:]   # (20000,21,2,128,128)
                vel = vel.transpose(0,1,3,4,2) # (20000,21,128,128,2)
                
                # concat A and B
                data = vel # (160,1001,128,128,2)
                print(f'Shape of data: {data.shape}')
                
            arrays.append(data.astype('float32'))

        if arrays:
            return np.concatenate(arrays, axis=0)
        else:
            return np.empty((0,), dtype='float32')

    def split_train(self):
        print(f"[{self.dataset_name}] Importing training data...")
        train_data = self.load_split('train')
        train_data = self.inflate_array(train_data, axes=[2,6]) # add 'D' and 'F'
        
        print(f"[{self.dataset_name}] Importing validation data...")
        val_data = self.load_split('val')
        val_data = self.inflate_array(val_data, axes=[2,6])
        return train_data, val_data

    def split_test(self):
        print(f"[{self.dataset_name}] Importing test data...")
        test_data = self.load_split('test')
        test_data = self.inflate_array(test_data, axes=[2,6])
        return test_data
