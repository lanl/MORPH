import os
import numpy as np
import h5py

from src.utils.main_process_ddp import is_main_process

def split_and_save_h5(raw_h5_loadpath, savepath, dataset_name='CE', 
                      train_frac=0.8, val_frac=0.1, 
                      var_name = 'data', rand=True):
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
            data = f_in[var_name][:]  # shape: (N, 21, 2, 128, 128)
            file_attrs = dict(f_in.attrs)
            ds_attrs   = dict(f_in[var_name].attrs)

        N = len(data)
        print(f'Total traj: {N}')
        base = os.path.splitext(os.path.basename(raw_path))[0]
        
        # --- shuffle & split ---
        if rand:
            np.random.seed(42)
            idx = np.random.permutation(N) 
        else:
            idx = np.arange(N)
            
        train_end = int(train_frac * N)
        val_end   = int((train_frac + val_frac) * N)
        
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
                dset = f_out.create_dataset(var_name, data=data[idxs])
                # (optional but useful) preserve attrs
                for k, v in file_attrs.items():
                    f_out.attrs[k] = v
                for k, v in ds_attrs.items():
                    dset.attrs[k] = v

            print(f"[{dataset_name}] Saved {len(idxs)} → {out_path}")
            
class CE2dDataLoader:
    def __init__(self, data_path, var_name = 'data', dataset_name='CE'):
        self.data_path = data_path
        self.var_name = var_name
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
        
        if is_main_process():
            print(f"[{self.dataset_name}] Loading {len(files)} files from {split_path}...")
            
        arrays = []

        for fname in files:
            path = os.path.join(split_path, fname)
            # print(f'Current file: {path}')
            with h5py.File(path, 'r') as f5:
                # read data
                data = f5[self.var_name][:]   # (20000,21,5,128,128)
                data = data.transpose(0,1,3,4,2) # (20000,21,128,128,5)
                
                # read fields (field - 5 is energy, not taking it)
                dens = data[:,:,:,:,0:1]  # rho: (10000,21,128,128,1)
                vel = data[:,:,:,:,1:3]  # (Vx,Vy): (10000,21,128,128,2)
                pres = data[:,:,:,:,3:4] # P: (10000,21,128,128,2)
                
                # expand all the fields
                dens = np.expand_dims(dens, axis=5) # rho: (10000,21,128,128,1,1)          
                vel = np.expand_dims(vel, axis=5) # (Vx,Vy): (10000,21,128,128,2,1)
                pres = np.expand_dims(pres, axis=(5)) # pres: (10000,21,128,128,1,1)   
                
                # structure into the concat shape
                dens = np.repeat(dens, repeats=2, axis=4)       # (10000,21,128,128,2,1)      
                pres = np.repeat(pres, repeats=2, axis=4)       # (10000,21,128,128,2,1)
                # print(f'dens:{dens.shape}, vel:{vel.shape}, pres:{pres.shape}')

                # concat fields (20000,21,128,128,2,3)  
                data = np.concatenate((vel, dens, pres), axis=5).astype(np.float32)
                # print(f'Shape of loaded data: {data.shape}')
            arrays.append(data.astype('float32'))

        if arrays:
            arrays = np.concatenate(arrays, axis=0)
            if is_main_process():
                print(f"[{self.dataset_name}] Shape of concatenated arrays: {arrays.shape}")
            return arrays
        else:
            return np.empty((0,), dtype='float32')

    def split_train(self):
        if is_main_process():
            print(f"[{self.dataset_name}] Importing training data...")
        train_data = self.load_split('train')
        train_data = self.inflate_array(train_data, axes=[2]) # add 'D'
        if is_main_process():
            print(f"[{self.dataset_name}] Training data shape after inflation: {train_data.shape}")
        
        if is_main_process():
            print(f"[{self.dataset_name}] Importing validation data...")
        val_data = self.load_split('val')
        val_data = self.inflate_array(val_data, axes=[2])
        if is_main_process():
            print(f"[{self.dataset_name}] Validation data shape after inflation: {val_data.shape}")

        return train_data, val_data

    def split_test(self):
        if is_main_process():
            print(f"[{self.dataset_name}] Importing test data...")
        test_data = self.load_split('test')
        test_data = self.inflate_array(test_data, axes=[2])
        if is_main_process():
            print(f"[{self.dataset_name}] Test data shape after inflation: {test_data.shape}")
        return test_data
