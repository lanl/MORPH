import os
import numpy as np
import h5py

class HEAT2dDataLoader:
    def __init__(self, data_path, dataset_name='HEAT2d'):
        self.data_path = data_path
        self.dataset_name = dataset_name
    
    def load_split(self, split, num_files=None, fields_to_use=None):
        split_path = os.path.join(self.data_path, split)
        # only .h5 and .hdf5 files now
        files = [f for f in os.listdir(split_path)
                 if f.endswith('.h5') or f.endswith('.hdf5')]
        print(f"[{self.dataset_name}] Found {len(files)} files in {split_path}")

        # select the number of files
        if num_files is not None:
            files = files[:num_files] if num_files!=None else files
        print(f"[{self.dataset_name}-{split}] Loading {len(files)} files from {split_path} …")

        # based on channels names (all densities including av_temperature)
        if fields_to_use is not None:
            use_idx = np.asarray(fields_to_use, dtype=np.int64)
        else:
            use_idx = None

        arrays = []
        for fname in files:
            path = os.path.join(split_path, fname)
            with h5py.File(path, 'r') as f5:
                arr = f5["data"][...][:, :, use_idx, :, :]  # (N,2,F,H,W)
            arrays.append(arr)

        if arrays:
            return np.concatenate(arrays, axis=0).astype(np.float16)
        else:
            return np.empty((0,), dtype='float16')

    def split_train(self, num_files=None, fields_to_use=None):

        print(f"[{self.dataset_name}] Importing training data...")
        train_data = self.load_split('train', num_files=num_files, 
                                     fields_to_use=fields_to_use)

        print(f"[{self.dataset_name}] Importing validation data...")
        val_data = self.load_split('val', num_files=num_files, 
                                   fields_to_use=fields_to_use  )
        
        return train_data, val_data

    def split_test(self, num_files=None, fields_to_use=None):

        print(f"[{self.dataset_name}] Importing test data...")
        test_data = self.load_split('test', num_files=num_files, 
                                    fields_to_use=fields_to_use)

        return test_data