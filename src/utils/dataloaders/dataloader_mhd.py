import os
import numpy as np
import h5py

class MHDDataLoader:
    def __init__(self, data_path, dataset_name='MHD'):
        self.data_path = data_path
        self.dataset_name = dataset_name

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
                magnetic_field = f5['t1_fields/magnetic_field'][:]  # (N,T,D,H,W,3)
                velocity       = f5['t1_fields/velocity'][:]        # (N,T,D,H,W,3)
                density        = f5['t0_fields/density'][:]         # (N,T,D,H,W)

                # expand density to 3 channels
                density = np.expand_dims(density, axis=-1)  # (N,T,D,H,W,1)
                density = np.tile(density, (1,)* (density.ndim-1) + (3,))  # (N,T,D,H,W,3)
                
                # concat across last
                arr = np.stack([magnetic_field, velocity, density], axis=-1)  # (N,T,D,H,W,C=3,F=3)

            arrays.append(arr.astype('float32'))

        if arrays:
            return np.concatenate(arrays, axis=0)
        else:
            return np.empty((0,), dtype='float32')

    def split_train(self):
        print(f"[{self.dataset_name}] Importing training data...")
        train_data = self.load_split('train')
        print(f"[{self.dataset_name}] Importing validation data...")
        val_data = self.load_split('val')
        return train_data, val_data

    def split_test(self):
        print(f"[{self.dataset_name}] Importing test data...")
        test_data = self.load_split('test')
        return test_data