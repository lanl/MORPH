import os
import numpy as np
import h5py

class TGC3dDataLoader:
    def __init__(self, data_path, dataset_name='TGC3d'):
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
                dens = f5['t0_fields/density'][:]                                     # (80, 21, 64, 64, 64)
                # press = f5['t0_fields/pressure'][:]                                 # (80, 50, 64, 64, 64)
                temp = f5['t0_fields/temperature'][:]                                 # (80, 50, 64, 64, 64)
                vel = f5['t1_fields/velocity'][:]                                     # (80, 50, 64, 64, 64, 3)

                dens = np.expand_dims(dens, axis=(5,6))                               # (80, 50, 64, 64, 64, 1, 1)
                # press = np.expand_dims(press, axis=(5,6))                           # (80, 50, 64, 64, 64, 1, 1)
                temp = np.expand_dims(temp, axis=(5,6))                               # (80, 50, 64, 64, 64, 1, 1)
                vel = np.expand_dims(vel, axis=6)                                     # (80, 50, 64, 64, 64, 3, 1)
                
                dens = np.repeat(dens, repeats = 3, axis = 5)                         # (80, 50, 64, 64, 64, 3, 1)
                # press = np.repeat(press, repeats = 3, axis = 5)                     # (80, 50, 64, 64, 64, 3, 1)
                temp = np.repeat(temp, repeats = 3, axis = 5)                         # (80, 50, 64, 64, 64, 3, 1)
                
                print(f'Vel:{vel.shape}, dens:{dens.shape}, temp:{temp.shape}')
                arr = np.concatenate((vel, dens, temp), axis = 6).astype(np.float32) # (80,50,64,64,64,3,4)
                print("Shape of the data", arr.shape)

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