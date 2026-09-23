import os
import numpy as np
import h5py

class GSDR2dDataLoader:
    def __init__(self, data_path, dataset_name='GSDR'):
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
                act_A = f5['t0_fields/A'][:]   # (160,1001,128,128)
                act_B = f5['t0_fields/B'][:]
                
                # expand density to 3 channels
                act_A = np.expand_dims(act_A, axis=-1)  # (160,1001,128,128,1)
                act_B = np.expand_dims(act_B, axis=-1)  # (160,1001,128,128,1)
                
                # concat A and B
                data = np.concatenate((act_A, act_B), axis=4) # (160,1001,128,128,2)

            arrays.append(data.astype('float32'))

        if arrays:
            return np.concatenate(arrays, axis=0)
        else:
            return np.empty((0,), dtype='float32')

    def split_train(self):
        print(f"[{self.dataset_name}] Importing training data...")
        train_data = self.load_split('train')
        train_data = self.inflate_array(train_data, axes=[2,5])
        
        print(f"[{self.dataset_name}] Importing validation data...")
        val_data = self.load_split('val')
        val_data = self.inflate_array(val_data, axes=[2,5])
        return train_data, val_data

    def split_test(self):
        print(f"[{self.dataset_name}] Importing test data...")
        test_data = self.load_split('test')
        test_data = self.inflate_array(test_data, axes=[2,5])
        return test_data