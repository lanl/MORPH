import os
import numpy as np
import h5py

def split_and_save_h5(raw_dir: str,
                      out_dir: str,
                      dataset_name = 'cfd2dic',
                      train_frac: float = 0.8,
                      rand = True):
    """
    Splits each file in raw_dir into train/val/test (val==test) along the
    first axis (N trajectories). Saves only the raw 'force' and 'velocity'
    fields—dropping 'particles' entirely.
    """
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(raw_dir)
             if f.endswith(".h5") or f.endswith(".hdf5")]
    if not files:
        raise FileNotFoundError(f"No HDF5 files in {raw_dir}")
        
    print(f'Number of files in {raw_dir}: {len(files)}')
    
    for fname in files:
        base, _ = os.path.splitext(fname)
        path = os.path.join(raw_dir, fname)
        print(f"\nProcessing {fname}")

        # --- load raw fields ---
        with h5py.File(path, "r") as f5:
            force_raw = f5["force"][...]      # (N, H, W, 2)
            vel_raw   = f5["velocity"][...]   # (N, T, H, W, 2)

        # sanity check
        N, H, W, Cf = force_raw.shape
        N2, T, H2, W2, Cv = vel_raw.shape
        assert N == N2 and H == H2 and W == W2 and Cf == Cv == 2, (
            f"Got force {force_raw.shape}, velocity {vel_raw.shape}")

        print(f"  loaded force {force_raw.shape}, velocity {vel_raw.shape}")
        
        # --- shuffle & split ---
        if rand:
            np.random.seed(42)
            idx = np.random.permutation(N) 
        else:
            # --- fixed 80/10/10 split without randomness ---
            idx = np.arange(N)
            
        # train_end = int(train_frac * N)
        # val_end   = int((train_frac + 0.1) * N)
        train_end = 3
        
        splits  = {
            'train': idx[:train_end],
            'val':   idx[train_end:N],
            'test':  idx[train_end:N],   # same as validation since we have 4 sims per file
        }
        
        # --- write each split as its own .h5 ---
        for split_name, indices in splits.items():
            split_folder = os.path.join(out_dir, split_name)
            os.makedirs(split_folder, exist_ok=True)
            out_path = os.path.join(split_folder, f"{base}_{split_name}.h5")

            with h5py.File(out_path, "w") as fo:
                fo.create_dataset("force",    data=force_raw[indices],
                                  compression="gzip")
                fo.create_dataset("velocity", data=vel_raw[indices],
                                  compression="gzip")

            print(f"  saved {split_name}: "
                  f"force{force_raw[indices].shape}, "
                  f"velocity{vel_raw[indices].shape}")
            
class CFD2dicDataLoader:
    '''
    The spatial force field (no time variation) is removed
    '''
    def __init__(self, data_path, force = False, dataset_name='cfd2dic'):
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.force = force
        
    def inflate_array(self, arr, axes):
        # arr must be 5-D ⇒ (S, T, H, W, F)
        for ax in sorted(axes):
            arr = np.expand_dims(arr, axis=ax)
        return arr
    
    def load_split(self, split, num_files):
        split_path = os.path.join(self.data_path, split)
        files = sorted([f for f in os.listdir(split_path)
                 if f.endswith('.h5') or f.endswith('.hdf5')])
        if not files:
            raise FileNotFoundError(f"No HDF5 files found in {files}")
        
        print(f'Number of files in {split_path}: {len(files)}')
        
        # select the number of files
        selected_files = sorted(files[:num_files] if num_files!=None else files)
    
        # explore the structure of file 
        # explorer = ExploreHDF5Structure()
        # explorer.explore_hdf5(os.path.join(file_path_cfd2d_ic, files[0]))   
    
        # Open the HDF5 file
        print('Loading CFD-2D (incompressible) data from PDEBench...')
        arrays = []
        for fname in selected_files:
            fullpath = os.path.join(split_path, fname)
            with h5py.File(fullpath, 'r') as f5:
                print(f"\nProcessing {fname}")
                # Directly read your three main fields
                vel      = f5['velocity'][...]  
                vel   =  np.expand_dims(vel, axis = 5)
                
                if self.force:
                    # expand the fields
                    force    = f5['force'][...]
                    force =  np.expand_dims(force, axis = (1,5))  # time and field
                    force = np.repeat(force, repeats = 1000, axis = 1)  # repeat time
                    data = np.concatenate((force, vel), axis = 5).astype(np.float32)
                    print("Shape of the data", data.shape)
                else:
                    data = vel
                    print("Shape of the data", data.shape)
                    
            arrays.append(data)
            
        # if you have multiple HDF5s per split, stack them; otherwise take the one
        if len(arrays) > 1:
            arrays = np.concatenate(arrays, axis=0)
        else:
            arrays = arrays[0]

        return arrays

    def split_train(self, num_files = None):
        print(f"[{self.dataset_name}] Importing training data...")
        train = self.load_split('train', num_files = num_files)
        train = self.inflate_array(train, axes=[2])

        print(f"[{self.dataset_name}] Importing validation data...")
        val   = self.load_split('val', num_files = num_files)
        val   = self.inflate_array(val,   axes=[2])

        return train, val

    def split_test(self, num_files = None):
        print(f"[{self.dataset_name}] Importing test data...")
        test = self.load_split('test', num_files = num_files)
        test = self.inflate_array(test, axes=[2])
        return test
