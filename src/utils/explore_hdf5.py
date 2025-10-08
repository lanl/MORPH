import h5py

class ExploreHDF5Structure:
    def __init__(self):
        pass
        
    def print_hdf5_structure(self, name, obj):
        """Function to print HDF5 file structure"""
        if isinstance(obj, h5py.Dataset):
            print(f"DATASET: {name}")
            print(f"    Shape: {obj.shape}, Type: {obj.dtype}")
            if len(obj.attrs) > 0:
                print("    Attributes:")
                for key, value in obj.attrs.items():
                    print(f"        {key}: {value}")
        elif isinstance(obj, h5py.Group):
            print(f"GROUP: {name}")
            if len(obj.attrs) > 0:
                print("    Attributes:")
                for key, value in obj.attrs.items():
                    print(f"        {key}: {value}")
                    
    def explore_hdf5(self, file_path):
        """Explore the complete structure of an HDF5 file"""
        with h5py.File(file_path, 'r') as f:
            print("HDF5 FILE STRUCTURE:")
            print("===================")
            # Recursively visit all objects in the file
            f.visititems(self.print_hdf5_structure)
            
            print("\nROOT LEVEL KEYS:")
            print("===============")
            # for key in f.keys():
            #     print(f"- {key}")