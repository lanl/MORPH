from src.utils.dataloaders.dataloader_mhd import MHDDataLoader
from src.utils.dataloaders.dataloader_dr import DR2DDataLoader
from src.utils.dataloaders.dataloader_cfd1d import CFD1dDataLoader
from src.utils.dataloaders.dataloader_cfd2dic import CFD2dicDataLoader
from src.utils.dataloaders.dataloader_cfd3d import CFD3dDataLoader
from src.utils.dataloaders.dataloader_sw2d import SW2dDataLoader
from src.utils.dataloaders.dataloader_dr1d import DR1DDataLoader
from src.utils.dataloaders.dataloader_cfd2d import CFD2DDataLoader
from src.utils.dataloaders.dataloader_cfd3d_turb import CFD3dTurbDataLoader
from src.utils.dataloaders.dataloader_be1d import BE1DDataLoader
from src.utils.dataloaders.dataloader_gsdr2d import GSDR2dDataLoader
from src.utils.dataloaders.dataloader_tgc3d import TGC3dDataLoader
from src.utils.dataloaders.dataloader_fns_kf_2d import FNSKF2dDataLoader

class DataloaderChaos():
    
    @staticmethod
    def load_data(dataset_name, loadpath, split):
        
        # --- Pre-training datasets ----    
        if dataset_name == 'MHD':
            dataset = MHDDataLoader(loadpath)
            if split == 'test':
                test_data = dataset.split_test()
                return test_data
            else:
                train_data, val_data = dataset.split_train()
                return train_data, val_data
            
        elif dataset_name == 'DR':
            dataset = DR2DDataLoader(loadpath)
            if split == 'test':
                test_data = dataset.split_test()
                return test_data
            else:
                train_data, val_data = dataset.split_train()
                return train_data, val_data
            
        elif dataset_name == 'CFD1D':
            dataset = CFD1dDataLoader(loadpath)
            if split == 'test':
                test_data = dataset.split_test(num_files = 1)
                return test_data
            else:
                train_data, val_data = dataset.split_train(num_files = 1, sims = 2250)
                return train_data, val_data
            
        elif dataset_name == 'CFD2D-IC':
            dataset = CFD2dicDataLoader(loadpath)
            if split == 'test':
                test_data = dataset.split_test(num_files = 4)
                return test_data
            else:
                train_data, val_data = dataset.split_train(num_files = 4)
                return train_data, val_data
            
        elif dataset_name == 'CFD3D':
            dataset = CFD3dDataLoader(loadpath)
            if split == 'test':
                test_data = dataset.split_test()
                return test_data
            else:
                train_data, val_data = dataset.split_train()
                return train_data, val_data
            
        elif dataset_name == 'SW':
            dataset = SW2dDataLoader(loadpath)
            if split == 'test':
                test_data = dataset.split_test()
                return test_data
            else:
                train_data, val_data = dataset.split_train()
                return train_data, val_data
            
        # --- Fine-tuning datasets ----    
        elif dataset_name == 'DR1D':
            dataset = DR1DDataLoader(loadpath)
            if split == 'test':
                test_data = dataset.split_test(selected_idx = 0) 
                return test_data
            else:
                train_data, val_data = dataset.split_train(selected_idx = 0)
                return train_data, val_data
            
        elif dataset_name == 'CFD2D':
            dataset = CFD2DDataLoader(loadpath) 
            if split == 'test':
                test_data = dataset.split_test(selected_idx = 0)
                return test_data
            else:
                train_data, val_data = dataset.split_train(selected_idx = 0)
                return train_data, val_data
            
        elif dataset_name == 'CFD3D-TURB':
            dataset = CFD3dTurbDataLoader(loadpath) 
            if split == 'test':
                test_data = dataset.split_test()
                return test_data
            else:
                train_data, val_data = dataset.split_train()
                return train_data, val_data
            
        elif dataset_name == 'BE1D':
            dataset = BE1DDataLoader(loadpath) 
            if split == 'test':
                test_data = dataset.split_test(selected_idx = 0)
                return test_data
            else:
                train_data, val_data = dataset.split_train(selected_idx = 0)
                return train_data, val_data
            
        elif dataset_name == 'GSDR2D':
            dataset = GSDR2dDataLoader(loadpath) 
            if split == 'test':
                test_data = dataset.split_test()
                return test_data
            else:
                train_data, val_data = dataset.split_train()
                return train_data, val_data
        
        elif dataset_name == 'TGC3D':
            dataset = TGC3dDataLoader(loadpath) 
            if split == 'test':
                test_data = dataset.split_test()
                return test_data
            else:
                train_data, val_data = dataset.split_train()
                return train_data, val_data
            
        elif dataset_name == 'FNS_KF_2D':
            dataset = FNSKF2dDataLoader(loadpath) 
            if split == 'test':
                test_data = dataset.split_test()
                return test_data
            else:
                train_data, val_data = dataset.split_train()
                return train_data, val_data
            
            
            