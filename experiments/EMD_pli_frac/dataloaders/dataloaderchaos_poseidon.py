from src.utils.dataloaders.dataloader_heat2d_poseidon import HEAT2dDataLoader
from src.utils.dataloaders.dataloader_frac2d_poseidon import FRAC2dDataLoader

class DataloaderChaos():

    @staticmethod
    def load_heat2d(loadpath, split, num_files=None, fields_to_use=None):
        dataset = HEAT2dDataLoader(loadpath) 
        if split == 'test':
            test_data = dataset.split_test(num_files = num_files, 
                            fields_to_use = fields_to_use)
            return test_data
        else:
            train_data, val_data = dataset.split_train(num_files = num_files, 
                                        fields_to_use = fields_to_use)
            return train_data, val_data
    
    @staticmethod
    def load_frac2d(loadpath, split, num_files=None):
        dataset = FRAC2dDataLoader(loadpath) 
        if split == 'test':
            test_data = dataset.split_test(num_files = num_files)
            return test_data
        else:
            train_data, val_data = dataset.split_train(num_files = num_files)
            return train_data, val_data
            
            