import math
from torch.utils.data import DataLoader
# --- for pretraining datasets ---
from src.utils.datastreamers.datastreaming_mhd_1 import MHDIterableDataset
from src.utils.datastreamers.datastreaming_dr_1 import DRChunkedIterableDataset
from src.utils.datastreamers.datastreaming_cfd1d_1 import CFD1DChunkedIterableDataset
from src.utils.datastreamers.datastreaming_cfd2dic_1 import CFD2DICChunkedIterableDataset
from src.utils.datastreamers.datastreaming_cfd3d_1 import CFD3DChunkedIterableDataset
from src.utils.datastreamers.datastreaming_sw_1 import SWChunkedIterableDataset
# --- for fine-tuning datasets ---
from src.utils.datastreamers.datastreaming_dr1d_1 import DR1DChunkedIterableDataset
from src.utils.datastreamers.datastreaming_cfd2d_1 import CFD2DChunkedIterableDataset
from src.utils.datastreamers.datastreaming_cfd3d_turb_1 import CFD3DTurbChunkedIterableDataset
from src.utils.datastreamers.datastreaming_be1d_1 import BE1DChunkedIterableDataset
from src.utils.datastreamers.datastreaming_gsdr2d_1 import GSDR2DChunkedIterableDataset
from src.utils.datastreamers.datastreaming_tgc3d_1 import TGC3DChunkedIterableDataset
from src.utils.datastreamers.datastreaming_fnskf_1 import FNSKF2DChunkedIterableDataset

'''
Sharding is at the dataset level
'''

class DataStreamersChaos():
    def __init__(self, DATA_CONFIG, ar_order, 
                 batch_sizes, workers, pin_flag, persist_flag,
                 chunk_mhd, chunk_dr, chunk_cfd1d, chunk_cfd2dic, chunk_cfd3d, chunk_sw,             # pretraining chunk sizes
                 chunk_dr1d, chunk_cfd2d, chunk_cfd3d_turb, chunk_be1d, chunk_gsdr2d, 
                 chunk_tgc3d, chunk_fnskf2d):  # finetuning chunk sizes
        self.ar_order = ar_order
        self.DATA_CONFIG = DATA_CONFIG
        self.batch_sizes = batch_sizes
        
        self.workers = workers
        self.pin_flag = pin_flag
        self.persist_flag = persist_flag
        
        # chunks for different datasets
        self.chunk_mhd = chunk_mhd
        self.chunk_dr = chunk_dr
        self.chunk_cfd1d = chunk_cfd1d
        self.chunk_cfd2dic = chunk_cfd2dic
        self.chunk_cfd3d = chunk_cfd3d
        self.chunk_sw = chunk_sw
        self.chunk_dr1d = chunk_dr1d
        self.chunk_cfd2d = chunk_cfd2d
        self.chunk_cfd3d_turb = chunk_cfd3d_turb
        self.chunk_be1d = chunk_be1d
        self.chunk_gsdr2d = chunk_gsdr2d
        self.chunk_tgc3d = chunk_tgc3d
        self.chunk_fnskf2d = chunk_fnskf2d
        
    def datastreamers(self, dataset_choice):
        # ---- stream data from different datastreaming (iterable dataset) ----
        tr_stream, va_stream = [], []
        # MHD split streams
        if dataset_choice in ('MHD','FM'):
            r = self.DATA_CONFIG['MHD']['root']
            tr_stream.append(('MHD', MHDIterableDataset
                              (r, 'train', self.ar_order, self.chunk_mhd, 
                               f'MHD-train-ar{self.ar_order}'))) # out of 5 and 8 / file train examples. Total files = 10
            va_stream.append(('MHD', MHDIterableDataset
                              (r, 'val', self.ar_order, 1, 
                               f'MHD-val-ar{self.ar_order}'))) # out of 1 / file train examples. Total files = 10
            
        # DR split streams
        if dataset_choice in ('DR','FM'):
            r = self.DATA_CONFIG['DR']['root']
            tr_stream.append(('DR', DRChunkedIterableDataset
                              (r, 'train', self.ar_order, self.chunk_dr, 
                               f'DR-train-ar{self.ar_order}'))) # out of 900 train examples
            va_stream.append(('DR', DRChunkedIterableDataset
                              (r, 'val', self.ar_order, self.chunk_dr, 
                               f'DR-val-ar{self.ar_order}')))   # out of 100 val examples
        
        # CFD1D split streams
        if dataset_choice in ('CFD1D','FM'):
            r = self.DATA_CONFIG['CFD1D']['root']
            tr_stream.append(('CFD1D', CFD1DChunkedIterableDataset
                              (r, 'train', self.ar_order, self.chunk_cfd1d, 
                               f'CFD1D-train-ar{self.ar_order}',
                               num_loadfiles = 1,
                               num_sims = 2250))) 
            va_stream.append(('CFD1D', CFD1DChunkedIterableDataset
                              (r, 'val', self.ar_order, self.chunk_cfd1d, 
                               f'CFD1D-val-ar{self.ar_order}',
                               num_loadfiles = 1,
                               num_sims = 250)))   
        
        # CFD2D-IC split streams
        if dataset_choice in ('CFD2D-IC','FM'):
            r = self.DATA_CONFIG['CFD2D-IC']['root']
            tr_stream.append(('CFD2D-IC', CFD2DICChunkedIterableDataset
                              (r, 'train', self.ar_order, self.chunk_cfd2dic, 
                               f'CFD2D-IC-train-ar{self.ar_order}',
                               num_loadfiles = 4))) # out of 3 / file train examples. Total files = 16
            va_stream.append(('CFD2D-IC', CFD2DICChunkedIterableDataset
                              (r, 'val', self.ar_order, 1, 
                               f'CFD2D-IC-val-ar{self.ar_order}',
                               num_loadfiles = 4)))   # out of 1 / file train examples. Total files = 16
        
        # CFD3D split streams
        if dataset_choice in ('CFD3D','FM'):
            r = self.DATA_CONFIG['CFD3D']['root']
            tr_stream.append(('CFD3D', CFD3DChunkedIterableDataset
                              (r, 'train', self.ar_order, self.chunk_cfd3d, 
                               f'CFD3D-train-ar{self.ar_order}')))   # out of 90 / file train examples. Total files = 2
            va_stream.append(('CFD3D', CFD3DChunkedIterableDataset
                              (r, 'val', self.ar_order, self.chunk_cfd3d, 
                               f'CFD3D-val-ar{self.ar_order}')))     # out of 10 / file train examples. Total files = 2
        
        # SW split streams   
        if dataset_choice in ('SW','FM'):
            r = self.DATA_CONFIG['SW']['root']
            tr_stream.append(('SW', SWChunkedIterableDataset
                              (r, 'train', self.ar_order, self.chunk_sw, 
                               f'SW-train-ar{self.ar_order}'))) # out of 900 / file train examples. Total files = 1
            va_stream.append(('SW', SWChunkedIterableDataset
                              (r, 'val',   self.ar_order, 100, 
                               f'SW-val-ar{self.ar_order}')))             # out of 100 / file val examples. Total files = 1
        
        # --- Fine-tuning datastreamers (Do not include in FM) ----
        # DR1d stream
        if dataset_choice == 'DR1D':
            r = self.DATA_CONFIG['DR1D']['root']
            tr_stream.append(('DR1D', DR1DChunkedIterableDataset
                              (r, 'train', self.ar_order, self.chunk_dr1d, 
                               f'DR1D-train-ar{self.ar_order}',
                               num_loadfiles = 1))) # out of 9000 / file train examples. Total files = 1
            va_stream.append(('DR1D', DR1DChunkedIterableDataset
                              (r, 'val',   self.ar_order, self.chunk_dr1d, 
                               f'DR1D-val-ar{self.ar_order}',
                               num_loadfiles = 1)))             # out of 1000 / file val examples. Total files = 1
        # CFD2D stream
        if dataset_choice == 'CFD2D':
            r = self.DATA_CONFIG['CFD2D']['root']
            tr_stream.append(('CFD2D', CFD2DChunkedIterableDataset
                              (r, 'train', self.ar_order, self.chunk_cfd2d, 
                               f'CFD2D-train-ar{self.ar_order}',
                               num_loadfiles = 1))) # out of 9000 / file train examples. Total files = 1
            va_stream.append(('CFD2D', CFD2DChunkedIterableDataset
                              (r, 'val',   self.ar_order, self.chunk_cfd2d, 
                               f'CFD2D-val-ar{self.ar_order}',
                               num_loadfiles = 1)))             # out of 1000 / file val examples. Total files = 1
            
        # CFD3D-Turb split streams
        if dataset_choice == 'CFD3D-TURB':
            r = self.DATA_CONFIG['CFD3D-TURB']['root']
            tr_stream.append(('CFD3D-TURB', CFD3DTurbChunkedIterableDataset
                              (r, 'train', self.ar_order, self.chunk_cfd3d_turb, 
                               f'CFD3D-TURB-train-ar{self.ar_order}')))   # out of 90 / file train examples. Total files = 2
            va_stream.append(('CFD3D-TURB', CFD3DTurbChunkedIterableDataset
                              (r, 'val', self.ar_order, self.chunk_cfd3d_turb, 
                               f'CFD3D-TURB-val-ar{self.ar_order}')))     # out of 10 / file train examples. Total files = 2
            
        # BE1D stream
        if dataset_choice == 'BE1D':
            r = self.DATA_CONFIG['BE1D']['root']
            tr_stream.append(('BE1D', BE1DChunkedIterableDataset
                              (r, 'train', self.ar_order, self.chunk_be1d, 
                               f'BE1D-train-ar{self.ar_order}',
                               num_loadfiles = 1))) # out of 8000 / file train examples. Total files = 1
            va_stream.append(('BE1D', BE1DChunkedIterableDataset
                              (r, 'val',   self.ar_order, self.chunk_be1d, 
                               f'BE1D-val-ar{self.ar_order}',
                               num_loadfiles = 1))) # out of 1000 / file val examples. Total files = 1
            
        # GSDR2D stream
        if dataset_choice == 'GSDR2D':
            r = self.DATA_CONFIG['GSDR2D']['root']
            tr_stream.append(('GSDR2D', GSDR2DChunkedIterableDataset
                              (r, 'train', self.ar_order, self.chunk_gsdr2d, 
                               f'GSDR2D-train-ar{self.ar_order}'))) # out of 8000 / file train examples. Total files = 1
            va_stream.append(('GSDR2D', GSDR2DChunkedIterableDataset
                              (r, 'val',   self.ar_order, self.chunk_gsdr2d, 
                               f'GSDR2D-val-ar{self.ar_order}'))) # out of 1000 / file val examples. Total files = 1
            
        # TGC3D stream
        if dataset_choice == 'TGC3D':
            r = self.DATA_CONFIG['TGC3D']['root']
            tr_stream.append(('TGC3D', TGC3DChunkedIterableDataset
                              (r, 'train', self.ar_order, self.chunk_tgc3d, 
                               f'TGC3D-train-ar{self.ar_order}')))  
            va_stream.append(('TGC3D', TGC3DChunkedIterableDataset
                              (r, 'val',   self.ar_order, self.chunk_tgc3d, 
                               f'TGC3D-val-ar{self.ar_order}')))    
        
        # GSDR2D stream
        if dataset_choice == 'FNS_KF_2D':
            r = self.DATA_CONFIG['FNS_KF_2D']['root']
            tr_stream.append(('FNS_KF_2D', FNSKF2DChunkedIterableDataset
                              (r, 'train', self.ar_order, self.chunk_fnskf2d, 
                               f'FNS_KF_2D-train-ar{self.ar_order}'))) # out of 8000 / file train examples. Total files = 1
            va_stream.append(('FNS_KF_2D', FNSKF2DChunkedIterableDataset
                              (r, 'val',   self.ar_order, self.chunk_fnskf2d, 
                               f'FNS_KF_2D-val-ar{self.ar_order}'))) # out of 1000 / file val examples. Total files = 1
            
        # ---- wrapping each IterableDataset in its own DataLoader ----
        train_loaders = [DataLoader(ds, batch_size=self.batch_sizes[name], 
                                    num_workers=self.workers, 
                                    pin_memory=self.pin_flag,                       # pin_memory: faster cpu -> gpu copy
                                    persistent_workers=self.persist_flag) 
                                    for name, ds in tr_stream]
        val_loaders   = [DataLoader(ds, batch_size=self.batch_sizes[name], 
                                    num_workers=self.workers, 
                                    pin_memory=self.pin_flag, 
                                    persistent_workers=self.persist_flag)             # persistant_workers: worker processes alive across epoch
                                    for name, ds in va_stream]
        
        # save for later
        self.tr_stream = tr_stream
        self.va_stream = va_stream
        self.train_loaders = train_loaders
        self.val_loaders = val_loaders
        
        return train_loaders, val_loaders
    
    def data_sampling(self, task_weights):
        train_scores = [task_weights[name] for name, ds in self.tr_stream]
        val_scores = [task_weights[name] for name, ds in self.va_stream]
        
        return train_scores, val_scores
    
    def lengths(self):
        train_stats = {}
        for (label, ds), loader in zip(self.tr_stream, self.train_loaders):
            n = len(ds)
            b = loader.batch_size
            train_stats[label] = (n, math.ceil(n/b))

        val_stats = {}
        for (label, ds), loader in zip(self.va_stream, self.val_loaders):
            n = len(ds)
            b = loader.batch_size
            val_stats[label] = (n, math.ceil(n/b))

        return train_stats, val_stats