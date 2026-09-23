import argparse
import os

# ---- Different model sizes ----
MORPH_MODELS = {
    # filters, dims, heads, depth, mlp
    'Ti': [8, 256,  4,  4, 1024], # Ti
    'S' : [8, 512,  8,  4, 2048], # S
    'M' : [8, 768, 12,  8, 3072], # M
    'Lt' : [8, 1024, 16,  8, 3072], # Large-lite (with AR1:8 ~231M)
    'L' : [8, 1024, 16, 16, 4096],  # Large   (with AR1:16 ~480M)
    'XL' : [64, 1536, 24, 16, 8192]  # Extra Large (with AR1:8 ~1.12B)
    }

class ArgsConfig:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="device",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser = parser
        self._add_args(parser)

        ns = parser.parse_args()
        for key, val in vars(ns).items():
            setattr(self, key, val)

        # post-parse fill-in for tf_params
        if self.tf_params is None:
            self.tf_params = MORPH_MODELS[self.model_size]

    def _add_args(self, parser: argparse.ArgumentParser):
        # ---- data loading hyperparameters ----
        parser.add_argument('--dataset_root', type = str, default=None, help = "Location of dataset")
        parser.add_argument('--dataset', choices=['MHD','DR','CFD1D','CFD2D-IC','CFD3D','SW','DR1D',
                            'CFD2D','CFD3D-TURB','BE1D','GSDR2D','TGC3D','FNS_KF_2D',
                            'FM',
                            'HEAT2D','FRAC2D', # cl datasets added
                            'CL'],  
                            default='FM')
        
        # --- Pretraining datasets ---
        # chunk sizes for each dataset
        parser.add_argument('--chunk_mhd', type=int, default=3, help='max chunk size = 8')
        parser.add_argument('--chunk_dr',  type=int, default=200, help='max chunk size = 800')
        parser.add_argument('--chunk_cfd1d', type=int, default=50, help='max chunk size = 8000')
        parser.add_argument('--chunk_cfd2dic', type=int, default=1, help='max chunk size = 3')
        parser.add_argument('--chunk_cfd3d', type=int, default=3, help='max chunk size = 100')
        parser.add_argument('--chunk_sw', type=int, default=200, help='max chunk size = 800')
        # batchsizes for each dataset
        parser.add_argument('--bs_mhd', type=int, default=16)
        parser.add_argument('--bs_dr', type=int, default=64)
        parser.add_argument('--bs_cfd1d', type=int, default=128)
        parser.add_argument('--bs_cfd2dic', type=int, default=16)
        parser.add_argument('--bs_cfd3d', type=int, default=4)
        parser.add_argument('--bs_sw', type=int, default=64)

        # --- Finetuning datasets ---
        # chunk sizes for each finetuning dataset
        parser.add_argument('--chunk_dr1d', type=int, default=500, help='max chunk size = 8000')
        parser.add_argument('--chunk_cfd2d', type=int, default=10, help='max chunk size = 8000')
        parser.add_argument('--chunk_cfd3d_turb', type=int, default=5, help='max chunk size = 480')
        parser.add_argument('--chunk_be1d', type=int, default=50, help='max chunk size = 8000')
        parser.add_argument('--chunk_gsdr2d', type=int, default=50, help='max chunk size = 160')
        parser.add_argument('--chunk_tgc3d', type=int, default=5, help='max chunk size = 80')
        parser.add_argument('--chunk_fnskf2d', type=int, default=500, help='max chunk size = 16000')
        # batchsizes for each finetuning dataset
        parser.add_argument('--bs_dr1d', type=int, default=384)
        parser.add_argument('--bs_cfd2d', type=int, default=8)
        parser.add_argument('--bs_cfd3d_turb', type=int, default=16)
        parser.add_argument('--bs_be1d', type=int, default=384)
        parser.add_argument('--bs_gsdr2d', type=int, default=64)
        parser.add_argument('--bs_tgc3d', type=int, default=16)
        parser.add_argument('--bs_fnskf2d', type=int, default=128)

        # --- NEW CL datasets ---
        parser.add_argument('--chunk_heat2d', type=int, default=100, help='max chunk size = 100')
        parser.add_argument('--chunk_frac2d', type=int, default=1000, help='max chunk size = 1000')
        parser.add_argument('--bs_heat2d', type=int, default=1)
        parser.add_argument('--bs_frac2d', type=int, default=128)

        # --- num loadfiles for CL datasets ---
        parser.add_argument('--num_loadfiles_heat2d', type=int, default=None, 
                            help='number of files to load for HEAT2D out of 42 train files')
        parser.add_argument('--num_loadfiles_frac2d', type=int, default=None, 
                            help='number of files to load for FRAC2D out of 320 train files')
        
        # --- fields to use for HEAT2D dataset ---
        parser.add_argument('--fields_to_use_heat2d', nargs='+', type=int, default=None,
                            help='list of field indices to use for HEAT2D dataset')
        
        # --- patch size ---
        parser.add_argument('--patch_size', type=int, default=8)

        # ---- model hyperparameters ----
        parser.add_argument('--model_size', type=str,
                            choices = list(MORPH_MODELS.keys()),
                            default='Ti', help='choose from Ti, S, M, Lt, L, XL')
        parser.add_argument('--max_ar_order', type=int, default=1)
        parser.add_argument('--ar_order', type=int, default=1)
        parser.add_argument('--resume_pt', action='store_true', help='resume from pretrained FM model')
        parser.add_argument('--resume_cl', action='store_true', help='resume from continual learning checkpoint')
        parser.add_argument('--ckpt_name', type=str, default=None)
        parser.add_argument('--tf_params', nargs=5, type=int,
                            metavar=('filters','dim','heads','depth','mlp_dim'),
                            default=None,
                            help='conv_filters, dim, heads, depth, mlp neurons')
        parser.add_argument('--tf_reg', nargs=2, type=float,
                            metavar=('dropout','emb_dropout'),
                            default=[0.1,0.1],
                            help='transformer regularization: dropouts')

        # ---- training hyperparameters ----
        parser.add_argument('--lr', type=float, default=1e-5)
        parser.add_argument('--wd', type=float, default=1e-2, help = 'weight decay')
        parser.add_argument('--num_epochs', type=int, default=150)
        parser.add_argument('--warm_epochs', type = int, default = 5)
        parser.add_argument('--patience', type=int, default=10, help='early stopping')

        # for schedular
        parser.add_argument('--min_lr', type=float, default=1e-7)

        # ---- infra hyperparameters ----
        parser.add_argument('--parallel', type=str, choices=['ddp','dp','no'], default='dp', help='dp vs ddp vs no')
        parser.add_argument('--scale_gpu_utils', type=str, choices=['1x','2x','4x','0.5x','0.25x'], default='1x', 
                            help='scale batches based on gpu utilization, 1x for 40GB, M model')
        parser.add_argument('--cpu_cores_per_node', type=int, default=48, help='number of physical cores')
        parser.add_argument('--local_rank', type=int, default=int(os.getenv('LOCAL_RANK',0)))
        parser.add_argument('--device_idx', type=int, default=0, help = 'select gpu for parallel = no')
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--pin_flag', action='store_true')
        parser.add_argument('--persist_flag', action='store_true')
        parser.add_argument('--save_every', type=int, default=1)
        parser.add_argument('--save_batch_ckpt', action='store_true')
        parser.add_argument('--save_batch_freq', type=int, default=1000)
        parser.add_argument('--overwrite_weights', action='store_true')

        # ---- hyperparameters for continual learning ----
        parser.add_argument("--cl_old_frac", type=float, default=None,   
        help="CL mode only: fraction of sampling probability mass allocated to OLD datasets "
            "(MHD, DR, CFD1D, CFD2D-IC, CFD3D, SW). Remainder goes to (HEAT2D, FRAC2D).")
        
        # ---lora parameters for continual learning ---
        parser.add_argument('--rank_lora_attn', type = int, default = 16, 
                            help = "Rank of attention layers in transformer module")
        parser.add_argument('--rank_lora_mlp', type = int, default = 12, 
                            help = "Rank of MLP layers in transformer module")
        parser.add_argument('--lora_p', type = float, default = 0.05, 
                            help = "Dropout inside LoRA layers")
        
        # --- Finetune levels ---                     
        parser.add_argument('--ft_level1', action='store_true', help = "Level-1 finetuning (LoRA, PE, LN)")
        parser.add_argument('--ft_level2', action='store_true', help = "Level-2 finetuning (Encoder)")
        parser.add_argument('--ft_level3', action='store_true', help = "Level-3 finetuning (Decoder)")
        parser.add_argument('--ft_level4', action='store_true', help = "All model parameters")

        # ---- Knowledge Distillation parameters ----
        parser.add_argument('--kd_lambda', type = float, default = 0.0, 
                            help = "Weight for KD loss during CL finetuning")