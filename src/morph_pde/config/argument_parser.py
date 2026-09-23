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
        parser.add_argument('--dataset', choices=['MHD','DR','CFD1D','CFD2D-IC',
        'CFD3D','SW','DR1D','CFD2D','CFD3D-TURB','BE1D','GSDR2D','TGC3D','FNS_KF_2D','FM'], 
        default='FM')
        # --- Pretraining datasets ---
        parser.add_argument('--chunk_mhd', type=int, default=3, help='max chunk size = 8')
        parser.add_argument('--chunk_dr',  type=int, default=200, help='max chunk size = 800')
        parser.add_argument('--chunk_cfd1d', type=int, default=50, help='max chunk size = 8000')
        parser.add_argument('--chunk_cfd2dic', type=int, default=1, help='max chunk size = 3')
        parser.add_argument('--chunk_cfd3d', type=int, default=3, help='max chunk size = 100')
        parser.add_argument('--chunk_sw', type=int, default=200, help='max chunk size = 800')
        # --- Finetuning datasets ---
        parser.add_argument('--chunk_dr1d', type=int, default=500, help='max chunk size = 8000')
        parser.add_argument('--chunk_cfd2d', type=int, default=10, help='max chunk size = 8000')
        parser.add_argument('--chunk_cfd3d_turb', type=int, default=5, help='max chunk size = 480')
        parser.add_argument('--chunk_be1d', type=int, default=50, help='max chunk size = 8000')
        parser.add_argument('--chunk_gsdr2d', type=int, default=50, help='max chunk size = 160')
        parser.add_argument('--chunk_tgc3d', type=int, default=5, help='max chunk size = 80')
        parser.add_argument('--chunk_fnskf2d', type=int, default=500, help='max chunk size = 16000')

        # ---- model hyperparameters ----
        parser.add_argument('--model_size', type=str,
                            choices = list(MORPH_MODELS.keys()),
                            default='Ti', help='choose from Ti, S, M, Lt, L, XL')
        parser.add_argument('--max_ar_order', type=int, default=1)
        parser.add_argument('--activated_ar1k', action='store_true',
                            help = 'train time-axial-attention everytime')
        parser.add_argument('--ar_order', type=int, default=1)
        parser.add_argument('--resume', action='store_true')
        parser.add_argument('--ckpt_name', type=str, default=None)
        parser.add_argument('--finetune_ar1k', action='store_true',
                            help = 'Second stage pretraining')
        parser.add_argument('--tf_params', nargs=5, type=int,
                            metavar=('filters','dim','heads','depth','mlp_dim'),
                            default=None,
                            help='conv_filters, dim, heads, depth, mlp neurons')
        parser.add_argument('--tf_reg', nargs=2, type=float,
                            metavar=('dropout','emb_dropout'),
                            default=[0.1,0.1],
                            help='transformer regularization: dropouts')
        parser.add_argument('--heads_xa', type=int, default=32,
                            help='number of heads of cross attention')

        # ---- training hyperparameters ----
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--new_lr_ckpt', type=float, default=None, help = 'Adjust training while using previous ckpt')
        parser.add_argument('--num_epochs', type=int, default=150)
        parser.add_argument('--warm_epochs', type = int, default = 20)
        parser.add_argument('--patience', type=int, default=10, help='early stopping')

        # ---- batch sizes ----
        parser.add_argument('--bs', type=int, nargs=13,
        metavar=('MHD','DR','CFD1D','CFD2D-IC','CFD3D','SW',
                 'DR1D','CFD2D','CFD3D-TURB','BE1D','GSDR2D','TGC3D', 'FNS_KF_2D'),
                 default=[16, 64, 128, 16, 4, 64,
                          384, 8, 16, 384, 64, 16, 64],
                 help='Batch sizes for each dataset')

        # ---- infra hyperparameters ----
        parser.add_argument('--parallel', type=str, choices=['ddp','dp','no'], default='dp', help='dp vs ddp vs no')
        parser.add_argument('--scale_gpu_utils', type=str, choices=['1x','2x','4x','0.5x','0.25x'], default='1x', 
                            help='scale batches based on gpu utilization, 1x for 40GB, M model')
        parser.add_argument('--cpu_cores_per_node', type=int, default=56, help='number of physical cores')
        parser.add_argument('--local_rank', type=int, default=int(os.getenv('LOCAL_RANK',0)))
        parser.add_argument('--device_idx', type=int, default=0, help = 'select gpu for parallel = no')
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--pin_flag', action='store_true')
        parser.add_argument('--persist_flag', action='store_true')
        parser.add_argument('--save_every', type=int, default=1)
        parser.add_argument('--save_batch_ckpt', action='store_true')
        parser.add_argument('--save_batch_freq', type=int, default=1000)
        parser.add_argument('--overwrite_weights', action='store_true')