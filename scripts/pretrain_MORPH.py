import os
import sys
import gc
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import platform
import time
import math
import torch.distributed as dist
import socket

# Add project root to path
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

# Hyperparameters for data
patch_size = 8

# ---- Data paths (training on 6 datasets) ----
from config.data_config import DataConfig
DATA_CONFIG = DataConfig(project_root, patch_size)

# ---- Hyperparameters for data ----
max_imsize_1d, max_imsize_2d, max_imsize_3d = 1024, 512, 128
max_patches = max((max_imsize_1d//patch_size), 
                  (max_imsize_2d//patch_size)**2, 
                  (max_imsize_3d//patch_size)**3)  # max(128, 4096, 4096)

savepath_model = os.path.join(project_root, 'models')
savepath_results = os.path.join(project_root, 'experiments', 'results')
    
#%% Main entry
# Import model, trainers, and data loader
from config.argument_parser import ArgsConfig
from src.utils.device_manager import DeviceManager
from src.utils.vit_conv_xatt_axialatt2 import ViT3DRegression
from src.utils.trainers import Trainer
from src.utils.datastreamers.datastreamerschaos_1 import DataStreamersChaos
from src.utils.stream_iterabledatasets import WeightedMultiSourceIterableDataset_1
from src.utils.main_process_ddp import is_main_process
from src.utils.finetune_ar1k import FineTuneAR

# main process
def main():
    # ---- arguments ----
    args = ArgsConfig()
    choice = args.dataset
    ar_order = args.ar_order
    max_ar_order = args.max_ar_order
    filters, dim, heads, depth, mlp_dim = args.tf_params
    dropout, emb_dropout                = args.tf_reg
    
    # --- Data paths (pretraining on 6 datasets) ---
    if args.dataset_root is None:
        dataset_root = project_root
    else:
        dataset_root = args.dataset_root
    
    if is_main_process(): print(f"→ Current dataset root: {dataset_root}")
    
    # --- Data configuration ---
    DATA_CONFIG = DataConfig(dataset_root, patch_size = 8)
    
    # --- batch sizes ---
    bs_mhd, bs_dr, bs_cfd1d, bs_cfd2dic, bs_cfd3d, bs_sw, bs_dr1d, \
    bs_cfd2d, bs_cfd3d_turb, bs_be1d, bs_gsdr2d, bs_tgc3d, bs_fnskf2d = args.bs
    
    ##########################################################################
    ####################### --- Batch sizes --- ##############################
    ##########################################################################
    
    batch_sizes = {'MHD': bs_mhd, 'DR': bs_dr, 'CFD1D': bs_cfd1d, 
                   'CFD2D-IC': bs_cfd2dic, 'CFD3D': bs_cfd3d, 'SW': bs_sw,
                   'DR1D': bs_dr1d, 'CFD2D': bs_cfd2d, 'CFD3D-TURB': bs_cfd3d_turb,
                   'BE1D': bs_be1d, 'GSDR2D': bs_gsdr2d, 'TGC3D': bs_tgc3d,
                   'FNS_KF_2D': bs_fnskf2d}

    # scale batch size accordingly to size of GPUs
    if args.scale_gpu_utils == '2x':
        batch_sizes = {k: v * 2 for k, v in batch_sizes.items()}
    elif args.scale_gpu_utils == '4x':
        batch_sizes = {k: v * 4 for k, v in batch_sizes.items()}
    elif args.scale_gpu_utils == '0.5x':
        batch_sizes = {k: v // 2 for k, v in batch_sizes.items()}
    elif args.scale_gpu_utils == '0.25x':
        batch_sizes = {k: v // 4 for k, v in batch_sizes.items()}
    
    # scale batch size accordingly for parallel method dp
    if args.parallel == 'dp': ## scale batch size gpus
        ng = torch.cuda.device_count() 
        # scale batch size but not more than 4 times
        if ng > 4:
            ng_lim = 4
        else:
            ng_lim = ng
        batch_sizes = {k: v * ng_lim for k, v in batch_sizes.items()}
    
    ##########################################################################
    ################## ---- set up devices ---- ##############################
    ##########################################################################
    # ---- set up devices ---
    if args.parallel == 'ddp':
        if platform.system() == 'Windows':
            ddp_backend = 'gloo'
        else:
            ddp_backend = 'nccl' # pick the right backend
        
        # ----  Set-up GPUs ----
        local_rank = int(os.environ["LOCAL_RANK"])
        rank       = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=ddp_backend, init_method='env://',
                                rank=rank, world_size=world_size)
        dist.barrier(device_ids=[local_rank]) # wait for everyone
        device = torch.device(f'cuda:{local_rank}')
        print(f"[rank={rank} | local_rank={local_rank} | world_size={world_size}] "
            f"host={socket.gethostname()} current_device={torch.cuda.current_device()}")
        
        # --- set-up CPU ---
        # OpenMP and MKL not to oversubscribe
        cpu_cores = int(args.cpu_cores_per_node)
        tot_dataloaders = 6 if choice == 'FM' else 1
        gpus_per_node = max(1, torch.cuda.device_count())  # 1 training proc per GPU
        train_procs = gpus_per_node
        worker_procs_per_train = args.num_workers * tot_dataloaders
        worker_procs_total = train_procs * worker_procs_per_train
                
        # Reserve ~1 core per worker; split the rest across training procs
        reserve_for_workers = worker_procs_total
        avail_for_train = max(0, cpu_cores - reserve_for_workers)
        omp_num_threads = max(1, avail_for_train // train_procs)
        
        # Minimum CPU cores required for this exact config
        required_cores = reserve_for_workers + train_procs * omp_num_threads

        # oversubscription check 
        assert omp_num_threads >= 1, (
            "CPU oversubscription: with "
            f"{cpu_cores=} and {worker_procs_total=} workers, "
            "per-proc OMP threads would drop below 1. Reduce workers or increase CPUs.")        

        # Apply to *training* processes
        os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)
        os.environ["MKL_NUM_THREADS"] = str(omp_num_threads)
        torch.set_num_threads(omp_num_threads)   # intra-op
        torch.set_num_interop_threads(1)

        if is_main_process():
            print(f"→ CPUs provided: {cpu_cores}, "
                  f" DataLoader workers (total): {worker_procs_total} "
                  f" OMP_NUM_THREADS per training proc: {omp_num_threads} "
                  f" Minimum CPU cores required for this config: {required_cores}")
        
    else:
        # --- set up gpu ----
        devices = DeviceManager.list_devices()
        device = devices[args.device_idx] if devices else 'cpu'
        # --- use default cpus ---
        
    # ---- determine data configuration based on {choice} ----
    if choice != 'FM':
        patch_size  = DATA_CONFIG[choice]['patch_size']
        max_patches = DATA_CONFIG[choice]['max_patches']
        max_fields = DATA_CONFIG[choice]['fields']
        max_components = DATA_CONFIG[choice]['components']
        
    else: # default
        patch_size  = 8
        max_patches = 4096
        max_fields = 3
        max_components = 3

    if is_main_process(): 
        print(f'→ [Max.] num_tokens: {max_patches}, fields: {max_fields}, '
              f'components: {max_components}')
    
    ##########################################################################
    ###################### ---- Data Loading ---- ############################
    ##########################################################################
    # ----Train and val loaders are appended iterable datasets----
    if args.ar_order > 1: 
        if is_main_process(): print(f'→ Current AR order = {args.ar_order}')
    itterdatasets = DataStreamersChaos(DATA_CONFIG, ar_order, batch_sizes, 
                    args.num_workers, args.pin_flag, args.persist_flag,
                    args.chunk_mhd, args.chunk_dr, args.chunk_cfd1d, args.chunk_cfd2dic,
                    args.chunk_cfd3d, args.chunk_sw,   # pretraining 
                    args.chunk_dr1d, args.chunk_cfd2d, args.chunk_cfd3d_turb, 
                    args.chunk_be1d, args.chunk_gsdr2d, args.chunk_tgc3d,
                    args.chunk_fnskf2d) # finetuning
    
    train_loaders, val_loaders = itterdatasets.datastreamers(choice) # dataloaders
    train_stats, val_stats = itterdatasets.lengths()  # find lengths of datasets
    
    # ---- determine total number of batches ----
    n_samp_all, n_batches_all = [], []
    if is_main_process(): print("=== Dataset sample / batch counts ===")
    for name in train_stats:
        n_samp, n_batch = train_stats[name]
        n_samp_all.append(n_samp)
        n_batches_all.append(n_batch)
        if is_main_process():
            bs = batch_sizes[name]
            print(f" {name:7s} → train: {n_samp:8d} samples, batch size: {bs}"
                  f", {n_batch:5d} batches")
    
    for name in val_stats:
        n_samp, n_batch = val_stats[name]
        if is_main_process():
            bs = batch_sizes[name]
            print(f" {name:7s} →   val: {n_samp:8d} samples, batch size: {bs}"
                  f", {n_batch:5d} batches")
    
    # batch ratio and inverse batch ratio ==> used for task weights
    total_batches = sum(n_batches_all)
    br     = [round(n / max(n_batches_all), 2) for n in n_batches_all]
    inv_br = [max(n_batches_all) / n for n in n_batches_all]
    
    # build task_weights
    dataset_names = list(train_stats.keys())
    task_weights = {name: inv_br[i] for i,name in enumerate(dataset_names)}
    
    # check if task weights are same as train_stats
    assert len(task_weights) == len(train_stats), \
    f"Expected {len(train_stats)} tasks, got {len(task_weights)}"
    
    if is_main_process() and choice == 'FM': 
        print("Task weights → " + ", ".join(f"{v:.2f}" 
              for v in task_weights.values()))
        
    if is_main_process(): 
        print(f"Total train batches → {total_batches}")
        print(f"Ratio w.r.t max batches → {br}")
    
    # sample the dataset according to task weights
    train_scores, val_scores = itterdatasets.data_sampling(task_weights)
    
    # ---- weighted multi source iterations ----
    '''Datastreamer level sharding'''
    tr_loader = WeightedMultiSourceIterableDataset_1(*train_loaders, 
                                                     weights=train_scores)
    va_loader = WeightedMultiSourceIterableDataset_1(*val_loaders,   
                                                     weights=val_scores)
    
    # ---- Sampling probabilties of each dataset ----
    final_weights = tr_loader.weights
    if is_main_process(): print("Sampling p_k → ", [f"{final_weights[i]:.2f}" 
                                for i in range(len(final_weights))])
    
    ##########################################################################    
    ####################### ---- Model setup ---- ############################
    ##########################################################################
    
    # instantiate with this “initial” shape; future calls to forward()
    # will reconfigure it automatically when DR batches arrive.
    model_name = f'morph-{args.model_size}-{choice}-max_ar{args.max_ar_order}'
    if args.max_ar_order > 1: 
        if is_main_process(): print(f'→ Max_AR_order = {args.max_ar_order}')
    if args.activated_ar1k:
        if is_main_process(): print('→ Activated AR1K training')
    model = ViT3DRegression(patch_size = patch_size, dim = dim, depth = depth,
            heads = heads, heads_xa = args.heads_xa, mlp_dim = mlp_dim,
            max_components = max_components, conv_filter = filters, 
            max_ar = max_ar_order, max_patches = max_patches, max_fields = max_fields,
            dropout = dropout, emb_dropout = emb_dropout,
            lora_r_attn = 0, lora_r_mlp = 0,
            lora_alpha = None, lora_p = 0.0,
            model_size = args.model_size, 
            activated_ar1k = args.activated_ar1k).to(device)
    
    num_params_model = sum(p.numel() for p in model.parameters()) / 1e6
    if is_main_process(): print(f"→ NUMBER OF PARAMETERS OF THE ViT (in millions): "
                                f"{num_params_model:.3g}")
    # print('Model architecture:', model)
    if is_main_process(): 
        print(f"Training on Dataset ==> {choice}" if choice != 'FM' 
              else "→ Training on All Datasets")
    
    ###########################################################################
    ######## ---- DistributedDataParallel vs DataParallel ---- ################
    ###########################################################################
    
    if args.parallel == 'ddp':
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_gpus = torch.cuda.device_count()
        
        # Gather hostnames from all ranks
        host = socket.gethostname()
        host_list = [None] * world_size
        dist.all_gather_object(host_list, host)
        
        if is_main_process():
            num_nodes = len(set(host_list))
            print(f"\n → DDP on {world_size} GPUs across {num_nodes} node(s) \t")
        
        # Print per-rank info from every rank
        print(f"[Rank {rank}/{world_size}] "
              f"Host={host}, node has {local_gpus} GPUs, "
              f"using cuda:{local_rank}")
        
        dist.barrier(device_ids=[local_rank])
        model = nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], 
                output_device = local_rank,
                find_unused_parameters=True)
        
    elif args.parallel == 'dp':
        if torch.cuda.device_count() > 1:
            print(f'→ DataParallel on {torch.cuda.device_count()} GPUs')
            model = nn.DataParallel(model)
        else:
            print('→ Only one GPU available, running unwrapped')
    
    elif args.parallel == 'no':
        print('→ Single-GPU mode, running unwrapped')

    ##########################################################################
    ################### ---- loss and optimizer ------- ######################
    ##########################################################################
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                          factor=0.5, patience=5)
    
    ##########################################################################
    ##################### ---- resume logic ---- #############################
    ##########################################################################
    
    start_epoch = 0
    if args.resume and args.ckpt_name is None:
        args.parser.error("--resume requires you to also pass --ckpt_name <CHECKPOINT>")
    
    if args.ckpt_name:
        resume_path = os.path.join(savepath_model, f'{choice}', args.ckpt_name)
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict"))
        
        if state_dict is None:
            raise KeyError("Checkpoint missing model_state_dict/state_dict")
        
        # pick the real model if wrapped
        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            target = model.module 
        else:
            target = model
        
        # normalize keys (strip "module." only where present)
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
            
        # If we are going to fine-tune only a subset (LN + pos + att_t), load model weights
        # with strict=False and DO NOT restore optimizer/scheduler.
        finetune_subset = bool(getattr(args, "finetune_ar1k", False) and args.ar_order > 1)
        
        incompatible = target.load_state_dict(state_dict, strict=not finetune_subset)
        if is_main_process() and (incompatible.missing_keys or incompatible.unexpected_keys):
            print(f"load_state_dict: missing={incompatible.missing_keys}, "
                  f"unexpected={incompatible.unexpected_keys}")
        
        if args.finetune_ar1k and args.ar_order>1:
            if is_main_process():
                print(f"Resumed (weights only) from {resume_path}; fine-tune subset → "
                      f"skipping optimizer/scheduler restore, resetting start_epoch to 0.")
            start_epoch = 0
        else:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            
            # Optionally override LR
            if getattr(args, "new_lr_ckpt", None) is not None:
                for g in optimizer.param_groups:
                    g["lr"] = float(args.new_lr_ckpt)
                    
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt.get("epoch", 0)
            if is_main_process():
                print(f"Resumed from {resume_path}, starting at epoch {start_epoch}")
        
    ##########################################################################
    ##################### ---- AR Finetuning ---- ############################
    ##########################################################################
    
    if args.finetune_ar1k and args.ar_order > 1:
        if is_main_process(): print('→ Finetuning over AR(1)')
        FineTuner = FineTuneAR(model, ln_last_k_blocks=4, att_last_k_blocks=4, pe = None)
        optimizer = FineTuner.configure()  # returns fresh opt & sched bound to selected params
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                              factor=0.5, patience=5)
        start_epoch = 0                     # start FT counter fresh
        
    ##########################################################################
    ###################### ---- training ---- ################################
    ##########################################################################
    # free up any leftover memory from previous runs
    gc.collect()
    torch.cuda.empty_cache()
    ep_st = time.time()
    
    savepath_model_folder = os.path.join(savepath_model, f'{choice}')
    os.makedirs(savepath_model_folder, exist_ok=True)
    model_path = os.path.join(savepath_model_folder, model_name)
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(start_epoch, args.num_epochs):
        # ---- seed based shuffling of the chunk ----
        # advance epoch on all datasets
        for loader in train_loaders: loader.dataset.set_epoch(epoch)
        for loader in val_loaders: loader.dataset.set_epoch(epoch)
        
        tr_loss = Trainer.train_singlestep(model, tr_loader, criterion, optimizer, device, 
                                           epoch, scheduler, model_path, 
                                           args.save_batch_ckpt, args.save_batch_freq)
        vl_loss = Trainer.validate_singlestep(model, va_loader, criterion, device)
        
        if dist.is_initialized():
            t = torch.tensor(vl_loss, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            vl_loss = t.item() / dist.get_world_size()
    
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        
        scheduler.step(vl_loss)
        
        # Get current LR (from first param group)
        current_lr = optimizer.param_groups[0]['lr']

        if is_main_process():
            print(f"Time = {(time.time()-ep_st)/60:.2f} min., LR:{current_lr:.6f}, "
                  f"Epoch {epoch+1}/{args.num_epochs} |"
                  f"Train:{tr_loss:.5f}, Val:{vl_loss:.5f}")
        
        # --- warm-up epochs ---
        if (epoch + 1) > args.warm_epochs: 
            # --early stopping logic ---
            if vl_loss < best_val_loss:
                best_val_loss = vl_loss
                epochs_no_improve = 0
                
                # --- save checkpoint (save_every vs noverwrite) ---
                if args.overwrite_weights:
                    if is_main_process():
                        checkpoint = {"epoch": epoch + 1,
                                      "model_state_dict": model.state_dict(),
                                      "optimizer_state_dict": optimizer.state_dict(),
                                      "scheduler_state_dict": scheduler.state_dict()}
                        ckpt_path = f"{model_path}_best.pth" if args.finetune_ar1k == False else \
                            f"{model_path}_ftAR1-{args.ar_order}_best.pth"
                        torch.save(checkpoint, ckpt_path)
                        print(f" Saved (Over-write) previous checkpoint: {ckpt_path}")
                
                else:
                    if (epoch + 1) % args.save_every == 0:
                        if is_main_process():
                            checkpoint = {"epoch": epoch + 1,
                                          "model_state_dict": model.state_dict(),
                                          "optimizer_state_dict": optimizer.state_dict(),
                                          "scheduler_state_dict": scheduler.state_dict()}
                            ckpt_path = f"{model_path}_ep{epoch+1}.pth" if args.finetune_ar1k == False else \
                                f"{model_path}_ftAR1-{args.ar_order}_ep{epoch+1}.pth"
                            torch.save(checkpoint, ckpt_path)
                            print(f" Saved checkpoint: {ckpt_path}")
                        
            else:
                epochs_no_improve += 1
                if is_main_process():
                    print(f"Not improved for {epochs_no_improve}/{args.patience} epochs")
    
        if epochs_no_improve >= args.patience:
            if is_main_process(): 
                print(f"Early stopping triggered Validation loss did not "
                      f"improve for {args.patience} epochs.")
            break
        
        # free up any leftover memory before the next epoch
        gc.collect()
        torch.cuda.empty_cache()

    # --- Plot losses ---
    if is_main_process():
        fig, ax = plt.subplots()
        ax.plot(train_losses, label='Train')
        ax.plot(val_losses, label='Val')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend()
        fig.savefig(os.path.join(savepath_results, 
                    f'loss_{choice}_max_ar_{args.max_ar_order}.png'))
    
    # --- clean up DDP ----
    if args.parallel == 'ddp':
        dist.destroy_process_group()

if __name__ == "__main__":
    main()