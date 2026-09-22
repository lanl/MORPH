# Set cwd to the directory 
import os
from pathlib import Path
import numpy as np
import argparse
import torch
import torch.optim as optim
import wandb
import torch.nn as nn
import time
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from sklearn.metrics import precision_score, recall_score, \
      f1_score, accuracy_score, classification_report

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)
print("Current directory:", current_dir)
print("Project root:", project_root)

# built-in functions call
from experiments.ft_llnl_jag.model_morph_ft import morph_ft
from experiments.lansce.dataloading_lansce import Dataloading
from experiments.ft_llnl_jag.lr_schedulars import LRSched
from src.utils.device_manager import DeviceManager
from experiments.lansce.trainer_lansce import Trainer
from experiments.lansce.post_training_viz_lansce import (
    learning_curves, plot_rollout)
from experiments.lansce.compute_rollout_metrics import rollout_metrics

# Define important directories
dataset_dir = os.path.join(project_root, "experiments", "lansce", "datasets")
model_dir = os.path.join(project_root, "experiments", "lansce", "models")
results_dir = os.path.join(project_root, "experiments", "lansce", "results")
rollout_dir = os.path.join(project_root, "experiments", "lansce", "results", "rollouts")
metrics_dir = os.path.join(project_root, "experiments", "lansce", "results", "metrics")
print("Dataset directory:", dataset_dir)
print("Model directory:", model_dir)
print("Results directory:", results_dir)
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(rollout_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

# argument parser
parser = argparse.ArgumentParser(description="Fine-tuning MORPH on wave propagation dataset")
parser.add_argument('--hyperparameter_study', action='store_true', help="Flag for hyperparameter study")
parser.add_argument('--dev_idx', type=int, default=1, help="Device index to use")
parser.add_argument('--parallel', action='store_true', help="Use multiple GPUs for training")
parser.add_argument('--data_frac', type=float, default=1.0, help="Fraction of data to use for training")
parser.add_argument('--run_single_channel', type=int, default=None, 
                    help="If set, use only this projection channel for training (0-14)")    
parser.add_argument('--model_variant', type=str, default='S', 
                    choices=['Ti', 'S', 'M', 'L'], help="Model variant to use")
parser.add_argument('--standalone', action='store_true', help="Use standalone MORPH model without fine-tuning")
parser.add_argument('--run', choices = ['forward', 'reverse'], default = 'forward', help="forward or reverse rollouts")
parser.add_argument('--test_only', action='store_true', help="Only run testing with a pre-trained model")
parser.add_argument('--ckpt', type=str, help="Path to the checkpoint for testing")
parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
parser.add_argument('--min_lr', type=float, default=1e-9, help="Minimum learning rate for scheduler")
parser.add_argument('--warm_epochs', type=int, default=5, help="Number of warmup epochs for scheduler")
parser.add_argument('--epochs', type=int, default=500, help="Number of training epochs")
parser.add_argument('--lr_morph', type=float, default=1e-4, help="Learning rate for optimizer-1")
parser.add_argument('--wd_morph', type=float, default=1e-2, help="Weight decay for optimizer-1")
parser.add_argument('--use_abs_posenc', action='store_true',
                    help="Override learned positional encoding with fixed absolute positional encoding")
parser.add_argument('--no_log_scale', action='store_false', help="Disable log scaling of inputs and targets")
args = parser.parse_args()

# model variants
MORPH_MODELS = {
    'Ti': [8, 256,  4,  4, 1024],
    'S' : [8, 512,  8,  4, 2048],
    'M' : [8, 768, 12,  8, 3072],
    'L' : [8, 1024,16, 16, 4096]
    }

# run tag
# print args
print("===== Arguments Used =====")
for k, v in vars(args).items():
    print(f"{k}: {v}")
print("==========================")

# tags for figs save
run_tag = (
    f"standalone-{args.standalone}_"
    f"run-{args.run}_"
    f"mv-{args.model_variant}_"
    f"df-{args.data_frac}_"
    f"bs-{args.batch_size}_"
    f"ep-{args.epochs}_"
    f"lrm-{args.lr_morph}_"
    f"wdm-{args.wd_morph}_"
    f"l4-True"
)

# --- hyperparameter study ---
if args.hyperparameter_study:
    print("→ Hyperparameter study mode ON: using small settings for quick runs")
    run = wandb.init(project=f"morph_shm_finetuning_2", config=args)

# device info
devices = DeviceManager.list_devices()
device = devices[args.dev_idx] if devices else 'cpu'
print(f'→ Using device: {device}')

print(f'=== Load the data ===')
dataloading = Dataloading(dataset_dir)

# load data
psp, rf = dataloading.load_data() # load

# use only 1 projection
if args.run_single_channel is not None:
    proj_to_plot = 0
    dataloading.visualize_data(results_dir, psp, projection = proj_to_plot, plot_remarks="raw_log") # visualize
    psp = psp[:,:,args.run_single_channel:args.run_single_channel+1,:,:] # (N, T, 1, H, W)
    print(f"Using only projection {args.run_single_channel} for training, shape: {psp.shape}") 
    psp_norm, min_psp, max_psp = dataloading.MinMaxNormalizePSP(psp)  # normalize data
    dataloading.visualize_data(results_dir, psp_norm, projection = proj_to_plot, plot_remarks="norm_log") # visualize
else:
    print(f"Using all projections for training, shape: {psp.shape}")
    proj_to_plot = 11
    dataloading.visualize_data(results_dir, psp, projection = proj_to_plot, plot_remarks="raw_log") # visualize
    psp_norm, min_psp, max_psp = dataloading.MinMaxNormalizePSP(psp) # normalize data
    dataloading.visualize_data(results_dir, psp_norm, projection = proj_to_plot, plot_remarks="norm_log") # visualize

# normalize rf
rf_norm, min_rfs, max_rfs = dataloading.normalize_rfs(rf) # normalize rfs

# split the trajectories
(trainval_psp, trainval_rf), (test_psp, test_rf) = dataloading.splits(psp_norm, rf_norm) # get the tuples
print(f"Train/Val PSP shape: {trainval_psp.shape}, RF shape: {trainval_rf.shape}")
print(f"Test PSP shape: {test_psp.shape}, RF shape: {test_rf.shape}")

# convert to input and target tensors
psp_prev_tv, psp_next_tv = dataloading.nsp_f(trainval_psp) # nsp for morph
psp_prev_te, psp_next_te = dataloading.nsp_f(test_psp) # nsp for testing

# dataloading
if args.run == 'forward':
    inputs = dataloading.uptf7_f(psp_prev_tv)
    targets = dataloading.uptf7_f(psp_next_tv).squeeze(1) # remove T dimension
    print(f'Input shape: {inputs.shape}, Target shape: {targets.shape}')
    train_ds, val_ds = dataloading.datasets(inputs, targets, args.data_frac, random_state=42) # datasets
    dl_tr, dl_val = dataloading.dataloaders(train_ds, val_ds, batch_size=args.batch_size) # dataloaders
elif args.run == 'reverse':
    inputs = dataloading.uptf7_f(psp_next_tv)
    targets = dataloading.uptf7_f(psp_prev_tv).squeeze(1) # remove T dimension
    print(f'Input shape: {inputs.shape}, Target shape: {targets.shape}')
    train_ds, val_ds = dataloading.datasets(inputs, targets, args.data_frac, random_state=42) # datasets
    dl_tr, dl_val = dataloading.dataloaders(train_ds, val_ds, batch_size=args.batch_size) # dataloaders
else:
    raise ValueError("Please specify either --forward_run or "
    "--reverse_run flag to determine the input-output setup for training.")

print(f'=== Instantiate the model ===')
morph , optimizer_1 = morph_ft(model_variant= args.model_variant, 
        device = device, standalone = args.standalone, 
        rank_lora_attn=0, rank_lora_mlp=0, lora_p=0.05,
        lr_morph=args.lr_morph, wd_morph=args.wd_morph, 
        l1=False, l2=False, l3=False, l4=True,
        use_absolute_positional_encoding=args.use_abs_posenc,
        model_dir=model_dir)
modelname = f"morph_ft_lansce_{run_tag}.pth"
print('Model architecture:', morph)

# dataparallel
if args.parallel and torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for data parallelism")
    morph = nn.DataParallel(morph)

if not args.test_only:
    print(f'=== Training/Fine-tuning  ===')

    # schedular
    scheduler_1 = LRSched.warmup_cosine(optimizer_1, args)
    print(f'→ Min LR: {args.min_lr} | Warm epochs: {args.warm_epochs} '
        f'| LR: {args.lr_morph} | Weight Decay: {args.wd_morph}')
    
    # training loop
    diz_loss = {'train_loss_morph':[], 'val_loss_morph':[] }
    begin_time = time.time()

    for epoch in range(args.epochs):
        train_loss_morph = Trainer.train_epoch(dl_tr, morph, optimizer_1, device, log_scale=args.no_log_scale)
        val_loss_morph = Trainer.test_epoch(dl_val, morph, device, log_scale=args.no_log_scale)

        # --- step the schedular ---
        scheduler_1.step()        # no val loss needed

        # Get current LR (from first param group)
        current_lr_1 = optimizer_1.param_groups[0]['lr']

        # epoch time
        epoch_time = (time.time()-begin_time)/60

        print(
        f"EPOCH {epoch+1}/{args.epochs} TIME: {time.time()-begin_time:.2f}s, "
        f"Morph LR: {current_lr_1:.6e}, "
        f"train loss morph {train_loss_morph:.8f}, "
        f"val loss morph {val_loss_morph:.8f}, "
        )

        # store the losses per epoch
        diz_loss['train_loss_morph'].append(train_loss_morph)
        diz_loss['val_loss_morph'].append(val_loss_morph)

        # --- log to wandb ---
        if args.hyperparameter_study:
            run.log({"train/loss_morph": train_loss_morph, 
                    "val/loss_morph": val_loss_morph, 
                    "lr_morph": current_lr_1,
                    "epoch": epoch})

    # Save the model
    checkpoint = {"args": args,
                "morph_state_dict": morph.state_dict(),
                "diz_loss": diz_loss}
    torch.save(checkpoint, os.path.join(model_dir, modelname))

    # post training visualization
    learning_curves(diz_loss, run_tag, results_dir)
    print(f"Training visualization saved to {results_dir}")

else:
    print(f'=== No training/Fine-tuning (using checkpoint) ===')
    # load the model
    if args.ckpt is None:
        raise ValueError("Please provide a checkpoint path using --ckpt when --test_only is set.")
    
    checkpoint = torch.load(os.path.join(model_dir, args.ckpt), map_location=device)
    morph.load_state_dict(checkpoint["morph_state_dict"])
    diz_loss = checkpoint["diz_loss"]
    print(f"Model loaded from {model_dir} with name {args.ckpt}")

### --- Testing - NSP predictions on the test set ---
if args.run == 'forward':
    morph_loss_nsp = Trainer.testing_nsp(psp_prev_te, 
                                        psp_next_te,
                                        morph, 
                                        device) # default = True
else:
    morph_loss_nsp = Trainer.testing_nsp(psp_next_te, 
                                        psp_prev_te,
                                        morph, 
                                        device) # default = True

morph_loss_nsp_mean = np.mean(morph_loss_nsp) # across all trajectories
print(f"Mean Test Loss - Morph (NSP): {morph_loss_nsp_mean:.8f}")
# save the test losses as txt file
with open(os.path.join(metrics_dir, f'test_loss_nsp_{run_tag}.txt'), 'w') as f:
    f.write(f"Mean Test Loss - Morph: {morph_loss_nsp_mean:.8f}\n")
    f.write("Test Losses for each snapshot:\n")
    for idx, loss in enumerate(morph_loss_nsp):
        f.write(f"Trajectory {idx}: {loss:.8f}\n")

### --- Testing - Rollouts predictions on the test set ---
mse_avg_traj_list, mse_final_frame_list = [], []
for traj_no in range(test_psp.shape[0]):
    if args.run == 'forward':
        gt_traj_eval = test_psp[traj_no]
        input_frame = gt_traj_eval[0][None, None, :, None, None]
    else:
        gt_traj_eval = test_psp[traj_no][::-1].copy()
        input_frame = gt_traj_eval[0][None, None, :, None, None]

    # testing for rollouts
    morph_full_traj = Trainer.testing_ro(input_frame, 
                                         morph, 
                                         horizon=gt_traj_eval.shape[0] - 1, 
                                         device=device) # default = True
    mse_step, mse_avg_traj, mse_final_frame = rollout_metrics(gt_traj_eval[1:], morph_full_traj[1:])

    print(f"Rollout MSE per timestep: {mse_step}")
    print(f"Rollout MSE averaged over trajectory: {mse_avg_traj:.8f}")
    print(f"Rollout MSE for final frame: {mse_final_frame:.8f}")
    mse_avg_traj_list.append(mse_avg_traj)
    mse_final_frame_list.append(mse_final_frame)

    # plot rollouts
    plot_rollout(gt_traj_eval, 
                 morph_full_traj, 
                 rollout_dir, 
                 projection=proj_to_plot, 
                 traj_no=traj_no, fs=16)

print(f"==>> Average Rollout MSE across trajectories: {np.mean(mse_avg_traj_list):.8f}")
print(f"==>> Average Rollout MSE for final frame across trajectories: {np.mean(mse_final_frame_list):.8f}")

# save the metrics as txt file
with open(os.path.join(metrics_dir, f'test_loss_rollouts_{run_tag}.txt'), 'w') as f:
    f.write(f"Average MSE loss for all steps: {np.mean(mse_avg_traj_list):.8f}\n")
    f.write(f"Average MSE loss for final step: {np.mean(mse_final_frame_list):.8f}\n")
    f.write("Test Losses for each trajectory:\n")
    for idx, loss in enumerate(mse_avg_traj_list):
        f.write(f"Trajectory {idx}, Per trajectory MSE across all time steps: {loss:.8f}\n")
    for idx, loss in enumerate(mse_final_frame_list):
        f.write(f"Trajectory {idx}: Per trajectory MSE for final time step: {loss:.8f}\n")

# --- log test losses to wandb ---
if args.hyperparameter_study:
    run.log({
        "test/morph_nsp_mean": morph_loss_nsp_mean,
        "epoch": args.epochs
    })

# hyperparameter study finish
if args.hyperparameter_study:
    run.finish()