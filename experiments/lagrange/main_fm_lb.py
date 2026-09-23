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
from experiments.lagrange.dataloading_lb import Dataloading
from experiments.ft_llnl_jag.lr_schedulars import LRSched
from morph_pde.utils.device_manager import DeviceManager
from experiments.lagrange.trainer_lb import Trainer
from experiments.lagrange.post_training_viz_lb import (
    learning_curves, visualize_true_pred)
from experiments.lagrange.compute_rollout_metrics import rollout_metrics

# Define important directories
dataset_dir = os.path.join(project_root, "experiments", "lagrange", "datasets", "2D_DAM_5740_20kevery100")
model_dir = os.path.join(project_root, "experiments", "lagrange", "models")
results_dir = os.path.join(project_root, "experiments", "lagrange", "results")
rollout_dir = os.path.join(project_root, "experiments", "lagrange", "results", "rollouts")
metrics_dir = os.path.join(project_root, "experiments", "lagrange", "results", "metrics")
print("Dataset directory:", dataset_dir)
print("Model directory:", model_dir)
print("Results directory:", results_dir)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(rollout_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

# argument parser
parser = argparse.ArgumentParser(description="Fine-tuning MORPH on wave propagation dataset")
parser.add_argument('--hyperparameter_study', action='store_true', help="Flag for hyperparameter study")
parser.add_argument('--device_idx', type=int, default=0, help="Device index to use")
parser.add_argument('--parallel', action='store_true', help="Use multiple GPUs for training")
parser.add_argument('--data_frac', type=float, default=1.0, help="Fraction of data to use for training") 
parser.add_argument('--model_variant', type=str, default='S', 
                    choices=['Ti', 'S', 'M', 'L'], help="Model variant to use")
parser.add_argument('--standalone', action='store_true', help="Use standalone MORPH model without fine-tuning")
parser.add_argument('--test_only', action='store_true', help="Only run testing with a pre-trained model")
parser.add_argument('--ckpt', type=str, help="Path to the checkpoint for testing")
parser.add_argument('--batch_size', type=int, default=96, help="Batch size for training")
parser.add_argument('--min_lr', type=float, default=1e-9, help="Minimum learning rate for scheduler")
parser.add_argument('--warm_epochs', type=int, default=5, help="Number of warmup epochs for scheduler")
parser.add_argument('--epochs', type=int, default=200, help="Number of training epochs")
parser.add_argument('--lr_morph', type=float, default=1e-4, help="Learning rate for optimizer-1")
parser.add_argument('--wd_morph', type=float, default=1e-2, help="Weight decay for optimizer-1")
parser.add_argument('--use_abs_posenc', action='store_true',
                    help="Override learned positional encoding with fixed absolute positional encoding")
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
    run = wandb.init(project=f"morph_langrange_ft", config=args)

# device info
devices = DeviceManager.list_devices()
device = devices[args.device_idx] if devices else 'cpu'
print(f'→ Using device: {device}')

print(f'=== Load the data ===')
dataloading = Dataloading(dataset_dir)

# load data: Shape of each: (num_trajs, T, Particles, Coordinates)
train_data = dataloading.load_data(split='train') # load
valid_data = dataloading.load_data(split='valid') # load
test_data = dataloading.load_data(split='test') # load

# data processing
train_data = train_data[:,:,0:5736]
valid_data = valid_data[:,:,0:5736]
test_data = test_data[:,:,0:5736]
print(f"Data shapes after processing: Train: {train_data.shape}, "
      f"Valid: {valid_data.shape}, Test: {test_data.shape}")

# visualize the data
dataloading.visualize_data(results_dir, train_data, plot_remarks='train_5736') # visualize single trajectory from the training set

# normalization
train_data_norm = dataloading.BoxNormalization(train_data, box_x_size=5.486, box_y_size=2.12)
valid_data_norm = dataloading.BoxNormalization(valid_data, box_x_size=5.486, box_y_size=2.12)
test_data_norm = dataloading.BoxNormalization(test_data, box_x_size=5.486, box_y_size=2.12)

# visualize the normalized data
dataloading.visualize_data(results_dir, train_data_norm, plot_remarks='train_norm_5736') # visualize single trajectory from the training set

# create inputs and targets for next-step prediction (NSP)
dam_prev_tr, dam_next_tr = dataloading.nsp_f(train_data_norm) # nsp for morph
dam_prev_val, dam_next_val = dataloading.nsp_f(valid_data_norm) # nsp for validation
dam_prev_te, dam_next_te = dataloading.nsp_f(test_data_norm) # nsp for testing

# dataloading
inputs_tr, targets_tr = dataloading.uptf7_f(dam_prev_tr), dataloading.uptf7_f(dam_next_tr).squeeze(1) # remove T dimension
print(f'Input shape: {inputs_tr.shape}, Target shape: {targets_tr.shape}')
inputs_val, targets_val = dataloading.uptf7_f(dam_prev_val), dataloading.uptf7_f(dam_next_val).squeeze(1) # remove T dimension
print(f'Input shape: {inputs_val.shape}, Target shape: {targets_val.shape}')
inputs_test, targets_test = dataloading.uptf7_f(dam_prev_te), dataloading.uptf7_f(dam_next_te).squeeze(1) # remove T dimension
print(f'Input shape: {inputs_test.shape}, Target shape: {targets_test.shape}')

# --- datasets and dataloaders ---
# training data
train_ds = dataloading.datasets(inputs_tr, targets_tr, data_frac = args.data_frac, split='train', random_state=42) # datasets
dl_tr = dataloading.dataloaders(train_ds, batch_size=args.batch_size, split='train') # dataloaders
# validation data
val_ds = dataloading.datasets(inputs_val, targets_val, data_frac = args.data_frac, split='valid', random_state=43) # datasets
dl_val = dataloading.dataloaders(val_ds, batch_size=args.batch_size, split='valid') # dataloaders
# testing data
test_ds = dataloading.datasets(inputs_test, targets_test, data_frac = 1.0, split='test', random_state=44) # datasets
dl_test = dataloading.dataloaders(test_ds, batch_size=args.batch_size, split='test') # dataloaders

print(f'=== Initialize the model ===')
morph , optimizer_1 = morph_ft(model_variant= args.model_variant, 
        device = device, standalone = args.standalone, 
        rank_lora_attn=0, rank_lora_mlp=0, lora_p=0.1,
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
        train_loss_morph = Trainer.train_epoch(dl_tr, morph, optimizer_1, device)
        val_loss_morph = Trainer.test_epoch(dl_val, morph, device)

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
print(f'=== Testing - NSP predictions on the test set ===')
morph_loss_nsp = Trainer.testing_nsp(inputs_test, targets_test, morph, device)
morph_loss_nsp_mean = np.mean(morph_loss_nsp) # across all trajectories
print(f"Mean Test Loss - Morph (NSP): {morph_loss_nsp_mean:.8f}")
# save the test losses as txt file
with open(os.path.join(metrics_dir, f'test_loss_nsp_{run_tag}.txt'), 'w') as f:
    f.write(f"Mean Test Loss - Morph: {morph_loss_nsp_mean:.8f}\n")
    f.write("Test Losses for each snapshot:\n")
    for idx, loss in enumerate(morph_loss_nsp):
        f.write(f"Trajectory {idx}: {loss:.8f}\n")

### --- Testing - Rollouts predictions on the test set ---
print(f'=== Testing - Rollout predictions on the test set ===')
mse_avg_steps_list, mse_steps_list = [], []
test_trajectory_count = test_data_norm.shape[0]
for traj_no in range(test_trajectory_count):
    gt_traj = test_data_norm[traj_no]  # (T, P, C)
    print(f"Testing trajectory {traj_no}: Ground truth trajectory shape: {gt_traj.shape}")

    # Normalized model input format
    gt_traj_rs = np.transpose(gt_traj, (0, 2, 1))  # (T, C, P)
    print(f"Ground truth trajectory shape (T, C, P): {gt_traj_rs.shape}")

    input_frame = gt_traj_rs[0]  # (C, P)

    morph_full_traj = Trainer.testing_ro(
        input_frame,
        morph,
        horizon=gt_traj_rs.shape[0] - 1,
        device=device 
    ) # (T, F=C, P)
    print(f"Predicted full trajectory shape: {morph_full_traj.shape}")

    # Convert prediction back to (T, P, C) before unnormalization
    morph_full_traj_pc = np.transpose(morph_full_traj, (0, 2, 1))  # (T, P, C)

    # Unnormalize in original coordinate-last format
    gt_traj_unnorm_pc = dataloading.BoxUnnormalization(
        gt_traj,
        box_x_size=5.486,
        box_y_size=2.12
    ) # (T, P, C)

    morph_full_traj_unnorm_pc = dataloading.BoxUnnormalization(
        morph_full_traj_pc,
        box_x_size=5.486,
        box_y_size=2.12
    ) # (T, P, C)

    # plot rollouts
    if traj_no == 0: # only plot for first trajectory to avoid too many figs
        visualize_true_pred(results_dir=results_dir, 
                            true_traj=gt_traj_unnorm_pc,  # (T, P, C)
                            pred_traj=morph_full_traj_unnorm_pc,  # (T, P, C)
                            t_viz=[1,2,3,4,5,10,20,30,40,50,100,200,300,400],
                            plot_remarks='test')

    assert gt_traj_unnorm_pc.shape == morph_full_traj_unnorm_pc.shape

    # compute rollout metrics
    mse_steps, mse_avg_steps = rollout_metrics(gt_traj_unnorm_pc[1:], morph_full_traj_unnorm_pc[1:])
    
    mse_steps_list.append(np.array(mse_steps))
    mse_avg_steps_list.append(mse_avg_steps)

MSE_5 = np.mean([np.mean(mse_steps_list[i][0:5]) for i in range(len(mse_steps_list))])
MSE_20 = np.mean([np.mean(mse_steps_list[i][0:20]) for i in range(len(mse_steps_list))])
print(f"==>> Average Rollout MSE for first 5 frames: {MSE_5:.8f}")
print(f"==>> Average Rollout MSE for first 20 frames: {MSE_20:.8f}")
print(f"==>> Average Rollout MSE across all frames: {np.mean(mse_avg_steps_list):.8f}")

# --- log test losses to wandb ---
if args.hyperparameter_study:
    run.log({
        "test/morph_nsp_mean": morph_loss_nsp_mean,
        "epoch": args.epochs
    })

# hyperparameter study finish
if args.hyperparameter_study:
    run.finish()