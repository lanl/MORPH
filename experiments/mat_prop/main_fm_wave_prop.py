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

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)
print("Current directory:", current_dir)
print("Project root:", project_root)

# built-in functions call
from experiments.ft_llnl_jag.model_morph_ft import morph_ft
from experiments.ft_llnl_jag.lr_schedulars import LRSched
from src.utils.device_manager import DeviceManager
from dataloading_2_wave_prop import dataloaders
from trainers_wave_prop import Trainer
from model_tsh_wave_prop import TaskSpecificHead_FC
from post_training_visualization_wave_prop import learning_curves
from post_training_visualization_wave_prop import plot_ytrue_vs_ypred
from post_training_visualization_wave_prop import plot_original_vs_predicted_images
from compute_metrics import metrics

# Define important directories
dataset_dir = os.path.join(project_root, "experiments", "mat_prop", "datasets")
model_dir = os.path.join(project_root, "experiments", "mat_prop", "models")
results_dir = os.path.join(project_root, "experiments", "mat_prop", "results")
print("Dataset directory:", dataset_dir)
print("Model directory:", model_dir)
print("Results directory:", results_dir)
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# argument parser
parser = argparse.ArgumentParser(description="Fine-tuning MORPH on wave propagation dataset")
parser.add_argument('--hyperparameter_study', action='store_true', help="Flag for hyperparameter study")
parser.add_argument('--dev_idx', type=int, default=0, help="Device index to use")
parser.add_argument('--patch_size', type=int, default=8, help="Patch size for MORPH model")
parser.add_argument('--data_frac', type=float, default=1.0, help="Fraction of data to use for training")
parser.add_argument('--model_variant', type=str, default='S', 
                    choices=['Ti', 'S', 'M', 'L'], help="Model variant to use")
parser.add_argument('--standalone', action='store_true', help="Use standalone MORPH model without fine-tuning")
parser.add_argument('--test_only', action='store_true', help="Only run testing with a pre-trained model")
parser.add_argument('--ckpt', type=str, help="Path to the checkpoint for testing")
parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training")
parser.add_argument('--min_lr', type=float, default=1e-8, help="Minimum learning rate for scheduler")
parser.add_argument('--warm_epochs', type=int, default=5, help="Number of warmup epochs for scheduler")
parser.add_argument('--epochs', type=int, default=200, help="Number of training epochs")
parser.add_argument('--lr_morph', type=float, default=1e-4, help="Learning rate for optimizer-1")
parser.add_argument('--wd_morph', type=float, default=1e-2, help="Weight decay for optimizer-1")
parser.add_argument('--lr_head', type=float, default=1e-4, help="Learning rate for optimizer-2")
parser.add_argument('--wd_head', type=float, default=1e-2, help="Weight decay for optimizer-2")
parser.add_argument('--l1', action='store_true', help="Fine-tune level-1 parameters")
parser.add_argument('--l2', action='store_true', help="Fine-tune level-2 parameters")
parser.add_argument('--l3', action='store_true', help="Fine-tune level-3 parameters")
parser.add_argument('--l4', action='store_true', help="Fine-tune level-4 parameters")
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
    f"ps-{args.patch_size}_"
    f"df-{args.data_frac}_"
    f"bs-{args.batch_size}_"
    f"ep-{args.epochs}_"
    f"lrm-{args.lr_morph}_"
    f"wdm-{args.wd_morph}_"
    f"lrh-{args.lr_head}_"
    f"wdh-{args.wd_head}_"
    f"l1-{args.l1}_"
    f"l2-{args.l2}_"
    f"l3-{args.l3}_"
    f"l4-{args.l4}"
)

# --- hyperparameter study ---
if args.hyperparameter_study:
    print("→ Hyperparameter study mode ON: using small settings for quick runs")
    run = wandb.init(project=f"morph_waveprop_finetuning_2", config=args)

# device info
devices = DeviceManager.list_devices()
device = devices[args.dev_idx] if devices else 'cpu'
print(f'→ Using device: {device}')

# get dataloaders
imgs, params, minval_list, maxval_list, \
    dl_tr, dr_val, dr_test = dataloaders(dataset_dir, 
                                results_dir, 
                                data_frac = args.data_frac, 
                                batch_size = args.batch_size)
print(f'Images shape: {imgs.shape}, Params shape: {params.shape}')

print(f'=== Fine-tuning setup ===')
dim = MORPH_MODELS[args.model_variant][1]
N_patches_W = imgs.shape[-2] // args.patch_size
N_patches_H = imgs.shape[-1] // args.patch_size
N_patches = N_patches_W * N_patches_H
feat_per_patch = args.patch_size * args.patch_size * imgs.shape[2] * imgs.shape[3]
print(f'Number of patches along W: {N_patches_W}, H: {N_patches_H}, Total Patches: {N_patches} \n'
      f'Feat_per_patch (patch_size * patch_size * C * F): {feat_per_patch}')

print(f'=== Instantiate the model ===')
if args.l1:
    r_attn, r_mlp = 16, 12
else:
    r_attn, r_mlp = 0, 0

if not (args.l1 or args.l2 or args.l3 or args.l4):
    args.l4 = True

morph , optimizer_1 = morph_ft(model_variant= args.model_variant, 
                        device = device,
                        standalone = args.standalone, 
                        rank_lora_attn=r_attn, rank_lora_mlp=r_mlp, lora_p=0.05,
                        lr_morph=args.lr_morph, wd_morph=args.wd_morph, 
                        l1=args.l1, l2=args.l2, l3=args.l3, l4=args.l4,
                        model_dir=model_dir)
modelname = f"morph_ft_waveprop_{run_tag}.pth"

# Instantiate the task-specific head
output_dim = params.shape[1]  # number of parameters to predict
head = TaskSpecificHead_FC(n_patches = N_patches, feat_dim = dim, 
                        output_dim = output_dim).to(device)
print("Num params encoder (in K): ", sum(p.numel()//10**3 for p in head.parameters()))
print('Model architecture', head)

if not args.test_only:
    print(f'=== Training/Fine-tuning  ===')

    # schedular
    scheduler_1 = LRSched.warmup_cosine(optimizer_1, args)
    print(f'→ Min LR: {args.min_lr} | Warm epochs: {args.warm_epochs} '
        f'| LR: {args.lr_morph} | Weight Decay: {args.wd_morph}')

    # optimizer, loss and scheduler
    optimizer_2 = optim.AdamW(head.parameters(), lr = args.lr_head, weight_decay=args.wd_head)
    scheduler_2 = LRSched.warmup_cosine(optimizer_2, args)
    print(f'→ Min LR: {args.min_lr} | Warm epochs: {args.warm_epochs} '
        f'| LR: {args.lr_head} | Weight Decay: {args.wd_head}')
    
    # training loop
    diz_loss = {'train_loss_morph':[],'train_loss_head':[], 'val_loss_morph':[], 'val_loss_head':[]}
    begin_time = time.time()
    for epoch in range(args.epochs):
        train_loss_morph, train_loss_head = Trainer.train_epoch(dl_tr, morph, head, 
                                                optimizer_1, optimizer_2, device)
        val_loss_morph, val_loss_head = Trainer.test_epoch(dr_val, morph, head, device)

        # --- step the schedular ---
        scheduler_1.step()        # no val loss needed
        scheduler_2.step()        # no val loss needed

        # Get current LR (from first param group)
        current_lr_1 = optimizer_1.param_groups[0]['lr']
        print(f"→ Epoch {epoch+1}: Morph LR: {current_lr_1:.6e}")
        current_lr_2 = optimizer_2.param_groups[0]['lr']
        print(f"→ Epoch {epoch+1}: Head LR: {current_lr_2:.6e}")

        # epoch time
        epoch_time = (time.time()-begin_time)/60

        print(
        f"\n EPOCH {epoch+1}/{args.epochs} TIME: {time.time()-begin_time:.2f}s, "
        f"train loss morph {train_loss_morph:.4f}, "
        f"val loss morph {val_loss_morph:.4f}, "
        f"train loss head {train_loss_head:.4f}, "
        f"val loss head {val_loss_head:.4f}"
        )

        # store the losses per epoch
        diz_loss['train_loss_morph'].append(train_loss_morph)
        diz_loss['train_loss_head'].append(train_loss_head)
        diz_loss['val_loss_morph'].append(val_loss_morph)
        diz_loss['val_loss_head'].append(val_loss_head)

        # --- log to wandb ---
        if args.hyperparameter_study:
            run.log({"train/loss_morph": train_loss_morph, "val/loss_morph": val_loss_morph, "lr_morph": current_lr_1,
                        "train/loss_head": train_loss_head, "val/loss_head": val_loss_head, "lr_head": current_lr_2,
                        "epoch": epoch})

    # Save the model
    checkpoint = {"args": args,
                "morph_state_dict": morph.state_dict(),
                "head_state_dict": head.state_dict(),
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
    head.load_state_dict(checkpoint["head_state_dict"])
    diz_loss = checkpoint["diz_loss"]
    print(f"Model loaded from {model_dir} with name {args.ckpt}")

# get the test losses and predictions
mse_loss_main, mse_loss_head, x_org, x_pred, y_org, y_pred = Trainer.testing(dr_test, 
                                                                morph, head, device)
mean_mse_loss_main, mean_mse_loss_head = np.mean(mse_loss_main), np.mean(mse_loss_head)
print(f"Test Loss - Morph: {mean_mse_loss_main:.4f}, Head: {mean_mse_loss_head:.4f}")
print(f'Number of test samples: {len(x_org)}')
print(f'x_org shape: {x_org[0].shape}, x_pred shape: {x_pred[0].shape}')
print(f'y_org shape: {y_org[0].shape}, y_pred shape: {y_pred[0].shape}')

# calculate metrics for each parameter
mse_lst, r2_lst, mape_lst = metrics(y_org, y_pred, minval_list, maxval_list)
print("\n=== Metrics for each parameter ===")
for i, (mse, r2, mape) in enumerate(zip(mse_lst, r2_lst, mape_lst)):
    print(f"Parameter {i}: MSE = {mse:.4f}, R^2 = {r2:.4f}, MAPE = {mape:.2f}%")

# --- log test losses to wandb ---
if args.hyperparameter_study:
    run.log({
        "test/loss_morph": mean_mse_loss_main,
        "test/loss_head": mean_mse_loss_head,
        "epoch": args.epochs
    })

# plot for all parameters
plot_ytrue_vs_ypred(y_org, y_pred, run_tag, results_dir)
print(f"Y true vs Y pred visualization saved to {results_dir}")

# plot for original vs predicted images
plot_original_vs_predicted_images(x_org, x_pred, run_tag, results_dir)
print(f"Original vs Predicted images visualization saved to {results_dir}")

# hyperparameter study finish
if args.hyperparameter_study:
    run.finish()