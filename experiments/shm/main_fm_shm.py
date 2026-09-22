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
from experiments.shm.dataloading_shm import Dataloading
from experiments.ft_llnl_jag.model_morph_ft import morph_ft
from experiments.ft_llnl_jag.lr_schedulars import LRSched
from src.utils.device_manager import DeviceManager
from experiments.shm.model_tsh_shm import TaskSpecificHead_FC
from experiments.shm.trainers_shm import Trainer
from experiments.shm.metrics import Metrics
from experiments.shm.post_training_viz_shm import learning_curves, \
    plot_original_vs_predicted_images

# Define important directories
dataset_dir = os.path.join(project_root, "experiments", "shm", "datasets", "tapered_composite_wing")
model_dir = os.path.join(project_root, "experiments", "shm", "models")
results_dir = os.path.join(project_root, "experiments", "shm", "results")
print("Dataset directory:", dataset_dir)
print("Model directory:", model_dir)
print("Results directory:", results_dir)
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# argument parser
parser = argparse.ArgumentParser(description="Fine-tuning MORPH on wave propagation dataset")
parser.add_argument('--hyperparameter_study', action='store_true', help="Flag for hyperparameter study")
parser.add_argument('--dev_idx', type=int, default=1, help="Device index to use")
parser.add_argument('--patch_size', type=int, default=8, help="Patch size for MORPH model")
parser.add_argument('--data_frac', type=float, default=1.0, help="Fraction of data to use for training")
parser.add_argument('--model_variant', type=str, default='Ti', 
                    choices=['Ti', 'S', 'M', 'L'], help="Model variant to use")
parser.add_argument('--standalone', action='store_true', help="Use standalone MORPH model without fine-tuning")
parser.add_argument('--test_only', action='store_true', help="Only run testing with a pre-trained model")
parser.add_argument('--ckpt', type=str, help="Path to the checkpoint for testing")
parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training")
parser.add_argument('--min_lr', type=float, default=1e-8, help="Minimum learning rate for scheduler")
parser.add_argument('--warm_epochs', type=int, default=5, help="Number of warmup epochs for scheduler")
parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
parser.add_argument('--lr_morph', type=float, default=1e-4, help="Learning rate for optimizer-1")
parser.add_argument('--wd_morph', type=float, default=1e-2, help="Weight decay for optimizer-1")
parser.add_argument('--lr_head', type=float, default=1e-5, help="Learning rate for optimizer-2")
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
    run = wandb.init(project=f"morph_shm_finetuning_2", config=args)

# device info
devices = DeviceManager.list_devices()
device = devices[args.dev_idx] if devices else 'cpu'
print(f'→ Using device: {device}')

print(f'=== Load the data ===')
dataloading = Dataloading(dataset_dir)
df, df_base, df_dam, time_stamps, rmsd = dataloading.load_data() # load
dataloading.plot_rmsd(results_dir, rmsd) # plot RMSD
dataloading.visualize_data(results_dir, df, df_base, df_dam, time_stamps) # visualize
labels = dataloading.create_labels_detection(df_base, df_dam) # create labels
X, df_mu, df_sigma = dataloading.normalize_data(df) # normalize data
X = dataloading.uptf7(X) # convert to UPTF7 format
train_ds, val_ds, test_ds = dataloading.datasets(X, labels, data_frac = args.data_frac, random_state=42) # datasets
dl_tr, dl_val, dl_test = dataloading.dataloaders(train_ds, val_ds, test_ds, batch_size=args.batch_size) # dataloaders
print(f'Shape of X: {X.shape}, Shape of labels: {labels.shape}')

print(f'=== Fine-tuning setup ===')
dim = MORPH_MODELS[args.model_variant][1]
N_patches_W = X.shape[-1] // args.patch_size  # patches along width
patch_size_W = args.patch_size # patches along width
N_patches_H = X.shape[-2] // args.patch_size if X.shape[-2] != 1 else 1 # patches along height
patch_size_H = args.patch_size if X.shape[-2] != 1 else 1 # patches along height

# total patches and features per patch
N_patches = N_patches_W * N_patches_H 
feat_per_patch = patch_size_W * patch_size_H * X.shape[2] * X .shape[3]

print(f'Number of patches along W: {N_patches_W}, H: {N_patches_H}, Total Patches: {N_patches}')
print(f'Feat_per_patch along W: {patch_size_W}, H: {patch_size_H}, Total Feats per patch: {feat_per_patch}')

print(f'=== Instantiate the model ===')
if args.l1:
    r_attn, r_mlp = 16, 12
else:
    r_attn, r_mlp = 0, 0

if not (args.l1 or args.l2 or args.l3 or args.l4):
    args.l4 = True

morph , optimizer_1 = morph_ft(model_variant= args.model_variant, 
        device = device, standalone = args.standalone, 
        rank_lora_attn=r_attn, rank_lora_mlp=r_mlp, lora_p=0.05,
        lr_morph=args.lr_morph, wd_morph=args.wd_morph, 
        l1=args.l1, l2=args.l2, l3=args.l3, l4=args.l4,
        model_dir=model_dir)
modelname = f"morph_ft_shm_{run_tag}.pth"

# Instantiate the task-specific head
output_dim = labels.shape[1]  # number of parameters to predict
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
    diz_loss = {'train_loss_morph':[],'train_loss_head':[], 
                'val_loss_morph':[], 'val_loss_head':[],
                'train_acc_head':[], 'val_acc_head':[]}
    begin_time = time.time()
    for epoch in range(args.epochs):
        train_loss_morph, train_loss_head, train_acc = Trainer.train_epoch(
            dl_tr, morph, head, optimizer_1, optimizer_2, device)
        val_loss_morph, val_loss_head, val_acc = Trainer.test_epoch(
            dl_val, morph, head, device)

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
        f"val loss head {val_loss_head:.4f}, "
        f"train acc head {train_acc:.4f}, "
        f"val acc head {val_acc:.4f}"
        )

        # store the losses per epoch
        diz_loss['train_loss_morph'].append(train_loss_morph)
        diz_loss['val_loss_morph'].append(val_loss_morph)
        diz_loss['val_loss_head'].append(val_loss_head)
        diz_loss['train_acc_head'].append(train_acc)
        diz_loss['val_acc_head'].append(val_acc)

        # --- log to wandb ---
        if args.hyperparameter_study:
            run.log({"train/loss_morph": train_loss_morph, 
                    "val/loss_morph": val_loss_morph, 
                    "lr_morph": current_lr_1,
                    "train/loss_head": train_loss_head, 
                    "val/loss_head": val_loss_head, 
                    "train/acc_head": train_acc,
                    "val/acc_head": val_acc,
                    "lr_head": current_lr_2,
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
loss_main, loss_head, x_org, x_pred, y_org, y_prob, y_pred, test_acc = Trainer.testing(
    dl_test, morph, head, device)
mean_mse_loss_main, mean_mse_loss_head = np.mean(loss_main), np.mean(loss_head)
print(f"Test Loss - Morph: {mean_mse_loss_main:.4f}, Head: {mean_mse_loss_head:.4f}")
print(f'Number of test samples: {len(x_org)}')
print(f'x_org shape: {x_org[0].shape}, x_pred shape: {x_pred[0].shape}')
print(f'y_org shape: {y_org[0].shape}, y_pred shape: {y_pred[0].shape}')

# calculate metrics for each parameter
y_true = np.concatenate(y_org, axis=0).reshape(-1).astype(int)
y_hat  = np.concatenate(y_pred, axis=0).reshape(-1).astype(int)
test_precision = precision_score(y_true, y_hat, zero_division=0)
test_recall    = recall_score(y_true, y_hat, zero_division=0)
test_f1        = f1_score(y_true, y_hat, zero_division=0)
test_acc       = accuracy_score(y_true, y_hat)

print(f"Precision = {test_precision:.4f}")
print(f"Recall    = {test_recall:.4f}")
print(f"F1 Score  = {test_f1:.4f}")
print(f"Accuracy  = {test_acc:.4f}")

# save precision, recall, f1-score, accuracy to a text file
with open(os.path.join(results_dir, f"metrics_shm_{run_tag}.txt"), 'w') as f:
    f.write(f"Test Precision: {test_precision:.4f}\n")
    f.write(f"Test Recall: {test_recall:.4f}\n")
    f.write(f"Test F1 Score: {test_f1:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")

# --- log test losses to wandb ---
if args.hyperparameter_study:
    run.log({
        "test/loss_morph": mean_mse_loss_main,
        "test/loss_head": mean_mse_loss_head,
        "test/acc_head": test_acc,
        "test/recall": test_recall,
        "test/precision": test_precision,
        "test/f1_score": test_f1,
        "epoch": args.epochs
    })

# plot for original vs predicted images
plot_original_vs_predicted_images(x_org, x_pred, run_tag, results_dir)
print(f"Original vs Predicted images visualization saved to {results_dir}")

# hyperparameter study finish
if args.hyperparameter_study:
    run.finish()