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
from experiments.mdr.dataloading_sst import Dataloading
from experiments.mdr.pre_viz_sst import PreVisualization
from experiments.ft_llnl_jag.lr_schedulars import LRSched
from src.utils.device_manager import DeviceManager
from experiments.mdr.trainer_sst import Trainer
from experiments.mdr.post_viz_sst import learning_curves, visualize_target_prediction_samples
# from experiments.lansce.compute_rollout_metrics import rollout_metrics

# Define important directories
dataset_dir = os.path.join(project_root, "experiments", "mdr", "datasets", "noaa_sst")
model_dir = os.path.join(project_root, "experiments", "mdr", "models")
results_dir = os.path.join(project_root, "experiments", "mdr", "results")
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
parser.add_argument('--parallel', action='store_true', help="Use DataParallel for multi-GPU training")
parser.add_argument('--patch_size', type=int, default=8, help="Patch size for MORPH model")
parser.add_argument('--data_frac', type=float, default=1.0, help="Fraction of data to use for training")
parser.add_argument('--model_variant', type=str, default='S', 
                    choices=['Ti', 'S', 'M', 'L'], help="Model variant to use")
parser.add_argument('--standalone', action='store_true', help="Use standalone MORPH model without fine-tuning")
parser.add_argument('--run', choices = ['forward', 'reverse'], default = 'forward', help="forward or reverse rollouts")
parser.add_argument('--test_only', action='store_true', help="Only run testing with a pre-trained model")
parser.add_argument('--train_sensor_count', type=int, nargs='+', 
                    default=[1,5,10,20,30,40,50,60,70,80,90,100], 
                    help="Number of sensors for training")
parser.add_argument('--test_sensor_count', type=int, nargs='+', default=[100], 
                    help="Number of sensors for testing")
parser.add_argument('--ckpt', type=str, help="Path to the checkpoint for testing")
parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
parser.add_argument('--min_lr', type=float, default=1e-8, help="Minimum learning rate for scheduler")
parser.add_argument('--warm_epochs', type=int, default=5, help="Number of warmup epochs for scheduler")
parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
parser.add_argument('--lr_morph', type=float, default=1e-4, help="Learning rate for optimizer-1")
parser.add_argument('--wd_morph', type=float, default=1e-2, help="Weight decay for optimizer-1")
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
    f"testsensors-{args.test_sensor_count}_"
    f"mv-{args.model_variant}_"
    f"ps-{args.patch_size}_"
    f"df-{args.data_frac}_"
    f"bs-{args.batch_size}_"
    f"ep-{args.epochs}_"
    f"lrm-{args.lr_morph}_"
    f"wdm-{args.wd_morph}_"
    f"l4-True"  # always fine-tune level-4 for best performance
)

# --- hyperparameter study ---
if args.hyperparameter_study:
    print("→ Hyperparameter study mode ON: using small settings for quick runs")
    run = wandb.init(config=args)

# device info
devices = DeviceManager.list_devices()
device = devices[args.dev_idx] if devices else 'cpu'
print(f'→ Using device: {device}')

print(f'=== Load the data ===')
dataloading = Dataloading(dataset_dir)
viz = PreVisualization(results_dir)

# load data
sst, lat, lon, time_steps, land_mask = dataloading.load_data() # load

# idx to year and week mapping
def idx_to_year_week(global_idx, start_year=1981, weeks_per_year=52):
    year = start_year + global_idx // weeks_per_year
    week = global_idx % weeks_per_year + 1
    return [int(week), int(year)]

# visualize the data
if not args.hyperparameter_study:
    print("=== Visualize the raw data ===")
    for t in [0, 10, 100, 500, 1000]: # snapshot (corresponding to weeks)
        my = idx_to_year_week(t)  # print the corresponding year and week for the snapshot
        print(f"Snapshot {t}: Week {my[0]}, Year {my[1]}")
        viz.visualize_data(sst, lat, lon, t = t, 
                        title_remarks = f"Week {my[0]}, Year {my[1]}", 
                        save_remarks='raw_data') # visualize

# split the data
trainval_idx = np.arange(0, 1040)
test_idx = np.arange(1040, sst.shape[0])

sst_min, sst_max = dataloading.compute_sst_minmax(
        sst, lat, lon, trainval_idx
)

print("=== multi sensor set-up ===")
train_sensor_counts = args.train_sensor_count
train_sensor_seeds = [300]

X_train, Y_train, train_sensor_locs = dataloading.build_multi_sensor_dataset(
    dataloading=dataloading,
    sst=sst,
    lat=lat,
    lon=lon,
    t_indices=trainval_idx,
    sensor_counts=train_sensor_counts,
    sensor_seeds=train_sensor_seeds,
    sst_min=sst_min,
    sst_max=sst_max,
    split_name="train",
)

# Evaluate on one sensor count
test_sensor_counts = args.test_sensor_count
test_sensor_seeds = [np.random.randint(0, 10000, 1).item()]

X_test, Y_test, test_sensor_locs = dataloading.build_multi_sensor_dataset(
    dataloading=dataloading,
    sst=sst,
    lat=lat,
    lon=lon,
    t_indices=test_idx,
    sensor_counts=test_sensor_counts,
    sensor_seeds=test_sensor_seeds,
    sst_min=sst_min,
    sst_max=sst_max,
    split_name="test",
)

# pad the data
X_train_padH, Y_train_padH = dataloading.pad_data(X_train, Y_train, sst_min=sst_min, sst_max=sst_max)
X_test_padH, Y_test_padH = dataloading.pad_data(X_test, Y_test, sst_min=sst_min, sst_max=sst_max)

# visualize voronoi and sensor mask for one time step
if not args.hyperparameter_study:
    print("=== Visualize the padded data (normalized) in voronoi ===")

    nsensors = 100
    selected_idx = train_sensor_counts.index(nsensors)

    n_train_snapshots = X_train.shape[0] // len(train_sensor_counts)

    print(f"Using {nsensors} sensors")
    print(f"Sensor-count block index = {selected_idx}")
    print(f"Snapshots per sensor count = {n_train_snapshots}")

    for raw_t in [0, 10, 100, 500, 1000]:
        my = idx_to_year_week(raw_t)

        print(f"Raw snapshot {raw_t}: Week {my[0]}, Year {my[1]}")

        plot_idx = raw_t + n_train_snapshots * selected_idx

        print(f"Plotting X_train index = {plot_idx}")

        viz.visualize_voronoi(
            X_train,
            idx=plot_idx,
            title_remarks=f"{nsensors} sensors | Week {my[0]}, Year {my[1]}",
            save_remarks=f'pad_data_norm_voronoi_{nsensors}_sensors_t{raw_t}'
        )

# transform the data to uptf7 format
print("=== UPTF7, Datasets and Dataloaders ===")
X_train_uptf7 = dataloading.uptf7(X_train_padH, set='inputs')
Y_train_uptf7 = dataloading.uptf7(Y_train_padH, set='targets')
X_test_uptf7 = dataloading.uptf7(X_test_padH, set='inputs')
Y_test_uptf7 = dataloading.uptf7(Y_test_padH, set='targets')
print(f"Uptf7 transformed shapes: X_train_uptf7={X_train_uptf7.shape}, "
    f"Y_train_uptf7={Y_train_uptf7.shape}, "
    f"X_test_uptf7={X_test_uptf7.shape}, Y_test_uptf7={Y_test_uptf7.shape}")

# datasets and dataloaders
train_ds = dataloading.datasets(X_train_uptf7, Y_train_uptf7, data_frac = args.data_frac, split='train', random_state=42)
dl_train = dataloading.dataloaders(train_ds, batch_size=args.batch_size, shuf = True, split='train')
val_ds = dataloading.datasets(X_test_uptf7, Y_test_uptf7, data_frac = args.data_frac, split='test', random_state=42)
dl_val = dataloading.dataloaders(val_ds, batch_size=args.batch_size, shuf = True, split='val')
test_ds = dataloading.datasets(X_test_uptf7, Y_test_uptf7, data_frac = 1.0, split='test', random_state=42)
dl_test = dataloading.dataloaders(test_ds, batch_size=args.batch_size, shuf = False, split='test')

print(f'=== Instantiate the model ===')
morph , optimizer_1 = morph_ft(model_variant= args.model_variant, 
        device = device, standalone = args.standalone, 
        rank_lora_attn=0, rank_lora_mlp=0, lora_p=0.05,
        lr_morph=args.lr_morph, wd_morph=args.wd_morph, 
        l1=False, l2=False, l3=False, l4=True,
        model_dir=model_dir)
modelname = f"morph_ft_noaa-sst_{run_tag}.pth"

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
        train_loss_morph = Trainer.train_epoch(dl_train, morph, optimizer_1, device)
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
    model_to_save = morph.module if isinstance(morph, nn.DataParallel) else morph
    checkpoint = {
        "args": vars(args),
        "morph_state_dict": model_to_save.state_dict(),
        "diz_loss": diz_loss,
    }
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
    model_to_load = morph.module if isinstance(morph, nn.DataParallel) else morph
    model_to_load.load_state_dict(checkpoint["morph_state_dict"])
    diz_loss = checkpoint["diz_loss"]
    print(f"Model loaded from {model_dir} with name {args.ckpt}")

### --- Testing - NSP predictions on the test set ---
inp, tar, pred, mse_nsp, l2_error_norm = Trainer.test_epoch_nsp(dl_test, morph, sst_max, sst_min, device)
print(f"Shape of input frames list: {len(inp)} each with shape {inp[0].shape}")
print(f"Shape of target frames list: {len(tar)} each with shape {tar[0].shape}")
print(f"Shape of pred frames list: {len(pred)} each with shape {pred[0].shape}")
print(f"Mean Test Loss - Morph (NSP): {np.mean(mse_nsp):.8f}")
print(f"Mean Relative L2 Loss - Morph (NSP): {np.mean(l2_error_norm):.8f}")

# save the test losses as txt file
with open(os.path.join(results_dir, f'test_loss_nsp_{run_tag}.txt'), 'w') as f:
    f.write(f"Mean Test Loss - Morph: {np.mean(mse_nsp):.8f}\n")
    f.write(f"Mean Relative L2 Loss - Morph: {np.mean(l2_error_norm):.8f}\n")
    f.write("Test Losses for each trajectory:\n")

# convert uptf7 format back to (N, F, H, W)
input_frames_np = inp[:,0,:,0,0,:,:]  # shape: (N, T, F, C, D, H, W) -> (N, F, H, W)
target_frames_np = tar[:,:,0,0,:,:]  # shape: (N, F, C, D, H, W) -> (N, F, H, W)
pred_frames_np = pred[:,:,0,0,:,:]      # shape: (N, F, C, D, H, W) -> (N, F, H, W)
print(f"Converted input frames shape: {input_frames_np.shape}")
print(f"Converted target frames shape: {target_frames_np.shape}")
print(f"Converted pred frames shape: {pred_frames_np.shape}")

# visualize the predictions (not for hyperparameter study)
test_sample_idxs = [0, 400, 800]
my = [idx_to_year_week(test_idx[i]) for i in test_sample_idxs]
if not args.hyperparameter_study:
    visualize_target_prediction_samples(
        results_dir,
        inputs_np = input_frames_np,
        targets_np = target_frames_np,
        preds_np = pred_frames_np,
        sample_idxs = test_sample_idxs,
        plot_remarks = f'test_nsp_samples_test-sensor_{args.test_sensor_count}'
    )
    print(f"Test NSP sample visualizations saved to {results_dir}")

# hyperparameter study
if args.hyperparameter_study:
    run.log({
        "test/mse_nsp": float(np.mean(mse_nsp)),
        "test/rel_l2_nsp": float(np.mean(l2_error_norm)),
    })

if args.hyperparameter_study:
    run.finish()