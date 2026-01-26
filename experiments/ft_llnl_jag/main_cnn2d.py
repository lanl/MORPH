# Set cwd to the directory 
import os
from pathlib import Path
import numpy as np
import argparse
import torch
from types import SimpleNamespace
import torch.optim as optim
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
from dataloading import dataloaders
from model_standalone import CNN2D
from trainers_standalone import Trainer2

# Define important directories
data_dir = os.path.join(project_root, "experiments", "ft_llnl_jag", "data")
model_dir = os.path.join(project_root, "experiments", "ft_llnl_jag", "models")
results_dir = os.path.join(project_root, "experiments", "ft_llnl_jag", "results")
print("Data directory:", data_dir)
print("Model directory:", model_dir)
print("Results directory:", results_dir)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# argument parser
parser = argparse.ArgumentParser(description="Fine-tuning MORPH on ICF-JAG dataset")
parser.add_argument('--data_frac', type=float, default=0.5, help="Fraction of data to use for training")
parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training")
parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for optimizer-1")
parser.add_argument('--wd', type=float, default=1e-5, help="Weight decay for optimizer-1")
args = parser.parse_args()

# run tag
# print args
print("===== Arguments Used =====")
for k, v in vars(args).items():
    print(f"{k}: {v}")
print("==========================")

# tags for figs save
run_tag = (
    f"df-{args.data_frac}_"
    f"bs-{args.batch_size}_"
    f"ep-{args.epochs}_"
    f"lr-{args.lr}_"
    f"wd-{args.wd}_"
)

# device info
print("Number of CUDA devices",torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

#Load ICF data
imgs, params, scalars, dl_tr, dr_val, dr_test = dataloaders(current_dir, 
                                                            model_dir, results_dir,
                                                            data_frac = args.data_frac)

# Instantiate the model
output_dim = params.shape[1]  # number of parameters to predict
scalar_dim = scalars.shape[1]  # number of scalar features
model_ss = CNN2D(in_channels = imgs.shape[1], final_out_channels = 256, 
                 scalar_dim = scalar_dim, output_dim=output_dim).to(device)
print("Num params encoder (in M): ", sum(p.numel()//10**6 for p in model_ss.parameters()))
print('Model architecture', model_ss)

# loss and optimizer
loss_fn = nn.MSELoss()
opt = optim.Adam(model_ss.parameters(), lr=args.lr, weight_decay=args.wd)

diz_loss_ss = {'train_loss':[], 'val_loss':[]}
begin_time = time.time()
for epoch in range(args.epochs):
   train_loss = Trainer2.train_epoch_ss(dl_tr, model_ss, opt, loss_fn, device)
   val_loss = Trainer2.test_epoch_ss(dr_val, model_ss, loss_fn, device)

   print(
    f"\n EPOCH {epoch+1}/{args.epochs} TIME: {time.time()-begin_time:.2f}s, "
    f"train loss {train_loss:.4f}, "
    f"val loss {val_loss:.4f}"
)

   # store the losses per epoch
   diz_loss_ss['train_loss'].append(train_loss)
   diz_loss_ss['val_loss'].append(val_loss)

# Save the model
checkpoint = {"args": args,
            "model_state_dict": model_ss.state_dict(),
            "diz_loss": diz_loss_ss}
torch.save(checkpoint, os.path.join(model_dir, f"model_ss_icf_{run_tag}.pth"))