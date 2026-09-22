import numpy as np

def rollout_metrics(true_traj, pred_traj):
    mse_per_timestep = np.mean((true_traj - pred_traj) ** 2, axis=(1,2))  # shape: (horizon,)
    mse_avg_traj = np.mean((true_traj - pred_traj) ** 2)  # shape: (1,)
    return mse_per_timestep, mse_avg_traj
    