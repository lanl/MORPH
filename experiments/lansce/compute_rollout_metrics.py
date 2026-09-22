import numpy as np

def rollout_metrics(true_traj, pred_traj):
    mse_per_timestep = np.mean((true_traj - pred_traj) ** 2, axis=(1,2,3))  # shape: (horizon,)
    mse_avg_traj = np.mean((true_traj - pred_traj) ** 2)  # shape: (1,)
    mse_final_frame = np.mean((true_traj[-1] - pred_traj[-1]) ** 2)  # shape: (1,)
    return mse_per_timestep, mse_avg_traj, mse_final_frame
    