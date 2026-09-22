import os
import matplotlib.pyplot as plt
import numpy as np

def learning_curves(diz_loss, run_tag, results_dir):
    # print train and val loss
    plt.figure(figsize=(8, 4))
    plt.plot(diz_loss['train_loss_morph'], '-ok', label='Train',)
    plt.plot(diz_loss['val_loss_morph'], '-^r', label='Valid')
    plt.xlabel('Epoch',fontsize=20)
    plt.ylabel('Average Loss (MORPH)',fontsize=20)
    plt.legend(["tr_total", "val_total"])
    plt.title('Training & Validation loss', fontsize = 20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'loss_ft_morph_lansce_{run_tag}.png'))
    plt.close()

# plot rollouts
def visualize_true_pred(results_dir, true_traj, pred_traj,
                        t_viz=None, plot_remarks='test'):

    T, P, C = true_traj.shape
    print(f"Visualizing true trajectory shape: {true_traj.shape}")
    print(f"Visualizing pred trajectory shape: {pred_traj.shape}")

    # Choose 10 time steps evenly across the trajectory
    if t_viz is None:
        t_viz = np.linspace(0, T - 1, 10, dtype=int)

    fig, axes = plt.subplots(2, len(t_viz), figsize=(len(t_viz) * 3, 6))

    for i, t in enumerate(t_viz):
        # Ground truth row
        axes[0, i].scatter(true_traj[t, :, 0], true_traj[t, :, 1], s=10)
        axes[0, i].set_title(f'Time {t}')
        axes[0, i].set_aspect('equal')

        # Predicted row
        axes[1, i].scatter(pred_traj[t, :, 0], pred_traj[t, :, 1], s=10)
        axes[1, i].set_aspect('equal')

    axes[0, 0].set_ylabel('Ground Truth')
    axes[1, 0].set_ylabel('Predicted')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'true_vs_pred_trajectory_{plot_remarks}.png'))
    plt.close()