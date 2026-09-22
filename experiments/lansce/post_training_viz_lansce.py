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
def plot_rollout(
    x_org, # true trajectory of shape (48, 15, 256, 256)
    x_pred, # predicted trajectory of shape (48, 15, 256, 256)
    rollout_dir,
    projection,
    traj_no = 0,
    time_steps=(0, 2, 4, 6, 8, 16, 32, 47),
    fs=16,
):
    """
    x_org, x_pred: arrays of shape (48, 15, 256, 256)
    projection: int in [0, 14]
    """

    ncols = len(time_steps)
    fig, axes = plt.subplots(2, ncols, figsize=(32, 8))

    for j, t in enumerate(time_steps):
        im0 = axes[0, j].imshow(x_org[t, projection], cmap="viridis", origin="lower",)
        axes[0, j].set_title(f"True t={t}", fontsize=fs)
        axes[0, j].axis("off")

        axes[1, j].imshow(x_pred[t, projection], cmap="viridis", origin="lower")
        axes[1, j].set_title(f"Pred t={t}", fontsize=fs)
        axes[1, j].axis("off")

    axes[0, 0].set_ylabel("Ground truth", fontsize=fs)
    axes[1, 0].set_ylabel("Prediction", fontsize=fs)

    fig.suptitle(f"Rollouts for test traj = {traj_no} and projection={projection}", fontsize=fs)
    fig.colorbar(im0, ax=axes, fraction=0.015, pad=0.02)
    plt.tight_layout()

    save_path = os.path.join(rollout_dir, f"ro_traj-{traj_no}_proj-{projection}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved rollout plot to: {save_path}")