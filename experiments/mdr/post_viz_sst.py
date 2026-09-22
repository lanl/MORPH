import os
import matplotlib.pyplot as plt
import numpy as np
import torch

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

# idx to year and week mapping
def idx_to_year_week(global_idx, start_year=2001, weeks_per_year=52):
    year = start_year + global_idx // weeks_per_year
    week = global_idx % weeks_per_year + 1
    return [int(week), int(year)]

def visualize_target_prediction_samples(
    results_dir,
    inputs_np, 
    targets_np,
    preds_np,
    sample_idxs=(0, 100),
    plot_remarks=None,
):
    """
    inputs, targets, preds: expected shape (N, 2, H, W)

    Row 1: Voronoi input + sensor overlay
    Row 2: Target SST, no sensor overlay
    Row 3: Prediction SST, no sensor overlay
    """

    cbar_label = "SST"
    ncols = len(sample_idxs)

    fig, axes = plt.subplots(3, ncols, figsize=(10 * ncols, 12))

    if ncols == 1:
        axes = axes.reshape(3, 1)

    for j, idx in enumerate(sample_idxs):
        # convert idx to year and week for title
        year, week = idx_to_year_week(idx)

        vor_sst = inputs_np[idx, 0]
        target_sst = targets_np[idx, 0]
        pred_sst = preds_np[idx, 0]

        ocean_mask = targets_np[idx, 1] > 0.5
        sensor_mask = inputs_np[idx, 1] > 0.5

        vor_plot = np.ma.masked_where(~ocean_mask, vor_sst)
        target_plot = np.ma.masked_where(~ocean_mask, target_sst)
        pred_plot = np.ma.masked_where(~ocean_mask, pred_sst)

        # Use target scale for all three rows
        vmin = target_plot.min()
        vmax = target_plot.max()

        # Row 1: Voronoi input + sensor overlay
        last_im = axes[0, j].imshow(
            vor_plot,
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )

        sensor_rows, sensor_cols = np.where(sensor_mask)
        axes[0, j].scatter(
            sensor_cols,
            sensor_rows,
            s=10,
            c="black",
            marker="o",
        )

        axes[0, j].set_title(f"SST Voronoi input (Year={year}, Week={week})", fontsize=16)
        axes[0, j].set_xticks([])
        axes[0, j].set_yticks([])

        # Row 2: target, no sensor overlay
        axes[1, j].imshow(
            target_plot,
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )

        axes[1, j].set_title(f"SST Target (Year={year}, Week={week})", fontsize=16)
        axes[1, j].set_xticks([])
        axes[1, j].set_yticks([])

        # Row 3: prediction, no sensor overlay
        axes[2, j].imshow(
            pred_plot,
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )

        axes[2, j].set_title(f"SST Prediction/Reconstructions (Year={year}, Week={week})", fontsize=16)
        axes[2, j].set_xticks([])
        axes[2, j].set_yticks([])

    fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.85, label=cbar_label)

    if plot_remarks is not None:
        fname = f"target_prediction_samples_{plot_remarks}.png"
    else:
        fname = "target_prediction_samples.png"

    save_path = os.path.join(results_dir, fname)
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    print(f"Saved target/prediction visualization to: {save_path}")
    plt.close()