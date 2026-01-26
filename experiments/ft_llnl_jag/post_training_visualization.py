import os
import matplotlib.pyplot as plt
import numpy as np

def learning_curves(diz_loss, run_tag, results_dir):
    # print train and val loss
    plt.figure(figsize=(6, 4))
    plt.plot(diz_loss['train_loss_morph'], '-ok', label='Train',)
    plt.plot(diz_loss['val_loss_morph'], '-^r', label='Valid')
    plt.xlabel('Epoch',fontsize=20)
    plt.ylabel('Average Loss (MORPH)',fontsize=20)
    plt.legend(["tr_total", "val_total"])
    plt.title('Training & Validation loss', fontsize = 20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(os.path.join(results_dir, f'loss_ft_morph_icf_{run_tag}.png'))
    plt.close()

    # print train and val loss
    plt.figure(figsize=(6, 4))
    plt.plot(diz_loss['train_loss_head'], '-ok', label='Train',)
    plt.plot(diz_loss['val_loss_head'], '-^r', label='Valid')
    plt.xlabel('Epoch',fontsize=20)
    plt.ylabel('Average Loss (HEAD)',fontsize=20)
    plt.legend(["tr_total", "val_total"])
    plt.title('Training & Validation loss', fontsize = 20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(os.path.join(results_dir, f'loss_ft_head_icf_{run_tag}.png'))
    plt.close()

# plot y_org vs y_pred
def plot_ytrue_vs_ypred(y_org, y_pred, run_tag, results_dir):
    y_org_arr = np.concatenate(y_org, axis=0)
    y_pred_arr =  np.concatenate(y_pred, axis=0)
    print(f'y_org shape: {y_org_arr.shape}, y_pred shape: {y_pred_arr.shape}')
    plt.figure(figsize=(32, 4))
    for c in range(y_org_arr.shape[1]):
        r2 = np.corrcoef(y_org_arr[:, c], y_pred_arr[:, c])[0, 1] ** 2
        mse = np.mean((y_org_arr[:, c] - y_pred_arr[:, c])**2)
        print(f'Parameter {c+1}: R^2 = {r2:.4f}, MSE = {mse:.4f}')
        plt.subplot(1, 5, c+1)
        plt.plot(y_org_arr[:, c], y_pred_arr[:, c], 'o', label='Original vs Predicted')
        plt.title(f'Parameter {c+1}, r2 = {r2:.4f}, MSE = {mse:.4f}')
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.legend()
    plt.savefig(os.path.join(results_dir, f'ytrue_vs_ypred_{run_tag}.png'), dpi=300)
    plt.close()

# Plots of x_org vs x_pred
def plot_original_vs_predicted(x_org, x_pred, run_tag, results_dir):
    len_samples = len(x_org)
    print(f'Number of test samples: {len_samples}, each of shape: {x_org[0].shape}')
    select_samples = np.random.choice(len_samples, 5, replace=False)
    print(f'Selected samples: {select_samples}')

    for idx in select_samples:
        x_o = x_org[idx][:,0,:,:,0,:,:]       # (1, 2, 2, 1, 64, 64) -> (2, 2, 64, 64)
        x_p = x_pred[idx][:,0,:,:,0,:,:]      # (1, 2, 2, 1, 64, 64) -> (2, 2, 64, 64)
        x_o = x_o.reshape(-1, 4, 64, 64)      # (B, 4, 64, 64)
        x_p = x_p.reshape(-1, 4, 64, 64)      # (B, 4, 64, 64)
        print(f'x_org shape: {x_o.shape}, x_pred shape: {x_p.shape}')
        
        idx = np.random.randint(0, x_o.shape[0])  # pick a random sample from the batch
        x_o = x_o[idx]  # (4, 64, 64)
        x_p = x_p[idx]  # (4, 64, 64)

        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for c in range(4):
            axes[0, c].imshow(x_o[c, :, :], cmap='plasma', origin='lower')
            axes[0, c].set_title(f'Original - Channel {c}')
            axes[0, c].axis('off')

            axes[1, c].imshow(x_p[c, :, :], cmap='plasma', origin='lower')
            axes[1, c].set_title(f'Predicted - Channel {c}')
            axes[1, c].axis('off')
        plt.suptitle(f'Sample {idx}: Original vs Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'original_vs_predicted_sample_{idx}_{run_tag}.png'), dpi=300)
        plt.close()