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
    plt.savefig(os.path.join(results_dir, f'loss_ft_morph_icf_{run_tag}.png'))
    plt.close()

    # print train and val loss
    plt.figure(figsize=(8, 4))
    plt.plot(diz_loss['train_loss_head'], '-ok', label='Train',)
    plt.plot(diz_loss['val_loss_head'], '-^r', label='Valid')
    plt.xlabel('Epoch',fontsize=20)
    plt.ylabel('Average Loss (HEAD)',fontsize=20)
    plt.legend(["tr_total", "val_total"])
    plt.title('Training & Validation loss', fontsize = 20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'loss_ft_head_icf_{run_tag}.png'))
    plt.close()

# plot y_org vs y_pred
def plot_ytrue_vs_ypred(y_org, y_pred, run_tag, results_dir, params = (1,2,4), fs=22):
    y_org_arr = np.concatenate(y_org, axis=0)
    y_pred_arr =  np.concatenate(y_pred, axis=0)
    print(f'y_org shape: {y_org_arr.shape}, y_pred shape: {y_pred_arr.shape}')
    markers = ['o', 's', '^', 'x', 'D']
    colors = ['tab:blue', 'tab:orange', 'tab:purple', 'tab:green', 'tab:red']
    plt.figure(figsize=(24, 6))
    for c in range(y_org_arr.shape[1]):
        r2 = np.corrcoef(y_org_arr[:, c], y_pred_arr[:, c])[0, 1] ** 2
        mse = np.mean((y_org_arr[:, c] - y_pred_arr[:, c])**2)
        print(f'param {params[c]}: $R^2$ = {r2:.4f}, $L_2$ = {mse:.4f}')
        plt.subplot(1, y_org_arr.shape[1], c+1)
        plt.plot(y_org_arr[:, c], y_pred_arr[:, c], marker=markers[c], linestyle='None', color=colors[c])
        plt.title(f'param {params[c]}, $R^2$  = {r2:.3f}, $L_2$ = {mse:.3f}', fontsize=fs)
        plt.xlabel('True', fontsize=fs)
        plt.ylabel('Predicted', fontsize=fs)
        plt.xlim(-2.0, 2.0)
        plt.ylim(-2.0, 2.0)
        plt.xticks(np.arange(-2.0, 2.1, 1.0), fontsize=fs)
        plt.yticks(np.arange(-2.0, 2.1, 1.0), fontsize=fs)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'ytrue_vs_ypred_{run_tag}.png'), dpi=300)
    plt.close()

# Plots of x_org vs x_pred
def plot_original_vs_predicted_images(x_org, x_pred, run_tag, results_dir, fs=24):
    len_samples = len(x_org)
    print(f'Number of test samples: {len_samples}, each of shape: {x_org[0].shape}')
    select_samples = np.random.choice(len_samples, 5, replace=False)
    print(f'Selected samples: {select_samples}')

    for idx in select_samples:
        x_o = x_org[idx][:,:,:,0,0,:,:]       # (1, 4, 1, 1, 64, 64) -> (4, 1, 64, 64)
        x_p = x_pred[idx][:,:,:,0,0,:,:]      # (1, 4, 1, 1, 64, 64) -> (4, 1, 64, 64)
        x_o = x_o.reshape(-1, 4, 64, 64)      # (B, 4, 64, 64)
        x_p = x_p.reshape(-1, 4, 64, 64)      # (B, 4, 64, 64)
        print(f'x_org shape: {x_o.shape}, x_pred shape: {x_p.shape}')
        
        sample_idx = np.random.randint(0, x_o.shape[0])  # pick a random sample from the batch
        x_o = x_o[sample_idx]  # (4, 64, 64)
        x_p = x_p[sample_idx]  # (4, 64, 64)

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for c in range(4):
            axes[0, c].imshow(x_o[c, :, :], cmap='plasma', origin='lower')
            axes[0, c].set_title(f'True Ch. {c}', fontsize=fs)
            axes[0, c].axis('off')

            axes[1, c].imshow(x_p[c, :, :], cmap='plasma', origin='lower')
            axes[1, c].set_title(f'Recon. Ch. {c}', fontsize=fs)
            axes[1, c].axis('off')
        #plt.suptitle(f'Sample {sample_idx}: Original vs Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'original_vs_predicted_sample_{sample_idx}_{run_tag}.png'), dpi=300)
        plt.close()