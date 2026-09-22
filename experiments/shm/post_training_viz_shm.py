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
    plt.savefig(os.path.join(results_dir, f'loss_ft_morph_shm_{run_tag}.png'))
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
    plt.savefig(os.path.join(results_dir, f'loss_ft_head_shm_{run_tag}.png'))
    plt.close()

    # print train and val accuracy
    plt.figure(figsize=(8, 4))
    plt.plot(diz_loss['train_acc_head'], '-ok', label='Train',)
    plt.plot(diz_loss['val_acc_head'], '-^r', label='Valid')
    plt.xlabel('Epoch',fontsize=20)
    plt.ylabel('Average Accuracy (HEAD)',fontsize=20)
    plt.legend(["tr_total", "val_total"])
    plt.title('Training & Validation Accuracy', fontsize = 20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'acc_ft_head_shm_{run_tag}.png'))
    plt.close()

# Plots of x_org vs x_pred
def plot_original_vs_predicted_images(x_org, x_pred, run_tag, results_dir, fs=22):
    len_samples = len(x_org)
    print(f'Number of test samples: {len_samples}, each of shape: {x_org[0].shape}')
    select_samples = np.random.choice(len_samples, 5, replace=False)
    print(f'Selected samples: {select_samples}')

    vmin, vmax = 0, 1
    for idx in select_samples:
        x_o = x_org[idx][:,0,0,0,0,0,:]       # (B, 1, 1, 1, 1, 1, 1000) -> (B, 1000)
        x_p = x_pred[idx][:,0,0,0,0,0,:]      # (B, 1, 1, 1, 1, 1, 1000) -> (B, 1000)
        
        sample_idx = np.random.randint(0, x_o.shape[0])  # pick a random sample from the batch
        x_o = x_o[sample_idx]  # (1000,)
        x_p = x_p[sample_idx]  # (1000,)

        print(f'For batch {idx} sample {sample_idx}: x_org shape: {x_o.shape}, x_pred shape: {x_p.shape}')

        plt.figure(figsize=(8, 8))
        plt.plot(x_o.flatten(), label='True', color='tab:blue')
        plt.plot(x_p.flatten(), label='Reconstructed', color='tab:orange')
        plt.xlabel('Time Steps (arbitrary units)', fontsize=fs)
        plt.ylabel('Normalized Amplitude', fontsize=fs)
        plt.xticks(fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.legend(fontsize=fs)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'original_vs_predicted_sample_{sample_idx}_{run_tag}_new.png'), dpi=300)
        plt.close()