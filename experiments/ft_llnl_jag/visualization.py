import numpy as np
import os
from matplotlib import pyplot as plt

def data_visualizer(images, scalars, params, save_dir=None):
    # --- visualize the scalars ---
    n_rows, n_cols = 3, 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10), sharey=True)

    for i in range(15):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        ax.hist(scalars[:, i], bins=50, alpha=0.7)
        ax.set_title(f'Scalar {i+1}', fontsize=14)
        ax.set_xlabel('Value', fontsize=14)
        if c == 0:
            ax.set_ylabel('Frequency', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'scalars.png'), dpi = 300)
    plt.close()

    # --- visualize the parameters ---
    n_rows, n_cols = 1, 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10), sharey=True)

    for i in range(5):
        r, c = divmod(i, n_cols)
        ax = axes[i]
        ax.hist(params[:, i], bins=50, alpha=0.7)
        ax.set_title(f'Parameter {i+1}', fontsize=14)
        ax.set_xlabel('Value', fontsize=14)
        if c == 0:
            ax.set_ylabel('Frequency', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'params.png'), dpi = 300)
    plt.close()

    # --- images visualization ---
    assert images.shape[3] == 4, "Expected 4 channels in images"

    # pick unique samples
    idxs = np.random.choice(images.shape[0], 3, replace=False)

    fig, axes = plt.subplots(len(idxs), 4, figsize=(8, 2*len(idxs)))

    # handle the case axes is 2D
    for r, idx in enumerate(idxs):
        for c in range(4):
            ax = axes[r, c]
            ax.imshow(images[idx, :, :, c], cmap='plasma', origin='lower')
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f'Channel {c}')
            if c == 0:
                ax.set_ylabel(f'Sample {idx}')

    plt.suptitle('N Samples × four channels')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'images.png'), dpi = 300)
    plt.close()