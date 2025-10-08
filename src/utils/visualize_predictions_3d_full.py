import matplotlib.pyplot as plt
import torch
import os 

class Visualize3DPredictions:
    def __init__(self, model, test_dataset, device):
        self.model = model
        self.test_dataset = test_dataset
        self.device = device
        
    def visualize_predictions(self, time_step = 0, component = 0, slice_dim='d', 
                              slice_pos=None, save_path = None,
                              figname = 'ViT_CC_XAF_AAST.png'):
        self.model.eval()
        with torch.no_grad():
            # Get a random sample from validation set 
            input_vol = self.test_dataset[:,time_step]      # [B=1,F,C,D,H,W]
            target_vol = self.test_dataset[:,time_step + 1] # [B=1,F,C,D,H,W]
            
            # Perform prediction
            '''
            input_tensor shape [N,T,F,C,D,H,W]
            '''
            input_tensor = input_vol.unsqueeze(1).to(self.device) # [B=1,T=1,F,C,D,H,W]        
            prediction = self.model(input_tensor).cpu()           # [B=1,F,C,D,H,W]
            
            #print(f'input:{input_tensor.shape}, pred:{prediction.shape}')
            # Get shapes
            fields = input_vol.shape[1]
            
            # Use a component for plotting (usually 0)
            inp = input_vol.squeeze(0)[:,:,component]               # [T=1,F,C,D,H,W]    
            tar = target_vol.squeeze(0)[:,:,component]              # [F,C,D,H,W]
            pred = prediction.squeeze(0)[:,:,component]             # [F,C,D,H,W]
            
            #print(f'inp;{inp.shape}, tar;{tar.shape}, pred;{pred.shape}')
        
            # Create figure – now 5 columns instead of 3
            fig, axes = plt.subplots(fields, 5, figsize=(5 * 5, 5 * fields))
            if fields == 1:
                axes = axes.reshape(1, -1)
            
            # Determine slice positions if not provided (center of the volume)
            if slice_pos is None:
                if slice_dim == 'd':
                    slice_pos = inp.shape[1] // 2 # if inp.shape[1] == 1, then == 0
                elif slice_dim == 'h':
                    slice_pos = inp.shape[2] // 2
                elif slice_dim == 'w':
                    slice_pos = inp.shape[3] // 2
                elif slice_dim == '1d':
                    slice_pos = 0
                    
            # Plot for each channel
            for field in range(fields):
                # Extract slices based on the chosen dimension
                # To avoid plotting 3d
                if slice_dim == 'd':
                    input_slice = inp[field, slice_pos, :, :].numpy()
                    target_slice = tar[field, slice_pos, :, :].numpy()
                    pred_slice = pred[field, slice_pos, :, :].numpy()
                    slice_info = f'd={slice_pos}'
                elif slice_dim == 'h':
                    input_slice = inp[field, :, slice_pos, :].numpy()
                    target_slice = tar[field, :, slice_pos, :].numpy()
                    pred_slice = pred[field, :, slice_pos, :].numpy()
                    slice_info = f'h={slice_pos}'
                elif slice_dim == 'w':
                    input_slice = inp[field, :, :, slice_pos].numpy()
                    target_slice = tar[field, :, :, slice_pos].numpy()
                    pred_slice = pred[field, :, :, slice_pos].numpy()
                    slice_info = f'w={slice_pos}'
                elif slice_dim == '1d':
                    input_slice = inp[field, slice_pos, slice_pos, :].numpy()
                    target_slice = tar[field, slice_pos, slice_pos, :].numpy()
                    pred_slice = pred[field, slice_pos, slice_pos, :].numpy()
                    slice_info = f'w={slice_pos}'
                
                # Compute pixelwise MSE maps
                mse_tp = (target_slice - pred_slice) ** 2
                mse_ip = (input_slice  - pred_slice) ** 2
                
                # common scale for plotting
                vmin, vmax = input_slice.min(), input_slice.max()
                
                # Choose colormap based on channel
                if field == 0:
                    cmap = 'viridis'
                elif field == 1:
                    cmap = 'plasma'
                elif field == 2:
                    cmap = 'inferno'
                
                # Plot input, target, and prediction for this channel
                names = ['Input', 'Target', 'Prediction', 'MSE(Target,Prediction)', 'MSE(Input,Prediction)']
                slices = [input_slice, target_slice, pred_slice, mse_tp, mse_ip]
                
                # Plot all five columns
                for col, (name, img) in enumerate(zip(names, slices)):
                    ax = axes[field, col]
                    if slice_dim == '1d':
                        # img is a 1D array in this case
                        ax.plot(img, label=name)
                        ax.set_title(f'{name} Ch{field} ({slice_info})', fontsize=18)
                        ax.set_ylim(vmin, vmax)
                        ax.grid(True)
                        ax.legend()
                    else:
                        if col < 3:
                            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
                        else:
                            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)  # 10% of the scale
                        ax.set_title(f'{name} Ch{field} ({slice_info})', fontsize = 18)
                        ax.axis('off')
                        fig.colorbar(im, ax=ax)
                    
                plt.tight_layout()
                plt.subplots_adjust(top=0.92)
                plt.suptitle(f'Time step: {time_step}')
                # plt.show()
            
            if save_path:
                fig_name  = os.path.join(save_path, figname)
                fig.savefig(fig_name, bbox_inches='tight')
                print(f"Figure saved to: {fig_name}")
                
            plt.close(fig)
