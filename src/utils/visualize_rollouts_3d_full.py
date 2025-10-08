import matplotlib.pyplot as plt
import torch
import os 

class Visualize3DRolloutPredictions:
    def __init__(self, model, test_dataset, device, field_names, component_names):
        self.model = model
        self.test_dataset = test_dataset
        self.device = device
        self.field_names = field_names
        self.component_names = component_names

    def rollout_predictions(self, start_step: int, num_steps: int):
        self.model.eval()
        preds = []
        current_vol = self.test_dataset[:,start_step] # time dimension
        with torch.no_grad():
            for _ in range(num_steps):
                inp  = current_vol.unsqueeze(1).to(self.device)   # (B, t, F, C, D, H, W)
                pred = self.model(inp).cpu()                      # (B, F, C, D, H, W)
                preds.append(pred.unsqueeze(1))                   # (B, t, F, C, D, H, W)
                current_vol = pred 
        return preds

    def visualize_rollout(self,
                          start_step: int = 0,
                          num_steps:  int = 5,
                          field:    int = 0,
                          component:  int = 0,
                          slice_dim:  str = 'd',
                          slice_pos:  int = None,
                          save_path = None,
                          figname = 'ViT_CC_XAF_AAST_RO.png'):
        # unpack dims
        B, t, F, C, D, H, W = self.test_dataset.shape
        if slice_pos is None:
            slice_pos = {'d': D//2, 'h': H//2, 'w': W//2, '1d': 0}[slice_dim]

        # 1) collect true volumes for the next num_steps
        true_vols = [
            self.test_dataset[:, start_step + i + 1].unsqueeze(1)  # add back time dimension
            for i in range(num_steps)
        ]
        
        # 2) roll out predictions
        preds = self.rollout_predictions(start_step, num_steps)
        
        # print the shapes
        print(f'Predict {len(preds)} steps: Current True vols:{true_vols[0].shape}, Pred vols: {preds[0].shape}')

        # prepare figure: 3 rows × num_steps columns
        fig, axes = plt.subplots(3, num_steps, figsize=(4 * num_steps, 16))
        
        # guard for single‐column case
        if num_steps == 1:
            axes = axes.reshape(3, 1)

        def get_slice(vol):
            """Extract a 2D slice from a (F,C,X,Y,Z) volume."""
            if slice_dim == 'd':
                return vol[field, component, slice_pos, :, :].numpy()
            elif slice_dim == 'h':
                return vol[field, component, :, slice_pos, :].numpy()
            elif slice_dim == 'w':
                return vol[field, component, :, :, slice_pos].numpy()
            elif slice_dim == '1d':
                return vol[field, component, slice_pos, slice_pos, :].numpy()
        
        # Choose colormap based on field
        if field == 0:
            c_map = 'viridis'
        elif field == 1:
            c_map = 'plasma'
        else:  # field == 2
            c_map = 'inferno'
        
        print(f"Current field:{field}...")
        if slice_dim != '1d':
            for i in range(num_steps):            
                # true
                true_img = get_slice(true_vols[i].squeeze(0,1))
                
                # common scale for plotting
                vmin, vmax = true_img.min(), true_img.max()
                
                ax = axes[0, i]
                im = ax.imshow(true_img, cmap=c_map, vmin = vmin, vmax = vmax)
                ax.set_title(f'True t={start_step + i + 1}', fontsize = 32)
                ax.axis('off')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
                # pred
                pred_img = get_slice(preds[i].squeeze(0,1))
                ax = axes[1, i]
                im = ax.imshow(pred_img, cmap=c_map, vmin = vmin, vmax = vmax)
                ax.set_title(f'Pred t={start_step + i + 1}', fontsize = 32)
                ax.axis('off')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
                # mse
                mse_img = (true_img - pred_img) ** 2
                ax = axes[2, i]
                im = ax.imshow(mse_img, cmap=c_map, vmin = vmin, vmax = vmax)
                ax.set_title('MSE', fontsize = 24)
                ax.axis('off')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
        elif slice_dim == '1d':
            for i in range(num_steps):
                # true
                true_img = get_slice(true_vols[i].squeeze(0,1))
                
                # common scale for plotting
                vmin, vmax = true_img.min(), true_img.max()
                
                ax = axes[0, i]
                im = ax.plot(true_img)
                ax.set_ylim(vmin, vmax)
                ax.set_title(f'True t={start_step + i + 1}', fontsize = 32)
    
                # pred
                pred_img = get_slice(preds[i].squeeze(0,1))
                ax = axes[1, i]
                im = ax.plot(pred_img)
                ax.set_ylim(vmin, vmax)
                ax.set_title(f'Pred t={start_step + i + 1}', fontsize = 32)
    
                # mse
                mse_img = (true_img - pred_img) ** 2
                ax = axes[2, i]
                im = ax.plot(mse_img)
                ax.set_ylim(vmin, vmax)
                ax.set_title('MSE', fontsize = 24)

        #plt.suptitle(f'{self.field_names[field]}_{self.component_names[component]}, slice {slice_dim}={slice_pos}', 
                     #fontsize=20)
        plt.suptitle(f'Autoregressive rollouts (Input: t = 0) for Field {component}', fontsize=40)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        #plt.show()
        
        if save_path:
            fig_name  = os.path.join(save_path, figname)
            fig.savefig(fig_name, bbox_inches='tight')
            print(f"Figure saved to: {fig_name}")
            
        plt.close(fig)

