import numpy as np
import matplotlib.pyplot as plt
import os 

class DataPlotter:
    def __init__(self, save_path = None):
        self.save_path = save_path
        
    def plot_sample_dr2d(self, data, sample_idx=None, start_t_idx=0, num_timesteps=5, dataset_name = 'DR'):
        # Select a random sample if not specified
        if sample_idx is None:
            sample_idx = np.random.randint(0, data.shape[0])
        
        # Get the sample
        sample = data[sample_idx]
        
        # Plot the first num_timesteps time steps
        timesteps = range(num_timesteps)
        
        # Field names for the two channels
        field_names = ["Vx", "Vy"]
        
        # Use different colormaps for the two fields
        cmaps = ["viridis", "plasma"]
        
        # Create a figure with two rows (inhibitor and activator)
        fig, axes = plt.subplots(2, num_timesteps, figsize=(20, 6))
        fig.suptitle(f"[{dataset_name}] Trajectory #{sample_idx}: First {num_timesteps} Time Steps", 
                     fontsize=20)
        
        # Plot each field and timestep
        for field_idx in range(len(field_names)):
            for i, t in enumerate(timesteps):
                img_data = sample[t + start_t_idx, :, :, field_idx]
                
                im = axes[field_idx, i].imshow(img_data, cmap=cmaps[field_idx], aspect='equal')
                axes[field_idx, i].set_title(f"{field_names[field_idx]}, t={t}", fontsize = 14)
                axes[field_idx, i].axis('off')
                cbar = fig.colorbar(im, ax=axes[field_idx, i])
                cbar.ax.tick_params(labelsize=12)
        plt.show()
        
        if self.save_path:
            fig_name = os.path.join(self.save_path, f'sample_{dataset_name}.png')
            fig.savefig(fig_name, bbox_inches='tight', dpi = 512)
            print(f"Figure saved to: {self.save_path}")
        
        plt.close(fig)
    
    def plot_sample_mhd3d(self, data, sample_idx=None, start_t_idx=0, num_timesteps=5, dataset_name = 'MHD'):
        # Select a random sample if not specified
        if sample_idx is None:
            sample_idx = np.random.randint(0, data.shape[0])
        
        # Get the sample
        sample = data[sample_idx]
        
        # Plot the first num_timesteps time steps
        timesteps = range(num_timesteps)
        
        # Field names for the three channels
        field_names = ["$B_x$", "$V_x$", "$rho_x$"]
        
        # Use different colormaps for the three fields
        cmaps = ["viridis", "plasma", "inferno"]
        
        # Create a figure with three rows (for three fields)
        fig, axes = plt.subplots(len(field_names), num_timesteps, figsize=(20, 9))
        fig.suptitle(f"[{dataset_name}] Trajectory #{sample_idx}: First {num_timesteps} Time Steps", 
                     fontsize=24)
        
        # Plot each field and timestep
        for field_idx in range(len(field_names)):
            for i, t in enumerate(timesteps):
                img_data = sample[t + start_t_idx, :, :, field_idx]
                
                im = axes[field_idx, i].imshow(img_data, cmap=cmaps[field_idx], 
                                               aspect='equal')
                axes[field_idx, i].set_title(f"{field_names[field_idx]}, t={t}", fontsize = 14)
                axes[field_idx, i].axis('off')
                cbar = fig.colorbar(im, ax=axes[field_idx, i])
                cbar.ax.tick_params(labelsize=12)
        plt.show()
        #plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        # Save to disk if requested
        if self.save_path:
            fig_name = os.path.join(self.save_path, f'sample_{dataset_name}.png')
            fig.savefig(fig_name, bbox_inches='tight')
            print(f"Figure saved to: {self.save_path}")
        
        plt.close(fig)
        
    def plot_sample_cfd1d(self, data, sample_idx=None, start_t_idx=0, num_timesteps=5, dataset_name = 'CFD1d'):
        # Select a random sample if not specified
        if sample_idx is None:
            sample_idx = np.random.randint(0, data.shape[0])
        
        # Get the sample: shape (time, h, fields=3)
        sample = data[sample_idx]
        
        timesteps = range(num_timesteps)
        
        # ← updated field names & cmaps
        field_names = ["$V_x$", "$rho$", "$P$"]
        cmaps       = ["viridis", "plasma", "inferno"]
        
        # ← change to 3 rows (one per field)
        fig, axes = plt.subplots(len(field_names), num_timesteps, figsize=(20, 8))
        fig.suptitle(f"[{dataset_name}] Trajectory #{sample_idx}: First {num_timesteps} Time Steps", fontsize=24)
        
        for field_idx in range(len(field_names)):
            cmap = plt.get_cmap(cmaps[field_idx])
            color = cmap(0.5)  # pick a representative color from each map
            
            for i, t in enumerate(timesteps):
                # slice out the 1D profile
                y = sample[t + start_t_idx, :, field_idx]  # shape (h,)
                x = np.arange(y.shape[0])
                
                ax = axes[field_idx, i]
                ax.plot(x, y, '.', color=color)
                ax.set_title(f"{field_names[field_idx]}, t={t}", fontsize = 14)
                ax.set_xlabel("x index")
                ax.set_ylabel(field_names[field_idx])
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
        if self.save_path:
            fig_name = os.path.join(self.save_path, f'sample_{dataset_name}.png')
            fig.savefig(fig_name, bbox_inches='tight', dpi = 512)
            print(f"Figure saved to: {fig_name}")
        
        plt.close(fig)
    
    def plot_sample_cfd2d_ic(self, data, sample_idx=None, start_t_idx=0, num_timesteps=5, dataset_name = 'CFD2d(IC)'):
        # Select a random sample if not specified
        if sample_idx is None:
            sample_idx = np.random.randint(0, data.shape[0])
        
        # Get the sample
        sample = data[sample_idx]
        
        # Plot the first num_timesteps time steps
        timesteps = range(num_timesteps)
        
        # Field names for the two channels
        field_names = ["$F$", "$V$"]
        
        # Use different colormaps for the two fields
        cmaps = ["viridis", "plasma"]
        
        # Create a figure with two rows (inhibitor and activator)
        fig, axes = plt.subplots(len(field_names), num_timesteps, figsize=(20, 6))
        fig.suptitle(f"[{dataset_name}] Trajectory #{sample_idx}: First {num_timesteps} Time Steps", fontsize=24)
        
        # Plot each field and timestep
        for field_idx in range(len(field_names)):
            for i, t in enumerate(timesteps):
                img_data = sample[t + start_t_idx, :, :, field_idx]
                
                im = axes[field_idx, i].imshow(img_data, cmap=cmaps[field_idx], aspect='equal')
                axes[field_idx, i].set_title(f"{field_names[field_idx]}, t={t}", fontsize = 14)
                axes[field_idx, i].axis('off')
                cbar = fig.colorbar(im, ax=axes[field_idx, i])
                cbar.ax.tick_params(labelsize=12)
        plt.show()
        
        if self.save_path:
            fig_name = os.path.join(self.save_path, f'sample_{dataset_name}.png')
            fig.savefig(fig_name, bbox_inches='tight', dpi = 512)
            print(f"Figure saved to: {self.save_path}")
        
        plt.close(fig)
        
    def plot_sample_cfd3d(self, data, sample_idx=None, start_t_idx=0, num_timesteps=5, dataset_name = 'CFD3d'):
        # Select a random sample if not specified
        if sample_idx is None:
            sample_idx = np.random.randint(0, data.shape[0])
        
        # Get the sample
        sample = data[sample_idx]
        
        # Plot the first num_timesteps time steps
        timesteps = range(num_timesteps)
        
        # Field names for the three channels
        field_names = ["$V$", "$rho$", "$P$"]
        
        # Use different colormaps for the three fields
        cmaps = ["viridis", "plasma", "inferno"]
        
        # Create a figure with three rows (for three fields)
        fig, axes = plt.subplots(len(field_names), num_timesteps, figsize=(20, 9))
        fig.suptitle(f"[{dataset_name}] Trajectory #{sample_idx}: First {num_timesteps} Time Steps", 
                     fontsize=24)
        
        # Plot each field and timestep
        for field_idx in range(len(field_names)):
            for i, t in enumerate(timesteps):
                img_data = sample[t + start_t_idx, :, :, field_idx]
                
                im = axes[field_idx, i].imshow(img_data, cmap=cmaps[field_idx], 
                                               aspect='equal')
                axes[field_idx, i].set_title(f"{field_names[field_idx]}, t={t}", fontsize = 14)
                axes[field_idx, i].axis('off')
                cbar = fig.colorbar(im, ax=axes[field_idx, i])
                cbar.ax.tick_params(labelsize=12)
        plt.show()
        #plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        # Save to disk if requested
        if self.save_path:
            fig_name = os.path.join(self.save_path, f'sample_{dataset_name}.png')
            fig.savefig(fig_name, bbox_inches='tight', dpi = 512)
            print(f"Figure saved to: {self.save_path}")
        
        plt.close(fig)
            
    def plot_sample_sw2d(self, data, sample_idx=None, start_t_idx=0, num_timesteps=5, dataset_name = 'SW'):
        # Select a random sample if not specified
        if sample_idx is None:
            sample_idx = np.random.randint(0, data.shape[0])
        
        # Get the sample
        sample = data[sample_idx]
        
        # Plot the first num_timesteps time steps
        timesteps = range(num_timesteps)
        
        # Create a figure with two rows (inhibitor and activator)
        fig, axes = plt.subplots(1, num_timesteps, figsize=(30, 4))
        fig.suptitle(f"[{dataset_name}] Trajectory #{sample_idx}: First {num_timesteps} Time Steps", 
                     fontsize=24)
        
        # Plot each field and timestep
        for i, t in enumerate(timesteps):
            img_data = sample[t + start_t_idx, :, :, 0]
            im = axes[i].imshow(img_data, cmap="viridis")
            axes[i].set_title(f"h, t={t}", fontsize = 16)
            axes[i].axis('off')
            #cbar = fig.colorbar(im, ax=axes[i])
            #cbar.ax.tick_params(labelsize=12)
        plt.show()
        
        if self.save_path:
            fig_name = os.path.join(self.save_path, f'sample_{dataset_name}.png')
            fig.savefig(fig_name, bbox_inches='tight', dpi = 512)
            print(f"Figure saved to: {self.save_path}")
        
        plt.close(fig)
        
    def plot_sample_dr1d(self, data, sample_idx=None, start_t_idx=0, num_timesteps=5, 
                         dataset_name = 'DR1d', figsize=(20, 4)):
        # Select a random sample if not specified
        if sample_idx is None:
            sample_idx = np.random.randint(0, data.shape[0])
        
        # Get the sample: shape (time, h, fields=3)
        sample = data[sample_idx]
        
        timesteps = range(num_timesteps)
        
        # Field names for the three channels
        field_names = ["v"]
        
        # Use different colormaps for the two fields
        cmaps = ["viridis"]
        
        # Create a figure with three rows (for three fields)
        fig, axes = plt.subplots(len(field_names), num_timesteps, figsize=figsize)
        fig.suptitle(f"[{dataset_name}] Trajectory #{sample_idx}: First {num_timesteps} Time Steps", 
                     fontsize=24)
        
        # ← change to 3 rows (one per field)
        for field_idx in range(len(field_names)):
            cmap = plt.get_cmap(cmaps[field_idx])
            color = cmap(0.5)  # pick a representative color from each map
            for i, t in enumerate(timesteps):
                # slice out the 1D profile
                y = sample[t + start_t_idx, :]  # shape (h,)
                x = np.arange(y.shape[0])
                
                ax = axes[i]
                ax.plot(x, y, '.', color=color)
                ax.set_title(f"{field_names[field_idx]}, t={t}", fontsize = 14)
                ax.set_xlabel("x index")
                ax.set_ylabel(field_names[field_idx])
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
        if self.save_path:
            fig_name = os.path.join(self.save_path, f'sample_{dataset_name}.png')
            fig.savefig(fig_name, bbox_inches='tight', dpi = 512)
            print(f"Figure saved to: {fig_name}")
        
        plt.close(fig)
        
    def plot_sample_cfd2d(self, data, sample_idx=None, start_t_idx=0, num_timesteps=5, 
                          dataset_name = 'CFD2d'):
        # Select a random sample if not specified
        if sample_idx is None:
            sample_idx = np.random.randint(0, data.shape[0])
        
        # Get the sample
        sample = data[sample_idx]
        
        # Plot the first num_timesteps time steps
        timesteps = range(num_timesteps)
        
        # Field names for the three channels
        field_names = ["$V$", "$rho$", "$P$"]
        
        # Use different colormaps for the three fields
        cmaps = ["viridis", "plasma", "inferno"]
        
        # Create a figure with three rows (for three fields)
        fig, axes = plt.subplots(len(field_names), num_timesteps, figsize=(20, 9))
        fig.suptitle(f"[{dataset_name}] Trajectory #{sample_idx}: First {num_timesteps} Time Steps", 
                     fontsize=24)
        
        # Plot each field and timestep
        for field_idx in range(len(field_names)):
            for i, t in enumerate(timesteps):
                img_data = sample[t + start_t_idx, :, :, field_idx]
                
                im = axes[field_idx, i].imshow(img_data, cmap=cmaps[field_idx], 
                                               aspect='equal')
                axes[field_idx, i].set_title(f"{field_names[field_idx]}, t={t}", fontsize = 14)
                axes[field_idx, i].axis('off')
                cbar = fig.colorbar(im, ax=axes[field_idx, i])
                cbar.ax.tick_params(labelsize=12)
        plt.show()
        #plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        # Save to disk if requested
        if self.save_path:
            fig_name = os.path.join(self.save_path, f'sample_{dataset_name}.png')
            fig.savefig(fig_name, bbox_inches='tight', dpi = 512)
            print(f"Figure saved to: {self.save_path}")
        
        plt.close(fig)
        
    def plot_sample_tgc3d(self, data, sample_idx=None, start_t_idx=0, num_timesteps=5, dataset_name = 'TGC3d'):
        # Select a random sample if not specified
        if sample_idx is None:
            sample_idx = np.random.randint(0, data.shape[0])
        
        # Get the sample
        sample = data[sample_idx]
        
        # Plot the first num_timesteps time steps
        timesteps = range(num_timesteps)
        
        # Field names for the three channels
        field_names = ["$V$", "$rho$", "$P$", "T"]
        
        # Use different colormaps for the three fields
        cmaps = ["viridis", "plasma", "inferno", "coolwarm"]
        
        # Create a figure with three rows (for three fields)
        fig, axes = plt.subplots(len(field_names), num_timesteps, figsize=(20, 12))
        fig.suptitle(f"[{dataset_name}] Trajectory #{sample_idx}: First {num_timesteps} Time Steps", 
                     fontsize=24)
        
        # Plot each field and timestep
        for field_idx in range(len(field_names)):
            for i, t in enumerate(timesteps):
                img_data = sample[t + start_t_idx, :, :, field_idx]
                
                im = axes[field_idx, i].imshow(img_data, cmap=cmaps[field_idx], 
                                               aspect='equal')
                axes[field_idx, i].set_title(f"{field_names[field_idx]}, t={t}", fontsize = 14)
                axes[field_idx, i].axis('off')
                cbar = fig.colorbar(im, ax=axes[field_idx, i])
                cbar.ax.tick_params(labelsize=12)
        plt.show()
        #plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        # Save to disk if requested
        if self.save_path:
            fig_name = os.path.join(self.save_path, f'sample_{dataset_name}.png')
            fig.savefig(fig_name, bbox_inches='tight', dpi = 512)
            print(f"Figure saved to: {self.save_path}")
        
        plt.close(fig)
    
    def plot_sample_ns2d_pdegym(self, data, sample_idx=None, start_t_idx=0, num_timesteps=5, dataset_name = 'NS'):
        # Select a random sample if not specified
        if sample_idx is None:
            sample_idx = np.random.randint(0, data.shape[0])
        
        # Get the sample
        sample = data[sample_idx]
        
        # Plot the first num_timesteps time steps
        timesteps = range(num_timesteps)
        
        # Field names for the two channels
        field_names = ["Vx", "Vy", "tracer"]
        
        # Use different colormaps for the two fields
        cmaps = ["viridis", "plasma", "inferno"]
        
        # Create a figure with two rows (inhibitor and activator)
        fig, axes = plt.subplots(len(field_names), num_timesteps, figsize=(27, 6))
        fig.suptitle(f"[{dataset_name}] Trajectory #{sample_idx}: First {num_timesteps} Time Steps", fontsize=24)
        
        # Plot each field and timestep
        for field_idx in range(len(field_names)):
            for i, t in enumerate(timesteps):
                img_data = sample[t + start_t_idx, :, :, field_idx]
                
                im = axes[field_idx, i].imshow(img_data, cmap=cmaps[field_idx], aspect='equal')
                axes[field_idx, i].set_title(f"{field_names[field_idx]}, t={t}", fontsize = 14)
                axes[field_idx, i].axis('off')
                cbar = fig.colorbar(im, ax=axes[field_idx, i])
                cbar.ax.tick_params(labelsize=12)
        plt.show()
        
        if self.save_path:
            fig_name = os.path.join(self.save_path, f'sample_{dataset_name}.png')
            fig.savefig(fig_name, bbox_inches='tight', dpi = 512)
            print(f"Figure saved to: {self.save_path}")
        
        plt.close(fig)
        
    def plot_sample_ce2d_pdegym(self, data, sample_idx=None, start_t_idx=0, num_timesteps=5, 
                          dataset_name = 'CE'):
        # Select a random sample if not specified
        if sample_idx is None:
            sample_idx = np.random.randint(0, data.shape[0])
        
        # Get the sample
        sample = data[sample_idx]
        
        # Plot the first num_timesteps time steps
        timesteps = range(num_timesteps)
        
        # Field names for the three channels
        field_names = ["$rho$", "$V_x$", "$V_y$", "$P$"]
        field_names = ["$rho$", "$V_x$", "$V_y$", "$P$", "$E$"]
        
        # Use different colormaps for the three fields
        cmaps = ["viridis", "plasma", "plasma", "coolwarm"]
        cmaps = ["viridis", "plasma", "plasma", "inferno", "coolwarm"]
        
        # Create a figure with three rows (for three fields)
        fig, axes = plt.subplots(len(field_names), num_timesteps, figsize=(22, 12))
        fig.suptitle(f"[{dataset_name}] Trajectory #{sample_idx}: First {num_timesteps} Time Steps", 
                     fontsize=24)
        
        # Plot each field and timestep
        for field_idx in range(len(field_names)):
            for i, t in enumerate(timesteps):
                img_data = sample[t + start_t_idx, :, :, field_idx]
                
                im = axes[field_idx, i].imshow(img_data, cmap=cmaps[field_idx], 
                                               aspect='equal')
                axes[field_idx, i].set_title(f"{field_names[field_idx]}, t={t}", fontsize = 14)
                axes[field_idx, i].axis('off')
                cbar = fig.colorbar(im, ax=axes[field_idx, i])
                cbar.ax.tick_params(labelsize=12)
        plt.show()
        #plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        # Save to disk if requested
        if self.save_path:
            fig_name = os.path.join(self.save_path, f'sample_{dataset_name}.png')
            fig.savefig(fig_name, bbox_inches='tight', dpi = 512)
            print(f"Figure saved to: {self.save_path}")
        
        plt.close(fig)