import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class RolloutTester:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()
    
    def test_single_sample(self, test_data, sample_idx=0):
        # Select a sample
        sample = test_data[sample_idx]  # Shape: (101, 2, 128, 128)
        
        # Get the first timestep as input and the rest as labels
        test_x_0 = sample[0:1]  # Shape: (1, 2, 128, 128)
        labels = sample[1:]     # Shape: (100, 2, 128, 128)
              
        # Convert to torch tensors
        test_x_0 = torch.tensor(test_x_0, dtype=torch.float32).to(self.device)
        labels = torch.tensor(labels, dtype=torch.float32)  # Keep on CPU
        
        # Create list to store all predictions
        all_predictions = []
        
        # Run rollout
        current_input = test_x_0.squeeze(0)  # Shape: (2, 128, 128)
        
        with torch.no_grad():
            for t in tqdm(range(labels.shape[0]), desc="Running rollout"):
                # Forward pass
                current_input = current_input.unsqueeze(0)  # Add batch dimension
                next_prediction = self.model(current_input)

                # Save prediction
                all_predictions.append(next_prediction.cpu().squeeze(0))
                
                # Use prediction as next input
                current_input = next_prediction.squeeze(0)
        
        # Stack predictions
        all_predictions = torch.stack(all_predictions)  # Shape: (100, 2, 128, 128)
        print(f"stack all predictions: {all_predictions.shape}")
        
        # Calculate MSE for each timestep and channel
        mse_values = ((all_predictions - labels) ** 2).mean(dim=(2, 3))  # Shape: (100, 2)
        
        return {
            'predictions': all_predictions,
            'labels': labels,
            'mse_values': mse_values
        }
    
    def visualize_results(self, results, start_t_idx = 0, num_timesteps_plot=5):
        predictions = results['predictions']
        labels = results['labels']
        mse_values = results['mse_values']
        
        # Calculate absolute error
        abs_error = torch.abs(predictions - labels)
        
        # Calculate equally spaced timestep indices
        indices = list(range(num_timesteps_plot))
        
        # Create figure for both channels
        for channel in range(2):
            field_name = "Inhibitor" if channel == 0 else "Activator"
            cmap = "viridis" if channel == 0 else "plasma"
            
            fig, axes = plt.subplots(3, num_timesteps_plot, figsize=(num_timesteps_plot*4, 10))
            fig.suptitle(f"{field_name} Field Rollout - First {num_timesteps_plot} Timesteps", fontsize=16)
            
            # Plot for each selected timestep
            for col, t_idx in enumerate(indices):
                # Ground truth
                ax = axes[0, col]
                im = ax.imshow(labels[start_t_idx + t_idx, channel].numpy(), cmap=cmap)
                ax.set_title(f"Ground Truth t={start_t_idx + t_idx+1}")
                ax.axis("off")
                fig.colorbar(im, ax=ax)
                
                # Prediction
                ax = axes[1, col]
                im = ax.imshow(predictions[start_t_idx + t_idx, channel].numpy(), cmap=cmap)
                ax.set_title(f"Prediction t={start_t_idx + t_idx+1}")
                ax.axis("off")
                fig.colorbar(im, ax=ax)
                
                # Absolute error
                ax = axes[2, col]
                im = ax.imshow(abs_error[start_t_idx + t_idx, channel].numpy(), cmap="hot")
                ax.set_title(f"Error, MSE={mse_values[start_t_idx + t_idx, channel]:.4f}")
                ax.axis("off")
                fig.colorbar(im, ax=ax)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()
        
        # Plot MSE over time
        plt.figure(figsize=(10, 5))
        plt.plot(mse_values[:, 0].numpy(), '*', label='Inhibitor')
        plt.plot(mse_values[:, 1].numpy(), 's', label='Activator')
        plt.xlabel("Timestep")
        plt.ylabel("MSE")
        plt.title("MSE Evolution Over Time")
        plt.yscale("log")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Print average MSE
        avg_mse = mse_values.mean().item()
        print(f"Average MSE over all timesteps: {avg_mse:.6f}")
        
        return avg_mse