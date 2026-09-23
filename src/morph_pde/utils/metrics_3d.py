import torch
import torch.nn.functional as F

class Metrics3DCalculator:
    @staticmethod
    def calculate_rmse(vol1, vol2):
        mse = torch.mean((vol1 - vol2) ** 2)
        rmse = torch.sqrt(mse)
        return rmse
    
    @staticmethod
    def calculate_psnr(vol1, vol2, max_val=1.0):
        mse = torch.mean((vol1 - vol2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(max_val / torch.sqrt(mse))
    
    @staticmethod
    def calculate_ssim(vol1, vol2, window_size=11, size_average=True):
        # Use 3D average pooling instead of 2D
        mu1 = F.avg_pool3d(vol1, window_size, stride=1, padding=window_size // 2)
        mu2 = F.avg_pool3d(vol2, window_size, stride=1, padding=window_size // 2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Calculate variance and covariance using 3D pooling
        sigma1_sq = F.avg_pool3d(vol1 * vol1, window_size, stride=1, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.avg_pool3d(vol2 * vol2, window_size, stride=1, padding=window_size // 2) - mu2_sq
        sigma12 = F.avg_pool3d(vol1 * vol2, window_size, stride=1, padding=window_size // 2) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            # Handle the 3D case - mean across all three dimensions
            return ssim_map.mean(1).mean(1).mean(1)
        
    @staticmethod

    def calculate_VRMSE(pred: torch.Tensor,  
                        truth: torch.Tensor,
                        eps: float = 1e-7) -> torch.Tensor:
        """
        Compute per-example Variance-scaled RMSE:
        VRMSE = sqrt(  ⟨|pred − truth|^2⟩ / ( ⟨|truth − mean(truth)|^2⟩ + eps ) )
        """
        
        # 1) point-wise squared error, then spatial-mean over all dims except batch → (B,)
        mse_per_sample = F.mse_loss(pred, truth, reduction='none').mean(dim=list(range(1, pred.dim())))
    
        # 2) compute spatial mean of truth for each sample: shape (B,1,1,1,1,1)
        truth_mean = truth.mean(dim=list(range(1, truth.dim())), keepdim=True)
    
        # 3) compute per-sample variance = ⟨|truth − truth_mean|²⟩ → (B,)
        var_per_sample = ((truth - truth_mean) ** 2).mean(dim=list(range(1, truth.dim())))
    
        # 4) variance-scaled MSE, then sqrt → VRMSE
        VRMSE = torch.sqrt(mse_per_sample / (var_per_sample + eps))
        return VRMSE
    
    def calculate_NRMSE(pred: torch.Tensor,
                        truth: torch.Tensor,
                        eps: float = 1e-7) -> torch.Tensor:
        """
        Per-example Relative L2 Error (aka NRMSE):
        NRMSE = sqrt( ⟨|pred − truth|²⟩ / (⟨|truth|²⟩ + eps))
        """
        mse_per_sample = F.mse_loss(pred, truth, reduction='none').mean(dim=list(range(1, pred.dim()))) 
        ms_truth      = (truth ** 2).mean(dim=list(range(1, truth.dim()))) 
        NRMSE = torch.sqrt(mse_per_sample / (ms_truth + eps))             
        return NRMSE


