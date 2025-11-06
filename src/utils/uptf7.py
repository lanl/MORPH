import torch
import numpy as np
import argparse
from pathlib import Path

class UPTF7():
    def __init__(self, 
                 dataset: torch.Tensor | np.ndarray,
                 num_samples: int = None, 
                 traj_len: int = None, 
                 fields: int = None, 
                 components: int = None,
                 image_depth: int = None, 
                 image_height: int = None, 
                 image_width: int = None):
        
        self.dataset = dataset
        self.num_samples = num_samples
        self.traj_len = traj_len
        self.fields = fields
        self.components = components
        self.image_depth = image_depth
        self.image_height = image_height
        self.image_width = image_width

    def transform(self):
        # need tensor dataset (float32)
        if not isinstance(self.dataset, torch.Tensor):
            dataset_tensor = torch.from_numpy(self.dataset).to(dtype=torch.float32)
        else:
            dataset_tensor = self.dataset

        # need atleast 3D tensor and max 7D tensor
        if dataset_tensor.ndims < 3:
            raise ValueError(f"Expected at least (N, T, ...), but got shape {tuple(dataset_tensor.shape)}")
        
        if dataset_tensor.ndims > 7:
            raise ValueError(f"Expected at most 7D tensor, but got shape {tuple(dataset_tensor.shape)}")

        # set default values if None
        N = self.num_samples if self.num_samples is not None else dataset_tensor.shape[0]
        T = self.traj_len if self.traj_len is not None else dataset_tensor.shape[1]
        F = self.fields if self.fields is not None else 1
        C = self.components if self.components is not None else 1
        D = self.image_depth if self.image_depth is not None else 1
        H = self.image_height if self.image_height is not None else 1
        W = self.image_width if self.image_width is not None else 1

        # morph-v1 restrictions
        if F > 3 or C > 3:
            raise ValueError(f"MORPH-v1 only supports up to 3 fields and 3 components. Got F={F}, C={C}")
        
        if D * H * W > 128 * 128 * 128:
            raise ValueError(f"MORPH-v1 only supports up to 128x128x128 images (max tokens 4096). Got D*H*W={D*H*W}")
        
        # Apply UPTF7 transformation
        dataset_uptf7 = dataset_tensor.contiguous().view(N, T, F, C, D, H, W)

        return dataset_uptf7
    
    def main():
        # load dataset
        parser = argparse.ArgumentParser(description="UPTF7 Transformation")
        parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset (numpy file)')
        parser.add_argument('--num_samples', type=int, default=None, help='Number of samples (N)')
        parser.add_argument('--traj_len', type=int, default=None, help='Trajectory length (T)')
        parser.add_argument('--fields', type=int, default=None, help='Number of fields (F)')
        parser.add_argument('--components', type=int, default=None, help='Number of components (C)')
        parser.add_argument('--image_depth', type=int, default=None, help='Image depth (D)')
        parser.add_argument('--image_height', type=int, default=None, help='Image height (H)')
        parser.add_argument('--image_width', type=int, default=None, help='Image width (W)')
        args = parser.parse_args()

        # Load dataset
        ext = Path(args.dataset_path).suffix.lower()
        if ext == ".npy":
            dataset = np.load(args.dataset_path)  # -> np.ndarray
        elif ext == ".pt" or ext == ".pth":
            dataset = torch.load(args.dataset_path)  # -> torch.Tensor

        # Create UPTF7 instance
        uptf7 = UPTF7(
            dataset=dataset,
            num_samples=args.num_samples,
            traj_len=args.traj_len,
            fields=args.fields,
            components=args.components,
            image_depth=args.image_depth,
            image_height=args.image_height,
            image_width=args.image_width
        )

        # Transform dataset
        dataset_uptf7 = uptf7.transform()

        # Save transformed dataset
        np.save("dataset_uptf7.npy", dataset_uptf7)

    if __name__ == "__main__":
        main()