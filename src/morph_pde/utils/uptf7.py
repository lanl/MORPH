import argparse
from pathlib import Path
import numpy as np

class UPTF7:
    def __init__(self, 
                 dataset,            # NumPy array
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
        # ensure float32 NumPy array
        x = np.asarray(self.dataset, dtype=np.float32)

        # need at least 3D and at most 7D
        if x.ndim < 3:
            raise ValueError(f"Expected at least (N, T, ...), but got shape {tuple(x.shape)}")
        if x.ndim > 7:
            raise ValueError(f"Expected at most 7D tensor, but got shape {tuple(x.shape)}")

        # set default values if None
        N = self.num_samples if self.num_samples is not None else x.shape[0]
        T = self.traj_len if self.traj_len is not None else x.shape[1]
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

        # reshape to (N, T, F, C, D, H, W)
        return x.reshape(N, T, F, C, D, H, W)

def main():
    parser = argparse.ArgumentParser(description="UPTF7 Transformation (NumPy-only)")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset (.npy)')
    parser.add_argument('--num_samples', type=int, default=None, help='N')
    parser.add_argument('--traj_len', type=int, default=None, help='T')
    parser.add_argument('--fields', type=int, default=None, help='F')
    parser.add_argument('--components', type=int, default=None, help='C')
    parser.add_argument('--image_depth', type=int, default=None, help='D')
    parser.add_argument('--image_height', type=int, default=None, help='H')
    parser.add_argument('--image_width', type=int, default=None, help='W')
    args = parser.parse_args()

    # Load dataset (.npy only)
    if Path(args.dataset_path).suffix.lower() != ".npy":
        raise ValueError("Only .npy files are supported in the NumPy-only version.")
    dataset = np.load(args.dataset_path)

    # Transform and save
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
    dataset_uptf7 = uptf7.transform()
    np.save("dataset_uptf7.npy", dataset_uptf7)

if __name__ == "__main__":
    main()
