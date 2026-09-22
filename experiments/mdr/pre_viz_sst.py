import matplotlib.pyplot as plt
import numpy as np
import os

class PreVisualization():
    def __init__(self, results_dir):
        self.results_dir = results_dir

    def _frame_from_raw(self, sst, H, W, t):
        return sst[t].reshape(H, W, order="F")
        
    def visualize_data(self, sst, lat, lon, t = 0, 
                       title_remarks=None,
                       save_remarks=None):
        """
        Visualize one SST frame.
        """
        H, W = len(lat), len(lon)
        frame = np.ma.masked_invalid(self._frame_from_raw(sst, H, W, t))

        plt.figure(figsize=(10, 4))
        plt.imshow(
            frame,
            origin="lower",
            aspect="auto",
            extent=[lon.min(), lon.max(), lat.min(), lat.max()],
        )
        plt.colorbar(label="SST")
        plt.xlabel("longitude")
        plt.ylabel("latitude")
        plt.title(f"SST: {title_remarks}", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'sst_snapshot_{save_remarks}.png'), dpi=300)
        print(f"Saved SST snapshot visualization to: {self.results_dir}")
        plt.close()
        
    def visualize_voronoi(self, X, idx=0,
                        lat=None, lon=None, sensor_locs=None, 
                        title_remarks=None, save_remarks=None):
        """
        X: (N, 2, H, W)
        """
        vor = X[idx, 0]
        mask = X[idx, 1]

        fig, axes = plt.subplots(1, 2, figsize=(20, 4))
        if lat is not None and lon is not None:
            extent = [lon.min(), lon.max(), lat.min(), lat.max()]
            im0 = axes[0].imshow(vor, origin="lower", aspect="auto", extent=extent)
            im1 = axes[1].imshow(mask, origin="lower", aspect="auto", extent=extent)
        else:
            im0 = axes[0].imshow(vor, origin="lower", aspect="auto")
            im1 = axes[1].imshow(mask, origin="lower", aspect="auto")

        axes[0].set_title(f"Voronoi / nearest fill ({title_remarks})")
        axes[1].set_title(f"Sensor mask ({title_remarks})")

        if sensor_locs is not None and lat is not None and lon is not None:
            axes[0].scatter(
                lon[sensor_locs[:, 1]],
                lat[sensor_locs[:, 0]],
                s=20
            )

        plt.colorbar(im0, ax=axes[0], shrink=0.8)
        plt.colorbar(im1, ax=axes[1], shrink=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'voronoi_visualization_{save_remarks}.png'), dpi=300)
        print(f"Saved Voronoi visualization to: {self.results_dir}")
        plt.close()