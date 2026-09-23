import json
from pathlib import Path
import torch
from torch import nn
from morph.utils.vit_conv_xatt_axialatt2 import ViT3DRegression

class MorphFieldNormalize(nn.Module):
    def __init__(self, fm: nn.Module, mu = None, scale = None) -> None:
        super().__init__()
        self.fm = fm
        self.eps = 1e-6
        self.mu = mu 
        self.scale = scale 

    def forward(self, x: torch.Tensor):
        # x: (B, AR, F, C, D, H, W)
        s = self.scale + self.eps
        x_hat = (x - self.mu) / s
        enc, z, next_state = self.fm(x_hat)
        next_state = next_state * s + self.mu
        return enc, z, next_state

    @classmethod
    def from_pretrained(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        config = json.loads(Path(model_dir, "config.json").read_text())
        model = ViT3DRegression(**config["model"])
        weights = torch.load(Path(model_dir, "checkpoint.pth"))
        model.load_state_dict(weights)

        mu, scale = torch.load(model_dir / "normstats_heat_avd.pt")
        return MorphFieldNormalize(model, mu, scale)
