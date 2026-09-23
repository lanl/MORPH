from collections.abc import Sequence
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, Field

from nomad.fm_base_tool import TorchModuleTool, default_device
from nomad.well_format import Tensor

from morph_pde import MORPH


class MORPHInput(BaseModel):
    """Input to MORPH."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    state: Tensor = Field(description="MORPH input tensor with shape (T, F, C, D, H, W)")


class MORPHOutput(BaseModel):
    """Output from MORPH."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    prediction: Tensor = Field(description="Predicted next state with shape (F, C, D, H, W)")

class MORPHTool(
    TorchModuleTool[
        MORPHInput,
        MORPHOutput,
        torch.Tensor,
        torch.Tensor,
    ]
):

    args_schema: type[MORPHInput] = MORPHInput
    output_schema: type[MORPHOutput] = MORPHOutput

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):

        model_dir = Path(pretrained_model_name_or_path)

        # NOMAD Windows local-path workaround:
        # "F:FM\..." -> "F:\FM\..."
        if (
            not model_dir.exists()
            and len(pretrained_model_name_or_path) > 2
            and pretrained_model_name_or_path[1] == ":"
            and pretrained_model_name_or_path[2] not in ("\\", "/")
        ):
            repaired_path = (pretrained_model_name_or_path[:2] + "\\"
                + pretrained_model_name_or_path[2:])
            repaired_dir = Path(repaired_path)

            if repaired_dir.exists():
                model_dir = repaired_dir

        checkpoint_path = model_dir / "model.pth"

        # MORPH-Ti foundation-model configuration
        model = MORPH(
            patch_size=8,
            dim=256,
            depth=4,
            heads=4,
            heads_xa=32,
            mlp_dim=1024,
            max_components=3,
            conv_filter=8,
            max_ar=1,
            max_patches=4096,
            max_fields=3,
            dropout=0.1,
            emb_dropout=0.1,
            lora_r_attn=0,
            lora_r_mlp=0,
            lora_alpha=None,
            lora_p=0.0,
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True,)

        state_dict = checkpoint["model_state_dict"]

        # Checkpoints trained with DistributedDataParallel may contain "module."
        if next(iter(state_dict)).startswith("module."):
            state_dict = {key.replace("module.", "", 1): value
                          for key, value in state_dict.items()}

        model.load_state_dict(state_dict)
        model.eval()

        return cls(
            name="morph-ti",
            description=(
                "MORPH-Ti PDE foundation model for next-state prediction. "
                "Input tensor shape is (T, F, C, D, H, W)."
            ),
            fm=model,
            batch_size=1,
            device=kwargs.pop("device", default_device()),
            **kwargs,
        )

    def preprocess(self, inputs: Sequence[MORPHInput]) -> torch.Tensor:
        # Each input: (T,F,C,D,H,W)
        # MORPH expects: (B,T,F,C,D,H,W)
        batch = torch.stack([item.state for item in inputs], dim=0)

        if batch.ndim != 7:
            raise ValueError(f"MORPH expects input shape (B,T,F,C,D,H,W); received {tuple(batch.shape)}")

        return batch.to(self.device)

    def _forward(self, model_inputs: torch.Tensor) -> torch.Tensor:
        _, _, prediction = self.fm(model_inputs)
        return prediction.cpu()

    def postprocess(self, model_output: torch.Tensor):
        for prediction in model_output:
            yield MORPHOutput(prediction=prediction)