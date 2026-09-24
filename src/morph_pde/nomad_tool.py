from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field

from nomad.fm_base_tool import TorchModuleTool, default_device
from nomad.well_format import Tensor

from morph_pde import MORPH


class MORPHInput(BaseModel):
    """Input to MORPH."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    state: Tensor = Field(
        description=(
            "Unnormalized MORPH input tensor in physical units with shape "
            "(T, F, C, D, H, W)"
        )
    )


class MORPHOutput(BaseModel):
    """Output from MORPH."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    prediction: Tensor = Field(
        description=(
            "Denormalized MORPH prediction in physical units with shape "
            "(F, C, D, H, W)"
        )
    )


class MORPHFieldNormalize(torch.nn.Module):
    """Apply checkpoint-specific affine normalization around MORPH."""

    def __init__(self, fm: torch.nn.Module, mean, scale) -> None:
        super().__init__()
        self.fm = fm

        # Registered buffers move with the model when NOMAD changes devices.
        self.register_buffer(
            "mean",
            torch.as_tensor(mean, dtype=torch.float32),
        )
        self.register_buffer(
            "scale",
            torch.as_tensor(scale, dtype=torch.float32),
        )

        if torch.any(self.scale == 0):
            raise ValueError("Normalization scale must be nonzero.")

    def forward(self, x: torch.Tensor):
        x_norm = (x - self.mean) / self.scale
        enc, z, prediction_norm = self.fm(x_norm)
        prediction = prediction_norm * self.scale + self.mean
        return enc, z, prediction


def _load_config(model_dir: Path) -> dict:
    config_path = model_dir / "config.yaml"

    if not config_path.is_file():
        raise FileNotFoundError(
            f"MORPH config file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict) or "model" not in config:
        raise ValueError(
            f"Invalid MORPH config: expected a top-level 'model' section in "
            f"{config_path}"
        )

    return config


def _load_normalization(model_dir: Path, config: dict):
    norm_config = config.get("normalization", {})
    norm_file = norm_config.get("file", "normalization.npy")
    norm_path = model_dir / norm_file

    if not norm_path.is_file():
        raise FileNotFoundError(
            f"MORPH normalization file not found: {norm_path}"
        )

    stats = np.load(norm_path)

    # Current PLI deployment format:
    # normalization.npy = [mean, variance]
    if not isinstance(stats, np.ndarray) or stats.shape != (2,):
        raise ValueError(
            "For the current PLI checkpoint, normalization.npy must have "
            f"shape (2,) containing [mean, variance]. Received {stats.shape}."
        )

    statistics = norm_config.get(
        "statistics",
        ["mean", "variance"],
    )

    if statistics != ["mean", "variance"]:
        raise ValueError(
            "Current PLI integration expects normalization statistics "
            "['mean', 'variance']; "
            f"config specifies {statistics}."
        )

    mean = stats[0]
    variance = stats[1]

    return mean, variance


def _load_state_dict(checkpoint_path: Path, state_dict_key: str):
    # Fine-tuning checkpoints contain optimizer/argparse metadata, so this
    # trusted local artifact is intentionally loaded with weights_only=False.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and state_dict_key in checkpoint:
        state_dict = checkpoint[state_dict_key]
    elif (
        isinstance(checkpoint, dict)
        and checkpoint
        and all(torch.is_tensor(value) for value in checkpoint.values())
    ):
        # Also support a clean weights-only state_dict.
        state_dict = checkpoint
    else:
        raise ValueError(
            f"Could not find state dict under key '{state_dict_key}' in "
            f"{checkpoint_path}"
        )

    if state_dict and next(iter(state_dict)).startswith("module."):
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }

    return state_dict


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
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ):
        model_dir = Path(pretrained_model_name_or_path)

        # NOMAD Windows local-path workaround:
        # "F:FM\..." -> "F:\FM\..."
        if (
            not model_dir.exists()
            and len(pretrained_model_name_or_path) > 2
            and pretrained_model_name_or_path[1] == ":"
            and pretrained_model_name_or_path[2] not in ("\\", "/")
        ):
            repaired_path = (
                pretrained_model_name_or_path[:2]
                + "\\"
                + pretrained_model_name_or_path[2:]
            )
            repaired_dir = Path(repaired_path)

            if repaired_dir.exists():
                model_dir = repaired_dir

        config = _load_config(model_dir)

        checkpoint_config = config.get("checkpoint", {})
        checkpoint_path = model_dir / checkpoint_config.get(
            "file",
            "model.pth",
        )
        state_dict_key = checkpoint_config.get(
            "state_dict_key",
            "model_state_dict",
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"MORPH checkpoint not found: {checkpoint_path}"
            )

        # Architecture comes entirely from the model artifact.
        model_config = config["model"]
        model = MORPH(**model_config)

        state_dict = _load_state_dict(
            checkpoint_path,
            state_dict_key,
        )
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        mean, variance = _load_normalization(
            model_dir,
            config,
        )

        wrapped_model = MORPHFieldNormalize(
            fm=model,
            mean=mean,
            scale=variance,
        )
        wrapped_model.eval()

        task_config = config.get("task", {})
        task_name = task_config.get(
            "name",
            "morph_prediction",
        )

        return cls(
            name=task_name,
            description=(
                "MORPH model loaded from an artifact-local configuration. "
                "Input is supplied in physical units; checkpoint-specific "
                "normalization and denormalization are applied internally."
            ),
            fm=wrapped_model,
            batch_size=1,
            device=kwargs.pop(
                "device",
                default_device(),
            ),
            **kwargs,
        )

    def preprocess(
        self,
        inputs: Sequence[MORPHInput],
    ) -> torch.Tensor:
        # Each input: (T,F,C,D,H,W)
        # MORPH expects: (B,T,F,C,D,H,W)
        batch = torch.stack(
            [item.state for item in inputs],
            dim=0,
        )

        if batch.ndim != 7:
            raise ValueError(
                "MORPH expects input shape (B,T,F,C,D,H,W); "
                f"received {tuple(batch.shape)}"
            )

        return batch.to(self.device)

    def _forward(
        self,
        model_inputs: torch.Tensor,
    ) -> torch.Tensor:
        _, _, prediction = self.fm(model_inputs)
        return prediction.cpu()

    def postprocess(
        self,
        model_output: torch.Tensor,
    ):
        for prediction in model_output:
            yield MORPHOutput(
                prediction=prediction,
            )
