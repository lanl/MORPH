"""Shared utilities for the PLI/FRAC MORPH reproduction scripts.

The paper task is endpoint prediction: X(t0) -> X(tT).  The historical
normalization script computes RevIN statistics independently for every sample
using both endpoint frames and all spatial locations.  This module reproduces
that behavior on-the-fly, without writing a second normalized copy of the
HDF5 dataset.
"""

from __future__ import print_function

import bisect
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


MORPH_MODELS = {
    "Ti": (8, 256, 4, 4, 1024),
    "S": (8, 512, 8, 4, 2048),
    "M": (8, 768, 12, 8, 3072),
    "L": (8, 1024, 16, 16, 4096),
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    directory: str
    field_name: str
    default_lr: float
    default_batch_size: int
    default_patch: Tuple[int, int, int]
    expected_hw: Tuple[int, int]


DATASET_SPECS: Dict[str, DatasetSpec] = {
    # The executable historical fine-tuning script overrides DataConfig and
    # uses (1, 8, 8) for PLI, matching the paper's stated patch size.  A stale
    # DataConfig entry instead says (1, 16, 8) to obtain 3500 tokens; this
    # inconsistency is documented in README_MORPH_PLI_FRAC.md.
    "pli": DatasetSpec(
        key="pli",
        directory="2dHEAT",
        field_name="av_density",
        default_lr=1.0e-4,
        default_batch_size=8,
        default_patch=(1, 8, 8),
        expected_hw=(1120, 400),
    ),
    "frac": DatasetSpec(
        key="frac",
        directory="2dFRAC_tung",
        field_name="damage",
        default_lr=1.0e-3,
        default_batch_size=512,
        default_patch=(1, 4, 4),
        expected_hw=(128, 128),
    ),
}


def infer_repo_root() -> Path:
    """Infer MORPH repository root from experiments/EMD_pli_frac/morph_emd."""
    return Path(__file__).resolve().parents[3]


def default_dataset_dir(dataset_key: str) -> Path:
    spec = DATASET_SPECS[dataset_key]
    return infer_repo_root() / "datasets" / spec.directory


def h5_files(dataset_dir: Union[str, Path], split: str) -> List[Path]:
    split_dir = Path(dataset_dir) / split
    if not split_dir.is_dir():
        raise FileNotFoundError("Missing split directory: {}".format(split_dir))
    files = sorted(
        p for p in split_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".h5", ".hdf5")
    )
    if not files:
        raise FileNotFoundError("No .h5/.hdf5 files found in {}".format(split_dir))
    return files


def _normalize_channel_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def read_channel_names(dataset_dir: Union[str, Path]) -> Optional[List[str]]:
    """Read PLI channel names if one of the historical filenames is present."""
    root = Path(dataset_dir)
    candidates = [
        root / "channel_names.txt",
        root / "common_channels.txt",
        root.parent / "channel_names.txt",
        root.parent / "common_channels.txt",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) == 1 and "," in lines[0]:
            lines = [part.strip() for part in lines[0].split(",") if part.strip()]
        return lines
    return None


def resolve_pli_field_index(
    dataset_dir: Union[str, Path],
    field_name: str = "av_density",
    field_index: Optional[int] = None,
) -> int:
    if field_index is not None:
        if field_index < 0:
            raise ValueError("field_index must be >= 0")
        return int(field_index)

    names = read_channel_names(dataset_dir)
    if not names:
        raise ValueError(
            "Could not find channel_names.txt/common_channels.txt for PLI. "
            "Pass --field-index explicitly. The historical visualization appears "
            "to use index 2 for average density, but this script intentionally "
            "does not assume that silently."
        )

    wanted = _normalize_channel_name(field_name)
    aliases = {
        "avdensity",
        "avgdensity",
        "averagedensity",
        "mixtureaverageddensity",
    }
    aliases.add(wanted)
    for idx, name in enumerate(names):
        if _normalize_channel_name(name) in aliases:
            return idx
    raise ValueError(
        "PLI field {!r} not found in channel names: {}".format(field_name, names)
    )


def pair_revin_normalize(
    pair: torch.Tensor,
    eps: float = 1.0e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize one endpoint pair with the historical RevIN convention.

    Parameters
    ----------
    pair:
        Tensor shaped (2, H, W) after selecting the single PLI/FRAC field.
        Statistics are computed jointly across t0, tT, H, and W.

    Returns
    -------
    normalized, mean, variance
        Variance uses the population convention (unbiased=False), matching
        NumPy's default and the original MORPH RevIN implementation.
    """
    if pair.ndim != 3 or pair.shape[0] != 2:
        raise ValueError("Expected endpoint pair with shape (2,H,W), got {}".format(tuple(pair.shape)))
    pair = torch.nan_to_num(pair.float(), nan=0.0, posinf=0.0, neginf=0.0)
    mean = pair.mean()
    var = pair.var(unbiased=False)
    normalized = (pair - mean) / torch.sqrt(var + eps)
    return normalized, mean, var


def pair_revin_denormalize(
    value: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    return value * torch.sqrt(var + eps) + mean


class EndpointH5Dataset(Dataset):
    """Lazy HDF5 loader for already-created PLI/FRAC train/val/test splits.

    Supported layouts:
      PLI  : (N, 2, F, H, W)
      FRAC : (N, 2, H, W) or (N, 2, 1, H, W)

    A sample is returned in the tensor layout expected by ViT3DRegression:
      input  -> (T=1, F=1, C=1, D=1, H, W)
      target -> (F=1, C=1, D=1, H, W)
    """

    def __init__(
        self,
        dataset_dir: Union[str, Path],
        split: str,
        dataset_key: str,
        field_index: Optional[int] = None,
        field_name: Optional[str] = None,
        normalization: str = "pair-revin",
        eps: float = 1.0e-6,
        max_samples: Optional[int] = None,
        validate_spatial_shape: bool = True,
    ) -> None:
        super().__init__()
        if dataset_key not in DATASET_SPECS:
            raise ValueError("Unknown dataset_key: {}".format(dataset_key))
        if normalization not in ("pair-revin", "none"):
            raise ValueError("normalization must be 'pair-revin' or 'none'")

        self.dataset_key = dataset_key
        self.spec = DATASET_SPECS[dataset_key]
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.normalization = normalization
        self.eps = float(eps)
        self.files = h5_files(self.dataset_dir, split)
        self.field_name = field_name or self.spec.field_name
        self.field_index = None
        if dataset_key == "pli":
            self.field_index = resolve_pli_field_index(
                self.dataset_dir,
                field_name=self.field_name,
                field_index=field_index,
            )

        lengths = []
        spatial_shape = None
        for path in self.files:
            with h5py.File(str(path), "r") as handle:
                if "data" not in handle:
                    raise KeyError("{} has no 'data' dataset".format(path))
                shape = tuple(handle["data"].shape)
                if len(shape) < 4 or shape[1] != 2:
                    raise ValueError(
                        "{} must contain endpoint pairs with time dimension 2; got {}".format(path, shape)
                    )
                lengths.append(int(shape[0]))
                hw = (int(shape[-2]), int(shape[-1]))
                if spatial_shape is None:
                    spatial_shape = hw
                elif spatial_shape != hw:
                    raise ValueError("Inconsistent spatial shapes: {} vs {}".format(spatial_shape, hw))

                if dataset_key == "pli":
                    if len(shape) != 5:
                        raise ValueError("PLI expects (N,2,F,H,W), got {} in {}".format(shape, path))
                    if self.field_index is None or self.field_index >= shape[2]:
                        raise IndexError(
                            "PLI field index {} is invalid for {} fields in {}".format(
                                self.field_index, shape[2], path
                            )
                        )
                else:
                    if len(shape) not in (4, 5):
                        raise ValueError("FRAC expects (N,2,H,W) or (N,2,1,H,W), got {}".format(shape))
                    if len(shape) == 5 and shape[2] != 1:
                        raise ValueError("FRAC rank-5 data must have singleton field axis, got {}".format(shape))

        if validate_spatial_shape and spatial_shape != self.spec.expected_hw:
            raise ValueError(
                "{} spatial shape is {}, expected {}. "
                "Use validate_spatial_shape=False only for tests/debugging.".format(
                    dataset_key.upper(), spatial_shape, self.spec.expected_hw
                )
            )

        self.spatial_shape = spatial_shape
        self._file_lengths = lengths
        self._cumulative = []
        total = 0
        for length in lengths:
            total += length
            self._cumulative.append(total)
        if max_samples is None:
            self._length = total
        else:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
            self._length = min(total, int(max_samples))

        # Handles are opened lazily.  With DataLoader workers each process owns
        # its own dataset copy/cache, avoiding one open/close cycle per sample.
        self._handles = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles = {}

    def __del__(self):
        self.close()

    def __len__(self) -> int:
        return self._length

    def _locate(self, index: int) -> Tuple[Path, int]:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        file_idx = bisect.bisect_right(self._cumulative, index)
        prev = 0 if file_idx == 0 else self._cumulative[file_idx - 1]
        return self.files[file_idx], index - prev

    def _read_pair(self, index: int) -> torch.Tensor:
        path, local_idx = self._locate(index)
        key = str(path)
        handle = self._handles.get(key)
        if handle is None:
            handle = h5py.File(key, "r")
            self._handles[key] = handle
        data = handle["data"]
        if self.dataset_key == "pli":
            arr = data[local_idx, :, self.field_index, :, :]
        else:
            if data.ndim == 4:
                arr = data[local_idx, :, :, :]
            else:
                arr = data[local_idx, :, 0, :, :]
        return torch.from_numpy(np.asarray(arr, dtype=np.float32))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        pair = self._read_pair(index)
        pair = torch.nan_to_num(pair, nan=0.0, posinf=0.0, neginf=0.0)

        if self.normalization == "pair-revin":
            pair_norm, mean, var = pair_revin_normalize(pair, eps=self.eps)
        else:
            pair_norm = pair.float()
            mean = torch.tensor(0.0, dtype=torch.float32)
            var = torch.tensor(1.0, dtype=torch.float32)

        h, w = pair_norm.shape[-2:]
        x = pair_norm[0].reshape(1, 1, 1, 1, h, w)
        y = pair_norm[1].reshape(1, 1, 1, h, w)
        return {
            "input": x,
            "target": y,
            "mean": mean,
            "var": var,
            "index": torch.tensor(index, dtype=torch.long),
        }


def extract_prediction(model_output):
    """Handle ViT3DRegression's (enc, latent, prediction) return signature."""
    if isinstance(model_output, (tuple, list)):
        return model_output[-1]
    return model_output


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def build_morph_model(
    model_size: str,
    patch_size: Tuple[int, int, int],
    max_ar: int = 1,
) -> nn.Module:
    """Build MORPH from the pip-installable morph_pde package."""
    if model_size not in MORPH_MODELS:
        raise ValueError("Unknown model size: {}".format(model_size))
    try:
        from morph_pde.utils.vit_conv_xatt_axialatt2 import ViT3DRegression
    except ImportError as exc:
        raise ImportError(
            "Could not import morph_pde. Install the package-morph branch first, e.g. "
            "python -m pip install -e . from the MORPH repository root."
        ) from exc

    filters, dim, heads, depth, mlp_dim = MORPH_MODELS[model_size]
    return ViT3DRegression(
        patch_size=patch_size,
        dim=dim,
        depth=depth,
        heads=heads,
        heads_xa=32,
        mlp_dim=mlp_dim,
        max_components=3,
        conv_filter=filters,
        max_ar=max_ar,
        max_patches=4096,
        max_fields=3,
        dropout=0.1,
        emb_dropout=0.1,
        lora_r_attn=0,
        lora_r_mlp=0,
        lora_alpha=None,
        lora_p=0.0,
        model_size=model_size,
        activated_ar1k=False,
    )


def _torch_load(path: Union[str, Path], map_location="cpu"):
    """Load checkpoints across PyTorch versions without requiring weights_only."""
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def extract_state_dict(checkpoint) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise ValueError("Could not find a model state dict in checkpoint")


def strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    output = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        output[new_key] = value
    return output


def load_pretrained_compatible(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    map_location="cpu",
) -> Dict[str, object]:
    """Load matching pretrained tensors and explicitly report patch-shape skips.

    PyTorch's strict=False still errors on same-name tensors whose shapes differ.
    FRAC's 4x4 patching changes patch-dependent projection/decoder dimensions, so
    those tensors must be deliberately skipped/reinitialized when loading an 8x8
    foundation checkpoint.
    """
    checkpoint = _torch_load(checkpoint_path, map_location=map_location)
    incoming = strip_module_prefix(extract_state_dict(checkpoint))
    current = unwrap_model(model).state_dict()

    matched = {}
    mismatched = {}
    unexpected = []
    for key, value in incoming.items():
        if key not in current:
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(current[key].shape):
            mismatched[key] = {
                "checkpoint": list(value.shape),
                "model": list(current[key].shape),
            }
            continue
        matched[key] = value

    result = unwrap_model(model).load_state_dict(matched, strict=False)
    report = {
        "loaded_tensor_count": len(matched),
        "checkpoint_tensor_count": len(incoming),
        "mismatched_shapes": mismatched,
        "unexpected_keys": unexpected,
        "missing_keys": list(result.missing_keys),
    }
    return report


def load_finetuned_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    map_location="cpu",
) -> Mapping[str, object]:
    checkpoint = _torch_load(checkpoint_path, map_location=map_location)
    state = strip_module_prefix(extract_state_dict(checkpoint))
    unwrap_model(model).load_state_dict(state, strict=True)
    return checkpoint


def epoch_learning_rate(
    epoch: int,
    epochs: int,
    peak_lr: float,
    warmup_epochs: int = 5,
    min_lr: float = 1.0e-7,
) -> float:
    """Five-epoch linear warmup followed by cosine decay."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if epoch < 0 or epoch >= epochs:
        raise ValueError("epoch must be in [0, epochs)")
    if warmup_epochs <= 1:
        warmup_epochs = 1

    if epoch < warmup_epochs:
        if warmup_epochs == 1:
            return peak_lr
        alpha = float(epoch) / float(warmup_epochs - 1)
        return min_lr + alpha * (peak_lr - min_lr)

    remaining = max(1, epochs - warmup_epochs)
    progress = float(epoch - warmup_epochs) / float(max(1, remaining - 1))
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    loss_fn = nn.MSELoss(reduction="mean")

    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = extract_prediction(model(x))
        if pred.shape != y.shape:
            raise RuntimeError("Prediction {} != target {}".format(tuple(pred.shape), tuple(y.shape)))
        loss = loss_fn(pred, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        batch_size = int(x.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size

    if total_items == 0:
        raise RuntimeError("Training loader produced zero samples")
    return total_loss / total_items


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable[Mapping[str, torch.Tensor]],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    elements = 0
    samples = 0
    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        pred = extract_prediction(model(x))
        if pred.shape != y.shape:
            raise RuntimeError("Prediction {} != target {}".format(tuple(pred.shape), tuple(y.shape)))
        if not torch.isfinite(pred).all():
            raise FloatingPointError("Non-finite prediction")
        diff = pred - y
        mse_sum += float(torch.sum(diff * diff).cpu())
        mae_sum += float(torch.sum(torch.abs(diff)).cpu())
        elements += int(diff.numel())
        samples += int(x.shape[0])
    if elements == 0:
        raise RuntimeError("Evaluation loader produced zero samples")
    return {
        "mse": mse_sum / elements,
        "rmse": math.sqrt(mse_sum / elements),
        "mae": mae_sum / elements,
        "samples": float(samples),
    }


def checkpoint_payload(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    dataset_key: str,
    model_size: str,
    patch_size: Tuple[int, int, int],
    normalization: str,
    eps: float,
    train_args: Mapping[str, object],
    best_val_mse: float,
    preload_report: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    payload = {
        "epoch": int(epoch),
        "model_state_dict": unwrap_model(model).state_dict(),
        "dataset_key": dataset_key,
        "model_size": model_size,
        "patch_size": list(patch_size),
        "normalization": normalization,
        "normalization_eps": float(eps),
        "best_val_mse": float(best_val_mse),
        "train_args": dict(train_args),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if preload_report is not None:
        payload["pretrained_load_report"] = dict(preload_report)
    return payload


def save_json(path: Union[str, Path], value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(requested)
