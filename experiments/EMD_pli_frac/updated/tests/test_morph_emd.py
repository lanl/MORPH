import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from morph_emd.common import (
    EndpointH5Dataset,
    build_morph_model,
    checkpoint_payload,
    epoch_learning_rate,
    evaluate,
    load_finetuned_checkpoint,
    load_pretrained_compatible,
    pair_revin_denormalize,
    pair_revin_normalize,
    train_one_epoch,
)


class TinyMORPH(nn.Module):
    """Minimal model with the same input/output tensor contract as MORPH."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        # x: (B,T,F,C,D,H,W); output: (B,F,C,D,H,W)
        pred = self.scale * x[:, -1] + self.bias
        return None, None, pred


def _write_split(root: Path, split: str, data: np.ndarray) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(split_dir / "batch_000.h5", "w") as handle:
        handle.create_dataset("data", data=data)


def _make_frac_root(tmp_path: Path, n: int = 6) -> Path:
    root = tmp_path / "2dFRAC_tung"
    rng = np.random.default_rng(7)
    for split in ("train", "val", "test"):
        initial = rng.normal(size=(n, 8, 8)).astype(np.float32)
        final = (1.25 * initial + 0.1).astype(np.float32)
        data = np.stack([initial, final], axis=1)
        _write_split(root, split, data)
    return root


def test_pair_revin_round_trip():
    pair = torch.arange(2 * 4 * 5, dtype=torch.float32).reshape(2, 4, 5)
    norm, mean, var = pair_revin_normalize(pair)
    recovered = pair_revin_denormalize(norm, mean, var)
    assert torch.allclose(recovered, pair, atol=1e-5, rtol=1e-5)
    assert abs(float(norm.mean())) < 1e-6
    assert abs(float(norm.var(unbiased=False)) - 1.0) < 1e-5


def test_pli_loader_selects_named_field_and_shapes(tmp_path):
    root = tmp_path / "2dHEAT"
    root.mkdir(parents=True)
    (root / "channel_names.txt").write_text("pressure\nav_density\ntemperature\n", encoding="utf-8")

    rng = np.random.default_rng(4)
    for split in ("train", "val", "test"):
        data = rng.normal(size=(3, 2, 3, 12, 10)).astype(np.float32)
        _write_split(root, split, data)

    ds = EndpointH5Dataset(
        root,
        "train",
        "pli",
        normalization="pair-revin",
        validate_spatial_shape=False,
    )
    sample = ds[0]
    assert sample["input"].shape == (1, 1, 1, 1, 12, 10)
    assert sample["target"].shape == (1, 1, 1, 12, 10)
    joined = torch.cat([sample["input"].reshape(-1), sample["target"].reshape(-1)])
    assert abs(float(joined.mean())) < 1e-5
    assert torch.isfinite(joined).all()


def test_frac_loader_shapes_and_max_samples(tmp_path):
    root = _make_frac_root(tmp_path, n=5)
    ds = EndpointH5Dataset(
        root,
        "train",
        "frac",
        normalization="pair-revin",
        max_samples=3,
        validate_spatial_shape=False,
    )
    assert len(ds) == 3
    sample = ds[1]
    assert sample["input"].shape == (1, 1, 1, 1, 8, 8)
    assert sample["target"].shape == (1, 1, 1, 8, 8)
    assert torch.isfinite(sample["input"]).all()
    assert torch.isfinite(sample["target"]).all()


def test_lr_schedule_matches_warmup_and_cosine_endpoints():
    lrs = [epoch_learning_rate(e, 20, 1e-3, warmup_epochs=5, min_lr=1e-7) for e in range(20)]
    assert lrs[0] == pytest.approx(1e-7)
    assert lrs[4] == pytest.approx(1e-3)
    assert lrs[-1] == pytest.approx(1e-7)
    assert max(lrs) == pytest.approx(1e-3)


def test_shape_compatible_checkpoint_loading(tmp_path):
    source = nn.Linear(3, 2)
    target = nn.Linear(4, 2)
    ckpt = tmp_path / "pretrained.pth"
    torch.save({"model_state_dict": source.state_dict()}, ckpt)

    report = load_pretrained_compatible(target, ckpt)
    assert "weight" in report["mismatched_shapes"]
    assert report["loaded_tensor_count"] == 1  # bias matches
    assert torch.allclose(target.bias, source.bias)


def test_training_checkpoint_and_inference_smoke(tmp_path):
    root = _make_frac_root(tmp_path, n=8)
    train_ds = EndpointH5Dataset(
        root,
        "train",
        "frac",
        validate_spatial_shape=False,
    )
    val_ds = EndpointH5Dataset(
        root,
        "val",
        "frac",
        validate_spatial_shape=False,
    )
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    device = torch.device("cpu")
    model = TinyMORPH().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    metrics = evaluate(model, val_loader, device)
    assert np.isfinite(train_loss)
    assert np.isfinite(metrics["mse"])

    checkpoint = tmp_path / "best.pth"
    torch.save(
        checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=1,
            dataset_key="frac",
            model_size="Ti",
            patch_size=(1, 4, 4),
            normalization="pair-revin",
            eps=1e-6,
            train_args={"smoke": True},
            best_val_mse=metrics["mse"],
        ),
        checkpoint,
    )

    restored = TinyMORPH().to(device)
    load_finetuned_checkpoint(restored, checkpoint)
    restored_metrics = evaluate(restored, val_loader, device)
    assert restored_metrics["mse"] == pytest.approx(metrics["mse"], rel=1e-7, abs=1e-9)


def test_entrypoint_help_does_not_require_morph_package():
    for name in ("finetune_pli.py", "infer_pli.py", "finetune_frac.py", "infer_frac.py"):
        proc = subprocess.run(
            [sys.executable, str(ROOT / name), "--help"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert "usage:" in proc.stdout.lower()


def test_build_morph_model_uses_current_package_constructor(monkeypatch):
    import types

    class StubViT(nn.Module):
        def __init__(
            self,
            patch_size,
            dim,
            depth,
            heads,
            heads_xa,
            mlp_dim,
            max_components=3,
            conv_filter=32,
            max_ar=5,
            max_patches=512,
            max_fields=3,
            dropout=0.1,
            emb_dropout=0.1,
            lora_r_attn=0,
            lora_r_mlp=0,
            lora_alpha=None,
            lora_p=0.0,
            model_size="Ti",
            activated_ar1k=False,
        ):
            super().__init__()
            self.patch_size = patch_size
            self.heads_xa = heads_xa
            self.dropout = dropout
            self.emb_dropout = emb_dropout
            self.weight = nn.Parameter(torch.ones(1))

    pkg = types.ModuleType("morph_pde")
    utils = types.ModuleType("morph_pde.utils")
    vit = types.ModuleType("morph_pde.utils.vit_conv_xatt_axialatt2")
    vit.ViT3DRegression = StubViT
    monkeypatch.setitem(sys.modules, "morph_pde", pkg)
    monkeypatch.setitem(sys.modules, "morph_pde.utils", utils)
    monkeypatch.setitem(sys.modules, "morph_pde.utils.vit_conv_xatt_axialatt2", vit)

    model = build_morph_model("Ti", (1, 8, 8), max_ar=1)
    assert model.patch_size == (1, 8, 8)
    assert model.heads_xa == 32
    assert model.dropout == pytest.approx(0.1)
    assert model.emb_dropout == pytest.approx(0.1)
