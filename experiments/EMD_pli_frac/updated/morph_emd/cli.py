"""CLI implementation shared by the four small PLI/FRAC entry points."""

from __future__ import print_function

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .common import (
    DATASET_SPECS,
    EndpointH5Dataset,
    build_morph_model,
    checkpoint_payload,
    choose_device,
    default_dataset_dir,
    epoch_learning_rate,
    evaluate,
    load_finetuned_checkpoint,
    load_pretrained_compatible,
    pair_revin_denormalize,
    save_json,
    set_learning_rate,
    train_one_epoch,
    unwrap_model,
)


def _dataset_args(parser: argparse.ArgumentParser, dataset_key: str) -> None:
    spec = DATASET_SPECS[dataset_key]
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=default_dataset_dir(dataset_key),
        help="Dataset root containing train/val/test (default: %(default)s)",
    )
    parser.add_argument(
        "--normalization",
        choices=("pair-revin", "none"),
        default="pair-revin",
        help=(
            "pair-revin reproduces the historical endpoint-pair RevIN. Use none "
            "only if the HDF5 files are already normalized."
        ),
    )
    parser.add_argument("--normalization-eps", type=float, default=1.0e-6)
    parser.add_argument("--field-name", default=spec.field_name)
    parser.add_argument(
        "--field-index",
        type=int,
        default=None,
        help="PLI only: explicit channel index if no channel-name text file is present",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")


def _patch_args(parser: argparse.ArgumentParser, dataset_key: str) -> None:
    p_d, p_h, p_w = DATASET_SPECS[dataset_key].default_patch
    parser.add_argument("--patch-d", type=int, default=p_d)
    parser.add_argument("--patch-h", type=int, default=p_h)
    parser.add_argument("--patch-w", type=int, default=p_w)


def _make_dataset(args, dataset_key: str, split: str, max_samples: Optional[int]):
    return EndpointH5Dataset(
        dataset_dir=args.dataset_dir,
        split=split,
        dataset_key=dataset_key,
        field_index=args.field_index,
        field_name=args.field_name,
        normalization=args.normalization,
        eps=args.normalization_eps,
        max_samples=max_samples,
    )


def _loader(dataset, batch_size: int, shuffle: bool, args):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def finetune_parser(dataset_key: str) -> argparse.ArgumentParser:
    spec = DATASET_SPECS[dataset_key]
    parser = argparse.ArgumentParser(
        description="Fine-tune MORPH-Ti for {} first-frame -> terminal-state prediction".format(dataset_key.upper())
    )
    _dataset_args(parser, dataset_key)
    _patch_args(parser, dataset_key)
    parser.add_argument("--model-size", choices=("Ti", "S", "M", "L"), default="Ti")
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=None,
        help="Foundation-model checkpoint. Omit only with --random-init.",
    )
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=spec.default_lr)
    parser.add_argument("--min-lr", type=float, default=1.0e-7)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=spec.default_batch_size,
        help="Global/effective batch size (paper default: %(default)s)",
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument(
        "--data-parallel",
        action="store_true",
        help="Use torch.nn.DataParallel when multiple CUDA devices are visible",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / (dataset_key + "_morph_ti"),
    )
    return parser


def run_finetune_cli(dataset_key: str) -> None:
    args = finetune_parser(dataset_key).parse_args()
    if args.random_init and args.pretrained_checkpoint is not None:
        raise SystemExit("Choose either --random-init or --pretrained-checkpoint, not both")
    if not args.random_init and args.pretrained_checkpoint is None:
        raise SystemExit("Pass --pretrained-checkpoint PATH, or use --random-init")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("--epochs and --batch-size must be positive")

    patch_size = (args.patch_d, args.patch_h, args.patch_w)
    device = choose_device(args.device)
    print("Dataset : {}".format(dataset_key.upper()))
    print("Data    : {}".format(args.dataset_dir))
    print("Device  : {}".format(device))
    print("Patch   : {}".format(patch_size))
    print("Norm    : {}".format(args.normalization))

    train_ds = _make_dataset(args, dataset_key, "train", args.max_train_samples)
    val_ds = _make_dataset(args, dataset_key, "val", args.max_val_samples)
    print("Samples : train={} val={}".format(len(train_ds), len(val_ds)))

    train_loader = _loader(train_ds, args.batch_size, True, args)
    val_loader = _loader(val_ds, args.batch_size, False, args)

    model = build_morph_model(args.model_size, patch_size, max_ar=1)
    preload_report = None
    if args.pretrained_checkpoint is not None:
        preload_report = load_pretrained_compatible(model, args.pretrained_checkpoint, map_location="cpu")
        print(
            "Pretrained load: {}/{} tensors loaded; {} shape mismatches".format(
                preload_report["loaded_tensor_count"],
                preload_report["checkpoint_tensor_count"],
                len(preload_report["mismatched_shapes"]),
            )
        )
        if preload_report["mismatched_shapes"]:
            print("Shape-mismatched tensors were reinitialized:")
            for key, shapes in sorted(preload_report["mismatched_shapes"].items()):
                print("  {}: checkpoint {} -> model {}".format(key, shapes["checkpoint"], shapes["model"]))

    model.to(device)
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print("DataParallel GPUs: {}".format(torch.cuda.device_count()))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best.pth"
    last_path = args.output_dir / "last.pth"
    history = []
    best_val = float("inf")
    stale_epochs = 0

    serializable_args = vars(args).copy()
    for key, value in list(serializable_args.items()):
        if isinstance(value, Path):
            serializable_args[key] = str(value)

    for epoch in range(args.epochs):
        lr = epoch_learning_rate(
            epoch,
            args.epochs,
            peak_lr=args.lr,
            warmup_epochs=args.warmup_epochs,
            min_lr=args.min_lr,
        )
        set_learning_rate(optimizer, lr)
        start = time.time()
        train_mse = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - start
        row = {
            "epoch": epoch + 1,
            "lr": lr,
            "train_mse": train_mse,
            "val_mse": val_metrics["mse"],
            "val_rmse": val_metrics["rmse"],
            "seconds": elapsed,
        }
        history.append(row)
        print(
            "epoch {:4d}/{:4d} lr={:.3e} train_mse={:.6g} val_mse={:.6g}".format(
                epoch + 1, args.epochs, lr, train_mse, val_metrics["mse"]
            )
        )

        improved = val_metrics["mse"] < best_val
        if improved:
            best_val = val_metrics["mse"]
            stale_epochs = 0
            torch.save(
                checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    dataset_key=dataset_key,
                    model_size=args.model_size,
                    patch_size=patch_size,
                    normalization=args.normalization,
                    eps=args.normalization_eps,
                    train_args=serializable_args,
                    best_val_mse=best_val,
                    preload_report=preload_report,
                ),
                str(best_path),
            )
        elif epoch + 1 > args.warmup_epochs:
            stale_epochs += 1

        torch.save(
            checkpoint_payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                dataset_key=dataset_key,
                model_size=args.model_size,
                patch_size=patch_size,
                normalization=args.normalization,
                eps=args.normalization_eps,
                train_args=serializable_args,
                best_val_mse=best_val,
                preload_report=preload_report,
            ),
            str(last_path),
        )
        save_json(args.output_dir / "history.json", history)

        if stale_epochs >= args.patience:
            print("Early stopping after {} non-improving validation epochs.".format(stale_epochs))
            break

    print("Best checkpoint: {}".format(best_path))
    print("Last checkpoint: {}".format(last_path))


def inference_parser(dataset_key: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned MORPH checkpoint on {} test endpoints".format(dataset_key.upper())
    )
    _dataset_args(parser, dataset_key)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--allow-config-override",
        action="store_true",
        help="Allow CLI normalization to differ from the checkpoint metadata",
    )
    return parser


@torch.no_grad()
def _raw_metrics(model, loader, device, eps: float) -> Dict[str, float]:
    """Compute optional physical-scale metrics by undoing pair RevIN per sample."""
    model.eval()
    sq = 0.0
    ab = 0.0
    n = 0
    for batch in loader:
        x = batch["input"].to(device)
        y = batch["target"].to(device)
        output = model(x)
        pred = output[-1] if isinstance(output, (tuple, list)) else output
        mean = batch["mean"].to(device).view(-1, 1, 1, 1, 1)
        var = batch["var"].to(device).view(-1, 1, 1, 1, 1)
        pred_raw = pair_revin_denormalize(pred, mean, var, eps=eps)
        y_raw = pair_revin_denormalize(y, mean, var, eps=eps)
        diff = pred_raw - y_raw
        sq += float(torch.sum(diff * diff).cpu())
        ab += float(torch.sum(torch.abs(diff)).cpu())
        n += int(diff.numel())
    return {
        "raw_mse": sq / n,
        "raw_rmse": (sq / n) ** 0.5,
        "raw_mae": ab / n,
    }


def run_inference_cli(dataset_key: str) -> None:
    args = inference_parser(dataset_key).parse_args()
    device = choose_device(args.device)
    try:
        checkpoint = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(args.checkpoint), map_location="cpu")

    ckpt_dataset = checkpoint.get("dataset_key", dataset_key)
    if ckpt_dataset != dataset_key:
        raise SystemExit(
            "Checkpoint dataset_key={!r}, but this is the {} inference script".format(
                ckpt_dataset, dataset_key.upper()
            )
        )
    model_size = checkpoint.get("model_size", "Ti")
    patch_size = tuple(checkpoint.get("patch_size", DATASET_SPECS[dataset_key].default_patch))
    ckpt_norm = checkpoint.get("normalization", "pair-revin")
    ckpt_eps = float(checkpoint.get("normalization_eps", args.normalization_eps))
    if not args.allow_config_override:
        args.normalization = ckpt_norm
        args.normalization_eps = ckpt_eps

    test_ds = _make_dataset(args, dataset_key, "test", args.max_test_samples)
    test_loader = _loader(test_ds, args.batch_size, False, args)
    model = build_morph_model(model_size, patch_size, max_ar=1)
    load_finetuned_checkpoint(model, args.checkpoint, map_location="cpu")
    model.to(device)

    normalized = evaluate(model, test_loader, device)
    results = {
        "dataset": dataset_key,
        "checkpoint": str(args.checkpoint),
        "normalization": args.normalization,
        "test_mse": normalized["mse"],
        "test_rmse": normalized["rmse"],
        "test_mae": normalized["mae"],
        "test_samples": int(normalized["samples"]),
    }
    if args.normalization == "pair-revin":
        results.update(_raw_metrics(model, test_loader, device, args.normalization_eps))

    print(json.dumps(results, indent=2, sort_keys=True))
    if args.output_json is not None:
        save_json(args.output_json, results)


def main_finetune(dataset_key: str) -> None:
    run_finetune_cli(dataset_key)


def main_inference(dataset_key: str) -> None:
    run_inference_cli(dataset_key)
