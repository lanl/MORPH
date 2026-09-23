# MORPH PLI / FRAC experiment cleanup

This directory keeps the copied historical experiment scripts for provenance and adds a small,
package-compatible path for reproducing the MORPH terminal-state experiments.

The new code intentionally does **not** modify the Poseidon scripts.

## New entry points

- `finetune_pli.py` - MORPH fine-tuning on `datasets/2dHEAT`
- `infer_pli.py` - held-out PLI evaluation
- `finetune_frac.py` - MORPH fine-tuning on `datasets/2dFRAC_tung`
- `infer_frac.py` - held-out FRAC evaluation
- `morph_emd/` - shared lazy HDF5 loader, pair-RevIN, model/checkpoint and training utilities
- `tests/` - unit and smoke tests

The only MORPH package import used by the clean path is:

```python
from morph_pde.utils.vit_conv_xatt_axialatt2 import ViT3DRegression
```

This matches the `package-morph` package layout. The old `src.utils.*` imports are intentionally
not used by the new entry points.

## Expected data layout

The existing repository-level dataset folders are used directly; no new 80/10/10 split is made:

```text
MORPH/
├── datasets/
│   ├── 2dHEAT/
│   │   ├── train/*.h5
│   │   ├── val/*.h5
│   │   ├── test/*.h5
│   │   └── channel_names.txt        # or common_channels.txt
│   └── 2dFRAC_tung/
│       ├── train/*.h5
│       ├── val/*.h5
│       └── test/*.h5
└── experiments/
    └── EMD_pli_frac/
        └── ...
```

Supported HDF5 `data` layouts are:

- PLI / `2dHEAT`: `(N, 2, F, H, W)`
- FRAC / `2dFRAC_tung`: `(N, 2, H, W)` or `(N, 2, 1, H, W)`

The two time entries are interpreted as `t0` and `tT`.

PLI uses the `av-density` / `av_density` field. If no channel-name text file is available,
pass `--field-index INDEX` explicitly. The historical visualization appears to use index 2 for
average density, but the clean loader does not assume that silently.

## Normalization reproduced from the historical script

The copied `normalization/data_normalization_revin_cl.py` computes MORPH `RevIN` statistics per
sample and per field over the complete two-frame sample:

```text
[t0, tT] -> mean and variance over time + space -> normalize t0 and tT
```

For the single field used here this is:

```python
mu  = pair.mean()
var = pair.var()
pair_normalized = (pair - mu) / sqrt(var + eps)
```

The clean scripts do this on-the-fly with `--normalization pair-revin` (the default), so separate
`stats_*.npy` files are not required.

**Important:** this is an offline benchmark normalization because the statistics use both `t0` and
the target `tT`. It reproduces the historical/paper evaluation, but it is not a target-free deployment
normalization. A deployment/NOMAD path should instead use statistics computable from the input alone
or training-set statistics, and should be retrained/evaluated consistently with that choice.

If `datasets/2dHEAT` or `datasets/2dFRAC_tung` already contains the output of the old RevIN
preprocessing script, use:

```bash
--normalization none
```

Do not normalize an already-normalized HDF5 copy a second time.

## Paper-reproduction defaults

| Setting | PLI | FRAC |
|---|---:|---:|
| task | `t0 -> tT` | `t0 -> tT` |
| MORPH size | Ti | Ti |
| field | `av-density` | damage/fracture field |
| native grid | 1120 x 400 | 128 x 128 |
| patch | 8 x 8 | 4 x 4 |
| peak LR | 1e-4 | 1e-3 |
| AdamW weight decay | 1e-2 | 1e-2 |
| effective/global batch | 8 | 512 |
| warmup | 5 epochs from 1e-7 | 5 epochs from 1e-7 |
| schedule | cosine after warmup | cosine after warmup |
| early-stop patience | 10 | 10 |
| max epochs | 200 | 200 |
| loss/reporting scale | normalized MSE | normalized MSE |

### PLI patch-count inconsistency

There is a reproducibility inconsistency in the copied sources/paper:

- the paper says `8 x 8` PLI patches but also says this gives about 3,500 patches;
- `1120 x 400` with `8 x 8` actually gives 7,000 patches;
- copied `dataloaders/data_config.py` contains `(1,16,8)` and explicitly comments that it gives
  3,500 patches;
- copied executable `finetune_MORPH.py`, however, overrides the config and uses `(1,8,8)`.

The new PLI script defaults to `8 x 8` because that is both the paper's stated patch size and the
setting used by the copied executable fine-tuning script. To test the stale/config alternative, use
`--patch-h 16 --patch-w 8` explicitly.

## Fine-tuning

Install the package branch from the MORPH repository root first:

```bash
python -m pip install -e .
```

### PLI

```bash
python experiments/EMD_pli_frac/finetune_pli.py \
  --dataset-dir datasets/2dHEAT \
  --pretrained-checkpoint /path/to/MORPH-Ti-pretrained.pth \
  --data-parallel \
  --output-dir experiments/results/pli_morph_ti
```

If `channel_names.txt` / `common_channels.txt` is not present, add `--field-index INDEX`.

### FRAC

```bash
python experiments/EMD_pli_frac/finetune_frac.py \
  --dataset-dir datasets/2dFRAC_tung \
  --pretrained-checkpoint /path/to/MORPH-Ti-pretrained.pth \
  --data-parallel \
  --output-dir experiments/results/frac_morph_ti
```

For training from scratch, replace `--pretrained-checkpoint ...` with `--random-init`.

The scripts save:

```text
<output-dir>/best.pth
<output-dir>/last.pth
<output-dir>/history.json
```

Checkpoint metadata records the dataset, patch size, normalization mode, normalization epsilon,
training arguments, validation score, and the pretrained-load compatibility report.

### FRAC checkpoint compatibility

The released/pretrained MORPH model uses width-8 patch-dependent projection/decoder tensors.
Fine-tuning FRAC at 4 x 4 changes those tensor shapes. PyTorch `strict=False` does not ignore a
same-name tensor with a different shape. The clean loader therefore loads every matching pretrained
tensor and explicitly reports/reinitializes only shape-mismatched patch-dependent tensors.

This behavior is visible in the console and stored in the fine-tuned checkpoint metadata; it is not
silently ignored.

## Inference / test evaluation

### PLI

```bash
python experiments/EMD_pli_frac/infer_pli.py \
  --dataset-dir datasets/2dHEAT \
  --checkpoint experiments/results/pli_morph_ti/best.pth \
  --batch-size 1 \
  --output-json experiments/results/pli_morph_ti/test_metrics.json
```

### FRAC

```bash
python experiments/EMD_pli_frac/infer_frac.py \
  --dataset-dir datasets/2dFRAC_tung \
  --checkpoint experiments/results/frac_morph_ti/best.pth \
  --batch-size 64 \
  --output-json experiments/results/frac_morph_ti/test_metrics.json
```

The headline `test_mse` is computed in normalized space, matching the paper protocol. For
`pair-revin`, the script also reports denormalized/raw MSE/RMSE/MAE as auxiliary diagnostics.

## Tests

From `experiments/EMD_pli_frac`:

```bash
pytest -q
```

The included tests cover:

- PLI and FRAC HDF5 layouts and MORPH tensor shapes;
- pair-RevIN normalize/denormalize round-trip;
- five-epoch warmup + cosine schedule endpoints;
- safe partial loading of patch-shape-mismatched pretrained checkpoints;
- a CPU train -> save -> reload -> inference smoke path using a tiny model with MORPH's tensor contract;
- compatibility of the clean model-builder arguments with the current `ViT3DRegression` constructor;
- all four entry-point `--help` commands;
- preservation of the copied Poseidon files.

## Legacy script audit

The copied files are useful provenance, but most should not be the public/recommended execution path.

| File | Status | Main finding |
|---|---|---|
| `dataloaders/argument_parser_cl.py` | archive | Continual-learning/general parser; much broader than PLI/FRAC reproduction. |
| `dataloaders/data_config.py` | archive | Has PLI/FRAC metadata, but assumes `datasets/normalized_revin/...`; contains the 16x8-vs-8x8 PLI inconsistency and a 320k FRAC training count inconsistent with the paper's 160k horizontal-BC training set. |
| `dataloaders/dataloader_heat2d.py` | useful legacy | Correct basic HDF5 layout, but loads full files into RAM and uses unsorted directory order. Clean loader is lazy and deterministic. |
| `dataloaders/dataloader_frac2d.py` | useful legacy | Same issue: eager full-file loading. Clean loader is lazy. |
| `dataloaders/*poseidon*.py` | untouched | Preserved exactly as requested. `dataloaderchaos_poseidon.py` still has old `src.utils.*` imports. |
| `finetune_MORPH.py` | superseded for PLI/FRAC | Generic and cluttered; old `src.utils.*`; old dataset paths; expects modified `DataloaderChaos` methods not in current package; outdated fine-tuning-selector call; unconditional `wandb`; batch defaults do not directly match the paper's stated effective batches. |
| `morph_fast_ft.py` | do not use for paper result | Separate PLI raw-NPZ experiment. Uses one global scalar mean/variance and divides by **variance** rather than standard deviation; this is not the paper RevIN path. |
| `morph_fast_infer.py` | do not use | Matches the separate global-stat path, contains a `breakpoint()`, and is PLI-only. |
| `morph_infer_first_to_last.py` | do not use | Uses old namespace, contains breakpoints, expects combined stats filenames not produced by the copied RevIN preprocessing script, and its shown denormalization multiplies by variance rather than `sqrt(var+eps)`. |
| `morph_infer_hf.py` | separate future deployment idea | Uses `morph.utils...` instead of `morph_pde...` and a stored field-normalization scheme different from the paper's per-sample pair-RevIN. Do not mix it into paper reproduction. |
| `normalization/data_normalization_revin_cl.py` | key provenance, superseded operationally | Confirms two-frame per-sample/per-field RevIN, but has hardcoded Windows paths and imports a missing `config.data_config_vis`. Clean code reproduces its math on-the-fly. |
| `visualizers/process_and_visualize_HEAT.py` | preprocessing provenance | Builds endpoint HDF5 data from raw HEAT; hardcoded local paths. Writes `common_channels.txt` in a different location from where old training expects `channel_names.txt`. |
| `visualizers/process_hf_fracture.py` | preprocessing provenance | Processes both `combined_bc` and `horizontal_bc`, whereas the paper uses the tungsten horizontal-BC subset. Hardcoded local paths and old helper import. |
| `visualizers/visualize_heat_sim1_tAll.py` | visualization only | Hardcoded local raw-data path. |
| `visualizers/visualize_hf_fracture.py` | visualization only | Hardcoded local path and old helper import. |

## What is missing for bit-for-bit reproduction of the published numbers

The copied folder is enough to reconstruct the intended task and normalization, but it is **not a
complete bit-for-bit experiment snapshot**. The following provenance is still missing/ambiguous:

1. The exact released MORPH-Ti pretrained checkpoint filename/hash used for each run.
2. The exact fine-tuned checkpoints that produced Table 1.
3. The exact PLI channel-name file/index mapping used for `av-density` if it is not inside `datasets/2dHEAT`.
4. The old modified `DataloaderChaos` implementation that supplied `load_heat2d` and `load_frac2d` to `finetune_MORPH.py`.
5. `config/data_config_vis.py`, imported by the historical normalization script.
6. The saved `data/stats_*` arrays from the historical preprocessing; these are no longer required by the clean scripts because the same per-pair statistics are recomputed on-the-fly.
7. Resolution of the PLI `8x8 / 7000` versus `16x8 / 3500` patch inconsistency.
8. Confirmation that the local `datasets/2dFRAC_tung` split is the paper's **horizontal-boundary-condition-only** subset (paper: ~200k total, 160k training), rather than the older combined+horizontal preprocessing pool.
9. Exact random seeds/subsample indices used for every data-scaling point, if the scaling curves need exact numerical reproduction.

For normal full-shot PLI/FRAC fine-tuning and test inference, the new scripts remove dependencies on
items 4-6 and make items 1, 3, 7, and 8 explicit instead of hiding them in old local paths.
