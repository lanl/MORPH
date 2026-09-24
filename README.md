<div align="center">

<img src="docs/assets/morph_logo.png" alt="MORPH Physics Foundation Model" width="780">

# MORPH: PDE Foundation Models with Arbitrary Data Modality

<a href="https://mahindrautela.github.io/morph/"><img src="https://img.shields.io/badge/projectpage-morph-blue"></a> <a href="https://arxiv.org/abs/2509.21670"><img src="https://img.shields.io/badge/ArXiv-Preprint-red"></a> <a href="https://huggingface.co/mahindrautela/MORPH"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue"></a>

</div>

MORPH is a multimodal PDE foundation model designed to learn across heterogeneous scientific datasets with different spatial dimensions (1D–3D), resolutions, and physical fields. It is pretrained across diverse PDE systems and transferred to downstream tasks including autoregressive rollouts, terminal key-frame prediction, composite material property estimation, structural damage detection, inertial-confinement-fusion parameter estimation, and sparse sea-surface-temperature reconstruction.

The figure below summarizes the MORPH architecture and its unified treatment of heterogeneous scientific data.

<p align="center">
  <img src="docs/assets/morph_main.png" width="850" alt="Architecture of MORPH">
</p>

<div align="center">

### Pretraining Sets

</div>

<p align="center">
  <img src="docs/assets/pretraining_sets.png" width="850" alt="Pretraining sets">
</p>

<div align="center">

### Finetuning Sets (Fluid Systems - Autoregressive Rollouts)

</div>

<p align="center">
  <img src="docs/assets/fluid_systems_finetuning_sets.png" width="850" alt="Finetuning sets for autoregressive rollouts">
</p>

---

## User Guide

The guide for using MORPH as a standalone surrogate model and as a foundation model is available in [`./docs`](./docs).

## Installation

MORPH is packaged as `morph-pde` and can be imported as:

```python
from morph_pde import MORPH
```

### Install the PLI NOMAD/MCP integration branch

The real-data PLI integration is developed on:

```text
morph-nomad-pli_t0_t99
```

Because this branch contains the fine-tuned MORPH-S checkpoint through Git LFS, cloning the repository is the recommended installation method.

First install and initialize Git LFS:

```bash
git lfs install
```

Clone the integration branch:

```bash
git clone -b morph-nomad-pli_t0_t99 https://github.com/lanl/MORPH.git
cd MORPH
git lfs pull
```

Install MORPH in editable mode:

```bash
python -m pip install -e .
```

Install NOMAD:

```bash
python -m pip install nomad-scifm
```

Optional development dependencies can be installed with:

```bash
python -m pip install -e ".[dev]"
```

For GPU support, install a PyTorch build compatible with your CUDA environment before installing MORPH. See the official PyTorch installation instructions at https://pytorch.org/get-started/locally/.

A quick MORPH installation check is:

```bash
python -c "from morph_pde import MORPH; print(MORPH)"
```

> **Note:** Core MORPH supports Python 3.8+. The NOMAD/MCP integration described below requires Python 3.12+.

---

## NOMAD / MCP Integration

MORPH can be exposed as a Model Context Protocol (MCP) tool using [LANL NOMAD](https://github.com/lanl/nomad). This allows MCP-compatible clients and AI agents to discover and invoke MORPH through a standard tool interface.

The `morph-nomad-pli_t0_t99` branch extends the original `morph-nomad-integration` proof-of-concept with a real scientific inference problem:

* a fine-tuned **MORPH-S** model,
* real PLI input data,
* checkpoint-specific normalization,
* configuration-driven model construction, and
* end-to-end validation through NOMAD/MCP.

The original `morph-nomad-integration` branch is retained as the baseline MORPH-Ti synthetic integration.

### PLI terminal-frame prediction

The current integration demonstrates terminal-frame prediction for PLI:

```text
PLI t=0 physical field
        |
        v
checkpoint-specific normalization
        |
        v
      MORPH-S
        |
        v
    PLI t=99
        |
        v
   denormalization
        |
        v
physical prediction
        |
        v
      NOMAD
        |
        v
       MCP
```

The model predicts the final PLI frame at `t=99` from the initial PLI frame at `t=0`.

This is a single terminal-frame prediction task rather than an autoregressive rollout.

---

## Model Bundle

The complete model artifact is stored at:

```text
models/
└── morph-s-pli/
    ├── config.yaml
    ├── normalization.npy
    └── model.pth
```

The three files have distinct roles:

```text
config.yaml
    Model architecture and task metadata

normalization.npy
    Normalization statistics associated with the fine-tuned checkpoint

model.pth
    Fine-tuned MORPH-S checkpoint
```

`model.pth` is tracked using Git LFS.

### Model configuration

`config.yaml` describes the MORPH-S architecture used for this checkpoint. The relevant model configuration includes:

```yaml
model:
  patch_size: 8
  dim: 512
  depth: 4
  heads: 8
  heads_xa: 32
  mlp_dim: 2048
  max_components: 3
  conv_filter: 8
  max_ar: 1
  max_patches: 4096
  max_fields: 3
  dropout: 0.1
  emb_dropout: 0.1
  lora_r_attn: 0
  lora_r_mlp: 0
  lora_alpha: null
  lora_p: 0.0
  model_size: S
  activated_ar1k: false
```

This avoids hard-coding a specific MORPH model size inside the NOMAD adapter.

---

## Normalization

Normalization is part of the MORPH model bundle rather than the NOMAD configuration.

The PLI checkpoint was fine-tuned using normalization statistics computed from the training/validation data.

For the current checkpoint:

```text
normalization.npy = [mean, variance]
```

and the model input is normalized as:

```text
x_normalized = (x - mean) / variance
```

After inference, the predicted field is transformed back to physical units:

```text
x_physical = x_normalized * variance + mean
```

The current model bundle contains approximately:

```text
mean     = 0.7192955
variance = 3.9962928
```

The MCP client therefore sends the **physical, unnormalized input field**. Normalization and denormalization are performed internally by MORPH.

The normalization tensors are registered as PyTorch buffers so they move with the MORPH model when NOMAD dynamically moves the model between CPU and GPU.

---

## MORPH/NOMAD Interface

The MCP tool accepts a MORPH state tensor in physical units with shape:

```text
(T, F, C, D, H, W)
```

NOMAD batches the request internally, producing:

```text
(B, T, F, C, D, H, W)
```

The terminal-frame prediction returned by MORPH has shape:

```text
(F, C, D, H, W)
```

The current PLI sample therefore enters MCP as:

```text
(1, 1, 1, 1, 1120, 400)
```

and returns:

```text
(1, 1, 1, 1120, 400)
```

---

## NOMAD Configuration

The repository includes [`nomad.yml`](./nomad.yml):

```yaml
fmod_models:
  - model_class: morph_pde.nomad_tool.MORPHTool
    name_or_path: models/morph-s-pli
    tool_name: morph-s-pli
    batch_size: 1
```

`name_or_path` points NOMAD to the complete MORPH model bundle:

```text
models/morph-s-pli/
```

NOMAD exposes the model through MCP as:

```text
morph_s_pli
```

No MORPH-specific normalization parameters are stored in `nomad.yml`.

---

## Start the NOMAD MCP Server

Create or activate a Python 3.12 environment and install MORPH and NOMAD as described above.

From the MORPH repository root, start the server:

```bash
nomad serve --transport http --port 8181 nomad.yml
```

A successful startup should contain messages similar to:

```text
Loading server configuration from nomad.yml
Registering torch model 'morph-s-pli'
Offloaded tool 'morph_s_pli' to CPU
Starting MCP server 'nomad' with transport 'http'
Application startup complete.
Uvicorn running on http://localhost:8181
```

The MCP endpoint is:

```text
http://localhost:8181/mcp
```

Leave this terminal running while testing the MCP client.

If a CUDA device is available, NOMAD dynamically loads the model onto an available GPU when an inference request is received.

For example:

```text
Loading tool 'morph_s_pli' onto cuda:0
```

This behavior has been tested successfully with NVIDIA RTX A6000 GPUs.

---

## Real PLI Input

The integration test uses a PLI HDF5 file containing the initial state.

For the validated example:

```text
datasets/
└── average_jet_first_frame_full_res.h5
```

The test reads:

```text
t0_fields/av_density
```

with source shape:

```text
(1, 1, 1120, 400)
```

and extracts the initial PLI field using:

```python
with h5py.File(input_h5, "r") as f:
    frame = f["t0_fields/av_density"][0, 0, :, :].astype(np.float32)
```

NaN values, if present, are replaced with zero before inference.

The dataset directory is intentionally excluded from Git because scientific datasets can be large.

---

## Run the End-to-End PLI Test

With the NOMAD server running in the first terminal, execute:

```bash
python tests/test_mcp_pli.py \
    --input-h5 datasets/average_jet_first_frame_full_res.h5
```

On Windows Command Prompt:

```bat
python tests\test_mcp_pli.py --input-h5 datasets\average_jet_first_frame_full_res.h5
```

The test uses:

```text
http://localhost:8181/mcp
```

by default.

A different MCP endpoint can be supplied explicitly:

```bash
python tests/test_mcp_pli.py \
    --input-h5 datasets/average_jet_first_frame_full_res.h5 \
    --mcp-url http://localhost:8181/mcp
```

---

## What the Test Validates

The PLI test performs three equivalent inference paths.

### 1. Manual reference inference

```text
physical PLI t=0
        |
        v
manual normalization
        |
        v
bare MORPH-S
        |
        v
manual denormalization
        |
        v
reference prediction
```

### 2. MORPH normalization wrapper

```text
physical PLI t=0
        |
        v
MORPHFieldNormalize
        |
        +-- normalization
        +-- MORPH-S
        +-- denormalization
        |
        v
direct wrapped prediction
```

The wrapped result is compared against the manually constructed reference.

### 3. NOMAD/MCP inference

```text
physical PLI t=0
        |
        v
       MCP
        |
        v
      NOMAD
        |
        v
    MORPHTool
        |
        v
MORPHFieldNormalize
        |
        v
      MORPH-S
        |
        v
physical prediction
```

The MCP prediction is compared against the same manual reference.

This design ensures that an error in the normalization wrapper cannot be hidden by comparing two paths that use the same implementation.

---

## Validated Result

The real PLI integration was validated using:

```text
average_jet_first_frame_full_res.h5
```

with input shape:

```text
(1, 1, 1, 1, 1120, 400)
```

The normalization statistics loaded from the model bundle were:

```text
Normalization mean:     0.7192955017089844
Normalization variance: 3.99629282951355
```

The direct normalization-aware MORPH wrapper reproduced the independent reference exactly:

```text
Reference vs wrapper max abs diff: 0.0
Reference vs wrapper match: True
```

NOMAD successfully exposed:

```text
Available tools: ['get_model_card', 'morph_s_pli']
```

The prediction returned through MCP matched the direct reference exactly:

```text
Reference output shape: (1, 1, 1, 1120, 400)
MCP output shape:       (1, 1, 1, 1120, 400)

Reference vs MCP max abs diff:  0.0
Reference vs MCP mean abs diff: 0.0
Reference vs MCP match: True
```

This validates the complete inference path:

```text
real PLI input
    |
    v
checkpoint-specific normalization
    |
    v
fine-tuned MORPH-S
    |
    v
denormalization
    |
    v
NOMAD
    |
    v
MCP
```

for the tested PLI sample.

---

## MCP Tool Discovery

MCP clients can discover the model through the NOMAD server.

The expected tools include:

```text
get_model_card
morph_s_pli
```

The optional MCP Inspector can also be used:

```bash
npx @modelcontextprotocol/inspector \
    --cli http://localhost:8181/mcp \
    --transport http \
    --method tools/list
```

Node.js is only required for the MCP Inspector and is not required for normal MORPH/NOMAD inference.

---

## NOMAD/MCP Troubleshooting

### `ModuleNotFoundError: No module named 'morph_pde.nomad_tool'`

Make sure the current repository is installed:

```bash
python -m pip install -e .
```

Confirm which MORPH installation Python is using:

```bash
python -c "import morph_pde; print(morph_pde.__file__)"
```

### `ModuleNotFoundError: No module named 'nomad'`

Install NOMAD:

```bash
python -m pip install nomad-scifm
```

### `ModuleNotFoundError: No module named 'yaml'`

Install the current MORPH package again:

```bash
python -m pip install -e .
```

`PyYAML` is included as a MORPH package dependency on this branch.

### Missing Git LFS checkpoint

Confirm Git LFS is installed:

```bash
git lfs install
```

Then download the LFS artifacts:

```bash
git lfs pull
```

Verify:

```bash
git lfs ls-files
```

The output should include:

```text
models/morph-s-pli/model.pth
```

### `FileNotFoundError` for the model artifact

Confirm that the following files exist:

```text
models/morph-s-pli/config.yaml
models/morph-s-pli/normalization.npy
models/morph-s-pli/model.pth
```

### MCP connection failure

Make sure the NOMAD server is running:

```bash
nomad serve --transport http --port 8181 nomad.yml
```

and that the client uses:

```text
http://localhost:8181/mcp
```

### Checkpoint archive error

A valid PyTorch checkpoint should load independently:

```bash
python -c "import torch; x=torch.load('models/morph-s-pli/model.pth', map_location='cpu', weights_only=False); print(x.keys())"
```

The current fine-tuning checkpoint contains training metadata in addition to `model_state_dict`, so the integration currently loads this trusted local artifact with `weights_only=False`.

---

## Current Integration Limitations

The current NOMAD/MCP integration is intentionally focused on validating one complete real scientific inference workflow.

Current limitations are:

* the demonstrated model is a fine-tuned **MORPH-S** checkpoint for PLI `t=0 -> t=99` terminal-frame prediction;
* the current PLI normalization artifact stores a scalar `[mean, variance]` pair;
* HDF5 parsing and extraction of `t0_fields/av_density` currently occur in the client/test rather than inside the MCP tool;
* the MCP interface currently receives an already prepared MORPH tensor;
* the current model artifact contains the full fine-tuning checkpoint, including training metadata, rather than an inference-only weights checkpoint;
* autoregressive multi-step rollout through NOMAD/MCP has not yet been implemented; and
* additional MORPH model sizes, datasets, fields, and downstream tasks have not yet been validated through the same adapter.

The next integration step is to extend the same model-bundle abstraction:

```text
checkpoint
+ model configuration
+ normalization statistics
```

to MORPH autoregressive rollout tasks, where an initial physical state is normalized internally and rolled forward to a predefined prediction horizon.

---

## Development Branches

The NOMAD/MCP work is intentionally separated into branches.

### `morph-nomad-integration`

Baseline proof-of-concept:

```text
MORPH-Ti
+ synthetic tensor
+ single forward prediction
+ direct vs MCP equivalence
```

This branch is retained as the original minimal NOMAD/MCP integration.

### `morph-nomad-pli_t0_t99`

Real-data integration:

```text
MORPH-S
+ config-driven model construction
+ checkpoint-specific normalization
+ real PLI t=0 field
+ t=99 terminal-frame prediction
+ NOMAD device management
+ MCP inference
+ direct vs MCP equivalence
```

This branch is the current real-data validation of the MORPH/NOMAD integration.

---

## Citation

If you use MORPH in your research, please cite:

```bibtex
@article{rautela2025morph,
  title={MORPH: PDE Foundation Models with Arbitrary Data Modality},
  author={Rautela, Mahindra Singh and Most, Alexander and Mansingh, Siddharth and Love, Bradley C and Biswas, Ayan and Oyen, Diane and Lawrence, Earl},
  journal={arXiv preprint arXiv:2509.21670},
  year={2025}
}
```

#### Note

EIDR number **O#4999** — *MORPH: Shape-agnostic PDE Foundational Models.*

This program is Open-Source under the MIT License.
