<div align="center">

<img src="docs/assets/morph_logo.png" alt="MORPH Physics Foundation Model" width="780">

# MORPH: PDE Foundation Models with Arbitrary Data Modality
<a href="https://mahindrautela.github.io/morph/"><img src="https://img.shields.io/badge/projectpage-morph-blue"></a> <a href="https://arxiv.org/abs/2509.21670"><img src="https://img.shields.io/badge/ArXiv-Preprint-red"></a> <a href="https://huggingface.co/mahindrautela/MORPH"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue"></a>

</div>

MORPH is a multimodal PDE foundation model designed to learn across heterogeneous scientific datasets with different spatial dimensions (1D–3D), resolutions, and physical fields. It is pretrained across diverse PDE systems and transferred to downstream tasks including autoregressive rollouts, terminal key-frame prediction, composite material property estimation, structural damage detection, inertial-confinement-fusion parameter estimation, and sparse sea-surface-temperature reconstruction. The figure below summarizes the MORPH architecture and its unified treatment of heterogeneous scientific data.


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

### User Guide

The guide for using MORPH as a standalone surrogate model and as a foundation model is available in [`./docs`](./docs).

### Installation

MORPH can be installed directly from GitHub:

```bash
pip install git+https://github.com/lanl/MORPH.git
```

For GPU support, install the appropriate PyTorch version for your CUDA environment before installing MORPH. For example, for CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

MORPH can then be imported as:

```python
from morph_pde import MORPH
```

For development, clone the repository and install MORPH in editable mode:

```bash
git clone https://github.com/lanl/MORPH.git
cd MORPH
pip install -e .
```

Optional development dependencies can be installed with:

```bash
pip install -e ".[dev]"
```


MORPH can also be installed directly from GitHub:

```bash
pip install git+https://github.com/lanl/MORPH.git
```

### Citation

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