---
language:
  - en
tags:
  - type:model
  - science:fusion
  - pde
  - morph
  - pli
  - terminal-frame-prediction
  - risk:general
license: mit
base_model: MORPH-S
---

# MORPH-S PLI t0-to-t99

MORPH-S fine-tuned for terminal-frame prediction of the PLI
average-density field.

The model takes the physical `av_density` field at `t=0` and
predicts the corresponding field at `t=99`.

## Model short description

- Architecture: MORPH-S
- Task: terminal-frame prediction
- Input frame: t=0
- Output frame: t=99
- Physical field: `av_density`
- Spatial resolution: 1120 x 400
- MORPH tensor input: `(T, F, C, D, H, W)`

## Model Type

MORPH-S PDE foundation model fine-tuned for the PLI terminal-frame
prediction task.

## Inputs and outputs

### Input

The model receives the physical, unnormalized PLI `av_density`
field at `t=0`.

For the current MCP interface, one request has shape:

`(T, F, C, D, H, W) = (1, 1, 1, 1, 1120, 400)`.

### Output

The model returns the predicted physical `av_density` field at
`t=99` with shape:

`(F, C, D, H, W) = (1, 1, 1, 1120, 400)`.

## Normalization

Normalization is part of the model artifact and is performed
internally during inference.

`normalization.npy` contains:

`[mean, variance]`

The transformation used during fine-tuning and inference is:

`x_norm = (x - mean) / variance`

The predicted normalized field is transformed back to physical
space as:

`x = x_norm * variance + mean`

The current artifact contains approximately:

- mean: 0.7192955
- variance: 3.9962928

Clients should provide data in physical units and should not
normalize the input themselves.

## Model configuration

The architecture is defined by `config.yaml`.

Current model:

- model size: S
- patch size: 8
- embedding dimension: 512
- depth: 4
- attention heads: 8
- MLP dimension: 2048
- maximum autoregressive order: 1

## How to use

This model is exposed through NOMAD as:

`morph_s_pli`

Start NOMAD from the MORPH repository:

```bash
nomad serve --transport http --port 8181 nomad.yml

## Papers and Scientific Outputs

The underlying MORPH foundation model is introduced in:

```bibtex
@article{rautela2025morph,
  title={Morph: Pde foundation models with arbitrary data modality},
  author={Rautela, Mahindra Singh and Most, Alexander and Mansingh, Siddharth and Love, Bradley C and Scheinker, Alexander and Oyen, Diane and Debardeleben, Nathan and Lawrence, Earl and Biswas, Ayan},
  journal={arXiv preprint arXiv:2509.21670},
  year={2025},
  url={https://arxiv.org/abs/2509.21670}
}
```

The PLI downstream transfer setting associated with this model is described in:

```bibtex
@article{rautela2026out,
  title={Out-of-distribution transfer of PDE foundation models to material dynamics under extreme loading},
  author={Rautela, Mahindra and Most, Alexander and Mansingh, Siddharth and Pachalieva, Aleksandra and Love, Bradley and Malley, Daniel O and Scheinker, Alexander and Hickmann, Kyle and Oyen, Diane and Debardeleben, Nathan and others},
  journal={arXiv preprint arXiv:2603.04354},
  year={2026},
  url={https://arxiv.org/abs/2603.04354}
}
```
