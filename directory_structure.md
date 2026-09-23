# MORPH Repository Structure

```text
MORPH/
|-- pyproject.toml          # Python package configuration
|-- README.md               # Project overview and installation
|-- LICENSE
|-- environment.yml
|
|-- src/
|   `-- morph_pde/          # Installable MORPH Python package
|       |-- __init__.py
|       |-- model.py        # Public MORPH model interface
|       |
|       |-- config/         # Model and dataset configuration
|       |   |-- __init__.py
|       |   |-- argument_parser.py
|       |   |-- data_config.py
|       |   `-- data_config_vis.py
|       |
|       `-- utils/          # MORPH model components and utilities
|           |-- __init__.py
|           |-- dataloaders/
|           |-- datastreamers/
|           `-- ...
|
|-- scripts/                # Pretraining, finetuning, and inference scripts
|-- experiments/            # Downstream-task experiments
|-- docs/                   # User documentation
`-- tests/                  # Package tests
    `-- test_import.py
```

The installable Python package is located under `src/morph_pde`. Research workflows and downstream experiments remain outside the package under `scripts/` and `experiments/`.
