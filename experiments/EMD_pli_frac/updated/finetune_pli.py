#!/usr/bin/env python
"""Fine-tune MORPH on PLI (2dHEAT): first frame -> terminal frame."""

from morph_emd.cli import main_finetune


if __name__ == "__main__":
    main_finetune("pli")
