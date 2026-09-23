#!/usr/bin/env python
"""Evaluate a PLI MORPH checkpoint on the held-out test split."""

from morph_emd.cli import main_inference


if __name__ == "__main__":
    main_inference("pli")
