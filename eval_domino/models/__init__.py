"""Model-family adapters. Each module exposes: FAMILY, CKPT_ROOT, checkpoint_dir(),
verify_layout(), smoke_cmd(). Adding a family = adding a module here."""
from __future__ import annotations

from importlib import import_module


def get(family: str):
    return import_module(f"eval_domino.models.{family}")
