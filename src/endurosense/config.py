"""Project configuration, paths and random seeding.

All tunable settings live in ``config.yaml`` at the project root. Code reads
them through ``load_config()`` instead of hard-coding numbers, so a change
requested in review (e.g. the reserve voltage) is a one-line edit.
"""
from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.yaml"


@lru_cache(maxsize=None)
def load_config(path: str | Path = CONFIG_PATH) -> dict:
    """Read the YAML config (cached; call ``load_config.cache_clear()`` after editing)."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def data_path(key: str) -> Path:
    """Absolute path for a key under ``paths`` in the config, e.g. ``data_path("raw")``."""
    return ROOT / load_config()["paths"][key]


def set_seed(seed: int | None = None) -> int:
    """Seed Python, NumPy and (if installed) PyTorch; returns the seed used."""
    seed = load_config()["seed"] if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed
