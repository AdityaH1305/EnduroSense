"""Common interface for every Model A predictor.

All models take whole data frames (rows of ``data/features/model_a.parquet``)
rather than bare arrays, so baselines, tabular models and sequence models can
each pull the columns they need and be cross-validated by the same code.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class Regressor(Protocol):
    name: str

    def fit(self, train: pd.DataFrame, target: str) -> "Regressor": ...

    def predict(self, df: pd.DataFrame) -> np.ndarray: ...
