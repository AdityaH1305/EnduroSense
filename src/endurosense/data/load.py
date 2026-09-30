"""Read the raw CMU dataset files.

Phase 0 keeps this minimal; Phase 1 adds explicit dtypes, cleaning and a
parquet cache.
"""
from __future__ import annotations

import pandas as pd

from endurosense.config import data_path


def load_parameters() -> pd.DataFrame:
    """One row per flight: planned speed, payload, altitude, date, time and route."""
    return pd.read_csv(data_path("raw") / "parameters.csv")


def load_flights() -> pd.DataFrame:
    """All sensor readings (~258k rows, 28 columns) from ``flights.csv``.

    ``low_memory=False`` because the ``altitude`` column mixes numbers with
    labels such as ``"25-50-100-25"`` for flights that change altitude.
    """
    return pd.read_csv(data_path("raw") / "flights.csv", low_memory=False)
