"""Read the raw CMU dataset files and the processed Phase 1 tables."""
from __future__ import annotations

import pandas as pd

from endurosense.config import data_path

PROCESSED_TABLES = ("samples", "flights", "chains")


def load_parameters() -> pd.DataFrame:
    """One row per flight: planned speed, payload, altitude, date, time and route."""
    return pd.read_csv(data_path("raw") / "parameters.csv")


def load_flights() -> pd.DataFrame:
    """All sensor readings (~258k rows, 28 columns) from ``flights.csv``.

    ``low_memory=False`` because the ``altitude`` column mixes numbers with
    labels such as ``"25-50-100-25"`` for flights that change altitude.
    """
    return pd.read_csv(data_path("raw") / "flights.csv", low_memory=False)


def save_processed(name: str, df: pd.DataFrame) -> None:
    """Write a processed table to ``data/processed/<name>.parquet``."""
    if name not in PROCESSED_TABLES:
        raise ValueError(f"unknown table {name!r}")
    out = data_path("processed")
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"{name}.parquet", index=False)


def load_processed(name: str) -> pd.DataFrame:
    """Read a table written by ``scripts/02_prepare_data.py`` (samples, flights or chains)."""
    path = data_path("processed") / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/02_prepare_data.py first")
    return pd.read_parquet(path)
