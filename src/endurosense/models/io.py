"""Load the fitted models saved by the pipeline scripts (no retraining).

``scripts/05_model_a.py`` saves every Model A algorithm fitted on all development
data: tabular models and baselines as pickles, the LSTM and GRU as weights plus
their configuration and input scalers. ``scripts/06_model_b.py`` saves every
Model B variant as a pickle.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import torch

from endurosense.config import data_path
from endurosense.models.sequence import SequenceModel, WindowStore, _Net


def model_a_path(name: str, kind: str) -> Path:
    """File a Model A algorithm was saved to (same naming rule as ``scripts/05_model_a.py``)."""
    folder = data_path("models") / "model_a"
    if kind in ("lstm", "gru"):
        return folder / f"{kind}.pt"
    return folder / f"{name.replace(' ', '_').replace('+', 'plus').replace('(', '').replace(')', '')}.pkl"


def load_sequence_model(path, windows: WindowStore, device: str | None = None) -> SequenceModel:
    """Rebuild a saved LSTM / GRU. ``windows`` holds the sequence inputs of the rows to predict."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    m = SequenceModel(blob["kind"], windows, device=device, **blob["config"])
    for k, v in blob["scalers"].items():
        setattr(m, k, v)
    m.net = _Net(blob["kind"], windows.X.shape[2], len(m.features), m.hidden, m.layers, m.dropout)
    m.net.load_state_dict(blob["state"])
    m.net.to(m.device).eval()
    return m


def load_model_a(name: str, kind: str, windows: WindowStore):
    path = model_a_path(name, kind)
    if kind in ("lstm", "gru"):
        return load_sequence_model(path, windows)
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_model_b(name: str):
    """A saved Model B variant by its display name, e.g. ``"Physics-first"``."""
    with open(data_path("models") / "model_b" / f"{name.replace(' ', '_').replace('+', 'plus')}.pkl", "rb") as fh:
        return pickle.load(fh)
