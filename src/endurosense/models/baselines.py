"""Model A baselines that machine learning has to beat.

These are what a standard battery management system (BMS) can do without
machine learning. They are deliberately *fair*: each knows the battery's state
of charge when it was inserted (from its rest voltage) rather than assuming
every battery starts full, which would make ML look better than it is
(the naive fixed-capacity version errs by ~13.6 Wh in Phase 2 checks).

- ``FixedCapacity``: average usable energy of a battery minus energy drawn so
  far. Assumes every battery starts identical and full. Reference only.
- ``EnergyCounting``: state of charge at insertion from the chain-start rest
  voltage (via the discharge curve), then subtract energy drawn ("coulomb
  counting", as a BMS does).
- ``VoltageLookup``: re-reads the state of charge from the latest rest
  voltage (before this flight) and subtracts only this flight's energy.

The discharge curve is fitted on the training fold's chains only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from endurosense.config import load_config
from endurosense.features.model_a import DischargeCurve, fit_discharge_curve


class _CurveBaseline:
    name = "baseline"

    def __init__(self, flights: pd.DataFrame):
        self.flights = flights
        self.reserve = load_config()["battery"]["reserve_v"]
        self.curve: DischargeCurve | None = None
        self.fallback_wh = np.nan

    def _fit_curve(self, train: pd.DataFrame, target: str):
        self.curve = fit_discharge_curve(self.flights, train["flight"].unique())
        # used when no rest voltage is known (a few rows): average starting energy
        start = train.sort_values("t_chain_s").groupby("battery_chain").first()
        self.fallback_wh = float((start[target] + start["e_chain_wh"]).mean())
        self.flights = None          # only the fitted curve is part of the model (keeps saved size honest)

    def _wh_to_reserve(self, v: pd.Series) -> np.ndarray:
        return np.array([self.curve.energy_between(x, self.reserve) if np.isfinite(x) else np.nan for x in v])

    def _cached(self, v: pd.Series) -> np.ndarray:
        # rest voltages repeat across many rows; compute each distinct value once
        uniq = pd.unique(v.dropna())
        table = dict(zip(uniq, self._wh_to_reserve(pd.Series(uniq))))
        return v.map(table).to_numpy(dtype=float)


class FixedCapacity(_CurveBaseline):
    name = "Fixed capacity (reference)"

    def fit(self, train, target):
        self._fit_curve(train, target)
        return self

    def predict(self, df):
        return self.fallback_wh - df["e_chain_wh"].to_numpy()


class EnergyCounting(_CurveBaseline):
    name = "Energy counting (BMS)"

    def fit(self, train, target):
        self._fit_curve(train, target)
        return self

    def predict(self, df):
        start = self._cached(df["v_rest_chain_start"])
        start = np.where(np.isfinite(start), start, self.fallback_wh)
        return start - df["e_chain_wh"].to_numpy()


class VoltageLookup(_CurveBaseline):
    name = "Voltage lookup"

    def fit(self, train, target):
        self._fit_curve(train, target)
        return self

    def predict(self, df):
        now = self._cached(df["v_rest_flight_start"])
        counting = EnergyCounting.predict(self, df)
        return np.where(np.isfinite(now), now - df["e_flight_wh"].to_numpy(), counting)
