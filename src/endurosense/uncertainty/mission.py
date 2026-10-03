"""Calibrated predictive distribution for Model B (energy a mission needs).

Plain idea: on held-out flights we know how wrong each part of the mission
estimate was. To get a range for a *new* mission we replay every held-out
flight's set of errors on the new mission's parts and look at the spread of
totals.

Errors are replayed **together, per flight**, because they are correlated:
e.g. energy counted as "descent" on one flight is energy not counted as
"ground" (correlation -0.63), so treating parts as independent would overstate
the total uncertainty (0.82 vs the real 0.69 Wh).

Climb, descent, hover and ground errors are kept in Wh (they barely grow with
altitude); the cruise-leg error is kept as a *percentage*, so that a mission
with more or longer legs gets proportionally more uncertainty.

``method="total"`` is the simpler alternative (one percentage error on the
whole mission), kept for comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from endurosense.mission import MissionSpec
from endurosense.uncertainty.quantiles import PredictiveDistribution, conformal_quantiles

ABS_PARTS = ("climb", "descent", "hover", "ground")


def predicted_parts(model, flights: pd.DataFrame, legs: pd.DataFrame) -> pd.DataFrame:
    """Per-flight predicted energy of each part (Wh), indexed by flight."""
    pos = lambda comp, df: np.maximum(model.predictors[comp].predict(df), 0.0)
    legs = legs.copy()
    legs["e"] = pos("leg_power", legs) * (legs["distance_m"] / legs["speed"] + pos("leg_overhead", legs)) / 3600.0
    f = flights.set_index("flight")
    return pd.DataFrame({"climb": pos("climb", f), "descent": pos("descent", f), "hover": pos("hover", f),
                         "legs": legs.groupby("flight")["e"].sum().reindex(f.index).to_numpy(),
                         "ground": model.ground}, index=f.index)


def residual_tuples(pred: pd.DataFrame, flights: pd.DataFrame) -> pd.DataFrame:
    """One row of errors per held-out flight: Wh for the fixed parts, a ratio for the legs and the total."""
    f = flights.set_index("flight").loc[pred.index]
    actual_legs = f["legs_wh"] + f["other_wh"]
    out = pd.DataFrame({p: f[f"{p}_wh"] - pred[p] for p in ABS_PARTS})
    out["legs_rel"] = actual_legs / pred["legs"] - 1.0
    out["total_rel"] = f["total_wh"] / pred.sum(axis=1) - 1.0
    out["battery_chain"] = f["battery_chain"]
    return out


def replay(parts: dict, tuples: pd.DataFrame, method: str = "parts") -> np.ndarray:
    """Mission totals implied by each calibration flight's errors."""
    total = sum(parts[p] for p in (*ABS_PARTS, "legs"))
    if method == "total":
        return total * (1.0 + tuples["total_rel"].to_numpy())
    return (total + sum(tuples[p].to_numpy() for p in ABS_PARTS) + parts["legs"] * tuples["legs_rel"].to_numpy())


def quantiles_from_replay(totals: np.ndarray, groups, levels, correction: str | None = None) -> np.ndarray:
    return conformal_quantiles(totals, groups, levels, correction)


def replay_compound(parts_list: list[dict], tuples: pd.DataFrame, n: int, rng: np.random.Generator,
                    method: str = "parts") -> np.ndarray:
    """Totals for a mission made of several sorties flown one after another.

    Each sortie gets its own, independently drawn, held-out error set (the errors
    of different flights are not tied together), and the sorties are added up.
    """
    total = np.zeros(n)
    for parts in parts_list:
        idx = rng.integers(0, len(tuples), n)
        total += replay(parts, tuples.iloc[idx], method)
    return total


def oof_parts(make_predictor, flights: pd.DataFrame, legs: pd.DataFrame, folds) -> tuple[pd.DataFrame, pd.Series]:
    """Out-of-fold predicted parts for every flight: each flight is predicted by a
    model trained without its fold. ``folds`` yields ``(k, train_flights, val_flights)``."""
    from endurosense.models.model_b import fit_mission_model

    preds, fold = [], []
    for k, tr, va in folds:
        m = fit_mission_model(make_predictor, tr, legs[legs["flight"].isin(tr["flight"])])
        preds.append(predicted_parts(m, va, legs[legs["flight"].isin(va["flight"])]))
        fold.append(pd.Series(k, index=preds[-1].index))
    return pd.concat(preds), pd.concat(fold)


def leave_fold_out_quantiles(pred: pd.DataFrame, tuples: pd.DataFrame, fold: pd.Series, levels,
                             method: str = "parts", correction: str | None = None) -> np.ndarray:
    """Held-out mission quantiles per flight, using only the other folds' error sets."""
    out = np.zeros((len(pred), len(levels)))
    for i, fl in enumerate(pred.index):
        cal = tuples[(fold != fold[fl]).to_numpy()]
        out[i] = quantiles_from_replay(replay(pred.loc[fl].to_dict(), cal, method), cal["battery_chain"], levels, correction)
    return out


class CalibratedMissionModel:
    """Mission-energy model + held-out error tuples: gives a distribution for any mission."""

    def __init__(self, model, tuples: pd.DataFrame, levels, method: str = "parts"):
        self.model, self.tuples, self.levels, self.method = model, tuples, np.asarray(levels, float), method

    def distribution_from_parts(self, parts: dict) -> PredictiveDistribution:
        totals = np.maximum(replay(parts, self.tuples, self.method), 0.0)
        return PredictiveDistribution(self.levels, quantiles_from_replay(totals, self.tuples["battery_chain"], self.levels))

    def distribution(self, spec: MissionSpec) -> PredictiveDistribution:
        m = self.model.mission(spec)
        return self.distribution_from_parts({"climb": m["climb_wh"], "descent": m["descent_wh"], "hover": m["hover_wh"],
                                             "legs": m["legs_wh"], "ground": m["ground_wh"]})

    def warnings(self, spec: MissionSpec) -> list[str]:
        return self.model.extrapolation_warnings(spec)
