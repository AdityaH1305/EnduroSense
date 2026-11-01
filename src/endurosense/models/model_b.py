"""Model B: energy a planned mission will need, built from per-part predictors.

Plain idea: predict each part of a mission separately (power and time of each
leg, climb, descent, the pauses between legs, ground time) and add them up
with ``endurosense.mission.mission_energy``. Each part has:

- a **physics model**: a few physical parameters fitted to development data;
- **machine-learning models** (Linear Regression, Random Forest, XGBoost) on
  the same planning-time inputs;
- a **hybrid**: the physics model plus XGBoost learning only its errors.

Physics models
--------------
Leg power (rotor momentum theory). For total mass m (airframe + payload):

    v_h = sqrt(m g / (2 rho A))                                   hover induced velocity
    v_i = sqrt(-v^2/2 + sqrt(v^4/4 + v_h^4))                       induced velocity at speed v
    P   = P0 + k_ind * m g * v_i + k_par * v^3

P0 covers avionics and blade profile power, k_ind the inefficiency of real
rotors (ideal = 1), k_par body drag. Induced power falls with speed and drag
rises with speed cubed, which gives the flat-then-rising power curve seen in
the data.

The airframe mass m0 (with sensor rig) is undocumented, so it is fitted within
the DJI Matrice 100's real range: 2.4 kg (with battery) to 3.6 kg (maximum
take-off). Left free, the fit drifts to an unphysical ~5 kg with P0 = 0 at the
*same* accuracy (cross-validated MAE 22.0 vs 22.3 W): this data cannot
separate mass, constant power and rotor inefficiency, so the physics model is
a physically shaped curve and its individual parameters should not be read
as measurements.

Leg time (kinematics). Accelerating to speed v and braking at rate a adds about
v / a seconds to distance / v:  T = d / v + v / a + c.

Climb, descent and hover. Energy proportional to the work involved:
height x (alpha + beta * payload) for climb and descent, and
n_legs x (alpha + beta * payload) for the pauses between legs, plus an offset.

LSTM/GRU are not used here: the inputs are a mission's settings, not a time
series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from endurosense.config import data_path, load_config
from endurosense.mission import MissionSpec, mission_energy
from endurosense.models import tabular as T
from endurosense.models.tabular import TabularModel

G, RHO = 9.81, 1.225
ROTOR_AREA_M2 = 4 * np.pi * 0.165 ** 2        # DJI Matrice 100: four 13-inch propellers

# component -> (target column, planning-time features)
COMPONENTS = {
    "leg_power": ("power_w", ["speed", "payload", "alt_cruise_m", "ambient_wind"]),
    "leg_overhead": ("overhead_s", ["speed", "distance_m", "payload", "ambient_wind"]),
    "climb": ("climb_wh", ["alt_cruise_m", "payload", "ambient_wind"]),
    "descent": ("descent_wh", ["alt_cruise_m", "payload", "ambient_wind"]),
    "hover": ("hover_wh", ["n_legs", "payload", "ambient_wind"]),
}
LEG_COMPONENTS = ("leg_power", "leg_overhead")

FAMILIES = ["Physics", "Linear Regression", "Random Forest", "XGBoost", "Physics + XGBoost"]
PHYSICS_STRUCTURED = ["Physics", "Physics + XGBoost"]
MAIN = "Physics-first"                         # the Model B combination used by later phases
_TABULAR_KIND = {"Linear Regression": "linear", "Random Forest": "random_forest", "XGBoost": "xgboost"}


# ------------------------------------------------------------------- physics
class LegPowerPhysics:
    name = "Physics"
    _x0 = np.array([3.0, 100.0, 1.2, 0.02])          # m0 kg, P0 W, k_ind, k_par
    _lo = np.array([2.4, 0.0, 0.5, 0.0])             # m0: M100 spec mass with battery ...
    _hi = np.array([3.6, 400.0, 3.0, 0.5])           # ... up to its maximum take-off mass

    @staticmethod
    def power(theta, speed, payload_g):
        m0, p0, k_ind, k_par = theta
        w = (m0 + np.asarray(payload_g) / 1000.0) * G
        v = np.asarray(speed, dtype=float)
        vh4 = (w / (2 * RHO * ROTOR_AREA_M2)) ** 2
        vi = np.sqrt(-v ** 2 / 2 + np.sqrt(v ** 4 / 4 + vh4))
        return p0 + k_ind * w * vi + k_par * v ** 3

    def fit(self, df, target):
        res = least_squares(lambda th: self.power(th, df["speed"], df["payload"]) - df[target].to_numpy(),
                            self._x0, bounds=(self._lo, self._hi))
        self.theta = res.x
        return self

    def predict(self, df):
        return self.power(self.theta, df["speed"], df["payload"])


class LegOverheadPhysics:
    """Extra leg time from accelerating and braking: v / a + c."""
    name = "Physics"

    def fit(self, df, target):
        X = np.c_[df["speed"].to_numpy(), np.ones(len(df))]
        (self.inv_a, self.c), *_ = np.linalg.lstsq(X, df[target].to_numpy(), rcond=None)
        return self

    def predict(self, df):
        return self.inv_a * df["speed"].to_numpy() + self.c


class WorkPhysics:
    """offset + scale x (alpha + beta * payload_kg); scale is height or number of legs."""
    name = "Physics"

    def __init__(self, scale_col: str):
        self.scale_col = scale_col

    def _X(self, df):
        s, pay = df[self.scale_col].to_numpy(dtype=float), df["payload"].to_numpy() / 1000.0
        return np.c_[np.ones(len(df)), s, s * pay]

    def fit(self, df, target):
        self.coef, *_ = np.linalg.lstsq(self._X(df), df[target].to_numpy(), rcond=None)
        return self

    def predict(self, df):
        return self._X(df) @ self.coef


def physics_model(component: str):
    return {"leg_power": LegPowerPhysics, "leg_overhead": LegOverheadPhysics,
            "climb": lambda: WorkPhysics("alt_cruise_m"), "descent": lambda: WorkPhysics("alt_cruise_m"),
            "hover": lambda: WorkPhysics("n_legs")}[component]()


class HybridModel:
    """Physics first; XGBoost learns what the physics model gets wrong."""
    name = "Physics + XGBoost"

    def __init__(self, physics, residual_model: TabularModel):
        self.physics, self.residual = physics, residual_model

    def fit(self, df, target):
        self.physics.fit(df, target)
        self.residual.fit(df.assign(_r=df[target] - self.physics.predict(df)), "_r")
        return self

    def predict(self, df):
        return self.physics.predict(df) + self.residual.predict(df)


# ------------------------------------------------------- mission-level model
class MissionEnergyModel:
    """Fitted component predictors that together implement ``mission.Components``."""

    def __init__(self, predictors: dict, ground_wh: float, ranges: dict | None = None):
        self.predictors, self.ground = predictors, ground_wh
        self.ranges = ranges or {}                 # (min, max) of each input in the training data

    def _one(self, comp, **cols):
        # energies and overheads cannot be negative; a model may predict so when extrapolating
        return max(0.0, float(self.predictors[comp].predict(pd.DataFrame({k: [v] for k, v in cols.items()}))[0]))

    def extrapolation_warnings(self, spec: MissionSpec) -> list[str]:
        """Inputs of ``spec`` that lie outside the range the model was trained on.

        Predictions there are extrapolations: physics-shaped parts degrade
        gracefully (2.5-3.5% error on left-out settings), but nothing beyond the
        recorded ranges has been tested at all.
        """
        values = {"speed": [spec.speed], "alt_cruise_m": [spec.altitude_m], "ambient_wind": [spec.wind],
                  "payload": list(spec.payload_g), "distance_m": list(spec.legs_m), "n_legs": [spec.n_legs]}
        label = {"speed": ("speed", "m/s"), "alt_cruise_m": ("altitude", "m"), "ambient_wind": ("wind", "m/s"),
                 "payload": ("payload", "g"), "distance_m": ("leg length", "m"), "n_legs": ("number of legs", "")}
        out = []
        for key, vals in values.items():
            if key not in self.ranges:
                continue
            lo, hi = self.ranges[key]
            name, unit = label[key]
            for v in sorted(set(vals)):
                if not lo <= v <= hi:
                    out.append(f"{name} {v:g} {unit} is outside the tested range "
                               f"{round(lo, 1):g}-{round(hi, 1):g} {unit}".replace("  ", " ").strip())
        return out

    def climb_wh(self, altitude_m, payload_g, wind):
        return self._one("climb", alt_cruise_m=altitude_m, payload=payload_g, ambient_wind=wind)

    def descent_wh(self, altitude_m, payload_g, wind):
        return self._one("descent", alt_cruise_m=altitude_m, payload=payload_g, ambient_wind=wind)

    def hover_wh(self, n_legs, payload_g, wind):
        return self._one("hover", n_legs=n_legs, payload=payload_g, ambient_wind=wind)

    def leg_power_w(self, speed, payload_g, altitude_m, wind):
        return self._one("leg_power", speed=speed, payload=payload_g, alt_cruise_m=altitude_m, ambient_wind=wind)

    def leg_time_s(self, distance_m, speed, payload_g, wind):
        over = self._one("leg_overhead", speed=speed, distance_m=distance_m, payload=payload_g, ambient_wind=wind)
        return distance_m / speed + over

    def ground_wh(self):
        return self.ground

    def mission(self, spec: MissionSpec) -> dict:
        return mission_energy(spec, self)

    def predict_flights(self, flights: pd.DataFrame, legs: pd.DataFrame) -> pd.Series:
        """Vectorised mission energy for recorded flights, rebuilt from their plan:
        measured leg lengths, commanded speed, payload, cruise altitude, wind."""
        pos = lambda comp, df: np.maximum(self.predictors[comp].predict(df), 0.0)   # same clamp as _one
        legs = legs.copy()
        legs["t_s"] = legs["distance_m"] / legs["speed"] + pos("leg_overhead", legs)
        legs["e_wh"] = pos("leg_power", legs) * legs["t_s"] / 3600.0
        leg_wh = legs.groupby("flight")["e_wh"].sum()
        f = flights.set_index("flight")
        total = (pos("climb", f) + pos("descent", f) + pos("hover", f) + self.ground
                 + leg_wh.reindex(f.index).to_numpy())
        return pd.Series(total, index=f.index, name="pred_total_wh")


def fit_mission_model(make_predictor, flights: pd.DataFrame, legs: pd.DataFrame) -> MissionEnergyModel:
    """Fit one predictor per component. ``make_predictor(component)`` returns an unfitted model."""
    preds = {}
    for comp, (target, _) in COMPONENTS.items():
        rows = legs[legs["at_commanded_speed"]] if comp in LEG_COMPONENTS else flights
        preds[comp] = make_predictor(comp).fit(rows, target)
    used = legs[legs["at_commanded_speed"]]
    ranges = {c: (float(flights[c].min()), float(flights[c].max()))
              for c in ("speed", "payload", "alt_cruise_m", "ambient_wind", "n_legs")}
    ranges["distance_m"] = (float(used["distance_m"].min()), float(used["distance_m"].max()))
    return MissionEnergyModel(preds, ground_wh=float(flights["ground_wh"].median()), ranges=ranges)


# ------------------------------------------------------ building combinations
def choose_family(scores: dict, rule: str) -> str:
    """Pick the algorithm for one part from its CV errors.

    ``best``: lowest error. ``physics_first``: the better physics-structured
    part, unless a pure-ML part is more than ``physics_first_margin`` better.
    Physics-structured parts extrapolate to unseen settings and showed half the
    systematic bias at the highest speed, at the same overall accuracy.
    """
    best_all = min(scores, key=scores.get)
    if rule == "best":
        return best_all
    best_phys = min(PHYSICS_STRUCTURED, key=scores.get)
    margin = load_config()["model_b"]["physics_first_margin"]
    return best_all if scores[best_all] < (1 - margin) * scores[best_phys] else best_phys


def make_component(family: str, comp: str, cfg: dict | None, drop: tuple = ()):
    """Unfitted predictor for one component (``drop``: input columns to withhold)."""
    feats = [f for f in COMPONENTS[comp][1] if f not in drop]
    if family == "Physics":
        return physics_model(comp)
    if family == "Physics + XGBoost":
        resid = T.xgboost(**cfg)
        resid.features = feats
        return HybridModel(physics_model(comp), resid)
    m = T.FACTORIES[_TABULAR_KIND[family]](**cfg)
    m.features = feats
    return m


def mission_factory(choice: dict, drop: tuple = ()):
    """``choice``: component -> (family, config). Returns ``make_predictor(component)``."""
    return lambda comp: make_component(choice[comp][0], comp, choice[comp][1], drop)


def main_choice() -> dict:
    """The main Model B combination and its tuned configurations, as selected by
    ``scripts/06_model_b.py`` (read from its saved results)."""
    import json

    out = data_path("results") / "model_b"
    log = pd.read_csv(out / "component_search_log.csv")
    best_cfg = {(r.family, r.component): json.loads(r.config)
                for r in log.loc[log.groupby(["family", "component"])["cv_mae"].idxmin()].itertuples()}
    families = json.loads((out / "combinations.json").read_text())[MAIN]
    return {c: (fam, best_cfg[(fam, c)]) for c, fam in families.items()}
