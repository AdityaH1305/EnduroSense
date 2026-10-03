"""Model B component models and mission assembly."""
import numpy as np
import pandas as pd
import pytest

from endurosense.mission import MissionSpec
from endurosense.models.model_b import (COMPONENTS, LegOverheadPhysics, LegPowerPhysics, WorkPhysics,
                                        fit_mission_model, physics_model)


def test_momentum_theory_power_curve_has_the_right_shape():
    theta = np.array([3.0, 100.0, 1.3, 0.05])
    p = LegPowerPhysics.power(theta, np.array([0.0, 4.0, 8.0, 16.0]), np.zeros(4))
    assert p[1] < p[0]                 # induced power falls as forward speed rises...
    assert p[3] > p[2]                 # ...until drag (v^3) dominates
    heavy = LegPowerPhysics.power(theta, np.array([8.0]), np.array([500.0]))
    assert heavy[0] > p[2]             # more mass, more power


def test_physics_power_fit_respects_airframe_mass_range():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"speed": rng.choice([4, 6, 8, 10, 12], 300).astype(float),
                       "payload": rng.choice([0, 250, 500], 300).astype(float)})
    df["power_w"] = LegPowerPhysics.power([2.8, 120, 1.4, 0.04], df["speed"], df["payload"]) + rng.normal(0, 5, 300)
    m = LegPowerPhysics().fit(df, "power_w")
    assert 2.4 <= m.theta[0] <= 3.6
    assert np.abs(m.predict(df) - df["power_w"]).mean() < 6


def test_kinematic_overhead_recovers_acceleration():
    v = np.repeat([4.0, 6.0, 8.0, 10.0, 12.0], 20)
    df = pd.DataFrame({"speed": v, "overhead_s": v / 2.5 - 1.0})
    m = LegOverheadPhysics().fit(df, "overhead_s")
    assert 1 / m.inv_a == pytest.approx(2.5) and m.c == pytest.approx(-1.0)


def test_work_physics_is_linear_in_height_and_payload():
    df = pd.DataFrame({"alt_cruise_m": [25, 50, 100, 25, 50, 100.0], "payload": [0, 0, 0, 500, 500, 500.0]})
    df["climb_wh"] = 0.1 + df["alt_cruise_m"] * (0.05 + 0.02 * df["payload"] / 1000)
    m = WorkPhysics("alt_cruise_m").fit(df, "climb_wh")
    np.testing.assert_allclose(m.coef, [0.1, 0.05, 0.02], atol=1e-9)


@pytest.fixture(scope="module")
def dev_b():
    from endurosense.config import data_path
    from endurosense.data.split import DEV, select
    path = data_path("features") / "model_b_legs.parquet"
    if not path.exists():
        pytest.skip("run scripts/04_build_features.py first")
    legs = select(pd.read_parquet(path), DEV)
    flights = select(pd.read_parquet(data_path("features") / "model_b_flights.parquet"), DEV)
    return flights, legs


@pytest.mark.data
def test_batch_and_single_mission_paths_agree(dev_b):
    """predict_flights (vectorised, used for evaluation) must equal mission() (used by
    the scheduler and demo) for the same plan."""
    flights, legs = dev_b
    model = fit_mission_model(physics_model, flights, legs)
    row = flights.iloc[10]
    fl_legs = legs[legs["flight"] == row["flight"]].sort_values("leg")
    spec = MissionSpec.loop(fl_legs["distance_m"].tolist(), row["payload"], row["speed"], row["alt_cruise_m"],
                            row["ambient_wind"])
    batch = model.predict_flights(flights.iloc[[10]], fl_legs).iloc[0]
    assert model.mission(spec)["total_wh"] == pytest.approx(batch, rel=1e-9)


@pytest.mark.data
def test_component_models_never_see_validation_flights(dev_b):
    from endurosense.data.split import dev_folds
    flights, legs = dev_b
    for _, tr, va in dev_folds(flights):
        assert set(tr["flight"]).isdisjoint(va["flight"])
        assert set(tr["battery_chain"]).isdisjoint(va["battery_chain"])


def test_every_component_has_planning_time_features_only():
    banned = {"ground_speed", "power_w", "duration_s", "energy_wh", "overhead_s", "total_wh"}
    for comp, (target, feats) in COMPONENTS.items():
        assert target not in feats
        assert not banned & set(feats), comp
