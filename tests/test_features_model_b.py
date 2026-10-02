"""Model B tables and mission assembly."""
import pytest

from endurosense.mission import ConstantComponents, MissionSpec, mission_energy


def test_delivery_is_out_and_back_with_payload_dropped():
    m = MissionSpec.delivery(300, payload_g=500, speed=8, altitude_m=50)
    assert m.legs_m == (300, 300) and m.payload_g == (500, 0.0) and m.distance_m == 600


def test_mission_spec_validates_input():
    with pytest.raises(ValueError):
        MissionSpec((100.0, 200.0), (0.0,), 8, 50)
    with pytest.raises(ValueError):
        MissionSpec((100.0,), (0.0,), 0, 50)


def test_assembly_arithmetic_with_constant_components():
    c = ConstantComponents()
    m = MissionSpec.loop([160.0, 160.0], payload_g=0, speed=8, altitude_m=50)
    out = mission_energy(m, c)
    leg = c.power_w * (160 / 8 + c.overhead_s) / 3600
    expected = c.climb_wh_per_m * 50 + 2 * leg + 2 * c.hover_wh_per_leg + c.descent_wh_per_m * 50 + c.ground
    assert out["total_wh"] == pytest.approx(expected)
    assert out["leg_wh"] == pytest.approx([leg, leg])


def test_longer_mission_needs_more_energy():
    c = ConstantComponents()
    short = mission_energy(MissionSpec.delivery(200, 250, 8, 50), c)["total_wh"]
    long = mission_energy(MissionSpec.delivery(400, 250, 8, 50), c)["total_wh"]
    assert long > short


@pytest.mark.data
def test_flight_breakdown_adds_up_exactly():
    import pandas as pd
    from endurosense.config import data_path
    t = pd.read_parquet(data_path("features") / "model_b_flights.parquet")
    parts = t[["climb_wh", "legs_wh", "hover_wh", "other_wh", "descent_wh", "ground_wh"]].sum(axis=1)
    assert (parts - t["total_wh"]).abs().max() < 1e-9
    assert 250 not in set(t["flight"])                          # truncated landing excluded


@pytest.mark.data
def test_legs_are_physical():
    import pandas as pd
    from endurosense.config import data_path
    legs = pd.read_parquet(data_path("features") / "model_b_legs.parquet")
    assert (legs["duration_s"] >= 3.0).all()
    assert legs["power_w"].between(300, 800).all()
    # legs faster than commanded are rare repositioning segments and must be flagged
    ok = legs[legs["at_commanded_speed"]]
    assert (ok["ground_speed"] <= ok["speed"] * 1.05).all()
    assert (~legs["at_commanded_speed"]).sum() <= 0.01 * len(legs)
