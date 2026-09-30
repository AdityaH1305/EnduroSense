import numpy as np
import pandas as pd

from endurosense.data.clean import fix_glitches, parse_altitude


def test_parse_altitude_numeric_and_varying_labels():
    out = parse_altitude(pd.Series(["25", "100", "25-50-100-25", "0"]))
    assert out["alt_cruise_m"].tolist() == [25.0, 100.0, 100.0, 0.0]
    assert out["varying_altitude"].tolist() == [False, False, True, False]


def _flight(v, i):
    n = len(v)
    return pd.DataFrame({"flight": 1, "time": np.arange(n) * 0.12,
                         "battery_voltage": np.array(v, float), "battery_current": np.array(i, float)})


def test_isolated_voltage_glitch_is_replaced():
    df = fix_glitches(_flight([24.0] * 5 + [22.5] + [24.0] * 5, [20.0] * 11))
    assert df["v_glitch"].sum() == 1
    assert df["battery_voltage"].iloc[5] == 24.0


def test_real_load_step_is_kept():
    # voltage sags because current rises: a real load change, not a glitch
    v = [24.0] * 5 + [22.5] + [24.0] * 5
    i = [20.0] * 5 + [35.0] + [20.0] * 5
    df = fix_glitches(_flight(v, i))
    assert not df["v_glitch"].any() and not df["i_glitch"].any()
    assert df["battery_voltage"].iloc[5] == 22.5


def test_current_glitch_without_voltage_response_is_replaced():
    df = fix_glitches(_flight([24.0] * 11, [20.0] * 5 + [45.0] + [20.0] * 5))
    assert df["i_glitch"].sum() == 1
    assert df["battery_current"].iloc[5] == 20.0
