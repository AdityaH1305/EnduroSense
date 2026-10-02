import numpy as np
import pandas as pd
import pytest

from endurosense.config import data_path
from endurosense.data.clean import (chronological, correct_start_times, fix_glitches,
                                    load_time_corrections, parse_altitude)


def test_chronological_orders_single_digit_hours_correctly():
    # as text, "9:22" sorts after "10:05"; real time order must not
    df = pd.DataFrame({"flight": [1, 2, 3], "date": ["2019-07-09"] * 3,
                       "local_time": ["10:05", "9:22", "16:30"]})
    assert chronological(df)["flight"].tolist() == [2, 1, 3]


def test_time_correction_refuses_unexpected_raw_value():
    fid, c = next(iter(load_time_corrections().items()))
    df = pd.DataFrame({"flight": [fid], "date": [c["date"]], "local_time": ["00:00"]})
    with pytest.raises(ValueError):
        correct_start_times(df)


@pytest.mark.data
def test_corrected_times_make_flight_ids_chronological():
    # flight ids are assigned in time order; after the 4 corrections the raw
    # parameters agree with that everywhere (independent check of the corrections)
    p = correct_start_times(pd.read_csv(data_path("raw") / "parameters.csv"))
    assert chronological(p)["flight"].tolist() == sorted(p["flight"])


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
