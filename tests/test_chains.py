import pandas as pd

from endurosense.data.chains import assign_battery_chains


def _summary(rows):
    s = pd.DataFrame(rows, columns=["date", "local_time", "v_start", "v_end"])
    s["flight"] = range(1, len(s) + 1)
    return s


def test_matching_voltage_continues_chain_and_jump_starts_new_one():
    s = _summary([
        ("2019-06-01", "10:00", 25.6, 24.4),
        ("2019-06-01", "10:10", 24.5, 23.3),   # continues (0.1 V)
        ("2019-06-01", "10:20", 25.7, 24.5),   # fresh battery (jump 2.4 V)
    ])
    assert assign_battery_chains(s, tol_v=0.3).tolist() == [1, 1, 2]


def test_new_day_always_starts_new_chain():
    s = _summary([
        ("2019-06-01", "17:00", 25.6, 24.4),
        ("2019-06-02", "09:00", 24.4, 23.2),   # same voltage, different day
    ])
    assert assign_battery_chains(s, tol_v=0.3).tolist() == [1, 2]


def test_window_rejects_voltage_drop_between_flights():
    # a resting battery only recovers voltage; a clear drop means a different battery
    s = pd.DataFrame({"flight": [1, 2, 3], "date": ["2019-06-01"] * 3, "local_time": ["10:00", "10:10", "10:20"],
                      "v_rest_start": [25.6, 24.6, 23.2], "v_rest_end": [24.5, 23.5, 22.4]})
    ids = assign_battery_chains(s, start_col="v_rest_start", end_col="v_rest_end", window=(-0.10, 0.40))
    assert ids.tolist() == [1, 1, 2]        # +0.10 V links; -0.30 V does not


def test_overrides_force_break_and_join():
    s = _summary([("2019-06-01", "10:00", 25.6, 24.4), ("2019-06-01", "10:10", 24.5, 23.3), ("2019-06-01", "10:20", 25.7, 24.5)])
    s["flight"] = [1, 2, 3]
    assert assign_battery_chains(s, tol_v=0.3, overrides={"break_before": [2]}).tolist() == [1, 2, 3]
    assert assign_battery_chains(s, tol_v=0.3, overrides={"join_to_previous": [3]}).tolist() == [1, 1, 1]


def test_ids_follow_time_order_not_row_order():
    s = _summary([
        ("2019-06-01", "10:20", 24.5, 23.3),
        ("2019-06-01", "10:00", 25.6, 24.4),
    ])
    assert assign_battery_chains(s, tol_v=0.3).tolist() == [1, 1]
