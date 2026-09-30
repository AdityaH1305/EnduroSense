import pandas as pd

from endurosense.data.chains import assign_battery_chains


def _summary(rows):
    return pd.DataFrame(rows, columns=["date", "local_time", "v_start", "v_end"])


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


def test_ids_follow_time_order_not_row_order():
    s = _summary([
        ("2019-06-01", "10:20", 24.5, 23.3),
        ("2019-06-01", "10:00", 25.6, 24.4),
    ])
    assert assign_battery_chains(s, tol_v=0.3).tolist() == [1, 1]
