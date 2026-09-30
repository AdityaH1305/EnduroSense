import numpy as np
import pandas as pd

from endurosense.data.phases import check_phase_order, merge_short_runs, phase_runs, segment_flight


def synthetic_flight(drift_after_landing_m: float = 0.0) -> pd.DataFrame:
    """Ground 5 s -> climb to 30 m at 2.5 m/s -> cruise 20 s at 8 m/s -> descend -> ground 5 s."""
    dt = 0.1
    segs = [("ground", 5.0, 0.0, 0.0, 0.2),
            ("climb", 12.0, 2.5, 0.0, 22.0),
            ("cruise", 20.0, 0.0, 8.0, 24.0),
            ("descent", 15.0, -2.0, 0.0, 18.0),
            ("ground", 5.0, 0.0, 0.0, 0.0)]
    rows, z = [], 300.0
    for name, dur, vz, vx, cur in segs:
        for _ in range(int(dur / dt)):
            z += vz * dt
            rows.append((vz, vx, cur, z, name))
    df = pd.DataFrame(rows, columns=["velocity_z", "velocity_x", "battery_current", "position_z", "truth"])
    df["velocity_y"] = 0.0
    df["time"] = np.arange(len(df)) * dt
    df.loc[df.index[-50:], "position_z"] += drift_after_landing_m   # altimeter drift after landing
    return df


def test_segments_a_clean_flight_in_order():
    runs = phase_runs(segment_flight(synthetic_flight()))
    assert runs == ["ground", "climb", "cruise", "descent", "ground"]
    assert check_phase_order(runs) == []


def test_motors_off_counts_as_ground_despite_altitude_drift():
    runs = phase_runs(segment_flight(synthetic_flight(drift_after_landing_m=4.0)))
    assert runs[-1] == "ground"


def test_short_fragment_is_merged_into_previous_phase():
    labels = np.array(["cruise"] * 20 + ["hover"] * 3 + ["cruise"] * 20)
    time = np.arange(len(labels)) * 0.1          # 3 hover readings = 0.3 s < 1 s
    assert set(merge_short_runs(labels, time, 1.0)) == {"cruise"}


def test_order_check_flags_truncated_landing():
    assert "does not end on ground" in check_phase_order(["ground", "climb", "cruise", "descent"])
