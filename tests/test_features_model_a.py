"""Model A labels and features: label maths on synthetic chains, and causality
(no feature may depend on readings after its own timestamp)."""
import numpy as np
import pandas as pd
import pytest

from endurosense.features import model_a as A

FLAT = A.DischargeCurve(a=0.04, b=0.0, clip=(0.0, 1.0))          # constant 0.04 V per Wh


def test_curve_energy_between_constant_slope():
    assert FLAT.energy_between(23.0, 22.6) == pytest.approx(0.4 / 0.04)
    assert FLAT.energy_between(22.6, 23.0) == pytest.approx(-10.0)


def test_curve_puts_more_energy_where_it_is_flatter():
    curve = A.DischargeCurve(a=-0.3484, b=0.01669, clip=(0.025, 0.055))
    # the same 0.4 V holds more energy near the reserve than higher up
    assert curve.energy_between(23.0, 22.6) > curve.energy_between(24.0, 23.6)


def test_measured_crossing_interpolates_between_rest_points():
    e = np.array([0.0, 20.0, 40.0])
    v = np.array([23.4, 22.8, 22.2])                  # reserve 22.6 crossed in 2nd segment
    r = A.reserve_energy(e, v, np.zeros(3, bool), 22.6, 0.3, FLAT)
    assert r["source"] == "measured"
    assert r["e_res_wh"] == pytest.approx(20.0 + 20.0 * (0.2 / 0.6))


def test_extrapolates_only_within_margin():
    e, est = np.array([0.0, 20.0]), np.zeros(2, bool)
    r = A.reserve_energy(e, np.array([23.6, 22.8]), est, 22.6, 0.3, FLAT)
    assert r["source"] == "extrapolated" and r["e_res_wh"] == pytest.approx(20.0 + 0.2 / 0.04)
    r = A.reserve_energy(e, np.array([24.6, 23.5]), est, 22.6, 0.3, FLAT)
    assert r["source"] == "none" and np.isnan(r["e_res_wh"])


def test_running_rest_voltage_is_causal_and_freezes_at_takeoff():
    v = np.array([25.0, 25.2, 25.4, 23.0, 22.0])
    i = np.array([0.0, 0.0, 0.0, 20.0, 25.0])
    out = A.running_rest_voltage(v, i, motors_on_a=2.0, n=10)
    np.testing.assert_allclose(out, [25.0, 25.1, 25.2, 25.2, 25.2])
    assert np.isnan(A.running_rest_voltage(v, np.full(5, 20.0), 2.0)).all()


def _synthetic_flight(n=600, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 0.125
    return pd.DataFrame({"time": t, "battery_voltage": 24 - 0.002 * t + rng.normal(0, 0.05, n),
                         "battery_current": 20 + rng.normal(0, 3, n)})


def test_rolling_features_never_look_ahead():
    f = _synthetic_flight()
    base = A._rolling_flight_features(f)
    cut = 300                                         # perturb everything after row 300
    g = f.copy()
    g.loc[cut + 1:, ["battery_voltage", "battery_current"]] *= 1.5
    pert = A._rolling_flight_features(g)
    pd.testing.assert_frame_equal(base.iloc[:cut + 1], pert.iloc[:cut + 1])
    assert not base.iloc[cut + 1:].equals(pert.iloc[cut + 1:])   # and the perturbation does matter


def test_sequence_windows_use_only_completed_bins():
    series = pd.DataFrame({"flight": 1, "bin": np.arange(10), **{c: np.arange(10.0) for c in A.SEQ_CHANNELS}})
    rows = pd.DataFrame({"flight": [1], "time": [2.3]})        # 2 Hz: bin of t=2.3 s is 4 (incomplete)
    X, mask = A.sequence_windows(series, rows, seq_len=6)
    assert X[0, -1, 0] == 3.0                                  # last completed bin
    assert mask[0].tolist() == [0, 0, 1, 1, 1, 1]              # bins 0..3, older slots padded


@pytest.mark.data
def test_model_a_features_are_causal_on_real_data():
    """Perturb all readings after t* in one real flight; no earlier row may change."""
    from endurosense.data.load import load_processed
    from endurosense.data.split import DEV, select

    samples, flights = load_processed("samples"), load_processed("flights")
    dev_ids = select(flights, DEV)["flight"]
    curve, rec = A.fit_discharge_curve(flights, dev_ids), A.dev_recovery_v(flights, dev_ids)
    lab = A.chain_labels(flights, curve)
    chain = lab.loc[lab["label_source"] == "measured", "battery_chain"].iloc[0]
    s = samples[samples["battery_chain"] == chain]
    fid = s["flight"].iloc[0]
    t_cut = s.loc[s["flight"] == fid, "time"].median()

    base = A.build_model_a(s, flights, lab, rec)
    pert_s = s.copy()
    late = (pert_s["flight"] == fid) & (pert_s["time"] > t_cut)
    pert_s.loc[late, ["battery_voltage", "battery_current"]] *= 1.3
    pert = A.build_model_a(pert_s, flights, lab, rec)

    early = (base["flight"] == fid) & (base["time"] <= t_cut)
    pd.testing.assert_frame_equal(base.loc[early, A.FEATURES], pert.loc[early, A.FEATURES])


@pytest.mark.data
def test_labels_fall_one_for_one_with_energy_drawn():
    from endurosense.config import data_path
    df = pd.read_parquet(data_path("features") / "model_a.parquet")
    # within a chain, remaining energy + energy drawn is constant (= energy at the reserve)
    total = (df["remaining_wh"] + df["e_chain_wh"]).groupby(df["battery_chain"]).agg(["min", "max"])
    assert ((total["max"] - total["min"]) < 1e-9).all()
    assert df["label_source"].isin(["measured", "extrapolated"]).all()
