"""Conformal quantiles, predictive distributions, coverage metrics and the calibrated models."""
import numpy as np
import pandas as pd
import pytest
import torch

from endurosense.uncertainty import battery as UA
from endurosense.uncertainty import metrics as UM
from endurosense.uncertainty import mission as UB
from endurosense.uncertainty.quantiles import (PredictiveDistribution, adjusted_level, conformal_quantiles,
                                               group_weights, weighted_quantile)

LEVELS = np.array([0.05, 0.25, 0.5, 0.75, 0.95])


def test_group_weights_give_every_group_the_same_total():
    g = np.array(["a"] * 100 + ["b"] * 5 + ["c"])
    w = group_weights(g)
    assert pd.Series(w).groupby(g).sum().round(12).tolist() == [1.0, 1.0, 1.0]


def test_weighted_quantile_matches_plain_quantile_for_equal_weights():
    v = np.arange(1, 101, dtype=float)
    assert weighted_quantile(v, 0.5, np.ones(100)) == 50
    assert weighted_quantile(v, 0.9, np.ones(100)) == 90
    # a heavy group must not dominate: 1000 rows of one group vs 1 row each of 9 others
    v2 = np.r_[np.zeros(1000), np.arange(1, 10)]
    g2 = np.r_[np.zeros(1000), np.arange(1, 10)]
    assert weighted_quantile(v2, 0.5, group_weights(g2)) == 4          # the median *group*, not 0


def test_small_sample_corrections_push_levels_outwards():
    assert adjusted_level(0.95, 40, "none") == 0.95
    for mode in ("smooth", "ceil"):                    # never inwards; which is wider depends on G and q
        assert 0.95 <= adjusted_level(0.95, 40, mode) <= 1.0
        assert 0.0 <= adjusted_level(0.05, 40, mode) <= 0.05
        assert adjusted_level(0.5, 40, mode) >= 0.5
    assert adjusted_level(0.95, 40, "smooth") == pytest.approx(0.95 * 41 / 40)
    assert adjusted_level(0.95, 40, "ceil") == pytest.approx(39 / 40)
    with pytest.raises(ValueError):
        adjusted_level(0.9, 40, "bogus")


def test_predictive_distribution_is_consistent():
    d = PredictiveDistribution(LEVELS, [10.0, 14.0, 16.0, 18.0, 22.0])
    assert d.median == 16.0 and d.interval(0.9) == (10.0, 22.0)
    for q in (0.01, 0.05, 0.3, 0.5, 0.8, 0.95, 0.99):                 # cdf is the inverse of quantile, tails included
        assert d.cdf(d.quantile(q)) == pytest.approx(q, abs=1e-9)
    assert d.quantile(0.01) < 10.0 < 22.0 < d.quantile(0.99)           # tails continue past the stored levels
    s = d.sample(20000, np.random.default_rng(0))
    assert np.median(s) == pytest.approx(16.0, abs=0.2)
    assert PredictiveDistribution(LEVELS, [1, 3, 2, 4, 5]).values.tolist() == [1, 3, 3, 4, 5]   # crossings repaired


def test_conformal_ranges_cover_new_groups_at_the_nominal_rate():
    """Errors shared within a group (like a per-battery offset): calibrate on some groups,
    check coverage on others."""
    rng = np.random.default_rng(0)

    def make(n_groups):
        g = np.repeat(np.arange(n_groups), 50)
        return g, rng.normal(0, 3, n_groups)[g] + rng.normal(0, 1, len(g))      # offset per group + noise

    g_cal, e_cal = make(400)
    g_new, e_new = make(400)
    lo, hi = conformal_quantiles(e_cal, g_cal, [0.05, 0.95], correction="none")
    assert np.mean((e_new >= lo) & (e_new <= hi)) == pytest.approx(0.90, abs=0.03)


def test_interval_table_and_pinball_on_known_values():
    y = np.array([0.0, 5.0, 10.0, 20.0])
    qv = np.tile([1.0, 4.0, 6.0, 8.0, 12.0], (4, 1))                  # same quantiles for every row
    t = UM.interval_table(y, qv, LEVELS, groups=[1, 2, 3, 4], coverages=(0.9,)).iloc[0]
    assert t["coverage"] == 0.5 and t["mean_width"] == 11.0           # 5 and 10 inside [1, 12]
    assert t["too_high"] == 0.25 and t["too_low"] == 0.25
    assert UM.pinball_loss([6.0], [[6.0] * 5], LEVELS) == 0.0
    by = UM.coverage_by(y, qv, LEVELS, by=["a", "a", "b", "b"])
    assert by.set_index("by")["coverage"].to_dict() == {"a": 0.5, "b": 0.5}


def test_battery_quantiles_scale_with_sigma_and_abstain_without_voltage():
    q = UA.quantile_values([50.0, 50.0], [1.0, 3.0], [-2.0, 0.0, 2.0])
    assert q.tolist() == [[48.0, 50.0, 52.0], [44.0, 50.0, 56.0]]
    df = pd.DataFrame({"v_rest_chain_start": [25.0, np.nan, 25.0], "v_rest_flight_start": [24.0, 24.0, np.nan]})
    assert UA.supported(df).tolist() == [True, False, False]


def test_mission_replay_keeps_errors_together_and_scales_the_leg_error():
    parts = {"climb": 3.0, "descent": 4.0, "hover": 1.0, "ground": 1.0, "legs": 10.0}
    tuples = pd.DataFrame({"climb": [0.5, -0.5], "descent": [-0.5, 0.5], "hover": [0.0, 0.0], "ground": [0.0, 0.0],
                           "legs_rel": [0.10, -0.10], "total_rel": [0.10, -0.10], "battery_chain": [1, 2]})
    np.testing.assert_allclose(UB.replay(parts, tuples, "parts"), [20.0, 18.0])   # +-0.5 cancel; legs +-10% of 10
    np.testing.assert_allclose(UB.replay(parts, tuples, "total"), [20.9, 17.1])   # +-10% of 19
    longer = {**parts, "legs": 30.0}                                   # three times the cruise energy
    spread = lambda p: np.ptp(UB.replay(p, tuples, "parts"))
    assert spread(longer) == pytest.approx(3 * spread(parts))


def test_spread_head_cannot_change_the_mean_path():
    from endurosense.models.probabilistic import _ProbNet
    net = _ProbNet("gru", n_channels=6, n_static=4, hidden=8, layers=1, dropout=0.0)
    mean, logvar = net(torch.randn(5, 12, 6), torch.randn(5, 4))
    logvar.sum().backward()                                            # a loss that depends on the spread only
    assert all(p.grad is None or torch.all(p.grad == 0) for p in net.rnn.parameters())
    assert all(p.grad is None or torch.all(p.grad == 0) for p in net.mean_head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.spread_head.parameters())


@pytest.mark.data
def test_saved_mission_model_gives_wider_ranges_for_longer_missions():
    import pickle
    from endurosense.config import data_path
    from endurosense.mission import MissionSpec
    path = data_path("models") / "model_b" / "model_b_calibrated.pkl"
    if not path.exists():
        pytest.skip("run scripts/07_uncertainty.py first")
    with open(path, "rb") as fh:
        m = pickle.load(fh)
    short, long = m.distribution(MissionSpec.delivery(300, 500, 8, 50)), m.distribution(MissionSpec.delivery(600, 500, 8, 50))
    width = lambda d: np.subtract(*d.interval(0.9)[::-1])
    assert long.median > short.median and width(long) > width(short)
    assert short.cdf(short.median) == pytest.approx(0.5, abs=0.02)
    assert m.warnings(MissionSpec.delivery(300, 500, 20, 50)) != []


@pytest.mark.data
def test_saved_battery_model_loads_and_reproduces_its_calibrated_ranges():
    from endurosense.config import data_path, load_config
    from endurosense.data.split import DEV, select
    from endurosense.features.model_a import sequence_windows
    from endurosense.models.sequence import WindowStore
    path = data_path("models") / "model_a" / "model_a_calibrated.pt"
    if not path.exists():
        pytest.skip("run scripts/07_uncertainty.py first")
    df = pd.read_parquet(data_path("features") / "model_a.parquet")
    dev = select(df, DEV)
    rows = dev[UA.supported(dev)].iloc[::400]
    series = pd.read_parquet(data_path("features") / "model_a_series.parquet")
    X, mask = sequence_windows(series, rows)
    full = np.zeros((len(df),) + X.shape[1:], np.float32); fm = np.zeros((len(df), X.shape[1]), np.float32)
    full[rows.index], fm[rows.index] = X, mask
    model = UA.load_calibrated(path, WindowStore(full, fm))
    q = model.quantiles(rows)
    assert q.shape == (len(rows), len(load_config()["uncertainty"]["quantiles"]))
    assert np.isfinite(q).all() and (np.diff(q, axis=1) >= -1e-9).all()        # quantiles never cross
    med = q[:, list(model.levels).index(0.5)]
    assert np.mean(np.abs(med - rows["remaining_wh"])) < 3.0                   # in-sample, so better than CV
    assert model.distributions(rows.iloc[:2])[0].interval(0.9)[0] < med[0]
