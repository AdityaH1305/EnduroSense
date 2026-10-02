"""Model A predictors, baselines and evaluation helpers."""
import numpy as np
import pandas as pd
import pytest
import torch

from endurosense.evaluate import metrics
from endurosense.models import tabular as T
from endurosense.models.sequence import _Net


def test_metrics_on_known_values():
    m = metrics([0.0, 2.0, 4.0], [1.0, 2.0, 3.0])
    assert m["mae"] == pytest.approx(2 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert m["bias"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1 - 2 / 8)


def test_paired_comparison_detects_a_better_model_and_a_tie():
    from endurosense.evaluate import paired_comparison
    rng = np.random.default_rng(0)
    y = rng.normal(size=600)
    groups = np.repeat(np.arange(30), 20)
    good, bad = y + rng.normal(0, 0.5, 600), y + rng.normal(0, 2.0, 600)
    c = paired_comparison(y, good, bad, groups)
    assert c["mae_diff"] < 0 and c["ci_high"] < 0          # clearly better: CI excludes zero
    tie = paired_comparison(y, good, good, groups)
    assert tie["mae_diff"] == 0 and tie["ci_low"] == 0 == tie["ci_high"]


def test_single_threaded_sets_inner_estimators():
    from endurosense.evaluate import single_threaded
    m = single_threaded(T.random_forest(n_estimators=5))
    assert m.pipe[-1].n_jobs == 1
    phys = type("P", (), {})()
    phys.model = T.xgboost(n_estimators=5)
    single_threaded(phys)
    assert phys.model.pipe[-1].n_jobs == 1


def test_search_configs_are_distinct_valid_and_reproducible():
    a = T.sample_configs("xgboost", 30, seed=42)
    assert a == T.sample_configs("xgboost", 30, seed=42)
    assert len({tuple(sorted(c.items())) for c in a}) == 30
    for c in a:
        for k, v in c.items():
            assert v in T.SEARCH_SPACES["xgboost"][k]
    assert len(T.sample_configs("linear", 0, seed=0)) == len(T.SEARCH_SPACES["linear"]["alpha"])


def test_sequence_net_output_shape():
    for kind in ("lstm", "gru"):
        net = _Net(kind, n_channels=6, n_static=25, hidden=16, layers=2, dropout=0.1)
        out = net(torch.zeros(7, 120, 6), torch.zeros(7, 25))
        assert out.shape == (7,)


def test_tabular_model_handles_missing_values():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
    df.loc[::7, "b"] = np.nan
    df["y"] = 2 * df["a"] - df["c"]
    m = T.TabularModel("lin", T.Ridge(alpha=0.01), features=["a", "b", "c"], scale=True).fit(df, "y")
    assert np.abs(m.predict(df) - df["y"]).max() < 0.05


@pytest.fixture(scope="module")
def model_a_data():
    from endurosense.config import data_path
    from endurosense.data.load import load_processed
    path = data_path("features") / "model_a.parquet"
    if not path.exists():
        pytest.skip("run scripts/04_build_features.py first")
    return pd.read_parquet(path), load_processed("flights")


@pytest.mark.data
def test_baselines_follow_their_definitions(model_a_data):
    from endurosense.data.split import DEV, select
    from endurosense.models.baselines import EnergyCounting, VoltageLookup
    df, flights = model_a_data
    dev = select(df, DEV)
    ec = EnergyCounting(flights).fit(dev, "remaining_wh")
    row = dev.dropna(subset=["v_rest_chain_start", "v_rest_flight_start"]).iloc[[100]]
    expect = ec.curve.energy_between(row["v_rest_chain_start"].iloc[0], 22.6) - row["e_chain_wh"].iloc[0]
    assert ec.predict(row)[0] == pytest.approx(expect)
    vl = VoltageLookup(flights).fit(dev, "remaining_wh")
    expect = vl.curve.energy_between(row["v_rest_flight_start"].iloc[0], 22.6) - row["e_flight_wh"].iloc[0]
    assert vl.predict(row)[0] == pytest.approx(expect)


@pytest.mark.data
def test_cross_validation_predicts_every_dev_row_once_without_seeing_it(model_a_data):
    from endurosense.data.split import DEV, select
    from endurosense.evaluate import cross_validate
    df, _ = model_a_data
    dev = select(df, DEV)
    seen_chains = []

    class Spy:
        def fit(self, train, target):
            self.train_chains = set(train["battery_chain"])
            return self

        def predict(self, d):
            assert self.train_chains.isdisjoint(d["battery_chain"])     # grouped: no shared battery
            seen_chains.extend(d["battery_chain"].unique())
            return np.zeros(len(d))

    oof, fm = cross_validate(Spy, dev, "remaining_wh")
    assert oof.notna().all() and len(fm) == 5
    assert len(seen_chains) == dev["battery_chain"].nunique()
