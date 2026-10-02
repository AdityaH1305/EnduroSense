"""Data checks on the Phase 1 tables (built from the raw dataset)."""
import numpy as np
import pytest

from endurosense.data.clean import chronological
from endurosense.data.load import load_flights
from endurosense.data.prepare import build_tables

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def tables():
    return build_tables(load_flights())


def test_every_reading_kept_and_phased(tables):
    samples, flights, _, _ = tables
    assert len(samples) == 257_896
    assert samples["phase"].notna().all()
    assert set(flights["flight"]) == set(samples["flight"])


def test_phase_order_ok_except_truncated_flight(tables):
    _, flights, _, report = tables
    cruise = flights[flights["route"].str.startswith("R")]
    assert cruise["phase_order_ok"].sum() == 195
    assert report["phase_order_problems"] == [250] == report["truncated_landing"]


def test_phase_energies_add_up_to_flight_energy(tables):
    _, flights, _, _ = tables
    parts = flights[[c for c in flights if c.endswith("_energy_wh") and c != "energy_wh"]].sum(axis=1)
    np.testing.assert_allclose(parts, flights["energy_wh"], rtol=1e-9)


def test_energy_matches_phase0_profile(tables):
    _, flights, _, _ = tables
    assert flights.loc[flights["route"].str.startswith("R"), "energy_wh"].mean() == pytest.approx(21.23, abs=0.01)


def test_chain_energy_never_decreases(tables):
    samples, _, _, _ = tables
    order = chronological(samples[["flight", "date", "local_time"]].drop_duplicates())["flight"]
    rank = {f: i for i, f in enumerate(order)}
    s = samples.assign(_r=samples["flight"].map(rank)).sort_values(["battery_chain", "_r", "time"])
    steps = s.groupby("battery_chain")["cum_chain_energy_wh"].diff().dropna()
    assert (steps >= -0.01).all()        # tiny negatives only from ~0 A ground readings


def test_rest_voltage_falls_along_every_chain(tables):
    _, flights, _, _ = tables
    s = chronological(flights)
    rises = s.groupby("battery_chain")["v_rest_start"].apply(lambda v: (v.diff() > 0.05).any())
    assert not rises.any()


def test_corrected_flights_join_their_neighbours(tables):
    # with true start times, the mis-logged flights continue the right batteries
    _, flights, _, _ = tables
    chain = flights.set_index("flight")["battery_chain"]
    assert chain[108] == chain[109]
    assert chain[110] == chain[111] == chain[112]
    assert chain[149] == chain[150] == chain[151] == chain[152]
    assert chain[172] == chain[173] == chain[174]


def test_flight_81_is_linked_to_82(tables):
    # flight 81's recording ends under load; with motors-off rest voltages it links correctly
    _, flights, _, _ = tables
    chain = flights.set_index("flight")["battery_chain"]
    assert chain[81] == chain[82] == chain[83]
