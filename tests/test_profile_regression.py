"""Regression checks: the packaged code must reproduce the profiling results
reported in the project plan (docs/EnduroSense_Project_Plan.docx)."""
import pytest

pytestmark = pytest.mark.data


def test_flight_counts(flight_summary):
    assert len(flight_summary) == 209
    assert flight_summary["route"].str.startswith("R").sum() == 196


def test_design_grid_is_complete(flight_summary):
    core = flight_summary[
        flight_summary["route"].str.startswith("R")
        & flight_summary["payload"].isin([0, 250, 500])
        & flight_summary["altitude"].isin(["25", "50", "75", "100"])
    ]
    counts = core.groupby(["speed", "payload", "altitude"]).size()
    assert len(counts) == 60
    assert counts.min() >= 3 and counts.max() <= 5


def test_battery_chains(flight_summary):
    # Phase 0's quick chain profile (first/last-10-reading voltages, symmetric 0.3 V),
    # after the 2026-10-02 fix to chronological ordering and start-time corrections.
    # The chains used for modelling are Phase 1's (motors-off rest voltages); see test_prepare.py.
    from endurosense.data.clean import chronological
    s = chronological(flight_summary)
    chains = s.groupby("battery_chain").agg(
        n=("flight", "size"), v0=("v_start", "first"), v1=("v_end", "last"))
    assert len(chains) == 92
    assert (chains["n"] >= 2).sum() == 64
    assert ((chains["v0"] >= 25.0) & (chains["v1"] <= 22.6)).sum() == 28


def test_energy_and_wind_ranges(flight_summary):
    cruise = flight_summary[flight_summary["route"].str.startswith("R")]
    assert cruise["energy_wh"].mean() == pytest.approx(21.23, abs=0.01)
    assert cruise["ambient_wind"].min() == pytest.approx(1.29, abs=0.01)
    assert cruise["ambient_wind"].max() == pytest.approx(6.71, abs=0.01)
