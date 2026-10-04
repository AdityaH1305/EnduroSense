"""The locked split must stay leak-free, balanced and protected."""
import json

import pytest

from endurosense.data import split as sp
from endurosense.data.load import load_processed

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def flights():
    if not sp.split_path().exists():
        pytest.skip("split not created yet (run scripts/03_make_split.py)")
    return load_processed("flights")


def test_hash_verifies(flights):
    assert sp.load_split()["sha256"]


def test_no_battery_chain_on_both_sides(flights):
    sp.check_consistency(flights)          # raises if violated


def test_test_share_and_coverage(flights):
    roles = flights["flight"].map(sp.flight_roles())
    cruise = flights[flights["route"].str.startswith("R")]
    test = cruise[roles[cruise.index] == sp.TEST]
    assert 0.18 <= len(test) / len(cruise) <= 0.22
    assert test["speed"].nunique() == 5 and test["payload"].nunique() >= 3
    assert (flights.loc[flights["route"] == "R6", "flight"].map(sp.flight_roles()) == sp.TEST).all()


def test_test_rows_are_locked(flights, monkeypatch):
    monkeypatch.delenv("ENDUROSENSE_FINAL", raising=False)
    with pytest.raises(sp.TestSetLocked):
        sp.select(flights, sp.TEST)
    assert len(sp.select(flights, sp.TEST, final=True)) > 0


def test_folds_partition_the_dev_data(flights):
    dev = sp.select(flights, sp.DEV)
    seen = []
    for _, train, val in sp.dev_folds(flights):
        assert set(train["battery_chain"]).isdisjoint(val["battery_chain"])
        seen += val["flight"].tolist()
    assert sorted(seen) == sorted(dev["flight"])


def test_existing_split_is_never_overwritten(flights):
    with pytest.raises(FileExistsError):
        sp.save_split(sp.load_split())


def test_edited_split_file_is_detected(flights, tmp_path, monkeypatch):
    payload = sp.load_split()
    payload["flight_role"]["1"] = "test" if payload["flight_role"]["1"] == "dev" else "dev"
    fake = tmp_path / "split.json"
    fake.write_text(json.dumps(payload))
    monkeypatch.setattr(sp, "split_path", lambda: fake)
    with pytest.raises(ValueError, match="modified"):
        sp.load_split()


def test_every_opening_of_the_test_set_is_logged_and_changes_need_a_reason(tmp_path):
    import json
    from endurosense.access import ReasonRequired, code_fingerprint, register_opening
    log = tmp_path / "final" / "test_access_log.json"
    assert register_opening(log, fingerprint="aaa") == "aaa"
    assert register_opening(log, fingerprint="aaa") == "aaa"                # same code: a reproducibility rerun, nothing added
    entries = json.loads(log.read_text())
    assert len(entries) == 1 and entries[0]["reason"] == "first final evaluation"
    with pytest.raises(ReasonRequired):
        register_opening(log, fingerprint="bbb")                            # changed code and no explanation
    assert len(json.loads(log.read_text())) == 1
    register_opening(log, reason="fixed a plotting bug", fingerprint="bbb")
    entries = json.loads(log.read_text())
    assert [e["fingerprint"] for e in entries] == ["aaa", "bbb"] and entries[1]["reason"] == "fixed a plotting bug"
    assert code_fingerprint() == code_fingerprint() and len(code_fingerprint()) == 16

