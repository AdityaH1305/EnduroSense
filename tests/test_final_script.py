"""The final-evaluation script and the pipeline runner: helpers, criteria and the test-set gate."""
import importlib.util
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from endurosense import whatif as W
from endurosense.config import ROOT
from endurosense.data.split import DEV, TEST


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def final():
    return load_script("09_final_test.py")


LEVELS = np.array([0.05, 0.25, 0.5, 0.75, 0.95])


def test_chain_interval_resamples_whole_batteries(final):
    # one long battery always covered, one short battery never covered: 100 readings vs 10
    values, chains = np.r_[np.ones(100), np.zeros(10)], np.r_[np.zeros(100), np.ones(10)]
    lo, hi = final.chain_ci(values, chains)
    assert lo == 0.0 and hi == 1.0                                     # a resample can contain only one of them
    assert final.chain_ci(np.ones(50), np.arange(50) // 10) == (1.0, 1.0)
    # batteries of unequal length: within a resample the readings are pooled, not averaged per battery
    sizes, cover = np.array([100, 100, 10, 10, 50, 50]), np.array([1.0, 1.0, 0.0, 0.0, 0.5, 1.0])
    chains = np.repeat(np.arange(6), sizes)
    values = np.concatenate([np.r_[np.ones(int(c * n)), np.zeros(n - int(c * n))] for n, c in zip(sizes, cover)])
    idx = np.random.default_rng(42).integers(0, 6, (2000, 6))
    pooled = np.percentile((sizes * cover)[idx].sum(1) / sizes[idx].sum(1), [2.5, 97.5])
    per_battery = np.percentile(cover[idx].mean(1), [2.5, 97.5])
    assert final.chain_ci(values, chains) == pytest.approx(tuple(pooled)) and not np.allclose(pooled, per_battery)


def test_range_summary_counts_both_kinds_of_miss(final):
    y = np.array([0.0, 0.5, 5.0, 20.0])
    qv = np.tile([1.0, 4.0, 6.0, 8.0, 12.0], (4, 1))
    r = final.range_summary("m", y, qv, LEVELS, chains=[1, 2, 3, 4])
    assert r["rows"] == 4 and r["chains"] == 4 and r["cov90"] == 0.25 and r["width90"] == 11.0
    assert r["truth_below_90_range"] == 0.5 and r["truth_above_90_range"] == 0.25
    assert r["mae_median"] == pytest.approx(np.mean(np.abs(6.0 - y))) and r["cov90_ci_low"] <= r["cov90"] <= r["cov90_ci_high"]


def criteria_inputs(main=2.0, lookup=3.0, counting=4.0, b_all=3.0, b_unseen=4.0, cov_a=0.90, cov_b=0.90,
                    auc=(0.97, 0.99, 0.99), unsafe=(0.10, 0.03, 0.005)):
    a_table = pd.DataFrame({"model": ["GRU ensemble, calibrated (main)", "Voltage lookup", "Energy counting (BMS)"],
                            "test_mae_supported": [main, lookup, counting]})
    rng = lambda c: pd.DataFrame([{"cov90": c, "cov90_ci_low": c - 0.08, "cov90_ci_high": min(c + 0.08, 1.0)}])
    b_table = pd.DataFrame({"model": "Physics-first", "flights_group": ["all test flights", "unseen routes"], "mape_pct": [b_all, b_unseen]})
    summary = pd.DataFrame({"pairs": "all pairs", "policy": [W.POLICIES[c] for c in ("P1", "P2", "P3")], "auc": list(auc)})
    points = pd.DataFrame({"pairs": "all pairs", "margins": "tuned on development data",
                           "policy": ["P1 minutes left, as in the brief (ratio >= 1)", "P2 energy point estimates (margin >= 0 Wh)",
                                      "P3 EnduroSense, tau = 0.95"], "unsafe_approval_rate": list(unsafe)})
    return a_table, rng(cov_a), b_table, rng(cov_b), summary, points, 0.95


def test_success_criteria_follow_the_protocol(final):
    met = lambda **kw: final.criteria(*criteria_inputs(**kw))["met"].tolist()
    assert met() == [True, True, True, True]
    assert met(main=3.5) == [False, True, True, True]                  # worse than the voltage lookup (still better than counting)
    assert met(counting=1.5) == [False, True, True, True]              # must beat *both* baselines
    assert met(b_unseen=5.4) == [True, False, True, True]              # the cut-off is 5%, not "about 5%"
    assert met(b_all=5.4) == [True, False, True, True]
    assert met(cov_a=0.87) == [True, True, False, True] and met(cov_b=0.93) == [True, True, False, True]
    assert met(cov_a=0.88, cov_b=0.92) == [True, True, True, True]     # the band includes its ends
    assert met(auc=(0.97, 0.995, 0.99)) == [True, True, True, False]   # ranking clearly below point estimates
    assert met(auc=(0.97, 0.991, 0.99)) == [True, True, True, True]    # a difference under 0.002 counts as equal
    assert met(unsafe=(0.10, 0.004, 0.005)) == [True, True, True, False]   # must approve fewer failing missions than *both*
    assert met(unsafe=(0.004, 0.03, 0.005)) == [True, True, True, False]
    text = final.criteria(*criteria_inputs(cov_a=0.94, cov_b=0.80)).set_index("criterion")["evidence"]
    assert "90% inside both intervals: no" in text["3. 90% ranges cover 88-92%"]          # 0.80 + 0.08 < 0.90
    assert "90% inside both intervals: yes" in final.criteria(*criteria_inputs(cov_a=0.94, cov_b=0.84)).iloc[2]["evidence"]


def test_only_the_final_flag_reaches_the_test_set(final, monkeypatch):
    calls = []
    monkeypatch.setattr(final, "select", lambda df, role, final=False: calls.append((role, final)) or df)
    df = pd.DataFrame({"flight": [1, 2]})
    final.evaluation_part(df, rehearsal=False)
    assert calls == [(TEST, True)]


@pytest.mark.data
def test_rehearsal_uses_development_rows_only(final):
    from endurosense.data.load import load_processed
    from endurosense.data.split import flight_roles, load_split
    flights = load_processed("flights")[["flight"]]
    part = final.evaluation_part(flights, rehearsal=True)
    roles, fold = flight_roles(), load_split()["dev_fold"]
    assert len(part) > 0 and (part["flight"].map(roles) == DEV).all()
    assert all(fold[str(f)] == 0 for f in part["flight"])


def test_the_final_script_refuses_to_run_without_a_mode():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "09_final_test.py")], capture_output=True, text=True)
    assert r.returncode != 0 and "--final" in r.stderr and "--rehearsal" in r.stderr


def test_pipeline_runner_lists_every_numbered_script_in_order():
    runner = load_script("run_all.py")
    steps = runner.STEPS + [runner.FINAL]
    assert [s[0] for s in steps] == list(range(1, 10))
    numbered = sorted(p.name for p in (ROOT / "scripts").glob("0*.py"))
    assert [s[1] for s in steps] == numbered                           # nothing missing, nothing extra, and in order
    assert runner.FINAL[1] == "09_final_test.py" and runner.FINAL not in runner.STEPS   # the test set only with --final


def run_pipeline(monkeypatch, tmp_path, argv, exit_codes=()):
    """Run the pipeline runner with the steps replaced by a recorder."""
    runner = load_script("run_all.py")
    (tmp_path / "results").mkdir(exist_ok=True)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_all.py", *argv])
    calls, codes = [], list(exit_codes)

    def fake_run(cmd, cwd=None):
        calls.append([str(c).replace("\\", "/").split("/")[-1] for c in cmd[2:]])     # script name and its arguments
        return type("R", (), {"returncode": codes.pop(0) if codes else 0})()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as stop:
        runner.main()
    return calls, stop.value.code


def test_pipeline_runner_only_opens_the_test_set_when_asked(monkeypatch, tmp_path):
    calls, code = run_pipeline(monkeypatch, tmp_path, [])
    assert [c[0] for c in calls] == [f"0{i}_" + n for i, n in zip(range(1, 9), (
        "profile_data.py", "prepare_data.py", "make_split.py", "build_features.py", "model_a.py", "model_b.py", "uncertainty.py", "decisions.py"))]
    assert all(len(c) == 1 for c in calls) and code == 0                    # no step 09 and no --final anywhere
    calls, _ = run_pipeline(monkeypatch, tmp_path, ["--from", "8", "--final"])
    assert calls == [["08_decisions.py"], ["09_final_test.py", "--final"]]
    calls, _ = run_pipeline(monkeypatch, tmp_path, ["--from", "7"])
    assert calls == [["07_uncertainty.py"], ["08_decisions.py"]]


def test_pipeline_runner_stops_at_the_first_failure(monkeypatch, tmp_path):
    import json
    calls, code = run_pipeline(monkeypatch, tmp_path, ["--final"], exit_codes=[0, 3])
    assert [c[0] for c in calls] == ["01_profile_data.py", "02_prepare_data.py"] and code == 1    # nothing after the failed step
    log = json.loads((tmp_path / "results" / "run_all_log.json").read_text())
    assert [s["exit_code"] for s in log["steps"]] == [0, 3]

