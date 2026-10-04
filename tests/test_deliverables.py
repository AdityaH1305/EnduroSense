"""The report and slide builders: they run, and what they say matches the result files."""
import importlib.util
import json
import re

import pandas as pd
import pytest

from endurosense.config import ROOT, data_path

FINAL = data_path("results") / "final"
pytestmark = [pytest.mark.data, pytest.mark.skipif(not (FINAL / "success_criteria.csv").exists(), reason="run scripts/run_all.py --final first")]


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), ROOT / "docs" / "build" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_report_builds_and_quotes_the_final_results(tmp_path):
    from docx import Document
    report = load("build_report.py")
    out = tmp_path / "report.docx"
    report.build(out)
    doc = Document(str(out))
    text = "\n".join(p.text for p in doc.paragraphs) + "\n" + "\n".join(c.text for t in doc.tables for row in t.rows for c in row.cells)
    a = pd.read_csv(FINAL / "model_a_test_metrics.csv").set_index("model")
    pts = pd.read_csv(FINAL / "operating_points.csv")
    pts = pts[(pts["pairs"] == "all pairs") & (pts["margins"] == "tuned on development data")].set_index("policy")["unsafe_approval_rate"]
    b = pd.read_csv(FINAL / "model_b_test_metrics.csv")
    b = b[b["model"] == "Physics-first"].set_index("flights_group")["mape_pct"]
    for expected in (f"{a.loc['GRU ensemble, calibrated (main)', 'test_mae_supported']:.2f} Wh",
                     f"{a.loc['Voltage lookup', 'test_mae_supported']:.2f} Wh",
                     f"{100 * pts['P1 minutes left, as in the brief (ratio >= 1)']:.1f}%", f"{100 * pts['P3 EnduroSense, tau = 0.95']:.2f}%",
                     f"{b['unseen routes']:.1f}%", "Not met", "Not met, within sampling noise"):
        assert expected in text, expected
    crit = pd.read_csv(FINAL / "success_criteria.csv")
    met_in_report = [row.cells[1].text for row in doc.tables[0].rows[1:]]
    assert [m == "Met" for m in met_in_report] == crit["met"].tolist()          # the verdicts are the evaluation's, not the author's
    assert len(doc.tables) == 5 and len(doc.inline_shapes) == 4                    # five tables, four figures
    assert "**" not in text and "`" not in text                                   # no stray markup


def test_slides_specification_is_complete_and_within_the_slide(tmp_path, monkeypatch):
    slides = load("build_slides.py")
    monkeypatch.setattr(slides, "FIG", tmp_path)                                  # charts go to a temporary folder
    deck = slides.deck()
    assert len(deck) == 13 and deck[0]["title"] == "" and all(s["title"] for s in deck[1:]) and all(s["notes"] for s in deck)
    for s in deck:
        for e in s["elements"]:
            assert e["x"] >= 0 and e["y"] >= 0 and e["x"] + e["w"] <= slides.W_IN + 0.01, (s["title"], e.get("text", e.get("path")))
            assert e["y"] + e["h"] <= slides.H_IN + 0.01, (s["title"], e.get("text", e.get("path")))
            if e["type"] == "text":
                assert "**" not in e["text"]
                for start, length in e["bold_ranges"]:                             # bold spans lie inside the text
                    assert 1 <= start and start + length - 1 <= len(e["text"])
            if e["type"] == "image":
                assert (tmp_path / e["path"].replace("\\", "/").split("/")[-1]).exists()
    everything = json.dumps(deck)
    pts = pd.read_csv(FINAL / "operating_points.csv")
    p3 = pts[(pts["pairs"] == "all pairs") & (pts["margins"] == "tuned on development data") & (pts["policy"] == "P3 EnduroSense, tau = 0.95")]
    assert f"{100 * p3['unsafe_approval_rate'].iloc[0]:.2f}%" in everything
    assert re.search(r"NO-GO: EnduroSense", everything) and everything.count("GO: EnduroSense") == 3   # two approvals, one refusal
