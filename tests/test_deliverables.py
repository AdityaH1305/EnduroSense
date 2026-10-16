"""The report and slide builders: they run, and what they say matches the result files."""
import importlib.util
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

from endurosense.config import ROOT, data_path

sys.path.insert(0, str(ROOT / "app"))
import logic as L  # noqa: E402  (the facts the deliverables are built from)

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
    ext = L.extension_results("final") is not None                                  # the extensions section adds two tables and a figure
    assert len(doc.tables) == 5 + 2 * ext and len(doc.inline_shapes) == 4 + ext
    fleet = pd.read_csv(FINAL / "fleet_simulation.csv").set_index("policy")["unsafe_per_100_missions"]
    cells = [row.cells[1].text for row in doc.tables[4].rows[1:]]                  # unsafe missions per 100: one decimal above 1, two below
    assert cells[0] == f"{fleet['P1 minutes left (brief)']:.1f}" and cells[3] == f"{fleet['P3 EnduroSense (tau = 0.95)']:.2f}"
    facts = L.headline_facts()
    assert [row.cells[1].text for row in doc.tables[0].rows[1:]] == facts["verdicts"]
    assert f"{facts['number_word'](facts['verification_passes']).capitalize()} verification passes" in text
    assert f"{facts['a_gru_cv']:.2f} Wh" in text and "judged once" not in text
    assert "**" not in text and "`" not in text                                   # no stray markup


def test_slides_specification_is_complete_and_within_the_slide(tmp_path, monkeypatch):
    slides = load("build_slides.py")
    monkeypatch.setattr(slides, "FIG", tmp_path)                                  # charts go to a temporary folder
    deck = slides.deck()
    assert len(deck) == 13 + (L.extension_results("final") is not None) and deck[0]["title"] == "" and all(s["title"] for s in deck[1:]) and all(s["notes"] for s in deck)
    for s in deck:
        for e in s["elements"]:
            assert e["x"] >= 0 and e["y"] >= 0 and e["x"] + e["w"] <= slides.W_IN + 0.01, (s["title"], e.get("text", e.get("path")))
            assert e["y"] + e["h"] <= slides.H_IN + 0.01, (s["title"], e.get("text", e.get("path")))
            if e["type"] == "text":
                assert "**" not in e["text"]
                for start, length in e["bold_ranges"]:                             # bold spans lie inside the text
                    assert 1 <= start and start + length - 1 <= len(e["text"])
            if e["type"] == "image":
                drawn = tmp_path / e["path"].replace("\\", "/").split("/")[-1]
                assert drawn.exists() or (Path(e["path"]).is_relative_to(data_path("results")) and Path(e["path"]).exists())   # a chart drawn here, or a saved result figure
    everything = json.dumps(deck)
    pts = pd.read_csv(FINAL / "operating_points.csv")
    p3 = pts[(pts["pairs"] == "all pairs") & (pts["margins"] == "tuned on development data") & (pts["policy"] == "P3 EnduroSense, tau = 0.95")]
    assert f"{100 * p3['unsafe_approval_rate'].iloc[0]:.2f}%" in everything
    boxes = [e["text"] for e in deck[7]["elements"] if e["type"] == "box"]
    assert boxes == ["GO: EnduroSense", "NO-GO: EnduroSense", "GO: EnduroSense"]   # the middle mission is the one refused
    lead = next(e for s in deck for e in s["elements"] if e["type"] == "text" and e["text"].startswith("The brief:"))
    bold = [lead["text"][a - 1:a - 1 + n] for a, n in lead["bold_ranges"]]          # positions are 1-based, as PowerPoint counts
    assert bold == ["The brief:", "The trouble with minutes:", "A single number hides how sure it is."]
    assert re.search(r"1 in \d+ missions", everything) and "verification passes" in everything


def test_project_overview_builds_and_matches_the_results(tmp_path, monkeypatch):
    from docx import Document
    overview = load("build_overview.py")
    out = tmp_path / "overview.docx"
    overview.build(out)
    doc = Document(str(out))
    text = "\n".join(p.text for p in doc.paragraphs) + "\n" + "\n".join(c.text for t in doc.tables for row in t.rows for c in row.cells)
    facts = L.headline_facts()
    for expected in (f"{100 * facts['unsafe_p1']:.1f}%", f"{100 * facts['unsafe_p3']:.2f}%", f"{facts['a_main']:.2f} Wh", f"{facts['b_unseen']:.1f}%",
                     "Decisions that need your approval", "What did not work as hoped", "Glossary"):
        assert expected in text, expected
    criteria = next(t for t in doc.tables if t.rows[0].cells[0].text.startswith("Criterion"))
    assert [row.cells[1].text for row in criteria.rows[1:]] == facts["verdicts"]            # the verdicts are the evaluation's
    decisions = doc.tables[0]
    assert len(decisions.rows) == 8 and all(row.cells[1].text and row.cells[2].text for row in decisions.rows[1:])   # every decision has an answer and an alternative
    assert "**" not in text and "`" not in text and len(doc.inline_shapes) == 3


def test_study_guide_builds_with_its_diagrams_and_worked_numbers(tmp_path):
    from docx import Document
    guide, figs = load("build_study_guide.py"), load("study_figures.py")
    out = tmp_path / "guide.docx"
    guide.build_docx(out)
    doc = Document(str(out))
    source = guide.SOURCE.read_text(encoding="utf-8")
    names = re.findall(r"^!\[.+\]\((\w+)\)$", source, flags=re.M)
    assert len(names) == len(set(names)) == 9 and len(doc.inline_shapes) == 9     # every diagram is placed once
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "![" not in text and "**" not in text                                   # no stray markup
    m = figs.example_mission()                                                     # the worked numbers are the models' own
    assert f"({m['total']:.1f} Wh)" in source and f"{m['low']:.1f} to {m['high']:.1f} Wh" in source and f"= {m['legs'][1]:.1f} Wh" in source
    (mu, sd), q = figs.combined(), figs.multipliers()
    assert f"= {mu + sd * q[0.05]:.1f} Wh" in source and f"= {mu + sd * q[0.95]:.1f} Wh" in source
    assert f"{-q[0.05]:.2f} spreads below" in source and f"{q[0.95]:.2f} spreads above" in source
    assert f"E_reserve = {figs.label_example()['e_reserve']:.0f} Wh" in source


def test_extension_verdicts_follow_from_their_tables_and_the_run_is_logged():
    ext = L.extension_results("final")
    if ext is None:
        pytest.skip("run scripts/10_extensions.py --final first")
    spec = importlib.util.spec_from_file_location("ext10", ROOT / "scripts" / "10_extensions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    again = mod.criteria(ext["planner"], ext["computing"])
    assert again["met"].tolist() == ext["criteria"]["met"].tolist() and again["evidence"].tolist() == ext["criteria"]["evidence"].tolist()
    log = json.loads((FINAL / "test_access_log.json").read_text())
    assert any(e["fingerprint"] == ext["summary"]["code_fingerprint"] and e["reason"] for e in log)   # the opening was logged, with a reason
    assert ext["summary"]["data"].startswith("unseen")
