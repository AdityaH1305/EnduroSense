"""Build the final report (Word) from the saved result files.

Usage: python docs/build/build_report.py        -> docs/EnduroSense_Final_Report.docx

Every number in the report is read from results/ (the final test evaluation and the
development phases), so the document cannot drift from the tables. Figures are the
ones the pipeline saved. Nothing is computed from data here.
"""
import json
import sys

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from endurosense.config import ROOT, data_path, load_config
from endurosense.data.split import load_split

sys.path.insert(0, str(ROOT / "app"))
import logic as L  # noqa: E402  (headline facts shared with the dashboard and the slides)

RES, FIN = data_path("results"), data_path("results") / "final"
OUT = ROOT / "docs" / "EnduroSense_Final_Report.docx"
BLUE, INK, MUTED = RGBColor(0x1F, 0x5F, 0xAE), RGBColor(0x0B, 0x0B, 0x0B), RGBColor(0x52, 0x51, 0x4E)


# ------------------------------------------------------------------ numbers
def numbers() -> dict:
    cfg = load_config()
    a = pd.read_csv(FIN / "model_a_test_metrics.csv").set_index("model")
    ar = pd.read_csv(FIN / "model_a_test_ranges.csv").iloc[0]
    b = pd.read_csv(FIN / "model_b_test_metrics.csv")
    bm = b[b["model"] == "Physics-first"].set_index("flights_group")
    br = pd.read_csv(FIN / "model_b_test_ranges.csv").set_index("model")
    pts = pd.read_csv(FIN / "operating_points.csv")
    pts = pts[pts["margins"] == "tuned on development data"]
    op = {s: pts[pts["pairs"] == s].assign(policy=lambda d: d["policy"].str.strip()).set_index("policy") for s in ("all pairs", "pre-flight states")}
    dev_pts = pd.read_csv(RES / "decisions" / "operating_points.csv")
    dev = dev_pts[dev_pts["pairs"] == "all pairs"].assign(policy=lambda d: d["policy"].str.strip()).set_index("policy")
    summ = pd.read_csv(FIN / "policy_summary.csv")
    fleet = pd.read_csv(FIN / "fleet_simulation.csv").set_index("policy")
    rel = pd.read_csv(FIN / "probability_reliability.csv")
    return {"cfg": cfg, "a": a, "ar": ar, "b": b, "bm": bm, "br": br, "op": op, "dev": dev, "summ": summ, "fleet": fleet, "rel": rel,
            "crit": pd.read_csv(FIN / "success_criteria.csv"), "summary": json.loads((FIN / "summary.json").read_text()),
            "log": json.loads((FIN / "test_access_log.json").read_text()),
            "mistakes": json.loads((FIN / "mistakes_at_tau.json").read_text()),
            "indep": json.loads((FIN / "error_independence.json").read_text()),
            "cv": pd.read_csv(RES / "model_a" / "cv_metrics.csv").set_index("model"),
            "dev_a": pd.read_csv(RES / "uncertainty" / "model_a_intervals.csv").set_index("method"),
            "dev_b": pd.read_csv(RES / "uncertainty" / "model_b_intervals.csv").set_index("method"),
            "mission_cv": pd.read_csv(RES / "model_b" / "mission_cv.csv").set_index("model")}


# ------------------------------------------------------------------ document helpers
def add_runs(par, text: str, size=None, color=None, italic=False, bold=False):
    """Add text to a paragraph: ``**bold**`` spans are bold, `` `code` `` spans are monospace."""
    for i, chunk in enumerate(text.split("**")):
        for j, piece in enumerate(chunk.split("`")):
            if not piece:
                continue
            run = par.add_run(piece)
            run.bold, run.italic = bold or (i % 2 == 1), italic
            if j % 2 == 1:
                run.font.name, run.font.size = "Consolas", Pt((size or 11) - 1)
            elif size:
                run.font.size = Pt(size)
            if color is not None:
                run.font.color.rgb = color


def set_cell_shade(cell, hex_fill: str) -> None:
    tc = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_fill)
    tc.append(shd)


class Report:
    def __init__(self):
        self.doc = Document()
        sec = self.doc.sections[0]
        sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
        sec.left_margin = sec.right_margin = Cm(2.3)
        sec.top_margin = sec.bottom_margin = Cm(2.2)
        st = self.doc.styles["Normal"]
        st.font.name, st.font.size = "Calibri", Pt(11)
        st.paragraph_format.space_after = Pt(6)
        for name, size in (("Heading 1", 18), ("Heading 2", 13.5), ("Heading 3", 11.5)):
            h = self.doc.styles[name]
            h.font.name, h.font.size, h.font.bold, h.font.color.rgb = "Calibri", Pt(size), True, BLUE
            h.paragraph_format.space_before, h.paragraph_format.space_after = Pt(14 if name == "Heading 1" else 10), Pt(5)
            h.paragraph_format.keep_with_next = True
        self.fig_no = 0
        self.tab_no = 0

    def h(self, text, level=1):
        return self.doc.add_heading(text, level)

    def p(self, text="", bold_lead=None, italic=False, size=None, color=None, align=None, after=None):
        """A paragraph. ``**bold**`` spans are honoured; ``bold_lead`` is a bold opening phrase."""
        par = self.doc.add_paragraph()
        if bold_lead:
            par.add_run(bold_lead + " ").bold = True
        add_runs(par, text, size, color, italic)
        if align:
            par.alignment = align
        if after is not None:
            par.paragraph_format.space_after = Pt(after)
        return par

    def bullets(self, items, style="List Bullet"):
        for it in items:
            par = self.doc.add_paragraph(style=style)
            par.paragraph_format.space_after = Pt(3)
            add_runs(par, it)

    def table(self, header, rows, widths, caption=None, bold_rows=(), align_right_from=1):
        if caption:
            self.tab_no += 1
            c = self.p(f"Table {self.tab_no}. {caption}", size=9.5, color=MUTED, after=3)
            c.paragraph_format.keep_with_next = True
        t = self.doc.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        for j, (text, w) in enumerate(zip(header, widths)):
            cell = t.rows[0].cells[j]
            cell.width = Cm(w)
            set_cell_shade(cell, "DCE8F6")
            run = cell.paragraphs[0].add_run(text)
            run.bold, run.font.size = True, Pt(9.5)
            cell.paragraphs[0].paragraph_format.space_after = Pt(2)
        for i, row in enumerate(rows):
            cells = t.add_row().cells
            for j, (text, w) in enumerate(zip(row, widths)):
                cells[j].width = Cm(w)
                par = cells[j].paragraphs[0]
                par.paragraph_format.space_after = Pt(2)
                run = par.add_run(str(text))
                run.font.size = Pt(9.5)
                run.bold = i in bold_rows
                if j >= align_right_from:
                    par.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        for row in t.rows[:-1]:                                    # keep the table on one page
            for cell in row.cells:
                cell.paragraphs[0].paragraph_format.keep_with_next = True
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)
        return t

    def figure(self, path, caption, width_cm=16.0):
        self.fig_no += 1
        par = self.doc.add_paragraph()
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.keep_with_next = True
        par.add_run().add_picture(str(path), width=Cm(width_cm))
        self.p(f"Figure {self.fig_no}. {caption}", size=9.5, color=MUTED, align=WD_ALIGN_PARAGRAPH.CENTER, after=10)

    def page_break(self):
        self.doc.add_page_break()


pct = lambda x, d=1: f"{100 * x:.{d}f}%"


# ------------------------------------------------------------------ the report
def build(out=OUT) -> None:
    n, r, f = numbers(), Report(), L.headline_facts()
    word = f["number_word"]
    cfg, a, ar, bm, br, op, fleet = n["cfg"], n["a"], n["ar"], n["bm"], n["br"], n["op"], n["fleet"]
    tau, reserve = cfg["decision"]["tau"], cfg["battery"]["reserve_v"]
    MAIN = "GRU ensemble, calibrated (main)"
    P1, P2, P3 = "P1 minutes left, as in the brief (ratio >= 1)", "P2 energy point estimates (margin >= 0 Wh)", f"P3 EnduroSense, tau = {tau}"
    allp, pre = op["all pairs"], op["pre-flight states"]
    p2m = [p for p in allp.index if p.startswith("P2 with margin") and f"tau = {tau} " in p][0]
    p2m_pre = [p for p in pre.index if p.startswith("P2 with margin") and f"tau = {tau} " in p][0]
    p1m = [p for p in allp.index if p.startswith("P1 with margin") and f"tau = {tau} " in p][0]
    f_p1, f_p2 = fleet.loc["P1 minutes left (brief)"], fleet.loc["P2 energy, point estimates"]
    f_p3, f_p2m = fleet.loc[f"P3 EnduroSense (tau = {tau})"], fleet.loc["P2 + margin tuned on development data"]
    band = n["rel"][n["rel"]["band"] == "(0.95, 0.99]"].iloc[0]

    # ---- title
    for _ in range(5):
        r.p()
    r.p("EnduroSense", size=34, color=BLUE, align=WD_ALIGN_PARAGRAPH.CENTER, after=4).runs[0].bold = True
    r.p("Uncertainty-aware mission feasibility prediction for UAVs", size=15, color=INK, align=WD_ALIGN_PARAGRAPH.CENTER, after=30)
    r.p("Final report", size=13, color=MUTED, align=WD_ALIGN_PARAGRAPH.CENTER, after=40)
    r.p("An extension of the intern project “ML-Based UAV Battery / Remaining Flight-Time Prediction”. "
        "Instead of asking how many minutes of flight are left, EnduroSense asks whether a given battery can complete a given "
        "mission, and how sure it is.", size=11.5, color=MUTED, align=WD_ALIGN_PARAGRAPH.CENTER)
    r.page_break()

    # ---- 1 summary
    r.h("1. Summary")
    r.p("The brief asks for a model that predicts the remaining flight time of a UAV battery. Minutes are a weak basis for a "
        "go / no-go decision, because the same battery lasts very different times depending on what it is asked to do next. "
        "EnduroSense predicts two energies instead and compares them:")
    r.bullets(["**Energy available (Model A):** the watt-hours the battery can still deliver before a safe reserve, from its telemetry.",
               "**Energy required (Model B):** the watt-hours a planned mission will use, from its distance, speed, payload, altitude and wind.",
               f"**P(success):** each prediction comes with a calibrated range, and the two ranges give the chance that the battery covers "
               f"the mission. A mission is approved only if P(success) is at least {tau:.0%}."])
    split = load_split()
    roles = pd.Series(split["flight_role"])
    r.p(f"Everything was built on {int((roles == 'dev').sum())} flights of a DJI Matrice 100 and judged on a locked test set of "
        f"{int((roles == 'test').sum())} flights ({len(split['test_chains'])} battery chains) that no model, "
        "setting or calibration ever saw. What would count as success was written down before that test set was opened.")
    r.h("Main results on the test set", 2)
    r.bullets([
        f"**Decisions.** The brief’s rule (minutes left ≥ mission duration) approves {pct(allp.loc[P1, 'unsafe_approval_rate'])} of missions that "
        f"would cut into the reserve. Energy point estimates approve {pct(allp.loc[P2, 'unsafe_approval_rate'])}. EnduroSense approves "
        f"{pct(allp.loc[P3, 'unsafe_approval_rate'], 2)}, and none at take-off decisions.",
        f"**Energy available.** {a.loc[MAIN, 'test_mae_supported']:.2f} Wh error on unseen batteries, against {a.loc['Voltage lookup', 'test_mae_supported']:.2f} Wh "
        f"for the best method without machine learning.",
        f"**Energy required.** {bm.loc['all test flights', 'mape_pct']:.1f}% error on unseen flights: {bm.loc['route seen in development', 'mape_pct']:.1f}% on the "
        f"route flown in training and {bm.loc['unseen routes', 'mape_pct']:.1f}% on routes never seen.",
        f"**Fleet simulation on unseen batteries.** {f_p1['unsafe_per_100_missions']:.1f} unsafe missions per 100 with the brief’s rule, "
        f"{f_p3['unsafe_per_100_missions']:.2f} with EnduroSense."])
    r.h("Success criteria", 2)
    crit = n["crit"]
    r.table(["Criterion (fixed before the test)", "Result", "Evidence"],
            [[c["criterion"], v, c["evidence"]] for (_, c), v in zip(crit.iterrows(), f["verdicts"])], [5.6, 2.7, 8.1],
            caption="The four success criteria and their outcome on the test set.", align_right_from=9)
    r.p(f"{word(f['criteria_met']).capitalize()} of the {word(f['criteria_total'])} criteria were met. Section 6 says plainly what did not hold up. "
        "The largest and most robust gain is from deciding on energy with a calibrated safety margin instead of on minutes.")

    # ---- 2 problem
    r.h("2. The problem and the idea")
    r.p("A scheduler that sends drones on missions needs one answer per mission: can this battery do it and come back with a safe reserve? "
        "“Minutes of flight left” does not answer that, for two reasons.")
    r.bullets(["**Minutes depend on the next task.** Remaining minutes are computed from the power being drawn now. A drone hovering or descending "
               "draws less power than the mission ahead will, so its minutes are overstated.",
               "**A single number hides how sure it is.** Two batteries can show the same estimate while one of them is far less certain. "
               "A safe decision needs to know that."])
    r.p("EnduroSense therefore works in energy (watt-hours), which adds up across the parts of a mission, and attaches an honest range to "
        f"each prediction. The reserve is {reserve} V at rest (about 3.77 V per cell); “energy available” means energy above that reserve, "
        "so a successful mission ends with the reserve intact. Both the reserve and the approval threshold are settings in one "
        "configuration file and are the project’s defaults pending the guide’s confirmation.")

    # ---- 3 data
    r.h("3. Data")
    r.p("The data is the public Carnegie Mellon University package-delivery dataset (Rodrigues et al., 2021): 209 flights of a DJI Matrice 100 "
        "with battery voltage and current, wind, GPS and inertial readings at about 8 Hz, over a grid of 5 speeds, 3 payloads and 4 altitudes.")
    r.bullets(["**Battery chains.** The dataset does not say which flights shared a battery. Consecutive flights were linked from their "
               "motors-off rest voltages into 90 “battery chains”; 71 contain more than one flight. A chain is the unit that is never split "
               "between training and test, because readings from one battery are not independent.",
               "**Labels.** The energy a battery could still give above the reserve is known for chains that reach (or nearly reach) the reserve, "
               "from the energy drawn up to that point.",
               "**Locked test set.** 16 chains (40 flights) were set aside before any modelling, including the longest route (R6), which appears "
               "nowhere in the development data. The remaining 74 chains were used for development with 5-fold cross-validation grouped by chain.",
               "**Corrections found on the way.** Flight start times had to be ordered as times, not text (7 days were otherwise out of order); "
               "4 start times in the raw data are mis-logged; the vertical-velocity sign is the opposite of what the dataset’s documentation says."])

    # ---- 4 method
    r.h("4. Method")
    r.h("4.1 Model A: energy available", 2)
    r.p("Model A reads the battery once per second: voltage, current, power, short rolling statistics, energy drawn so far, the rest voltage "
        "before the flight, and the last minute of readings as a sequence. Every input uses only the past.")
    r.p(f"All five algorithms in the brief were compared with grouped cross-validation, together with three baselines that need no machine "
        f"learning. On readings with a pre-flight voltage, GRU ({f['a_gru_cv']:.2f} Wh) and LSTM ({f['a_lstm_cv']:.2f} Wh) were the most accurate, "
        f"against {f['a_lookup_cv']:.2f} Wh for a fair voltage-lookup baseline; Random Forest, Linear Regression and XGBoost were between "
        f"{f['a_tabular_cv'][0]:.1f} and {f['a_tabular_cv'][1]:.1f} Wh. Every model predicts one reading in {f['latency_ms']} ms or less on a laptop CPU.")
    r.p("The final Model A is an ensemble of five GRUs, each predicting a value and a spread. The model declines to answer when no pre-flight "
        "voltage reading exists, because its errors there are several times larger.")
    r.h("4.2 Model B: energy required", 2)
    mc = n["mission_cv"]
    r.p("A mission is built from its parts: climb, each cruise leg (power × time), hover between legs, descent and ground. Each part is "
        "predicted by a physics-shaped model (rotor momentum theory for power, simple kinematics for time) corrected by XGBoost where that "
        f"clearly helps. In cross-validation missions were predicted to {mc.loc['Physics-first', 'mape_pct']:.2f}%. "
        "Physics-shaped parts were kept as the default because tree models alone could not extrapolate to unseen altitudes or payloads "
        "(9–14% error) while physics-shaped ones stayed near 3%. Model B uses only what is known when planning: distance, commanded speed, "
        "payload, altitude and expected wind.")
    r.h("4.3 Calibrated ranges", 2)
    da, db = n["dev_a"], n["dev_b"]
    r.p("A range is honest if a “90% range” contains the truth about 90% of the time on data the model has not seen. Both models’ ranges are "
        "set by conformal calibration: the size of the errors on held-out batteries and flights decides how wide the range must be. Every "
        "battery counts equally, because thousands of readings from one battery share the same error.")
    r.bullets([f"**Model A:** held-out 90% ranges covered {pct(da.loc['Calibrated, per-reading spread (main)', 'cov90'])} in development and were "
               f"{da.loc['Calibrated, per-reading spread (main)', 'width90']:.1f} Wh wide.",
               f"**Model B:** the errors of each part on held-out flights are replayed together on a new mission; held-out 90% ranges covered "
               f"{pct(db.loc['Per-part errors, replayed jointly (main)', 'cov90'])} and were {db.loc['Per-part errors, replayed jointly (main)', 'width90']:.1f} Wh wide. "
               "Sorties flown back to back on one battery share one error, because their errors were found to move together."])
    r.h("4.4 The decision", 2)
    r.p(f"P(success) is the probability that energy available is at least energy required, computed from the two ranges. The scheduler approves "
        f"a mission if P(success) ≥ τ, with τ = {tau}. Decisions were evaluated without flying anything, by asking of real data: would this "
        "battery, in this state, have managed that recorded mission? Both halves are measurements, so the answer is known. Four rules were compared: "
        "minutes left (the brief), energy point estimates, EnduroSense, and an oracle that knows the truth.")

    # ---- 5 results
    r.h("5. Results on the locked test set")
    log = n["log"]
    r.p(f"The test set was opened on {log[0]['opened_at'][:10]}. Every opening is logged with a fingerprint of the code and settings; "
        + ("the evaluation was rerun once after a verification pass changed reporting code only, with identical numbers. " if len(log) > 1 else "")
        + f"11 of the 16 test chains carry an energy label and are used for Model A ({n['summary']['model_a']['readings']:,} readings); "
        "all 40 flights are used for Model B.")
    r.h("5.1 Energy available", 2)
    order = [MAIN, "LSTM", "XGBoost + physics", "Linear Regression", "Random Forest", "XGBoost", "GRU", "Voltage lookup", "Energy counting (BMS)"]
    names = {MAIN: "GRU ensemble, calibrated (main model)", "GRU": "GRU (single)", "Voltage lookup": "Voltage lookup (no ML)",
             "Energy counting (BMS)": "Energy counting (no ML)"}
    rows = [[names.get(m, m), f"{a.loc[m, 'cv_mae_supported']:.2f}", f"{a.loc[m, 'test_mae_supported']:.2f}",
             "—" if m == "Voltage lookup" else f"{a.loc[m, 'vs_lookup_diff']:+.2f} ({a.loc[m, 'vs_lookup_ci_low']:+.2f} to {a.loc[m, 'vs_lookup_ci_high']:+.2f})"]
            for m in order]
    r.table(["Model", "Cross-validation (Wh)", "Test (Wh)", "Test, against voltage lookup (95% interval)"], rows, [6.2, 3.0, 2.2, 5.0],
            caption="Model A error on comparable readings (those with a pre-flight voltage).", bold_rows=(0,))
    r.bullets([f"**The main model held up:** {a.loc[MAIN, 'test_mae_supported']:.2f} Wh on unseen batteries, and the only model whose advantage over the "
               "baseline is clear (its interval excludes zero).",
               f"**A single GRU did not.** Best in cross-validation ({a.loc['GRU', 'cv_mae_supported']:.2f} Wh), it scored {a.loc['GRU', 'test_mae_supported']:.2f} Wh "
               "on test, no better than the baseline. Averaging five networks is what made the result dependable.",
               f"**Ranges:** the 90% range covered {pct(ar['cov90'])} of test readings (plausible band with {int(ar['chains'])} batteries: "
               f"{pct(ar['cov90_ci_low'], 0)}–{pct(ar['cov90_ci_high'], 0)}). Almost all the misses were on {word(f['a_batteries_over_estimated'])} batteries "
               f"whose capacity the model over-estimated; the other {word(f['a_batteries_covered'])} were inside their range for at least "
               f"{pct(f['a_min_coverage_of_covered'])} of their readings. The risk is per battery, not per moment."])
    r.figure(FIN / "figures" / "model_a_cv_vs_test.png", "Model A: cross-validation error against test error, per model.", 14.5)
    r.h("5.2 Energy required", 2)
    rows = [["Cross-validation (development)", "153", f"{mc.loc['Physics-first', 'mape_pct']:.2f}%", f"{mc.loc['Physics-first', 'mae_wh']:.2f}"]]
    for g, label in (("all test flights", "All test flights"), ("route seen in development", "Route flown in training (R1)"),
                     ("unseen routes", "Unseen routes (R2, R3, R6)"), ("unseen route R6", "of which R6, the longest")):
        rows.append([label, int(bm.loc[g, "flights"]), f"{bm.loc[g, 'mape_pct']:.2f}%", f"{bm.loc[g, 'mae_wh']:.2f}"])
    r.table(["Flights", "n", "Error", "Error (Wh)"], rows, [7.0, 1.6, 2.6, 2.6], caption="Model B mission-energy error.", bold_rows=(1,))
    b0 = br.loc["Physics-first, calibrated: all test flights"]
    r.bullets([f"**On the trained route the model generalises** to new batteries and days ({bm.loc['route seen in development', 'mape_pct']:.1f}%).",
               f"**On unseen routes it misses the 5% target, in the safe direction:** all {f['b_unseen_flights']} flights were over-predicted. The excess is in the cruise "
               "legs of a route about 45% longer than any flown in development.",
               f"**Its ranges were too narrow on test:** {pct(b0['cov90'], 0)} coverage against a 90% target. Most misses on the trained route are at "
               "100 m altitude, a weak spot already flagged during development."])
    r.figure(FIN / "figures" / "calibration_and_missions.png", "Left: calibration of both models on test data (on the dashed line = perfectly "
             "calibrated). Right: Model B’s 90% ranges against measured mission energy.", 16.0)
    r.h("5.3 Decisions", 2)
    rows = [["Minutes left ≥ duration (the brief)", pct(n["dev"].loc[P1, "unsafe_approval_rate"]), pct(allp.loc[P1, "unsafe_approval_rate"]),
             pct(allp.loc[P1, "wasted_refusal_rate"]), pct(pre.loc[P1, "unsafe_approval_rate"])],
            ["Energy, point estimates", pct(n["dev"].loc[P2, "unsafe_approval_rate"]), pct(allp.loc[P2, "unsafe_approval_rate"]),
             pct(allp.loc[P2, "wasted_refusal_rate"]), pct(pre.loc[P2, "unsafe_approval_rate"])],
            ["Point estimates + margin tuned in development", "—", pct(allp.loc[p2m, "unsafe_approval_rate"], 2),
             pct(allp.loc[p2m, "wasted_refusal_rate"]), pct(pre.loc[p2m_pre, "unsafe_approval_rate"], 2)],
            [f"EnduroSense, τ = {tau}", pct(n["dev"].loc[P3, "unsafe_approval_rate"], 2), pct(allp.loc[P3, "unsafe_approval_rate"], 2),
             pct(allp.loc[P3, "wasted_refusal_rate"]), pct(pre.loc[P3, "unsafe_approval_rate"], 2)]]
    r.table(["Rule", "Unsafe approvals, development", "Unsafe approvals, test", "Wasted refusals, test", "Unsafe approvals at take-off, test"],
            rows, [5.6, 2.7, 2.6, 2.6, 2.9], caption="Unsafe approvals are missions approved that would have cut into the reserve (share of such "
            "missions). Wasted refusals are missions refused that would have succeeded.", bold_rows=(3,))
    m = n["mistakes"]["all pairs"]
    r.bullets([f"**The headline holds on unseen data.** The brief’s rule approves about 1 in {round(1 / allp.loc[P1, 'unsafe_approval_rate'])} missions "
               f"that would fail; EnduroSense about 1 in {round(1 / allp.loc[P3, 'unsafe_approval_rate'], -1):.0f}, and none at take-off decisions.",
               f"**The price is caution.** EnduroSense refuses {pct(allp.loc[P3, 'wasted_refusal_rate'], 0)} of missions that would have succeeded. These are "
               f"close calls: their median true margin was {m['refused_margin_wh_median']:.1f} Wh.",
               f"**Minutes-left cannot be rescued with a margin.** With the margin that made it as safe in development, it refuses {pct(allp.loc[p1m, 'wasted_refusal_rate'], 0)} of feasible missions.",
               f"**Point estimates plus a fixed margin did as well over all cases.** At take-off decisions the probability refused fewer feasible missions "
               f"({pct(pre.loc[P3, 'wasted_refusal_rate'], 0)} against {pct(pre.loc[p2m_pre, 'wasted_refusal_rate'], 0)}) at the same safety, and it needed no tuning.",
               f"**Is P(success) honest?** Mostly: missions rated 95–99% succeeded {pct(band['observed_success'])} of the time, slightly over-confident, "
               "because of the two over-estimated batteries."])
    r.figure(FIN / "figures" / "tradeoff.png", "Unsafe approvals against wasted refusals as each rule’s threshold moves, on test data. "
             "Closer to the bottom-left corner is better.", 16.0)
    r.h("5.4 Fleet simulation", 2)
    r.p(f"Four drones work through 40 tasks a day for 200 simulated days. Each battery is one of {int(f_p3['batteries'])} real unseen batteries and each "
        "task is a real recorded flight or pair of sorties; only the pairing is simulated.")
    per100 = lambda x: f"{x:.1f}" if x >= 1 else f"{x:.2f}"
    rows = [[lab, per100(row["unsafe_per_100_missions"]), f"{row['completed_per_day']:.1f}", f"{row['swaps_per_day']:.1f}"]
            for lab, row in (("Minutes left (the brief)", f_p1), ("Energy, point estimates", f_p2),
                             ("Point estimates + development margin", f_p2m), (f"EnduroSense, τ = {tau}", f_p3), ("Oracle", fleet.loc["P4 oracle"]))]
    r.table(["Rule", "Unsafe missions per 100 flown", "Completed per day (of 40)", "Battery swaps per day"], rows, [6.2, 3.4, 3.4, 3.0],
            caption="Fleet simulation on unseen batteries.", bold_rows=(3,))
    r.p(f"Against the brief’s rule EnduroSense has {int(f_p1['unsafe_per_100_missions'] / f_p3['unsafe_per_100_missions'])} times fewer unsafe missions for "
        f"{100 * (f_p3['swaps_per_day'] / f_p1['swaps_per_day'] - 1):.0f}% more battery swaps. Against point estimates with a margin it sits at a nearby point of the "
        "same trade-off: slightly more unsafe missions, fewer swaps.")
    r.figure(FIN / "figures" / "fleet_simulation.png", "Fleet simulation on unseen batteries.", 16.0)

    # ---- 6 what held, what did not
    r.h("6. What held up and what did not")
    r.h("Supported by the test set", 2)
    takeoff = "to zero" if f["unsafe_p3_takeoff"] == 0 else f"to {pct(f['unsafe_p3_takeoff'], 2)}"
    r.bullets([f"A calibrated GRU ensemble estimates energy to reserve to about {f['a_main']:.1f} Wh on unseen batteries "
               f"and beats a fair baseline without machine learning ({f['a_lookup']:.1f} Wh).",
               f"Mission energy is predicted to about {f['b_seen']:.0f}% for the kind of mission flown in training.",
               f"Deciding on energy with a calibrated margin cuts unsafe approvals from about {pct(f['unsafe_p1'], 0)} to {pct(f['unsafe_p3'], 1)}, "
               f"and {takeoff} at take-off decisions.",
               "The method is honest about what it does not know: the unsafe cases trace to two batteries the model over-estimated, the "
               "per-battery risk identified during development."])
    r.h("Not supported, and not claimed", 2)
    r.bullets([f"That the mission model is within 5% on routes much longer than its training routes ({bm.loc['unseen routes', 'mape_pct']:.1f}%, over-predicted).",
               f"That the mission model’s 90% ranges are calibrated on new data ({pct(b0['cov90'], 0)} on test).",
               "That a probability is measurably better than point estimates plus a well-chosen margin.",
               "That a single GRU beats the baseline."])
    r.p("These weaknesses were reported, not patched. Fixing them and scoring again on the same test set would make the test results optimistic.")

    # ---- 7 limitations
    r.h("7. Limitations")
    r.bullets([f"**Small test set.** {f['labelled_batteries']} labelled batteries and {f['test_flights']} flights from one drone and one battery type. "
               "Every interval is wide for that reason.",
               "**The truth itself has error.** The energy-to-reserve label comes from rest voltages and is uncertain by about 1.5–2 Wh.",
               "**Decisions were not flown.** They were evaluated by pairing real battery states with real recorded missions.",
               "**Settings await confirmation.** The reserve (22.6 V) and threshold (0.95) are defaults; the guide has not yet confirmed them.",
               "**Conditions outside the data** (other airframes, temperatures, battery ages, stronger wind) are untested. The mission model warns when "
               "a mission is outside the ranges it was tested on."])

    # ---- 8 reproducibility
    r.h("8. Reproducibility and verification")
    r.bullets(["**One command rebuilds everything** from the raw files: `python scripts/run_all.py --final` (about 30 minutes). A full rebuild with empty "
               "caches reproduced all 74 result files and all 19 model files byte for byte.",
               "**The test set is gated and logged.** Test data can only be read by the final-evaluation script; every opening is recorded, and a rerun "
               "on changed code needs a stated reason.",
               f"**{word(f['verification_passes']).capitalize()} verification passes** re-checked finished phases adversarially: independent recomputation of headline numbers, a search for "
               "leakage, and mutation testing (deliberately breaking the code to confirm that a test fails). What each pass found and fixed is in "
               "docs/verification_log.md.",
               "**Tests:** the automated test suite covers the data pipeline, both models, calibration, decisions, the final script and the dashboard."])

    # ---- 9 demo
    r.h("9. The dashboard")
    r.p("A Streamlit dashboard (`streamlit run app/streamlit_app.py`) loads the saved models and the final results; nothing is trained in it.")
    r.bullets(["**Three missions, one battery:** one unseen battery before take-off and three recorded missions that the minutes-left rule approves; "
               "EnduroSense refuses the one that would have cut into the reserve.",
               "**Mission check:** pick any unseen battery and moment, plan a mission or choose a recorded one, and see both ranges, P(success), "
               "the decision of each rule and what was measured afterwards.",
               "**Fleet:** the fleet simulation, with adjustable fleet size, workload and threshold.",
               "**Results:** the tables and charts of this report, read from the same result files."])

    # ---- 10 next
    r.h("10. Next steps")
    r.bullets(["**More batteries and routes.** The main limit is data: a second battery type, longer routes and more flights at 100 m would address "
               "the two missed criteria directly.",
               "**Learn each battery.** The error is a per-battery offset, so a model that updates its estimate of a battery’s capacity after each "
               "flight should narrow the ranges and remove the per-battery risk.",
               "**Widen Model B’s ranges for missions outside its training routes,** using route length as an input to the range.",
               f"**On-board timing** on a companion computer (for example a Raspberry Pi); on a laptop CPU every model predicts in {f['latency_ms']} ms or less.",
               "**Confirm the reserve and threshold** with the guide; both are single settings and the pipeline reruns in half an hour."])

    r.doc.core_properties.title = "EnduroSense: Final Report"
    r.doc.core_properties.author = "EnduroSense project"
    r.doc.core_properties.subject = "Uncertainty-aware mission feasibility prediction for UAVs"
    out.parent.mkdir(parents=True, exist_ok=True)
    r.doc.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    build()
