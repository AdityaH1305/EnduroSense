"""Build the project overview for the mentor (Word) from the saved result files.

Usage: python docs/build/build_overview.py        -> docs/EnduroSense_Project_Overview.docx

A plain-language account of the whole project: the idea, the data, what was built step by
step, the final results, what did not hold up, and the decisions that need the mentor's
approval. Every number is read from results/ (through the same `headline_facts()` the
dashboard, report and slides use), so the document follows the results if the pipeline is rerun.
"""
import glob
import importlib.util
import json
import re
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from docx.enum.text import WD_ALIGN_PARAGRAPH

from endurosense.config import ROOT, data_path
from endurosense.data.load import load_processed
from endurosense.data.split import load_split
from endurosense.features.model_a import FEATURES
from endurosense.plots import SERIES, apply_style, plt

sys.path.insert(0, str(ROOT / "app"))
import logic as L  # noqa: E402

BUILD = ROOT / "docs" / "build"
RES, FIN = data_path("results"), data_path("results") / "final"
OUT = ROOT / "docs" / "EnduroSense_Project_Overview.docx"


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), BUILD / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def idea_figure(path, tau) -> None:
    """Three boxes: energy available, energy required, and the decision."""
    apply_style()
    fig, ax = plt.subplots(figsize=(9.5, 2.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    boxes = [(0.2, "Energy available", "What the battery can still give\nabove a safe reserve", SERIES[0]),
             (3.6, "Energy required", "What the planned mission\nwill use", SERIES[1]),
             (7.0, f"P(success) ≥ {tau:.0%} ?", "GO only if the battery very\nlikely covers the mission", SERIES[2])]
    for x, title, sub, col in boxes:
        ax.add_patch(plt.Rectangle((x, 0.5), 2.8, 2.0, facecolor=col, alpha=0.14, edgecolor=col, linewidth=1.5))
        ax.text(x + 1.4, 1.85, title, ha="center", va="center", fontsize=13, fontweight="bold", color="#0b0b0b")
        ax.text(x + 1.4, 1.1, sub, ha="center", va="center", fontsize=10, color="#52514e")
    ax.text(3.3, 1.5, "vs", ha="center", va="center", fontsize=13, color="#52514e")
    ax.annotate("", xy=(6.95, 1.5), xytext=(6.45, 1.5), arrowprops={"arrowstyle": "-|>", "color": "#52514e", "lw": 1.8})
    fig.savefig(path, dpi=170)
    plt.close(fig)


def build(out=OUT) -> None:
    rep, slides = load("build_report.py"), load("build_slides.py")
    r, pct = rep.Report(), rep.pct
    f, split = L.headline_facts(), load_split()
    word, tau = f["number_word"], f["tau"]
    roles = pd.Series(split["flight_role"])

    # ---- numbers and figures
    n, world = slides.numbers(), slides.demo_world()
    case = L.three_missions(world["grid"], tau)
    figs = slides.charts(n, world, case, tau)
    slides.FIG.mkdir(parents=True, exist_ok=True)
    idea_png = slides.FIG / "idea.png"
    idea_figure(idea_png, tau)
    a = n["a"]
    MAIN = "GRU ensemble, calibrated (main)"
    cv = pd.read_csv(RES / "model_a" / "cv_metrics.csv").set_index("model")
    mcv = pd.read_csv(RES / "model_b" / "mission_cv.csv").set_index("model")
    gen = pd.read_csv(RES / "model_b" / "generalisation_summary.csv").pivot(index="model", columns="test", values="mape_pct")
    taus = pd.read_csv(FIN / "tau_table.csv").set_index("tau")
    da, db = n["dev_a"].loc["Calibrated, per-reading spread (main)"], n["dev_b"].loc["Per-part errors, replayed jointly (main)"]
    flights, chains = load_processed("flights"), load_processed("chains")
    readings = pq.ParquetFile(data_path("processed") / "samples.parquet").metadata.num_rows
    n_tests = sum(len(re.findall(r"^def test_", open(p, encoding="utf8").read(), flags=re.M)) for p in glob.glob(str(ROOT / "tests" / "test_*.py")))
    log = json.loads((FIN / "test_access_log.json").read_text())
    state = world["a"].loc[case["state"].iloc[0]]
    mis = [world["missions"].iloc[int(m)] for m in case["mission"]]
    est = float(np.interp(0.5, world["lv"], world["q"].loc[state.name]))
    reserve = f["reserve_v"]

    # ================================================================= title
    for _ in range(5):
        r.p()
    r.p("EnduroSense", size=34, color=rep.BLUE, align=WD_ALIGN_PARAGRAPH.CENTER, after=4).runs[0].bold = True
    r.p("Can this battery fly this mission?", size=16, color=rep.INK, align=WD_ALIGN_PARAGRAPH.CENTER, after=30)
    r.p("Project overview for review", size=13, color=rep.MUTED, align=WD_ALIGN_PARAGRAPH.CENTER, after=40)
    r.p("What the project set out to do, what was built step by step, what the final results are, and the decisions that need "
        "approval or changes. Written in plain language; technical terms are explained in the glossary at the end.",
        size=11.5, color=rep.MUTED, align=WD_ALIGN_PARAGRAPH.CENTER)
    r.page_break()

    # ================================================================= 1 at a glance
    r.h("1. At a glance")
    r.p("The intern brief asks for a machine-learning model that predicts the remaining flight time of a UAV battery, and for a comparison "
        "of five algorithms. This project does that, and adds one step that makes the prediction useful for a real decision.")
    r.bullets(["**The question it answers:** not only “how many minutes are left?” but “can this battery complete this mission and still "
               "keep a safe reserve?”",
               "**How:** one model estimates the energy the battery can still give. A second model estimates the energy the mission needs. "
               "Each estimate comes with an honest range, and the two together give a probability of success.",
               f"**The rule:** a mission is approved only if the probability of success is at least {tau:.0%}.",
               f"**The data:** {len(flights)} real flights of a DJI Matrice 100 drone (public data from Carnegie Mellon University)."])
    r.h("The main result", 2)
    r.p(f"On batteries and flights that were kept aside and never used for building the models, the rule in the brief (“minutes left is "
        f"at least the mission duration”) approved **{pct(f['unsafe_p1'])}** of the missions that would have cut into the battery’s safety reserve. "
        f"EnduroSense approved **{pct(f['unsafe_p3'], 2)}**, and none when the decision was made before take-off.")
    r.h("Where it stands", 2)
    r.bullets(["**All planned work is finished:** data preparation, both models, the decision rule, the final evaluation, a live dashboard, "
               "a final report and a slide deck.",
               f"**Success criteria:** {word(f['criteria_total'])} were written down before the final evaluation. "
               f"{word(f['criteria_met']).capitalize()} were met and {word(f['criteria_total'] - f['criteria_met'])} were not. Section 7 explains both.",
               "**The two extensions you suggested are built:** an energy-aware fleet planner, and a choice of where a task's data is "
               "processed (on board, on an edge server or on another drone). They are simulations on the real batteries and missions; "
               "step 9 in section 5 describes them.",
               "**What is needed now:** approval of a few choices that were made using sensible defaults, or instructions to change them. "
               "They are listed in the next section. Changing any of them takes about half an hour of computer time."])

    # ================================================================= 2 decisions
    r.h("2. Decisions that need your approval")
    r.p("These choices were made with default values because no confirmation was available at the time. Each is a single setting or a "
        "clearly separated piece of work, so any of them can be changed.")
    t90, t95, t99 = taus.loc[0.90], taus.loc[tau], taus.loc[0.99]
    rows = [
        ["1. Scope: answering “can this battery fly this mission?” in addition to the brief’s “minutes left”",
         f"Both are delivered. Minutes left is still predicted (average error {a.loc[MAIN, 'minutes_mae']:.1f} min) and is one of the rules compared.",
         "If only minutes left is wanted, the first model alone covers the brief."],
        [f"2. Safety reserve: the battery level a mission must not go below ({reserve} V at rest, about {reserve / 6:.2f} V per cell)",
         f"{reserve} V. Chosen because it is a common conservative level and many recorded batteries reach it.",
         "A different reserve changes how much energy counts as “available”. Rerun needed (about 30 minutes)."],
        [f"3. Approval threshold: how sure the system must be before saying GO ({tau:.0%})",
         f"{tau:.0%}. A stricter value is safer but refuses more missions (see Table 2).",
         "Any value can be set; no rerun of the models is needed, only of the decision step."],
        ["4. Range level: the ranges reported with each prediction (90%)", "90%, with 50%, 80% and 95% also reported in the result tables.", "A single setting."],
        ["5. The five algorithms of the brief are compared on the battery model only",
         "The mission model uses physics formulas corrected by XGBoost, because a mission plan is a handful of numbers, not a time series, "
         "so LSTM and GRU do not apply to it.", "If all five must be compared for the mission model too, that can be added."],
        ["6. Two success criteria were missed and are reported as missed",
         "The models were not adjusted after seeing the final results, because that would make those results look better than they are.",
         "If improvements are wanted, they should be made and then judged on new data."],
        ["7. Two optional extras were not done", "Timing on a Raspberry Pi, and a simulated high-wind test.", "Either can be added if required."],
    ]
    r.table(["Decision", "What was used, and why", "If you prefer otherwise"], rows, [5.6, 6.2, 4.6],
            caption="Decisions for review.", align_right_from=9)
    r.table(["Threshold", "Failing missions approved (all moments)", "Failing missions approved (before take-off)", "Good missions refused"],
            [[f"{t:.0%}" + (" (used)" if t == tau else ""), pct(row["unsafe_approval_rate"], 2), pct(row["preflight_unsafe_approval_rate"], 2),
              pct(row["wasted_refusal_rate"], 0)] for t, row in ((0.90, t90), (tau, t95), (0.99, t99))], [3.0, 4.6, 4.8, 4.0],
            caption="What the approval threshold trades off, measured on the final test data.", bold_rows=(1,))

    # ================================================================= 3 the idea
    r.h("3. The idea in plain words")
    r.p("A drone scheduler has to decide, mission by mission, whether a battery is good enough. “Minutes of flight left” is a weak basis "
        "for that decision, for two reasons.")
    r.bullets(["**Minutes depend on what the drone is doing right now.** They are worked out from the power being drawn at this moment. "
               "A drone that is hovering or descending draws little power, so it appears to have plenty of time, whatever the next mission needs.",
               "**One number hides how sure it is.** Two batteries can show the same estimate while one of them is much less certain."])
    r.p("EnduroSense therefore works in energy (watt-hours, Wh), which can be added up over the parts of a mission, and gives every estimate "
        "a range.")
    r.figure(idea_png, "The idea: compare the energy a battery can give with the energy a mission needs, and approve only when success is very likely.", 15.5)
    r.h("A real example from the test data", 2)
    r.p(f"One battery, just before take-off. The model estimated {est:.0f} Wh above the reserve; {state['remaining_wh']:.0f} Wh were really there. "
        "Three missions that were really flown are offered to it. The brief’s minutes-left rule says GO to all three.")
    r.table(["Mission", "Energy it really needed", "EnduroSense: chance of success", "EnduroSense says", "What would have happened"],
            [[m["label"].replace("Flight ", "Flight ").split(":")[0] if m["sorties"] == 1 else f"{int(m['sorties'])} flights back to back",
              f"{m['true_wh']:.0f} Wh", f"{row['P3']:.0%}", "GO" if row["P3"] >= tau else "NO-GO",
              "Fine" if row["feasible"] else f"{-row['true_margin_wh']:.0f} Wh into the reserve"]
             for (_, row), m in zip(case.iterrows(), mis)], [4.0, 3.0, 3.2, 2.6, 3.6],
            caption="Three real missions for one real battery. The case is picked by a fixed rule, not by hand.", bold_rows=(1,))
    r.figure(figs["three"][0], "The same three missions. Blue: what the battery can give. Orange: what the mission needs. Darker means more likely; "
             "diamonds show what was measured. In the middle case the two ranges overlap, so the mission is refused.", 16.0)

    # ================================================================= 4 data
    r.h("4. The data")
    r.p(f"The project uses the public “package delivery quadcopter” dataset from Carnegie Mellon University: **{len(flights)} flights** of a DJI "
        f"Matrice 100 over {flights['date'].nunique()} days, with **{readings:,} sensor readings** (battery voltage and current, wind, position, motion). "
        "The flights cover 5 speeds, 3 payloads and 4 altitudes.")
    r.bullets([f"**Which flights shared a battery?** The data does not say. It was worked out from the battery’s resting voltage before and after each "
               f"flight, which links the flights into **{len(chains)} “battery chains”** ({int((chains['n_flights'] > 1).sum())} of them with more than one flight). "
               "This matters: readings from one battery are alike, so a battery must be either entirely in the training data or entirely in the test data.",
               f"**A test set locked away from the start.** {len(split['test_chains'])} chains ({int((roles == 'test').sum())} flights) were set aside before any "
               "model was built, including routes that appear nowhere else. They were used only for the final evaluation. The computer code refuses "
               "to read them otherwise, and every time they are read is recorded.",
               f"**The rest** ({len(chains) - len(split['test_chains'])} chains, {int((roles == 'dev').sum())} flights) was used for building and comparing models, "
               "always testing on batteries the model had not seen (5-fold cross-validation grouped by battery).",
               "**Problems found in the data and fixed:** flight start times were stored as text and sorted wrongly (7 days were out of order); "
               "4 start times were mis-recorded; the documentation gives the wrong sign for vertical speed."])

    # ================================================================= 5 what was built
    r.h("5. What was built, step by step")
    r.h("Step 1. Preparing the data", 2)
    r.bullets(["Cleaned the readings (removed sensor glitches, checked time gaps).",
               "Split each flight into phases: on the ground, climbing, cruising, hovering, descending.",
               "Worked out the energy used at every moment (voltage × current × time) and linked flights into battery chains.",
               "Locked the test set."])
    r.h("Step 2. Defining what the models should learn", 2)
    r.bullets([f"**For the battery model:** at every second, the true “energy left above the reserve”. It is known for batteries that were flown down "
               f"to (or near) the reserve. The model gets {len(FEATURES)} inputs describing the battery right now (voltage, current, energy drawn so far, "
               "recent trends) plus the last minute of readings. Every input uses only the past, never the future.",
               "**For the mission model:** the energy each part of a flight used (climb, each straight leg, hover, descent, ground), together with the "
               "plan that was known beforehand (distance, speed, payload, altitude, wind)."])
    r.h("Step 3. The battery model (Model A): comparing the five algorithms", 2)
    r.p("All five algorithms named in the brief were trained and compared, together with three simple methods that use no machine learning "
        "(for example, reading the charge from the resting voltage and subtracting the energy used). A model is only worth having if it beats these.")
    order = [MAIN, "LSTM", "GRU", "XGBoost", "Random Forest", "Linear Regression", "Voltage lookup", "Energy counting (BMS)"]
    names = {MAIN: "Five GRUs averaged (final model)", "GRU": "GRU (single)", "Voltage lookup": "Voltage lookup (no ML)", "Energy counting (BMS)": "Energy counting (no ML)"}
    r.table(["Method", "Error while developing (Wh)", "Error on the final test (Wh)", "Time per prediction (ms)", "Size (MB)"],
            [[names.get(m, m), f"{a.loc[m, 'cv_mae_supported']:.2f}", f"{a.loc[m, 'test_mae_supported']:.2f}",
              "—" if m == MAIN else f"{cv.loc[m, 'latency_ms']:.1f}", "—" if m == MAIN else (f"{cv.loc[m, 'size_mb']:.2f}" if cv.loc[m, "size_mb"] >= 0.005 else "<0.01")]
             for m in order], [5.4, 3.0, 3.0, 2.8, 2.2],
            caption="Battery model: average error in watt-hours (lower is better). A battery holds about 65 Wh above the reserve.", bold_rows=(0,))
    r.bullets(["**GRU and LSTM were the most accurate while developing.** The other three algorithms were about as good as the simple voltage method.",
               f"**The final model is five GRUs averaged.** A single GRU looked best while developing but was no better than the simple method on the final "
               f"test ({a.loc['GRU', 'test_mae_supported']:.2f} Wh against {f['a_lookup']:.2f} Wh). The average of five was the best of all ({f['a_main']:.2f} Wh).",
               f"**Speed:** every model answers in {f['latency_ms']} ms or less on a laptop processor."])
    r.h("Step 4. The mission model (Model B)", 2)
    r.p("A mission is treated as a sum of parts: climb, each straight leg (power × time), hover between legs, descent and ground time. Each part "
        "is first estimated with a physics formula and then corrected by XGBoost where the formula is clearly off.")
    r.bullets([f"**Accuracy while developing:** {mcv.loc['Physics-first', 'mape_pct']:.1f}% error on missions the model had not seen, against "
               f"{mcv.loc['Average flight (naive)', 'mape_pct']:.0f}% for simply guessing the average flight.",
               f"**Why physics first:** models made only of decision trees could not handle an altitude or payload they had never seen "
               f"({gen.loc['Random Forest', 'unseen payload']:.0f}–{gen.loc['Random Forest', 'unseen altitude']:.0f}% error), while the physics-shaped model "
               f"stayed near {gen.loc['Physics-first', 'unseen altitude']:.0f}–{gen.loc['Physics-first', 'unseen payload']:.0f}%.",
               "**It warns** when asked about a mission outside what it was tested on (for example a speed never flown)."])
    r.h("Step 5. Honest ranges", 2)
    r.p("A single number is not enough for a safety decision, so each prediction comes with a range. A range is honest if a “90% range” contains "
        "the true value about 90% of the time on data the model has not seen. The ranges were set from the models’ actual errors on unseen batteries "
        "and flights (a method called conformal calibration).")
    r.bullets([f"**While developing:** the battery model’s 90% range contained the truth {pct(da['cov90'])} of the time, the mission model’s {pct(db['cov90'])}.",
               "**No pre-flight reading, no answer:** the battery model declines to answer if the battery’s voltage before take-off is unknown, "
               "because its errors are several times larger there."])
    r.h("Step 6. The decision, and how it was tested without flying", 2)
    r.p(f"From the two ranges the system works out the chance that the battery’s energy covers the mission, and says GO only at {tau:.0%} or more.")
    r.p("New flights could not be flown, so decisions were tested on recorded ones by asking: would this battery, at this moment, have managed that "
        "recorded mission? Both the battery’s true remaining energy and the mission’s true energy were measured, so the right answer is known. "
        "Four rules were compared:")
    r.bullets(["**Minutes left** (the brief): approve if the minutes left are at least the mission’s duration.",
               "**Energy, best estimates:** approve if the estimated energy available is at least the estimated energy needed.",
               f"**EnduroSense:** approve if the chance of success is at least {tau:.0%}.",
               "**Oracle:** knows the truth; shows the best any rule could do."])
    r.p("A small fleet was also simulated: four drones working through 40 tasks a day, using real recorded batteries and real recorded flights.")
    r.h("Step 7. The final evaluation", 2)
    r.bullets([f"What would count as success was written down first ({word(f['criteria_total'])} criteria).",
               "Then the locked test set was opened and every model, range and decision rule was scored on it, with nothing adjusted.",
               "One command rebuilds the whole project from the raw data in about 30 minutes; a full rebuild gave exactly the same files."])
    r.h("Step 8. Dashboard, report and slides", 2)
    r.bullets(["**A live dashboard** with six pages: a worked example, a mission checker (choose a battery, plan a mission, see the decision), "
               "the fleet simulation, the extensions (step 9), the results, and an explanation.",
               "**A final report** (10 pages) and **a slide deck** (14 slides with speaker notes).",
               "All three take their numbers directly from the result files, so they cannot disagree with the results."])

    ext = L.extension_results("final")
    if ext is not None:
        pl, cm = ext["planner"].set_index("code"), ext["computing"].set_index("rule")
        pc = ext["summary"]["settings"]["planner"]
        r.h("Step 9. The extensions you suggested", 2)
        r.p("Both were added after the final evaluation, on top of the finished models. The batteries and missions in them are real recordings. "
            "What the dataset does not contain is simulated: when tasks arrive, their priorities and deadlines, each drone's payload limit, the "
            "chargers, and the radio and edge-server settings. What would count as success was again written down first, and the unseen "
            "batteries were used once, with that use logged.")
        r.bullets([f"**Scheduling several drones.** {len(pc['drone_max_payload_g'])} drones, {pc['tasks_per_day']} tasks a day, each with a priority and a "
                   "deadline, battery swaps and a charging station. The planner matches all waiting tasks to all free drones at once: urgent and "
                   "important tasks first, only if the payload fits, the task can be finished in time, and the battery very likely covers it. "
                   f"It got {pl.loc['S4', 'weighted_on_time_pct']:.0f}% of the priority-weighted work done on time, against "
                   f"{pl.loc['S2', 'weighted_on_time_pct']:.0f}% for a simple first-come queue, with "
                   f"{pl.loc['S4', 'unsafe_per_100_flown']:.2f} missions per 100 cutting into the reserve (the brief's rule: "
                   f"{pl.loc['S1', 'unsafe_per_100_flown']:.1f}).",
                   "**Where to process a task's data.** On board, on an edge server at the depot, or on another drone. The radio and computer use "
                   "very little energy; what costs energy is the drone hovering while it waits for a result. Choosing by energy and deadline met "
                   f"{cm.loc['energy-aware', 'jobs_on_time_pct']:.0f}% of the computing deadlines at {cm.loc['energy-aware', 'computing_wh_per_task']:.1f} Wh per "
                   f"task, against {cm.loc['local', 'jobs_on_time_pct']:.0f}% at {cm.loc['local', 'computing_wh_per_task']:.1f} Wh when always processing on board.",
                   ("**Criteria:** all three were met." if ext["criteria"]["met"].all() else
                    f"**Criteria:** {word(int(ext['criteria']['met'].sum()))} of {word(len(ext['criteria']))} met.") + " The simulated settings (priorities, "
                   "deadlines, chargers, radio and server) are assumptions; please confirm them or suggest others."])

    # ================================================================= 6 results
    r.h("6. Final results on the test data")
    r.p(f"The test data is {int((roles == 'test').sum())} flights from {len(split['test_chains'])} battery chains that nothing in the project had seen. "
        f"{f['labelled_batteries']} of those chains have a known “energy left” value and are used to judge the battery model.")
    r.table(["What was measured", "Result"],
            [["Battery model: average error", f"{f['a_main']:.2f} Wh (best simple method: {f['a_lookup']:.2f} Wh)"],
             ["Battery model: how often its 90% range held the truth", f"{pct(f['a_cov90'])} (aim: 88–92%)"],
             ["Mission model: average error, all test flights", f"{f['b_all']:.1f}%"],
             ["Mission model: on the route it was trained on / on routes never seen", f"{f['b_seen']:.1f}% / {f['b_unseen']:.1f}%"],
             ["Mission model: how often its 90% range held the truth", f"{pct(f['b_cov90'], 0)} (aim: 88–92%)"],
             ["Failing missions approved by the minutes-left rule", pct(f["unsafe_p1"])],
             ["Failing missions approved using best estimates only", pct(f["unsafe_p2"])],
             ["Failing missions approved by EnduroSense", f"{pct(f['unsafe_p3'], 2)} ({pct(f['unsafe_p3_takeoff'], 2)} before take-off)"],
             ["Good missions refused by EnduroSense (the price of caution)", pct(f["wasted_p3"], 0)],
             ["Fleet simulation: unsafe missions per 100, minutes-left rule / EnduroSense", f"{f['fleet_p1']:.1f} / {f['fleet_p3']:.2f}"]],
            [10.4, 6.0], caption="Headline results on the test data.", align_right_from=9)
    r.figure(figs["unsafe"][0], "Share of failing missions that each rule approved, on the test data. Lower is safer.", 13.0)
    r.h("The success criteria", 2)
    r.table(["Criterion (written down before the test)", "Result", "Evidence"],
            [[c["criterion"], v, c["evidence"]] for (_, c), v in zip(f["criteria"].iterrows(), f["verdicts"])], [5.6, 2.7, 8.1],
            caption="Success criteria and outcome.", align_right_from=9)

    # ================================================================= 7 worked / did not
    r.h("7. What worked and what did not")
    r.h("What worked", 2)
    r.bullets([f"**The main idea.** Deciding on energy with a safety margin, instead of on minutes, cut unsafe approvals from {pct(f['unsafe_p1'], 0)} to "
               f"{pct(f['unsafe_p3'], 1)} on data never seen before. In the fleet simulation that is {f['fleet_ratio']} times fewer unsafe missions, "
               f"for about {pct(f['fleet_extra_swaps'], 0)} more battery swaps.",
               f"**The battery model.** {f['a_main']:.1f} Wh average error on unseen batteries, clearly better than the best method without machine learning.",
               f"**The mission model on familiar missions.** {f['b_seen']:.1f}% error on new batteries and days.",
               "**The process.** The results were reproduced exactly from scratch, and the checks described in section 8 found no sign that test data "
               "had leaked into the models."])
    r.h("What did not work as hoped", 2)
    r.bullets([f"**The mission model on unseen routes:** {f['b_unseen']:.1f}% error against a 5% aim. It over-estimated every one of them, which is the safe "
               "direction, but it is a miss. The longest route is about 45% longer than anything the model was trained on.",
               f"**The mission model’s ranges were too narrow on new data:** they held the truth {pct(f['b_cov90'], 0)} of the time, not 90%.",
               f"**The risk sits with individual batteries.** The battery model over-estimated {word(f['a_batteries_over_estimated'])} of the "
               f"{f['labelled_batteries']} test batteries, and all of EnduroSense’s unsafe approvals came from those.",
               "**The probability is not clearly better than a simpler alternative.** “Best estimates plus a fixed safety margin”, with the margin chosen "
               "during development, did about as well overall. The probability refused fewer good missions before take-off "
               f"({pct(f['wasted_p3_takeoff'], 0)} against {pct(f['wasted_p2_margin_takeoff'], 0)}) and needs no margin to be chosen. The large gain is over minutes-left, "
               "not over this alternative.",
               "**A single GRU was not enough.** It looked best during development but not on the test; averaging five fixed that."])
    r.p("These weaknesses were reported as they are. The models were not changed after seeing the test results, because scoring a changed model "
        "on the same test data would give flattering numbers.")

    # ================================================================= 8 trust
    r.h("8. How the results were checked")
    r.bullets(["**The test data was protected.** It was chosen before any modelling, the code refuses to read it except in the final evaluation, and "
               f"every reading of it is logged ({len(log)} entries: the evaluation, one repeat after a check that changed only how results are reported "
               "(the numbers were identical)" + (", and the run of the extensions)." if len(log) > 2 else ")."),
               "**Success was defined first.** The criteria were written down before the test data was opened.",
               "**Everything can be rebuilt.** One command recreates all results from the raw data. A complete rebuild produced identical files.",
               f"**Automatic tests.** {n_tests} tests check the calculations, including by deliberately breaking the code to confirm that a test notices.",
               f"**{word(f['verification_passes']).capitalize()} review rounds.** After each stage the finished work was re-examined as if by a sceptic: "
               "key numbers were recalculated with separately written code, and the project was searched for mistakes."])
    r.table(["Review round", "Most important thing it found and fixed"],
            [["1. Data preparation", "Flights were being sorted by start time as text, which put 7 days in the wrong order and broke the battery chains."],
             ["2. Battery model", "Some summary figures and charts had included test data (no model had used it); they were redone on development data only."],
             ["3. Mission model", "The first choice of model under-estimated fast missions (the unsafe direction); it was replaced by the physics-first model."],
             ["4. Ranges and decisions", "Missions of several flights on one battery were treated as if their errors were unrelated; they are not, and this was corrected."],
             ["5. Final evaluation", "A comparison in the write-up was unfair to the development figures; it was corrected. A full rebuild from scratch matched exactly."],
             ["6. Dashboard, report, slides", "Statements such as “two of four criteria met” were typed by hand; they are now taken from the result files."]],
            [4.4, 12.0], caption="What each review round found.", align_right_from=9)

    # ================================================================= 9 vs brief
    r.h("9. How this compares with the brief")
    r.table(["The brief asks for", "What was done"],
            [["Predict remaining flight time", f"Done (average error {a.loc[MAIN, 'minutes_mae']:.1f} min), and extended to “can this battery fly this mission?”"],
             ["Compare Linear Regression, Random Forest, XGBoost, LSTM and GRU", "Done for the battery model, with error, speed and size for each, plus three simple methods as a reference."],
             ["A working model", "A battery model, a mission model, a decision rule, and a dashboard that runs them."],
             ["(Not asked) How sure is the prediction?", "Every prediction has a range, checked on unseen data."],
             ["(Not asked) Is the decision actually safer?", "Measured on unseen data and in a fleet simulation, against the brief’s own rule."]],
            [6.4, 10.0], caption="The brief and the project.", align_right_from=9)

    # ================================================================= 10 deliverables
    r.h("10. What is in the project")
    r.table(["Item", "Where", "What it is"],
            [["Dashboard", "app/", "Live demo. Start with: streamlit run app/streamlit_app.py"],
             ["Final report", "docs/EnduroSense_Final_Report.docx", "10 pages, with tables and figures"],
             ["Slides", "docs/EnduroSense_Slides.pptx", "14 slides with speaker notes"],
             ["Results summary", "docs/results_summary.md", "All final numbers, and what may and may not be claimed"],
             ["Review log", "docs/verification_log.md", "What each review round found and fixed"],
             ["Step-by-step notes", "docs/phase_notes/", "One note per stage of the work"],
             ["Code", "src/, scripts/, tests/", "10 numbered steps; one command runs steps 1 to 9, the extensions run on their own"]],
            [3.4, 6.2, 6.8], caption="Deliverables.", align_right_from=9)

    # ================================================================= 11 limits / next
    r.h("11. Limits and possible next steps")
    r.bullets([f"**Small test set:** {f['labelled_batteries']} batteries with known energy and {f['test_flights']} flights, from one drone and one battery type. "
               "The results are honest but the uncertainty around them is wide.",
               "**Decisions were not flown:** they were judged by pairing recorded battery states with recorded missions.",
               "**Other drones, temperatures and battery ages are untested.** The method would carry over; the models would need that drone’s data."])
    r.p("If more work is wanted, the most useful steps would be:", after=3)
    r.bullets(["More batteries and longer routes, which address both missed criteria directly.",
               "Letting the battery model learn each battery’s capacity as it flies, to remove the per-battery risk.",
               "Wider mission ranges for routes outside the training data.",
               "Timing the models on on-board hardware."])

    # ================================================================= glossary
    r.h("Glossary")
    r.table(["Term", "Meaning"],
            [["Watt-hour (Wh)", "A unit of energy. This battery pack holds about 100 Wh in total and about 65 Wh above the reserve."],
             ["Reserve", "The battery level a mission must not go below, kept as a safety buffer."],
             ["Energy available", "Energy the battery can still give before reaching the reserve."],
             ["Energy required", "Energy a planned mission will use."],
             ["Battery chain", "The flights flown one after another on the same battery."],
             ["Development data / test data", "Data used to build and compare models / data kept aside and used only for the final results."],
             ["Cross-validation", "Building a model on part of the development data and checking it on the rest, repeated so every battery is checked once."],
             ["Baseline", "A simple method without machine learning, used as the bar a model must beat."],
             ["Linear Regression, Random Forest, XGBoost", "Algorithms that learn from a table of numbers."],
             ["LSTM, GRU", "Neural networks that read a sequence of readings over time."],
             ["Range (90% range)", "An interval that should contain the true value about 90% of the time."],
             ["Calibration", "Adjusting the ranges so that they are as reliable as they claim."],
             ["P(success)", "The estimated chance that the battery’s energy covers the mission."],
             ["Threshold (τ)", f"The chance of success required to approve a mission ({tau:.0%} here)."],
             ["Unsafe approval", "Approving a mission that would have cut into the reserve."],
             ["Wasted refusal", "Refusing a mission that would have been fine."]],
            [5.0, 11.4], caption=None, align_right_from=9)

    r.doc.core_properties.title = "EnduroSense: project overview for review"
    r.doc.core_properties.author = "EnduroSense project"
    out.parent.mkdir(parents=True, exist_ok=True)
    r.doc.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    build()
