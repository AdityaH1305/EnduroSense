"""Build the presentation (PowerPoint) from the saved result files.

Usage: python docs/build/build_slides.py        -> docs/EnduroSense_Slides.pptx

Numbers are read from results/; charts are drawn here at slide-readable sizes. The deck
itself is assembled by PowerPoint (docs/build/build_slides.ps1 drives the installed
application), so Microsoft PowerPoint on Windows is needed to rebuild it.
"""
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from endurosense.config import ROOT, data_path, load_config
from endurosense.data.split import load_split
from endurosense.plots import INK_MUTED, SERIES, apply_style, plt

sys.path.insert(0, str(ROOT / "app"))
import logic as L  # noqa: E402  (the dashboard's logic: the same real cases as the demo)
from endurosense import whatif as W  # noqa: E402

RES, FIN = data_path("results"), data_path("results") / "final"
BUILD = ROOT / "docs" / "build"
FIG = BUILD / "_figures"
OUT = ROOT / "docs" / "EnduroSense_Slides.pptx"
BLUE, DARK, GREY, LIGHT, GREEN, RED = "1F5FAE", "0B0B0B", "52514E", "EAF1FB", "0C7A0C", "C0392B"
W_IN, H_IN, M = 13.333, 7.5, 0.6
pct = lambda x, d=1: f"{100 * x:.{d}f}%"


# ------------------------------------------------------------------ charts
def style():
    apply_style()
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 14,
                         "legend.fontsize": 13, "savefig.dpi": 160})


def save(fig, name) -> tuple[str, float]:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path)
    w, h = fig.get_size_inches()
    plt.close(fig)
    return str(path), h / w


def hbars(labels, values, highlight, xlabel, fmt, name, size=(8.6, 3.6)):
    fig, ax = plt.subplots(figsize=size)
    y = np.arange(len(labels))
    ax.barh(y, values, color=[SERIES[0] if i == highlight else "#9a9992" for i in range(len(labels))], height=0.62)
    for yi, v in zip(y, values):
        ax.text(v + max(values) * 0.015, yi, fmt(v), va="center", fontsize=15, color="#0b0b0b")
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlabel(xlabel); ax.set_xlim(0, max(values) * 1.16)
    ax.grid(axis="y", visible=False)
    return save(fig, name)


def three_missions_chart(case, world, tau, name):
    lv, q, q_b, missions, a = world["lv"], world["q"], world["q_b"], world["missions"], world["a"]
    state = a.loc[case["state"].iloc[0]]
    qa = q.loc[state.name].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.1), sharex=True)
    for ax, (_, r) in zip(axes, case.iterrows()):
        qb = q_b[int(r["mission"])]
        for row, (qq, col) in enumerate(((qa, SERIES[0]), (qb, SERIES[1]))):
            for lo, hi, alpha in ((0.01, 0.99, 0.22), (0.05, 0.95, 0.45), (0.25, 0.75, 0.9)):
                ax.barh(row, np.interp(hi, lv, qq) - np.interp(lo, lv, qq), left=np.interp(lo, lv, qq), color=col, alpha=alpha, height=0.5)
            ax.plot([np.interp(0.5, lv, qq)] * 2, [row - 0.32, row + 0.32], color="#0b0b0b", lw=2.5)
        ax.scatter([state["remaining_wh"], missions["true_wh"].iloc[int(r["mission"])]], [0, 1], marker="D", s=70, color="white", edgecolor="#0b0b0b", zorder=5, linewidth=1.8)
        ax.set_yticks([0, 1], ["Battery\ncan give", "Mission\nneeds"] if ax is axes[0] else ["", ""]); ax.invert_yaxis()
        ax.set_title(f"P(success) {r['P3']:.0%}", color="#0b0b0b"); ax.set_xlabel("energy (Wh)"); ax.grid(axis="y", visible=False)
    return save(fig, name)


def charts(n, world, case, tau) -> dict:
    style()
    out = {}
    allp, a, fleet = n["allp"], n["a"], n["fleet"]
    out["unsafe"] = hbars(["Minutes left\n(the brief)", "Energy,\nbest estimates", "EnduroSense"],
                          [100 * allp.loc[k, "unsafe_approval_rate"] for k in (n["P1"], n["P2"], n["P3"])], 2,
                          "missions approved that would have cut into the reserve (%)", lambda v: f"{v:.1f}%" if v >= 1 else f"{v:.2f}%", "unsafe.png")
    models = ["GRU ensemble, calibrated (main)", "LSTM", "GRU", "Voltage lookup", "Energy counting (BMS)"]
    labels = ["GRU ensemble\n(main model)", "LSTM", "GRU (single)", "Voltage lookup\n(no ML)", "Energy counting\n(no ML)"]
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    y = np.arange(len(models))
    ax.barh(y - 0.2, a.loc[models, "test_mae_supported"], height=0.38, color=SERIES[0], label="test (unseen batteries)")
    ax.barh(y + 0.2, a.loc[models, "cv_mae_supported"], height=0.38, color="#9a9992", label="cross-validation (development)")
    for yi, m in zip(y, models):
        ax.text(a.loc[m, "test_mae_supported"] + 0.05, yi - 0.2, f"{a.loc[m, 'test_mae_supported']:.2f}", va="center", fontsize=13)
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlabel("mean absolute error (Wh), lower is better")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, columnspacing=1.2)
    ax.grid(axis="y", visible=False)
    out["model_a"] = save(fig, "model_a.png")
    d = pd.read_csv(FIN / "model_b_test_flights.csv")
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    lim = [d[["total_wh", "predicted_wh"]].min().min() - 1, d[["total_wh", "predicted_wh"]].max().max() + 1]
    ax.plot(lim, lim, color=INK_MUTED, lw=1, ls="--")
    for seen, col, lab in ((True, SERIES[0], "trained route"), (False, SERIES[1], "unseen routes")):
        g = d[d["seen_route"] == seen]
        ax.scatter(g["total_wh"], g["predicted_wh"], s=55, color=col, label=lab, edgecolor="white", linewidth=0.8)
    ax.set_xlabel("measured mission energy (Wh)"); ax.set_ylabel("predicted (Wh)"); ax.legend(loc="upper left")
    out["model_b"] = save(fig, "model_b.png")
    order = ["P1 minutes left (brief)", "P2 energy, point estimates", "P2 + margin tuned on development data", f"P3 EnduroSense (tau = {tau})", "P4 oracle"]
    out["fleet"] = hbars(["Minutes left\n(the brief)", "Energy,\nbest estimates", "Best estimates\n+ fixed margin", "EnduroSense", "Oracle\n(knows the truth)"],
                         [fleet.loc[k, "unsafe_per_100_missions"] for k in order], 3, "unsafe missions per 100 flown", lambda v: f"{v:.1f}" if v >= 1 else f"{v:.2f}",
                         "fleet.png", size=(8.6, 4.4))
    rel = n["rel"]
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot([0, 100], [0, 100], color=INK_MUTED, lw=1, ls="--")
    ax.plot(100 * rel["mean_predicted"], 100 * rel["observed_success"], "o-", color=SERIES[0], lw=2.2, ms=7)
    ax.set_xlabel("predicted P(success) (%)"); ax.set_ylabel("missions that succeeded (%)")
    out["reliability"] = save(fig, "reliability.png")
    out["three"] = three_missions_chart(case, world, tau, "three_missions.png")
    return out


# ------------------------------------------------------------------ slide elements (inches)
def text(x, y, w, h, s, size=20, bold=False, color=DARK, align="left", bullets=False, after=8):
    """``**bold**`` spans become bold ranges; lines are separated by newlines."""
    plain, ranges, pos = "", [], 1
    for i, chunk in enumerate(s.replace("\n", "\r").split("**")):
        if i % 2 == 1 and chunk:
            ranges.append([pos, len(chunk)])
        plain += chunk
        pos += len(chunk)
    return {"type": "text", "x": x, "y": y, "w": w, "h": h, "text": plain, "size": size, "bold": bold, "color": color, "align": align,
            "bullets": bullets, "after": after, "bold_ranges": ranges}


def image(x, y, w, fig):
    path, aspect = fig
    return {"type": "image", "x": x, "y": y, "w": w, "h": w * aspect, "path": path}


def box(x, y, w, h, s, fill=LIGHT, color=DARK, size=18, bold=False, shape="round"):
    return {"type": "box", "x": x, "y": y, "w": w, "h": h, "text": s, "fill": fill, "color": color, "size": size, "bold": bold, "shape": shape}


def table(x, y, w, h, rows, widths, size=15):
    return {"type": "table", "x": x, "y": y, "w": w, "h": h, "rows": rows, "widths": widths, "size": size}


def slide(title, elements, notes=""):
    return {"title": title, "elements": elements, "notes": notes}


def big(x, y, w, value, label, color=BLUE):
    return [text(x, y, w, 0.9, value, size=44, bold=True, color=color), text(x, y + 0.95, w, 0.9, label, size=16, color=GREY)]


# ------------------------------------------------------------------ content
def numbers():
    cfg = load_config()
    tau = cfg["decision"]["tau"]
    pts = pd.read_csv(FIN / "operating_points.csv")
    pts = pts[pts["margins"] == "tuned on development data"].assign(policy=lambda d: d["policy"].str.strip())
    allp, pre = (pts[pts["pairs"] == s].set_index("policy") for s in ("all pairs", "pre-flight states"))
    b = pd.read_csv(FIN / "model_b_test_metrics.csv")
    return {"cfg": cfg, "tau": tau, "allp": allp, "pre": pre,
            "P1": "P1 minutes left, as in the brief (ratio >= 1)", "P2": "P2 energy point estimates (margin >= 0 Wh)", "P3": f"P3 EnduroSense, tau = {tau}",
            "a": pd.read_csv(FIN / "model_a_test_metrics.csv").set_index("model"), "ar": pd.read_csv(FIN / "model_a_test_ranges.csv").iloc[0],
            "bm": b[b["model"] == "Physics-first"].set_index("flights_group"),
            "br": pd.read_csv(FIN / "model_b_test_ranges.csv").set_index("model").loc["Physics-first, calibrated: all test flights"],
            "fleet": pd.read_csv(FIN / "fleet_simulation.csv").set_index("policy"), "rel": pd.read_csv(FIN / "probability_reliability.csv"),
            "crit": pd.read_csv(FIN / "success_criteria.csv"), "dev_a": pd.read_csv(RES / "uncertainty" / "model_a_intervals.csv").set_index("method"),
            "dev_b": pd.read_csv(RES / "uncertainty" / "model_b_intervals.csv").set_index("method"),
            "mission_cv": pd.read_csv(RES / "model_b" / "mission_cv.csv").set_index("model"), "cv": pd.read_csv(RES / "model_a" / "cv_metrics.csv").set_index("model"),
            "log": json.loads((FIN / "test_access_log.json").read_text())}


def demo_world():
    lv = L.levels()
    a, q = L.load_battery_states()
    model = L.load_mission_model()
    missions, q_b, plan = L.load_recorded_missions(model, lv)
    tw = L.typical_flying_power()
    pre = W.preflight_states(a)
    return {"lv": lv, "a": a, "q": q, "missions": missions, "q_b": q_b, "tw": tw, "grid": L.score_grid(pre, q, missions, q_b, lv, tw)}


def deck() -> list:
    n, world = numbers(), demo_world()
    tau, a, ar, bm, br, allp, pre, fleet = n["tau"], n["a"], n["ar"], n["bm"], n["br"], n["allp"], n["pre"], n["fleet"]
    case = L.three_missions(world["grid"], tau)
    figs = charts(n, world, case, tau)
    MAIN = "GRU ensemble, calibrated (main)"
    split = load_split()
    roles = pd.Series(split["flight_role"])
    state = world["a"].loc[case["state"].iloc[0]]
    mis = [world["missions"].iloc[int(m)] for m in case["mission"]]
    f1, f3 = fleet.loc["P1 minutes left (brief)"], fleet.loc[f"P3 EnduroSense (tau = {tau})"]
    band = n["rel"][n["rel"]["band"] == "(0.95, 0.99]"].iloc[0]
    da, db = n["dev_a"].loc["Calibrated, per-reading spread (main)"], n["dev_b"].loc["Per-part errors, replayed jointly (main)"]
    cw = W_IN - 2 * M
    s = []

    s.append(slide("", [
        text(M, 2.2, cw, 1.2, "EnduroSense", size=60, bold=True, color=BLUE),
        text(M, 3.45, cw, 0.8, "Can this battery fly this mission?", size=30, color=DARK),
        text(M, 4.5, cw, 1.0, "Uncertainty-aware mission feasibility prediction for UAVs.\nAn extension of the intern project "
             "“ML-Based UAV Battery / Remaining Flight-Time Prediction”.", size=18, color=GREY)],
        "One sentence: instead of predicting minutes of flight left, EnduroSense predicts whether a given battery can complete a given mission, and how sure it is."))

    s.append(slide("The brief asks for minutes left. A scheduler needs a different answer.", [
        text(M, 1.6, 6.3, 4.8, "**The brief:** predict the remaining flight time of a UAV battery.\n"
             "**The trouble with minutes:** they are computed from the power being drawn now. A drone that is hovering or descending "
             "looks as if it has plenty of time, whatever the next mission needs.\n"
             "**A single number hides how sure it is.** Two batteries can show the same estimate while one is far less certain.", size=20, bullets=True, after=14),
        box(7.4, 1.9, 5.3, 1.5, "Minutes left ≥ mission duration?", fill="F2F1EE", size=22),
        text(7.4, 3.55, 5.3, 0.6, "approves 1 in 8 missions that would cut into the reserve", size=16, color=RED, align="center"),
        box(7.4, 4.5, 5.3, 1.5, "What is the chance this battery has the energy this mission needs?", fill=LIGHT, size=22, bold=True, color=BLUE)],
        f"The 1-in-8 figure is the test-set result: {pct(allp.loc[n['P1'], 'unsafe_approval_rate'])} of missions that would fail are approved by the minutes-left rule."))

    s.append(slide("The idea: predict two energies, each with an honest range", [
        box(M, 1.9, 3.6, 2.0, "Energy available\n(Model A)", fill=LIGHT, size=22, bold=True, color=BLUE),
        text(M, 4.0, 3.6, 1.6, "From battery telemetry: how many watt-hours are left above a safe reserve.", size=16, color=GREY, align="center"),
        box(4.85, 1.9, 3.6, 2.0, "Energy required\n(Model B)", fill="FDEEE6", size=22, bold=True, color="B4491C"),
        text(4.85, 4.0, 3.6, 1.6, "From the mission plan: distance, speed, payload, altitude, wind.", size=16, color=GREY, align="center"),
        box(8.6, 2.55, 0.7, 0.7, "", fill="9A9992", shape="arrow"),
        box(9.45, 1.9, 3.3, 2.0, f"P(success)\n≥ {tau:.0%} ?", fill="E8F5E8", size=24, bold=True, color=GREEN),
        text(9.45, 4.0, 3.3, 1.6, "Approve only if the battery covers the mission with high probability.", size=16, color=GREY, align="center"),
        text(M, 5.9, cw, 0.9, "Energy adds up across the parts of a mission; minutes do not. A range says how sure each prediction is.", size=18, color=DARK)],
        "Reserve: 22.6 V at rest. Energy available means energy above that reserve, so a successful mission ends with the reserve intact."))

    s.append(slide("Data: 209 real flights, and a test set locked away from the start", [
        *big(M, 1.7, 2.8, "209", "flights of a DJI Matrice 100\n(Carnegie Mellon, public)"),
        *big(3.7, 1.7, 2.8, "90", "battery chains: flights linked\nby battery from rest voltages"),
        *big(6.8, 1.7, 2.8, str(len(split["test_chains"])), f"chains ({int((roles == 'test').sum())} flights) locked away\nbefore any modelling"),
        *big(9.9, 1.7, 2.8, "4", "success criteria written down\nbefore the test set was opened"),
        text(M, 4.5, cw, 2.3, "**Never split a battery:** readings from one battery are not independent, so a battery is entirely in training or entirely in test.\n"
             "**The test set includes routes never flown in development,** among them the longest.\n"
             "**Data problems found and fixed:** start times sorted as text, four mis-logged times, a sign error in the dataset’s documentation.", size=18, bullets=True, after=10)],
        "The chain reconstruction matters: without it, the same battery would appear in both training and test, and every result would be optimistic."))

    s.append(slide("Model A: energy available", [
        image(M, 1.5, 7.4, figs["model_a"]),
        text(8.3, 1.6, 4.5, 5.2, f"**All five algorithms in the brief compared,** plus three baselines that need no machine learning.\n"
             f"**Main model: five GRUs averaged.** {a.loc[MAIN, 'test_mae_supported']:.2f} Wh error on unseen batteries, against "
             f"{a.loc['Voltage lookup', 'test_mae_supported']:.2f} Wh for the best non-ML method.\n"
             f"**A single GRU won in development but not on test** ({a.loc['GRU', 'test_mae_supported']:.2f} Wh). The ensemble is what held up.\n"
             "**Fast enough for on-board use:** about 5 ms per prediction on a laptop CPU.", size=17, bullets=True, after=10)],
        "Error is in watt-hours; a pack has about 65 Wh usable above the reserve, so 2.2 Wh is about 3%."))

    s.append(slide("Model B: energy a mission needs", [
        image(M, 1.55, 5.6, figs["model_b"]),
        text(6.6, 1.6, 6.2, 5.2, "**Built from the mission’s parts:** climb, each cruise leg (power × time), hover, descent, ground.\n"
             "**Physics first, ML corrects it:** rotor physics gives the shape, XGBoost fixes what physics misses.\n"
             f"**{bm.loc['all test flights', 'mape_pct']:.1f}% error on unseen flights:** {bm.loc['route seen in development', 'mape_pct']:.1f}% on the trained route, "
             f"{bm.loc['unseen routes', 'mape_pct']:.1f}% on routes never seen.\n"
             "**On the longest unseen route it over-predicts** (the safe direction) and misses our 5% target.", size=17, bullets=True, after=10)],
        "Only planning-time inputs are used: distance, commanded speed, payload, altitude, expected wind."))

    s.append(slide("Honest ranges: how sure is each prediction?", [
        text(M, 1.6, 6.0, 2.4, "A “90% range” should contain the truth about 90% of the time on data the model has not seen. "
             "Both ranges are set from the models’ errors on held-out batteries and flights, with every battery counted equally.", size=19),
        table(M, 4.0, 6.0, 2.2, [["90% range covers", "Development", "Test"],
                                 ["Energy available", pct(da["cov90"]), pct(ar["cov90"])],
                                 ["Energy required", pct(db["cov90"]), pct(br["cov90"])]], [2.8, 1.6, 1.6], size=17),
        image(7.3, 1.5, 5.3, figs["reliability"]),
        text(7.3, 6.15, 5.3, 0.8, f"P(success) against what happened, on test. Missions rated 95–99% succeeded {pct(band['observed_success'], 0)}.", size=14, color=GREY)],
        "Model A's ranges held on test. Model B's were too narrow on new data (80% against 90%): a real weakness, reported as such."))

    labels = [m["label"] for m in mis]
    s.append(slide("Three missions, one battery: the brief’s rule approves all three", [
        text(M, 1.45, cw, 0.9, f"An unseen battery before take-off (chain {int(state['battery_chain'])}). The model says {np.interp(0.5, world['lv'], world['q'].loc[state.name]):.0f} Wh "
             f"above reserve; {state['remaining_wh']:.0f} Wh were really there. Diamonds show what was measured.", size=17, color=GREY),
        image(M, 2.3, cw, figs["three"]),
        *[box(M + 0.33 + i * 4.12, 5.55, 3.4, 0.6, ("GO" if r["P3"] >= tau else "NO-GO") + ": EnduroSense",
              fill="E8F5E8" if r["P3"] >= tau else "FBE9E7", color=GREEN if r["P3"] >= tau else RED, size=18, bold=True) for i, (_, r) in enumerate(case.iterrows())],
        *[text(M + 0.33 + i * 4.12, 6.2, 3.4, 0.8, ("needed {:.0f} Wh: fine".format(m["true_wh"]) if r["feasible"] else
                                                   "needed {:.0f} Wh: {:.0f} Wh into the reserve".format(m["true_wh"], -r["true_margin_wh"])),
               size=15, color=GREY, align="center") for i, ((_, r), m) in enumerate(zip(case.iterrows(), mis))]],
        "Real test data, picked by a fixed rule. Missions: " + " | ".join(labels) + ". The minutes-left rule says GO for all three."))

    s.append(slide("Result: unsafe approvals fall from 12% to under 1%", [
        image(M, 1.5, 8.0, figs["unsafe"]),
        *big(9.0, 1.5, 3.8, pct(allp.loc[n["P3"], "unsafe_approval_rate"], 2), "of failing missions approved by EnduroSense\n(test set, all battery states)"),
        *big(9.0, 3.6, 3.8, pct(pre.loc[n["P3"], "unsafe_approval_rate"], 2), "at take-off decisions", color=GREEN),
        text(M, 5.3, cw, 1.6, f"**The price is caution:** it refuses {pct(allp.loc[n['P3'], 'wasted_refusal_rate'], 0)} of missions that would have succeeded, mostly those within a few Wh of the limit.\n"
             "**Minutes-left cannot be rescued with a safety margin:** to be as safe it must refuse 9 in 10 feasible missions.", size=17, bullets=True, after=8)],
        "Unsafe approval = a mission approved that would have cut into the reserve, as a share of all such missions. Evaluated by pairing real unseen battery states with real unseen missions."))

    s.append(slide("In a simulated fleet on unseen batteries", [
        image(M, 1.5, 7.8, figs["fleet"]),
        *big(8.9, 1.6, 3.9, f"{int(f1['unsafe_per_100_missions'] / f3['unsafe_per_100_missions'])}×", "fewer unsafe missions than the brief’s rule"),
        text(8.9, 3.7, 3.9, 3.0, f"4 drones, 40 tasks a day, 200 days.\n{int(f3['batteries'])} real unseen batteries; tasks are real recorded flights.\n"
             f"Cost: {100 * (f3['swaps_per_day'] / f1['swaps_per_day'] - 1):.0f}% more battery swaps.", size=17, color=GREY, after=10)],
        "Only the pairing of batteries with tasks is simulated. Best estimates plus a fixed margin tuned in development does about as well as EnduroSense here."))

    s.append(slide("What did not hold up", [
        text(M, 1.6, cw, 4.3,
             f"**Mission model on unseen, longer routes:** {bm.loc['unseen routes', 'mape_pct']:.1f}% error against a 5% target (over-predicted, the safe direction).\n"
             f"**Mission model’s ranges:** {pct(br['cov90'], 0)} coverage on test against a 90% target. Too narrow on new data.\n"
             "**The probability against a well-tuned fixed margin:** no measurable difference over all cases. The probability refused fewer "
             "feasible missions at take-off and needs no tuning.\n"
             "**Risk is per battery:** all of EnduroSense’s unsafe approvals came from two batteries whose capacity the model over-estimated.", size=19, bullets=True, after=14),
        box(M, 6.0, cw, 0.9, "Two of four pre-set success criteria met. The misses are reported, not patched.", fill="F2F1EE", size=19, bold=True)],
        "These weaknesses were found on the test set. Fixing them and re-scoring on the same test set would make the numbers dishonest, so they are reported as they are."))

    s.append(slide("Why the numbers can be trusted", [
        text(M, 1.6, cw, 5.2, "**A locked test set,** chosen before any modelling. Every opening is logged: the evaluation, and one rerun after verification that gave identical numbers.\n"
             "**Success criteria written down first,** so they could not be adjusted after seeing results.\n"
             "**One command rebuilds everything** from the raw data in 30 minutes. A rebuild with empty caches reproduced every result and model file byte for byte.\n"
             "**Five verification passes:** headline numbers recomputed with separate code, a search for leakage, and deliberate bugs planted to confirm the tests catch them.\n"
             "**Errors found were written up,** including the ones in our own reporting.", size=19, bullets=True, after=14)],
        "The verification log (docs/verification_log.md) lists what each pass found and fixed."))

    s.append(slide("Demo and next steps", [
        text(M, 1.6, 5.8, 0.6, "Live dashboard", size=22, bold=True, color=BLUE),
        text(M, 2.3, 5.8, 4.0, "**Three missions, one battery:** the case on the earlier slide, live.\n"
             "**Mission check:** any unseen battery, any planned or recorded mission.\n"
             "**Fleet:** the simulation with adjustable fleet, workload and threshold.\n"
             "**Results:** every table in this talk, read from the result files.", size=18, bullets=True, after=10),
        text(6.9, 1.6, 5.9, 0.6, "Next steps", size=22, bold=True, color=BLUE),
        text(6.9, 2.3, 5.9, 4.4, "More batteries and longer routes: the two missed criteria are data limits.\n"
             "Learn each battery’s capacity as it flies, to remove the per-battery risk.\n"
             "Widen the mission range outside trained routes.\n"
             "Time the model on on-board hardware.\n"
             "Confirm the reserve and threshold with the guide.", size=18, bullets=True, after=10)],
        "Run: streamlit run app/streamlit_app.py"))
    return s


def main() -> None:
    slides = deck()
    spec = {"out": str(OUT), "width_in": W_IN, "height_in": H_IN, "margin": M, "title_color": BLUE, "footer": "EnduroSense", "slides": slides,
            "preview_dir": str(BUILD / "_preview")}
    (BUILD / "_slides_spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(BUILD / "build_slides.ps1"),
                        str(BUILD / "_slides_spec.json")], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    if r.returncode != 0:
        sys.exit(r.stderr)


if __name__ == "__main__":
    main()
