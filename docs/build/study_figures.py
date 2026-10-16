"""Simple diagrams for the study guide (docs/study_guide.md).

Usage: python docs/build/study_figures.py        -> docs/build/_figures/study/*.png

Nine plain pictures: the pipeline, battery chains, the split, the label, the
inside of Model A, how its range is made, the parts of a mission, P(success),
and the two kinds of mistake.

Where a picture uses recorded data it uses development flights only. The one
exception is the worked example (battery chain 86), which the final evaluation
already published and the dashboard shows.
"""
import importlib.util
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from endurosense.config import ROOT, data_path, load_config
from endurosense.data.load import load_processed
from endurosense.data.split import load_split, select
from endurosense.features import model_a as MA
from endurosense.mission import MissionSpec
from endurosense.plots import INK, INK_MUTED, PHASE_COLORS, SERIES, apply_style

sys.path.insert(0, str(ROOT / "app"))
import logic as L  # noqa: E402

BUILD = ROOT / "docs" / "build"
FIG = BUILD / "_figures" / "study"
BLUE, ORANGE, GREEN, AMBER, RED, GREY = SERIES[0], SERIES[1], SERIES[2], SERIES[3], SERIES[7], "#8a8982"
W_IN = 7.2                                  # every diagram is this wide, so text prints at a readable size
PT = W_IN * 72 / 100                        # points per drawing unit (drawings are 100 units wide)

LINK_FLIGHTS = list(range(91, 99))          # eight development flights flown one after another on 19 June 2019
LABEL_CHAIN = 25                            # a development battery flown from full to below the reserve
EXAMPLE_LOOP = {"legs_m": [140, 198, 119], "payload_g": 250, "speed": 8, "altitude_m": 50}
MEMBERS = [(36.9, 3.1), (38.8, 3.6), (37.2, 3.3), (39.4, 3.9), (37.7, 3.4)]   # illustrative, not model output


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), BUILD / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ drawing helpers
def canvas(height_in: float):
    """A blank drawing 100 units wide, with the same unit vertically."""
    apply_style()
    fig, ax = plt.subplots(figsize=(W_IN, height_in))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100 * height_in / W_IN); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, sub="", color=BLUE, size=10, sub_size=8.4, fill=0.14) -> None:
    for face, edge, alpha in ((color, "none", fill), ("none", color, 1.0)):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", facecolor=face, edgecolor=edge,
                                    alpha=alpha, linewidth=1.2))
    lines = sub.count("\n") + 1 if sub else 0
    block = (size * 1.3 + lines * sub_size * 1.25) / PT
    top = y + h / 2 + block / 2
    ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=size, fontweight="bold", color=INK)
    if sub:
        ax.text(x + w / 2, top - size * 1.45 / PT, sub, ha="center", va="top", fontsize=sub_size, color=INK_MUTED, linespacing=1.2)


def arrow(ax, x0, y0, x1, y1, color=INK_MUTED) -> None:
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.4, "shrinkA": 0, "shrinkB": 0, "mutation_scale": 11})


def save(fig, name: str):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / f"{name}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ the numbers behind the pictures
def worked_example() -> dict:
    """The three-missions case of the dashboard: one battery's range and three missions' ranges."""
    slides = load("build_slides.py")
    world, tau = slides.demo_world(), L.tau_default()
    case = L.three_missions(world["grid"], tau)
    lv = world["lv"]
    at = lambda q, p: float(np.interp(p, lv, q))
    qa = world["q"].loc[case["state"].iloc[0]].to_numpy()
    missions = []
    for _, r in case.iterrows():
        m = world["missions"].iloc[int(r["mission"])]
        qb = world["q_b"][int(r["mission"])]
        name = f"Flight {m['flights'][0]}" if len(m["flights"]) == 1 else "Flights " + " + ".join(str(f) for f in m["flights"])
        missions.append({"name": name, "p": float(r["P3"]), "go": bool(r["P3"] >= tau), "true_wh": float(m["true_wh"]),
                         "range": [at(qb, p) for p in (0.05, 0.5, 0.95)]})
    return {"tau": tau, "available": [at(qa, p) for p in (0.05, 0.5, 0.95)], "true_available": float(case["available_wh"].iloc[0]),
            "missions": missions}


def example_mission() -> dict:
    """Model B's parts and range for a loop like route R1."""
    model = L.load_mission_model()
    spec = MissionSpec.loop(EXAMPLE_LOOP["legs_m"], EXAMPLE_LOOP["payload_g"], EXAMPLE_LOOP["speed"], EXAMPLE_LOOP["altitude_m"])
    parts = model.model.mission(spec)
    lo, hi = model.distribution(spec).interval()
    return {"climb": parts["climb_wh"], "legs": list(parts["leg_wh"]), "hover": parts["hover_wh"], "descent": parts["descent_wh"],
            "ground": parts["ground_wh"], "total": parts["total_wh"], "low": lo, "high": hi}


def label_example() -> dict:
    """Resting voltage against energy drawn for one development battery, and where it meets the reserve."""
    flights = load_processed("flights")
    dev = select(flights, "dev")
    curve = MA.fit_discharge_curve(flights, dev["flight"])
    e, v, _ = MA.rest_points(dev[dev["battery_chain"] == LABEL_CHAIN])
    reserve = load_config()["battery"]["reserve_v"]
    xs, ys = [], []
    for k in range(len(e) - 1):                                    # between resting points, follow the fitted discharge curve
        grid = np.linspace(v[k], v[k + 1], 60)
        frac = np.array([curve.energy_between(v[k], g) for g in grid]) / curve.energy_between(v[k], v[k + 1])
        xs.extend(e[k] + frac * (e[k + 1] - e[k])); ys.extend(grid)
    xs, ys = np.array(xs), np.array(ys)
    return {"e": e, "v": v, "curve_x": xs, "curve_y": ys, "reserve": reserve, "e_reserve": float(np.interp(-reserve, -ys, xs))}


def combined(members=MEMBERS) -> tuple[float, float]:
    """Estimate and spread of several networks: the average, and their own spreads plus their disagreement."""
    mu, sd = np.array(members).T
    return float(mu.mean()), float(np.sqrt((sd ** 2 + mu ** 2).mean() - mu.mean() ** 2))


def multipliers() -> pd.Series:
    """Measured calibration multipliers of Model A, by probability level."""
    m = pd.read_csv(data_path("results") / "uncertainty" / "model_a_calibration_multipliers.csv")
    return m.set_index(m["level"].round(3))["q_hat_sigmas"]


# ------------------------------------------------------------------ the nine pictures
def pipeline(ex: dict):
    fig, ax = canvas(4.4)
    a, b = ex["available"], ex["missions"][2]["range"]
    ax.text(0.5, 59.5, "Step 1. Prepare the data", fontsize=9.5, fontweight="bold", color=INK_MUTED, va="center")
    row1 = [("Flight data", "209 flights"), ("Clean", "phases, energy"), ("Battery chains", "90 batteries"), ("Split by battery", "74 to build, 16 to test")]
    for k, (title, sub) in enumerate(row1):
        box(ax, 0.5 + 26 * k, 45.5, 21, 11, title, sub, color=GREY)
        if k:
            arrow(ax, 26 * k - 4, 51, 26 * k, 51)
    ax.text(0.5, 41, "Step 2. Predict two energies, combine them, decide", fontsize=9.5, fontweight="bold", color=INK_MUTED, va="center")
    lanes = [(26, BLUE, [("Battery readings", "voltage, current"), ("Model A", "5 GRU networks"),
                         ("Energy available", f"{a[1]:.0f} Wh ({a[0]:.0f} to {a[2]:.0f})")]),
             (8, ORANGE, [("Mission plan", "legs, speed, payload"), ("Model B", "physics + XGBoost"),
                          ("Energy required", f"{b[1]:.0f} Wh ({b[0]:.0f} to {b[2]:.0f})")])]
    for y, col, items in lanes:
        for k, (title, sub) in enumerate(items):
            box(ax, 0.5 + 25 * k, y, 20.5, 12, title, sub, color=col)
            if k:
                arrow(ax, 25 * k - 4, y + 6, 25 * k, y + 6)
    box(ax, 77, 15, 22.5, 17, "P(success)", "chance that available\nis at least required", color=GREEN)
    arrow(ax, 71, 32, 76.5, 27); arrow(ax, 71, 14, 76.5, 20)
    box(ax, 77, 1, 22.5, 9.5, f"GO if {ex['tau']:.0%} or more", color=GREEN)
    arrow(ax, 88.2, 15, 88.2, 11)
    return save(fig, "pipeline")


def chains():
    apply_style()
    f = load_processed("flights")
    f = select(f[f["flight"].isin(LINK_FLIGHTS)], "dev").sort_values("start_time").reset_index(drop=True)
    assert len(f) == len(LINK_FLIGHTS)
    lo, hi = load_config()["chains"]["link_window_v"]
    fig, ax = plt.subplots(figsize=(W_IN, 3.1))
    for k, r in f.iterrows():
        ax.plot([k - 0.28, k + 0.28], [r["v_rest_start"], r["v_rest_end"]], color=BLUE, lw=3.2, solid_capstyle="round")
        if k + 1 < len(f):
            nxt = f.loc[k + 1, "v_rest_start"]
            jump = nxt - r["v_rest_end"]
            same = lo <= jump <= hi
            ax.plot([k + 0.28, k + 0.72], [r["v_rest_end"], nxt], color=GREEN if same else RED, lw=1.6, ls="-" if same else "--")
            ax.text(k + (0.5 if same else 0.42), (r["v_rest_end"] + nxt) / 2 + (0.2 if same else 0), f"{jump:+.1f} V", ha="center" if same else "right",
                    va="bottom", fontsize=8, color=INK)
    for n, (ch, g) in enumerate(f.groupby("battery_chain", sort=False)):
        ax.axvspan(g.index.min() - 0.5, g.index.max() + 0.5, color=INK_MUTED, alpha=0.05 if n % 2 else 0.11, lw=0)
        ax.text(g.index.to_numpy().mean(), 26.15, f"battery {n + 1}", ha="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.plot([], [], color=BLUE, lw=3.2, label="during a flight: resting voltage falls")
    ax.plot([], [], color=GREEN, lw=1.6, label=f"small recovery ({lo:+.1f} to {hi:+.1f} V): same battery")
    ax.plot([], [], color=RED, lw=1.6, ls="--", label="big jump up: a fresh battery was put in")
    ax.legend(loc="lower left", fontsize=8, ncol=1, bbox_to_anchor=(0.0, -0.02))
    ax.set_xticks(range(len(f)), [f"flight\n{n}" for n in f["flight"]], fontsize=8)
    ax.set_ylabel("resting voltage (V)"); ax.set_ylim(21.2, 26.5); ax.set_xlim(-0.5, len(f) - 0.5); ax.grid(axis="x", visible=False)
    return save(fig, "chains")


def split():
    fig, ax = canvas(3.1)
    s = load_split()
    flights = load_processed("flights").set_index("flight")["battery_chain"]
    fold_of_chain = pd.Series({flights[int(fl)]: k for fl, k in s["dev_fold"].items()})
    sizes = fold_of_chain.value_counts().sort_index()
    n_dev, n_test = int(sizes.sum()), len(s["test_chains"])
    x = 0.5
    for k, n in sizes.items():
        w = 78.5 * n / n_dev
        box(ax, x, 33.5, w - 0.8, 7, f"fold {k + 1}", f"{n} batteries", color=BLUE, size=8.6, sub_size=7.8)
        x += w
    box(ax, 81, 33.5, 18.5, 7, "test: locked", f"{n_test} batteries", color=GREY, size=8.6, sub_size=7.8, fill=0.3)
    ax.text(0.5, 29, f"Cross-validation: five rounds on the {n_dev} development batteries", fontsize=9.5, fontweight="bold", color=INK_MUTED, va="center")
    for r in range(5):
        y = 21.5 - 5 * r
        ax.text(0.5, y + 1.9, f"Round {r + 1}", fontsize=8.6, color=INK, va="center")
        for j in range(5):
            hit = j == r
            box(ax, 11 + 13.6 * j, y, 12.8, 3.8, "predict" if hit else "train", color=ORANGE if hit else BLUE, size=8.2, fill=0.45 if hit else 0.1)
    ax.text(90.2, 12, "Every battery is\npredicted once,\nby a model that\nnever saw it", ha="center", va="center", fontsize=9, color=INK, linespacing=1.3)
    return save(fig, "split")


def label(lab: dict):
    apply_style()
    e, v, res, e_res = lab["e"], lab["v"], lab["reserve"], lab["e_reserve"]
    fig, ax = plt.subplots(figsize=(W_IN, 3.1))
    ax.plot(lab["curve_x"], lab["curve_y"], color=BLUE, lw=2)
    ax.scatter(e, v, s=42, color=BLUE, zorder=5, edgecolor="white", linewidth=1, label="resting voltage, measured between flights")
    ax.axhline(res, color=RED, lw=1.4, ls="--")
    ax.text(1, res + 0.07, f"reserve: {res} V", color=RED, fontsize=9, va="bottom")
    ax.plot([e_res, e_res], [res, v[1]], color=INK_MUTED, lw=1, ls=":")
    ax.scatter([e_res], [res], s=70, color=RED, zorder=6, marker="D", edgecolor="white", linewidth=1)
    ax.annotate(f"reserve reached after {e_res:.0f} Wh", xy=(e_res, res), xytext=(e_res - 2, res - 0.42), fontsize=9, color=INK, ha="right",
                arrowprops={"arrowstyle": "-", "color": INK_MUTED, "lw": 0.8})
    for x0, x1, text in ((0, e[1], f"drawn so far: {e[1]:.0f} Wh"), (e[1], e_res, f"energy left: {e_res - e[1]:.0f} Wh")):
        ax.annotate("", xy=(x1, v[1]), xytext=(x0, v[1]), arrowprops={"arrowstyle": "<->", "color": INK, "lw": 1.1, "shrinkA": 3, "shrinkB": 3})
        ax.text((x0 + x1) / 2, v[1] + 0.07, text, ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xlabel("energy drawn from the battery since it was put in (Wh)"); ax.set_ylabel("resting voltage (V)")
    ax.set_xlim(-2, e[-1] + 4); ax.set_ylim(21.9, 26.0); ax.legend(loc="upper right", fontsize=8.5)
    return save(fig, "label")


def model_a():
    fig, ax = canvas(3.2)
    box(ax, 0.5, 28, 24, 14, "Last 60 seconds", "120 readings of voltage,\ncurrent, power, energy")
    box(ax, 30.5, 28, 22, 14, "GRU, 2 layers", "reads them in order,\nkeeps a memory")
    box(ax, 0.5, 3, 24, 14, "25 features", "now, last 10 s and 30 s,\nresting voltages, energy")
    box(ax, 58.5, 15.5, 15, 14, "Join", "64 + 25\nnumbers", color=GREY)
    box(ax, 79.5, 28, 20, 14, "Estimate", "energy left (Wh)", color=GREEN)
    box(ax, 79.5, 3, 20, 14, "Spread", "how unsure (± Wh)", color=AMBER)
    arrow(ax, 24.5, 35, 30, 35); arrow(ax, 52.5, 33, 58, 27); arrow(ax, 24.5, 11, 58, 19)
    arrow(ax, 73.5, 26, 79, 32); arrow(ax, 73.5, 19, 79, 13)
    return save(fig, "model_a")


def ensemble():
    fig, ax = canvas(3.0)
    mu, sd = combined()
    q = multipliers()
    lo, hi = mu + sd * q[0.05], mu + sd * q[0.95]
    for k, (m, s) in enumerate(MEMBERS):
        y = 33.5 - 7.3 * k
        box(ax, 0.5, y, 20, 5.6, f"GRU {k + 1}:  {m:.1f} ± {s:.1f}", size=8.8)
        arrow(ax, 20.5, y + 2.8, 27.5, 19 + 0.6 * (2 - k))
    box(ax, 28, 9, 25, 20, "Combine", f"estimate = average = {mu:.1f}\nspread = {sd:.1f}\n(own spreads + disagreement)", color=GREY, sub_size=8)
    box(ax, 58.5, 9, 20, 20, "Calibrate", "stretch the spread\nusing real errors\non unseen batteries", color=AMBER, sub_size=8)
    box(ax, 84, 9, 15.5, 20, "90% range", f"{lo:.1f} to {hi:.1f}\nWh", color=GREEN)
    arrow(ax, 53, 19, 58, 19); arrow(ax, 78.5, 19, 83.5, 19)
    ax.text(99.5, 2, "numbers in Wh, for illustration", ha="right", fontsize=8, color=INK_MUTED, style="italic")
    return save(fig, "ensemble")


def mission(m: dict):
    fig, ax = canvas(2.9)
    base, top = 17, 32
    ax.plot([2, 98], [base, base], color=INK_MUTED, lw=0.8, ls=":")
    xs = [4, 12, 24, 42, 64, 78, 90, 96]
    segs = [("ground", base, base), ("climb", base, top), ("cruise", top, top), ("cruise", top, top), ("cruise", top, top),
            ("descent", top, base), ("ground", base, base)]
    for (phase, y0, y1), x0, x1 in zip(segs, xs[:-1], xs[1:]):
        ax.plot([x0, x1], [y0, y1], color=PHASE_COLORS[phase], lw=4, solid_capstyle="round")
    ax.scatter(xs[2:6], [top] * 4, s=55, color=PHASE_COLORS["hover"], zorder=5, edgecolor="white", linewidth=1)
    for k, (x0, x1) in enumerate(zip(xs[2:5], xs[3:6])):
        ax.text((x0 + x1) / 2, top + 2, f"leg {k + 1}: {m['legs'][k]:.1f} Wh", ha="center", fontsize=9, color=INK)
    ax.text(51, top - 3.2, f"pauses at the corners (hover): {m['hover']:.1f} Wh", ha="center", va="top", fontsize=9, color=INK)
    ax.text(18, base - 2, f"climb\n{m['climb']:.1f} Wh", ha="center", va="top", fontsize=9, color=INK, linespacing=1.2)
    ax.text(84, base - 2, f"descent\n{m['descent']:.1f} Wh", ha="center", va="top", fontsize=9, color=INK, linespacing=1.2)
    ax.text(51, base - 2, f"on the ground, before and after: {m['ground']:.1f} Wh", ha="center", va="top", fontsize=9, color=INK)
    total = (f"climb {m['climb']:.1f}  +  legs {sum(m['legs']):.1f}  +  hover {m['hover']:.1f}  +  descent {m['descent']:.1f}  +  ground {m['ground']:.1f}"
             f"  =  {m['total']:.1f} Wh")
    ax.text(50, 5.2, total, ha="center", fontsize=10, fontweight="bold", color=INK)
    ax.text(50, 1.2, f"90% range: {m['low']:.1f} to {m['high']:.1f} Wh", ha="center", fontsize=9, color=INK_MUTED)
    return save(fig, "mission")


def p_success(ex: dict):
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(W_IN, 2.3), sharex=True)
    for ax, m in zip(axes, ex["missions"]):
        for row, (rng, col) in enumerate(((ex["available"], BLUE), (m["range"], ORANGE))):
            ax.barh(row, rng[2] - rng[0], left=rng[0], color=col, alpha=0.55, height=0.5)
            ax.plot([rng[1]] * 2, [row - 0.3, row + 0.3], color=INK, lw=2)
        ax.set_yticks([0, 1], ["battery\ncan give", "mission\nneeds"] if ax is axes[0] else ["", ""]); ax.invert_yaxis()
        ax.set_title(f"{m['name']}\nP(success) {m['p']:.0%}: {'GO' if m['go'] else 'NO-GO'}", fontsize=9.5, color=INK, loc="center",
                     fontweight="bold" if not m["go"] else "normal")
        ax.set_xlabel("energy (Wh)"); ax.grid(axis="y", visible=False); ax.set_ylim(1.6, -0.6)
    fig.tight_layout(w_pad=1.2)
    return save(fig, "p_success")


def mistakes():
    fig, ax = canvas(2.6)
    ax.text(42, 33, "Mission would succeed", ha="center", va="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(80, 33, "Mission would fail", ha="center", va="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(21, 23, "Rule says GO", ha="right", va="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(21, 9, "Rule says NO-GO", ha="right", va="center", fontsize=9.5, fontweight="bold", color=INK)
    box(ax, 24, 17, 36, 12, "Correct", "mission flown", color=GREEN)
    box(ax, 62, 17, 36, 12, "Unsafe approval", "cuts into the reserve", color=RED, fill=0.22)
    box(ax, 24, 3, 36, 12, "Wasted refusal", "a good mission not flown", color=AMBER, fill=0.22)
    box(ax, 62, 3, 36, 12, "Correct", "mission refused", color=GREEN)
    return save(fig, "mistakes")


def figures() -> dict:
    """Draw every diagram; returns name -> path."""
    ex = worked_example()
    return {"pipeline": pipeline(ex), "chains": chains(), "split": split(), "label": label(label_example()), "model_a": model_a(),
            "ensemble": ensemble(), "mission": mission(example_mission()), "p_success": p_success(ex), "mistakes": mistakes()}


if __name__ == "__main__":
    for name, path in figures().items():
        print(f"{name}: {path}")
