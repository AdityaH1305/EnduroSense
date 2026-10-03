"""What-if evaluation of go / no-go decisions on real data.

Plain idea: we cannot fly new missions, but we can ask "would *this* battery,
in *this* state, have managed *that* recorded mission?" and know the answer:

- a **battery state** is a real held-out reading; its true energy above the
  reserve is known from its chain (Phase 2 labels);
- a **mission** is a real held-out flight, or several flown one after another
  ("sorties"); its true energy is what those flights measured;
- the mission **would have succeeded** if true energy available >= true energy
  required.

Every model prediction used here is out-of-fold: the battery model never saw
that battery, the mission model never saw that flight, and each distribution
is calibrated on other folds only.

Policies (each produces a score; a mission is approved if the score reaches a
threshold, and sweeping the threshold traces a safety/efficiency curve):

- **P1 minutes left** (the brief): predicted minutes left at the *current*
  power draw, divided by the mission's duration. Approve if >= 1 (+ margin).
  The mission's true airborne duration is given to P1, which favours it.
- **P2 energy, point estimates**: predicted energy available minus predicted
  energy required. Approve if >= 0 Wh (+ margin).
- **P3 EnduroSense**: P(success) from both calibrated distributions. Approve
  if >= tau.
- **P4 oracle**: the true margin (the best any policy could do).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from endurosense.config import load_config
from endurosense.feasibility import p_success_batch
from endurosense.uncertainty import mission as UB
from endurosense.uncertainty.quantiles import group_weights

POLICIES = {"P1": "P1 minutes left (brief)", "P2": "P2 energy, point estimates",
            "P3": "P3 EnduroSense P(success)", "P4": "P4 oracle"}


def decision_states(a: pd.DataFrame, every_s: int) -> pd.DataFrame:
    """One battery state every ``every_s`` seconds of each flight."""
    return a[(np.floor(a["time"]).astype(int) % every_s) == 0]


def preflight_states(a: pd.DataFrame) -> pd.DataFrame:
    """The last motors-off reading before each flight's take-off: where a real
    go / no-go decision is made."""
    first_on = a[a["motors_on"] == 1].groupby("flight")["time"].min()
    before = a[(a["motors_on"] == 0) & (a["time"] < a["flight"].map(first_on))]
    return before.loc[before.groupby("flight")["time"].idxmax()]


def typical_power_by_fold(a: pd.DataFrame) -> pd.Series:
    """Typical flying power, from the *other* folds, for the minutes-left conversion."""
    on = a[a["motors_on"] == 1]
    return pd.Series({k: float(on.loc[on["fold"] != k, "p"].median()) for k in np.unique(a["fold"])})


def build_missions(parts: pd.DataFrame, flights_b: pd.DataFrame, levels, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Single-flight missions and multi-sortie missions, with their held-out distributions.

    ``parts`` is Phase 5's ``model_b_oof_parts`` (pred_*, err_*, fold; indexed by flight).
    Multi-sortie missions join flights of the same fold, so one set of "other
    folds" calibrates the whole mission. Returns the mission table and a
    (missions, levels) array of predictive quantiles.
    """
    cfg = load_config()["decision"]
    pred = parts[[c for c in parts if c.startswith("pred_")]].rename(columns=lambda c: c[5:])
    tuples = parts[[c for c in parts if c.startswith("err_")]].rename(columns=lambda c: c[4:])
    fold = parts["fold"]
    f = flights_b.set_index("flight").loc[parts.index]
    air_min = (f["climb_s"] + f["cruise_s"] + f["descent_s"] + f["hover_s"]) / 60.0

    rows = [{"flights": (fl,), "sorties": 1, "fold": int(fold[fl]), "true_wh": float(f.loc[fl, "total_wh"]),
             "duration_min": float(air_min[fl])} for fl in parts.index]
    q = [UB.leave_fold_out_quantiles(pred, tuples, fold, levels)]

    compound_q = []
    for k in cfg["compound_sizes"]:
        for _ in range(cfg["compound_per_size"]):
            fd = int(rng.choice(np.unique(fold)))
            members = tuple(int(x) for x in rng.choice(fold.index[fold == fd], k, replace=False))
            cal = tuples[(fold != fd).to_numpy()]
            w = group_weights(cal["battery_chain"])
            draws = cal.iloc[rng.choice(len(cal), 4 * cfg["mc_samples"], p=w / w.sum())]    # chain-weighted pool
            totals = UB.replay_compound([pred.loc[m].to_dict() for m in members], draws, cfg["mc_samples"], rng)
            compound_q.append(np.quantile(totals, levels))
            rows.append({"flights": members, "sorties": k, "fold": fd, "true_wh": float(f.loc[list(members), "total_wh"].sum()),
                         "duration_min": float(air_min[list(members)].sum())})
    if compound_q:
        q.append(np.array(compound_q))
    missions = pd.DataFrame(rows)
    return missions, np.vstack(q)


def make_pairs(states: pd.DataFrame, missions: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Random battery-state x mission pairs with their true outcome."""
    si, mi = rng.integers(0, len(states), n), rng.integers(0, len(missions), n)
    p = pd.DataFrame({"state": states.index.to_numpy()[si], "mission": mi,
                      "battery_chain": states["battery_chain"].to_numpy()[si],
                      "available_wh": states["remaining_wh"].to_numpy()[si],
                      "required_wh": missions["true_wh"].to_numpy()[mi]})
    p["true_margin_wh"] = p["available_wh"] - p["required_wh"]
    p["feasible"] = p["true_margin_wh"] >= 0
    return p


def policy_scores(pairs: pd.DataFrame, states: pd.DataFrame, q_a: pd.DataFrame, missions: pd.DataFrame,
                  q_b: np.ndarray, levels, typical_w: pd.Series) -> pd.DataFrame:
    """Score of each policy for every pair (higher = more willing to approve)."""
    levels = np.asarray(levels, float)
    mid = int(np.argmin(np.abs(levels - 0.5)))
    s = states.loc[pairs["state"]]
    qa, qb = q_a.loc[pairs["state"]].to_numpy(), q_b[pairs["mission"].to_numpy()]
    med_a, med_b = qa[:, mid], qb[:, mid]
    flying = (s["motors_on"].to_numpy() == 1) & (s["p_mean_30s"].to_numpy() >= 100)
    power = np.where(flying, s["p_mean_30s"].to_numpy(), s["fold"].map(typical_w).to_numpy())
    minutes_left = med_a / power * 60.0
    return pd.DataFrame({
        "P1": minutes_left / missions["duration_min"].to_numpy()[pairs["mission"].to_numpy()],
        "P2": med_a - med_b,
        "P3": p_success_batch(qa, qb, levels),
        "P4": pairs["true_margin_wh"].to_numpy(),
    }, index=pairs.index)


def rates(approve: np.ndarray, feasible: np.ndarray) -> dict:
    """Decision quality at one operating point."""
    approve, feasible = np.asarray(approve, bool), np.asarray(feasible, bool)
    n_inf, n_fea, n_app = (~feasible).sum(), feasible.sum(), approve.sum()
    return {"approved_share": float(approve.mean()),
            "unsafe_approval_rate": float((approve & ~feasible).sum() / max(n_inf, 1)),      # of missions that would fail
            "wasted_refusal_rate": float((~approve & feasible).sum() / max(n_fea, 1)),       # of missions that would succeed
            "unsafe_share_of_approvals": float((approve & ~feasible).sum() / max(n_app, 1))}


def tradeoff_curve(score: np.ndarray, feasible: np.ndarray) -> pd.DataFrame:
    """Unsafe-approval rate vs wasted-refusal rate as the threshold moves."""
    fpr, tpr, thr = roc_curve(feasible, score)
    return pd.DataFrame({"threshold": thr, "unsafe_approval_rate": fpr, "wasted_refusal_rate": 1 - tpr})


def approved_at_unsafe(score: np.ndarray, feasible: np.ndarray, max_unsafe: float) -> tuple[float, float]:
    """Best share of feasible missions approved while the unsafe-approval rate
    stays at or below ``max_unsafe``; also returns the threshold that achieves it."""
    fpr, tpr, thr = roc_curve(feasible, score)
    ok = fpr <= max_unsafe
    i = int(np.flatnonzero(ok)[np.argmax(tpr[ok])])
    return float(tpr[i]), float(thr[i])


def summary_table(scores: pd.DataFrame, pairs: pd.DataFrame, unsafe_levels=(0.01, 0.02, 0.05),
                  n_boot: int = 300, seed: int = 42) -> pd.DataFrame:
    """Per policy: ranking quality (AUC) and the share of feasible missions approved
    at fixed unsafe-approval rates, with 95% intervals from resampling battery chains."""
    feasible = pairs["feasible"].to_numpy()
    chains = pairs["battery_chain"].to_numpy()
    uniq = np.unique(chains)
    by_chain = {c: np.flatnonzero(chains == c) for c in uniq}
    rng = np.random.default_rng(seed)
    rows = []
    for col in scores:
        sc = scores[col].to_numpy()
        row = {"policy": POLICIES[col], "auc": float(roc_auc_score(feasible, sc))}
        boot = {u: [] for u in unsafe_levels}
        for _ in range(n_boot):
            idx = np.concatenate([by_chain[c] for c in rng.choice(uniq, len(uniq))])
            if feasible[idx].all() or (~feasible[idx]).all():
                continue
            for u in unsafe_levels:
                boot[u].append(approved_at_unsafe(sc[idx], feasible[idx], u)[0])
        for u in unsafe_levels:
            row[f"approved_at_{int(u * 100)}pct_unsafe"] = approved_at_unsafe(sc, feasible, u)[0]
            row[f"ci_low_{int(u * 100)}"], row[f"ci_high_{int(u * 100)}"] = np.percentile(boot[u], [2.5, 97.5])
        rows.append(row)
    return pd.DataFrame(rows)


def paired_gain(scores: pd.DataFrame, pairs: pd.DataFrame, a: str = "P3", b: str = "P2",
                unsafe_levels=(0.01, 0.02, 0.05), n_boot: int = 500, seed: int = 42) -> pd.DataFrame:
    """Policy ``a`` minus policy ``b`` on the same pairs: difference in ranking quality
    and in the share of feasible missions approved at fixed unsafe-approval rates.
    95% intervals resample battery chains (both policies see the same resample)."""
    feasible, chains = pairs["feasible"].to_numpy(), pairs["battery_chain"].to_numpy()
    uniq = np.unique(chains)
    by_chain = {c: np.flatnonzero(chains == c) for c in uniq}
    sa, sb = scores[a].to_numpy(), scores[b].to_numpy()

    def diffs(idx):
        out = {"auc": roc_auc_score(feasible[idx], sa[idx]) - roc_auc_score(feasible[idx], sb[idx])}
        for u in unsafe_levels:
            out[f"approved_at_{int(u * 100)}pct_unsafe"] = (approved_at_unsafe(sa[idx], feasible[idx], u)[0]
                                                           - approved_at_unsafe(sb[idx], feasible[idx], u)[0])
        return out

    point, rng, boot = diffs(np.arange(len(pairs))), np.random.default_rng(seed), []
    for _ in range(n_boot):
        idx = np.concatenate([by_chain[c] for c in rng.choice(uniq, len(uniq))])
        if not (feasible[idx].all() or (~feasible[idx]).all()):
            boot.append(diffs(idx))
    boot = pd.DataFrame(boot)
    return pd.DataFrame([{"metric": k, "gain": v, "ci_low": np.percentile(boot[k], 2.5), "ci_high": np.percentile(boot[k], 97.5),
                          "share_of_resamples_positive": float((boot[k] > 0).mean())} for k, v in point.items()])


def probability_reliability(p: np.ndarray, feasible: np.ndarray, groups, bins=(0, .05, .2, .4, .6, .8, .9, .95, .99, 1.0)) -> pd.DataFrame:
    """Is P(success) honest? Within each band of predicted probability, how often
    did missions actually succeed (also weighted so every battery chain counts equally)."""
    d = pd.DataFrame({"p": p, "ok": np.asarray(feasible, float), "w": group_weights(groups)})
    d["band"] = pd.cut(d["p"], list(bins), include_lowest=True)
    g = d.groupby("band", observed=True)
    return pd.DataFrame({"pairs": g.size(), "mean_predicted": g["p"].mean(), "observed_success": g["ok"].mean(),
                         "observed_by_chain": g.apply(lambda x: np.average(x["ok"], weights=x["w"]))}).reset_index()
