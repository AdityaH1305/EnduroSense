"""Locked train/test split and grouped cross-validation folds.

Plain idea: set aside ~20% of the data *before any modelling* and never look
at it until the final evaluation, so the reported results are honest.

Rules:
- The split unit is the **battery chain**. All readings from one battery chain
  stay on the same side; otherwise a model could be tested on a battery it
  effectively trained on (data leakage).
- Chains containing a ``force_test_routes`` flight (R6, the longest route) go
  to test, so the final evaluation includes an unseen mission distance.
- The rest is chosen by scoring many random splits and keeping the one whose
  test set best matches the full data (share of flights, share of chains that
  reach the reserve, and the mix of speeds, payloads and altitudes).
- The development part is divided into grouped CV folds balanced the same way.

The saved file records a flight -> role map plus a content hash. Test flights
can only be read with ``final=True`` (or ``ENDUROSENSE_FINAL=1``).
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import date

import numpy as np
import pandas as pd

from endurosense.config import ROOT, load_config

DEV, TEST = "dev", "test"


class TestSetLocked(PermissionError):
    """Raised when test data is requested outside the final evaluation."""


def split_path():
    return ROOT / load_config()["split"]["file"]


def _content_hash(payload: dict) -> str:
    body = {k: payload[k] for k in ("flight_role", "dev_fold")}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def _distribution_gap(test: pd.DataFrame, full: pd.DataFrame) -> float:
    """Total-variation distance between test and full data over speed, payload, altitude."""
    gap = 0.0
    for col in ("speed", "payload", "alt_cruise_m"):
        p = full[col].value_counts(normalize=True)
        q = test[col].value_counts(normalize=True).reindex(p.index, fill_value=0.0)
        gap += 0.5 * (p - q).abs().sum()
    return gap


def choose_test_chains(flights: pd.DataFrame, chains: pd.DataFrame, seed: int) -> set:
    """Pick test chains by scoring random candidate splits (see module docstring)."""
    cfg = load_config()["split"]
    rng = np.random.default_rng(seed)
    cruise = flights[flights["route"].str.startswith("R")]
    forced = set(flights.loc[flights["route"].isin(cfg["force_test_routes"]), "battery_chain"])
    n_target = cfg["test_frac"] * len(cruise)
    reserve_share = chains["near_reserve"].mean()
    cruise_per_chain = cruise.groupby("battery_chain").size()
    free = [c for c in chains["battery_chain"] if c not in forced]
    near = set(chains.loc[chains["near_reserve"], "battery_chain"])

    best, best_score = None, np.inf
    for _ in range(cfg["candidates"]):
        order = rng.permutation(free)
        test, n = set(forced), cruise_per_chain.reindex(list(forced)).fillna(0).sum()
        for c in order:
            if n >= n_target:
                break
            test.add(c)
            n += cruise_per_chain.get(c, 0)
        t = cruise[cruise["battery_chain"].isin(test)]
        score = (abs(len(t) - n_target) / n_target
                 + abs(len(test & near) / len(test) - reserve_share)
                 + _distribution_gap(t, cruise))
        if t["speed"].nunique() < cruise["speed"].nunique() or t["payload"].nunique() < 3:
            score += 10.0
        if score < best_score:
            best, best_score = test, score
    return best


def assign_folds(dev_chains: pd.DataFrame, flights: pd.DataFrame, k: int, seed: int) -> dict:
    """Greedy balanced grouped folds: near-reserve and other chains are spread
    separately, each chain going to the fold with the fewest flights so far."""
    rng = np.random.default_rng(seed)
    size = flights.groupby("battery_chain").size()
    load = np.zeros(k)
    fold = {}
    for stratum in (True, False):
        ids = dev_chains.loc[dev_chains["near_reserve"] == stratum, "battery_chain"].to_numpy()
        ids = rng.permutation(ids)
        ids = sorted(ids, key=lambda c: -size[c])      # big chains first, random tie order
        for c in ids:
            j = int(np.argmin(load))
            fold[int(c)] = j
            load[j] += size[c]
    return fold


def make_split(flights: pd.DataFrame, chains: pd.DataFrame, seed: int | None = None) -> dict:
    """Build the split payload (not saved)."""
    cfg = load_config()
    seed = cfg["seed"] if seed is None else seed
    test_chains = choose_test_chains(flights, chains, seed)
    dev = chains[~chains["battery_chain"].isin(test_chains)]
    dev_fold = assign_folds(dev, flights, cfg["split"]["cv_folds"], seed)
    flight_role = {int(f): (TEST if c in test_chains else DEV)
                   for f, c in zip(flights["flight"], flights["battery_chain"])}
    payload = {
        "version": "v1",
        "created": date.today().isoformat(),
        "seed": seed,
        "rules": {"test_frac": cfg["split"]["test_frac"], "cv_folds": cfg["split"]["cv_folds"],
                  "force_test_routes": cfg["split"]["force_test_routes"],
                  "link_window_v": cfg["chains"]["link_window_v"]},
        "test_chains": sorted(int(c) for c in test_chains),
        "flight_role": {str(k): v for k, v in sorted(flight_role.items())},
        "dev_fold": {str(f): dev_fold[int(c)] for f, c in zip(flights["flight"], flights["battery_chain"])
                     if c not in test_chains},
    }
    payload["sha256"] = _content_hash(payload)
    return payload


def save_split(payload: dict, overwrite: bool = False) -> None:
    path = split_path()
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; the split is locked. Create a new version instead.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def load_split() -> dict:
    """Read the locked split and verify it has not been edited."""
    payload = json.loads(split_path().read_text(encoding="utf-8"))
    if _content_hash(payload) != payload["sha256"]:
        raise ValueError("split file content does not match its hash; it was modified")
    return payload


def flight_roles() -> pd.Series:
    """flight id -> 'dev' / 'test'."""
    roles = load_split()["flight_role"]
    return pd.Series({int(k): v for k, v in roles.items()}, name="role")


def _final_allowed(final: bool) -> bool:
    return final or os.environ.get("ENDUROSENSE_FINAL") == "1"


def select(df: pd.DataFrame, role: str, final: bool = False) -> pd.DataFrame:
    """Rows of ``df`` (must have a ``flight`` column) belonging to ``role``.

    Test rows are refused unless ``final=True``: they are for the one-time
    final evaluation only.
    """
    if role == TEST and not _final_allowed(final):
        raise TestSetLocked("test data is locked until the final evaluation (pass final=True)")
    roles = flight_roles()
    return df[df["flight"].map(roles) == role]


def dev_folds(df: pd.DataFrame):
    """Yield ``(k, train_df, val_df)`` over the grouped CV folds of the dev data."""
    folds = {int(k): v for k, v in load_split()["dev_fold"].items()}
    dev = select(df, DEV)
    fold_of_row = dev["flight"].map(folds)
    for k in sorted(set(folds.values())):
        yield k, dev[fold_of_row != k], dev[fold_of_row == k]


def check_consistency(flights: pd.DataFrame) -> None:
    """Fail if current chain definitions put one chain on both sides of the split."""
    roles = flights["flight"].map(flight_roles())
    mixed = flights.assign(role=roles).groupby("battery_chain")["role"].nunique()
    if (mixed > 1).any():
        raise ValueError(f"chains span dev and test: {mixed[mixed > 1].index.tolist()}; "
                         "chain definitions changed after the split was locked")
