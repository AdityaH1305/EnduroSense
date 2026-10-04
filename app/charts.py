"""Charts for the dashboard (Altair). Colours are the project's validated categorical palette
(endurosense.plots.SERIES): blue = the battery / EnduroSense, orange = the mission / measured values.
Identity is always carried by a label or legend as well as by colour."""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from endurosense.plots import SERIES

BATTERY, MISSION, NEUTRAL = SERIES[0], SERIES[1], "#8a8982"
BANDS = [("98% range", 0.01, 0.99, 0.22), ("90% range", 0.05, 0.95, 0.45), ("50% range", 0.25, 0.75, 0.85)]


def range_chart(available, required, truth: dict | None = None, height: int = 170, domain=None) -> alt.Chart:
    """Energy the battery can give vs energy the mission needs, as nested ranges on one Wh axis.
    ``truth`` may give measured values: {"available": Wh, "required": Wh}. ``domain`` fixes the Wh axis,
    so that several charts side by side can be compared."""
    names = {"available": "Battery can give", "required": "Mission needs"}
    rows, med = [], []
    for key, dist in (("available", available), ("required", required)):
        for band, lo, hi, opacity in BANDS:
            rows.append({"what": names[key], "range": band, "from": float(dist.quantile(lo)), "to": float(dist.quantile(hi)), "opacity": opacity})
        med.append({"what": names[key], "best estimate (Wh)": dist.median})
    df, md = pd.DataFrame(rows), pd.DataFrame(med)
    color = alt.Color("what:N", scale=alt.Scale(domain=list(names.values()), range=[BATTERY, MISSION]), legend=None)
    y = alt.Y("what:N", sort=list(names.values()), title=None, axis=alt.Axis(labelFontSize=13, labelLimit=160))
    x = alt.X("from:Q", title="energy (Wh)", scale=alt.Scale(domain=list(domain)) if domain else alt.Scale(zero=False, nice=True))
    bars = alt.Chart(df).mark_bar(height=30, cornerRadius=4).encode(
        x=x, x2="to:Q", y=y, color=color, opacity=alt.Opacity("opacity:Q", scale=None, legend=None),
        tooltip=[alt.Tooltip("what:N", title=" "), alt.Tooltip("range:N"), alt.Tooltip("from:Q", format=".1f"), alt.Tooltip("to:Q", format=".1f")])
    ticks = alt.Chart(md).mark_tick(thickness=3, size=38, color="#0b0b0b").encode(
        x="best estimate (Wh):Q", y=y, tooltip=[alt.Tooltip("what:N", title=" "), alt.Tooltip("best estimate (Wh):Q", format=".1f")])
    layers = [bars, ticks]
    if truth:
        tr = pd.DataFrame([{"what": names[k], "measured (Wh)": v} for k, v in truth.items() if v is not None])
        if len(tr):
            layers.append(alt.Chart(tr).mark_point(shape="diamond", size=170, filled=True, color="#ffffff", stroke="#0b0b0b", strokeWidth=2).encode(
                x="measured (Wh):Q", y=y, tooltip=[alt.Tooltip("what:N", title=" "), alt.Tooltip("measured (Wh):Q", format=".1f")]))
    return alt.layer(*layers).properties(height=height)


def battery_timeline(chain: pd.DataFrame, now_second: int, show_truth: bool, height: int = 230) -> alt.Chart:
    """The battery model's estimate over the whole recording of one battery, with the chosen moment marked.
    ``chain`` has columns second, q05, q50, q95, remaining_wh."""
    base = alt.Chart(chain).encode(x=alt.X("second:Q", title="seconds of recording on this battery (flights joined end to end)"))
    band = base.mark_area(opacity=0.25, color=BATTERY).encode(y=alt.Y("q05:Q", title="energy above reserve (Wh)"), y2="q95:Q")
    lines = [{"series": "Predicted (best estimate)", "col": "q50"}]
    if show_truth:
        lines.append({"series": "Measured afterwards", "col": "remaining_wh"})
    long = pd.concat([chain[["second", l["col"]]].rename(columns={l["col"]: "Wh"}).assign(series=l["series"]) for l in lines])
    line = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x="second:Q", y="Wh:Q",
        color=alt.Color("series:N", scale=alt.Scale(domain=["Predicted (best estimate)", "Measured afterwards"], range=[BATTERY, MISSION]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("second:Q"), alt.Tooltip("series:N", title=" "), alt.Tooltip("Wh:Q", format=".1f")])
    zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color=NEUTRAL, strokeDash=[4, 3]).encode(y="y:Q")
    now = alt.Chart(pd.DataFrame({"second": [now_second], "label": ["decision made here"]})).mark_rule(color="#0b0b0b", strokeWidth=2).encode(
        x="second:Q", tooltip=["label:N"])
    return alt.layer(band, zero, line, now).properties(height=height)


def policy_bars(table: pd.DataFrame, value: str, title: str, highlight: str = "P3", fmt: str = ".1f", height: int = 240) -> alt.Chart:
    """One horizontal bar per policy; EnduroSense in blue, the others grey, values written on the bars."""
    df = table.assign(kind=np.where(table["policy"].str.startswith(highlight), "EnduroSense", "other rule"))
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=18).encode(
        x=alt.X(f"{value}:Q", title=title), y=alt.Y("policy:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["EnduroSense", "other rule"], range=[BATTERY, NEUTRAL]), legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("policy:N"), alt.Tooltip(f"{value}:Q", format=fmt, title=title)])
    text = bars.mark_text(align="left", dx=4, color="#52514e").encode(text=alt.Text(f"{value}:Q", format=fmt), color=alt.value("#52514e"))
    return (bars + text).properties(height=height)


def reliability_chart(rel: pd.DataFrame, height: int = 300) -> alt.Chart:
    """Predicted P(success) against how often missions really succeeded."""
    df = rel.assign(predicted=rel["mean_predicted"] * 100, succeeded=rel["observed_success"] * 100)
    diag = alt.Chart(pd.DataFrame({"x": [0, 100], "y": [0, 100]})).mark_line(color=NEUTRAL, strokeDash=[4, 3]).encode(x="x:Q", y="y:Q")
    pts = alt.Chart(df).mark_line(point=alt.OverlayMarkDef(size=70, filled=True), color=BATTERY, strokeWidth=2).encode(
        x=alt.X("predicted:Q", title="predicted P(success) (%)"), y=alt.Y("succeeded:Q", title="missions that actually succeeded (%)"),
        tooltip=[alt.Tooltip("band:N", title="predicted band"), alt.Tooltip("pairs:Q", title="cases"),
                 alt.Tooltip("predicted:Q", format=".1f"), alt.Tooltip("succeeded:Q", format=".1f")])
    return (diag + pts).properties(height=height)


def cv_vs_test(table: pd.DataFrame, height: int = 330) -> alt.Chart:
    """Model A: cross-validation error next to test error, per model."""
    t = table[table["model"] != "Fixed capacity (reference)"].sort_values("test_mae_supported")
    long = pd.concat([t[["model", "cv_mae_supported"]].rename(columns={"cv_mae_supported": "Wh"}).assign(data="development (cross-validation)"),
                      t[["model", "test_mae_supported"]].rename(columns={"test_mae_supported": "Wh"}).assign(data="test (unseen batteries)")])
    return alt.Chart(long).mark_bar(cornerRadiusEnd=4, height=9).encode(
        x=alt.X("Wh:Q", title="mean absolute error (Wh), lower is better"),
        y=alt.Y("model:N", sort=t["model"].tolist(), title=None, axis=alt.Axis(labelLimit=260)),
        yOffset=alt.YOffset("data:N", sort=["test (unseen batteries)", "development (cross-validation)"]),
        color=alt.Color("data:N", scale=alt.Scale(domain=["test (unseen batteries)", "development (cross-validation)"], range=[BATTERY, NEUTRAL]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("model:N"), alt.Tooltip("data:N", title=" "), alt.Tooltip("Wh:Q", format=".2f")]).properties(height=height)
