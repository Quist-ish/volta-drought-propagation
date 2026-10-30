#!/usr/bin/env python3
"""
Graphical abstract (Elsevier: readable at 13 x 5 cm). Four panels from the derived tables:
(a) the 1969 break in annual basin rainfall (GPCC); (b) gauge-specific change in runoff at fixed rainfall
for the nine informative gauges, ordered from the driest to the wettest catchment (sensitivity_extensions.py);
(c) the observable sample: downstream droughts, testable ones and matched ones; (d) years of observation
needed for a 50%-power test under three monitoring designs (monitoring_horizon.json).
Writes out/GraphicalAbstract.{pdf,png,tiff}.
Author: I. Quist (NUIST)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pubstyle as S

S.setup()
R = Path(__file__).resolve().parents[1]
TC, TP = R / "output_climate/tables", R / "output_projection/tables"
RISE, FALL, GREY = S.VOLTA, S.EBRO, "#b9bec4"

fig = plt.figure(figsize=(133 * S.MM, 53 * S.MM))
gs = fig.add_gridspec(1, 4, left=0.045, right=0.99, top=0.80, bottom=0.17, wspace=0.62,
                      width_ratios=[1.15, 1.15, 0.95, 1.0])
title = dict(fontsize=7.2, fontweight="bold", color=S.INK, loc="left", pad=4)
lab = lambda ax, s: None   # no panel letters in the graphical abstract

# (a) rainfall break ---------------------------------------------------------------------------
ax = fig.add_subplot(gs[0])
P = pd.read_csv(TC / "precip_catchments_basin.csv", index_col=0, parse_dates=True)["basin"]
ann = P.groupby(P.index.year).sum().loc[1951:2020]
tr = pd.read_csv(TC / "trend_changepoint_summary.csv").set_index("series").loc["Basin annual precipitation"]
yb = int(tr.pettitt_year)
ax.bar(ann.index, ann.values, width=0.75, color=GREY, lw=0)
for (a, b, m) in ((1951, yb, tr.mean_before), (yb + 1, 2020, tr.mean_after)):
    ax.plot([a - 0.4, b + 0.4], [m, m], color=S.NAVY, lw=1.6, solid_capstyle="butt")
ax.axvline(yb + 0.5, color=S.INK, lw=0.6)
ax.text(yb - 0.8, 1340, f"{yb}", fontsize=6.3, color=S.INK, va="top", ha="right")
ax.text(2019, 1310, f"mean −{100 * (1 - tr.mean_after / tr.mean_before):.1f}%", fontsize=6.5,
        color=S.NAVY, ha="right", va="center", fontweight="bold")
ax.set_ylim(600, 1350); ax.set_xlim(1949, 2022)
ax.set_xticks([1960, 1990, 2020]); ax.set_ylabel("mm yr⁻¹", labelpad=1)
ax.set_title("Rainfall regime shift", **title); lab(ax, "a")

# (b) runoff efficiency by gauge -----------------------------------------------------------------
ax = fig.add_subplot(gs[1])
sl = pd.read_csv(TC / "runoff_gauge_slopes.csv")
sl = sl[(sl["size"] >= 5) & (sl.span_years >= 8)].sort_values("P_mean_mm", ascending=False)
y = np.arange(len(sl))
v = sl.change_pct_per_decade.values
ax.barh(y, v, height=0.62, color=[RISE if x > 0 else FALL for x in v], lw=0)
ax.axvline(0, color=S.INK, lw=0.6)
ax.set_yticks(y); ax.set_yticklabels(sl.station.values, fontsize=6)
ax.tick_params(axis="y", length=0, pad=1.5)
ax.spines["left"].set_visible(False)
ax.set_xlim(-40, 32); ax.set_xticks([-30, 0, 30])
ax.set_xlabel("% per decade at fixed rainfall", labelpad=1)
ax.text(30, len(sl) - 0.5, "drier", fontsize=6, color=S.INK2, ha="right", va="center", style="italic")
ax.text(30, -0.1, "wetter", fontsize=6, color=S.INK2, ha="right", va="center", style="italic")
ax.set_ylim(-0.6, len(sl) - 0.2)
ax.set_title("Runoff efficiency", **title); lab(ax, "b")

# (c) observable sample ----------------------------------------------------------------------------
ax = fig.add_subplot(gs[2])
vals, names = [104, 40, 10], ["downstream\ndroughts", "testable", "matched"]
cols = [GREY, "#7d8794", S.NAVY]
ax.barh([2, 1, 0], vals, height=0.6, color=cols, lw=0)
for yy, vv in zip([2, 1, 0], vals):
    ax.text(vv + 3, yy, str(vv), fontsize=6.8, va="center", color=S.INK, fontweight="bold")
ax.set_yticks([2, 1, 0]); ax.set_yticklabels(names, fontsize=6)
ax.tick_params(axis="y", length=0, pad=1.5)
ax.spines["left"].set_visible(False)
ax.set_xlim(0, 125); ax.set_xticks([0, 50, 100])
ax.set_xlabel("events, 1961–2007", labelpad=1)
ax.set_title("Observability", **title); lab(ax, "c")

# (d) monitoring horizon -----------------------------------------------------------------------------
ax = fig.add_subplot(gs[3])
mh = json.load(open(TP / "monitoring_horizon.json"))["designs"]
keys = list(mh)
pick = [k for k in keys if "observed" in k.lower()][:1] + [k for k in keys if "continuous" in k.lower()][:1] + \
       [k for k in keys if "reanalysis" in k.lower()][:1]
yrs = [mh[k]["years_n50"]["historical"] for k in pick]
names = ["observed\nrecords", "18 gauges,\ncontinuous", "reanalysis\ndensity"]
ax.barh([2, 1, 0], yrs, height=0.6, color=[GREY, S.NAVY, "#7d8794"], lw=0)
for yy, vv in zip([2, 1, 0], yrs):
    ax.text(vv * 1.25, yy, f"{vv:.0f} yr", fontsize=6.8, va="center", color=S.INK, fontweight="bold")
ax.set_xscale("log"); ax.set_xlim(1, 5000); ax.set_xticks([1, 10, 100, 1000])
ax.set_xticklabels(["1", "10", "100", "1,000"])
ax.set_yticks([2, 1, 0]); ax.set_yticklabels(names, fontsize=6)
ax.tick_params(axis="y", length=0, pad=1.5)
ax.spines["left"].set_visible(False)
ax.set_xlabel("years to 50% power", labelpad=1)
ax.set_title("Time to a decisive test", **title); lab(ax, "d")

print([round(x, 1) for x in yrs], pick)
S.save(fig, "GraphicalAbstract")
