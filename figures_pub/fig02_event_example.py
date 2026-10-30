#!/usr/bin/env python3
"""Fig. S5 (draft Fig. 2). Drought event identification at Nawuni (White Volta): discharge, daily SSI by severity class, SSI distribution."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import gaussian_kde
import pubstyle as S

S.setup()
R = Path(__file__).resolve().parents[1]
T = R / "output_observational/tables"
ssi = pd.read_csv(T / "SSI_daily.csv", parse_dates=["date"])
if "Q" not in ssi.columns:      # the public release ships SSI without daily discharge (GRDC data policy)
    raise SystemExit("Fig. S5 needs GRDC daily discharge: run prepare_grdc.py and run_observational.py first")
ev = pd.read_csv(T / "SSI_drought_events.csv", parse_dates=["start_date", "end_date", "peak_date"])
e = ev[ev.node_id == 1531450].sort_values("severity", ascending=False).iloc[0]
s0, e0, pk = e.start_date, e.end_date, e.peak_date
lo, hi = s0 - pd.Timedelta(days=25), e0 + pd.Timedelta(days=25)
d = ssi[(ssi.node_id == 1531450) & ssi.date.between(lo, hi)].sort_values("date")
ins = d[d.date.between(s0, e0)]
TH = {"moderate": -1.28, "severe": -1.65, "extreme": -2.0}

fig = plt.figure(figsize=(S.DOUBLE, 92 * S.MM))
axQ = fig.add_axes([0.075, 0.70, 0.70, 0.25])
axS = fig.add_axes([0.075, 0.12, 0.70, 0.55], sharex=axQ)
axH = fig.add_axes([0.80, 0.12, 0.17, 0.55], sharey=axS)

for ax in (axQ, axS):
    ax.axvspan(s0, e0, color="#f7f1ec", lw=0, zorder=0)
# (a) discharge
axQ.plot(d.date, d.Q, color=S.NAVY, lw=0.9)
axQ.set_yscale("log")
axQ.set_ylabel("$Q$ (m$^{3}$ s$^{-1}$)")
axQ.tick_params(labelbottom=False)
axQ.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
S.hgrid(axQ)
S.panel(axQ, "a", x=-0.055, y=0.98)

# (b) SSI with severity classes
x, y = d.date.values, d.SSI.values
inside = (d.date >= s0) & (d.date <= e0)
m = inside.values
axS.fill_between(x, np.maximum(y, TH["severe"]), TH["moderate"], where=m & (y < TH["moderate"]),
                 color=S.SEV["moderate"], lw=0, interpolate=True, zorder=2)
axS.fill_between(x, np.maximum(y, TH["extreme"]), TH["severe"], where=m & (y < TH["severe"]),
                 color=S.SEV["severe"], lw=0, interpolate=True, zorder=2)
axS.fill_between(x, y, TH["extreme"], where=m & (y < TH["extreme"]), color=S.SEV["extreme"], lw=0,
                 interpolate=True, zorder=2)
axS.plot(x, y, color=S.NAVY, lw=0.9, zorder=3)
for k, v in TH.items():
    axS.axhline(v, color=S.INK if k == "moderate" else S.SEV[k], lw=0.6 if k == "moderate" else 0.5, zorder=1)
axS.plot(pk, e.peak_SSI, marker="v", ms=4.2, color=S.INK, zorder=4, ls="")
axS.annotate(f"peak {e.peak_SSI:.2f}\n{pk:%d %b %Y}".replace("-", "−"), (pk, e.peak_SSI), xytext=(6, -10),
             textcoords="offset points", fontsize=5.7, color=S.INK, va="top", linespacing=1.15)
ytop = -0.98
axS.annotate("", xy=(s0, ytop), xytext=(e0, ytop),
             arrowprops=dict(arrowstyle="|-|,widthA=0.25,widthB=0.25", lw=0.6, color=S.INK))
axS.text(s0 + (e0 - s0) / 2, ytop + 0.03,
         f"event {int(e.event_id)}: {s0:%d %b %Y} – {e0:%d %b %Y}, duration {int(e.duration)} d",
         ha="center", va="bottom", fontsize=5.9, color=S.INK)
axS.text(0.985, 0.04, f"severity {e.severity:.1f}   ·   intensity {e.intensity:.3f} d$^{{-1}}$   ·   "
         f"volumetric deficit {e.severity_hm3:.1f} hm$^{{3}}$", transform=axS.transAxes, ha="right", va="bottom",
         fontsize=5.8, color=S.INK2)
axS.set_ylim(-2.15, -0.85)
axS.set_yticks([-2.0, -1.65, -1.28, -1.0]); axS.set_yticklabels(["−2.00", "−1.65", "−1.28", "−1.00"])
axS.set_ylabel("SSI")
axS.xaxis.set_major_locator(mdates.MonthLocator())
axS.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
axS.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=[8, 15, 22]))
axS.set_xlim(lo, hi)
S.hgrid(axS)
S.panel(axS, "b", x=-0.055, y=0.98)

# (c) distribution of SSI during the event, sharing the SSI axis
v = ins.SSI.dropna().values
bins = np.arange(-2.15, -0.84, 0.05)
cnt, ed = np.histogram(v, bins=bins)
mid = (ed[:-1] + ed[1:]) / 2
col = [S.SEV["extreme"] if c < TH["extreme"] else S.SEV["severe"] if c < TH["severe"]
       else S.SEV["moderate"] if c < TH["moderate"] else "#cfd6de" for c in mid]
axH.barh(ed[:-1], cnt, height=0.05, align="edge", color=col, edgecolor="white", lw=0.4)
yy = np.linspace(-2.15, -0.85, 300)
axH.plot(gaussian_kde(v)(yy) * len(v) * 0.05, yy, color=S.NAVY, lw=0.8)
for k, t in TH.items():
    axH.axhline(t, color=S.INK if k == "moderate" else S.SEV[k], lw=0.6 if k == "moderate" else 0.5)
shares = {"moderate": ((v < -1.28) & (v >= -1.65)).mean(), "severe": ((v < -1.65) & (v >= -2.0)).mean(),
          "extreme": (v < -2.0).mean(), "above threshold": (v >= -1.28).mean()}
for k, yy0 in (("moderate", -1.465), ("severe", -1.825), ("extreme", -2.075)):
    axH.text(0.98, yy0, f"{k} {shares[k] * 100:.0f}%", transform=axH.get_yaxis_transform(), ha="right",
             va="center", fontsize=5.6, color=S.INK2)
axH.text(0.98, -1.1, f"above −1.28\n(pooled) {shares['above threshold'] * 100:.0f}%", transform=axH.get_yaxis_transform(),
         ha="right", va="center", fontsize=5.6, color=S.INK2, linespacing=1.1)
axH.tick_params(labelleft=False)
axH.set_xlabel("Days in event")
S.hgrid(axH, "x")
S.panel(axH, "c", x=-0.06, y=0.98)

S.save(fig, "Fig02_event_example")
print(e[["event_id", "duration", "severity", "intensity", "peak_SSI", "severity_hm3"]].to_dict(), shares)
