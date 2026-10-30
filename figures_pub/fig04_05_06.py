#!/usr/bin/env python3
"""Fig. 4 (W-M sensitivity), Fig. 5 (event descriptors) and Fig. 6 (the ten observed chains)."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.stats import gaussian_kde
import pubstyle as S

S.setup()
R = Path(__file__).resolve().parents[1]
T = R / "output_observational/tables"
st = pd.read_csv(R / "data/grdc_stations_raw.csv").set_index("grdc_no")
name = lambda n: st.station_name[n].title().replace("(Halcrow)", "").strip()

# =================================================================================== Fig. 4
g = pd.read_csv(T / "sensitivity_grid.csv")
P = [("n_pairs", "Valid pairs (lag ≤ 0)", "count"),
     ("n_origin_matched", "Origin events matched", "count"),
     ("mean_propagation_fraction", "Mean propagation fraction", "cont"),
     ("median_propagation_fraction", "Median propagation fraction", "cont"),
     ("median_lag", "Median lag (d)", "lag"),
     ("mean_chain_size", "Mean chain size, $N_c - 1$", "cont")]
fig = plt.figure(figsize=(S.DOUBLE, 112 * S.MM))
gs = fig.add_gridspec(2, 3, left=0.065, right=0.985, bottom=0.085, top=0.94, wspace=0.42, hspace=0.5)
for k, (col, lab, kind) in enumerate(P):
    sub = gs[k // 3, k % 3].subgridspec(2, 2, height_ratios=[0.34, 1], width_ratios=[1, 0.045], hspace=0.08, wspace=0.06)
    ax = fig.add_subplot(sub[1, 0]); axp = fig.add_subplot(sub[0, 0], sharex=ax); cax = fig.add_subplot(sub[1, 1])
    piv = g.pivot(index="W", columns="M", values=col)
    v = piv.values
    if kind == "count":
        lv = np.arange(np.floor(v.min()) - 0.5, np.ceil(v.max()) + 1.5, 1)
        norm, cmap = BoundaryNorm(lv, 256), S.BLUES
    elif kind == "lag":
        norm, cmap = Normalize(v.min(), v.max()), S.BLUES.reversed()
    else:
        norm, cmap = Normalize(v.min(), v.max()), S.BLUES
    im = ax.pcolormesh(piv.columns, piv.index, v, cmap=cmap, norm=norm, shading="nearest", rasterized=True)
    ax.axvline(5, color="white", lw=0.5)
    ax.axhline(45, color="white", lw=0.5)
    ax.plot(5, 45, marker="P", ms=5.5, mfc="white", mec=S.INK, mew=0.6, zorder=5)
    ax.set_xlabel("Minimum overlap $M$ (d)"); ax.set_ylabel("Search window $W$ (d)")
    ax.set_yticks([0, 30, 60, 90, 120]); ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    for sp in ax.spines.values():
        sp.set_visible(True)
    prof = piv.loc[45]
    axp.step(prof.index, prof.values, where="mid", color=S.NAVY, lw=0.8)
    axp.plot(5, prof[5], marker="o", ms=2.6, color=S.VOLTA, zorder=4)
    axp.tick_params(labelbottom=False, labelsize=5.4)
    axp.set_title(lab, loc="left", fontsize=6.4, pad=2)
    axp.yaxis.set_major_locator(plt.MaxNLocator(3))
    assert np.ptp(v, axis=0).max() == 0      # every metric is invariant along W (Proposition 1)
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5.4, length=1.5, width=0.4)
    if kind == "count":
        cb.set_ticks(np.arange(np.floor(v.min()), np.ceil(v.max()) + 1, 2 if v.max() - v.min() > 6 else 1))
    S.panel(axp, "abcdef"[k], x=-0.2, y=1.12)
fig.text(0.985, 0.975, "Top strips: profile along $M$ at $W$ = 45 d (identical for every $W$). Cross: $W$ = 45 d, $M$ = 5 d.",
         ha="right", va="bottom", fontsize=5.6, color=S.INK2)
S.save(fig, "Fig04_sensitivity")

# =================================================================================== Fig. 5
ev = pd.read_csv(T / "SSI_drought_events.csv")
D = [("duration", "Duration (d)"), ("severity", "Severity"), ("intensity", "Intensity (d$^{-1}$)"),
     ("severity_hm3", "Volumetric severity (hm$^{3}$)")]
fig, axs = plt.subplots(1, 4, figsize=(S.DOUBLE, 62 * S.MM), gridspec_kw=dict(wspace=0.55, left=0.06, right=0.99,
                                                                             bottom=0.1, top=0.9))
rng = np.random.default_rng(7)
for k, (ax, (col, lab)) in enumerate(zip(axs, D)):
    a = ev[col].values
    pos = a[a > 0]
    lv = np.log10(pos)
    yy = np.linspace(lv.min() - 0.1, lv.max() + 0.1, 300)
    dens = gaussian_kde(lv, bw_method=0.35)(yy)
    dens = dens / dens.max() * 0.36
    ax.fill_betweenx(10 ** yy, 0.06, 0.06 + dens, color="#cde2fb", lw=0)
    ax.plot(0.06 + dens, 10 ** yy, color=S.EBRO, lw=0.6)
    q1, q2, q3 = np.percentile(pos, [25, 50, 75])
    ax.add_patch(plt.Rectangle((-0.045, q1), 0.09, q3 - q1, fc="white", ec=S.INK, lw=0.6, zorder=3))
    ax.plot([-0.045, 0.045], [q2, q2], color=S.INK, lw=1.0, zorder=4)
    iqr = np.log10(q3) - np.log10(q1)
    lo_w = pos[np.log10(pos) >= np.log10(q1) - 1.5 * iqr].min(); hi_w = pos[np.log10(pos) <= np.log10(q3) + 1.5 * iqr].max()
    ax.plot([0, 0], [lo_w, q1], color=S.INK, lw=0.6); ax.plot([0, 0], [q3, hi_w], color=S.INK, lw=0.6)
    ax.plot(0, a.mean(), marker="D", ms=3.2, mfc=S.VOLTA, mec=S.INK, mew=0.4, zorder=5)
    jit = rng.uniform(-0.16, -0.07, len(pos))
    ax.scatter(-0.11 + (jit + 0.115), pos, s=2.2, color=S.NAVY, alpha=0.55, lw=0, zorder=2)
    ax.set_yscale("log")
    ax.set_xlim(-0.24, 0.46)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel(lab)
    ax.yaxis.set_minor_formatter(NullFormatter())
    txt = f"median {np.median(a):.3g}\nmean {a.mean():.3g}\nIQR {np.percentile(a, 75) - np.percentile(a, 25):.3g}"
    if (a <= 0).any():
        ax.text(0.98, 0.02, f"+{int((a <= 0).sum())} zero-deficit\nevents, not drawn", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6, color=S.INK2, linespacing=1.1)
    ax.text(0.98, 0.98, txt, transform=ax.transAxes, ha="right", va="top", fontsize=6, color=S.INK2, linespacing=1.2)
    S.panel(ax, "abcd"[k], x=-0.42, y=1.02)
fig.text(0.5, 0.965, f"n = {len(ev)} events at 16 stations, 1961–2007", ha="center", fontsize=6.2, color=S.INK2)
S.save(fig, "Fig05_event_summary")

# =================================================================================== Fig. 6
ch = pd.read_csv(T / "Chains_final.csv", parse_dates=["origin_start", "origin_end", "chain_start", "chain_end"])
pr = pd.read_csv(T / "propagation_pairs.csv", parse_dates=["origin_start", "origin_end", "upstream_start", "upstream_end"])
ch["origin_event_id"] = ch.chain_id.str.split("_").str[1].astype(int)
ch = ch.merge(pr, on=["origin_node", "origin_event_id"], suffixes=("", "_p")).sort_values("origin_start").reset_index(drop=True)
n = len(ch)
fig = plt.figure(figsize=(S.DOUBLE, 86 * S.MM))
gs = fig.add_gridspec(1, 5, width_ratios=[2.5, 1, 1, 1, 1.1], left=0.215, right=0.99, bottom=0.13, top=0.9, wspace=0.12)
axg = fig.add_subplot(gs[0])
y = np.arange(n)[::-1]
for yi, (_, r) in zip(y, ch.iterrows()):
    s0 = r.origin_start
    us, ue = (r.upstream_start - s0).days, (r.upstream_end - s0).days
    oe = (r.origin_end - s0).days
    axg.barh(yi + 0.18, ue - us + 1, left=us, height=0.3, color=S.EBRO, alpha=0.9, lw=0)
    axg.barh(yi - 0.18, oe + 1, left=0, height=0.3, color=S.NAVY, lw=0)
    ov0, ov1 = max(us, 0), min(ue, oe)
    axg.barh(yi, ov1 - ov0 + 1, left=ov0, height=0.66, fc="none", ec=S.VOLTA, lw=0.6, zorder=4)
    axg.text(-58, yi, f"{int(r.lag_days):d} d", ha="right", va="center", fontsize=5.6, color=S.INK2)
axg.axvline(0, color=S.INK, lw=0.6)
axg.set_xlim(-55, 75)
axg.set_yticks(y)
axg.set_yticklabels([f"{name(r.origin_node)} ← {name(r.upstream_node)}, {r.origin_start:%b %Y}" for _, r in ch.iterrows()],
                    fontsize=5.8)
axg.tick_params(axis="y", length=0, pad=19)
axg.spines["left"].set_visible(False)
axg.set_xlabel("Days relative to origin onset $s_0$")
axg.text(-58, n - 0.35, "lag", ha="right", va="bottom", fontsize=5.6, color=S.INK2, fontweight="bold")
S.hgrid(axg, "x")
from matplotlib.patches import Patch
axg.legend(handles=[Patch(fc=S.EBRO, label="upstream event"), Patch(fc=S.NAVY, label="origin event"),
                    Patch(fc="none", ec=S.VOLTA, lw=0.6, label="overlap")], loc="lower right", fontsize=5.5,
           bbox_to_anchor=(1.0, 1.0), ncols=3, columnspacing=0.8, handlelength=1.2)
S.panel(axg, "a", x=-0.52, y=1.0)
cols = [("chain_duration", "Duration (d)", False), ("propagation_fraction", "Propagation\nfraction", False),
        ("chain_severity", "Severity", False), ("chain_severity_hm3", "Volumetric\nseverity (hm$^{3}$)", True)]
for k, (c, lab, lg) in enumerate(cols):
    ax = fig.add_subplot(gs[k + 1], sharey=axg)
    vals = ch[c].values
    base = 1 if lg else 0
    ax.hlines(y, base, vals, color="#c4ccd6", lw=1.0)
    ax.scatter(vals, y, s=11, color=S.NAVY, zorder=3, lw=0)
    if lg:
        ax.set_xscale("log"); ax.set_xlim(0.8, 800)
    else:
        ax.set_xlim(0, vals.max() * 1.15)
    ax.axvline(np.median(vals), color=S.VOLTA, lw=0.6)
    ax.tick_params(labelleft=False, axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(lab)
    S.hgrid(ax, "x")
    S.panel(ax, "bcde"[k], x=0.02, y=1.0, )
fig.text(0.99, 0.955, "dashed line: median of the ten chains · all chains contain two stations", ha="right",
         fontsize=5.5, color=S.INK2)
S.save(fig, "Fig06_chains")
print("done")
