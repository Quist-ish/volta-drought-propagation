#!/usr/bin/env python3
"""Figs 7-9 and Fig. S8. Spatiotemporal representation of observed propagation chains."""
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.lines import Line2D
import pubstyle as S

S.setup()
R = Path(__file__).resolve().parents[1]
T = R / "output_observational/tables"
st = pd.read_csv(R / "data/grdc_stations_raw.csv").set_index("grdc_no")
ssi = pd.read_csv(T / "SSI_daily.csv", parse_dates=["date"])
ev = pd.read_csv(T / "SSI_drought_events.csv", parse_dates=["start_date", "end_date", "peak_date"])
ch = pd.read_csv(T / "Chains_final.csv", parse_dates=["origin_start", "origin_end", "chain_start", "chain_end"])
pr = pd.read_csv(T / "propagation_pairs.csv", parse_dates=["origin_start", "origin_end", "upstream_start", "upstream_end"])
basin = gpd.read_file(R / "data/GRDC_VOLTA_POLITICAL_TRANSBOUNDARY_BASIN.shp").to_crs(4326)
reach = pd.read_parquet(R / "data/reanalysis/reaches_volta.parquet")
rr = reach[reach.area_km2 >= 800].set_index("LINKNO")
SEGS, LWS = [], []
for lk, r in rr.iterrows():
    if r.DSLINKNO in rr.index:
        d = rr.loc[r.DSLINKNO]
        SEGS.append([(r.lon, r.lat), (d.lon, d.lat)])
        LWS.append(0.2 + 0.9 * (np.log10(r.area_km2) - np.log10(800)) / (np.log10(4e5) - np.log10(800)))
nodes = sorted(ssi.node_id.unique())
name = lambda n: st.station_name[n].title().replace("(Halcrow)", "").strip()
river = lambda n: st.river[n].title()
TH = (-1.28, -1.65, -2.0)


def ssi_panel(ax, node, ev_s, ev_e, lo, hi, s0, role):
    d = ssi[(ssi.node_id == node) & ssi.date.between(lo, hi)].set_index("date").reindex(pd.date_range(lo, hi))
    x, y = d.index.values, d.SSI.values
    m = (d.index >= ev_s) & (d.index <= ev_e)
    ax.axvspan(ev_s, ev_e, color="#f7f1ec", lw=0, zorder=0)
    ax.fill_between(x, np.maximum(y, TH[1]), TH[0], where=m & (y < TH[0]), color=S.SEV["moderate"], lw=0, interpolate=True)
    ax.fill_between(x, np.maximum(y, TH[2]), TH[1], where=m & (y < TH[1]), color=S.SEV["severe"], lw=0, interpolate=True)
    ax.fill_between(x, y, TH[2], where=m & (y < TH[2]), color=S.SEV["extreme"], lw=0, interpolate=True)
    ax.plot(x, y, color=S.NAVY if role == "origin" else S.EBRO, lw=0.85, zorder=3)
    ax.axhline(TH[0], color=S.INK, lw=0.5, zorder=1)
    ax.axvline(s0, color=S.VOLTA, lw=0.8, zorder=1)
    ax.set_ylabel("SSI")
    S.hgrid(ax)
    return d.SSI


def chain_figure(cid, fname, case):
    c = ch[ch.chain_id == cid].iloc[0]
    p = pr[(pr.origin_node == c.origin_node) & (pr.origin_start == c.origin_start)].iloc[0]
    o, u = int(c.origin_node), int(p.upstream_node)
    s0 = c.origin_start
    lo, hi = c.chain_start - pd.Timedelta(days=30), c.chain_end + pd.Timedelta(days=30)
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    fig = plt.figure(figsize=(S.DOUBLE, 84 * S.MM))
    a1 = fig.add_axes([0.07, 0.53, 0.56, 0.32])
    a2 = fig.add_axes([0.07, 0.14, 0.56, 0.32], sharex=a1)
    am = fig.add_axes([0.665, 0.06, 0.33, 0.86])
    v1 = ssi_panel(a1, u, p.upstream_start, p.upstream_end, lo, hi, s0, "upstream")
    v2 = ssi_panel(a2, o, c.origin_start, c.origin_end, lo, hi, s0, "origin")
    vals = pd.concat([v1, v2]).dropna()
    ylo, yhi = min(-2.25, vals.min() - 0.15), max(0.8, vals.max() + 0.15)
    for ax, node, role, es, ee in ((a1, u, "upstream", p.upstream_start, p.upstream_end),
                                   (a2, o, "origin", c.origin_start, c.origin_end)):
        ax.set_ylim(ylo, yhi)
        lag = (es - s0).days
        dur = (ee - es).days + 1
        ax.text(0.005, 1.03, f"{name(node)} ({river(node)}), {role}", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=6.3, fontweight="bold", color=S.INK)
        ax.text(0.995, 1.03, f"onset lag L = {lag} d · duration D = {dur} d".replace("-", "−"), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=5.9, color=S.INK2)
    a1.tick_params(labelbottom=False)
    a2.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
    a2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(a2.xaxis.get_major_locator(), show_offset=False))
    a2.set_xlim(lo, hi)
    S.panel(a1, "a", x=-0.055, y=1.12)
    fig.text(0.07, 0.975, f"{case} · {name(o)} ← {name(u)}, {c.chain_start:%-d %b} – {c.chain_end:%-d %b %Y}",
             ha="left", va="center", fontsize=6.6, fontweight="bold", color=S.INK)
    hdr = (f"chain duration {int(c.chain_duration)} d · size {int(c.chain_size)} · propagation fraction "
           f"{c.propagation_fraction:.2f} · mean lag {c.lag_mean:.0f} d · severity {c.chain_severity:.1f} · "
           f"volumetric severity {c.chain_severity_hm3:.1f} hm$^{{3}}$").replace("-", "−")
    fig.text(0.07, 0.94, hdr, ha="left", va="center", fontsize=6.0, color=S.INK2)
    hd = [Patch(fc=S.SEV[k], label=k) for k in ("moderate", "severe", "extreme")] + \
         [Line2D([], [], color=S.INK, lw=0.5, label="SSI = −1.28"),
          Line2D([], [], color=S.VOLTA, lw=0.8, label="origin onset $s_0$"),
          Patch(fc="#f7f1ec", label="event span")]
    a2.legend(handles=hd, loc="upper center", bbox_to_anchor=(0.5, -0.19), ncols=6, fontsize=5.6, handlelength=1.3)
    # map
    basin.boundary.plot(ax=am, color=S.INK, lw=0.7, zorder=2)
    basin.plot(ax=am, color="#f5f6f7", lw=0, zorder=0)
    am.add_collection(LineCollection(SEGS, linewidths=LWS, colors="#8fb8e2", zorder=1))
    for n in nodes:
        if n in (o, u):
            continue
        am.scatter(st.lon[n], st.lat[n], s=8, fc="#d0d0d0", ec="#8c8c8c", lw=0.4, zorder=3)
    am.add_patch(FancyArrowPatch((st.lon[u], st.lat[u]), (st.lon[o], st.lat[o]), arrowstyle="-|>", mutation_scale=8,
                                 lw=1.0, color=S.VOLTA, shrinkA=4, shrinkB=5, connectionstyle="arc3,rad=0.15", zorder=4))
    am.scatter(st.lon[u], st.lat[u], s=26, fc=S.EBRO, ec="white", lw=0.7, zorder=5)
    am.scatter(st.lon[o], st.lat[o], s=30, fc=S.NAVY, ec="white", lw=0.7, zorder=5)
    halo = [pe.withStroke(linewidth=1.8, foreground="white")]
    for n, lab, dy in ((u, "upstream", 5), (o, "origin", -5)):
        dx = 5 if st.lon[n] < 0.5 else -5
        am.annotate(f"{name(n)} ({lab})", (st.lon[n], st.lat[n]), xytext=(dx, dy), textcoords="offset points",
                    ha="left" if dx > 0 else "right", va="bottom" if dy > 0 else "top", fontsize=5.8, color=S.INK,
                    zorder=6, path_effects=halo)
    minx, miny, maxx, maxy = basin.total_bounds
    am.set_xlim(minx - 0.2, maxx + 0.2); am.set_ylim(miny - 0.15, maxy + 0.15)
    am.set_aspect(1 / np.cos(np.radians(10.5)))
    am.set_xticks(range(-5, 3, 2)); am.set_yticks(range(6, 15, 2))
    am.set_xticklabels([f"{abs(v)}°{'W' if v < 0 else 'E' if v > 0 else ''}" for v in range(-5, 3, 2)])
    am.set_yticklabels([f"{v}°N" for v in range(6, 15, 2)])
    am.tick_params(direction="in", top=True, right=True, labelsize=5.4, length=1.8)
    for sp in am.spines.values():
        sp.set_visible(True)
    am.set_xlabel(""); am.set_ylabel("")
    am.legend(handles=[Line2D([], [], marker="o", ls="", mfc=S.NAVY, mec="white", ms=4.5, label="origin"),
                       Line2D([], [], marker="o", ls="", mfc=S.EBRO, mec="white", ms=4.2, label="matched upstream"),
                       Line2D([], [], marker="o", ls="", mfc="#d0d0d0", mec="#8c8c8c", ms=3, label="other stations")],
              loc="lower left", fontsize=5.5, handletextpad=0.2)
    S.panel(am, "b", x=0.0, y=1.005)
    S.save(fig, fname)


for cid, fname, case in (("1531100_1", "Fig07_case1_bamboi", "Case 1"), ("1531450_27", "Fig08_case2_nawuni", "Case 2"),
                         ("1731400_14", "Fig09_case3_porga", "Case 3"), ("1531550_9", "FigS08_pwalagu", "Most severe chain")):
    chain_figure(cid, fname, case)
print("done")
