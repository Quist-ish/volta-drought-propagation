#!/usr/bin/env python3
"""Fig. 1. Study area: Köppen-Geiger zones, river network, GRDC stations and gauged links; station mean discharge."""
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyArrowPatch, Rectangle
from matplotlib.collections import LineCollection
from scipy.stats import gaussian_kde
from shapely.geometry import Point
import pubstyle as S

S.setup()
R = Path(__file__).resolve().parents[1]
T = R / "output_observational/tables"
st = pd.read_csv(R / "data/grdc_stations_raw.csv").set_index("grdc_no")
ssi = pd.read_csv(T / "SSI_daily.csv", parse_dates=["date"])
nodes = sorted(ssi.node_id.unique())
conn = pd.read_csv(T / "upstream_connectivity.csv", sep=";")
chains = pd.read_csv(T / "Chains_final.csv")
basin = gpd.read_file(R / "data/GRDC_VOLTA_POLITICAL_TRANSBOUNDARY_BASIN.shp").to_crs(4326)
poly = basin.union_all()
countries = gpd.read_file(R / "data/external/naturalearth_lowres/naturalearth_lowres.shp").to_crs(4326)
reach = pd.read_parquet(R / "data/reanalysis/reaches_volta.parquet")

# ---- Köppen-Geiger grid (cached) -----------------------------------------------------------------
cache = Path(__file__).resolve().parent / "cache_koppen.npz"
g = 0.05
minx, miny, maxx, maxy = poly.bounds
xg = np.arange(minx + g / 2, maxx, g); yg = np.arange(miny + g / 2, maxy, g)
if cache.exists():
    grid = np.load(cache)["grid"]
else:
    import kgcpy
    from shapely import contains_xy
    X, Y = np.meshgrid(xg, yg)
    inside = contains_xy(poly, X, Y)
    grid = np.full(X.shape, "", dtype=object)
    for i, j in zip(*np.nonzero(inside)):
        grid[i, j] = kgcpy.lookupCZ(Y[i, j], X[i, j])
    grid = grid.astype(str)
    np.savez_compressed(cache, grid=grid)
w = np.cos(np.radians(yg))[:, None] * np.ones_like(grid, dtype=float)
cls = {"Aw": "Tropical savanna (Aw)", "BSh": "Hot semi-arid steppe (BSh)", "BWh": "Hot desert (BWh)"}
share = {k: w[grid == k].sum() / w[grid != ""].sum() * 100 for k in cls}
KC = {"Aw": "#d3ebdc", "BSh": "#f6dcc6", "BWh": "#e9a77f"}       # light tints: background, not data
code = np.full(grid.shape, np.nan)
for i, k in enumerate(cls):
    code[grid == k] = i

# ---- figure ---------------------------------------------------------------------------------------
fig = plt.figure(figsize=(S.DOUBLE, 112 * S.MM))
ax = fig.add_axes([0.045, 0.075, 0.585, 0.905])
lat0 = 10.5
ax.imshow(np.ma.masked_invalid(code), origin="lower", extent=(xg[0] - g / 2, xg[-1] + g / 2, yg[0] - g / 2, yg[-1] + g / 2),
          cmap=ListedColormap([KC[k] for k in cls]), vmin=-0.5, vmax=2.5, interpolation="nearest", zorder=0,
          rasterized=True)
countries.boundary.plot(ax=ax, color="#c9c5bd", lw=0.45, zorder=1)

# river network from the TDX-Hydro reach graph: reach point -> downstream reach point, width by drainage area
rr = reach[reach.area_km2 >= 800].set_index("LINKNO")
segs, lws = [], []
for lk, r in rr.iterrows():
    if r.DSLINKNO in rr.index:
        d = rr.loc[r.DSLINKNO]
        segs.append([(r.lon, r.lat), (d.lon, d.lat)])
        lws.append(0.25 + 1.15 * (np.log10(r.area_km2) - np.log10(800)) / (np.log10(4e5) - np.log10(800)))
ax.add_collection(LineCollection(segs, linewidths=lws, colors="#4f8fcf", capstyle="round", zorder=2))
basin.boundary.plot(ax=ax, color=S.INK, lw=0.9, zorder=3)

# gauged links
for _, r in conn.iterrows():
    if pd.notna(r.immediate_downstream):
        a, b = int(r.node_id), int(r.immediate_downstream)
        ax.add_patch(FancyArrowPatch((st.lon[a], st.lat[a]), (st.lon[b], st.lat[b]), arrowstyle="-|>",
                                     mutation_scale=6, lw=0.7, color=S.NAVY, shrinkA=3.5, shrinkB=3.8,
                                     connectionstyle="arc3,rad=0.12", zorder=4))
origins = set(chains.origin_node)
linked = {int(n) for n in conn.node_id[conn.n_upstream > 0]} | \
         {int(u) for s in conn.upstream_chain.dropna() for u in str(s).split(",") if u}
role = {n: ("origin" if n in origins else "linked" if n in linked else "isolated") for n in nodes}
MK = {"origin": dict(marker="o", s=24, fc=S.VOLTA, ec=S.INK, lw=0.6),
      "linked": dict(marker="^", s=20, fc="white", ec=S.INK, lw=0.7),
      "isolated": dict(marker="s", s=14, fc="#bdbdbd", ec=S.INK, lw=0.6)}
for n in nodes:
    m = MK[role[n]]
    ax.scatter(st.lon[n], st.lat[n], marker=m["marker"], s=m["s"], facecolor=m["fc"], edgecolor=m["ec"],
               linewidths=m["lw"], zorder=6)
OFF = {"Lawra": (-4, 3, "right"), "Dapola": (-4, -6, "right"), "Vonkoro": (-4, 1, "right"),
       "Bamboi": (4, -2, "left"), "Wiasi": (-4, 2, "right"), "Yagaba": (-4, -6, "right"),
       "Nawuni": (4, -3, "left"), "Pwalagu": (4, -3, "left"), "Nangodi": (-4, 2, "right"),
       "Yarugu": (4, 2, "left"), "Yakala": (4, 2, "left"), "Samboali": (4, 2, "left"),
       "Porga": (4, 1, "left"), "Tiele": (4, -4, "left"), "Lama Kara": (4, -2, "left"),
       "Sabari": (-4, -5, "right"), "Ekumdipe": (4, 1, "left"), "Prang": (4, -3, "left")}
halo = [pe.withStroke(linewidth=1.6, foreground="white")]
for n in nodes:
    nm = st.station_name[n].title().replace("(Halcrow)", "").strip()
    dx, dy, ha = OFF.get(nm, (4, 2, "left"))
    ax.annotate(nm, (st.lon[n], st.lat[n]), xytext=(dx, dy), textcoords="offset points", fontsize=5.6,
                color=S.INK, ha=ha, va="center", zorder=7, path_effects=halo)
for c, (x, y) in {"BURKINA FASO": (-1.55, 12.25), "GHANA": (-0.9, 6.55), "TOGO": (0.95, 7.55),
                  "BENIN": (1.75, 10.15), "MALI": (-3.1, 13.95), "CÔTE D'IVOIRE": (-3.35, 8.65)}.items():
    ax.text(x, y, c, fontsize=5.4, color="#7d776c", ha="center", va="center", zorder=2,
            fontweight="bold", path_effects=[pe.withStroke(linewidth=1.4, foreground="white")])
ax.set_xlim(minx - 0.35, maxx + 0.35); ax.set_ylim(miny - 0.25, maxy + 0.2)
ax.set_xlabel(""); ax.set_ylabel("")
ax.set_aspect(1 / np.cos(np.radians(lat0)))
ax.set_xticks(range(-5, 3)); ax.set_yticks(range(6, 15, 2))
ax.set_xticklabels([f"{abs(v)}°{'W' if v < 0 else 'E' if v > 0 else ''}" for v in range(-5, 3)])
ax.set_yticklabels([f"{v}°N" for v in range(6, 15, 2)])
ax.tick_params(direction="in", top=True, right=True, length=2)
for sp in ax.spines.values():
    sp.set_visible(True); sp.set_linewidth(0.5)

# scale bar (at 10.5°N) and north arrow
km_per_deg = 111.32 * np.cos(np.radians(lat0))
x0, y0 = 1.25, 6.2
for k, (a, b) in enumerate(((0, 25), (25, 50), (50, 100))):
    ax.add_patch(Rectangle((x0 + a / km_per_deg, y0), (b - a) / km_per_deg, 0.07,
                           fc=S.INK if k % 2 == 0 else "white", ec=S.INK, lw=0.5, zorder=8))
for v in (0, 50, 100):
    ax.text(x0 + v / km_per_deg, y0 + 0.13, f"{v}", ha="center", va="bottom", fontsize=5.4, color=S.INK)
ax.text(x0 + 100 / km_per_deg + 0.08, y0 + 0.035, "km", ha="left", va="center", fontsize=5.4, color=S.INK)
ax.annotate("", xy=(2.45, 14.65), xytext=(2.45, 14.05),
            arrowprops=dict(arrowstyle="-|>", mutation_scale=8, lw=0.8, color=S.INK))
ax.text(2.45, 14.72, "N", ha="center", va="bottom", fontsize=6.5, fontweight="bold")
S.panel(ax, "a", x=-0.01, y=1.0)

# Africa locator
ia = fig.add_axes([0.058, 0.09, 0.13, 0.20])
afr = countries[countries.continent == "Africa"] if "continent" in countries else countries
afr.plot(ax=ia, color="#e6e4df", edgecolor="white", lw=0.25)
basin.plot(ax=ia, color=S.VOLTA, lw=0)
ia.set_xlim(-19, 52); ia.set_ylim(-36, 38); ia.set_aspect("equal")
ia.set_xticks([]); ia.set_yticks([]); ia.set_xlabel(""); ia.set_ylabel("")
for sp in ia.spines.values():
    sp.set_visible(True); sp.set_linewidth(0.4); sp.set_color("#8c8c8c")
ia.set_facecolor("white")

# legend
hd = [Patch(fc=KC[k], ec="none", label=f"{cls[k]}, {share[k]:.1f}%") for k in cls]
hd += [Line2D([], [], color="#4f8fcf", lw=1.0, label="River network (≥ 800 km²)"),
       Line2D([], [], color=S.INK, lw=0.9, label="Volta Basin"),
       Line2D([], [], color="#c9c5bd", lw=0.6, label="National border")]
st_h = [Line2D([], [], marker="o", ls="", mfc=S.VOLTA, mec=S.INK, mew=0.6, ms=4.3, label="Origin of a chain"),
        Line2D([], [], marker="^", ls="", mfc="white", mec=S.INK, mew=0.7, ms=4.3, label="Gauged link, no chain"),
        Line2D([], [], marker="s", ls="", mfc="#bdbdbd", mec=S.INK, mew=0.6, ms=3.6, label="No gauged link"),
        Line2D([], [], color=S.NAVY, lw=0.7, marker=">", ms=3, label="Next gauged station downstream")]
fig.legend(handles=hd, loc="upper left", bbox_to_anchor=(0.655, 0.40), title="Köppen–Geiger zone (basin share)",
           title_fontsize=6.2, alignment="left", fontsize=5.9)
fig.legend(handles=st_h, loc="upper left", bbox_to_anchor=(0.655, 0.165), title="GRDC stations, 1961–2007",
           title_fontsize=6.2, alignment="left", fontsize=5.9, ncols=2, columnspacing=0.8)

# ---- (b) station mean discharge, ranked, with the KDE of log10 mean discharge ---------------------
CQ = Path(__file__).resolve().parent / "cache_station_mean_q.csv"   # station means only; daily Q is not redistributed
if "Q" in ssi.columns:
    mq = ssi[ssi.fill_tier == 0].groupby("node_id").Q.mean().sort_values()
    mq.rename("mean_Q_m3s").to_csv(CQ)
else:
    mq = pd.read_csv(CQ, index_col=0).iloc[:, 0].sort_values()
bx = fig.add_axes([0.765, 0.47, 0.215, 0.475])
names = [st.station_name[n].title().replace("(Halcrow)", "").strip() for n in mq.index]
yy = np.arange(len(mq))
for y, n in zip(yy, mq.index):
    m = MK[role[n]]
    bx.hlines(y, 0.7, mq[n], color=GRID if False else "#d9dde2", lw=0.6, zorder=1)
    bx.scatter(mq[n], y, marker=m["marker"], s=m["s"] * 0.9, facecolor=m["fc"], edgecolor=m["ec"],
               linewidths=m["lw"], zorder=3)
bx.set_xscale("log"); bx.set_xlim(0.7, 1200)
bx.set_yticks(yy); bx.set_yticklabels(names, fontsize=5.5)
bx.tick_params(axis="y", length=0, pad=2)
bx.spines["left"].set_visible(False)
med = mq.median()
bx.axvline(med, color=S.INK2, lw=0.6, zorder=0)
bx.text(med * 1.15, 0.0, f"median {med:.1f} m³ s⁻¹", fontsize=5.4, color=S.INK2, va="center", ha="left")
bx.set_xlabel("Mean discharge, 1961–2007 (m³ s⁻¹)")
bx.set_ylim(-0.7, len(mq) - 0.3)
# marginal density
kx = fig.add_axes([0.765, 0.945, 0.215, 0.045], sharex=bx)
xs = np.logspace(np.log10(0.7), np.log10(1200), 300)
dens = gaussian_kde(np.log10(mq.values))(np.log10(xs))
kx.fill_between(xs, 0, dens, color="#cde2fb", lw=0); kx.plot(xs, dens, color=S.EBRO, lw=0.8)
kx.axis("off")
S.panel(bx, "b", x=-0.36, y=1.06)

S.save(fig, "Fig01_study_area")
print({k: round(v, 1) for k, v in share.items()}, "median", round(med, 1))
