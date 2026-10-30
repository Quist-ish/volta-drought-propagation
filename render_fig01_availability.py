#!/usr/bin/env python3
"""
Fig. 1 (study area: Koppen-Geiger zones, stations, gauged links, borders;
distribution of station mean discharge) and Fig. S12 (station-by-year data
availability), in the layout of Crespillo López et al. (2026), Fig. 1.

Koppen-Geiger zones: 100-arc-second map of Rubel et al. (2017) via kgcpy.
Borders: Natural Earth 1:110m countries.
Author: I. Quist (NUIST)
"""
import json
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from shapely.geometry import Point
import kgcpy

T, O = Path("output_observational/tables"), Path("output_observational/figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.6,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
INK, INK2 = "#0b0b0b", "#52514e"

st = pd.read_csv("data/grdc_stations_raw.csv").set_index("grdc_no")
ssi = pd.read_csv(T / "SSI_daily.csv", parse_dates=["date"])
nodes = sorted(ssi.node_id.unique())
conn = pd.read_csv(T / "upstream_connectivity.csv", sep=";")
chains = pd.read_csv(T / "Chains_final.csv", parse_dates=["chain_start"])
basin = gpd.read_file("data/GRDC_VOLTA_POLITICAL_TRANSBOUNDARY_BASIN.shp").to_crs(4326)
poly = basin.union_all()
countries = gpd.read_file("data/external/naturalearth_lowres/naturalearth_lowres.shp")

# --- Koppen-Geiger grid -----------------------------------------------------
g = 0.05
minx, miny, maxx, maxy = poly.bounds
cells = [(y, x) for y in np.arange(miny + g / 2, maxy, g) for x in np.arange(minx + g / 2, maxx, g)
         if poly.contains(Point(x, y))]
zone = np.array([kgcpy.lookupCZ(y, x) for y, x in cells])
groups = {"Aw": "Tropical savanna (Aw)", "Am": "Tropical monsoon (Am)",
          "BSh": "Hot semi-arid steppe (BSh)", "BWh": "Hot desert (BWh)"}
lab = np.array([groups.get(z, "Other") for z in zone])
w = np.cos(np.radians([y for y, _ in cells]))
share = (pd.Series(w).groupby(lab).sum() / w.sum() * 100).round(1).sort_values(ascending=False)
# validated categorical slots (references/palette.md, light mode), in area order
slots = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
colour = {k: slots[i] for i, k in enumerate(share.index)}

fig = plt.figure(figsize=(7.2, 4.6))
ax = fig.add_axes([0.04, 0.08, 0.58, 0.86])
# zones as a raster (no cell edges), masked outside the basin
from matplotlib.colors import ListedColormap
xg = np.arange(minx + g / 2, maxx, g); yg = np.arange(miny + g / 2, maxy, g)
grid = np.full((len(yg), len(xg)), np.nan)
code = {k: i for i, k in enumerate(share.index)}
for (y, x), l in zip(cells, lab):
    grid[int(round((y - yg[0]) / g)), int(round((x - xg[0]) / g))] = code[l]
ax.imshow(np.ma.masked_invalid(grid), origin="lower", extent=(minx, minx + g * len(xg), miny, miny + g * len(yg)),
          cmap=ListedColormap([colour[k] for k in share.index]), vmin=-0.5, vmax=len(share) - 0.5,
          alpha=0.42, interpolation="nearest", zorder=0, rasterized=True)
countries.boundary.plot(ax=ax, color="#9a9890", lw=0.5, ls="--")
basin.boundary.plot(ax=ax, color=INK, lw=1.0)

# gauged links: immediate downstream pointer among the 18 stations
for _, r in conn.iterrows():
    if pd.notna(r.immediate_downstream):
        a, b = int(r.node_id), int(r.immediate_downstream)
        ax.annotate("", xy=(st.lon[b], st.lat[b]), xytext=(st.lon[a], st.lat[a]),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color=INK2, shrinkA=4, shrinkB=4,
                                    mutation_scale=6))
origins = set(chains.origin_node)
# label offsets (points) for stations that sit close together
LABEL_OFFSET = {1531050: (-4, 3, "right"), 1931400: (-4, -7, "right"), 1531600: (4, 3, "left"),
                1531650: (-4, 3, "right"), 1531430: (-4, 3, "right"), 1531420: (-4, -10, "right"),
                1531550: (4, -2, "left"), 1731400: (4, 2, "left"), 1731450: (4, 2, "left"),
                1531680: (-4, 3, "right"), 1931880: (4, 2, "left")}
linked = {int(n) for n in conn.node_id[conn.n_upstream > 0]} | \
         {int(u) for s in conn.upstream_chain.dropna() for u in str(s).split(",") if u}
for n in nodes:
    kind = "o" if n in origins else ("^" if n in linked else "s")
    ax.scatter(st.lon[n], st.lat[n], s=26 if n in origins else 18, marker=kind,
               facecolor="white" if n not in origins else INK, edgecolor=INK, lw=0.8, zorder=5)
    dx, dy, ha = LABEL_OFFSET.get(int(n), (3, 2, "left"))
    ax.annotate(st.station_name[n].title().replace("(Halcrow)", ""), (st.lon[n], st.lat[n]),
                xytext=(dx, dy), textcoords="offset points", fontsize=5.6, color=INK, zorder=6, ha=ha)
# country names placed by hand inside each country's share of the basin
for c, (x, y) in {"Burkina Faso": (-1.4, 11.8), "Ghana": (-0.8, 7.0), "Togo": (0.85, 7.6),
                  "Benin": (1.7, 9.95), "Mali": (-3.1, 13.85), "Côte d'Ivoire": (-3.2, 8.5)}.items():
    ax.text(x, y, c, fontsize=6.4, color=INK2, style="italic", ha="center")
ax.set_xlim(minx - 0.3, maxx + 0.3); ax.set_ylim(miny - 0.2, maxy + 0.2); ax.set_aspect("equal")
ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
ax.text(0.01, 0.99, "(a)", transform=ax.transAxes, va="top", fontweight="bold", fontsize=9)
handles = [mpatches.Patch(facecolor=colour[k], edgecolor=colour[k], alpha=0.6, label=f"{k}, {share[k]:.1f}%")
           for k in share.index]
handles += [Line2D([], [], marker="o", ls="", mfc=INK, mec=INK, ms=4.5, label="Origin of a chain"),
            Line2D([], [], marker="^", ls="", mfc="white", mec=INK, ms=4.5, label="Gauged link, no chain"),
            Line2D([], [], marker="s", ls="", mfc="white", mec=INK, ms=4, label="No gauged link"),
            Line2D([], [], color=INK2, lw=0.7, label="Next gauged station downstream")]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.665, 0.06), fontsize=6.2, frameon=False,
           title="Köppen–Geiger zone (share of basin) and stations", title_fontsize=6.4, alignment="left")

# (b) distribution of mean discharge
bx = fig.add_axes([0.71, 0.52, 0.27, 0.40])
mq = ssi[ssi.fill_tier == 0].groupby("node_id").Q.mean()
lq = np.log10(mq.values)
bx.hist(lq, bins=np.arange(0.25, 2.9, 0.25), color="#86b6ef", edgecolor="#fcfcfb", lw=1.2)
xs = np.linspace(0.2, 2.9, 200)
kde = gaussian_kde(lq)(xs) * len(lq) * 0.25
bx.plot(xs, kde, color="#1c5cab", lw=1.6)
bx.axvline(np.log10(mq.median()), color=INK, lw=0.8, ls="--")
bx.text(np.log10(mq.median()) + 0.04, bx.get_ylim()[1] * 0.92, f"median {mq.median():.1f} m³ s⁻¹",
        fontsize=6, color=INK)
bx.set_xlabel("log₁₀ mean discharge (m³ s⁻¹)"); bx.set_ylabel("Stations")
bx.spines[["top", "right"]].set_visible(False)
bx.text(-0.18, 1.06, "(b)", transform=bx.transAxes, fontweight="bold", fontsize=9)
for p in ("pdf", "png"):
    fig.savefig(O / f"fig01_study_area.{p}", dpi=400, facecolor="white")
plt.close(fig)

# --- Fig. S12: availability ----------------------------------------------------
order = sorted(nodes, key=lambda n: (str(st.subbasin.get(n, "")), -st.area_km2[n]))
yrs = np.arange(1961, 2008)
mat = np.zeros((len(order), len(yrs)))
obs = ssi[ssi.fill_tier.isin([0, 1])].assign(y=lambda d: d.date.dt.year)
cnt = obs.groupby(["node_id", "y"]).size()
for i, n in enumerate(order):
    for j, y in enumerate(yrs):
        mat[i, j] = cnt.get((n, y), 0) / (366 if y % 4 == 0 else 365)
fig, ax = plt.subplots(figsize=(7.2, 3.6))
im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest",
               extent=(yrs[0] - 0.5, yrs[-1] + 0.5, len(order) - 0.5, -0.5))
for _, c in chains.iterrows():
    i = order.index(int(c.origin_node))
    ax.plot(c.chain_start.year + c.chain_start.dayofyear / 365, i, marker="o", ms=4.5,
            mfc="#eb6834", mec=INK, mew=0.6, ls="")
ax.set_yticks(range(len(order)))
ax.set_yticklabels([f"{st.station_name[n].title()} ({st.river[n].title()})" for n in order], fontsize=6)
ax.set_xlabel("Year")
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("Share of days with data", fontsize=7)
ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="#eb6834", mec=INK, ms=4.5,
                          label="Onset of a reconstructed chain (origin station)")],
          loc="upper left", fontsize=6, frameon=True, edgecolor="#d0cfc9")
fig.tight_layout()
for p in ("pdf", "png"):
    fig.savefig(O / f"figS12_data_availability.{p}", dpi=400, facecolor="white")
json.dump({"koppen_pct": share.to_dict(), "n_cells": len(cells)},
          open(T / "fig01_koppen_shares.json", "w"), indent=2)
print(share)
