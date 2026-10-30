#!/usr/bin/env python3
"""Fig. 10 (identifiability on sparse networks) and Fig. 11 (reanalysis network and observing-system experiment)."""
import json, os
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from statsmodels.stats.proportion import proportion_confint
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.linear_model import LogisticRegression
import pubstyle as S

S.setup()
R = Path(__file__).resolve().parents[1]
TI, TR = R / "output_identifiability/tables", R / "output_reanalysis/tables"
CF, PL = S.NAVY, "#6250d6"          # quantity colours: connected fraction, per-link probability

# =================================================================================== Fig. 10
# Fig. 6 of the article. (a) Connectivity curves 1 - (1 - p)^k with ribbons spanning the 95% interval of p-hat,
# binned observations with Wilson intervals; (b) Ebro thinning draws as density contours with LOWESS medians and
# the Volta with its 95% intervals; (c) power curves with bootstrap ribbons, the Volta and Ebro positions on the
# curve and the opportunities needed for 50% power. No dashed lines: references are solid hairlines or shading.
import pubviz as V
fit = pd.read_csv(TI / "per_link_model_fit.csv")
ns = pd.read_csv(TI / "networks_summary.csv").set_index("network")
deg = pd.read_csv(TI / "ebro_degradation.csv")
pw = pd.read_csv(TI / "ebro_power_by_opportunities.csv")
IN = json.load(open(R / "output_identifiability/identifiability_numbers.json"))
ci = lambda s: [float(v) for v in str(s).strip("[]").split(",")]

fig, axs = plt.subplots(1, 3, figsize=(S.DOUBLE, 70 * S.MM),
                        gridspec_kw=dict(wspace=0.34, left=0.065, right=0.99, bottom=0.16, top=0.86))
ax = axs[0]
kk = np.linspace(0, 34, 300)
for net, col, mk, off in (("Ebro", S.EBRO, "o", -0.25), ("Volta observed", S.VOLTA, "s", 0.25)):
    f = fit[fit.network == net]
    p = ns.loc[net, "p_link"]; plo, phi = ci(ns.loc[net, "p_link_ci"])
    ax.fill_between(kk, 1 - (1 - plo) ** kk, 1 - (1 - phi) ** kk, color=col, alpha=0.16, lw=0, zorder=1)
    ax.plot(kk, 1 - (1 - p) ** kk, color=col, lw=1.2, zorder=2)
    lo, hi = proportion_confint((f.observed_cf * f.n).round(), f.n, method="wilson")
    ax.vlines(f.k_observing + off, lo, hi, color=col, lw=0.8, alpha=0.9, zorder=3)
    ax.scatter(f.k_observing + off, f.observed_cf, s=10 + 30 * np.sqrt(f.n / 250), marker=mk, fc=col, ec="white",
               lw=0.6, zorder=4)
    mk_ = ns.loc[net, "mean_k_observing"]
    ax.plot([mk_], [0.0], marker="^", ms=5, mfc=col, mec="white", mew=0.5, clip_on=False, zorder=5)
ax.set_xlim(0, 34); ax.set_ylim(0, 1.05)
ax.set_xlabel("Observing upstream stations, $k$"); ax.set_ylabel("Connected share of opportunities")
pe_, pv_ = ci(ns.loc["Ebro", "p_link_ci"]), ci(ns.loc["Volta observed", "p_link_ci"])
ax.legend(handles=[Line2D([], [], color=S.EBRO, marker="o", mec="white", lw=1.2, ms=4.5,
                          label=f"Ebro, $\\hat{{p}}$ = {ns.loc['Ebro', 'p_link']:.3f} ({pe_[0]:.2f}–{pe_[1]:.2f})"),
                   Line2D([], [], color=S.VOLTA, marker="s", mec="white", lw=1.2, ms=4.2,
                          label=f"Volta, $\\hat{{p}}$ = {ns.loc['Volta observed', 'p_link']:.3f} ({pv_[0]:.2f}–{pv_[1]:.2f})"),
                   Line2D([], [], ls="", marker="^", ms=4.5, mfc=S.INK2, mec="white", label="mean $k$ per opportunity")],
          loc="lower right", bbox_to_anchor=(1.0, 0.1), title="1 − (1 − $\\hat{p}$)$^{k}$, 95% ribbon",
          title_fontsize=6, alignment="left", fontsize=6)
S.hgrid(ax); S.panel(ax, "a", x=-0.15, y=1.0)
ax.set_title("Connected share rises with observing stations", loc="left", fontsize=7, pad=4)

ax = axs[1]
full = deg[deg["mode"] == "full"].dropna(subset=["mean_k_observing", "cf_opp", "p_link"])
from scipy.stats import gaussian_kde
gx, gy = np.meshgrid(np.linspace(0, 10, 160), np.linspace(0, 0.9, 160))
for col, c, lab in (("cf_opp", CF, "connected share of opportunities"), ("p_link", PL, "per-link probability")):
    xy = np.vstack([full.mean_k_observing, full[col]])
    kde = gaussian_kde(xy, bw_method=0.32)
    z = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    zs = np.sort(z.ravel())[::-1]; cs = np.cumsum(zs) / zs.sum()
    lev = [zs[np.searchsorted(cs, q)] for q in (0.9, 0.6, 0.3)]
    ax.contourf(gx, gy, z, levels=lev + [z.max()], colors=[V.tint(c, 0.82), V.tint(c, 0.66), V.tint(c, 0.5)],
                zorder=1)
    ax.scatter(full.mean_k_observing, full[col], s=2.5, color=c, alpha=0.35, lw=0, rasterized=True, zorder=2)
    lw_ = lowess(full[col], full.mean_k_observing, frac=0.5)
    ax.plot(lw_[:, 0], lw_[:, 1], color=c, lw=1.3, label=lab, zorder=3)
vo = ns.loc["Volta observed"]
lo_cf, hi_cf = proportion_confint(10, 40, method="wilson")
for yv, (lo_, hi_), mk in ((vo.cf_opp, (lo_cf, hi_cf), "s"), (vo.p_link, ci(vo.p_link_ci), "D")):
    ax.vlines(vo.mean_k_observing, lo_, hi_, color=S.VOLTA, lw=1.0, zorder=4)
    ax.scatter([vo.mean_k_observing], [yv], s=28, marker=mk, color=S.VOLTA, ec=S.INK, lw=0.5, zorder=5)
ax.annotate("Volta", (vo.mean_k_observing, vo.cf_opp), xytext=(6, 4), textcoords="offset points", fontsize=6)
ax.annotate("Volta", (vo.mean_k_observing, vo.p_link), xytext=(6, -3), textcoords="offset points", fontsize=6,
            va="top")
ax.set_xlabel("Mean observing upstream stations per opportunity"); ax.set_ylabel("Probability")
ax.set_ylim(0, 0.9); ax.set_xlim(0, 10)
ax.legend(loc="upper left", title="Ebro thinned to 5–33 stations", title_fontsize=6, alignment="left", fontsize=6)
S.hgrid(ax); S.panel(ax, "b", x=-0.15, y=1.0)
ax.set_title("Thinning lowers the share but not $\\hat{p}$", loc="left", fontsize=7, pad=4)

ax = axs[2]
d = deg[deg.n_opportunities > 0]
xx = np.linspace(np.log(3), np.log(3000), 200)
rng = np.random.default_rng(2026)
ax.axhline(0.5, color=S.GRID, lw=0.8, zorder=0)
ax.text(3.3, 0.515, "50%", fontsize=6, color=S.INK2, va="bottom")
xlog = np.log(d.n_opportunities.values).reshape(-1, 1)
for col, lab, c, pcol in (("p_cf_opp", "connected share", CF, "power_cf_opp"),
                          ("p_p_link", "per-link probability", PL, "power_p_link")):
    y = (d[col] < 0.05).astype(int).values
    m = LogisticRegression(C=np.inf).fit(xlog, y)
    boots = []
    for _ in range(300):
        i = rng.integers(0, len(y), len(y))
        boots.append(LogisticRegression(C=np.inf).fit(xlog[i], y[i]).predict_proba(xx.reshape(-1, 1))[:, 1])
    lo_b, hi_b = np.percentile(boots, [2.5, 97.5], axis=0)
    ax.fill_between(np.exp(xx), lo_b, hi_b, color=c, alpha=0.14, lw=0, zorder=1)
    ax.plot(np.exp(xx), m.predict_proba(xx.reshape(-1, 1))[:, 1], color=c, lw=1.3, label=lab, zorder=2)
    ax.scatter(pw.median_opp, pw[pcol], s=6 + 0.25 * pw.n, color=c, ec="white", lw=0.5, zorder=3)
n50 = IN["N50"]["p_cf_opp"]
pa = IN["power_at"]["p_cf_opp"]
ax.axvspan(40, n50, color=V.tint(S.VOLTA, 0.88), lw=0, zorder=0)
ax.plot([n50], [0.5], marker="o", ms=4.5, mfc="white", mec=CF, mew=0.9, zorder=5)
ax.annotate(f"{n50:.0f} for 50%", (n50, 0.5), xytext=(-5, 8), textcoords="offset points", ha="right", fontsize=6)
for xv, lab, c, ha, dx in ((40, "Volta, 40", S.VOLTA, "left", 5), (928, "Ebro, 928", S.EBRO, "right", -5)):
    yv = pa[str(xv)]
    ax.plot([xv], [yv], marker="o", ms=5, mfc=c, mec="white", mew=0.6, zorder=6)
    ax.annotate(lab, (xv, yv), xytext=(0, 20 if xv == 40 else -22), textcoords="offset points", ha="center",
                fontsize=6, color=S.INK, va="center",
                arrowprops=dict(arrowstyle="-", color=S.INK2, lw=0.5, shrinkA=2, shrinkB=3.5))
ax.set_xscale("log"); ax.set_ylim(0, 1); ax.set_xlim(3, 3000)
ax.set_xlabel("Opportunities"); ax.set_ylabel("Power, α = 0.05")
ax.legend(loc="upper left", title="test statistic (95% bootstrap ribbon)", title_fontsize=6, alignment="left",
          fontsize=6)
S.hgrid(ax); S.panel(ax, "c", x=-0.15, y=1.0)
ax.set_title("Power grows with opportunities", loc="left", fontsize=7, pad=4)
S.save(fig, "Fig10_identifiability")


# =================================================================================== Fig. 11
MAIN = json.load(open(R / "output_reanalysis/reanalysis_numbers.json"))["main_network"]
nodes = pd.read_csv(R / f"data/reanalysis/{MAIN}/nodes.csv").set_index("node_id")
cn = pd.read_csv(R / f"data/reanalysis/{MAIN}/upstream_connectivity.csv", sep=";").set_index("node_id")
org = pd.read_csv(TR / "origin_events_opportunities.csv")
cfn = org[org.n_observing > 0].groupby("origin_node").connected.mean()
basin = gpd.read_file(R / "data/GRDC_VOLTA_POLITICAL_TRANSBOUNDARY_BASIN.shp").to_crs(4326)
countries = gpd.read_file(R / "data/external/naturalearth_lowres/naturalearth_lowres.shp")
rch = pd.read_parquet(R / "data/reanalysis/reaches_volta.parquet")
rr = rch[rch.area_km2 >= 1000].set_index("LINKNO")
segs, lws = [], []
for lk, r in rr.iterrows():
    if r.DSLINKNO in rr.index:
        dd = rr.loc[r.DSLINKNO]
        segs.append([(r.lon, r.lat), (dd.lon, dd.lat)])
        lws.append(0.2 + 0.8 * (np.log10(r.area_km2) - 3) / (np.log10(4e5) - 3))

import panel_ose as PO
PO.style()                      # Fig. 11 follows the line-profile look of panel (c) throughout
fig = plt.figure(figsize=(S.DOUBLE, 124 * S.MM))
ax = fig.add_axes([0.0, 0.03, 0.47, 0.95])
ax.add_collection(LineCollection(segs, linewidths=lws, colors="#c3d9f1", zorder=1))
countries.boundary.plot(ax=ax, color="#cfcbc3", lw=0.4, zorder=1)
basin.boundary.plot(ax=ax, color=S.INK, lw=0.8, zorder=2)
for n, r in cn.iterrows():
    if pd.notna(r.immediate_downstream) and int(r.immediate_downstream) in nodes.index:
        a, b = nodes.loc[n], nodes.loc[int(r.immediate_downstream)]
        ax.plot([a.lon, b.lon], [a.lat, b.lat], color="#8f9aa6", lw=0.45, zorder=2)
orig = nodes.index.isin(cfn.index)
sc = ax.scatter(nodes.lon[orig], nodes.lat[orig], c=cfn.reindex(nodes.index[orig]).values, cmap=S.BLUES, vmin=0.3,
                vmax=1.0, s=13, ec=S.INK, lw=0.3, zorder=4)
ax.scatter(nodes.lon[~orig], nodes.lat[~orig], s=7, fc="white", ec=S.INK2, lw=0.4, zorder=3)
g = nodes[nodes.kind == "gauge"]
ax.scatter(g.lon, g.lat, s=30, marker="^", fc="none", ec=S.VOLTA, lw=0.9, zorder=5)
minx, miny, maxx, maxy = basin.total_bounds
ax.set_xlim(minx - 0.2, maxx + 0.25); ax.set_ylim(miny - 0.2, maxy + 0.1)
ax.set_aspect(1 / np.cos(np.radians(10.5)))
ax.set_xticks(range(-5, 3, 2)); ax.set_yticks(range(6, 15, 2))
ax.set_xticklabels([f"{abs(v)}°{'W' if v < 0 else 'E' if v > 0 else ''}" for v in range(-5, 3, 2)])
ax.set_yticklabels([f"{v}°N" for v in range(6, 15, 2)])
ax.tick_params(direction="in", top=True, right=True, length=1.8, labelsize=5.6)
for sp in ax.spines.values():
    sp.set_visible(True)
ax.set_xlabel(""); ax.set_ylabel("")
cax = fig.add_axes([0.06, 0.135, 0.15, 0.012])
cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
cb.set_label("Connected share of opportunities,\norigin nodes, 1961–2020", fontsize=5.8, labelpad=2)
cb.ax.xaxis.set_label_position("top")
cb.ax.tick_params(labelsize=5.4, length=1.5); cb.outline.set_linewidth(0.4)
ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=S.INK2, ms=3.2, label="headwater node"),
                   Line2D([], [], marker="^", ls="", mfc="none", mec=S.VOLTA, ms=4.5, mew=0.9, label="GRDC gauge"),
                   Line2D([], [], color="#8f9aa6", lw=0.6, label="next node downstream"),
                   Line2D([], [], color="#c3d9f1", lw=1.0, label="channel ≥ 1,000 km²")],
          loc="upper right", fontsize=5.6, bbox_to_anchor=(1.0, 0.995))
ax.set_title("GEOGLOWS v2 reanalysis node network", fontsize=7.5, fontweight="normal", pad=5, loc="left")
S.panel(ax, "a", x=-0.01, y=1.005)

# (b) chain size
bx = fig.add_axes([0.585, 0.70, 0.395, 0.25])
eb = pd.read_csv(R / os.environ.get("ORIGINAL_CODE", "third_party/Code_v.1") /
                 "revision_analyses/S8_national_catalogue/chains_national_validation.csv")
rc = pd.read_csv(TR / "Chains_final.csv")
bins = np.arange(2, 13)


def share(cs):
    cs = np.minimum(np.asarray(cs), 12)
    return np.array([(cs == b).mean() for b in bins])


wd = 0.26
for k, (vals, c, lab) in enumerate(((eb.chain_size, S.EBRO, f"Ebro, gauges (n = {len(eb)})"),
                                    (rc.chain_size, S.REAN, f"Volta, reanalysis (n = {len(rc):,})"),
                                    ([2] * 10, S.VOLTA, "Volta, gauges (n = 10)"))):
    bx.bar(bins + (k - 1) * wd, share(vals), width=wd * 0.92, color=c, lw=0, label=lab)
bx.set_xticks(bins); bx.set_xticklabels([str(b) for b in bins[:-1]] + ["≥12"])
bx.set_xlabel("Chain size (stations)"); bx.set_ylabel("Share of chains")
bx.set_ylim(0, 1.02)
bx.legend(loc="upper right")
S.hgrid(bx); bx.set_title("Chain size", fontsize=7.5, fontweight="normal", pad=5, loc="left")
S.panel(bx, "b", x=-0.12, y=1.0)

# (c) observing-system experiment, line-profile style (panel_ose.py)
cx = fig.add_axes([0.625, 0.255, 0.355, 0.255])
PO.draw(cx, fs=6.8, legend_kw=dict(bbox_to_anchor=(-0.17, -0.47), ncols=2, columnspacing=1.2, handlelength=2.3))
S.panel(cx, "c", x=-0.14, y=1.07)
S.save(fig, "Fig11_reanalysis")
print("done")
