"""
Figures for the regime shift and trends (article Fig. 3), teleconnections and synchrony (Fig. 4),
projected drought risk and the monitoring horizon (Fig. 8), and supplementary panels.

Inputs: output_climate/tables, output_climate/figdata, output_projection/tables,
output_observational/tables/r5_*.  Run: python figs_climate_projection.py [regime tele proj supp]
Author: I. Quist (NUIST)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import pubstyle as S

R = Path(__file__).resolve().parents[1]
TC, FDc = R / "output_climate/tables", R / "output_climate/figdata"
TP = R / "output_projection/tables"
TO = R / "output_observational/tables"
S.setup()
NORTH, SOUTH = "#8a5a00", "#0b6e8a"          # catchment groups (validated pair, labelled)
SSP = {"ssp245": "#3a7dc9", "ssp585": "#b8452a"}


def _title(ax, s):
    ax.set_title(s, loc="left", fontsize=7, color=S.INK, pad=3)


# ------------------------------------------------------------------ Fig. 3 regime shift
def fig_regime():
    P = pd.read_csv(TC / "precip_catchments_basin.csv", index_col=0, parse_dates=True)
    ann = P["basin"].resample("YS").sum(); ann.index = ann.index.year
    tr = pd.read_csv(TC / "trend_changepoint_summary.csv").set_index("series")
    b = tr.loc["Basin annual precipitation"]
    sy = tr.loc["Drought synchrony (share of 18 catchments with SPI-6 <= -1)"]
    S6 = pd.read_csv(TC / "spi6_catchments_basin.csv", index_col=0, parse_dates=True).drop(columns="basin")
    sync = (S6 <= -1).mean(axis=1).dropna(); sync = sync.groupby(sync.index.year).mean()
    ct = pd.read_csv(TC / "trend_changepoint_catchments.csv")
    rc = pd.read_csv(TC / "runoff_coefficient_by_year.csv")
    reg = json.load(open(TC / "runoff_coefficient_regression.json"))
    ref = pd.read_csv(TC / "gauge_reference_regime.csv")

    fig = plt.figure(figsize=(S.DOUBLE, 112 * S.MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], width_ratios=[1.25, 1, 1], hspace=0.46, wspace=0.42,
                          left=0.07, right=0.985, top=0.95, bottom=0.08)
    # a basin precipitation
    ax = fig.add_subplot(gs[0, :2])
    cp = int(b.pettitt_year)
    ax.bar(ann.index, ann.values, width=0.8, color=np.where(ann.index <= cp, "#c9d6e6", "#e9c8b8"), lw=0, zorder=2)
    ax.plot(ann.index, ann.rolling(11, center=True).mean(), color=S.INK, lw=0.9, zorder=3, label="11-year running mean")
    ax.hlines(b.mean_before, ann.index.min() - 0.5, cp + 0.5, color=S.EBRO, lw=1.1, zorder=4)
    ax.hlines(b.mean_after, cp + 0.5, ann.index.max() + 0.5, color=S.VOLTA, lw=1.1, zorder=4)
    ax.axvline(cp + 0.5, color=S.INK2, lw=0.6, zorder=4)
    ax.text(cp + 1.2, 1340, f"change point {cp}\nPettitt p = {b.pettitt_p:.3f}; SNHT p = {b.snht_p:.3f}", fontsize=6, color=S.INK2, va="top")
    ax.text(1952, b.mean_before + 18, f"{b.mean_before:,.0f} mm", color=S.EBRO, fontsize=6)
    ax.text(2008, b.mean_after + 18, f"{b.mean_after:,.0f} mm", color=S.VOLTA, fontsize=6)
    ax.set_ylim(600, 1440); ax.set_xlim(1949.5, 2020.5)
    ax.set_ylabel("Basin precipitation (mm yr$^{-1}$)")
    S.hgrid(ax); S.panel(ax, "a"); _title(ax, "Basin rainfall, GPCC 1951–2020")
    # gauge record density strip
    sd = pd.read_csv(TO / "SSI_daily.csv", usecols=["node_id", "date", "SSI"], parse_dates=["date"]).dropna(subset=["SSI"])
    dens = sd.groupby(sd.date.dt.year).node_id.nunique().reindex(range(1951, 2021), fill_value=0)
    ins = ax.inset_axes([0.0, 0.955, 1.0, 0.045], transform=ax.transAxes)
    ins.imshow(dens.values[None, :], aspect="auto", cmap="pub_blues", vmin=0, vmax=18,
               extent=[1950.5, 2020.5, 0, 1])
    ins.set_xlim(ax.get_xlim()); ins.axis("off")
    ax.text(2020.3, 1352, "strip: gauges with defined SSI per year (0–18)", ha="right", fontsize=6, color=S.INK2, va="top")
    # b synchrony
    ax = fig.add_subplot(gs[0, 2])
    cps = int(sy.pettitt_year)
    ax.plot(sync.index, sync.values, color=S.NAVY, lw=0.8)
    ax.hlines(sy.mean_before, 1951, cps, color=S.EBRO, lw=1.1)
    ax.hlines(sy.mean_after, cps, 2020, color=S.VOLTA, lw=1.1)
    ax.axvline(cps + 0.5, color=S.INK2, lw=0.6)
    ax.text(1988, 0.735, f"break {cps}, p < 0.001\nmean {sy.mean_before:.2f} → {sy.mean_after:.2f}", fontsize=6, color=S.INK2, va="top")
    ax.set_ylim(0, 0.75); ax.set_xlim(1951, 2020)
    ax.set_ylabel("Share of catchments in drought")
    S.hgrid(ax); S.panel(ax, "b"); _title(ax, "Drought synchrony (SPI-6 ≤ −1)")
    # c catchment change points
    ax = fig.add_subplot(gs[1, 0])
    ct = ct.sort_values("pettitt_year")
    pct = 100 * (ct.mean_after / ct.mean_before - 1)
    y = np.arange(len(ct))
    sig = ct.pettitt_q_bh < 0.05
    ax.scatter(ct.pettitt_year[sig], y[sig], s=np.clip(-pct[sig], 2, 30) * 2.2, color=S.NAVY, zorder=3,
               label="FDR q < 0.05")
    ax.scatter(ct.pettitt_year[~sig], y[~sig], s=np.clip(-pct[~sig], 2, 30) * 2.2, facecolor="white",
               edgecolor=S.NAVY, lw=0.6, zorder=3, label="not significant")
    ax.set_yticks(y); ax.set_yticklabels([s.title() for s in ct.station], fontsize=5.8)
    ax.set_xlim(1964, 1993); ax.set_xlabel("Change-point year (Pettitt); % = change in mean")
    for yy, (x, p) in enumerate(zip(ct.pettitt_year, pct)):
        ax.text(x + 0.8, yy, f"{p:+.0f}%", va="center", fontsize=5.6, color=S.INK2)
    ax.legend(loc="lower right", fontsize=6, handletextpad=0.2)
    S.hgrid(ax, "x"); S.panel(ax, "c"); _title(ax, "Catchment rainfall: break and change")
    # d runoff at fixed rainfall
    ax = fig.add_subplot(gs[1, 1])
    import statsmodels.formula.api as smf
    d = rc[(rc.R_mm > 0) & (rc.P_mm > 0)].copy()
    d["lnR"], d["lnP"] = np.log(d.R_mm), np.log(d.P_mm)
    f = smf.ols("lnR ~ C(station) + lnP", data=d).fit()
    d["res"] = f.resid
    north = d.lat_c > 10.5
    ax.scatter(d.year[north], d.res[north], s=6, color=NORTH, alpha=0.8, lw=0, label="catchment north of 10.5° N")
    ax.scatter(d.year[~north], d.res[~north], s=6, color=SOUTH, alpha=0.8, lw=0, label="south of 10.5° N")
    g = reg["all"]
    xs = np.array([1961, 2007]); beta = np.log(1 + g["runoff_change_pct_per_decade"] / 100) / 10
    yc = d.res.mean() + beta * (xs - d.year.mean())
    ax.plot(xs, yc, color=S.INK, lw=1.0)
    ax.text(1962, 1.55, f"+{g['runoff_change_pct_per_decade']:.1f}% per decade\n(95% CI {g['lo95']:.1f}–{g['hi95']:.1f}; "
                        f"{g['gauges']} gauges)", fontsize=6, color=S.INK, va="top")
    ax.set_xlim(1960, 2008); ax.set_ylim(-1.9, 1.7)
    ax.set_xlabel("Year"); ax.set_ylabel("ln runoff, rainfall-adjusted residual")
    ax.legend(loc="lower right", fontsize=5.8, handletextpad=0.1, markerscale=1.3)
    S.hgrid(ax); S.panel(ax, "d"); _title(ax, "Runoff for a given rainfall")
    # e reference regime of each gauge
    ax = fig.add_subplot(gs[1, 2])
    r = ref[ref.years_defined >= 3].sort_values("mean_spi12_reference")
    cols = [S.VOLTA if v < 0 else S.EBRO for v in r.mean_spi12_reference]
    ax.barh(np.arange(len(r)), r.mean_spi12_reference, color=cols, height=0.7)
    ax.set_yticks(np.arange(len(r))); ax.set_yticklabels([s.title() for s in r.station], fontsize=5.8)
    ax.axvline(0, color=S.INK2, lw=0.5)
    S12 = pd.read_csv(TC / "spi12_catchments_basin.csv", index_col=0, parse_dates=True)["basin"]
    cref = float(S12[(S12.index.month == 12) & (S12.index.year >= 1975) & (S12.index.year <= 1992)].mean())
    ax.axvline(cref, color=S.INK, lw=0.7)
    ax.text(cref + 0.02, len(r) - 0.7, f"common\nreference\n1975–1992\n({cref:+.2f})", ha="left", va="top", fontsize=5.6, color=S.INK)
    ax.set_xlabel("Mean SPI-12 over the years\nthat define the gauge's SSI")
    ax.set_xlim(-0.65, 0.3)
    S.hgrid(ax, "x"); S.panel(ax, "e"); _title(ax, "Climate of each SSI reference")
    S.save(fig, "Fig03_regime_shift")


# ------------------------------------------------------------------ Fig. 4 teleconnections
VEIL = dict(color="white", alpha=0.62, lw=0)        # outside the cone of influence: veiled, no boundary line


def _wct_panel(ax, k, lab, cmap="pub_blues"):
    z = np.load(FDc / f"wct_spi12_{k}.npz")
    t, per, W, coi, sig, ph = z["t"], z["period"], z["WCT"], z["coi"], z["sig"], z["aWCT"]
    m = ax.pcolormesh(t, per, W, cmap=cmap, vmin=0, vmax=1, shading="auto", rasterized=True)
    ax.contour(t, per, W / sig[:, None], [1], colors=S.INK, linewidths=0.7, linestyles="solid")
    ax.fill_between(t, coi, 64, **VEIL)
    # phase arrows where significant, thinned
    sel_t = np.arange(0, len(t), 30); sel_p = np.arange(0, len(per), 8)
    for i in sel_p:
        for j in sel_t:
            if W[i, j] >= sig[i] and per[i] <= coi[j] and 1.5 <= per[i] <= 32:
                a = ph[i, j]
                ax.annotate("", xy=(t[j] + 1.6 * np.cos(a), per[i] * 2 ** (0.12 * np.sin(a))), xytext=(t[j], per[i]),
                            arrowprops=dict(arrowstyle="-|>", lw=0.45, color=S.INK, mutation_scale=4))
    ax.set_yscale("log", base=2); ax.set_ylim(32, 1); ax.set_yticks([1, 2, 4, 8, 16, 32]); ax.set_yticklabels([1, 2, 4, 8, 16, 32])
    ax.set_xlim(t.min(), t.max()); ax.set_ylabel("Period (years)")
    _title(ax, lab)
    return m


def _cbar(fig, ax, m, ticks, labels, title, box=(0.035, 0.075, 0.30, 0.035)):
    cax = ax.inset_axes(list(box))
    cb = fig.colorbar(m, cax=cax, orientation="horizontal")
    cb.set_ticks(ticks); cb.set_ticklabels(labels); cb.ax.tick_params(labelsize=6, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4); cax.set_title(title, fontsize=6, pad=1.5, color=S.INK)


def fig_tele():
    """Fig. 4. Wavelet panels veil the region outside the cone of influence without a boundary line; the global
    spectrum fills the red-noise 95% level and the exceedance; event-rate ratios are gradient intervals (opacity
    following the confidence curve of the log-normal confidence distribution); onset synchrony is a raincloud."""
    import pubviz as V
    cw = np.load(FDc / "cwt_basin_spi12.npz")
    gl = pd.read_csv(TC / "event_rate_glm.csv")
    so = pd.read_csv(TC / "synchrony_at_onset.csv")
    ss = json.load(open(TC / "synchrony_summary.json"))
    fig = plt.figure(figsize=(S.DOUBLE, 124 * S.MM))
    g1 = fig.add_gridspec(1, 3, width_ratios=[1.55, 0.42, 1.55], left=0.065, right=0.985, top=0.95, bottom=0.575,
                          wspace=0.12)
    g2 = fig.add_gridspec(1, 4, width_ratios=[1.0, 0.66, 0.62, 1.45], left=0.09, right=0.985, top=0.43, bottom=0.075,
                          wspace=0.2)
    # a CWT of basin SPI-12
    ax = fig.add_subplot(g1[0, 0])
    t, per, pw, coi, s95 = cw["t"], cw["period"], cw["power"], cw["coi"], cw["sig95"]
    m = ax.pcolormesh(t, per, np.log2(pw), cmap="pub_blues", vmin=-4, vmax=4, shading="auto", rasterized=True)
    ax.contour(t, per, s95, [1], colors=S.INK, linewidths=0.7, linestyles="solid")
    ax.fill_between(t, coi, 64, **VEIL)
    ax.set_yscale("log", base=2); ax.set_ylim(32, 1); ax.set_yticks([1, 2, 4, 8, 16, 32]); ax.set_yticklabels([1, 2, 4, 8, 16, 32])
    ax.set_xlim(t.min(), t.max()); ax.set_ylabel("Period (years)")
    S.panel(ax, "a", x=-0.08); _title(ax, "Wavelet power, basin SPI-12")
    _cbar(fig, ax, m, [-4, 0, 4], ["1/16", "1", "16"], "normalised power")
    # b global wavelet spectrum: red-noise 95% level filled, exceedance highlighted
    ax2 = fig.add_subplot(g1[0, 1], sharey=ax)
    gws, gsig = cw["gws"], cw["gws_sig"]
    ax2.fill_betweenx(per, 1e-3, gsig, color="#e3e6ea", lw=0, zorder=1)
    ax2.fill_betweenx(per, gsig, gws, where=gws > gsig, color=V.tint(S.NAVY, 0.35), lw=0, zorder=2,
                      interpolate=True)
    ax2.text(0.013, 6.5, "95% red\nnoise", fontsize=6, color=S.INK2, va="center", ha="left", linespacing=0.95)
    ax2.plot(gws, per, color=S.NAVY, lw=1.0, zorder=3)
    pk = pd.read_csv(TC / "gws_peaks_basin_spi12.csv")
    for _, r in pk[(pk.significant) & (pk.period_yr > 1.5)].iterrows():
        ax2.annotate(f"{r.period_yr:.1f} yr", (r.power, r.period_yr), xytext=(4, 0), textcoords="offset points",
                     fontsize=6, va="center", color=S.INK)
    ax2.set_xscale("log"); ax2.set_xlim(1e-2, 5e2); ax2.set_xlabel("Power")
    plt.setp(ax2.get_yticklabels(), visible=False)
    S.panel(ax2, "b", x=-0.05)
    # c, f wavelet coherence
    axc = fig.add_subplot(g1[0, 2])
    m2 = _wct_panel(axc, "ATL3", "Coherence with the Atlantic Niño (ATL3)")
    S.panel(axc, "c", x=-0.1)
    _cbar(fig, axc, m2, [0, 0.5, 1], ["0", "0.5", "1"], "coherence")
    axf = fig.add_subplot(g2[0, 3])
    _wct_panel(axf, "AMO", "Coherence with the AMO")
    S.panel(axf, "f", x=-0.1); axf.set_xlabel("Year")
    _cbar(fig, axf, m2, [0, 0.5, 1], ["0", "0.5", "1"], "coherence")
    # d event-rate ratios: gradient intervals on a log axis
    ax = fig.add_subplot(g2[0, 0])
    g = gl.iloc[1:].copy()
    names = {"AMO (per SD)": "AMO", "Nino34 (per SD)": "Niño 3.4 (JAS)", "ATL3 (per SD)": "ATL3 (JJA)",
             "TNA_TSA (per SD)": "TNA − TSA", "DMI (per SD)": "DMI", "PDO (per SD)": "PDO"}
    yy = np.arange(len(g))[::-1]
    tx = fig.add_subplot(g2[0, 1], sharey=ax); tx.axis("off"); tx.set_xlim(0, 1)
    ax.axvline(1, color=S.INK2, lw=0.6, zorder=1)
    for y_, (_, r) in zip(yy, g.iterrows()):
        c = S.VOLTA if r.p_holm < 0.05 else S.NAVY
        sd = (np.log(r.hi95) - np.log(r.lo95)) / (2 * 1.96)
        ends = (r.rate_ratio * np.exp(-2.9 * sd), r.rate_ratio * np.exp(2.9 * sd))
        V.gradient_interval(ax, y_, r.lo95, r.rate_ratio, r.hi95, c if c == S.VOLTA else "#4a5d75", height=0.42,
                            logx=True, q=0.95, ends=ends, n=140, zorder=2, alpha_max=0.6)
        ax.plot([r.lo95, r.hi95], [y_, y_], color=c, lw=0.6, zorder=3, solid_capstyle="butt")
        ax.scatter(r.rate_ratio, y_, s=20, fc=c, ec="white", lw=0.6, zorder=4)
        tx.text(0.02, y_, f"{r.rate_ratio:.2f} ({r.lo95:.2f}–{r.hi95:.2f})", va="center", fontsize=6, color=S.INK)
        tx.text(0.86, y_, f"{r.p_holm:.2f}", va="center", ha="right", fontsize=6,
                color=S.INK if r.p_holm < 0.05 else S.INK2)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 1.6); ax.set_xticks([0.6, 0.8, 1.0, 1.25, 1.5]); ax.set_xticklabels(["0.6", "0.8", "1", "1.25", "1.5"])
    ax.minorticks_off()
    ax.set_yticks(yy); ax.set_yticklabels([names[c] for c in g.covariate])
    ax.set_ylim(-0.6, len(g) - 0.4)
    ax.set_xlabel("Event-rate ratio per SD of the index\n(gauges, 1961–2007)")
    tx.text(0.02, len(g) - 0.45, "ratio (95% CI)", fontsize=6, color=S.INK2, va="bottom")
    tx.text(0.86, len(g) - 0.45, "Holm p", fontsize=6, color=S.INK2, va="bottom", ha="right")
    S.hgrid(ax, "x"); S.panel(ax, "d", x=-0.2, y=1.0)
    # e onset synchrony: raincloud
    ax2b = fig.add_subplot(g2[0, 2])
    o = so[so.opportunity]
    ax2b.axhline(ss["sync_clim_mean_1961_2007"], color=S.INK2, lw=0.6, zorder=1)
    ax2b.text(1.62, ss["sync_clim_mean_1961_2007"] - 0.015, "1961–2007\nmean", fontsize=6, color=S.INK2,
              ha="right", va="top", linespacing=0.95)
    for i, (lab, d, c) in enumerate([("not\nmatched", o[~o.connected], "#8d96a0"), ("matched", o[o.connected], S.VOLTA)]):
        v = d.sync.values
        V.half_violin(ax2b, i - 0.04, v, 0.3, side="left", grid=np.linspace(max(v.min() - 0.03, 0), min(v.max() + 0.03, 1), 200),
                      fc=V.tint(c, 0.75), ec="none", lw=0, zorder=2)
        V.dotstrip(ax2b, i + 0.06, v, binw=0.04, dot_w=0.07, side="right", s=9, fc=c, ec="white", lw=0.4, zorder=3)
        ax2b.plot([i - 0.36, i + 0.36], [np.median(v)] * 2, color=S.INK, lw=1.2, solid_capstyle="butt", zorder=4)
    ax2b.text(0.5, 0.985, f"p = {ss['mw_connected_vs_unconnected_p']:.2f}", ha="center", va="top", fontsize=6,
              color=S.INK2, transform=ax2b.get_xaxis_transform())
    ax2b.set_xticks([0, 1]); ax2b.set_xticklabels(["not\nmatched", "matched"], fontsize=6)
    ax2b.set_xlim(-0.5, 1.65); ax2b.set_ylim(0, 1)
    ax2b.set_ylabel("Drought synchrony at onset")
    S.hgrid(ax2b); S.panel(ax2b, "e", x=-0.32)
    S.save(fig, "Fig04_teleconnections")


# ------------------------------------------------------------------ Fig. 8 projections
def fig_proj():
    """Fig. 8. (a) Raincloud of the basin precipitation change in the 16 models; (b) one dot per model for the
    change in drought months, diamonds the multi-model median (open where fewer than 80% of models agree on the
    sign); (c) the transfer-model skill plane with the acceptance region of the rule fixed before fitting and the
    three member models of each gauge; (d) the monitoring horizon: years for 50% and 80% power in the historical
    climate and gradient intervals (10th-90th percentiles, opacity following the confidence curve) under the two
    scenarios in 2051-2080. No dashed lines: reference levels are solid hairlines or shaded regions."""
    import pubviz as V
    from matplotlib.patches import Rectangle, Patch
    lg = pd.read_csv(TP / "projection_metrics_long.csv", dtype={"unit": str})
    s = pd.read_csv(TP / "projection_summary.csv", dtype={"unit": str})
    pr = pd.read_csv(TP / "projection_basin_precip.csv")
    sk = pd.read_csv(TP / "transfer_model_skill.csv")
    mon = json.load(open(TP / "monitoring_horizon.json"))
    st = pd.read_csv(R / "data/grdc_stations_raw.csv").set_index("grdc_no").station_name.str.title().to_dict()
    COMBO = [("ssp245", "2021-2050"), ("ssp245", "2051-2080"), ("ssp585", "2021-2050"), ("ssp585", "2051-2080")]
    shade = lambda sc, per: V.tint(SSP[sc], 0.5) if per == "2021-2050" else SSP[sc]
    LAB = {"ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}

    fig = plt.figure(figsize=(S.DOUBLE, 132 * S.MM))
    gt = fig.add_gridspec(1, 2, width_ratios=[0.86, 1.75], left=0.075, right=0.985, top=0.875, bottom=0.585,
                          wspace=0.27)
    gb = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], left=0.075, right=0.985, top=0.40, bottom=0.075,
                          wspace=0.2)

    # (a) raincloud of the basin precipitation change, 16 models
    ax = fig.add_subplot(gt[0, 0])
    p = pr[pr.kind == "bias-adjusted"]
    XA = [0, 1, 2.35, 3.35]
    for x0, (sc, per) in zip(XA, COMBO):
        d = p[p.scenario == sc]
        v = (100 * (d[per] / d["1985-2014"] - 1)).values
        assert len(v) == 16
        c = shade(sc, per)
        V.half_violin(ax, x0 - 0.05, v, 0.36, side="left", grid=np.linspace(v.min() - 2, v.max() + 2, 200),
                      fc=V.tint(c, 0.45), ec="none", lw=0, zorder=2)
        V.dotstrip(ax, x0 + 0.06, v, binw=2.0, dot_w=0.085, side="right", s=7, fc=c, ec="white", lw=0.35,
                   zorder=4)
        med = np.median(v)
        ax.plot([x0 - 0.36, x0 + 0.36], [med, med], color=S.INK, lw=1.1, solid_capstyle="butt", zorder=5)
        ax.text(x0, 43, f"{(v > 0).sum()}/16", ha="center", va="top", fontsize=6, color=S.INK)
    ax.axhline(0, color=S.INK2, lw=0.5, zorder=1)
    ax.set_xticks(XA); ax.set_xticklabels(["2021–\n2050", "2051–\n2080"] * 2, fontsize=6)
    for xm, sc in ((0.5, "ssp245"), (2.85, "ssp585")):
        ax.text(xm, -0.25, LAB[sc], transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.2,
                color=S.INK)
        ax.plot([xm - 0.75, xm + 0.75], [-0.205, -0.205], transform=ax.get_xaxis_transform(), color=SSP[sc],
                lw=1.6, solid_capstyle="butt", clip_on=False)
    ax.set_xlim(-0.65, 3.85); ax.set_ylim(-32, 46)
    ax.set_ylabel("Basin precipitation change (%)")
    S.hgrid(ax); S.panel(ax, "a", x=-0.2, y=1.0)

    # (b) change in drought-month share: one dot per model, median diamond, agreement
    ax = fig.add_subplot(gt[0, 1])
    rows = [("SPI-12", "basin", "SPI-12\nbasin"), ("SRI-12", "basin", "SRI-12\nbasin")]
    rows += [("SSI-12 (ensemble)", str(g), f"SSI-12\n{st.get(int(g), g)}") for g in sk[sk.selected].grdc_no]
    DX = [-0.34, -0.12, 0.12, 0.34]
    for j, (ind, u, lab) in enumerate(rows):
        if j % 2 == 0:
            ax.add_patch(Rectangle((j - 0.5, -40), 1.0, 120, fc="#f5f6f7", ec="none", zorder=0))
        for dx, (sc, per) in zip(DX, COMBO):
            d = lg[(lg["index"] == ind) & (lg.unit == u) & (lg.scenario == sc)]
            b = d[d.period == "1985-2014"].set_index("source").p_below_1
            f = d[d.period == per].set_index("source").p_below_1
            ch = 100 * (f - b).dropna().values
            r = s[(s["index"] == ind) & (s.unit == u) & (s.scenario == sc) & (s.period == per) &
                  (s.metric == "p_below_1")].iloc[0]
            assert len(ch) == 16 and abs(np.median(ch) - 100 * r.change_median) < 1e-6
            c = shade(sc, per)
            V.dotstrip(ax, j + dx, ch, binw=2.5, dot_w=0.042, s=4.5, fc=c, ec="none", lw=0, zorder=3)
            agree = r.share_models_increase >= 0.8 or r.share_models_increase <= 0.2
            ax.scatter(j + dx, np.median(ch), s=24, marker="D", fc=SSP[sc] if agree else "white", ec=S.INK,
                       lw=0.6, zorder=5)
    ax.axhline(0, color=S.INK2, lw=0.5, zorder=1)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[2] for r in rows], fontsize=6)
    ax.set_xlim(-0.5, len(rows) - 0.5); ax.set_ylim(-26, 70)
    ax.set_ylabel("Change in drought months\n(index < −1; percentage points)")
    hand = [Line2D([], [], ls="", marker="o", ms=3.4, mfc=shade(sc, per), mec="none",
                   label=f"{LAB[sc]}, {per.replace('-', '–')}") for sc, per in COMBO]
    hand += [Line2D([], [], ls="", marker="D", ms=3.6, mfc=S.INK2, mec=S.INK, mew=0.6, label="median, ≥ 80% agree on sign"),
             Line2D([], [], ls="", marker="D", ms=3.6, mfc="white", mec=S.INK, mew=0.6, label="median, < 80% agree")]
    ax.legend(handles=hand, loc="lower left", ncols=3, fontsize=5.9, handletextpad=0.2, columnspacing=1.0,
              bbox_to_anchor=(0.0, 1.0), borderaxespad=0.15)
    S.hgrid(ax); S.panel(ax, "b", x=-0.075, y=1.0)

    # (c) skill plane: ensemble and member models, acceptance region of the rule fixed before fitting
    ax = fig.add_subplot(gb[0, 0])
    NSE0, RHO0, YT = 0.5, 0.4, 0.8
    ax.add_patch(Rectangle((NSE0, RHO0), 1.08 - NSE0, YT - RHO0, fc=V.tint(S.VOLTA, 0.85), ec="none", zorder=0))
    ax.plot([NSE0, NSE0, 1.08], [YT, RHO0, RHO0], color=V.tint(S.VOLTA, 0.3), lw=0.6, zorder=1)
    ax.text(NSE0 + 0.03, YT - 0.03, "skill rule, fixed\nbefore fitting", ha="left", va="top", fontsize=5.9,
            color=S.INK2, linespacing=0.95)
    for _, r in sk.iterrows():
        c = S.VOLTA if r.selected else "#8d96a0"
        for m in ("ridge", "rf", "xgb"):
            ax.plot([r.nse_log_ensemble, r[f"nse_log_{m}"]], [r.anom_rho_ensemble, r[f"anom_rho_{m}"]],
                    color=V.tint(c, 0.45), lw=0.45, zorder=2)
            ax.scatter(r[f"nse_log_{m}"], r[f"anom_rho_{m}"], s=4, fc=V.tint(c, 0.35), ec="none", zorder=2)
        ax.scatter(r.nse_log_ensemble, r.anom_rho_ensemble, s=22, fc=c, ec="white", lw=0.6, zorder=4)
    OFF = {"LAWRA": (-7, 8, "right"), "SABARI": (-3, 7, "right"), "PORGA": (3, 8, "left"), "TIELE": (5, -3, "left"),
           "NAWUNI": (2, -8, "left"), "PRANG": (-5, 3, "right"), "PWALAGU": (5, -2, "left"),
           "BAMBOI": (5, 3, "left"), "WIASI": (-1, 8, "center"), "YARUGU": (-4, 6, "right")}
    for _, r in sk.iterrows():
        dx, dy, ha = OFF[r.station]
        ax.annotate(r.station.title(), (r.nse_log_ensemble, r.anom_rho_ensemble), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="center", fontsize=6, color=S.INK, zorder=6,
                    path_effects=[__import__("matplotlib.patheffects", fromlist=["x"]).withStroke(linewidth=1.6,
                                                                                                    foreground="white")])
    ax.axhline(0, color=S.INK2, lw=0.4, zorder=1); ax.axvline(0, color=S.INK2, lw=0.4, zorder=1)
    ax.set_xlim(-0.75, 1.08); ax.set_ylim(-0.8, YT)
    ax.set_xlabel("Out-of-fold NSE, log discharge"); ax.set_ylabel("Anomaly correlation ρ")
    ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=4, mfc=S.VOLTA, mec="white", label="projected gauge"),
                       Line2D([], [], ls="", marker="o", ms=4, mfc="#8d96a0", mec="white", label="not projected"),
                       Line2D([], [], color="#b9bfc6", lw=0.6, marker="o", ms=1.8, mfc="#b9bfc6", mec="none",
                              label="member models")],
              loc="upper left", fontsize=5.8, handletextpad=0.3)
    S.hgrid(ax); S.panel(ax, "c", x=-0.17, y=1.0)

    # (d) monitoring horizon
    ax = fig.add_subplot(gb[0, 1])
    des = mon["designs"]
    nice = {"Observed record pattern, 18 gauges": "Observed record pattern (18 gauges, 1961–2007)",
            "18 gauges, continuous records": "The 18 gauges with continuous records",
            "Reanalysis-node density (129), continuous": "Gauges at reanalysis-node density (129), continuous"}
    XMIN, XMAX = 0.5, 1e5
    ax.add_patch(Rectangle((XMIN, -0.62), 46 - XMIN, 3.3, fc="#eef1f4", ec="none", zorder=0))
    for y_, lab in zip([2, 1, 0], list(des)):
        v50, v80 = des[lab]["years_n50"], des[lab]["years_n80"]
        h50, h80 = v50["historical"], v80["historical"]
        yt = y_ + 0.2
        ax.plot([h50, h80], [yt, yt], color=S.INK2, lw=0.7, zorder=3)
        ax.scatter([h50], [yt], s=26, fc=S.NAVY, ec="white", lw=0.6, zorder=5)
        ax.scatter([h80], [yt], s=26, fc="white", ec=S.NAVY, lw=0.8, zorder=5)
        ax.text(h50, yt + 0.13, f"{h50:,.0f} yr", ha="center", va="bottom", fontsize=6, color=S.INK)
        ax.text(h80, yt + 0.13, f"{h80:,.0f} yr", ha="center", va="bottom", fontsize=6, color=S.INK2)
        for k, (sc, yo) in enumerate((("ssp245", -0.06), ("ssp585", -0.24))):
            key = f"{sc}_2051_2080"
            lo, md, hi = v50[f"{key}_p10"], v50[f"{key}_median"], v50[f"{key}_p90"]
            V.gradient_interval(ax, y_ + yo, lo, md, hi, SSP[sc], height=0.13, logx=True, q=0.8, n=160, zorder=3)
            ax.plot([md, md], [y_ + yo - 0.085, y_ + yo + 0.085], color=SSP[sc], lw=1.2, solid_capstyle="butt",
                    zorder=4)
        ax.text(XMIN * 1.15, y_ + 0.52, nice.get(lab, lab), fontsize=6.2, color=S.INK, va="bottom", ha="left")
    ax.set_xscale("log"); ax.set_xlim(XMIN, XMAX); ax.set_ylim(-0.45, 2.8)
    ax.set_xticks([1, 10, 100, 1000, 10000, 100000]); ax.set_xticklabels(["1", "10", "100", "1,000", "10,000", "100,000"])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Years of observation needed to test propagation")
    gl = [Line2D([], [], ls="", marker="o", ms=4.2, mfc=S.NAVY, mec="white", label="50% power, historical climate"),
          Line2D([], [], ls="", marker="o", ms=4.2, mfc="white", mec=S.NAVY, mew=0.8, label="80% power, historical climate"),
          Patch(fc=SSP["ssp245"], ec="none", label="50% power, SSP2-4.5, 2051–2080"),
          Patch(fc=SSP["ssp585"], ec="none", label="50% power, SSP5-8.5, 2051–2080"),
          Patch(fc="#eef1f4", ec="none", label="length of the 1961–2007 record")]
    ax.legend(handles=gl, loc="lower left", ncols=3, fontsize=5.9, handlelength=1.2, columnspacing=1.2,
              bbox_to_anchor=(0.0, 1.0), borderaxespad=0.15)
    S.hgrid(ax, "x"); S.panel(ax, "d", x=-0.02, y=1.0)
    S.save(fig, "Fig10_projection")


# ------------------------------------------------------------------ supplementary figures
def figS_wtc_all():
    idxs = [("AMO", "AMO"), ("Nino34", "Niño 3.4"), ("ATL3", "ATL3"), ("TNA_TSA", "TNA − TSA"), ("DMI", "DMI"), ("PDO", "PDO")]
    fig = plt.figure(figsize=(S.DOUBLE, 150 * S.MM))
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 0.04], hspace=0.5, wspace=0.18, left=0.07, right=0.94, top=0.95, bottom=0.07)
    m = None
    for i, (k, lab) in enumerate(idxs):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        m = _wct_panel(ax, k, f"Basin SPI-12 and {lab}")
        S.panel(ax, "abcdef"[i])
        if i % 2:
            ax.set_ylabel("")
        if i >= 4:
            ax.set_xlabel("Year")
    cax = fig.add_subplot(gs[:, 2])
    cb = fig.colorbar(m, cax=cax); cb.set_label("Wavelet coherence"); cb.outline.set_linewidth(0.4)
    S.save(fig, "FigS_wtc_all")


def figS_sync_wtc():
    idxs = [("AMO", "AMO"), ("Nino34", "Niño 3.4"), ("ATL3", "ATL3"), ("TNA_TSA", "TNA − TSA"), ("DMI", "DMI"), ("PDO", "PDO")]
    fig = plt.figure(figsize=(S.DOUBLE, 150 * S.MM))
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 0.04], hspace=0.5, wspace=0.18, left=0.07, right=0.94, top=0.95, bottom=0.07)
    m = None
    for i, (k, lab) in enumerate(idxs):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        z = np.load(FDc / f"wct_sync_{k}.npz")
        t, per, W, coi, sig = z["t"], z["period"], z["WCT"], z["coi"], z["sig"]
        m = ax.pcolormesh(t, per, W, cmap="pub_blues", vmin=0, vmax=1, shading="auto", rasterized=True)
        ax.contour(t, per, W / sig[:, None], [1], colors=S.INK, linewidths=0.6)
        ax.fill_between(t, coi, 64, **VEIL)
        ax.set_yscale("log", base=2); ax.set_ylim(32, 1); ax.set_yticks([1, 2, 4, 8, 16, 32]); ax.set_yticklabels([1, 2, 4, 8, 16, 32])
        ax.set_xlim(t.min(), t.max())
        _title(ax, f"Drought synchrony and {lab}"); S.panel(ax, "abcdef"[i])
        if i % 2 == 0:
            ax.set_ylabel("Period (years)")
        if i >= 4:
            ax.set_xlabel("Year")
    cax = fig.add_subplot(gs[:, 2])
    cb = fig.colorbar(m, cax=cax); cb.set_label("Wavelet coherence"); cb.outline.set_linewidth(0.4)
    S.save(fig, "FigS_sync_wtc")


def figS_lagcorr():
    lc = pd.read_csv(TC / "lag_correlation.csv")
    names = {"AMO": "AMO", "Nino34": "Niño 3.4", "ATL3": "ATL3", "TNA_TSA": "TNA − TSA", "DMI": "DMI", "PDO": "PDO"}
    cols = dict(zip(names, ["#2a78d6", "#b8452a", "#1baf7a", "#8a5a00", "#7a5fb0", "#4d4d4d"]))
    fig, axs = plt.subplots(1, 2, figsize=(S.DOUBLE, 62 * S.MM), sharey=True)
    for ax, (ser, lab) in zip(axs, [("basin SPI-12", "Basin SPI-12"), ("synchrony", "Drought synchrony (12-month mean)")]):
        d = lc[lc.series == ser]
        for k in names:
            e = d[d["index"] == k].sort_values("lag_months")
            ax.plot(e.lag_months, e.r, color=cols[k], lw=0.9, label=names[k])
            sig = e[e.p < 0.05]
            ax.scatter(sig.lag_months, sig.r, s=5, color=cols[k], zorder=3)
        ax.axhline(0, color=S.INK2, lw=0.5); ax.axvline(0, color=S.INK2, lw=0.5)
        ax.set_xlabel("Lag (months; positive = index leads)"); _title(ax, lab); S.hgrid(ax)
    axs[0].set_ylabel("Correlation"); axs[1].legend(loc="lower right", fontsize=5.8, ncol=2)
    S.panel(axs[0], "a"); S.panel(axs[1], "b")
    S.save(fig, "FigS_lagcorr")


def figS_transfer():
    sk = pd.read_csv(TP / "transfer_model_skill.csv")
    oof = pd.read_csv(TP / "transfer_model_oof.csv")
    if "obs" not in oof.columns:  # the public release ships predictions without observed monthly discharge
        print("Fig. S19 needs observed monthly discharge (GRDC): run prepare_grdc.py and projection_cmip6.py first")
        return
    oof.columns = ["grdc_no", "date"] + list(oof.columns[2:])
    oof["date"] = pd.to_datetime(oof.date)
    st = pd.read_csv(R / "data/grdc_stations_raw.csv").set_index("grdc_no").station_name.str.title().to_dict()
    show = list(sk[sk.selected].grdc_no) + [1531450]
    fig, axs = plt.subplots(len(show), 1, figsize=(S.DOUBLE, 30 * len(show) * S.MM), sharex=True)
    for ax, g in zip(axs, show):
        d = oof[oof.grdc_no == g].set_index("date").asfreq("MS")
        ax.plot(d.index, d.obs, color=S.INK, lw=0.7, label="observed")
        ax.plot(d.index, d.ens, color=S.VOLTA, lw=0.7, label="ensemble, out of fold")
        r = sk[sk.grdc_no == g].iloc[0]
        _title(ax, f"{st.get(g, g)}: NSE {r.nse_log_ensemble:.2f}, anomaly ρ {r.anom_rho_ensemble:.2f}" + ("" if r.selected else " (not projected)"))
        ax.set_ylabel("ln Q")
        S.hgrid(ax)
    axs[0].legend(loc="upper right", fontsize=5.8, ncol=2)
    for i, ax in enumerate(axs):
        S.panel(ax, "abcdefgh"[i])
    S.save(fig, "FigS_transfer")


def figS_projmodels():
    lg = pd.read_csv(TP / "projection_metrics_long.csv")
    sk = pd.read_csv(TP / "transfer_model_skill.csv")
    st = pd.read_csv(R / "data/grdc_stations_raw.csv").set_index("grdc_no").station_name.str.title().to_dict()
    rows = [("SPI-12", "basin", "SPI-12, basin"), ("SRI-12", "basin", "SRI-12, basin")] + \
           [("SSI-12 (ensemble)", str(g), f"SSI-12, {st.get(int(g), g)}") for g in sk[sk.selected].grdc_no]
    srcs = sorted(lg.source.unique())
    fig, axs = plt.subplots(1, 2, figsize=(S.DOUBLE, 80 * S.MM), sharey=True)
    for ax, sc in zip(axs, ["ssp245", "ssp585"]):
        M = np.full((len(srcs), len(rows)), np.nan)
        for j, (ind, u, _) in enumerate(rows):
            d = lg[(lg["index"] == ind) & (lg.unit.astype(str) == u) & (lg.scenario == sc)]
            b = d[d.period == "1985-2014"].set_index("source").p_below_1
            f = d[d.period == "2051-2080"].set_index("source").p_below_1
            for i, s_ in enumerate(srcs):
                if s_ in b.index and s_ in f.index:
                    M[i, j] = 100 * (f[s_] - b[s_])
        from matplotlib.colors import TwoSlopeNorm
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0, -30, 60))
        ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[2] for r in rows], rotation=35, ha="right", fontsize=5.8)
        ax.set_yticks(range(len(srcs))); ax.set_yticklabels(srcs, fontsize=5.8)
        _title(ax, "SSP2-4.5, 2051–2080" if sc == "ssp245" else "SSP5-8.5, 2051–2080")
    cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02); cb.set_label("Change in share of months below −1 (percentage points)")
    S.panel(axs[0], "a"); S.panel(axs[1], "b")
    S.save(fig, "FigS_projmodels")


if __name__ == "__main__":
    which = sys.argv[1:] or ["regime", "tele", "proj"]
    for w in which:
        {"regime": fig_regime, "tele": fig_tele, "proj": fig_proj, "wtcall": figS_wtc_all, "syncwtc": figS_sync_wtc,
         "lag": figS_lagcorr, "transfer": figS_transfer, "projmodels": figS_projmodels}[w]()
