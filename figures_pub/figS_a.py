#!/usr/bin/env python3
"""Supplementary Figs S1-S7 and S9-S12 (observed network), in the shared publication style."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import chi2, gaussian_kde
import pubstyle as S

S.setup()
R = Path(__file__).resolve().parents[1]
H = Path(__file__).resolve().parent
T = R / "output_observational/tables"
st = pd.read_csv(R / "data/grdc_stations_raw.csv").set_index("grdc_no")
name = lambda n: st.station_name[int(n)].title().replace("(Halcrow)", "").strip()
RB = json.load(open(T / "robust_numbers.json")); RV = json.load(open(T / "review_numbers.json"))
IDN = json.load(open(R / "output_identifiability/identifiability_numbers.json"))
ev = pd.read_csv(T / "SSI_drought_events.csv", parse_dates=["start_date", "end_date"])
DIV = LinearSegmentedColormap.from_list("div", ["#256abf", "#9ec5f4", "#f0efec", "#f2a99a", "#c73a3a"])
REF = dict(color=S.MUTED, lw=0.6)


def note(ax, s, x=0.98, y=0.97, ha="right", va="top", **kw):
    ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va, fontsize=5.6, color=S.INK2, linespacing=1.15, **kw)


# ------------------------------------------------------------------------------------------- S1
if not (H / "cache_s1.csv").exists() or not (H / "cache_s4_null.csv").exists():
    import sys
    sys.path.insert(0, str(R))
    from voltaprop import supplementary as SU
    from voltaprop.propagation import PropagationConfig, load_connectivity
    from voltaprop.analysis import permutation_null_test
    cn_ = load_connectivity(T / "upstream_connectivity.csv")
    SU.s1_event_level_m_sensitivity(ev, cn_)[0].to_csv(H / "cache_s1.csv", index=False)
    sm_, nl_ = permutation_null_test(ev, cn_, PropagationConfig(), n_permutations=1999, seed=42, node_col="node_id", progress=False)
    sm_.to_csv(H / "cache_s4_summary.csv", index=False); nl_.to_csv(H / "cache_s4_null.csv", index=False)
s1 = pd.read_csv(H / "cache_s1.csv")
fig, axs = plt.subplots(1, 3, figsize=(S.DOUBLE, 52 * S.MM), gridspec_kw=dict(wspace=0.38, left=0.06, right=0.99, bottom=0.2, top=0.9))
specs = [("connected_events_pct", "Connected origin events (%)", None, None),
         ("spearman_rho_vs_reference", "Spearman ρ of chain size vs $M$ = 5", 0.97, "Ebro: ρ > 0.97 for $M$ ≤ 3"),
         ("flip_vs_reference_pct", "Events changing class vs $M$ = 5 (%)", 2.2, "Ebro: < 2.2% for $M$ ≤ 5")]
for k, (ax, (c, lab, ref, rl)) in enumerate(zip(axs, specs)):
    ax.plot(s1.M_days, s1[c], color=S.NAVY, lw=0.9, marker="o", ms=3.2, mfc="white", mec=S.NAVY, mew=0.8)
    ax.plot(5, s1.loc[s1.M_days == 5, c], marker="o", ms=3.6, color=S.VOLTA, zorder=4, ls="")
    if ref is not None:
        ax.axhline(ref, color=S.EBRO, lw=0.6)
        ax.text(15, ref, rl, ha="right", va="bottom", fontsize=5.6, color=S.EBRO)
    ax.set_xlabel("Minimum overlap $M$ (d)"); ax.set_ylabel(lab); ax.set_xticks(s1.M_days)
    S.hgrid(ax); S.panel(ax, "abc"[k], x=-0.2, y=1.0)
S.save(fig, "FigS01_M_sensitivity")

# ------------------------------------------------------------------------------------------- S2
v = pd.DataFrame(RB["index_variants"])
v["label"] = ["Default", "GEV only", "GLO only", "GPA only", "LNO only", "PE3 only", "WEI only", "Pool ±30 d", "Ref. 1975–92"]
grp = ["default"] + ["family"] * 6 + ["construction"] * 2
gc = {"default": S.VOLTA, "family": S.NAVY, "construction": "#6250d6"}
fig, axs = plt.subplots(1, 3, figsize=(S.DOUBLE, 56 * S.MM), sharey=True,
                        gridspec_kw=dict(wspace=0.12, left=0.1, right=0.99, bottom=0.19, top=0.9))
y = np.arange(len(v))[::-1]
for k, (ax, (c, lab)) in enumerate(zip(axs, [("default_events_reproduced", "Share of default events reproduced"),
                                             ("chains", "Chains"), ("cf_opportunities", "Connected share of opportunities")])):
    ax.hlines(y, 0, v[c], color="#d3d8de", lw=1.0)
    ax.scatter(v[c], y, s=16, c=[gc[g] for g in grp], zorder=3, lw=0)
    for yi, xv in zip(y, v[c]):
        ax.text(xv, yi + 0.28, f"{xv:.2f}" if c != "chains" else f"{int(xv)}", ha="center", va="bottom", fontsize=5.2,
                color=S.INK2)
    ax.set_xlabel(lab); ax.set_xlim(0, v[c].max() * 1.15 if c != "cf_opportunities" else 0.66)
    S.hgrid(ax, "x"); S.panel(ax, "abc"[k], x=0.02 if k else -0.3, y=1.0)
axs[2].axvline(0.566, color=S.EBRO, lw=0.7)
axs[2].text(0.56, 8.6, "Ebro 0.566", ha="right", va="center", fontsize=5.6, color=S.EBRO)
axs[0].set_yticks(y); axs[0].set_yticklabels(v.label); axs[0].tick_params(axis="y", length=0)
for ax in axs:
    ax.spines["left"].set_visible(False)
axs[0].legend(handles=[Line2D([], [], marker="o", ls="", color=gc[g], ms=3.5, label=l) for g, l in
                       (("default", "default"), ("family", "single distribution family"), ("construction", "pooling or reference period"))],
              loc="lower left", bbox_to_anchor=(-0.45, 1.02), ncols=3)
S.save(fig, "FigS02_index_sensitivity")

# ------------------------------------------------------------------------------------------- S3
p3 = pd.read_csv(T / "review_s3_pairs.csv")
fig, axs = plt.subplots(1, 3, figsize=(S.DOUBLE, 56 * S.MM), sharey=True,
                        gridspec_kw=dict(wspace=0.1, left=0.06, right=0.99, bottom=0.2, top=0.92))
for k, (ax, (x, xl, key)) in enumerate(zip(axs, [("dist_km", "Straight-line distance (km)", "lag_vs_distance"),
                                                ("dA_km2", "Intervening drainage area (10³ km²)", "lag_vs_intervening_area"),
                                                ("upstream_duration", "Upstream event duration (d)", "lag_vs_upstream_duration")])):
    xv = p3[x] / (1000 if x == "dA_km2" else 1)
    ax.scatter(xv, p3.lag_days, s=18, color=S.EBRO, ec="white", lw=0.5, zorder=3)
    r, p = RV["s3"][key]
    note(ax, f"Spearman ρ = {r:.2f}, p = {p:.3f}".replace("-", "−"), x=0.03, y=0.04, ha="left", va="bottom")
    ax.set_xlabel(xl); S.hgrid(ax); S.panel(ax, "abc"[k], x=0.0 if k else -0.14, y=1.0)
d = np.linspace(5, p3.upstream_duration.max() + 5, 50)
axs[2].plot(d, -(d - 5), color=S.INK2, lw=0.6)
axs[2].annotate("bound −($D_u$ − $M$)", (58, -53), xytext=(44, -66), textcoords="data", ha="right", va="center",
                fontsize=6, color=S.INK2, arrowprops=dict(arrowstyle="-", lw=0.5, color=S.INK2, shrinkB=1))
axs[0].set_ylabel("Lag (d)")
S.save(fig, "FigS03_lag_relations")

# ------------------------------------------------------------------------------------------- S4
sm = pd.read_csv(H / "cache_s4_summary.csv"); nl = pd.read_csv(H / "cache_s4_null.csv")
labs = {"connected_fraction": "Connected fraction", "mean_chain_size": "Mean chain size",
        "mean_propagation_fraction": "Mean propagation fraction", "chain_count": "Chains (count)",
        "total_valid_pairs": "Valid pairs (count)"}
fig, axs = plt.subplots(1, 5, figsize=(S.DOUBLE, 46 * S.MM), gridspec_kw=dict(wspace=0.32, left=0.05, right=0.99, bottom=0.24, top=0.86))
for k, (ax, (_, r)) in enumerate(zip(axs, sm.iterrows())):
    col = nl[r.statistic].dropna()
    disc = r.statistic in ("chain_count", "total_valid_pairs")
    bins = np.arange(col.min() - 0.5, col.max() + 1.5) if disc else 30
    ax.hist(col, bins=bins, color="#c9d3de", edgecolor="white", lw=0.3)
    ax.axvline(r.observed, color=S.VOLTA, lw=1.1)
    ax.axvline(r.null_mean, color=S.INK2, lw=0.6)
    ax.set_title(f"p = {r.p_value_one_sided:.3f}", fontsize=6.2, loc="left")
    note(ax, f"observed {r.observed:.3g}\nnull {r.null_mean:.3g} ± {r.null_sd:.2g}")
    ax.set_xlabel(labs[r.statistic]); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    S.panel(ax, "abcde"[k], x=-0.04, y=1.07)
fig.legend(handles=[Patch(fc="#c9d3de", label=f"null, {int(sm.n_permutations[0]):,} permutations (published design)"),
                    Line2D([], [], color=S.VOLTA, lw=1.1, label="observed Volta"),
                    Line2D([], [], color=S.INK2, lw=0.6, label="null mean")],
           loc="lower center", ncols=3, bbox_to_anchor=(0.5, 0.0))
S.save(fig, "FigS04_null_test")

# ------------------------------------------------------------------------------------------- S5
pw = pd.DataFrame(RB["power_simulation"])
fig, ax = plt.subplots(figsize=(S.SINGLE, 62 * S.MM))
fig.subplots_adjust(left=0.15, right=0.97, bottom=0.17, top=0.96)
lo, hi = [], []
from statsmodels.stats.proportion import proportion_confint
for _, r in pw.iterrows():
    a, b = proportion_confint(round(r.power * r.n_sim), r.n_sim, method="wilson"); lo.append(a); hi.append(b)
ax.fill_between(pw.mean_cf_opportunities, lo, hi, color="#cde2fb", lw=0)
ax.plot(pw.mean_cf_opportunities, pw.power, color=S.NAVY, lw=0.9, marker="o", ms=3.2, mfc="white", mew=0.8)
ax.axvline(0.25, color=S.VOLTA, lw=0.7); ax.text(0.255, 0.98, "observed\nVolta 0.25", fontsize=5.6, va="top")
ax.axvline(0.566, color=S.EBRO, lw=0.7); ax.text(0.561, 0.05, "Ebro 0.566", fontsize=5.6, ha="right", color=S.EBRO)
ax.axhline(0.8, **REF)
v0 = IDN["networks"][1]["null_cf_opp"]; ex = IDN["analytic_bound"][0]["excess"]
note(ax, f"Volta availability-aware null mean {v0:.2f};\nEbro excess over its own null {ex:.3f}", x=0.97, y=0.42)
ax.set_xlabel("Connected share of opportunities after injection"); ax.set_ylabel("Power, α = 0.05")
ax.set_ylim(-0.02, 1.02); S.hgrid(ax)
S.save(fig, "FigS05_power_injection")

# ------------------------------------------------------------------------------------------- S6
d5 = pd.read_csv(T / "tableS5_detection_sensitivity.csv")
P6 = [("n_events", "Event count", S.BLUES, None), ("mean_duration", "Mean duration (d)", S.BLUES, None),
      ("pct_change_vs_default", "Change in event count (%)", DIV, "div"),
      ("rho_count_vs_default", "ρ, per-station event count", S.BLUES, None),
      ("rho_severity_vs_default", "ρ, per-station mean severity", S.BLUES, None)]
fig, axs = plt.subplots(1, 5, figsize=(S.DOUBLE, 50 * S.MM), gridspec_kw=dict(wspace=0.5, left=0.05, right=0.98, bottom=0.2, top=0.86))
for k, (ax, (c, lab, cm, kind)) in enumerate(zip(axs, P6)):
    pv = d5.pivot(index="pool_gap", columns="min_duration", values=c)
    norm = TwoSlopeNorm(0, pv.values.min(), pv.values.max()) if kind else Normalize(pv.values.min(), pv.values.max())
    im = ax.imshow(pv.values, cmap=cm, norm=norm, origin="lower", aspect="auto")
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            val = pv.values[i, j]
            dark = norm(val) > 0.6 if not kind else abs(norm(val) - 0.5) > 0.3
            ax.text(j, i, (f"{val:.0f}" if abs(val) >= 10 else f"{val:.2f}").replace("-", "−"), ha="center", va="center", fontsize=4.9,
                    color="white" if dark else S.INK)
    di, dj = list(pv.index).index(10), list(pv.columns).index(5)
    ax.add_patch(plt.Rectangle((dj - 0.5, di - 0.5), 1, 1, fc="none", ec=S.VOLTA, lw=1.0))
    ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels(pv.columns)
    ax.set_yticks(range(pv.shape[0])); ax.set_yticklabels(pv.index)
    ax.set_xlabel("Minimum duration (d)")
    if k == 0:
        ax.set_ylabel("Pooling gap (d)")
    ax.set_title(lab, fontsize=6.2, loc="left")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    S.panel(ax, "abcde"[k], x=-0.08, y=1.09)
note(axs[-1], "", x=1, y=1)
fig.text(0.98, 0.02, "outlined cell: published configuration (gap 10 d, minimum duration 5 d)", ha="right", fontsize=5.6,
         color=S.INK2)
S.save(fig, "FigS06_detection_sensitivity")

# ------------------------------------------------------------------------------------------- S7
z = pd.read_csv(T / "robust_sahel_jjaso_z.csv", index_col=0).iloc[:, 0]
s8 = RB["s8_analogue"]
fig, (a, b) = plt.subplots(1, 2, figsize=(S.DOUBLE, 56 * S.MM), gridspec_kw=dict(width_ratios=[2.6, 1], wspace=0.28,
                                                                                  left=0.06, right=0.99, bottom=0.14, top=0.92))
dry = z.values <= -0.5
a.bar(z.index, z.values, color=np.where(dry, S.VOLTA, "#c4c9cf"), width=0.78, lw=0)
a.axhline(-0.5, color=S.INK2, lw=0.6); a.axhline(0, color=S.INK2, lw=0.4)
for yv, lbl in [(1976, "Case 1"), (1997, "Case 2"), (1983, "Case 3"), (1990, "Fig. S8")]:
    a.annotate(lbl, (yv, z.loc[yv]), xytext=(0, -2 if z.loc[yv] < 0 else 2), textcoords="offset points", ha="center",
               va="top" if z.loc[yv] < 0 else "bottom", fontsize=6, color=S.INK)
a.set_ylabel("Sahel JJASO rainfall anomaly ($z$)"); a.set_xlim(1960, 2008); a.set_ylim(-2.55, 2.15); S.hgrid(a)
a.legend(handles=[Patch(fc=S.VOLTA, label=f"dry season ($z$ ≤ −0.5), {int(dry.sum())} years"), Patch(fc="#c4c9cf", label="other")],
         loc="upper right", ncols=2)
S.panel(a, "a", x=-0.07, y=1.0)
sy = 145.3
nd, no = round(s8["event_share_in_dry_years"] * 174), 174 - round(s8["event_share_in_dry_years"] * 174)
ed, eo = sy * s8["exposure_share_in_dry_years"], sy * (1 - s8["exposure_share_in_dry_years"])
for i, (n_, e_, c) in enumerate(((nd, ed, S.VOLTA), (no, eo, "#c4c9cf"))):
    l_, u_ = chi2.ppf(0.025, 2 * n_) / 2 / e_, chi2.ppf(0.975, 2 * (n_ + 1)) / 2 / e_
    b.bar(i, n_ / e_, width=0.55, color=c, lw=0)
    b.vlines(i, l_, u_, color=S.INK, lw=0.7)
    b.text(i, u_ + 0.04, f"{n_ / e_:.2f}", ha="center", va="bottom", fontsize=5.8)
b.set_xticks([0, 1]); b.set_xticklabels(["dry seasons", "other"])
b.set_ylabel("Events per station-year"); b.set_ylim(0, 2.3)
b.set_title(f"rate ratio {s8['rate_ratio']:.2f}, one-sided p = {s8['p_rate_ratio_one_sided']:.3f}", fontsize=6, loc="left")
S.hgrid(b); S.panel(b, "b", x=-0.3, y=1.0)
S.save(fig, "FigS07_sahel_context")

# ------------------------------------------------------------------------------------------- S9
op = pd.read_csv(T / "review_opportunities.csv")
ch = pd.read_csv(T / "Chains_final.csv")
ch["eid"] = ch.chain_id.str.split("_").str[1].astype(int)
conn_set = set(zip(ch.origin_node, ch.eid))
oe = op.merge(ev, on=["node_id", "event_id"])
oe["connected"] = [(n, e) in conn_set for n, e in zip(oe.node_id, oe.event_id)]
uc = pd.read_csv(T / "review_unconnected_by_station.csv")
up = pd.read_csv(T / "upstream_contrast.csv")
fig, axs = plt.subplots(1, 4, figsize=(S.DOUBLE, 56 * S.MM), gridspec_kw=dict(width_ratios=[1.25, 1, 1, 1.25], wspace=0.5,
                                                                                left=0.09, right=0.99, bottom=0.17, top=0.9))
ax = axs[0]
u2 = uc.sort_values("unconnected_pct")
yy = np.arange(len(u2))
ax.hlines(yy, 0, u2.unconnected_pct, color="#d3d8de", lw=1.0)
ax.scatter(u2.unconnected_pct, yy, s=16, color=S.NAVY, zorder=3, lw=0)
for yi, (_, r) in zip(yy, u2.iterrows()):
    ax.text(r.unconnected_pct + 3, yi, f"{int(r['size'] - r['sum'])}/{int(r['size'])}", va="center", fontsize=5.4, color=S.INK2)
ax.set_yticks(yy); ax.set_yticklabels([name(n) for n in u2.node_id]); ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_xlim(0, 110); ax.set_xlabel("Origin events without a match (%)")
S.hgrid(ax, "x"); S.panel(ax, "a", x=-0.42, y=1.0)
rng = np.random.default_rng(3)
for k, (ax, c, lab) in enumerate(((axs[1], "duration", "Duration (d)"), (axs[2], "severity", "Severity"))):
    for i, flag in enumerate((False, True)):
        vals = oe.loc[oe.connected == flag, c].values
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(vals)), vals, s=5 if not flag else 11, lw=0,
                   color=S.NAVY if flag else "#9aa5b1", alpha=0.75 if not flag else 1, zorder=3)
        q1, q2, q3 = np.percentile(vals, [25, 50, 75])
        ax.add_patch(plt.Rectangle((i + 0.24, q1), 0.12, q3 - q1, fc="white", ec=S.INK, lw=0.6, zorder=4))
        ax.plot([i + 0.24, i + 0.36], [q2, q2], color=S.INK, lw=1.0, zorder=5)
    ax.set_yscale("log"); ax.set_xticks([0, 1])
    ax.set_xticklabels([f"isolated\n(n = {int((~oe.connected).sum())})", f"connected\n(n = {int(oe.connected.sum())})"])
    ax.set_ylabel(lab); ax.set_xlim(-0.45, 1.55); S.hgrid(ax); S.panel(ax, "bc"[k], x=-0.32, y=1.0)
axs[3].axis("off")
sub = axs[3].get_subplotspec().subgridspec(2, 1, hspace=0.9)
for i, (c, lab, xmax) in enumerate((("median_duration", "Median duration (d)", 40), ("median_severity", "Median severity", 10))):
    ax = fig.add_subplot(sub[i])
    for j, (_, r) in enumerate(up.sort_values("accepted").iterrows()):
        col = S.EBRO if r.accepted else "#9aa5b1"
        ax.hlines(j, 0, r[c], color=col, lw=2.2)
        ax.text(r[c] + xmax * 0.03, j, f"{r[c]:.3g}", va="center", fontsize=5.4, color=S.INK2)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["never accepted", "accepted"]); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False); ax.set_xlim(0, xmax); ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel(lab, labelpad=1)
    S.hgrid(ax, "x")
    if i == 0:
        ax.set_title("upstream candidates (n = 10 each)", fontsize=5.8, loc="left")
        S.panel(ax, "d", x=-0.62, y=1.15)
S.save(fig, "FigS09_non_propagating")

# ------------------------------------------------------------------------------------------- S10
h = pd.read_csv(T / "tableS9_homogeneity.csv").dropna(subset=["ks_statistic"])
h["years"] = h.n_valid / 365.25
fig, ax = plt.subplots(figsize=(S.SINGLE, 64 * S.MM))
fig.subplots_adjust(left=0.15, right=0.97, bottom=0.16, top=0.97)
ax.axhline(0.15, color=S.INK2, lw=0.6); ax.text(29.5, 0.155, "flag threshold", ha="right", va="bottom", fontsize=5.6)
for _, r in h.iterrows():
    ax.scatter(r.years, r.ks_statistic, s=18, color=S.VOLTA if r.flag else S.EBRO, ec="white", lw=0.5, zorder=3)
    off = {"Lama Kara": (-1, 6, "left"), "Vonkoro": (3, -4, "left"), "Prang": (-3, 2, "right"), "Bamboi": (3, -3, "left"),
           "Ekumdipe": (3, -7, "left"), "Tiele": (3, 2, "left"), "Yagaba": (3, 2, "left"), "Nangodi": (3, -3, "left")}
    dx, dy, ha = off.get(name(r.node_id), (3, 2, "left"))
    ax.annotate(name(r.node_id), (r.years, r.ks_statistic), xytext=(dx, dy), textcoords="offset points", fontsize=5.0, ha=ha,
                color=S.INK2, path_effects=[pe.withStroke(linewidth=1.4, foreground="white")])
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=S.VOLTA, ms=3.5, label="flagged"),
                   Line2D([], [], marker="o", ls="", color=S.EBRO, ms=3.5, label="not flagged")], loc="upper right")
ax.set_xlabel("Years of defined index"); ax.set_ylabel("KS statistic, first vs second half"); S.hgrid(ax)
S.save(fig, "FigS10_homogeneity")

# ------------------------------------------------------------------------------------------- S11 (article Fig. S17)
from figS_redesign import s17  # noqa: E402
s17()

# ------------------------------------------------------------------------------------------- S12
ssi = pd.read_csv(T / "SSI_daily.csv", parse_dates=["date"])
nodes = sorted(ssi.node_id.unique())
basin_of = {"BLACK VOLTA": 0, "WHITE VOLTA": 1, "RED VOLTA": 1, "SISSILI": 1, "KULPAWN": 1, "OTI": 2, "PENDJARI": 2,
            "KARA": 2, "MAGOU": 2, "SINGOU": 2, "DAKA": 3, "PRU": 3}
order = sorted(nodes, key=lambda n: (basin_of.get(st.river[n], 4), -st.area_km2[n]))
yrs = np.arange(1961, 2008)
obs = ssi[ssi.fill_tier.isin([0, 1])].assign(y=lambda d: d.date.dt.year)
cn = obs.groupby(["node_id", "y"]).size()
mat = np.array([[cn.get((n, yv), 0) / (366 if yv % 4 == 0 else 365) for yv in yrs] for n in order])
fig, ax = plt.subplots(figsize=(S.DOUBLE, 82 * S.MM))
fig.subplots_adjust(left=0.17, right=0.93, bottom=0.11, top=0.96)
cmap = LinearSegmentedColormap.from_list("av", ["#ffffff"] + S.BLUE_RAMP[1:])
im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
               extent=(yrs[0] - 0.5, yrs[-1] + 0.5, len(order) - 0.5, -0.5))
chs = pd.read_csv(T / "Chains_final.csv", parse_dates=["chain_start"])
for _, c in chs.iterrows():
    ax.plot(c.chain_start.year + c.chain_start.dayofyear / 365 - 0.5, order.index(int(c.origin_node)), marker="o", ms=3.6,
            mfc=S.VOLTA, mec=S.INK, mew=0.5, ls="")
ax.set_yticks(range(len(order))); ax.set_yticklabels([f"{name(n)} ({st.river[n].title()})" for n in order], fontsize=5.6)
for b in np.where(np.diff([basin_of.get(st.river[n], 4) for n in order]))[0]:
    ax.axhline(b + 0.5, color=S.INK2, lw=0.5)
ax.tick_params(axis="y", length=0)
ax.set_xlabel("Year")
cax = fig.add_axes([0.945, 0.3, 0.012, 0.5])
cb = fig.colorbar(im, cax=cax); cb.set_label("Share of days with data", fontsize=6); cb.outline.set_linewidth(0.4)
ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc=S.VOLTA, mec=S.INK, mew=0.5, ms=3.6,
                          label="onset of a reconstructed chain (origin station)")], loc="lower left",
          bbox_to_anchor=(0.0, 1.0), ncols=1)
S.save(fig, "FigS12_availability")
print("S1-S12 done")
