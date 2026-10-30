#!/usr/bin/env python3
"""
Article Fig. 5 (the ten matched pairs) and Fig. S18 (drawn by figS_redesign.s18); the functions
s20 and s22-s25 draw exploratory panels that the article does not report.

Fig. 5   Network, timing geometry, deficit alignment and severity balance of the ten matched pairs.
Fig. S19 Landscape and storage associations against the detectability limit, and the recession-timescale scatter.
Fig. S20 SIEREM water-holding capacity: location on the network and relation to drought duration (descriptive).
Fig. S22 Intervening-catchment models: skill against the training-rate baseline, discrimination, gauge base rates.
Fig. S23 Antecedent streamflow state of the 37 testable events and tercile per-link estimates.
Fig. S24 Month-level models: discrimination, precision and accuracy under the two holdouts.
Fig. S25 Transfer of the Brier-loss differences from withheld gauges to later years.

Every number is read from tables written by the analysis pipeline (output_*/, supp_tables/) and re-derived where
possible (assertions below). Only the schematic gauge positions in Fig. 5a are typed in. Sub-basin colours are
Okabe-Ito blue / bluish green / reddish purple (validated: normal-vision dE >= 18.7; green-purple CVD dE 7.6, so
marker shape and direct labels also carry identity). Usage: python fig05_landscape_panels.py [fig5 s19 s20 s22 s23 s24 s25]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle, Patch
from matplotlib.path import Path as MPath
from scipy.optimize import minimize_scalar, brentq
from scipy.stats import norm
import pubstyle as S

S.setup()
R = Path(__file__).resolve().parents[1]
T = R / "output_observational/tables"
ST = Path(__file__).resolve().parent / "supp_tables"
st = pd.read_csv(R / "data/grdc_station_list.csv").set_index("grdc_no")
NAME = {n: st.station_name[n].title().replace("(Halcrow)", "").strip() for n in st.index}
halo = [pe.withStroke(linewidth=1.8, foreground="white")]
M = "−"                                    # typographic minus for hand-written labels

SB = {"Black Volta": "#0072B2", "White Volta": "#009E73", "Oti–Pendjari": "#CC79A7"}
MK = {"Black Volta": "o", "White Volta": "s", "Oti–Pendjari": "^"}
BASIN = {1531050: "Black Volta", 1931400: "Black Volta", 1431300: "Black Volta", 1531100: "Black Volta",
         1931785: "White Volta", 1531600: "White Volta", 1531650: "White Volta", 1531550: "White Volta",
         1531420: "White Volta", 1531430: "White Volta", 1531450: "White Volta",
         1931880: "Oti–Pendjari", 1731450: "Oti–Pendjari", 1731400: "Oti–Pendjari", 1631500: "Oti–Pendjari",
         1531800: "Oti–Pendjari"}
GREY, GREY2, NAVY = "#c3c7cd", "#7f8790", S.NAVY
TC = {"Gauge holdout": "#0072B2", "Later years": "#D55E00"}     # test colours (Okabe-Ito blue / vermilion)
TLAB = {"Gauge holdout": "withheld gauges", "Later years": "1995–2007"}


def mfmt(v, fmt):
    return format(v, fmt).replace("-", M)


def haversine(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (st.lat[a], st.lon[a], st.lat[b], st.lon[b]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(h))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def letter(fig, x, y, s, title=None):
    fig.text(x, y, s, fontsize=8, fontweight="bold", ha="left", va="baseline", color=S.INK)
    if title:
        fig.text(x + 0.018, y, title, fontsize=7.5, ha="left", va="baseline", color=S.INK)


# ======================================================================================================= Fig. 5
def load_pairs():
    pr = pd.read_csv(T / "propagation_pairs.csv",
                     parse_dates=["origin_start", "origin_end", "upstream_start", "upstream_end"])
    ch = pd.read_csv(T / "Chains_final.csv", parse_dates=["origin_start"])
    pr["lead"] = -pr.lag_days
    pr["end_off"] = (pr.upstream_end - pr.origin_end).dt.days
    # along-channel distance on TDX-Hydro (reference_equivalence_simulation.py, part D)
    cd = pd.read_csv(T / "r5_pairs_channel_distance.csv", parse_dates=["origin_start"])
    key = {(o, u, d.normalize()): km for o, u, d, km in zip(cd.origin, cd.upstream, cd.origin_start, cd.channel_km)}
    pr["dist"] = [key[(NAME[o], NAME[u], d.normalize())] for o, u, d in zip(pr.origin_node, pr.upstream_node, pr.origin_start)]
    pr["basin"] = [BASIN[o] for o in pr.origin_node]
    pr["pair"] = [f"{NAME[o]} ← {NAME[u]}" for o, u in zip(pr.origin_node, pr.upstream_node)]
    pr["label"] = [f"{p}, {d:%b %Y}" for p, d in zip(pr.pair, pr.origin_start)]
    pr["cls"] = np.where(pr.lead <= 1, "near", np.where(pr.end_off < 0, "relay", "nested"))
    pr = pr.merge(ch[["origin_node", "origin_start", "chain_severity", "chain_severity_hm3"]],
                  on=["origin_node", "origin_start"], how="left")
    assert len(pr) == 10 and np.allclose(pr.chain_severity, pr.origin_severity + pr.upstream_severity, atol=1e-3)
    return pr


def bez(p0, p1, rad, n=160, tl=0.5):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    c = (p0 + p1) / 2 + rad * np.array([d[1], -d[0]])
    t = np.linspace(0, 1, n)[:, None]
    lab = (1 - tl) ** 2 * p0 + 2 * (1 - tl) * tl * c + tl ** 2 * p1
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p1, lab


def fig5():
    pr = load_pairs()
    opp = pd.read_csv(T / "review_opportunities.csv", dtype={"observing": str})
    con = pd.read_csv(T / "upstream_connectivity.csv", sep=";")
    rec = []
    for _, r in opp.iterrows():
        if isinstance(r.observing, str):
            rec += [(int(r.node_id), int(float(u))) for u in r.observing.split(",")]
    rec = pd.Series(1, index=pd.MultiIndex.from_tuples(rec, names=["o", "u"])).groupby(level=[0, 1]).sum()
    match = pr.groupby(["origin_node", "upstream_node"]).size()
    links = [(int(u), int(r.node_id)) for _, r in con.iterrows() if isinstance(r.upstream_chain, str)
             for u in r.upstream_chain.split(",")]
    n_or = int((opp.groupby("node_id").n_upstream.first() > 0).sum())
    assert len(links) == 22 and len(rec) == 12 and rec.sum() == 67 and len(match) == 8 and match.sum() == 10
    assert len(opp[opp.n_upstream > 0]) == 104 and (opp.n_observing > 0).sum() == 40

    H_MM, TOP_MM = 192.0, 158.0                 # figure height; the top block keeps its 158-mm layout
    fig = plt.figure(figsize=(S.DOUBLE, H_MM * S.MM))
    ty = lambda y: (y * TOP_MM + H_MM - TOP_MM) / H_MM      # top-block position, from the 158-mm layout
    th = lambda h: h * TOP_MM / H_MM                        # top-block height
    fy = lambda mm: mm / H_MM                               # bottom-block position in mm from the bottom
    # ---------------------------------------------------------------- (a) schematic network
    ax = fig.add_axes([0.0, ty(0.535), 0.6, th(0.405)])
    XL, YL = (-1.05, 8.75), (0.35, 4.25)
    ax.set_xlim(*XL); ax.set_ylim(*YL); ax.axis("off")
    W_in, H_in = 0.6 * S.DOUBLE, 0.405 * TOP_MM * S.MM
    sx, sy = W_in / (XL[1] - XL[0]), H_in / (YL[1] - YL[0])
    POS = {1531050: (0.0, 3.65), 1931400: (0.0, 3.05), 1431300: (0.0, 2.05), 1531100: (0.0, 0.75),
           1931785: (3.55, 3.9), 1531600: (3.55, 3.25), 1531650: (4.55, 3.3), 1531550: (3.55, 2.3),
           1531420: (2.45, 1.5), 1531430: (1.75, 1.1), 1531450: (3.55, 0.75),
           1931880: (6.55, 3.3), 1731450: (7.65, 3.3), 1731400: (7.1, 2.35), 1631500: (7.95, 1.5),
           1531800: (7.1, 0.75)}
    RIVER = [[(0.0, 4.05), (0.0, 0.45)], [(3.55, 4.2), (3.55, 0.45)], [(4.55, 3.3), (3.55, 2.65)],
             [(2.45, 1.5), (3.05, 1.12), (3.55, 0.98)], [(1.75, 1.1), (3.05, 1.12)],
             [(7.1, 3.75), (7.1, 0.45)], [(6.55, 3.3), (7.1, 2.85)], [(7.65, 3.3), (7.1, 2.7)],
             [(7.95, 1.5), (7.1, 1.05)]]
    for seg in RIVER:
        xs, ys = zip(*seg)
        ax.plot(xs, ys, color="#d9e6f2", lw=3.4, solid_capstyle="round", zorder=0)
    # (rad, label dx, label dy, ha) for observed links; rad only for the others
    SPEC = {(1531050, 1431300): (-0.42, 0.07, 0.0, "left"), (1531050, 1531100): (-0.36, 0.07, 0.0, "left"),
            (1531050, 1931400): -0.45, (1931400, 1431300): -0.35, (1931400, 1531100): -0.22,
            (1431300, 1531100): -0.3,
            (1531600, 1531550): (-0.45, 0.06, 0.02, "left"), (1531650, 1531550): (0.28, -0.02, 0.1, "center"),
            (1531550, 1531450): (0.32, -0.07, 0.0, "right"), (1531600, 1531450): (0.32, -0.07, 0.0, "right"),
            (1931785, 1531450): (0.34, -0.06, 0.0, "right"), (1531650, 1531450): (-0.24, 0.07, 0.0, "left"),
            (1531420, 1531450): (-0.12, 0.0, -0.11, "center"), (1531430, 1531450): (0.12, 0.0, -0.12, "center"),
            (1931785, 1531600): -0.45, (1931785, 1531550): -0.32,
            (1931880, 1731400): (0.3, -0.07, 0.0, "right"), (1731450, 1731400): (-0.3, 0.07, 0.0, "left"),
            (1931880, 1531800): 0.3, (1731450, 1531800): -0.28, (1731400, 1531800): -0.45, (1631500, 1531800): 0.1}
    TL = {(1931785, 1531450): 0.27}
    for (u, o) in links:
        sp = SPEC[(u, o)]
        rad = sp[0] if isinstance(sp, tuple) else sp
        pts, mid = bez(POS[u], POS[o], rad, tl=TL.get((u, o), 0.5))
        dd0 = np.hypot((pts[:, 0] - pts[0, 0]) * sx, (pts[:, 1] - pts[0, 1]) * sy)
        dd1 = np.hypot((pts[:, 0] - pts[-1, 0]) * sx, (pts[:, 1] - pts[-1, 1]) * sy)
        pts = pts[(dd0 > 0.055) & (dd1 > 0.065)]
        n_rec, n_m = rec.get((o, u), 0), match.get((o, u), 0)
        if n_rec == 0:
            kw = dict(color=GREY, lw=0.6)
        elif n_m == 0:
            kw = dict(color=GREY2, lw=0.55 + 0.1 * n_rec)
        else:
            kw = dict(color=SB[BASIN[o]], lw=0.55 + 0.1 * n_rec)
        ax.add_patch(FancyArrowPatch(path=MPath(pts), arrowstyle="-|>", mutation_scale=4.5,
                                     zorder=3 if n_m else 2, capstyle="round", **kw))
        if n_rec:
            _, dx, dy, ha = sp
            ax.text(mid[0] + dx, mid[1] + dy, f"{n_m}/{n_rec}", fontsize=6.2, ha=ha, va="center", zorder=6,
                    color=SB[BASIN[o]] if n_m else GREY2, fontweight="bold" if n_m else "normal",
                    path_effects=halo)
    has_up = set(con.node_id[con.n_upstream > 0].astype(int))
    LBL = {1531050: (-6, 0, "right"), 1931400: (-6, 0, "right"), 1431300: (-6, 0, "right"),
           1531100: (-6, 0, "right"), 1931785: (-6, 0, "right"), 1531600: (-6, 2, "right"),
           1531650: (6, 0, "left"), 1531550: (-6, 3, "right"), 1531420: (-6, 2, "right"),
           1531430: (-6, 0, "right"), 1531450: (0, -8, "center"), 1931880: (-6, 0, "right"),
           1731450: (6, 0, "left"), 1731400: (-6, 4, "right"), 1631500: (6, 0, "left"), 1531800: (0, -8, "center")}
    for n, (x, y) in POS.items():
        b = BASIN[n]
        ax.scatter(x, y, s=30, marker=MK[b], fc=SB[b] if n in has_up else "white", ec=SB[b], lw=1.0, zorder=5)
        dx, dy, ha = LBL[n]
        ax.annotate(NAME[n], (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.3, ha=ha,
                    va="center", color=S.INK, zorder=7, path_effects=halo)
    for b, x in (("Black Volta", 0.0), ("White Volta", 3.55), ("Oti–Pendjari", 7.1)):
        ax.text(x, 4.22, b, ha="center", va="bottom", fontsize=6.8, color=SB[b], fontweight="bold")
    letter(fig, 0.008, ty(0.965), "a", "Gauge links")
    leg = [Line2D([], [], color=GREY, lw=0.8, label="never observed together"),
           Line2D([], [], color=GREY2, lw=1.1, label="observed, no match"),
           Line2D([], [], color=S.INK2, lw=1.6, label="matched")]
    fig.legend(handles=leg, loc="upper left", bbox_to_anchor=(0.03, ty(0.532)), ncols=3, fontsize=6.2,
               columnspacing=1.6, handlelength=2.2, borderaxespad=0)
    # attrition funnel: links and origin events through the three conditions
    fx = fig.add_axes([0.105, ty(0.452), 0.47, th(0.048)])
    FUN = [("links", (22, 12, 8)), ("origin events", (104, 40, 10))]
    SH = ("#dfe2e6", "#a7aeb6", S.INK2)
    for j, (lab, vals) in enumerate(FUN):
        yj = 1 - j
        for v, c in zip(vals, SH):
            fx.barh(yj, v / vals[0], height=0.62, color=c, lw=0, zorder=2)
            fx.text(v / vals[0] - 0.008, yj, f"{v}", ha="right", va="center", fontsize=6,
                    color="white" if c != SH[0] else S.INK, zorder=3)
        fx.text(-0.015, yj, lab, ha="right", va="center", fontsize=6.2, color=S.INK)
    fx.set_xlim(0, 1.0); fx.set_ylim(-0.45, 1.45); fx.axis("off")

    # ---------------------------------------------------------------- (b) onset-recovery plane
    bx = fig.add_axes([0.685, ty(0.585), 0.3, th(0.355)])
    gn = __import__("json").load(open(T / "geometry_null.json"))
    sh = gn["null_pooled_share"]
    o_ = gn["observed"]
    CLS_C = {"near": "#b9bfc7", "relay": "#86c9a6", "nested": "#e5c98c"}
    bx.axvspan(-2.5, 1.5, color="#eceef1", zorder=0, lw=0)
    bx.fill_between([1.5, 52], -45, 0, color="#eef6f1", zorder=0, lw=0)
    bx.fill_between([1.5, 52], 0, 30, color="#fbf5e8", zorder=0, lw=0)
    # numbering keyed to panel c
    order = pd.concat([pr[pr.cls == c].sort_values("lead", ascending=False, kind="stable")
                       for c in ("relay", "nested", "near")])
    num = {lab: i + 1 for i, lab in enumerate(order.label)}
    bx.axhline(0, color=S.INK2, lw=0.5, zorder=1)
    sz = lambda d: 10 + 0.2 * d
    for _, r in pr.iterrows():
        bx.scatter(r.lead, r.end_off, s=sz(r.dist), marker=MK[r.basin], fc=SB[r.basin], ec="white", lw=0.7,
                   zorder=4)
    CASE = {(1531100, "1976-08-07"): "Case 1", (1531450, "1997-08-02"): "Case 2", (1731400, "1983-09-06"): "Case 3"}
    OFFN = {1: (6, 0), 2: (6, 3), 3: (6, 0), 4: (-7, 0), 5: (0, 6), 6: (6, 0), 7: (6, 0), 8: (7, -3), 9: (6, -5),
            10: (-5, 5)}
    for _, r in pr.iterrows():
        k_ = num[r.label]
        dxo, dyo = OFFN[k_]
        bx.annotate(str(k_), (r.lead, r.end_off), xytext=(dxo, dyo), textcoords="offset points", fontsize=6.4,
                    fontweight="bold", ha="center", va="center", color=S.INK, zorder=5, path_effects=halo)
    bx.text(51.5, 1.2, "nested", fontsize=6.4, color="#8a6a1f", ha="right", va="bottom")
    bx.text(51.5, -1.2, "relay", fontsize=6.4, color="#2d6a4f", ha="right", va="top")
    # observed against permutation-null class shares
    ia = bx.inset_axes([24.0, 20.0, 27.0, 8.5], transform=bx.transData)
    for j, (lab, vals) in enumerate((("observed", [o_[c] / 10 for c in ("near", "relay", "nested")]),
                                     ("null", [sh[c] for c in ("near", "relay", "nested")]))):
        left = 0.0
        for v, c in zip(vals, ("near", "relay", "nested")):
            ia.barh(1 - j, v, left=left, height=0.62, color=CLS_C[c], lw=0.4, ec="white")
            left += v
        ia.text(-0.03, 1 - j, lab, ha="right", va="center", fontsize=6, color=S.INK)
    ia.set_xlim(0, 1); ia.set_ylim(-0.5, 1.5); ia.axis("off")
    bx.text(-0.5, 29, "near-simultaneous", rotation=90, fontsize=6, color=S.INK2, ha="center", va="top")
    bx.set_xlim(-2.5, 52); bx.set_ylim(-45, 30)
    bx.set_xlabel("Onset lead of upstream event (d)")
    bx.set_ylabel("Upstream end − origin end (d)")
    S.hgrid(bx)
    for d in (150, 300, 450):
        bx.scatter([], [], s=sz(d), fc="white", ec=S.INK2, lw=0.6, label=f"{d}")
    bx.legend(title="channel distance (km)", loc="lower right", bbox_to_anchor=(1.0, 0.03), fontsize=6, title_fontsize=6,
              ncols=3, columnspacing=1.1, handletextpad=0.35, borderpad=0.2, frameon=False)
    letter(fig, 0.618, ty(0.965), "b", "Timing geometry")

    # ---------------------------------------------------------------- (c) deficit ribbons, (d) severity balance
    # Each pair is drawn as a mirrored ribbon aligned at the origin onset s0: the daily deficit tau - SSI of the
    # upstream event above the row baseline and of the origin event below it, so ribbon length is duration and,
    # within a pair, ribbon height is daily intensity and ribbon area the event severity of Eq. (2). Heights are
    # scaled to the larger daily deficit of each pair (deficits span 0.05-1.7 across pairs); panel d gives the
    # severities on one scale. Dots mark each event's day of largest deficit. A superposed-epoch strip on top
    # counts the events active on each day relative to s0 across the ten pairs.
    TAU, HMAX, GAP = -1.28, 0.44, 0.045        # drought threshold; ribbon height of the larger peak; half gap
    ssi = pd.read_csv(T / "SSI_daily.csv", usecols=["node_id", "date", "SSI"], parse_dates=["date"])
    ssi = ssi.set_index(["node_id", "date"]).SSI.sort_index()
    UPC = "#8d96a0"                              # upstream colour of the composite (all sub-basins)
    OVC = "#eef1f4"                              # overlap band
    X0, X1 = -60, 72

    def deficit(node, a, b, s0):
        x = ssi.loc[node].loc[a:b]
        assert len(x) == (b - a).days + 1 and x.notna().all()
        d = np.clip(TAU - x.values, 0, None)
        return (x.index - s0).days.values, d

    groups = [("relay", "Relay"), ("nested", "Nested"), ("near", "Near-simultaneous")]
    rows, heads, yy = [], [], 0.0
    for c, h in groups:
        heads.append((yy, h)); yy -= 0.85
        for _, r in pr[pr.cls == c].sort_values("lead", ascending=False, kind="stable").iterrows():
            rows.append((yy, r)); yy -= 1.0
        yy -= 0.25
    cx = fig.add_axes([0.2, fy(11.5), 0.5, fy(64.0)])
    ex = fig.add_axes([0.2, fy(79.0), 0.5, fy(13.5)], sharex=cx)
    dx_ = fig.add_axes([0.765, fy(11.5), 0.215, fy(64.0)], sharey=cx)
    n_up, n_or = np.zeros(X1 - X0 + 1), np.zeros(X1 - X0 + 1)
    days = np.arange(X0, X1 + 1)
    ext, peaks = [], []                         # ribbon extents for the overlap check; days of largest deficit
    for y, r in rows:
        col = SB[r.basin]
        s0 = r.origin_start
        tu, du = deficit(r.upstream_node, r.upstream_start, r.upstream_end, s0)
        to, do = deficit(r.origin_node, r.origin_start, r.origin_end, s0)
        assert abs(du.sum() - r.upstream_severity) < 2e-3 and abs(do.sum() - r.origin_severity) < 2e-3
        KD = HMAX / max(du.max(), do.max())
        peaks.append((r.label, int(tu[du.argmax()]), int(to[do.argmax()])))
        n_up += (days >= tu[0]) & (days <= tu[-1]); n_or += (days >= to[0]) & (days <= to[-1])
        ext.append((y, GAP + KD * du.max(), GAP + KD * do.max()))
        lo, hi = max(tu[0], to[0]), min(tu[-1], to[-1])
        if hi >= lo:                                                    # overlap of the two event spans
            cx.add_patch(Rectangle((lo - 0.5, y - 0.42), hi - lo + 1, 0.84, fc=OVC, ec="none", zorder=1))
        cx.plot([X0, X1], [y, y], color=S.GRID, lw=0.4, zorder=0)
        # event spans (duration stays visible where the deficit is small) and deficit ribbons
        cx.plot([tu[0] - 0.5, tu[-1] + 0.5], [y + GAP, y + GAP], color=col, lw=0.9, solid_capstyle="butt", zorder=3)
        cx.plot([to[0] - 0.5, to[-1] + 0.5], [y - GAP, y - GAP], color=NAVY, lw=0.9, solid_capstyle="butt", zorder=3)
        cx.fill_between(tu, y + GAP, y + GAP + KD * du, step="mid", fc=col, ec="none", lw=0, zorder=4)
        cx.fill_between(to, y - GAP, y - GAP - KD * do, step="mid", fc=NAVY, ec="none", lw=0, zorder=4)
        cx.scatter(tu[du.argmax()], y + GAP + KD * du.max(), s=7, fc=col, ec="white", lw=0.5, zorder=5)
        cx.scatter(to[do.argmax()], y - GAP - KD * do.max(), s=7, fc=NAVY, ec="white", lw=0.5, zorder=5)
        lab = f"{r.lag_days:+d} d".replace("-", M).replace("+0", "0")
        cx.text(tu[0] - 1.6, y + 0.17, lab, fontsize=6, ha="right", va="center", color=S.INK2, zorder=6,
                path_effects=halo)
        # (d) severity balance: one bar of pair severity, split into upstream and origin parts
        su, so = r.upstream_severity, r.origin_severity
        dx_.barh(y, su, height=0.5, color=col, lw=0, zorder=3)
        dx_.barh(y, so, left=su, height=0.5, color=NAVY, lw=0, zorder=3)
        dx_.plot([su, su], [y - 0.25, y + 0.25], color="white", lw=1.0, zorder=4)     # surface gap
        dx_.text(su + so + 0.9, y, f"{su / (su + so):.0%}", fontsize=6, va="center", ha="left", color=S.INK2)
    for (ya, ua, da), (yb, ub, db) in zip(ext[:-1], ext[1:]):
        assert ya - da > yb + ub, "ribbons of neighbouring rows touch"
    for (y, h), c in zip(heads, ("relay", "nested", "near")):
        cx.add_patch(Rectangle((X0, y - 0.3), 2.8, 0.6, fc=CLS_C[c], ec="none", zorder=6))
        cx.text(X0 + 4.2, y, h, fontsize=6.4, fontweight="bold", color=S.INK, ha="left", va="center", zorder=6,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    cx.axvline(0, color=S.INK, lw=0.6, zorder=5)

    def rlab(r):
        c = CASE.get((r.origin_node, f"{r.origin_start:%Y-%m-%d}"))
        return f"{num[r.label]}  {r.pair}, {r.origin_start:%Y}" + (f" ({c})" if c else "")
    cx.set_yticks([y for y, _ in rows]); cx.set_yticklabels([rlab(r) for _, r in rows], fontsize=6.3)
    cx.set_xlim(X0, X1); cx.set_ylim(yy + 0.6, 0.55)
    cx.set_xlabel("Days relative to origin onset $s_0$")
    S.hgrid(cx, axis="x"); cx.tick_params(axis="y", length=0)
    from matplotlib.legend_handler import HandlerTuple
    up_h = tuple(Patch(fc=SB[b], ec="none") for b in ("Black Volta", "White Volta", "Oti–Pendjari"))
    cx.legend(handles=[up_h, Patch(fc=NAVY, ec="none"), Patch(fc=OVC, ec="none")],
              labels=["upstream deficit (sub-basin)", "origin deficit", "overlap"],
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
              loc="upper right", bbox_to_anchor=(1.0, -0.115), ncols=3, fontsize=6.2, handlelength=2.2,
              columnspacing=1.2, borderaxespad=0, frameon=False)
    # superposed-epoch strip: events active on each day relative to s0
    ex.fill_between(days, 0, n_up, step="mid", fc=UPC, ec="none", lw=0, zorder=3)
    ex.fill_between(days, 0, -n_or, step="mid", fc=NAVY, ec="none", lw=0, zorder=3)
    ex.axhline(0, color="white", lw=0.8, zorder=4)
    ex.axvline(0, color=S.INK, lw=0.6, zorder=5)
    ex.set_ylim(-11, 11); ex.set_yticks([-10, -5, 0, 5, 10]); ex.set_yticklabels(["10", "5", "0", "5", "10"])
    ex.set_ylabel("events\nactive", fontsize=6.2, labelpad=2)
    ex.text(X0 + 1, 9.5, "upstream", fontsize=6, color=S.INK2, ha="left", va="top")
    ex.text(X0 + 1, -9.5, "origin", fontsize=6, color=S.INK2, ha="left", va="bottom")
    ex.tick_params(axis="x", labelbottom=False, length=0)
    S.hgrid(ex, axis="x"); ex.spines["bottom"].set_visible(False)
    letter(fig, 0.008, fy(98.0), "c", "Deficit alignment")
    dx_.set_xlabel("Standardised severity of the pair")
    dx_.tick_params(axis="y", labelleft=False, length=0)
    dx_.set_xlim(0, 50); S.hgrid(dx_, axis="x"); dx_.spines["left"].set_visible(False)
    letter(fig, 0.745, fy(98.0), "d", "Severity balance")
    dx_.text(1.0, 1.012, "% = upstream share", transform=dx_.transAxes, fontsize=6, color=S.INK2, ha="right",
             va="bottom")
    S.save(fig, "Fig05_chain_geometry")
    pk = pd.DataFrame(peaks, columns=["label", "upstream_peak_day", "origin_peak_day"])
    pr = pr.merge(pk, on="label")
    pr[["label", "cls", "lead", "end_off", "dist", "upstream_severity", "origin_severity", "chain_severity",
        "upstream_peak_day", "origin_peak_day"]].to_csv(
        ST / "Fig05_chain_geometry_data.csv", index=False)
    print(pr.groupby("cls").agg(n=("lead", "size"), lead_med=("lead", "median"), dist_med=("dist", "median"),
                                up_share=("upstream_severity", lambda s: (s / pr.loc[s.index, "chain_severity"]).median())))


# ======================================================================================================= Fig. S19
def r_detect(n, power=0.8, alpha=0.05, f=1.06):
    return np.tanh((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * np.sqrt(f / (np.asarray(n, float) - 3)))


def s19():
    """Article Fig. S18: drawn by figS_redesign.s18 (writes out/FigS19_landscape_power.*)."""
    from figS_redesign import s18
    s18()


def s20():
    sie = pd.read_csv(ST / "TableS38.csv")
    sie["node"] = [st.index[st.station_name.str.upper() == s.upper()][0] for s in sie.Station]
    basin = gpd.read_file(R / "data/GRDC_VOLTA_POLITICAL_TRANSBOUNDARY_BASIN.shp").to_crs(4326)
    reach = pd.read_parquet(R / "data/reanalysis/reaches_volta.parquet")
    rr = reach[reach.area_km2 >= 2000].set_index("LINKNO")
    segs, lws = [], []
    for _, r in rr.iterrows():
        if r.DSLINKNO in rr.index:
            d = rr.loc[r.DSLINKNO]
            segs.append([(r.lon, r.lat), (d.lon, d.lat)])
            lws.append(0.2 + 0.9 * (np.log10(r.area_km2) - np.log10(2000)) / (np.log10(4e5) - np.log10(2000)))
    fig = plt.figure(figsize=(S.ONEHALF, 74 * S.MM))
    ax = fig.add_axes([0.0, 0.17, 0.45, 0.8])
    basin.plot(ax=ax, color="#f7f7f5", ec=S.INK, lw=0.6, zorder=0)
    ax.add_collection(LineCollection(segs, linewidths=lws, colors="#9cc3e6", capstyle="round", zorder=1))
    an = st[st.analysed_1961_2007]
    ax.scatter(an.lon, an.lat, s=7, fc="white", ec="#8a929b", lw=0.6, zorder=3)
    vmin, vmax = 100, 190
    sc = ax.scatter(st.lon[sie.node], st.lat[sie.node], c=sie["Source WHC (mm)"], cmap=S.BLUES, vmin=vmin, vmax=vmax,
                    s=30, ec=S.INK, lw=0.5, zorder=4)
    OFF = {"BAMBOI": (5, 0, "left"), "LAMA KARA": (5, -2, "left"), "NAWUNI": (-5, 0, "right"),
           "PORGA": (-2, 7, "center"), "PWALAGU": (-5, 2, "right"), "TIELE": (5, 0, "left")}
    for _, r in sie.iterrows():
        dx, dy, ha = OFF[r.Station]
        ax.annotate(r.Station.title(), (st.lon[r.node], st.lat[r.node]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=6, ha=ha, va="center", path_effects=halo, zorder=5)
    ax.set_aspect(1 / np.cos(np.radians(10.5))); ax.axis("off")
    cax = fig.add_axes([0.07, 0.1, 0.3, 0.025])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("SIEREM water-holding capacity (mm)", fontsize=6.5); cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=6)
    ax.scatter([], [], s=7, fc="white", ec="#8a929b", lw=0.6, label="other analysed gauges")
    ax.legend(loc="upper right", bbox_to_anchor=(1.05, 0.97), fontsize=6, handletextpad=0.2)
    letter(fig, 0.005, 0.955, "a")
    bx = fig.add_axes([0.6, 0.2, 0.37, 0.72])
    bx.scatter(sie["Source WHC (mm)"], sie["Median drought duration (d)"], s=10 + 2.0 * sie.Events,
               c=sie["Source WHC (mm)"], cmap=S.BLUES, vmin=vmin, vmax=vmax, ec=S.INK, lw=0.5, zorder=4)
    OFF2 = {"BAMBOI": (7, 0, "left"), "LAMA KARA": (6, 0, "left"), "NAWUNI": (9, 3, "left"),
            "PORGA": (0, -11, "center"), "PWALAGU": (7, 0, "left"), "TIELE": (0, 8, "center")}
    for _, r in sie.iterrows():
        dx, dy, ha = OFF2[r.Station]
        bx.annotate(f"{r.Station.title()} ({r.Events})", (r["Source WHC (mm)"], r["Median drought duration (d)"]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=6, va="center", ha=ha)
    bx.set_xlim(100, 195); bx.set_ylim(0, 24)
    bx.set_xlabel("Water-holding capacity (mm)"); bx.set_ylabel("Median drought duration (d)")
    bx.text(0.03, 0.03, "n = 6 name-matched stations; descriptive only\n(number of events in brackets)",
            transform=bx.transAxes, fontsize=6, color=S.INK2)
    S.hgrid(bx)
    letter(fig, 0.5, 0.955, "b")
    S.save(fig, "FigS20_sierem_map")


# ======================================================================================================= Fig. S22
def s22():
    ivm = pd.read_csv(ST / "TableS42.csv")
    base = ivm["Brier loss"].iloc[0]
    lab = ["training-rate baseline", "antecedent SPI-6", "SPI-6 + slope", "SPI-6 + clay", "SPI-6 + lateral area"]
    bss = (1 - ivm["Brier loss"] / base).values
    grp = pd.read_csv(ST / "TableS58.csv")
    grp["name"] = [NAME[int(g)] for g in grp["Downstream gauge"]]
    assert grp["Accepted matches"].sum() == 10 and grp["Eligible links"].sum() == 67
    fig = plt.figure(figsize=(S.DOUBLE, 58 * S.MM))
    ax = fig.add_axes([0.135, 0.22, 0.24, 0.66])
    yy = np.arange(len(lab))[::-1]
    for y0, v in zip(yy, bss):
        if v != 0:
            ax.barh(y0, v, height=0.45, color=S.VOLTA, lw=0)
        ax.text(v - 0.006 if v < 0 else 0.006, y0, mfmt(v, "+.2f") if v else "reference", fontsize=6,
                ha="right" if v < 0 else "left", va="center", color=S.INK)
    ax.axvline(0, color=S.INK, lw=0.6)
    ax.set_yticks(yy); ax.set_yticklabels(lab); ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.3, 0.1); ax.set_xlabel("Brier skill vs baseline")
    S.hgrid(ax, axis="x"); letter(fig, 0.005, 0.93, "a")
    bx = fig.add_axes([0.415, 0.22, 0.2, 0.66], sharey=ax)
    bx.axvspan(0.1, 0.5, color="#f6eeee", lw=0, zorder=0)
    bx.axvline(0.5, color=S.INK2, lw=0.6)
    bx.scatter(ivm["ROC AUC"], yy, s=20, color=NAVY, zorder=4)
    for y0, v in zip(yy, ivm["ROC AUC"]):
        bx.text(v + 0.02, y0, f"{v:.2f}", fontsize=6, va="center")
    bx.text(0.48, yy.min() - 0.45, "below chance", fontsize=6, ha="right", va="center", color=S.INK2)
    ax.set_ylim(-0.75, yy.max() + 0.5)
    bx.set_xlim(0.1, 0.65); bx.set_xlabel("ROC AUC, held-out gauge")
    bx.tick_params(axis="y", labelleft=False, length=0); S.hgrid(bx, axis="x"); letter(fig, 0.4, 0.93, "b")
    cx = fig.add_axes([0.73, 0.22, 0.25, 0.66])
    k = grp["Accepted matches"].values; nrec = grp["Eligible links"].values
    lo, hi = wilson(k, nrec)
    yg = np.arange(len(grp))[::-1]
    pooled = k.sum() / nrec.sum()
    cx.axvline(pooled, color=S.INK, lw=0.6)
    for y0, kk, nn, l_, h_ in zip(yg, k, nrec, lo, hi):
        cx.plot([l_, h_], [y0, y0], color=S.VOLTA, lw=1.1)
        cx.scatter(kk / nn, y0, s=8 + 1.0 * nn, color=S.VOLTA, ec="white", lw=0.5, zorder=4)
        cx.text(0.99, y0, f"{kk}/{nn}", fontsize=6, va="center", ha="right", color=S.INK2)
    cx.set_yticks(yg); cx.set_yticklabels(grp.name); cx.tick_params(axis="y", length=0)
    cx.set_xlim(0, 1.0); cx.set_ylim(-0.6, len(grp) - 0.2)
    cx.set_xlabel("Matches per link record (95% Wilson)")
    cx.text(pooled + 0.02, len(grp) - 0.45, f"pooled {k.sum()}/{nrec.sum()}", fontsize=6, color=S.INK2, va="center")
    S.hgrid(cx, axis="x"); letter(fig, 0.655, 0.93, "c")
    S.save(fig, "FigS22_link_skill")


# ======================================================================================================= Fig. S23
def ll(p, k, y):
    return np.sum(np.log(1 - (1 - p) ** k[y])) + np.sum(k[~y] * np.log(1 - p))


def fit_ci(sub):
    k = sub.n_observing.values.astype(float); y = sub.connected.values.astype(bool)
    res = minimize_scalar(lambda p: -ll(p, k, y), bounds=(1e-6, 1 - 1e-6), method="bounded")
    ph, lmax = res.x, -res.fun
    f = lambda p: 2 * (lmax - ll(p, k, y)) - 3.841
    lo_ = brentq(f, 1e-6, ph) if f(1e-6) > 0 else 0.0
    return ph, lo_, brentq(f, ph, 1 - 1e-6)


def s23():
    src = R / "output_identifiability/tables/antecedent_events.csv"   # perlink_sensitivity.py (37 events)
    ae = pd.read_csv(src if src.exists() else ST / "FigS23_antecedent_events.csv")
    ae["terc"] = pd.qcut(ae.ante, 3, labels=["drier", "intermediate", "wetter"])
    fig = plt.figure(figsize=(S.ONEHALF, 64 * S.MM))
    ax = fig.add_axes([0.13, 0.2, 0.5, 0.62])
    rng = np.random.default_rng(7)
    for conn_, y0, c in ((False, 0, GREY2), (True, 1, S.VOLTA)):
        s = ae[ae.connected == conn_]
        ax.scatter(s.ante, y0 + rng.uniform(-0.14, 0.14, len(s)), s=8 + 8 * s.n_observing, fc=c, ec="white", lw=0.5,
                   zorder=4)
    cuts = []
    for t in ("drier", "intermediate"):
        a, b = ae[ae.terc == t].ante.max(), ae.ante[ae.ante > ae[ae.terc == t].ante.max()].min()
        cuts.append((a + b) / 2)
        ax.axvline(cuts[-1], color=S.INK2, lw=0.5)
    edges = [ae.ante.min() - 0.03] + cuts + [ae.ante.max() + 0.03]
    for t, x0, x1 in zip(("drier", "intermediate", "wetter"), edges[:-1], edges[1:]):
        ax.text((x0 + x1) / 2, 1.42, t, ha="center", fontsize=6.3, color=S.INK)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["unconnected", "connected"]); ax.set_ylim(-0.4, 1.58)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Mean daily SSI in the 30 days before origin onset")
    for kk in (1, 2, 4):
        ax.scatter([], [], s=8 + 8 * kk, fc="#d5d8dc", ec=S.INK2, lw=0.5, label=f"k = {kk}")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=3, fontsize=6, title="observing links",
              title_fontsize=6, borderaxespad=0.1, alignment="left")
    S.hgrid(ax, axis="x"); letter(fig, 0.005, 0.94, "a")
    bx = fig.add_axes([0.76, 0.2, 0.22, 0.62])
    allp = fit_ci(ae)
    bx.axhspan(allp[1], allp[2], color="#eef0f2", lw=0)
    bx.axhline(allp[0], color=S.INK2, lw=0.6)
    out = []
    for i, t in enumerate(("drier", "intermediate", "wetter")):
        sub = ae[ae.terc == t]
        ph, lo_, hi_ = fit_ci(sub)
        out.append((t, ph, lo_, hi_, int(sub.connected.sum()), len(sub)))
        bx.plot([i, i], [lo_, hi_], color=S.VOLTA, lw=1.1)
        bx.scatter(i, ph, s=20, color=S.VOLTA, zorder=4)
        bx.text(i + 0.13, ph, f"{ph:.2f}\n{int(sub.connected.sum())}/{len(sub)}", fontsize=6, va="center",
                linespacing=1.0)
    assert [round(o[1], 3) for o in out] == [0.099, 0.105, 0.206]
    bx.set_xticks([0, 1, 2]); bx.set_xticklabels(["drier", "inter-\nmediate", "wetter"])
    bx.set_xlim(-0.4, 2.7); bx.set_ylim(0, 0.6)
    bx.set_ylabel("Per-link $\\hat{p}$ (95% likelihood interval)")
    bx.text(0.03, 0.98, f"dashed, shaded: all 37 events,\n$\\hat{{p}}$ = {allp[0]:.2f} ({allp[1]:.2f}–{allp[2]:.2f})",
            transform=bx.transAxes, fontsize=6, ha="left", va="top", color=S.INK2)
    S.hgrid(bx); letter(fig, 0.655, 0.94, "b")
    S.save(fig, "FigS23_antecedent")
    pd.DataFrame(out, columns=["tercile", "p_hat", "lo95", "hi95", "connected", "events"]).to_csv(
        ST / "FigS23_tercile_estimates.csv", index=False)
    print(out, allp)


# ======================================================================================================= Fig. S24
def s24():
    mm = pd.read_csv(ST / "TableS39.csv")
    mm["bss"] = [1 - b / mm[(mm.Test == t) & (mm.Model == "Season")].Brier.iloc[0] for t, b in zip(mm.Test, mm.Brier)]
    prev = mm.groupby("Test").first().eval("Drought / n")
    models = ["Season", "Logit SPI", "Logit SPI+L", "RF SPI", "RF SPI+L", "HGB SPI", "HGB SPI+L"]
    nice = {"Season": "seasonal baseline", "Logit SPI": "logistic, SPI", "Logit SPI+L": "logistic, SPI + L",
            "RF SPI": "random forest, SPI", "RF SPI+L": "random forest, SPI + L", "HGB SPI": "boosting, SPI",
            "HGB SPI+L": "boosting, SPI + L"}
    ypos = np.array([8.0, 6.6, 5.8, 4.4, 3.6, 2.2, 1.4])
    fig = plt.figure(figsize=(S.DOUBLE, 64 * S.MM))
    panels = [("AUC", "ROC AUC", (0.5, 0.71), 0.5, "chance"), ("AP", "Average precision", (0.1, 0.28), None, None),
              ("bss", "Brier skill vs seasonal", (-0.45, 0.06), 0.0, "seasonal")]
    x0s = [0.175, 0.455, 0.735]
    for j, (col, xl, lim, ref, reftxt) in enumerate(panels):
        ax = fig.add_axes([x0s[j], 0.2, 0.245, 0.66])
        if ref is not None:
            ax.axvline(ref, color=S.INK2, lw=0.6)
        if col == "AP":
            for t, ls in (("Gauge holdout", (0, (3, 2))), ("Later years", (0, (1, 1.5)))):
                ax.axvline(prev[t], color=TC[t], lw=0.6, ls=ls, alpha=0.8)
            ax.text(0.102, 0.9, "vertical lines: event prevalence", fontsize=6, color=S.INK2, va="bottom")
        for y0, m_ in zip(ypos, models):
            a = mm[(mm.Model == m_) & (mm.Test == "Gauge holdout")][col].iloc[0]
            b = mm[(mm.Model == m_) & (mm.Test == "Later years")][col].iloc[0]
            ax.plot([a, b], [y0, y0], color="#c3c7cd", lw=1.4, zorder=1, solid_capstyle="round")
            for v, t in ((a, "Gauge holdout"), (b, "Later years")):
                ax.scatter(v, y0, s=20, color=TC[t], ec="white", lw=0.5, zorder=4)
        ax.set_xlim(*lim); ax.set_ylim(0.8, 8.6)
        ax.set_yticks(ypos)
        ax.set_yticklabels([nice[m_] for m_ in models] if j == 0 else [])
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(xl)
        for yl in (7.3, 5.1, 2.9):
            ax.axhline(yl, color=GRID_C, lw=0.4)
        S.hgrid(ax, axis="x")
        letter(fig, x0s[j] - (0.165 if j == 0 else 0.02), 0.94, "abc"[j])
        if j == 0:
            ax.legend(handles=[Line2D([], [], marker="o", ls="", color=TC[t],
                                      label=f"{TLAB[t]} (n = {mm[mm.Test == t].n.iloc[0]:,})")
                               for t in TC], loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncols=2, fontsize=6.2,
                      borderaxespad=0.2, columnspacing=1.0, handletextpad=0.1)
    fig.text(0.175, 0.025, "L = HydroATLAS landscape attributes (soil water, aridity, slope). SPI = GPCC SPI-3/6/12 of the "
                           "preceding month.", fontsize=6, color=S.INK2)
    S.save(fig, "FigS24_month_plane")


GRID_C = "#e6e8eb"


# ======================================================================================================= Fig. S25
def s25():
    md = pd.read_csv(ST / "TableS40.csv")
    cmp_ = [("Logit SPI", "Season"), ("RF SPI", "Season"), ("HGB SPI", "Season"),
            ("Logit SPI+L", "Logit SPI"), ("RF SPI+L", "RF SPI"), ("HGB SPI+L", "HGB SPI")]
    lab = {"Logit SPI": "logistic", "RF SPI": "random forest", "HGB SPI": "boosting",
           "Logit SPI+L": "logistic", "RF SPI+L": "random forest", "HGB SPI+L": "boosting"}
    fig = plt.figure(figsize=(S.ONEHALF, 74 * S.MM))
    ax = fig.add_axes([0.2, 0.15, 0.77, 0.7])
    ax.axvspan(0, 0.012, color="#eef6f1", lw=0, zorder=0)
    ax.axvline(0, color=S.INK, lw=0.6)
    ypos = [7, 6, 5, 3, 2, 1]
    for y0, (new, ref) in zip(ypos, cmp_):
        pts = []
        for test, dy in (("Gauge holdout", 0.16), ("Later years", -0.16)):
            r = md[(md.Test == test) & (md["New model"] == new) & (md.Reference == ref)].iloc[0]
            c = TC[test]
            excl = r["CI lower"] > 0 or r["CI upper"] < 0
            ax.plot([r["CI lower"], r["CI upper"]], [y0 + dy] * 2, color=c, lw=1.0)
            ax.scatter(r.Effect, y0 + dy, s=18, fc=c if excl else "white", ec=c, lw=1.0, zorder=4)
            pts.append((r.Effect, y0 + dy))
        ax.plot(*zip(*pts), color="#c3c7cd", lw=0.5, zorder=1)
    ax.set_yticks(ypos); ax.set_yticklabels([lab[n] for n, _ in cmp_]); ax.tick_params(axis="y", length=0)
    tr = ax.get_yaxis_transform()
    ax.text(-0.02, 7.85, "SPI added to season", fontsize=6.4, fontweight="bold", color=S.INK, ha="right",
            transform=tr)
    ax.text(-0.02, 3.85, "Landscape added to SPI", fontsize=6.4, fontweight="bold", color=S.INK, ha="right",
            transform=tr)
    ax.text(0.0115, 8.15, "extended\nmodel better", fontsize=6, color="#2d6a4f", ha="right", va="top",
            linespacing=1.0)
    ax.set_xlim(-0.056, 0.012); ax.set_ylim(0.4, 8.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: mfmt(v, ".2f") if v else "0"))
    ax.set_xlabel("Brier loss, reference − extended model (95% gauge-bootstrap interval)")
    S.hgrid(ax, axis="x")
    ax.legend(handles=[Line2D([], [], marker="o", color=TC["Gauge holdout"], label="withheld gauges"),
                       Line2D([], [], marker="o", color=TC["Later years"], label="1995–2007 test"),
                       Line2D([], [], marker="o", mfc="white", color=S.INK2, label="interval includes 0")],
              loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncols=3, fontsize=6.2, borderaxespad=0.2)
    S.save(fig, "FigS25_month_transfer")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["fig5", "s19"]
    for f in todo:
        globals()[f]()
    print("done", todo)
