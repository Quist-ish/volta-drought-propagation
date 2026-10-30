#!/usr/bin/env python3
"""
Supplementary Figs. S9, S17 and S18, drawn so that the encoding carries the result and text is kept to
axis labels, short panel titles and a few values.

  s9()   Fig. S9  matching rules on a constructed example: daily SSI of the origin and four upstream
                  candidates as anomaly bars (drought days below the threshold in red), the search window,
                  the verdict per candidate, the example on the network and the three conditions
                  (writes out/FigS_matching_rules.*)
  s17()  Fig. S17 origin events by testability: events per onset year (matched and testable above zero,
                  not testable below) and the number of upstream gauges per origin event as violins for the
                  Ebro, the mapped Volta network and the observing Volta network (writes out/FigS11_opportunities.*)
  s18()  Fig. S18 landscape and storage bounds: Spearman rho with 95% intervals against the detection limit,
                  the detectable |rho| against the number of stations with a 50-95% power band, and the
                  recession timescale against drought duration with a Theil-Sen line and bootstrap band
                  (writes out/FigS19_landscape_power.*)

Called by figS_matching_rules.py, figS_a.py and fig05_landscape_panels.py (s19), so reproduce.sh is unchanged.
Colour roles: matched or accepted orange (the Volta colour), observing but unmatched slate, not observing or
not testable grey, Ebro blue; HydroATLAS attributes navy and storage signatures orange in Fig. S18; drought
days red and wet days blue in Fig. S9. The S9 series are synthetic (seeded), as the figure is a constructed
example. No dashed or dotted lines (pubstyle.save refuses them).
Author: I. Quist (NUIST)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Circle
from matplotlib.ticker import FuncFormatter, FixedLocator
from scipy.stats import norm, gaussian_kde, theilslopes
import pubstyle as S

S.setup()
H = Path(__file__).resolve().parent
R = H.parents[0]
T = R / "output_observational/tables"
ST = H / "supp_tables"
ID = R / "output_identifiability/tables"

ORIGIN, MATCH, SLATE = S.NAVY, S.VOLTA, "#5f7189"
GREY, GREY_EC = "#d3d8de", "#a3abb4"
DRY, NEG, POS = "#c8382c", "#f2bcae", "#5b9bd1"       # drought day, other negative day, positive day
WIN = "#e8eef6"
FRAME = "#9aa3ad"
HALO = [pe.withStroke(linewidth=2.0, foreground="white")]


def letter(fig, x, y, s, title=None):
    fig.text(x, y, s, fontsize=8, fontweight="bold", ha="left", va="baseline", color=S.INK)
    if title:
        fig.text(x + 0.018, y, title, fontsize=7, ha="left", va="baseline", color=S.INK)


def boxed(ax):
    """Thin light frame on all four sides, as in small-multiple anomaly plots."""
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(FRAME)
        sp.set_linewidth(0.5)


# ============================================================================================ Fig. S9
def _ssi(seed, t, events, tau=-1.28):
    """Synthetic daily SSI: AR(1) noise above the threshold outside events, a dip below it inside events."""
    rng = np.random.default_rng(seed)
    x = np.empty(len(t))
    e = rng.normal(0, 1, len(t))
    x[0] = e[0]
    for i in range(1, len(t)):
        x[i] = 0.88 * x[i - 1] + np.sqrt(1 - 0.88 ** 2) * e[i]
    x = 0.25 + 0.7 * x
    ev = np.zeros(len(t), bool)
    for s, f in events:
        ev |= (t >= s) & (t <= f)
    x[~ev] = np.maximum(x[~ev], tau + 0.1)
    for s, f in events:
        m = (t >= s) & (t <= f)
        shape = np.sin(np.pi * (t[m] - s + 0.5) / (f - s + 1))
        x[m] = np.minimum(tau - 0.12 - 1.0 * shape + 0.12 * rng.normal(size=m.sum()), tau - 0.05)
        for k in (1, 2, 3):                            # shoulders that approach the threshold
            for d in (s - k, f + k):
                j = np.flatnonzero((t == d) & ~ev)
                if j.size:
                    x[j[0]] = min(x[j[0]], tau + 0.22 * k + 0.05)
    return x, ev


def s9():
    W, M, TAU = 45, 5, -1.28
    t = np.arange(-84, 53)
    rows = [("Origin", [(0, 37)], 11, None),
            ("A", [(-26, 24)], 21, ("✓", MATCH, "accepted")),
            ("B", [(-33, -29)], 31, ("✕", S.REJECT, "overlap 0 d < $M$")),
            ("C", [(6, 44)], 41, ("✕", S.REJECT, "onset after $s_0$")),
            ("D", [(-72, -58)], 51, ("✕", S.REJECT, "onset before $s_0-W$"))]
    fig = plt.figure(figsize=(S.DOUBLE, 80 * S.MM))
    x0, x1, top, bot = 0.045, 0.585, 0.885, 0.125
    gap = 0.012
    h = (top - bot - gap * (len(rows) - 1)) / len(rows)
    axes = []
    for i, (lab, evs, seed, verdict) in enumerate(rows):
        ax = fig.add_axes([x0, top - (i + 1) * h - i * gap, x1 - x0, h])
        axes.append(ax)
        x, ev = _ssi(seed, t, evs, TAU)
        col = np.where(ev, DRY, np.where(x < 0, NEG, POS))
        ax.axvspan(-W, 0, color=WIN, lw=0, zorder=0)
        ax.bar(t, x, width=0.82, color=col, lw=0, zorder=2)
        ax.axhline(0, color=S.INK2, lw=0.4, zorder=3)
        ax.axhline(TAU, color="#7d8794", lw=0.5, zorder=3)
        ax.axvline(0, color=S.INK, lw=0.8, zorder=4)
        ax.set_xlim(t[0] - 1, t[-1] + 1); ax.set_ylim(-2.75, 1.95)
        ax.set_yticks([]); boxed(ax)
        ax.tick_params(axis="x", length=2, labelbottom=(i == len(rows) - 1))
        ax.set_xticks(range(-80, 51, 20))
        fig.text(x0 - 0.006, top - (i + 0.5) * h - i * gap, lab, fontsize=7, fontweight="bold", va="center",
                 ha="right", color=S.INK)
        if verdict:
            g, c, why = verdict
            fig.text(x1 + 0.008, top - (i + 0.5) * h - i * gap, g, fontsize=10, color=c, va="center",
                     ha="left", fontweight="bold")
            fig.text(x1 + 0.03, top - (i + 0.5) * h - i * gap, why, fontsize=6.3, color=S.INK2, va="center",
                     ha="left")
        if lab == "A":                                       # overlap with the origin event and the lag
            ax.plot([0, 24], [1.45, 1.45], color=MATCH, lw=1.1, solid_capstyle="butt", zorder=5)
            for xx in (0, 24):
                ax.plot([xx, xx], [1.2, 1.7], color=MATCH, lw=0.8, zorder=5)
            ax.text(25.5, 1.45, "overlap 25 d ≥ $M$", fontsize=6.2, color=MATCH, va="center", ha="left",
                    path_effects=HALO, zorder=6)
            ax.annotate("", xy=(-26, 1.45), xytext=(0, 1.45),
                        arrowprops=dict(arrowstyle="<|-|>", mutation_scale=5, lw=0.6, color=S.INK2), zorder=5)
            ax.text(-27.5, 1.45, "λ = −26 d", fontsize=6.2, color=S.INK2, va="center", ha="right",
                    path_effects=HALO, zorder=6)
        if lab == "Origin":
            ax.text(-1.0, TAU, "τ", fontsize=6.5, color="#5b6570", va="bottom", ha="right", style="italic",
                    path_effects=HALO, zorder=6)
            ax.text(38.5, -1.9, "$e_0$", fontsize=6.5, color=S.INK, va="center", ha="left", zorder=6)
    a0 = axes[0]
    a0.annotate("", xy=(-W, 2.55), xytext=(0, 2.55), annotation_clip=False,
                arrowprops=dict(arrowstyle="<|-|>", mutation_scale=5, lw=0.7, color=S.EBRO))
    a0.text(-W / 2, 2.75, "$W$ = 45 d", fontsize=6.5, color=S.EBRO, ha="center", va="bottom", clip_on=False)
    a0.text(0.8, 2.75, "$s_0$", fontsize=6.8, color=S.INK, ha="left", va="bottom", clip_on=False)
    axes[-1].set_xlabel("Days from the origin onset $s_0$")
    leg = [Patch(fc=DRY, label="drought day (SSI < τ)"), Patch(fc=NEG, label="SSI < 0"), Patch(fc=POS, label="SSI > 0")]
    fig.legend(handles=leg, loc="upper right", bbox_to_anchor=(x1, 0.985), ncols=3, fontsize=6.2,
               handlelength=1.0, handleheight=0.8, columnspacing=1.0, borderaxespad=0)
    letter(fig, 0.004, 0.955, "a", "Candidate events")

    # ---------------------------------------------------------------- (b) the example on the network
    nx = fig.add_axes([0.745, 0.33, 0.235, 0.6])
    nx.set_xlim(0, 10); nx.set_ylim(0, 12); nx.set_aspect("equal"); nx.axis("off")

    def bez(p0, p1, p2, k=60):
        u = np.linspace(0, 1, k)[:, None]
        return (1 - u) ** 2 * np.array(p0) + 2 * (1 - u) * u * np.array(p1) + u ** 2 * np.array(p2)

    def river(c, w0, w1, col="#b9d3ee", z=1):
        segs = np.stack([c[:-1], c[1:]], axis=1)
        nx.add_collection(LineCollection(segs, linewidths=np.linspace(w0, w1, len(segs)), colors=col,
                                         capstyle="round", zorder=z))

    main = bez((5.0, 4.4), (5.3, 2.6), (5.0, 0.6))
    left = bez((3.0, 8.0), (3.6, 6.2), (5.0, 4.4))
    river(bez((2.2, 11.4), (2.4, 9.8), (3.0, 8.0)), 0.9, 1.6)
    river(bez((4.6, 10.6), (4.2, 9.4), (3.0, 8.0)), 0.7, 1.2)
    river(bez((7.6, 7.6), (6.9, 5.6), (5.0, 4.4)), 1.0, 2.0)
    river(bez((8.4, 3.9), (6.9, 2.9), (5.2, 2.7)), 0.8, 1.4)
    river(left, 1.7, 2.6)
    river(main, 2.8, 3.6)
    hl = np.vstack([left, main])
    hl = hl[hl[:, 1] >= 1.15]
    nx.add_collection(LineCollection(np.stack([hl[:-1], hl[1:]], axis=1), linewidths=1.6, colors=MATCH,
                                     capstyle="round", zorder=2))
    for i0 in (18, 70):
        nx.add_patch(FancyArrowPatch(tuple(hl[i0]), tuple(hl[i0 + 3]), arrowstyle="-|>", mutation_scale=7,
                                     color=MATCH, lw=0, zorder=3))
    nodes = {"O": (5.0, 1.1, ORIGIN, "white", 0.6), "A": (3.0, 8.0, MATCH, "white", 0.52),
             "B": (8.4, 3.9, SLATE, "white", 0.45), "C": (7.6, 7.6, SLATE, "white", 0.45),
             "D": (2.2, 11.4, SLATE, "white", 0.45), "5": (4.6, 10.6, GREY, GREY_EC, 0.45)}
    for k, (x, y, fc, ec, r) in nodes.items():
        nx.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=0.9, zorder=5))
    for k, (dx, dy, ha) in {"O": (0.85, 0, "left"), "A": (-0.8, 0, "right"), "B": (0, -0.95, "center"),
                            "C": (0.75, 0, "left"), "D": (-0.75, 0, "right"), "5": (0.7, 0, "left")}.items():
        x, y, *_ = nodes[k]
        nx.text(x + dx, y + dy, "origin" if k == "O" else k, ha=ha, va="center", fontsize=6.5, color=S.INK,
                fontweight="bold" if k in "OA" else "normal", path_effects=HALO, zorder=6)
    letter(fig, 0.715, 0.955, "b", "Network")

    # ---------------------------------------------------------------- (c) the three conditions, in order
    cx = fig.add_axes([0.80, 0.07, 0.17, 0.2])
    for y, (lab, v, c) in zip((2, 1, 0), (("mapped", 5, GREY), ("observing", 4, SLATE), ("matched", 1, MATCH))):
        cx.barh(y, v, height=0.62, color=c, lw=0)
        cx.text(-0.15, y, lab, ha="right", va="center", fontsize=6.4, color=S.INK)
        cx.text(v + 0.15, y, str(v), ha="left", va="center", fontsize=6.8, fontweight="bold", color=S.INK)
    cx.set_xlim(0, 5.8); cx.set_ylim(-0.6, 2.6); cx.axis("off")
    letter(fig, 0.715, 0.29, "c", "Conditions")
    S.save(fig, "FigS_matching_rules")


# ============================================================================================ Fig. S17
def _violin(ax, x, vals, fc, ec, width=0.38, bw=0.32, alpha=0.55):
    """Half-width-normalised KDE violin of log2 values; returns the density function for jitter."""
    lv = np.log2(vals)
    kde = gaussian_kde(lv, bw_method=bw)
    yy = np.linspace(lv.min() - 0.3, lv.max() + 0.3, 300)
    d = kde(yy)
    d = width * d / d.max()
    ax.fill_betweenx(2 ** yy, x - d, x + d, fc=fc, ec=ec, lw=0.6, alpha=alpha, zorder=2)
    return lambda v: np.interp(np.log2(v), yy, d)


def s17():
    op = pd.read_csv(T / "review_opportunities.csv")
    ch = pd.read_csv(T / "Chains_final.csv")
    ev = pd.read_csv(T / "SSI_drought_events.csv", usecols=["node_id", "event_id", "start_date"])
    ch["eid"] = ch.chain_id.str.split("_").str[1].astype(int)
    cs = set(zip(ch.origin_node, ch.eid))
    op["connected"] = [(a, b) in cs for a, b in zip(op.node_id, op.event_id)]
    op = op.merge(ev, on=["node_id", "event_id"], how="left")
    op["year"] = pd.to_datetime(op.start_date).dt.year
    op["status"] = np.where(op.n_observing == 0, "none", np.where(op.connected, "match", "test"))
    N, n0 = len(op), int((op.status == "none").sum())
    nopp, ncon = N - n0, int((op.status == "match").sum())
    assert (N, n0, nopp, ncon) == (104, 64, 40, 10) and op.year.notna().all()
    ek = pd.read_csv(ID / "per_link_model_fit.csv")
    ek = ek[ek.network == "Ebro"]
    ebro_k = np.repeat(ek.k_observing.values, ek.n.values)
    assert len(ebro_k) == 928 and abs(ebro_k.mean() - 9.2) < 0.01

    fig = plt.figure(figsize=(S.DOUBLE, 72 * S.MM))
    # ---------------------------------------------------------------- (a) events per onset year
    ax = fig.add_axes([0.06, 0.14, 0.555, 0.74])
    yrs = np.arange(1969, 2009)
    cnt = {s: np.array([int(((op.year == y) & (op.status == s)).sum()) for y in yrs]) for s in ("match", "test", "none")}
    ax.bar(yrs, cnt["test"], width=0.78, color=SLATE, lw=0, zorder=2)
    ax.bar(yrs, cnt["match"], bottom=cnt["test"], width=0.78, color=MATCH, lw=0, zorder=2)
    ax.bar(yrs, -cnt["none"], width=0.78, color=GREY, lw=0, zorder=2)
    ax.axhline(0, color=S.INK2, lw=0.5, zorder=3)
    ymax = max((cnt["test"] + cnt["match"]).max(), cnt["none"].max()) + 1
    ax.set_ylim(-ymax, ymax); ax.set_xlim(1968.3, 2008.7)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(int(v))}"))
    ax.set_ylabel("Origin events per year")
    ax.set_xlabel("Onset year")
    boxed(ax); S.hgrid(ax)
    ax.text(0.012, 0.95, "testable", transform=ax.transAxes, fontsize=6.4, color=S.INK2, va="top", style="italic")
    ax.text(0.012, 0.05, "not testable", transform=ax.transAxes, fontsize=6.4, color=S.INK2, va="bottom",
            style="italic")
    ax.legend(handles=[Patch(fc=MATCH, label=f"matched ({ncon})"),
                       Patch(fc=SLATE, label=f"unmatched ({nopp - ncon})"),
                       Patch(fc=GREY, label=f"no upstream gauge observing ({n0})")],
              loc="upper right", ncols=3, fontsize=6.3, handlelength=1.0, handleheight=0.8, columnspacing=1.0)
    letter(fig, 0.004, 0.95, "a", "Testability of the 104 origin droughts")

    # ---------------------------------------------------------------- (b) upstream gauges per origin event
    bx = fig.add_axes([0.71, 0.14, 0.275, 0.74])
    rng = np.random.default_rng(7)
    groups = [(0, ebro_k, S.EBRO, None), (1, op.n_upstream.values.astype(float), "#aeb6bf", op.status.values),
              (2, op.n_observing[op.n_observing > 0].values.astype(float), MATCH,
               op.status[op.n_observing > 0].values)]
    pcol = {"match": MATCH, "test": SLATE, "none": "#b3bac2"}
    for x, vals, fc, st in groups:
        dens = _violin(bx, x, vals, fc if x != 2 else "#f7c7ad", fc if x != 1 else "#8d97a2",
                       alpha=0.35 if x == 0 else 0.6)
        if st is not None:
            jit = (rng.uniform(-1, 1, len(vals))) * dens(vals) * 0.85
            order = np.argsort(st == "match")                 # matched drawn last, on top
            bx.scatter((x + jit)[order], (vals * 2 ** rng.uniform(-0.06, 0.06, len(vals)))[order], s=7,
                       c=[pcol[s] for s in st[order]], ec="white", lw=0.25, zorder=4)
        q1, md, q3 = np.percentile(vals, [25, 50, 75])
        bx.plot([x, x], [q1, q3], color=S.INK, lw=1.4, solid_capstyle="butt", zorder=5)
        bx.scatter([x], [vals.mean()], s=16, marker="D", fc="white", ec=S.INK, lw=0.7, zorder=6)
        bx.text(x, 46, f"$\\bar{{k}}$ = {vals.mean():.1f}", ha="center", va="bottom", fontsize=6.4, color=S.INK)
    bx.set_yscale("log", base=2)
    bx.yaxis.set_major_locator(FixedLocator([1, 2, 4, 8, 16, 32]))
    bx.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    bx.yaxis.set_minor_locator(FixedLocator([]))
    bx.set_ylim(0.75, 56); bx.set_xlim(-0.6, 2.6)
    bx.set_xticks([0, 1, 2]); bx.set_xticklabels(["Ebro", "Volta\nmapped", "Volta\nobserving"])
    bx.tick_params(axis="x", length=0)
    bx.set_ylabel("Upstream gauges per origin event, $k$")
    S.hgrid(bx)
    letter(fig, 0.655, 0.95, "b", "Upstream gauges per event")
    S.save(fig, "FigS11_opportunities")


# ============================================================================================ Fig. S18
def r_detect(n, power=0.8, alpha=0.05):
    return np.tanh((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * np.sqrt(1.06 / (np.asarray(n, float) - 3)))


def s18():
    att = pd.read_csv(ST / "TableS37.csv")
    sig = pd.read_csv(ST / "TableS64.csv")
    gs = pd.read_csv(R / "output_landscape/storage_persistence_gauges.csv", index_col=0)
    num = lambda s: float(str(s).replace("−", "-").replace("+", ""))
    ci = lambda s: tuple(num(v) for v in str(s).replace("−", "-").split(" to "))
    rows = []
    for nm, lab in (("Soil water content", "Soil water content"), ("Aridity", "Aridity index"),
                    ("Mean channel-network slope", "Channel slope")):
        a = att[(att.Attribute == nm) & (att.Sample == "All")].iloc[0]
        e = att[(att.Attribute == nm) & (att.Sample == "Exclusions")].iloc[0]
        rows.append(("H", lab, num(a["Spearman ρ"]), ci(a["95% CI"]), num(a["Holm-adjusted p"]),
                     num(e["Spearman ρ"]), ci(e["95% CI"])))
    for nm, lab in (("Baseflow index", "Baseflow index"), ("Recession timescale", "Recession timescale"),
                    ("Zero-flow share", "Zero-flow share")):
        a = sig[(sig.Signature == nm) & (sig.Sample == "All gauges")].iloc[0]
        e = sig[(sig.Signature == nm) & (sig.Sample == "Excluding flagged")].iloc[0]
        rows.append(("S", lab, num(a["Spearman ρ"]), ci(a["95% CI"]), num(a["Holm-adjusted p"]),
                     num(e["Spearman ρ"]), ci(e["95% CI"])))
    r16 = r_detect(16)
    assert abs(r16 - 0.66) < 0.01 and abs(r_detect(13) - 0.72) < 0.01
    COL = {"H": S.NAVY, "S": S.VOLTA}
    rec = [r_ for r_ in rows if r_[1] == "Recession timescale"][0]
    n_rec = int(np.ceil(1.06 * ((norm.ppf(0.975) + norm.ppf(0.8)) / np.arctanh(rec[2])) ** 2 + 3))
    assert n_rec == 48

    fig = plt.figure(figsize=(S.DOUBLE, 70 * S.MM))
    # ---------------------------------------------------------------- (a) associations against the limit
    ax = fig.add_axes([0.125, 0.16, 0.215, 0.72])
    yy = np.array([6.0, 5.0, 4.0, 2.6, 1.6, 0.6])
    ax.axvspan(-r16, r16, color="#eceff3", lw=0, zorder=0)
    ax.axvline(0, color=S.INK2, lw=0.5, zorder=1)
    for y0, (fam, nm, r, c_, holm, re, cie) in zip(yy, rows):
        c = COL[fam]
        ax.plot(c_, [y0 + 0.14] * 2, color=c, lw=3.2, alpha=0.25, solid_capstyle="round", zorder=2)
        ax.plot(c_, [y0 + 0.14] * 2, color=c, lw=0.8, solid_capstyle="butt", zorder=2)
        ax.scatter(r, y0 + 0.14, s=30, color=c, ec="white", lw=0.7, zorder=4)
        ax.plot(cie, [y0 - 0.2] * 2, color=c, lw=0.6, alpha=0.6, solid_capstyle="butt", zorder=2)
        ax.scatter(re, y0 - 0.2, s=17, fc="white", ec=c, lw=0.8, zorder=4)
    ax.set_yticks(yy); ax.set_yticklabels([r_[1] for r_ in rows], fontsize=6.5)
    for tl, r_ in zip(ax.get_yticklabels(), rows):
        tl.set_color(COL[r_[0]])
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-1, 1); ax.set_ylim(0.0, 6.75)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1]); ax.set_xticklabels(["−1", "−0.5", "0", "0.5", "1"])
    ax.set_xlabel("Spearman ρ with drought duration")
    ax.text(0, 6.62, f"|ρ| < {r16:.2f}", fontsize=6.2, ha="center", va="top", color="#6b7580")
    ax.legend(handles=[Line2D([], [], marker="o", color=S.INK2, mec="white", lw=2.0, ms=4.8, label="n = 16"),
                       Line2D([], [], marker="o", mfc="white", color=S.INK2, lw=0.7, ms=4.2, label="n = 13")],
              loc="upper left", bbox_to_anchor=(-0.02, -0.17), ncols=2, fontsize=6.3, borderaxespad=0,
              handlelength=1.6, columnspacing=1.0)
    letter(fig, 0.004, 0.95, "a", "Associations")

    # ---------------------------------------------------------------- (b) detection limit against n
    bx = fig.add_axes([0.43, 0.16, 0.235, 0.72])
    n = np.arange(6, 101)
    bx.fill_between(n, r_detect(n, 0.5), r_detect(n, 0.95), color="#dfe3e8", lw=0, zorder=1)
    bx.plot(n, r_detect(n, 0.8), color=S.INK, lw=1.3, zorder=3)
    for pw, dy in ((0.5, -0.02), (0.8, 0.0), (0.95, 0.02)):
        bx.text(101, r_detect(100, pw) + dy, f"{int(pw * 100)}%", fontsize=6, color=S.INK2 if pw != 0.8 else S.INK,
                va="center", ha="left")
    for nn, lab in ((16, "Volta"), (33, "Ebro")):
        bx.plot([nn, nn], [0, 1], color="#c3c9d0", lw=0.6, zorder=1)
        bx.text(nn, 1.01, lab, fontsize=6.2, color=S.INK2, va="bottom", ha="center")
    for fam, nm, r, *_ in rows:
        bx.scatter(16, abs(r), s=22, color=COL[fam], ec="white", lw=0.5, zorder=5)
    bx.annotate("", xy=(n_rec, rec[2]), xytext=(17.2, rec[2]),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=6, lw=0.7, color=S.VOLTA), zorder=4)
    bx.scatter(n_rec, rec[2], s=26, marker="D", fc="white", ec=S.VOLTA, lw=0.9, zorder=6)
    bx.text(n_rec + 2.5, rec[2] + 0.035, f"{n_rec}", fontsize=6.6, color=S.VOLTA, fontweight="bold", va="bottom")
    bx.set_xlim(5, 100); bx.set_ylim(0, 1)
    bx.set_xlabel("Stations with events, n"); bx.set_ylabel("Detectable |ρ| (α = 0.05)")
    S.hgrid(bx)
    letter(fig, 0.37, 0.95, "b", "Detection limit")

    # ---------------------------------------------------------------- (c) the strongest association
    cx = fig.add_axes([0.75, 0.16, 0.2, 0.72])
    g = gs[gs.n_events > 0].copy()
    assert len(g) == 16
    g["flag"] = g.flag_zero_artefact | g.flag_short_record
    xg, yg = g.tau_rec_d.values, g.median_duration.values
    xs = np.linspace(0, 33, 120)
    rng = np.random.default_rng(42)
    lines = []
    for _ in range(2000):
        i = rng.integers(0, len(g), len(g))
        if np.unique(xg[i]).size < 3:
            continue
        b, a, *_ = theilslopes(yg[i], xg[i])
        lines.append(a + b * xs)
    lo, hi = np.percentile(np.array(lines), [2.5, 97.5], axis=0)
    b, a, *_ = theilslopes(yg, xg)
    cx.fill_between(xs, lo, hi, color="#f6d2c0", lw=0, zorder=1)
    cx.plot(xs, a + b * xs, color=S.VOLTA, lw=1.2, zorder=2)
    cmap = LinearSegmentedColormap.from_list("ev", ["#fde3a7", "#f6a04d", "#d9582b", "#8f2414"])
    nrm = Normalize(0, 32)
    for fl in (False, True):                                  # flagged stations carry a thick dark ring
        m = (g.flag == fl).values
        cx.scatter(xg[m], yg[m], s=30 if fl else 26, c=cmap(nrm(g.n_events.values[m])),
                   ec=S.INK if fl else "white", lw=1.3 if fl else 0.5, zorder=4 + fl)
    cx.text(0.03, 0.97, f"ρ = {rec[2]:.2f}\nHolm p = {rec[4]:.2f}", transform=cx.transAxes, fontsize=6.4,
            va="top", ha="left", color=S.INK, linespacing=1.2)
    cx.set_xlim(0, 33); cx.set_ylim(0, 50)
    cx.set_xlabel("Recession timescale $\\tau_r$ (d)"); cx.set_ylabel("Median drought duration (d)")
    S.hgrid(cx)
    cb_ax = fig.add_axes([0.958, 0.16, 0.008, 0.36])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap=cmap), cax=cb_ax)
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=6, length=1.5, width=0.4)
    cb.set_ticks([0, 10, 20, 30])
    cb_ax.set_title("events", fontsize=6, pad=3, loc="left")
    letter(fig, 0.69, 0.95, "c", "Recession timescale")
    S.save(fig, "FigS19_landscape_power")


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:] or ["s9", "s17", "s18"]:
        globals()[f]()
    print("done")
