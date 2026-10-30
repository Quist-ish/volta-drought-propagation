#!/usr/bin/env python3
"""
Fig. 2. Methodological framework, redrawn as vector artwork on the authors' layout (static/fig02_frame_user.png).

Geometry follows the authors' frame: the Stage 1 data column with its down arrows, the dotted stage outlines
(blue, red, purple, orange), the title and stage boxes, the card rows and the long arrows into Stage 5. The
redraw keeps that design and makes it sharp and consistent:
  * every element is vector (PDF), exported at 600 dpi (PNG) and 1000 dpi (TIFF),
  * card rows are evenly spaced, and every card has the same stack of three sheets fanning up and to the right,
    tinted with its stage colour,
  * each card holds one picture that explains the step and one line of text (redundant labels and equation
    numbers removed; Results sections given as §),
  * Stage 5 panels are drawn from the derived tables (rainfall change point, wavelet coherence of drought
    synchrony with the AMO, gauged network and matched pairs, monitoring horizon),
  * the Drought index card shows a real SSI drought event and the Climate indices card the observed AMO and ATL3;
    the other pictures in Stages 1-4 are schematic.
Text is at least 6 pt at 190 mm width. The dotted stage outlines are the authors' design and are drawn as rows
of square dots; no plotted line is dashed or dotted.
Author: I. Quist (NUIST)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import (Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon,
                                Rectangle)
from matplotlib.path import Path as MPath
import pubstyle as S
from pubviz import tint

S.setup()
mpl.rcParams.update({"font.family": ["Carlito", "Liberation Sans", "DejaVu Sans"],
                     "mathtext.fontset": "custom", "mathtext.rm": "Carlito", "mathtext.it": "Carlito:italic",
                     "mathtext.bf": "Carlito:bold", "mathtext.default": "regular"})
H = Path(__file__).resolve().parent
R = H.parent
TC, FDc = R / "output_climate/tables", R / "output_climate/figdata"
TP, TO = R / "output_projection/tables", R / "output_observational/tables"

# ------------------------------------------------------------------ canvas: authors' frame is 945 x 714 px
PW, PH = 945.0, 714.0
WMM = 190.0
SC = WMM / PW
HMM = PH * SC
fig = plt.figure(figsize=(WMM / 25.4, HMM / 25.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, WMM); ax.set_ylim(0, HMM); ax.axis("off")
REN = fig.canvas.get_renderer()
DPI = fig.dpi

INK, INK2, MUTED, NAVY = "#141414", "#3a3a3a", "#6f6f6f", S.NAVY
BLUE, LBLUE, ORANGE, EBRO, REAN = "#2f6db5", "#9cc3ea", S.VOLTA, S.EBRO, S.REAN
FRAME = {1: "#f2b705", 2: "#2e5fb8", 3: "#e3261f", 4: "#6a2ca0", 5: "#c55a11"}
CARD = {1: "#7fa7d9", 2: "#2e5fb8", 3: "#d2453f", 4: "#7b4bb0", 5: "#c55a11"}
TAB_FC, STG_FC, COL_FC = "#ebeef2", "#e7e6e6", "#f2f2f2"
SMALL = []


def X(px):
    return px * SC


def Y(py):
    return HMM - py * SC


def T(x, y, s, fs, maxw=None, weight="normal", color=INK2, ha="center", va="center", min_fs=6.0, z=9, **kw):
    t = ax.text(x, y, s, fontsize=fs, fontweight=weight, color=color, ha=ha, va=va, zorder=z, **kw)
    if maxw:
        w = t.get_window_extent(REN).width / DPI * 25.4
        if w > maxw:
            t.set_fontsize(max(min_fs, fs * maxw / w))
            w2 = t.get_window_extent(REN).width / DPI * 25.4
            if w2 > maxw + 0.05:
                SMALL.append((s, round(w2 - maxw, 2)))
    return t


def box(x0, y0, x1, y1, fc, ec, lw=0.9, r=0.0, z=3):
    """box in authors' pixel coordinates (y down)"""
    style = f"round,pad=0,rounding_size={r}" if r else "square,pad=0"
    ax.add_patch(FancyBboxPatch((X(x0), Y(y1)), X(x1) - X(x0), Y(y0) - Y(y1), boxstyle=style, fc=fc, ec=ec, lw=lw,
                                zorder=z))


def dotted(x0, y0, x1, y1, color, z=1):
    """the authors' dotted outline, drawn as square dots (0.5 mm, 1.3 mm pitch)"""
    d, p = 0.55, 1.3
    xs0, xs1, ys0, ys1 = X(x0), X(x1), Y(y1), Y(y0)
    pts = []
    for xx in np.arange(xs0, xs1 + 1e-6, p):
        pts += [(xx, ys0), (xx, ys1)]
    for yy in np.arange(ys0, ys1 + 1e-6, p):
        pts += [(xs0, yy), (xs1, yy)]
    ax.add_collection(PatchCollection([Rectangle((a - d / 2, b - d / 2), d, d) for a, b in pts], fc=color, lw=0,
                                      zorder=z))


def arrow(p0, p1, z=6, tw=3.0, hw=7.2, hl=5.0, col="black"):
    ax.add_patch(FancyArrowPatch((X(p0[0]), Y(p0[1])), (X(p1[0]), Y(p1[1])),
                                 arrowstyle=f"simple,head_width={hw},head_length={hl},tail_width={tw}",
                                 mutation_scale=1, fc=col, ec="none", zorder=z, shrinkA=0, shrinkB=0))


def elbow(pts_px, r_px, tip_px, lw=3.6, z=6):
    """thick polyline with one rounded corner, ending in a triangular head at tip_px (authors' long arrows)"""
    (xa, ya), (xb, yb), (xc, yc) = pts_px            # start, corner, end of line
    sx = np.sign(xc - xb)
    v = [(X(xa), Y(ya)), (X(xb), Y(yb - r_px)), (X(xb), Y(yb)), (X(xb + sx * r_px), Y(yb)), (X(xc), Y(yc))]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO]
    ax.add_patch(PathPatch(MPath(v, codes), fill=False, lw=lw, ec="black", capstyle="butt", joinstyle="round",
                           zorder=z))
    hx, hy = X(tip_px[0]), Y(tip_px[1])
    L, Wd = 3.4, 4.6
    ax.add_patch(Polygon([[hx, hy], [hx - sx * L, hy + Wd / 2], [hx - sx * L, hy - Wd / 2]], fc="black", ec="none",
                         zorder=z))


def card_stack(x0, y0, x1, y1, stage, face_ec=None, z=3):
    c = FRAME[stage] if stage > 1 else CARD[1]
    for k, off in enumerate((9.0, 6.0, 3.0)):          # three sheets fanning up and to the right
        box(x0 + off, y0 - off, x1 + off, y1 - off, tint(c, 0.86 - 0.05 * k), tint(c, 0.35), lw=0.45, r=0.7,
            z=z + k * 0.1)
    box(x0, y0, x1, y1, "white", face_ec or CARD[stage], lw=0.8, r=0.7, z=z + 0.5)


# ------------------------------------------------------------------ pictures (mm; cx, cy centre, w, h size)
def ln(xs, ys, **kw):
    kw.setdefault("lw", 0.55); kw.setdefault("zorder", 7); kw.setdefault("solid_capstyle", "round")
    ax.plot(xs, ys, **kw)


def p_hydro(cx, cy, w, h):
    t = np.linspace(0, 3, 400)
    q = 0.08 + 0.85 * np.exp(-((np.mod(t, 1) - 0.62) / 0.11) ** 2) * (0.75 + 0.25 * np.sin(5 * t))
    gap = ((t > 0.95) & (t < 1.35)) | ((t > 2.3) & (t < 2.48))
    xs, ys = cx - w / 2 + t / 3 * w, cy - h / 2 + q * h
    ax.fill_between(xs, cy - h / 2, ys, where=~gap, color="#d6e6f6", lw=0, zorder=6)
    ys = np.where(gap, np.nan, ys)
    ln(xs, ys, color=BLUE, lw=0.6)
    ln([cx - w / 2, cx + w / 2], [cy - h / 2] * 2, color=INK2, lw=0.3)


def p_tree(cx, cy, w, h, dense=False):
    rng = np.random.default_rng(4)
    o = (cx + 0.35 * w, cy - 0.38 * h)
    trunk = [(cx - 0.45 * w, cy + 0.35 * h), (cx - 0.15 * w, cy + 0.05 * h), (cx + 0.12 * w, cy - 0.15 * h), o]
    ln([p[0] for p in trunk], [p[1] for p in trunk], color=LBLUE if not dense else tint(REAN, 0.3), lw=1.1)
    br = [((cx - 0.3 * w, cy + 0.45 * h), trunk[1]), ((cx + 0.02 * w, cy + 0.42 * h), trunk[2]),
          ((cx + 0.3 * w, cy + 0.3 * h), (cx + 0.24 * w, cy - 0.25 * h))]
    for a, b in br:
        ln([a[0], b[0]], [a[1], b[1]], color=LBLUE if not dense else tint(REAN, 0.3), lw=0.8)
    if dense:
        pts = [trunk[0], trunk[1], trunk[2], o] + [a for a, _ in br]
        for a, b in zip(trunk[:-1], trunk[1:]):
            for f in (0.33, 0.66):
                pts.append((a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])))
        for a, b in br:
            pts.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
        for x, y in pts:
            ax.add_patch(Circle((x, y), 0.32, fc=REAN, ec="white", lw=0.25, zorder=8))
    else:
        for x, y in [trunk[0], br[0][0], br[1][0], br[2][0], trunk[2]]:
            ax.add_patch(Circle((x, y), 0.42, fc="white", ec=NAVY, lw=0.4, zorder=8))
        ax.add_patch(Circle(o, 0.55, fc=NAVY, ec=NAVY, lw=0.4, zorder=8))


def p_grid(cx, cy, w, h):
    nx, ny = 12, 6
    xs = np.linspace(cx - w / 2, cx + w / 2, nx + 1); ys = np.linspace(cy - h / 2, cy + h / 2, ny + 1)
    z = np.linspace(0.85, 0.15, ny)[:, None] + 0.08 * np.sin(np.arange(nx))[None, :]
    yy, xx = np.meshgrid(np.linspace(-1, 1, ny), np.linspace(-1, 1, nx), indexing="ij")
    inside = (xx / 1.05) ** 2 + (yy / 1.15) ** 2 < 1.0
    for i in range(ny):
        for j in range(nx):
            if inside[i, j]:
                ax.add_patch(Rectangle((xs[j], ys[i]), xs[1] - xs[0], ys[1] - ys[0], fc=S.BLUES(float(z[i, j])),
                                       ec="white", lw=0.2, zorder=7))


def p_indices(cx, cy, w, h):
    ci = pd.read_csv(TC / "climate_indices_monthly.csv", index_col=0, parse_dates=True).loc["1952":"2020"]
    atl = ci["ATL3"][ci.index.month.isin([6, 7, 8])].groupby(ci.index.year[ci.index.month.isin([6, 7, 8])]).mean()
    atl = (atl - atl.mean()) / atl.std()
    bw = w / len(atl)
    for i, v in enumerate(atl.values):
        ax.add_patch(Rectangle((cx - w / 2 + i * bw, cy), bw * 0.8, np.clip(v, -2.4, 2.4) / 2.4 * h / 2,
                               fc="#8fb3dd", ec="none", zorder=6))
    amo = ci["AMO"].rolling(132, center=True, min_periods=60).mean()
    amo = (amo - amo.mean()) / amo.std()
    t = np.linspace(0, 1, len(amo))
    ln(cx - w / 2 + t * w, cy + np.clip(amo.values, -2.4, 2.4) / 2.4 * h / 2, color=ORANGE, lw=0.8)
    ln([cx - w / 2, cx + w / 2], [cy, cy], color=INK2, lw=0.25)


def p_spag(cx, cy, w, h):
    rng = np.random.default_rng(11)
    t = np.linspace(0, 1, 80)
    hist = cy - 0.05 * h + 0.06 * h * np.sin(14 * t)
    for k in range(16):
        sl = rng.normal(0.08, 0.45)
        y = np.where(t < 0.5, hist + rng.normal(0, 0.03 * h), hist + sl * h * (t - 0.5) * 1.6)
        ln(cx - w / 2 + t * w, np.clip(y, cy - h / 2, cy + h / 2), color=("#3a7dc9" if k % 2 else "#b8452a"),
           lw=0.3)
    ln(cx - w / 2 + t[t < 0.5] * w, hist[t < 0.5], color=INK, lw=0.6)


def p_raster(cx, cy, w, h, col=EBRO):
    rng = np.random.default_rng(6)
    for i in range(5):
        y = cy + h / 2 - (i + 0.5) * h / 5
        for _ in range(4):
            s0 = rng.uniform(0, 0.9); ln_ = rng.uniform(0.03, 0.1)
            ax.add_patch(Rectangle((cx - w / 2 + s0 * w, y - 0.32 * h / 5), ln_ * w, 0.64 * h / 5, fc=col, ec="none",
                                   zorder=7))
        ln([cx - w / 2, cx + w / 2], [y, y], color="#d5dbe2", lw=0.25, zorder=6)


def p_avail(cx, cy, w, h):
    rows = [[(0, 0.42), (0.47, 1.0)], [(0.1, 0.55), (0.58, 0.8)], [(0.0, 0.25), (0.6, 1.0)], [(0.3, 0.7), (0.73, 1.0)]]
    fills = [(0.42, 0.47), (0.55, 0.58), (0.7, 0.73)]
    for i, segs in enumerate(rows):
        y = cy + h / 2 - (i + 0.5) * h / 4
        for a, b in segs:
            ax.add_patch(Rectangle((cx - w / 2 + a * w, y - 0.3 * h / 4), (b - a) * w, 0.6 * h / 4, fc="#9fb4c9",
                                   ec="none", zorder=7))
    for (a, b), i in zip(fills, (0, 1, 3)):
        y = cy + h / 2 - (i + 0.5) * h / 4
        ax.add_patch(Rectangle((cx - w / 2 + a * w, y - 0.3 * h / 4), (b - a) * w, 0.6 * h / 4, fc="#2e9b5e",
                               ec="none", zorder=8))


def p_ssi(cx, cy, w, h):
    ev = pd.read_csv(TO / "SSI_drought_events.csv", parse_dates=["start_date", "end_date"])
    sd = pd.read_csv(TO / "SSI_daily.csv", usecols=["node_id", "date", "SSI"], parse_dates=["date"])
    best = None
    for _, e in ev[(ev.duration >= 25) & (ev.duration <= 60)].iterrows():
        win = sd[(sd.node_id == e.node_id) & (sd.date >= e.start_date - pd.Timedelta(days=80)) &
                 (sd.date <= e.end_date + pd.Timedelta(days=80))]
        if win.SSI.notna().mean() > 0.98 and (best is None or e.severity > best[0]):
            best = (e.severity, win)
    v = best[1].SSI.values
    t = np.linspace(0, 1, len(v))
    lo, hi = -3.0, 2.0
    yv = cy - h / 2 + (np.clip(v, lo, hi) - lo) / (hi - lo) * h
    thr = cy - h / 2 + (-1.28 - lo) / (hi - lo) * h
    ax.fill_between(cx - w / 2 + t * w, thr, yv, where=yv < thr, color=S.SEV["severe"], lw=0, zorder=6,
                    interpolate=True)
    ln(cx - w / 2 + t * w, yv, color=NAVY, lw=0.5)
    ln([cx - w / 2, cx + w / 2], [thr, thr], color=ORANGE, lw=0.45)


def p_spi(cx, cy, w, h):
    rng = np.random.default_rng(9)
    v = np.convolve(rng.normal(0, 1, 40), np.ones(3) / 3, mode="same") * 1.6
    bw = w / len(v)
    for i, a in enumerate(v):
        ax.add_patch(Rectangle((cx - w / 2 + i * bw, cy), bw * 0.85, np.clip(a, -1.6, 1.6) / 1.6 * h / 2,
                               fc=ORANGE if a <= -1 else "#8fb3dd", ec="none", zorder=7))
    ln([cx - w / 2, cx + w / 2], [cy, cy], color=INK2, lw=0.25)


def p_recess(cx, cy, w, h):
    t = np.linspace(0, 1, 120)
    q = 0.1 + 0.85 * np.exp(-6 * (t - 0.15)) * (t > 0.15) + 0.95 * (t <= 0.15) * (t / 0.15) ** 2
    bf = np.minimum(q, 0.1 + 0.32 * np.exp(-2.2 * np.clip(t - 0.15, 0, None)))
    xs = cx - w / 2 + t * w
    ax.fill_between(xs, cy - h / 2, cy - h / 2 + bf * h, color="#cfe5d8", lw=0, zorder=6)
    ln(xs, cy - h / 2 + q * h, color="#2e8b57", lw=0.6)
    ln([cx - w / 2, cx + w / 2], [cy - h / 2] * 2, color=INK2, lw=0.25)


def p_qdm(cx, cy, w, h):
    x = np.linspace(-1, 1, 80)
    xs = cx - w / 2 + (x + 1) / 2 * w
    cm = cy - h / 2 + h / (1 + np.exp(-5 * (x + 0.25)))
    co = cy - h / 2 + h / (1 + np.exp(-5 * (x - 0.2)))
    ln(xs, cm, color="#9aa5b1", lw=0.6); ln(xs, co, color=BLUE, lw=0.6)
    for q in (0.3, 0.55, 0.8):
        xm = -0.25 - np.log(1 / q - 1) / 5; xo = 0.2 - np.log(1 / q - 1) / 5
        y = cy - h / 2 + q * h
        ax.add_patch(FancyArrowPatch((cx - w / 2 + (xm + 1) / 2 * w, y), (cx - w / 2 + (xo + 1) / 2 * w, y),
                                     arrowstyle="-|>", mutation_scale=3.2, lw=0.35, color=INK2, zorder=8,
                                     shrinkA=0, shrinkB=0))


def p_wave(cx, cy, w, h):
    rng = np.random.default_rng(2)
    x = np.cumsum(rng.normal(0, 1, 96)) * 0.15 + rng.normal(0, 1, 96)
    t = np.linspace(0, 1, 96)
    comps = [x, np.convolve(x, np.ones(4) / 4, mode="full")[:96], np.convolve(x, np.ones(16) / 16, mode="full")[:96]]
    for i, v in enumerate(comps):
        v = (v - v.mean()) / (v.std() + 1e-9)
        ln(cx - w / 2 + t * w, cy + h * (0.34 - 0.34 * i) + 0.11 * h * np.clip(v, -2, 2),
           color=[INK2, "#2e8b57", ORANGE][i], lw=0.45)


def p_link(cx, cy, w, h):
    t = np.linspace(0, 1, 60)
    xs = cx - 0.42 * w + t * 0.84 * w
    ys = cy + 0.25 * h - 0.5 * h * t + 0.12 * h * np.sin(2 * np.pi * t)
    ln(xs, ys, color=LBLUE, lw=2.0)
    ax.add_patch(FancyArrowPatch((xs[30], ys[30]), (xs[36], ys[36]), arrowstyle="-|>", mutation_scale=5, lw=0.6,
                                 color=BLUE, zorder=8, shrinkA=0, shrinkB=0))
    ax.add_patch(Circle((xs[4], ys[4]), 0.7, fc="white", ec=NAVY, lw=0.6, zorder=9))
    ax.add_patch(Circle((xs[-5], ys[-5]), 0.85, fc=NAVY, ec=NAVY, lw=0.6, zorder=9))
    ax.text(xs[4] - 1.3, ys[4], "u", ha="right", va="center", fontsize=6.5, style="italic", color=INK, zorder=9)
    ax.text(xs[-5] + 1.4, ys[-5], "o", ha="left", va="center", fontsize=6.5, style="italic", color=INK, zorder=9)


def p_observe(cx, cy, w, h):
    L, Rr = cx - w / 2, cx + w / 2
    s0 = L + 0.48 * w
    ax.add_patch(Rectangle((s0, cy + 0.2 * h), 0.36 * w, 0.2 * h, fc=NAVY, ec="none", zorder=7))
    days = np.linspace(L + 0.03 * w, Rr - 0.03 * w, 34)
    keep = np.r_[np.zeros(8), np.ones(10), np.zeros(4), np.ones(12)].astype(bool)
    for d_, k_ in zip(days, keep):
        ln([d_, d_], [cy - 0.22 * h, cy + 0.06 * h], color=EBRO if k_ else "#cfd4da", lw=0.6, solid_capstyle="butt")
    ln([s0, s0], [cy - 0.5 * h, cy + 0.5 * h], color=INK, lw=0.45)
    a, b = L + 0.16 * w, s0 + 0.36 * w
    ln([a, b], [cy - 0.4 * h] * 2, color=INK2, lw=0.4)
    for xe in (a, b):
        ln([xe, xe], [cy - 0.48 * h, cy - 0.32 * h], color=INK2, lw=0.4)


def p_match(cx, cy, w, h):
    L = cx - w / 2
    s0 = L + 0.45 * w
    ax.add_patch(Rectangle((L + 0.12 * w, cy + 0.06 * h), 0.52 * w, 0.24 * h, fc=EBRO, ec="none", zorder=7))
    ax.add_patch(Rectangle((s0, cy + 0.06 * h), 0.19 * w, 0.24 * h, fc="#184f95", ec="none", zorder=8))
    ax.add_patch(Rectangle((s0, cy - 0.34 * h), 0.42 * w, 0.24 * h, fc=NAVY, ec="none", zorder=7))
    ln([s0, s0], [cy - 0.5 * h, cy + 0.5 * h], color=INK, lw=0.45)
    ln([s0, s0 + 0.19 * w], [cy + 0.42 * h] * 2, color="#184f95", lw=0.45)
    ax.text(s0 + 0.21 * w, cy + 0.42 * h, "≥ M", ha="left", va="center", fontsize=6, color="#184f95", zorder=9)


def p_pk(cx, cy, w, h):
    k = np.linspace(0, 8, 80)
    ln(cx - w / 2 + k / 8 * w, cy - h / 2 + (1 - (1 - 0.164) ** k) * h, color=EBRO, lw=1.3)
    ln(cx - w / 2 + k / 8 * w, cy - h / 2 + (1 - (1 - 0.162) ** k) * h, color=ORANGE, lw=0.6)
    for kk, f in ((1, 0.22), (2, 0.33), (3, 0.45)):
        ax.add_patch(Circle((cx - w / 2 + kk / 8 * w, cy - h / 2 + f * h), 0.38, fc=ORANGE, ec="white", lw=0.25,
                            zorder=9))
    ln([cx - w / 2, cx + w / 2], [cy - h / 2] * 2, color=INK2, lw=0.3)
    ln([cx - w / 2] * 2, [cy - h / 2, cy + h / 2], color=INK2, lw=0.3)


def p_step(cx, cy, w, h):
    rng = np.random.default_rng(5)
    v = np.r_[rng.normal(0.68, 0.09, 10), rng.normal(0.44, 0.1, 22)]
    bw = w / len(v)
    for i, a in enumerate(v):
        ax.add_patch(Rectangle((cx - w / 2 + i * bw, cy - h / 2), bw * 0.85, a * h, fc="#c9d6e6" if i < 10 else "#ecc9b8",
                               ec="none", zorder=6))
    ln([cx - w / 2, cx - w / 2 + 10 * bw], [cy - h / 2 + 0.68 * h] * 2, color=EBRO, lw=0.7)
    ln([cx - w / 2 + 10 * bw, cx + w / 2], [cy - h / 2 + 0.44 * h] * 2, color=ORANGE, lw=0.7)
    ln([cx - w / 2 + 10 * bw] * 2, [cy - h / 2, cy + h / 2], color=INK2, lw=0.4)


def p_null(cx, cy, w, h):
    x = np.linspace(-2.8, 2.8, 17)
    v = np.exp(-x ** 2 / 2); bw = w / len(x)
    for i, a in enumerate(v):
        ax.add_patch(Rectangle((cx - w / 2 + i * bw, cy - h / 2), bw * 0.88, a * h * 0.92, fc="#c8ced6", ec="none",
                               zorder=6))
    ln([cx + 0.33 * w] * 2, [cy - h / 2, cy + h / 2], color=ORANGE, lw=0.9)


def p_wtc(cx, cy, w, h):
    xx, yy = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 16))
    z = np.exp(-((xx - 0.6) / 0.2) ** 2 - ((yy - 0.62) / 0.22) ** 2) + 0.4 * np.exp(-((xx - 0.22) / 0.1) ** 2 - ((yy - 0.3) / 0.18) ** 2)
    ax.imshow(z, extent=[cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2], origin="lower", cmap="pub_blues",
              aspect="auto", zorder=6, vmin=0, vmax=1.1, interpolation="bilinear")
    ax.contour(np.linspace(cx - w / 2, cx + w / 2, 50), np.linspace(cy - h / 2, cy + h / 2, 16), z, [0.6],
               colors=INK, linewidths=0.35, linestyles="solid", zorder=7)


def p_equiv(cx, cy, w, h):
    lg = lambda v: cx - w / 2 + (np.log(v) - np.log(0.25)) / (np.log(4) - np.log(0.25)) * w
    ax.add_patch(Rectangle((lg(1 / 3), cy - h / 2), lg(3) - lg(1 / 3), h, fc="#e4e8ee", ec="none", zorder=6))
    ax.add_patch(Rectangle((lg(0.5), cy - h / 2), lg(2) - lg(0.5), h, fc="#c9d1dc", ec="none", zorder=6))
    ln([lg(1)] * 2, [cy - h / 2, cy + h / 2], color=INK2, lw=0.35)
    ln([lg(0.45), lg(1.69)], [cy, cy], color=ORANGE, lw=1.0, solid_capstyle="butt")
    ax.add_patch(Circle((lg(0.87), cy), 0.42, fc=ORANGE, ec="white", lw=0.3, zorder=8))


def p_fan(cx, cy, w, h):
    t = np.linspace(0, 1, 60)
    mid = cy - 0.08 * h + 0.07 * h * np.sin(7 * t)
    sp = 0.05 * h + 0.55 * h * np.clip(t - 0.4, 0, None)
    ax.fill_between(cx - w / 2 + t * w, np.maximum(mid - sp, cy - h / 2), np.minimum(mid + 1.2 * sp, cy + h / 2),
                    color="#ecc9bb", lw=0, zorder=6)
    ln(cx - w / 2 + t * w, mid, color="#b8452a", lw=0.7)


def p_osse(cx, cy, w, h):
    vals = [(0.638, REAN), (0.257, "#86cfb2"), (0.213, "#86cfb2"), (0.096, ORANGE)]
    bw = w / len(vals)
    for i, (v, c) in enumerate(vals):
        ax.add_patch(Rectangle((cx - w / 2 + i * bw + 0.12 * bw, cy - h / 2), bw * 0.76, v / 0.65 * h, fc=c, ec="none",
                               zorder=7))
    ln([cx - w / 2, cx + w / 2], [cy - h / 2] * 2, color=INK2, lw=0.3)


def p_accum(cx, cy, w, h):
    ln([cx - w / 2, cx + w / 2], [cy + 0.3 * h] * 2, color=INK2, lw=0.35)
    for sl, c in ((3.0, REAN), (1.0, NAVY), (0.18, ORANGE)):
        x_end = min(1.0, 0.8 / sl)
        ln([cx - w / 2, cx - w / 2 + x_end * w], [cy - h / 2, cy - h / 2 + min(0.8, sl * x_end) * h * 1.0], color=c,
           lw=0.7)
    ln([cx - w / 2] * 2, [cy - h / 2, cy + h / 2], color=INK2, lw=0.3)


def p_rho(cx, cy, w, h):
    rng = np.random.default_rng(3)
    ax.add_patch(Rectangle((cx - w / 2, cy - 0.33 * h), w, 0.66 * h, fc="#eceef1", ec="none", zorder=6))
    xs = np.linspace(-0.42, 0.42, 6)
    ys = np.array([0.05, -0.12, 0.16, -0.16, 0.02, 0.41]) / 1.0
    for x, y in zip(xs, ys):
        ax.add_patch(Circle((cx + x * w, cy + y / 0.5 * 0.33 * h), 0.36, fc=ORANGE, ec="white", lw=0.25, zorder=8))
    ln([cx - w / 2, cx + w / 2], [cy, cy], color=INK2, lw=0.3)


# ------------------------------------------------------------------ card content: title, picture, one line
def fill(x0, y0, x1, y1, title, line, pic, added=False, tfs=6.6, fs=6.2, pic_h=None, two=False, tag=None,
         pic_gap=False):
    xa, xb, yt, yb = X(x0), X(x1), Y(y0), Y(y1)
    w, xc = xb - xa, (xa + xb) / 2
    mw = w - 1.0
    ttl = title + (" ‡" if added else "")
    T(xc, yt - 1.55, ttl, tfs, mw, weight="bold", color=INK)
    lines = line if isinstance(line, (list, tuple)) else [line]
    lsp = 2.2
    y_text_top = yb + 0.75 + len(lines) * lsp
    for i, s in enumerate(lines):
        T(xc, y_text_top - lsp / 2 - i * lsp, s, fs, mw)
    top, bot = yt - (3.6 if pic_gap else 3.15), y_text_top + 0.35
    if pic:
        ph = min(pic_h or 99, top - bot)
        pic(xc, (top + bot) / 2, w - 3.2, ph)


# ================================================================== frame (authors' geometry, px)
dotted(34, 42, 927.5, 155, FRAME[2])
dotted(32, 199.5, 832.5, 329, FRAME[3])
dotted(33.5, 373, 790.5, 517, FRAME[4])
dotted(167.5, 562.5, 749.5, 706, FRAME[5])
box(64, 68, 177, 590, COL_FC, FRAME[1], lw=1.1, z=2)
box(12.5, 211.5, 54, 410, TAB_FC, INK, lw=1.0, z=3)
xm, ym = (X(12.5) + X(54)) / 2, (Y(211.5) + Y(410)) / 2
ax.text(xm - 1.6, ym, "Data", rotation=90, ha="center", va="center", fontsize=8.6, fontweight="bold", color=INK, zorder=9)
ax.text(xm + 1.8, ym, "Observations and references", rotation=90, ha="center", va="center", fontsize=8.6,
        fontweight="bold", color=INK, zorder=9)
TITLES = [((190.5, 9, 400, 50), "Processing", "Indices, signatures and features"),
          ((184, 167.5, 395, 211), "Conceptual core", "Three conditions for a drought link"),
          ((184, 340, 460, 382.5), "Inference", "Mechanisms, robustness and identifiability"),
          ((189, 530, 465, 571.5), "Outputs", "Key results for objectives (i)–(iv)")]
for (a, b, c, d), l1, l2 in TITLES:
    box(a, b, c, d, TAB_FC, INK, lw=1.0, z=8)
    xc, yc = (X(a) + X(c)) / 2, (Y(b) + Y(d)) / 2
    T(xc, yc + 1.75, l1, 8.6, X(c) - X(a) - 2, weight="bold", color=INK, z=10)
    T(xc, yc - 1.8, l2, 8.6, X(c) - X(a) - 2, weight="bold", color=INK, z=10)
for n, (a, b, c, d) in {1: (81, 49, 157.5, 77.5), 2: (831, 17, 907.5, 46), 3: (732.5, 178.5, 807.5, 206.5),
                        4: (704, 350, 776, 379), 5: (655, 539, 728, 567.5)}.items():
    box(a, b, c, d, STG_FC, INK, lw=1.0, z=8)
    T((X(a) + X(c)) / 2, (Y(b) + Y(d)) / 2, f"Stage {n}", 9.2, X(c) - X(a) - 1.5, weight="bold", color=INK, z=10)

# ------------------------------------------------------------------ Stage 1: data cards and down arrows
H1, G1, Y1 = 55.9, 17.0, 81.0
data = [("GRDC discharge", "18 gauges, 1961–2007", p_hydro),
        ("River network", "GRDC + TDX-Hydro", p_tree),
        ("Precipitation", "GPCC v2022, 0.5°", p_grid),
        ("Climate indices", "six SST indices", p_indices),
        ("CMIP6", "16 models, 2 SSPs", p_spag),
        ("GEOGLOWS v2", "routed ERA5 runoff", lambda cx, cy, w, h: p_tree(cx, cy, w, h, dense=True)),
        ("Ebro reference", "33 gauges, 101 links", p_raster)]
CX0, CX1 = 69, 172
for i, (t, l_, pic) in enumerate(data):
    y0 = Y1 + i * (H1 + G1)
    box(CX0, y0, CX1, y0 + H1, "white", CARD[1], lw=0.8, r=0.7, z=4)
    fill(CX0, y0, CX1, y0 + H1, t, l_, pic, tfs=6.6, fs=6.1)
    if i < len(data) - 1:
        arrow(((CX0 + CX1) / 2, y0 + H1 + 0.8), ((CX0 + CX1) / 2, y0 + H1 + G1 - 0.8), tw=2.6, hw=6.4, hl=4.4)

# ------------------------------------------------------------------ rows of cards (even spacing, same stack)
def row(x0, w, g, y0, y1, n):
    return [(x0 + i * (w + g), y0, x0 + i * (w + g) + w, y1) for i in range(n)]


R2 = row(197, 88.5, 26, 80, 136, 6)
R3 = row(230, 113, 32, 232, 309, 4)
R4 = row(229, 107, 32, 404, 505, 4)
R5 = row(200, 105.75, 30, 593, 694, 4)
YC = {2: 108, 3: 270.5, 4: 454.5, 5: 643.5}

for st, rr in ((2, R2), (3, R3), (4, R4), (5, R5)):
    for b in rr:
        card_stack(*b, stage=st)
    for a, b in zip(rr[:-1], rr[1:]):
        arrow((a[2] + 9 + 1.2, YC[st]), (b[0] - 1.2, YC[st]))
for st in (2, 3, 4):
    arrow((177.8, YC[st]), ({2: R2, 3: R3, 4: R4}[st][0][0] - 1.6, YC[st]))
arrow((634, 331.5), (634, 371.5))
arrow((633, 519.5), (633, 561.5))
elbow([(866, 157), (866, 643.5), (760, 643.5)], 33, (751.5, 643.5))
elbow([(111, 591), (111, 643.5), (160, 643.5)], 18, (166, 643.5))

# Stage 2: processing
S2 = [("Data quality", "gaps ≤ 15 d filled", p_avail, False),
      ("Drought index", "SSI below −1.28", p_ssi, False),
      ("SPI anomalies", r"SPI-6, SPI-12, $S_t$", p_spi, True),
      ("Runoff/storage", "runoff at fixed rain", p_recess, True),
      ("Bias adjustment", "QDM vs GPCC", p_qdm, True),
      ("Wavelet features", "2–64 months", p_wave, True)]
for b, (t, l_, pic, add) in zip(R2, S2):
    fill(*b, t, l_, pic, added=add, tfs=6.5, fs=6.1)

# Stage 3: conceptual core
S3 = [("1  Mapped connection", ["u drains to o;", "terrain-checked"], p_link, False),
      ("2  Concurrent record", ["≥ 5 SSI days in [s₀−W, e₀]", "k ≥ 1 gauges: testable"], p_observe, True),
      ("3  Accepted match", ["s₀ − W ≤ sᵤ ≤ s₀", "overlap ≥ 5 d; W = 45 d"], p_match, False),
      ("Connectivity measures", [r"$P(\mathrm{link}\,|\,k)=1-(1-p)^{k}$",
                                 r"$cf_{\mathrm{opp}},\ \hat{p};\ \mathrm{OR}_{\mathrm{MH}}$ vs Ebro"], p_pk, True)]
for b, (t, l_, pic, add) in zip(R3, S3):
    fill(*b, t, l_, pic, added=add, tfs=6.7, fs=6.2, pic_gap=True)

# Stage 4: inference, two sections per card
S4 = [[("Trend, change point", "Pettitt, Sen slope · §3.1", p_step),
       ("Permutation nulls", "availability-aware · §3.3", p_null)],
      [("Teleconnections", "coherence, rates · §3.2", p_wtc),
       ("Equivalence, power", "TOST, power · §3.4", p_equiv)],
      [("Projection ensemble", "16 GCMs × 3 ML · §3.7", p_fan),
       ("Observing system", "gauge sampling · §3.5", p_osse)],
      [("Monitoring horizon", "years to a test · §3.8", p_accum),
       ("Landscape, storage", "Spearman ρ · §3.6", p_rho)]]
for (x0, y0, x1, y1), secs in zip(R4, S4):
    ym_ = (y0 + y1) / 2
    ln([X(x0) + 0.3, X(x1) - 0.3], [Y(ym_)] * 2, color=tint(CARD[4], 0.45), lw=0.5, zorder=6)
    for (a, b), (t, l_, pic) in zip(((y0, ym_), (ym_, y1)), secs):
        fill(x0, a, x1, b, t, l_, pic, added=True, tfs=6.4, fs=6.0)

# ------------------------------------------------------------------ Stage 5: results
def panel_frame(b, title, note):
    xa, xb, yt, yb = X(b[0]), X(b[2]), Y(b[1]), Y(b[3])
    T((xa + xb) / 2, yt - 1.6, title, 6.6, xb - xa - 1.2, weight="bold", color=INK)
    T((xa + xb) / 2, yb + 1.35, note, 6.0, xb - xa - 1.2, color=INK2)
    return xa, xb, yt - 3.3, yb + 2.7


def mini(xa, xb, ya, yb):
    a = fig.add_axes([xa / WMM, ya / HMM, (xb - xa) / WMM, (yb - ya) / HMM])
    a.tick_params(labelsize=6, length=1.3, pad=0.8, width=0.35)
    for s_ in a.spines.values():
        s_.set_linewidth(0.35)
    a.set_facecolor("white")
    return a


# (i) basin rainfall and its 1969 change point
xa, xb, yt, yb = panel_frame(R5[0], "(i) Dated regime shift", "1969: rainfall −10.5%")
a = mini(xa + 5.0, xb - 0.9, yb + 2.3, yt - 0.3)
P = pd.read_csv(TC / "precip_catchments_basin.csv", index_col=0, parse_dates=True)
ann = P["basin"].resample("YS").sum(); ann.index = ann.index.year
tr = pd.read_csv(TC / "trend_changepoint_summary.csv").set_index("series").loc["Basin annual precipitation"]
cp = int(tr.pettitt_year)
a.bar(ann.index, ann.values, width=0.85, color=np.where(ann.index <= cp, "#c9d6e6", "#ecc9b8"), lw=0)
a.hlines(tr.mean_before, ann.index.min() - 0.5, cp + 0.5, color=EBRO, lw=0.9)
a.hlines(tr.mean_after, cp + 0.5, ann.index.max() + 0.5, color=ORANGE, lw=0.9)
a.axvline(cp + 0.5, color=INK2, lw=0.5)
a.set_ylim(650, 1560); a.set_xlim(1950, 2021); a.set_yticks([800, 1200]); a.set_xticks([1960, 2000])
a.text(0.97, 0.98, "mm yr$^{-1}$", transform=a.transAxes, ha="right", va="top", fontsize=6, color=INK2)

# (ii) wavelet coherence of drought synchrony with the AMO
xa, xb, yt, yb = panel_frame(R5[1], "(ii) Synchrony modes", "AMO, 8–32 yr")
a = mini(xa + 5.4, xb - 0.9, yb + 2.3, yt - 0.3)
z = np.load(FDc / "wct_sync_AMO.npz")
t, per, Wc, coi, sig = z["t"], z["period"], z["WCT"], z["coi"], z["sig"]
a.pcolormesh(t, per, Wc, cmap="pub_blues", vmin=0, vmax=1, shading="auto", rasterized=True)
a.contour(t, per, Wc / sig[:, None], [1], colors=INK, linewidths=0.45, linestyles="solid")
a.fill_between(t, coi, 64, color="white", alpha=0.62, lw=0)
a.set_yscale("log", base=2); a.set_ylim(32, 1); a.set_yticks([2, 8, 32]); a.set_yticklabels(["2", "8", "32"])
a.minorticks_off(); a.set_xlim(t.min(), t.max()); a.set_xticks([1960, 2000])
a.set_yticklabels(["2 yr", "8", "32"])

# (iii) gauged network and matched pairs
xa, xb, yt, yb = panel_frame(R5[2], "(iii) Gauged network", "10 of 40 matched")
st = pd.read_csv(R / "data/grdc_stations_raw.csv").set_index("grdc_no")
conn = pd.read_csv(TO / "upstream_connectivity.csv", sep=";")
pairs = pd.read_csv(TO / "propagation_pairs.csv")
basin = gpd.read_file(R / "data/GRDC_VOLTA_POLITICAL_TRANSBOUNDARY_BASIN.shp").to_crs(4326)
minx, miny, maxx, maxy = basin.total_bounds
asp = (maxy - miny) / ((maxx - minx) * np.cos(np.radians(10.5)))
mh = 12.4
mw_ = mh / asp
a = mini(xa + 0.6, xa + 0.6 + mw_, (yt + yb) / 2 - mh / 2 - 0.2, (yt + yb) / 2 + mh / 2 - 0.2)
basin.boundary.plot(ax=a, color=INK, lw=0.45, zorder=1)
reach = pd.read_parquet(R / "data/reanalysis/reaches_volta.parquet")
rr_ = reach[reach.area_km2 >= 20000].set_index("LINKNO")
for lk, r in rr_.iterrows():
    if r.DSLINKNO in rr_.index:
        d = rr_.loc[r.DSLINKNO]
        a.plot([r.lon, d.lon], [r.lat, d.lat], color=LBLUE, lw=0.5, zorder=1, solid_capstyle="round")
for _, r in conn.iterrows():
    if pd.notna(r.immediate_downstream):
        u, o = int(r.node_id), int(r.immediate_downstream)
        if u in st.index and o in st.index:
            a.plot([st.lon[u], st.lon[o]], [st.lat[u], st.lat[o]], color=NAVY, lw=0.45, zorder=2)
for (o, u) in set(zip(pairs.origin_node, pairs.upstream_node)):
    a.plot([st.lon[u], st.lon[o]], [st.lat[u], st.lat[o]], color=ORANGE, lw=1.2, zorder=3, solid_capstyle="round")
origins = set(pairs.origin_node)
for n in sorted(set(conn.node_id)):
    if n in origins:
        a.scatter(st.lon[n], st.lat[n], s=5, marker="o", fc=ORANGE, ec=INK, lw=0.3, zorder=5)
    else:
        a.scatter(st.lon[n], st.lat[n], s=3.2, marker="^", fc="white", ec=INK, lw=0.3, zorder=4)
a.set_xlim(minx - 0.1, maxx + 0.1); a.set_ylim(miny - 0.1, maxy + 0.1)
a.set_aspect(1 / np.cos(np.radians(10.5))); a.axis("off")
lx = xa + 0.6 + mw_ + 0.5
for i, (kind, lab) in enumerate((("link", "link"), ("pair", "matched"), ("origin", "origin"))):
    yy = yt - 1.2 - i * 2.3
    if kind in ("link", "pair"):
        ln([lx, lx + 1.6], [yy, yy], color=NAVY if kind == "link" else ORANGE, lw=0.5 if kind == "link" else 1.2,
           zorder=9)
    else:
        ax.scatter(lx + 0.8, yy, s=5, marker="o", fc=ORANGE, ec=INK, lw=0.3, zorder=9)
    T(lx + 2.0, yy, lab, 6.0, xb - lx - 2.3, ha="left", color=INK2)

# (iv) monitoring horizon
xa, xb, yt, yb = panel_frame(R5[3], "(iv) Risk, test horizon", "years to 50% power")
a = mini(xa + 0.9, xb - 0.9, yb + 2.3, yt - 0.1)
mon = json.load(open(TP / "monitoring_horizon.json"))["designs"]["18 gauges, continuous records"]["years_n50"]
rows_ = [("historical", mon["historical"], None, None, NAVY),
         ("SSP2-4.5", mon["ssp245_2051_2080_median"], mon["ssp245_2051_2080_p10"], mon["ssp245_2051_2080_p90"], "#3a7dc9"),
         ("SSP5-8.5", mon["ssp585_2051_2080_median"], mon["ssp585_2051_2080_p10"], mon["ssp585_2051_2080_p90"], "#b8452a")]
for k_, (lab, m_, lo, hi, col) in enumerate(rows_):
    yk = 2.8 - 1.4 * k_
    if lo is not None:
        a.plot([lo, hi], [yk, yk], color=tint(col, 0.35), lw=1.8, solid_capstyle="butt", zorder=2)
    a.scatter(m_, yk, s=9, color=col, ec="white", lw=0.4, zorder=3)
    a.text(6.2, yk + 0.25, f"{lab}: {m_:.0f} yr", ha="left", va="bottom", fontsize=6, color=INK, zorder=4)
a.set_xscale("log"); a.set_xlim(5, 6000); a.set_ylim(-0.35, 3.8); a.set_yticks([])
a.set_xticks([10, 100, 1000]); a.set_xticklabels(["10", "100", "1,000"]); a.minorticks_off()
a.spines["left"].set_visible(False)

if SMALL:
    print("text wider than its box at 6 pt:", SMALL)
S.save(fig, "Fig02_framework_redraw")
print("saved")
