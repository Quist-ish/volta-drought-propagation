#!/usr/bin/env python3
"""
Fig. 2. Methodological framework (data -> processing -> conceptual core -> inference -> outputs of objectives i-iv).

Drawn on a millimetre canvas (190 x 200 mm). The conceptual core (tier 3) is the three-condition test of a drought
link; every box carries the main-text equation numbers (S, supplementary) in brackets, a dagger-free double-dagger marks steps added
to Crespillo Lopez et al. (2026), and tier-4 boxes carry the Results section they feed.
Glyphs are schematic and carry no data. Author: I. Quist (NUIST)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Polygon, Rectangle
import pubstyle as S

S.setup()
WMM, HMM = 190, 214
fig = plt.figure(figsize=(WMM * S.MM, HMM * S.MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, WMM); ax.set_ylim(0, HMM); ax.set_aspect("equal"); ax.axis("off")

TIER = {  # bar colour, tint
    1: ("#3b6ea8", "#eef4fb"), 2: ("#3c8a66", "#edf6f0"), 3: ("#c25a2c", "#fdf1ea"),
    4: ("#5f53a0", "#f2f0fa"), 5: ("#4d4d4d", "#f3f3f1")}
ADD = S.VOLTA          # outline for steps added in this study
INK, INK2, NAVY = S.INK, S.INK2, S.NAVY
X0, X1 = 13.5, 187.5   # inner content area


def rbox(x, y, w, h, fc, ec, lw=0.6, r=1.4, ls="-", z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                ls=ls, zorder=z))


def tier(n, y0, y1, label):
    bar, tint = TIER[n]
    rbox(2.0, y0, 8.6, y1 - y0, bar, bar, r=1.6, z=1)
    ax.text(6.3, (y0 + y1) / 2, f"{n}. {label}", rotation=90, ha="center", va="center", color="white",
            fontsize=7, fontweight="bold", zorder=3)
    rbox(11.8, y0, 177.4, y1 - y0, tint, bar, lw=0.5, r=1.6, z=0)


def down(y_from, y_to, x=100.0):
    ax.add_patch(FancyArrowPatch((x, y_from), (x, y_to), arrowstyle="simple,head_width=5.2,head_length=3.4,tail_width=2.2",
                                 mutation_scale=1, fc="#9aa3ad", ec="none", zorder=1))


def right(x0, x1, y, col=INK2):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=col, zorder=3))


def card(x, y, w, h, title, lines, glyph=None, added=False, eq=None, tag=None, gh=7.0, title_size=6.6, title_dx=0.0):
    rbox(x, y, w, h, "white", ADD if added else "#9aa3ad", lw=0.9 if added else 0.6)
    t = title + (" ‡" if added else "")
    ax.text(x + w / 2 + title_dx, y + h - 2.6, t, ha="center", va="center", fontsize=title_size, fontweight="bold", color=INK,
            zorder=4)
    yy = y + h - 5.0
    if eq:
        ax.text(x + w / 2, yy - 0.2, f"[{eq}]", ha="center", va="center", fontsize=6, color=S.MUTED, zorder=4)
        yy -= 2.6
    if glyph:
        glyph(x + w / 2, yy - gh / 2 - 0.4, w - 6, gh)
        yy -= gh + 1.2
    for i, s in enumerate(lines):
        ax.text(x + w / 2, yy - 1.4 - 2.55 * i, s, ha="center", va="center", fontsize=6, color=INK2, zorder=4)
    if tag:
        ax.text(x + w - 1.3, y + 1.2, tag, ha="right", va="bottom", fontsize=6, color=INK2, fontweight="bold",
                zorder=4)


# ------------------------------------------------------------------ glyphs (schematic, centre cx, cy, size w x h)
def g_hydro(cx, cy, w, h):
    x = np.linspace(-w / 2, w / 2, 80)
    y = 0.35 * h * np.exp(-((x + w * 0.18) / (w * 0.12)) ** 2) + 0.18 * h * np.exp(-((x - w * 0.2) / (w * 0.1)) ** 2)
    ax.fill_between(cx + x, cy - h / 2, cy - h / 2 + 0.15 * h + y, color="#cfe0f3", lw=0, zorder=3)
    ax.plot(cx + x, cy - h / 2 + 0.15 * h + y, color=TIER[1][0], lw=0.8, zorder=4)
    ax.add_patch(Polygon([[cx + w * 0.38, cy + h * 0.45], [cx + w * 0.30, cy + h * 0.2], [cx + w * 0.46, cy + h * 0.2]],
                         fc=INK2, ec="none", zorder=4))


def g_net(cx, cy, w, h):
    pts = {"a": (-0.38, 0.38), "b": (0.0, 0.42), "c": (0.36, 0.25), "m": (-0.05, 0.0), "o": (0.05, -0.42)}
    P = {k: (cx + v[0] * w * 0.9, cy + v[1] * h) for k, v in pts.items()}
    for a, b in (("a", "m"), ("b", "m"), ("c", "o"), ("m", "o")):
        ax.plot([P[a][0], P[b][0]], [P[a][1], P[b][1]], color="#8fb8e2", lw=1.6, solid_capstyle="round", zorder=3)
    for k, (x, y) in P.items():
        ax.add_patch(Circle((x, y), 0.85 if k != "o" else 1.05, fc=NAVY if k == "o" else "white", ec=NAVY, lw=0.6,
                            zorder=4))


def g_rain(cx, cy, w, h):
    for dx, dy, rx in ((-0.18, 0.2, 0.32), (0.1, 0.28, 0.36), (0.3, 0.16, 0.26)):
        ax.add_patch(Ellipse((cx + dx * w, cy + dy * h), rx * w, 0.45 * h, fc="#d7dde5", ec="none", zorder=3))
    for i, dx in enumerate((-0.25, -0.05, 0.15, 0.35)):
        ax.plot([cx + dx * w, cx + dx * w - 0.6], [cy - 0.05 * h, cy - 0.38 * h], color=TIER[1][0], lw=0.8, zorder=4)


def g_grid(cx, cy, w, h):
    for f in np.linspace(-0.45, 0.45, 5):
        ax.plot([cx + f * w, cx + f * w], [cy - h / 2, cy + h / 2], color="#d5dbe2", lw=0.4, zorder=3)
        ax.plot([cx - w / 2, cx + w / 2], [cy + f * h, cy + f * h], color="#d5dbe2", lw=0.4, zorder=3)
    t = np.linspace(0, 1, 40)
    ax.plot(cx - w / 2 + t * w, cy + h * (0.35 - 0.7 * t + 0.12 * np.sin(9 * t)), color=S.REAN, lw=1.3, zorder=4)


def g_catalogue(cx, cy, w, h):
    for i, (s, l) in enumerate(((0.05, 0.25), (0.3, 0.2), (0.12, 0.35))):
        ax.add_patch(Rectangle((cx - w / 2 + s * w, cy + h * (0.28 - 0.3 * i) - 0.7), l * w, 1.4, fc=S.EBRO, ec="none",
                               zorder=4))
    ax.plot([cx - w / 2, cx + w / 2], [cy - h / 2 + 0.3, cy - h / 2 + 0.3], color=INK2, lw=0.5, zorder=4)


def g_layers(cx, cy, w, h):
    for i, c in enumerate(("#c9d8c3", "#e8d6b5", "#cfd9e6")):
        y = cy - h * 0.3 + i * h * 0.28
        ax.add_patch(Polygon([[cx - w * 0.36, y], [cx + w * 0.2, y], [cx + w * 0.36, y + h * 0.22],
                              [cx - w * 0.2, y + h * 0.22]], fc=c, ec="#8c96a0", lw=0.4, zorder=3 + i))


def g_gaps(cx, cy, w, h):
    segs = [(0, 0.28), (0.36, 0.62), (0.66, 1.0)]
    for a, b in segs:
        ax.add_patch(Rectangle((cx - w / 2 + a * w, cy - 0.9), (b - a) * w, 1.8, fc="#9fb4c9", ec="none", zorder=3))
    ax.add_patch(Rectangle((cx - w / 2 + 0.62 * w, cy - 0.9), 0.04 * w, 1.8, fc=TIER[2][0], ec="none", zorder=4))
    ax.text(cx - w / 2 + 0.32 * w, cy + 2.0, "gap > 15 d", ha="center", va="bottom", fontsize=6, color=INK2, zorder=4)
    ax.text(cx - w / 2 + 0.64 * w, cy - 2.0, "≤ 15 d", ha="center", va="top", fontsize=6, color=TIER[2][0], zorder=4)


def g_ssi(cx, cy, w, h):
    x = np.linspace(-1, 1, 60)
    ax.plot(cx - w * 0.42 + (x + 1) / 2 * w * 0.36, cy - h / 2 + h * (1 / (1 + np.exp(-4 * x))), color=TIER[2][0],
            lw=0.9, zorder=4)
    xx = np.linspace(-2.5, 2.5, 60)
    ax.plot(cx + w * 0.06 + (xx + 2.5) / 5 * w * 0.36, cy - h / 2 + h * np.exp(-xx ** 2 / 2), color=NAVY, lw=0.9,
            zorder=4)
    right(cx - w * 0.04, cx + w * 0.07, cy)


def g_event(cx, cy, w, h):
    t = np.linspace(0, 1, 120)
    y = 0.15 - 0.55 * np.exp(-((t - 0.5) / 0.18) ** 2) + 0.08 * np.sin(18 * t)
    yy = cy + y * h
    thr = cy - 0.12 * h
    ax.fill_between(cx - w / 2 + t * w, thr, yy, where=yy < thr, color=S.SEV["moderate"], lw=0, zorder=3)
    ax.plot(cx - w / 2 + t * w, yy, color=NAVY, lw=0.8, zorder=4)
    ax.plot([cx - w / 2, cx + w / 2], [thr, thr], color=INK2, lw=0.5, zorder=4)
    ax.text(cx - w / 2, thr - 0.4, "−1.28", ha="left", va="top", fontsize=6, color=INK2, zorder=4)


def g_bars(cx, cy, w, h):
    v = [0.5, 0.8, -0.4, 0.3, -0.9, -0.6, 0.2, -0.3]
    bw = w / len(v)
    for i, a in enumerate(v):
        ax.add_patch(Rectangle((cx - w / 2 + i * bw + 0.15, cy), bw - 0.3, a * h * 0.48,
                               fc=S.VOLTA if a <= -0.5 else "#c4c9cf", ec="none", zorder=4))
    ax.plot([cx - w / 2, cx + w / 2], [cy, cy], color=INK2, lw=0.4, zorder=4)


def g_recess(cx, cy, w, h):
    t = np.linspace(0, 1, 80)
    q = 0.12 + 0.85 * np.exp(-5 * t) * (t > 0.08) + 0.85 * (t <= 0.08) * t / 0.08
    b = np.minimum(q, 0.12 + 0.3 * np.exp(-1.6 * t))
    ax.fill_between(cx - w / 2 + t * w, cy - h / 2, cy - h / 2 + b * h, color="#cfe5d8", lw=0, zorder=3)
    ax.plot(cx - w / 2 + t * w, cy - h / 2 + q * h, color=TIER[2][0], lw=0.9, zorder=4)


def g_link(cx, cy, w, h):
    ax.plot([cx - w * 0.3, cx + w * 0.3], [cy + h * 0.25, cy - h * 0.25], color="#8fb8e2", lw=2.2,
            solid_capstyle="round", zorder=3)
    ax.add_patch(Circle((cx - w * 0.3, cy + h * 0.25), 1.2, fc="white", ec=NAVY, lw=0.7, zorder=4))
    ax.add_patch(Circle((cx + w * 0.3, cy - h * 0.25), 1.35, fc=NAVY, ec=NAVY, lw=0.7, zorder=4))
    ax.text(cx - w * 0.3 - 2.0, cy + h * 0.25, "u", ha="right", va="center", fontsize=6.5, style="italic", zorder=4)
    ax.text(cx + w * 0.3 + 2.0, cy - h * 0.25, "o", ha="left", va="center", fontsize=6.5, style="italic", zorder=4)


def g_observe(cx, cy, w, h):
    L, R = cx - w / 2, cx + w / 2
    s0 = L + 0.5 * w
    ax.add_patch(Rectangle((s0, cy + 1.2), 0.4 * w, 1.5, fc=NAVY, ec="none", zorder=4))
    days = np.linspace(L + 0.05 * w, R - 0.05 * w, 26)
    keep = np.r_[np.zeros(6), np.ones(8), np.zeros(3), np.ones(9)].astype(bool)
    for d, k in zip(days, keep):
        ax.plot([d, d], [cy - 1.9, cy - 0.5], color=S.EBRO if k else "#d0d4d9", lw=0.9, zorder=4)
    ax.annotate("", xy=(L + 0.08 * w, cy - 2.9), xytext=(R - 0.06 * w, cy - 2.9),
                arrowprops=dict(arrowstyle="|-|,widthA=0.2,widthB=0.2", lw=0.5, color=INK2))
    ax.plot([s0, s0], [cy - 3.3, cy + 3.2], color=INK, lw=0.5, zorder=4)


def g_match(cx, cy, w, h):
    L = cx - w / 2
    s0 = L + 0.48 * w
    ax.add_patch(Rectangle((s0, cy - 2.1), 0.42 * w, 1.6, fc=NAVY, ec="none", zorder=4))
    ax.add_patch(Rectangle((L + 0.18 * w, cy + 0.6), 0.55 * w, 1.6, fc=S.EBRO, ec="none", zorder=4))
    ax.add_patch(Rectangle((s0, cy + 0.6), 0.25 * w, 1.6, fc="#184f95", ec="none", zorder=5))
    ax.plot([s0, s0], [cy - 3.0, cy + 3.4], color=INK, lw=0.5, zorder=4)
    ax.annotate("", xy=(s0, cy + 3.2), xytext=(s0 + 0.25 * w, cy + 3.2),
                arrowprops=dict(arrowstyle="|-|,widthA=0.2,widthB=0.2", lw=0.5, color="#184f95"))
    ax.text(s0 + 0.27 * w, cy + 3.2, "≥ M", ha="left", va="center", fontsize=6, color="#184f95", zorder=5)


def g_pk(cx, cy, w, h):
    k = np.linspace(0, 6, 60)
    for p, c in ((0.16, S.VOLTA), (0.35, "#c4c9cf")):
        ax.plot(cx - w / 2 + k / 6 * w, cy - h / 2 + (1 - (1 - p) ** k) * h, color=c, lw=1.0, zorder=4)
    ax.plot([cx - w / 2, cx + w / 2], [cy - h / 2, cy - h / 2], color=INK2, lw=0.4, zorder=4)
    ax.plot([cx - w / 2, cx - w / 2], [cy - h / 2, cy + h / 2], color=INK2, lw=0.4, zorder=4)
    ax.text(cx + w / 2, cy - h / 2 - 0.3, "k", ha="right", va="top", fontsize=6, style="italic", color=INK2, zorder=4)


def g_hist(cx, cy, w, h):
    x = np.linspace(-2.6, 2.6, 13)
    v = np.exp(-x ** 2 / 2)
    bw = w / len(x)
    for i, a in enumerate(v):
        ax.add_patch(Rectangle((cx - w / 2 + i * bw, cy - h / 2), bw * 0.9, a * h * 0.9, fc="#cdd3da", ec="none",
                               zorder=3))
    ax.plot([cx + 0.33 * w] * 2, [cy - h / 2, cy + h / 2], color=S.VOLTA, lw=1.1, zorder=4)


def g_strata(cx, cy, w, h):
    for i, (a, b) in enumerate(((0.22, 0.23), (0.33, 0.37), (0.33, 0.49))):
        x = cx - w / 2 + 0.08 * w + i * 0.31 * w
        ax.add_patch(Rectangle((x, cy - h / 2), 0.11 * w, a * h * 1.6, fc=S.VOLTA, ec="none", zorder=4))
        ax.add_patch(Rectangle((x + 0.12 * w, cy - h / 2), 0.11 * w, b * h * 1.6, fc=S.EBRO, ec="none", zorder=4))
    ax.plot([cx - w / 2, cx + w / 2], [cy - h / 2, cy - h / 2], color=INK2, lw=0.4, zorder=4)


def g_power(cx, cy, w, h):
    x = np.linspace(0, 1, 60)
    ax.plot(cx - w / 2 + x * w, cy - h / 2 + h / (1 + np.exp(-9 * (x - 0.6))), color=NAVY, lw=1.0, zorder=4)
    ax.plot([cx - w / 2, cx + w / 2], [cy, cy], color=INK2, lw=0.4, zorder=4)
    ax.add_patch(Circle((cx - w / 2 + 0.6 * w, cy), 0.75, fc="white", ec=NAVY, lw=0.6, zorder=5))


def g_osse(cx, cy, w, h):
    for i, (v, c) in enumerate(((0.95, S.REAN), (0.8, S.REAN), (0.38, "#7fcbb0"), (0.3, "#7fcbb0"), (0.15, S.VOLTA))):
        ax.add_patch(Rectangle((cx - w / 2 + i * w / 5 + 0.4, cy - h / 2), w / 5 - 0.8, v * h, fc=c, ec="none",
                               zorder=4))
    ax.plot([cx - w / 2, cx + w / 2], [cy - h / 2, cy - h / 2], color=INK2, lw=0.4, zorder=4)


def g_scatter(cx, cy, w, h):
    rng = np.random.default_rng(3)
    x = rng.uniform(-0.45, 0.45, 14); y = 0.2 * x + rng.normal(0, 0.18, 14)
    ax.add_patch(Rectangle((cx - w / 2, cy - h * 0.32), w, h * 0.64, fc="#eceef1", ec="none", zorder=3))
    ax.scatter(cx + x * w, cy + y * h, s=3, color=S.VOLTA, zorder=4, lw=0)


def g_heat(cx, cy, w, h):
    vals = np.array([[0.2, 0.5, 0.8], [0.3, 0.6, 0.9], [0.1, 0.4, 0.7]])
    cw, ch = w * 0.6 / 3, h / 3
    for i in range(3):
        for j in range(3):
            ax.add_patch(Rectangle((cx - w * 0.3 + j * cw, cy - h / 2 + i * ch), cw - 0.3, ch - 0.3,
                                   fc=S.BLUES(vals[i, j]), ec="none", zorder=4))



def g_wave(cx, cy, w, h):
    t = np.linspace(0, 1, 120)
    ax.plot(cx - w / 2 + t * w, cy + 0.28 * h * np.sin(2 * np.pi * 2.2 * t), color=TIER[1][0], lw=0.9, zorder=4)
    ax.plot(cx - w / 2 + t * w, cy - 0.05 * h + 0.18 * h * np.sin(2 * np.pi * 0.7 * t + 1.0), color=S.VOLTA, lw=0.9, zorder=4)


def g_spag(cx, cy, w, h):
    rng = np.random.default_rng(7)
    t = np.linspace(0, 1, 40)
    for k in range(7):
        sl = rng.normal(0.15, 0.35)
        y = cy - 0.1 * h + 0.25 * h * np.sin(9 * t + k) * 0.25 + sl * h * np.clip(t - 0.45, 0, None)
        ax.plot(cx - w / 2 + t * w, y, color=["#3a7dc9", "#b8452a"][k % 2], lw=0.5, alpha=0.9, zorder=4)
    ax.plot([cx - w / 2 + 0.45 * w] * 2, [cy - h / 2, cy + h / 2], color=INK2, lw=0.4, zorder=4)


def g_qdm(cx, cy, w, h):
    x = np.linspace(-1, 1, 60)
    for sh, c in ((-0.25, INK2), (0.15, TIER[2][0])):
        ax.plot(cx - w * 0.4 + (x + 1) / 2 * w * 0.8, cy - h / 2 + h / (1 + np.exp(-5 * (x - sh))), color=c, lw=0.9, zorder=4)
    ax.annotate("", xy=(cx + 0.06 * w, cy), xytext=(cx - 0.12 * w, cy), arrowprops=dict(arrowstyle="-|>", lw=0.6, color=INK2,
                                                                                         mutation_scale=5))


def g_atrous(cx, cy, w, h):
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 64).cumsum() * 0.0 + rng.normal(0, 1, 64)
    t = np.linspace(0, 1, 64)
    for i, win in enumerate((1, 4, 16)):
        v = np.convolve(x, np.ones(win) / win, mode="full")[:64]
        v = (v - v.mean()) / (v.std() + 1e-9)
        ax.plot(cx - w / 2 + t * w, cy + h * (0.32 - 0.32 * i) + 0.09 * h * v, color=[S.INK2, TIER[2][0], S.VOLTA][i], lw=0.6,
                zorder=4)


def g_step(cx, cy, w, h):
    rng = np.random.default_rng(5)
    n = 24
    v = np.r_[rng.normal(0.62, 0.1, 8), rng.normal(0.38, 0.1, 16)]
    bw = w / n
    for i, a in enumerate(v):
        ax.add_patch(Rectangle((cx - w / 2 + i * bw + 0.08, cy - h / 2), bw - 0.16, a * h, fc="#c9d6e6" if i < 8 else "#e9c8b8",
                               ec="none", zorder=3))
    ax.plot([cx - w / 2, cx - w / 2 + 8 * bw], [cy - h / 2 + 0.62 * h] * 2, color=S.EBRO, lw=1.0, zorder=4)
    ax.plot([cx - w / 2 + 8 * bw, cx + w / 2], [cy - h / 2 + 0.38 * h] * 2, color=S.VOLTA, lw=1.0, zorder=4)
    ax.plot([cx - w / 2 + 8 * bw] * 2, [cy - h / 2, cy + h / 2], color=INK2, lw=0.5, zorder=4)


def g_wtc(cx, cy, w, h):
    xx, yy = np.meshgrid(np.linspace(0, 1, 30), np.linspace(0, 1, 12))
    z = np.exp(-((xx - 0.55) / 0.25) ** 2 - ((yy - 0.65) / 0.22) ** 2) + 0.3 * np.exp(-((xx - 0.2) / 0.1) ** 2 - ((yy - 0.25) / 0.15) ** 2)
    ax.imshow(z, extent=[cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2], origin="lower", cmap="pub_blues", aspect="auto",
              zorder=3, vmin=0, vmax=1.1)
    ax.contour(np.linspace(cx - w / 2, cx + w / 2, 30), np.linspace(cy - h / 2, cy + h / 2, 12), z, [0.6], colors=INK,
               linewidths=0.5, zorder=4)


def g_fan(cx, cy, w, h):
    t = np.linspace(0, 1, 50)
    mid = cy - 0.15 * h + 0.1 * h * np.sin(6 * t)
    spread = 0.05 * h + 0.42 * h * np.clip(t - 0.4, 0, None)
    ax.fill_between(cx - w / 2 + t * w, mid - spread, mid + spread * 1.4, color="#e3c2b6", lw=0, zorder=3)
    ax.plot(cx - w / 2 + t * w, mid, color="#b8452a", lw=0.9, zorder=4)
    ax.plot([cx - w / 2, cx + w / 2], [cy - 0.15 * h, cy - 0.15 * h], color=INK2, lw=0.4, zorder=4)


def g_horizon(cx, cy, w, h):
    for i, (v, c) in enumerate(((0.95, NAVY), (0.5, NAVY), (0.15, NAVY))):
        ax.add_patch(Rectangle((cx - w / 2, cy + h * (0.28 - 0.3 * i) - 0.7), v * w, 1.4, fc=c, ec="none", zorder=4))
        ax.plot([cx - w / 2 + v * w * 0.6, min(cx + w / 2, cx - w / 2 + v * w * 1.6)], [cy + h * (0.28 - 0.3 * i) - 1.3] * 2,
                color="#b8452a", lw=0.7, zorder=4)

# ------------------------------------------------------------------ tier 1: data
tier(1, 186.0, 212.0, "Data")
G = 2.4
W1 = (X1 - X0 - 6 * G) / 7
data = [("GRDC discharge", ["daily, 18 gauges", "1961–2007"], g_hydro),
        ("River network", ["GRDC pointers; TDX-", "Hydro; HydroATLAS"], g_net),
        ("Precipitation", ["GPCC v2022, 0.5°;", "Sahel index"], g_rain),
        ("Climate indices", ["AMO, Niño 3.4, ATL3,", "TNA−TSA, DMI, PDO"], g_wave),
        ("CMIP6", ["16 models; pr, tas,", "mrro; SSP2-4.5, 5-8.5"], g_spag),
        ("GEOGLOWS v2", ["ERA5 runoff routed;", "1950–2025"], g_grid),
        ("Ebro reference", ["33 gauges; event", "catalogue, 101 links"], g_catalogue)]
for i, (t, ls, g) in enumerate(data):
    card(X0 + i * (W1 + G), 188.0, W1, 22.0, t, ls, glyph=g, gh=6.5, title_size=6.3)
down(185.6, 181.4)

# ------------------------------------------------------------------ tier 2: processing
tier(2, 151.0, 181.0, "Processing")
W2 = (X1 - X0 - 5 * G) / 6
proc = [("Quality control", ["published screen", "reported; tier-1 gap", "fill only"], g_gaps, False, None, None),
        ("Drought index, events", ["SSI: ±15-d pools, six", "families, zero-flow mass;", "SSI < −1.28; D, S, I, V"], g_ssi, False, "1–2, S1–S3", None),
        ("Rainfall anomalies", ["catchment SPI-6, SPI-12;", "drought synchrony Sₜ"], g_bars, True, "4", None),
        ("Runoff, storage", ["runoff at fixed rainfall;", "BFI, recession, zero flow"], g_recess, True, "3, S15–S16", None),
        ("Bias adjustment", ["QDM per catchment", "and month vs GPCC"], g_qdm, True, "S18", None),
        ("Wavelet features", ["causal à trous Haar,", "2–64 months"], g_atrous, True, "12", None)]
for i, (t, ls, g, add, eq, tg) in enumerate(proc):
    card(X0 + i * (W2 + G), 153.0, W2, 26.0, t, ls, glyph=g, added=add, eq=eq, gh=6.0, tag=tg, title_size=6.3)
down(150.6, 146.4)

# ------------------------------------------------------------------ tier 3: conceptual core
tier(3, 99.0, 146.0, "Conceptual core")
ax.text(X0, 142.6, "Is the downstream drought linked to an upstream one?  Three conditions, tested in order",
        ha="left", va="center", fontsize=6.8, fontweight="bold", color=TIER[3][0], zorder=4)
CW, CH, CY = 35.0, 29.0, 110.0
cx = [X0, X0 + CW + 5.0, X0 + 2 * (CW + 5.0)]
conds = [("Mapped connection", ["upstream gauge u drains", "to origin gauge o;", "terrain-checked"], g_link, False, None),
         ("Concurrent observation", ["≥ 5 defined SSI days in", "[s₀ − W, e₀]; k observing", "gauges; k ≥ 1 testable"],
          g_observe, True, "7"),
         ("Accepted match", ["s₀ − W ≤ sᵤ ≤ s₀,", "overlap ≥ M, lag ≤ 0;", "W = 45 d, M = 5 d"], g_match, False, "6")]
for i, (t, ls, g, add, eq) in enumerate(conds):
    card(cx[i], CY, CW, CH, t, ls, glyph=g, added=add, eq=eq, gh=7.5, title_dx=1.6)
    col = ADD if add else NAVY
    ax.add_patch(Circle((cx[i] + 1.4, CY + CH - 1.2), 2.2, fc=col, ec="white", lw=0.8, zorder=6))
    ax.text(cx[i] + 1.4, CY + CH - 1.25, str(i + 1), ha="center", va="center", fontsize=6.8, fontweight="bold",
            color="white", zorder=7)
    if i:
        right(cx[i] - 4.6, cx[i] - 0.5, CY + CH / 2)
OX = cx[2] + CW + 5.0
right(OX - 4.6, OX - 0.5, CY + CH / 2)
card(OX, CY, X1 - OX, CH, "Connectivity measures",
     [r"$P(\mathrm{connected}\mid k) = 1-(1-p)^{k}$", r"$c,\ cf_{\mathrm{all}},\ cf_{\mathrm{opp}}$ [8]; per-link $\hat{p}$ [9]",
      r"Volta–Ebro $\mathrm{OR}_{\mathrm{MH}}$ [10], equivalence [S14]"],
     glyph=g_pk, added=True, eq="8–10", gh=7.0)
rbox(cx[1] - 1.0, 101.0, CW + 2.0, 6.2, "white", "#a4abb4", r=1.0)
ax.text(cx[1] + CW / 2, 104.1, "k = 0: untestable, not unconnected", ha="center", va="center", fontsize=6,
        color=INK2, zorder=4)
ax.add_patch(FancyArrowPatch((cx[1] + CW / 2, CY - 0.2), (cx[1] + CW / 2, 107.4), arrowstyle="-|>", mutation_scale=5,
                             lw=0.6, color="#8c96a0", zorder=3))
rbox(cx[2] + 2.0, 101.0, 62.0, 6.2, "white", ADD, r=1.0)
ax.text(cx[2] + 33.0, 104.1, "timing class: relay, nested or near-simultaneous", ha="center", va="center",
        fontsize=6, color=INK2, zorder=4)
ax.add_patch(FancyArrowPatch((cx[2] + CW / 2, CY - 0.2), (cx[2] + CW / 2, 107.4), arrowstyle="-|>", mutation_scale=5,
                             lw=0.6, color=ADD, zorder=3))
down(98.6, 94.4)

# ------------------------------------------------------------------ tier 4: inference
tier(4, 34.0, 94.0, "Inference")
W4 = (X1 - X0 - 3 * G) / 4
rowA = [("Trend and change point", ["Pettitt, SNHT, Buishand;", "Mann–Kendall (Hamed–Rao),", "Sen; runoff at fixed rain"], g_step, True,
         "3, S4–S6", "§3.1"),
        ("Teleconnections", ["Morlet power, wavelet", "coherence and phase;", "event-rate regression"], g_wtc, True, "4–5, S7–S8", "§3.2"),
        ("Projection ensemble", ["16 CMIP6 models × ridge,", "RF, XGBoost; SPI, SRI,", "SSI-12; model agreement"], g_fan, True,
         "12, S18–S19", "§3.7"),
        ("Monitoring horizon", ["years to the opportunities", "for 50% and 80% power,", "three network designs"], g_horizon, True, "11, 13",
         "§3.7")]
rowB = [("Permutation nulls", ["records moved with their", "availability; default and", "common-reference index"], g_hist, True, "S13",
         "§3.3–3.4"),
        ("Equivalence, identifiability", ["strata of equal k, TOST;", "Ebro thinned, degraded;", "synthetic truth"], g_power, True,
         "10–11, S14", "§3.4–3.5"),
        ("Observing system", ["reanalysis sampled as", "gauged: masked, then", "refitted"], g_osse, True, None, "§3.5"),
        ("Landscape, storage", ["Spearman ρ, Holm,", "detectable |ρ|;", "event covariates"], g_scatter, True, "S15–S17", "§3.6")]
for i, (t, ls, g, add, eq, tg) in enumerate(rowA):
    card(X0 + i * (W4 + G), 65.0, W4, 27.0, t, ls, glyph=g, added=add, eq=eq, tag=tg, gh=6.0, title_size=6.4)
for i, (t, ls, g, add, eq, tg) in enumerate(rowB):
    card(X0 + i * (W4 + G), 36.0, W4, 27.0, t, ls, glyph=g, added=add, eq=eq, tag=tg, gh=6.0, title_size=6.4)
down(33.6, 29.4)

# ------------------------------------------------------------------ tier 5: outputs
tier(5, 2.0, 29.0, "Outputs")
W5 = (X1 - X0 - 3 * G) / 4
outs = [("A dated regime shift and", "the index reference"), ("Climate modes that set", "drought synchrony"),
        ("What the network can test,", "and the sample needed"), ("Projected risk and the", "observation to verify it")]
for i, (a, b) in enumerate(outs):
    x = X0 + i * (W5 + G)
    rbox(x, 6.0, W5, 17.0, "white", TIER[5][0], lw=0.6)
    ax.add_patch(Circle((x + 4.6, 14.5), 2.6, fc=TIER[5][0], ec="none", zorder=4))
    ax.text(x + 4.6, 14.45, ("i", "ii", "iii", "iv")[i], ha="center", va="center", fontsize=7, fontweight="bold", color="white",
            zorder=5)
    ax.text(x + 9.0, 16.6, a, ha="left", va="center", fontsize=6.4, color=INK, zorder=4)
    ax.text(x + 9.0, 12.6, b, ha="left", va="center", fontsize=6.4, color=INK, zorder=4)

S.save(fig, "Fig02_framework")
print("saved")
