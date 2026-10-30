#!/usr/bin/env python3
"""Supplementary Fig. S16 (rainfall corroboration) in the shared publication style.

Draws the driest upstream SPI-6 (GPCC) at the origin onset for connected and unconnected
opportunities, on the observed gauges and on the reanalysis network, from the tables written by
rainfall_corroboration.py (output_rainfall/tables). The output keeps its first-draft file name,
FigS14_rainfall; collect_final_figures.py copies it to figures/FigS16_rainfall.pdf.
Figures of earlier drafts that this script used to draw are no longer part of the article.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import pubstyle as S

S.setup()
R = Path(__file__).resolve().parents[1]
TP = R / "output_rainfall/tables"
Q1, GREY = S.NAVY, "#9aa5b1"

fig, axs = plt.subplots(1, 2, figsize=(S.ONEHALF, 62 * S.MM), gridspec_kw=dict(wspace=0.38, left=0.1, right=0.99, bottom=0.17, top=0.9))
rng = np.random.default_rng(1)
for k, (ax, f, lab) in enumerate(zip(axs, ("opportunities_observed.csv", "opportunities_reanalysis.csv"),
                                     ("Observed gauges, 1961–2007", "Reanalysis network, 1961–2020"))):
    o = pd.read_csv(TP / f)
    data = [o.loc[o.connected, "up_spi_min"].dropna().values, o.loc[~o.connected, "up_spi_min"].dropna().values]
    for i, dd in enumerate(data):
        col = Q1 if i == 0 else GREY
        if len(dd) < 300:
            ax.scatter(i + rng.uniform(-0.14, 0.14, len(dd)), dd, s=7, color=col, lw=0, alpha=0.85, zorder=3)
        else:
            vp = ax.violinplot(dd, positions=[i], widths=0.7, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(col); b.set_alpha(0.35); b.set_edgecolor("none")
        q1, q2, q3 = np.percentile(dd, [25, 50, 75])
        ax.add_patch(plt.Rectangle((i + 0.2, q1), 0.1, q3 - q1, fc="white", ec=S.INK, lw=0.6, zorder=4))
        ax.plot([i + 0.2, i + 0.3], [q2, q2], color=S.INK, lw=1.0, zorder=5)
    ax.axhline(-0.5, color=S.VOLTA, lw=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"connected\n(n = {len(data[0]):,})", f"not connected\n(n = {len(data[1]):,})"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("Driest upstream SPI-6 at origin onset"); ax.set_title(lab, fontsize=6.2, loc="left")
    S.hgrid(ax); S.panel(ax, "ab"[k], x=-0.2, y=1.0)
S.save(fig, "FigS14_rainfall")
