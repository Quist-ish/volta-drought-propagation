"""
Fig. 11c, observing-system experiment, drawn in the line-profile style of the ALE panels:
two colours (charcoal = connected fraction of origin events, burnt orange = per-link probability),
thick line = observed statistic, thin light line with small diamonds = mean of the availability-aware
permutation null (no dashed lines are used in the paper).
The shaded band between the two lines is the excess over the null; filled
markers show where that excess is significant (one-sided p < 0.05), open markers where it is not.
The five configurations on the right sample one simulation with progressively less observation;
the Ebro gauges are a separate reference and are not joined to the sequence.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pubviz import tint

R = Path(__file__).resolve().parents[1]
CHAR, ORNG, ZERO, TXT = "#333333", "#c46d16", "#a2a2a2", "#1c1c1c"
FONT = ["Liberation Sans", "DejaVu Sans"]   # same family as every other figure


def load():
    ns = pd.read_csv(R / "output_identifiability/tables/networks_summary.csv").set_index("network")
    rr = json.load(open(R / "output_reanalysis/reanalysis_numbers.json"))["main"]
    os_ = pd.read_csv(R / "output_reanalysis/tables/osse_summary.csv")
    cols = ["cf_all", "null_cf_all", "p_cf_all", "p_link", "null_p_link", "p_p_link"]
    rows = [("Ebro\ngauges", ns.loc["Ebro", cols]),
            (f"All {rr['n_nodes']}\nnodes", pd.Series({c: rr[c] for c in cols})),
            ("18 gauge\nnodes", os_.iloc[0][cols]),
            ("Masked to\nobserved\ndays", os_.iloc[1][cols]),
            ("Refitted on\nobserved\ndays", os_.iloc[2][cols]),
            ("Volta\ngauges", ns.loc["Volta observed", cols])]
    return rows


def style():
    plt.rcParams.update({"font.family": FONT, "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
                         "mathtext.it": "Liberation Sans:italic", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": TXT, "axes.linewidth": 0.8, "xtick.color": TXT, "ytick.color": TXT,
                         "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3,
                         "ytick.major.size": 3, "axes.labelcolor": TXT, "text.color": TXT})


def draw(ax, fs=7.0, legend_ax=None, legend_kw=None, title="Observing-system experiment"):
    rows = load()
    x = np.array([0.0, 1.55, 2.55, 3.55, 4.55, 5.55])
    seq = slice(1, 6)
    lw = 1.5
    for key, nkey, pkey, col in (("cf_all", "null_cf_all", "p_cf_all", CHAR), ("p_link", "null_p_link", "p_p_link", ORNG)):
        obs = np.array([float(r[key]) for _, r in rows]); nul = np.array([float(r[nkey]) for _, r in rows])
        pv = np.array([float(r[pkey]) for _, r in rows])
        ax.fill_between(x[seq], nul[seq], obs[seq], color=col, alpha=0.10, lw=0, zorder=1)
        ax.plot(x[seq], obs[seq], color=col, lw=lw, solid_capstyle="round", zorder=3)
        ax.plot(x[seq], nul[seq], color=tint(col, 0.45), lw=0.9, marker="D", ms=2.6, mew=0, zorder=2)
        # Ebro reference: observed marker and a short light tick for its null mean
        ax.plot([x[0] - 0.22, x[0] + 0.22], [nul[0]] * 2, color=tint(col, 0.45), lw=1.2, zorder=2)
        ax.plot(x[0], nul[0], marker="D", ms=2.6, mew=0, color=tint(col, 0.45), zorder=2)
        ax.vlines(x[0], min(obs[0], nul[0]), max(obs[0], nul[0]), color=col, lw=4.5, alpha=0.12, zorder=1)
        for xi, o, p in zip(x, obs, pv):
            ax.plot(xi, o, marker="o", ms=4.0, mew=1.1, mec=col, mfc=col if p < 0.05 else "white", zorder=4)
    ax.axvline(0.78, color="#d4d4d4", lw=0.7, zorder=0)
    ax.text(0.12, 0.875, "reference", ha="center", va="bottom", fontsize=fs - 1.2, color="#6b6b6b")
    ax.text(3.55, 0.875, "reanalysis, observed ever more sparsely  →", ha="center", va="bottom",
            fontsize=fs - 1.2, color="#6b6b6b")
    # the masking step: excess of the connected fraction over its null collapses
    o = rows[2][1]; m = rows[3][1]
    ax.annotate(f"masking to gauge days:\nexcess {o.cf_all - o.null_cf_all:.2f} → {m.cf_all - m.null_cf_all:.2f}",
                xy=(3.25, 0.40), xytext=(3.95, 0.60), fontsize=fs - 1.1, color=CHAR, ha="left", va="center",
                linespacing=1.15, arrowprops=dict(arrowstyle="-|>", mutation_scale=6, lw=0.7, color=CHAR,
                                                  connectionstyle="arc3,rad=0.25"))
    ax.set_xlim(-0.45, 5.95); ax.set_ylim(0, 0.9)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=fs - 1.0, linespacing=1.05)
    ax.tick_params(axis="y", labelsize=fs - 0.3)
    ax.set_yticks(np.arange(0, 0.81, 0.2))
    ax.set_ylabel("Probability", fontsize=fs + 0.3)
    ax.set_xlabel("Network configuration", fontsize=fs + 0.3, labelpad=3)
    if title:
        ax.set_title(title, fontsize=fs + 0.5, fontweight="normal", pad=14, loc="left")
    hd = [Line2D([], [], color=CHAR, lw=lw, label="Connected fraction ← observed"),
          Line2D([], [], color=tint(CHAR, 0.45), lw=0.9, marker="D", ms=2.6, mew=0, label="Connected fraction ← null mean"),
          Line2D([], [], color=ORNG, lw=lw, label="Per-link probability ← observed"),
          Line2D([], [], color=tint(ORNG, 0.45), lw=0.9, marker="D", ms=2.6, mew=0, label="Per-link probability ← null mean"),
          Line2D([], [], color="#555555", marker="o", ls="", ms=4, mew=1.1, label="excess over null, p < 0.05"),
          Line2D([], [], color="#555555", marker="o", ls="", ms=4, mew=1.1, mfc="white", label="excess over null, p ≥ 0.05")]
    kw = dict(loc="upper left", bbox_to_anchor=(-0.02, -0.33), ncols=1, fontsize=fs - 0.4, frameon=False,
              handlelength=2.6, handletextpad=0.7, labelspacing=0.35, borderaxespad=0)
    kw.update(legend_kw or {})
    (legend_ax or ax).legend(handles=hd, **kw)
    return ax
