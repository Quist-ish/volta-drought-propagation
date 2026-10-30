"""
Publication figures, matching the Ebro figure set one for one.

Figure correspondence with Crespillo Lopez et al. (2026):

  fig01_study_area          Fig. 1  study area map + mean discharge distribution
  fig02_event_example       Fig. 2  single event: daily SSI with severity bands
  fig03_workflow            Fig. 3  conceptual workflow schematic (a) stages,
                                    (b) the same rules drawn on a timeline
  fig04_sensitivity         Fig. 4  (W, M) sensitivity heatmaps, six panels
  fig05_event_summary       Fig. 5  event descriptor boxplots, log axis
  fig06_chain_distributions Fig. 6  chain descriptor histograms, six panels
  fig07_chain_spatiotemporal Figs. 7-9  stacked SSI panel + chain extent map

  fig10_transboundary       NEW, Volta only: country transmission matrix and
                            lead-time gain
  fig11_validation          NEW, Volta only: GloFAS against GRDC

Panel layout, severity bands, colour choices, box styling and the inline
statistics boxes follow the originals so the two papers read as companions.

Author: I. Quist (NUIST)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
PALETTE = {
    "moderate": "#F5C26B",
    "severe": "#E07B39",
    "extreme": "#A4133C",
    "daily_ssi": "#1B3B6F",
    "box": "#A8CCE8",
    "mean": "#D62728",
    "grid": "#D7DCE2",
}
BREAKS = {"moderate": -1.28, "severe": -1.65, "extreme": -2.00}


def set_style(dpi: int = 400) -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": dpi,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.7,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )


def _statbox(ax, values: pd.Series, loc: str = "upper right", fmt: str = "{:.3g}") -> None:
    """The inline Mean/Median/Min/Max/IQR box used throughout the Ebro figures."""
    v = pd.Series(values).dropna()
    if v.empty:
        return
    txt = "\n".join(
        [
            f"Mean   {fmt.format(v.mean())}",
            f"Median {fmt.format(v.median())}",
            f"Min    {fmt.format(v.min())}",
            f"Max    {fmt.format(v.max())}",
            f"IQR    {fmt.format(v.quantile(0.75) - v.quantile(0.25))}",
        ]
    )
    xy = {"upper right": (0.97, 0.97), "upper left": (0.03, 0.97)}[loc]
    ax.text(
        *xy, txt, transform=ax.transAxes, ha="right" if "right" in loc else "left",
        va="top", fontsize=6.4, family="DejaVu Sans Mono",
        bbox=dict(boxstyle="square,pad=0.35", fc="white", ec="#9AA5B1", lw=0.6),
    )


def _save(fig, path_stem: str, formats: Sequence[str] = ("pdf", "png")) -> List[str]:
    out = []
    for ext in formats:
        p = f"{path_stem}.{ext}"
        fig.savefig(p)
        out.append(p)
    return out


# ===========================================================================
# Fig. 1 -- study area and discharge distribution
# ===========================================================================
def fig01_study_area(
    nodes: pd.DataFrame,
    q_df: pd.DataFrame,
    boundary_gdf=None,
    rivers_gdf=None,
    subbasins_gdf=None,
    countries_gdf=None,
    dem=None,
    dem_transform=None,
    reservoirs: Optional[pd.DataFrame] = None,
    path_stem: str = "output/figures/fig01_study_area",
):
    """
    Panel (a): basin map with the river network, the virtual gauging nodes,
    sub-basin boundaries, riparian state borders and the major impoundments,
    over shaded elevation. The Ebro version shaded Koppen-Geiger zones; for the
    Volta the four sub-basins and the national borders carry more information,
    since the transboundary structure is the point of the paper.

    Panel (b): distribution of mean discharge across nodes on a log10 axis,
    identical in construction to the original.
    """
    set_style()
    fig = plt.figure(figsize=(11.0, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.22)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_dist = fig.add_subplot(gs[0, 1])

    # ---- (a) map ----------------------------------------------------------
    if dem is not None and dem_transform is not None:
        import rasterio.plot as rplot

        rplot.show(dem, transform=dem_transform, ax=ax_map, cmap="terrain", alpha=0.75)
    if subbasins_gdf is not None:
        subbasins_gdf.boundary.plot(ax=ax_map, color="#5A6B7B", lw=0.5, ls=(0, (4, 2)))
    if countries_gdf is not None:
        countries_gdf.boundary.plot(ax=ax_map, color="#2B2B2B", lw=0.9)
    if rivers_gdf is not None:
        rivers_gdf.plot(ax=ax_map, color="#2E6F9E", lw=0.45)
    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=ax_map, color="black", lw=1.3)

    ax_map.scatter(
        nodes["lon"], nodes["lat"], s=26, marker="^",
        facecolor="#C0392B", edgecolor="black", lw=0.4, zorder=5, label="Virtual gauging node",
    )
    if "kind" in nodes.columns:
        g = nodes[nodes["kind"] == "gauge"]
        if not g.empty:
            ax_map.scatter(
                g["lon"], g["lat"], s=42, marker="o", facecolor="none",
                edgecolor="#111111", lw=1.0, zorder=6, label="Co-located observed gauge",
            )
    if reservoirs is not None and not reservoirs.empty:
        ax_map.scatter(
            reservoirs["lon"], reservoirs["lat"], s=50, marker="s",
            facecolor="#1B3B6F", edgecolor="white", lw=0.5, zorder=7, label="Major reservoir",
        )

    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.text(0.012, 0.975, "(a)", transform=ax_map.transAxes, va="top",
                fontsize=10, fontweight="bold")
    ax_map.legend(loc="lower left", frameon=True, framealpha=0.92)

    # ---- (b) mean discharge distribution ----------------------------------
    mean_q = q_df.groupby("node_id")["Q"].mean()
    logq = np.log10(mean_q[mean_q > 0])

    ax_dist.hist(logq, bins=14, density=True, color=PALETTE["box"],
                 edgecolor="#3B6E9B", lw=0.6, label="Histogram")
    try:
        from scipy.stats import gaussian_kde

        xs = np.linspace(logq.min() - 0.3, logq.max() + 0.3, 400)
        ax_dist.plot(xs, gaussian_kde(logq)(xs), color=PALETTE["daily_ssi"],
                     lw=1.4, label="KDE (Scott bandwidth)")
    except Exception:
        pass

    med = float(np.median(logq))
    ax_dist.axvline(med, color="#D35400", ls="--", lw=1.2,
                    label=f"Median ({10 ** med:.1f} m$^3$ s$^{{-1}}$)")
    ax_dist.plot(logq, np.full_like(logq, -0.012), "|", color="#3B6E9B", ms=6, mew=0.8)

    ax_dist.set_xlabel("Mean discharge  log$_{10}$ (m$^3$ s$^{-1}$)")
    ax_dist.set_ylabel("Density")
    ax_dist.text(0.012, 0.975, "(b)", transform=ax_dist.transAxes, va="top",
                 fontsize=10, fontweight="bold")
    ax_dist.text(0.97, 0.88, f"n = {len(mean_q)} nodes", transform=ax_dist.transAxes,
                 ha="right", fontsize=8)
    ax_dist.legend(loc="upper left", frameon=True)

    secax = ax_dist.secondary_xaxis("top", functions=(lambda v: v, lambda v: v))
    secax.set_xticks(ax_dist.get_xticks())
    secax.set_xticklabels([f"{10 ** t:g}" for t in ax_dist.get_xticks()], fontsize=7)
    secax.set_xlabel("Mean discharge (m$^3$ s$^{-1}$)", fontsize=8)

    return fig, _save(fig, path_stem)


# ===========================================================================
# Fig. 2 -- single drought event
# ===========================================================================
def fig02_event_example(
    ssi_df: pd.DataFrame,
    event_row: pd.Series,
    node_label: str,
    context_days: int = 20,
    path_stem: str = "output/figures/fig02_event_example",
):
    """Daily SSI through one event with severity bands, plus the SSI histogram."""
    set_style()
    nid = event_row["node_id"]
    s, e = pd.Timestamp(event_row["start_date"]), pd.Timestamp(event_row["end_date"])
    lo, hi = s - pd.Timedelta(days=context_days), e + pd.Timedelta(days=context_days)

    sub = ssi_df[(ssi_df["node_id"] == nid) & ssi_df["date"].between(lo, hi)].sort_values("date")
    inside = sub[sub["date"].between(s, e)]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10.2, 7.4), gridspec_kw={"height_ratios": [1.45, 1.0], "hspace": 0.33}
    )

    ax1.axvspan(lo, s, color="#E8E8E8", alpha=0.65)
    ax1.axvspan(e, hi, color="#E8E8E8", alpha=0.65, label="Context window")
    ax1.plot(sub["date"], sub["SSI"], color=PALETTE["daily_ssi"], lw=1.25, label="Daily SSI")
    ax1.axhline(BREAKS["moderate"], color="black", ls="--", lw=1.0,
                label=f"Threshold (SSI = {BREAKS['moderate']})")
    ax1.axhline(BREAKS["severe"], color=PALETTE["severe"], ls=":", lw=0.9)
    ax1.axhline(BREAKS["extreme"], color=PALETTE["extreme"], ls=":", lw=0.9)

    x, y = inside["date"], inside["SSI"]
    ax1.fill_between(x, y, BREAKS["moderate"], where=y < BREAKS["moderate"],
                     color=PALETTE["moderate"], alpha=0.95, interpolate=True)
    ax1.fill_between(x, y, BREAKS["severe"], where=y < BREAKS["severe"],
                     color=PALETTE["severe"], alpha=0.95, interpolate=True)
    ax1.fill_between(x, y, BREAKS["extreme"], where=y < BREAKS["extreme"],
                     color=PALETTE["extreme"], alpha=0.95, interpolate=True)

    peak = pd.Timestamp(event_row["peak_date"])
    ax1.axvline(peak, color="#6A1B9A", ls="--", lw=1.1,
                label=f"Peak ({peak:%d %b %Y})")

    axq = ax1.twinx()
    axq.plot(sub["date"], sub["Q"], color="#9E9E9E", lw=0.8, zorder=1)
    axq.set_yscale("log")
    axq.set_ylabel("$Q$ (m$^3$ s$^{-1}$)", color="#6E6E6E")
    axq.tick_params(axis="y", colors="#6E6E6E")

    stats_txt = (
        f"Duration : {int(event_row['duration'])} d\n"
        f"Severity : {event_row['severity']:.2f}\n"
        f"Intensity: {event_row['intensity']:.4f} d$^{{-1}}$\n"
        f"Peak SSI : {event_row['peak_SSI']:.2f}\n"
        f"Peak date: {peak:%d %b %Y}\n"
        f"Vol. deficit: {event_row.get('severity_hm3', float('nan')):.1f} hm$^3$"
    )
    ax1.text(0.012, 0.97, stats_txt, transform=ax1.transAxes, va="top", fontsize=7,
             family="DejaVu Sans Mono",
             bbox=dict(boxstyle="square,pad=0.4", fc="white", ec="#9AA5B1", lw=0.6))

    ax1.set_ylabel("SSI (--)")
    ax1.set_title(
        f"{node_label}  ·  Event {int(event_row['event_id'])} "
        f"({s:%d %b %Y} – {e:%d %b %Y})", fontsize=10
    )
    handles = ax1.get_legend_handles_labels()[0] + [
        Patch(fc=PALETTE["moderate"], label="Moderate drought"),
        Patch(fc=PALETTE["severe"], label="Severe drought"),
        Patch(fc=PALETTE["extreme"], label="Extreme drought"),
    ]
    ax1.legend(handles=handles, loc="upper right", ncol=2, framealpha=0.95)
    ax1.text(0.012, 0.055, "(a)", transform=ax1.transAxes, fontsize=10, fontweight="bold")

    # ---- (b) SSI distribution within the event ----------------------------
    vals = inside["SSI"].dropna()
    bins = np.linspace(min(vals.min(), -2.6), max(vals.max(), -0.6), 26)
    colours = []
    for left in bins[:-1]:
        if left < BREAKS["extreme"]:
            colours.append(PALETTE["extreme"])
        elif left < BREAKS["severe"]:
            colours.append(PALETTE["severe"])
        elif left < BREAKS["moderate"]:
            colours.append(PALETTE["moderate"])
        else:
            colours.append("#BBDEFB")
    counts, edges = np.histogram(vals, bins=bins, density=True)
    ax2.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
            color=colours, edgecolor="white", lw=0.4)
    try:
        from scipy.stats import gaussian_kde

        xs = np.linspace(bins[0], bins[-1], 400)
        ax2.plot(xs, gaussian_kde(vals)(xs), color=PALETTE["daily_ssi"], lw=1.4, label="KDE")
    except Exception:
        pass

    for name, c in [("moderate", "black"), ("severe", PALETTE["severe"]),
                    ("extreme", PALETTE["extreme"])]:
        ax2.axvline(BREAKS[name], color=c, ls="--", lw=1.0)
        ax2.text(BREAKS[name], ax2.get_ylim()[1] * 0.96,
                 f"{name.capitalize()} ({BREAKS[name]:.2f})", rotation=90,
                 ha="right", va="top", fontsize=6.4, color=c)

    ax2.set_xlabel("SSI (--)")
    ax2.set_ylabel("Density")
    ax2.legend(
        handles=[
            Patch(fc="#BBDEFB", label="Near normal"),
            Patch(fc=PALETTE["moderate"], label="Moderate drought"),
            Patch(fc=PALETTE["severe"], label="Severe drought"),
            Patch(fc=PALETTE["extreme"], label="Extreme drought"),
            Line2D([], [], color=PALETTE["daily_ssi"], lw=1.4, label="KDE"),
        ],
        loc="upper right", framealpha=0.95,
    )
    ax2.text(0.012, 0.93, "(b)", transform=ax2.transAxes, fontsize=10, fontweight="bold")

    return fig, _save(fig, path_stem)


# ===========================================================================
# Fig. 3 -- conceptual workflow of the propagation algorithm
# ===========================================================================
def fig03_workflow(
    W: int = 45,
    M: int = 5,
    path_stem: str = "output/figures/fig03_workflow",
    formats: Sequence[str] = ("pdf", "png"),
) -> List[str]:
    """
    Conceptual workflow of the drought propagation algorithm.

    Panel (a) is the procedure as a sequence of stages. Panel (b) draws the
    same rules on a timeline, because the asymmetric window and the overlap
    criterion are geometric and a reader grasps them faster from the geometry
    than from prose. Four upstream candidates are shown against one origin
    event: one retained, one rejected for insufficient overlap, one rejected
    for onset after the origin, one rejected for lying outside the window.

    Nothing is fitted. The figure is schematic and takes W and M so the drawn
    geometry always matches the configuration in use.
    """
    set_style()
    fig = plt.figure(figsize=(7.4, 6.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.82, 1.0], hspace=0.42)

    ink, accent, bad, muted = "#1F2933", "#2C6E9B", "#B4472F", "#8C98A4"

    # ---------------------------------------------------------------- (a)
    ax = fig.add_subplot(gs[0]); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.1)

    top = [(1.55, 2.30, "Inputs", "Event catalogue per node\nUpstream connectivity"),
           (5.00, 2.30, "Origin event", "Select a downstream\nevent, onset $s_o$"),
           (8.45, 2.30, "Search window", f"$[s_o - W,\\, s_o]$, asymmetric\nW = {W} d")]
    bot = [(1.22, 0.60, "Candidates", "Upstream events\nmeeting the window"),
           (3.70, 0.60, "Overlap filter", f"Retain overlap $\\geq$ M\nM = {M} d"),
           (6.18, 0.60, "Tie-break", "Overlap, severity,\n|onset diff|, id"),
           (8.66, 0.60, "Chain", "Retained pairs,\ntemporally ordered")]

    def box(x, y, title, body, hw, fc="white", ec=None, lw=0.9):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - hw, y - 0.52), 2 * hw, 1.04,
            boxstyle="round,pad=0.03,rounding_size=0.07",
            fc=fc, ec=ec or accent, lw=lw, zorder=2))
        ax.text(x, y + 0.26, title, ha="center", va="center",
                fontsize=8.2, fontweight="bold", color=ink, zorder=3)
        ax.text(x, y - 0.20, body, ha="center", va="center",
                fontsize=6.5, color=ink, zorder=3, linespacing=1.35)

    for x, y, ti, bo in top:
        box(x, y, ti, bo, 1.42)
    for i, (x, y, ti, bo) in enumerate(bot):
        last = i == len(bot) - 1
        box(x, y, ti, bo, 1.14,
            fc="#EEF3F7" if last else "white",
            ec=ink if last else accent, lw=1.1 if last else 0.9)

    arrow = dict(arrowstyle="-|>", color=muted, lw=0.9,
                 shrinkA=1, shrinkB=1, mutation_scale=9)
    for a, b in (((2.97, 2.30), (3.58, 2.30)), ((6.42, 2.30), (7.03, 2.30)),
                 ((2.36, 0.60), (2.56, 0.60)), ((4.84, 0.60), (5.04, 0.60)),
                 ((7.32, 0.60), (7.52, 0.60))):
        ax.annotate("", xy=b, xytext=a, arrowprops=arrow)
    # descend from the search window to the candidate stage
    # orthogonal return path, routed along the clear band between the rows so
    # it cannot cross the stage boxes
    y_mid = 1.47
    ax.plot([8.45, 8.45], [1.78, y_mid], color=muted, lw=0.9,
            solid_capstyle="round", zorder=1)
    ax.plot([8.45, 1.22], [y_mid, y_mid], color=muted, lw=0.9,
            solid_capstyle="round", zorder=1)
    ax.annotate("", xy=(1.22, 1.12), xytext=(1.22, y_mid), arrowprops=arrow)
    ax.text(-0.012, 1.10, "(a)", transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left")

    # ---------------------------------------------------------------- (b)
    ax2 = fig.add_subplot(gs[1])
    s_o, e_o = 0, 38
    ax2.axvspan(-W, s_o, color=accent, alpha=0.075, zorder=0)
    ax2.axvline(s_o, color=ink, lw=0.9, ls="--", zorder=3)
    ax2.axvline(-W, color=accent, lw=0.9, ls=":", zorder=3)

    rows = [("Origin, node 12",     s_o, e_o, ink,    "origin event"),
            ("Upstream A, node 7",  -26,  24, accent, f"retained, overlap 25 d $\\geq$ M, lag $-$26 d"),
            ("Upstream B, node 9",  -33, -29, bad,    "rejected, overlap 0 d"),
            ("Upstream C, node 4",    6,  44, bad,    "rejected, lag $> 0$"),
            ("Upstream D, node 2",  -72, -58, bad,    "rejected, outside window")]

    names, verdicts, colours = [], [], []
    for i, (lab, s, e, col, tag) in enumerate(rows):
        y = len(rows) - 1 - i
        ax2.barh(y, e - s, left=s, height=0.44, color=col,
                 alpha=0.92 if col == ink else 0.80, zorder=2,
                 edgecolor=col, linewidth=0.8)
        names.append(lab); verdicts.append(tag); colours.append(col)

    # overlap bracket, placed in the gap between the origin and candidate A
    ov_s, ov_e = max(s_o, -26), min(e_o, 24)
    ax2.plot([ov_s, ov_e], [3.44, 3.44], color=accent, lw=1.3, zorder=4)
    for xv in (ov_s, ov_e):
        ax2.plot([xv, xv], [3.37, 3.51], color=accent, lw=1.3, zorder=4)
    ax2.text((ov_s + ov_e) / 2, 3.52, f"overlap = {ov_e - ov_s + 1} d",
             ha="center", va="bottom", fontsize=6.8, color=accent)

    ax2.annotate("", xy=(s_o, -0.80), xytext=(-W, -0.80),
                 arrowprops=dict(arrowstyle="<|-|>", color=accent, lw=0.9,
                                 mutation_scale=8))
    ax2.text(-W / 2, -1.00, f"look-back window, W = {W} d",
             ha="center", va="top", fontsize=7.0, color=accent)
    ax2.text(s_o + 2.0, -0.80, "$s_o$", ha="left", va="center",
             fontsize=8.0, color=ink)

    ax2.set_ylim(-1.55, 4.15)
    ax2.set_xlim(-88, 60)
    ax2.set_yticks(range(len(rows))[::-1])
    ax2.set_yticklabels(names, fontsize=7.4)
    ax2.tick_params(axis="y", length=0, pad=4)
    ax2.set_xlabel("Days relative to origin event onset $s_o$")
    for side in ("top", "right", "left"):
        ax2.spines[side].set_visible(False)

    # verdicts as right-hand tick labels, so nothing can overrun the axes
    ax2r = ax2.twinx()
    ax2r.set_ylim(ax2.get_ylim())
    ax2r.set_yticks(range(len(rows))[::-1])
    ax2r.set_yticklabels(verdicts, fontsize=6.8, style="italic")
    ax2r.tick_params(axis="y", length=0, pad=5)
    for lbl, col in zip(ax2r.get_yticklabels(), colours):
        lbl.set_color(col)
    for side in ("top", "right", "left"):
        ax2r.spines[side].set_visible(False)

    ax2.text(-0.012, 1.10, "(b)", transform=ax2.transAxes, fontsize=10,
             fontweight="bold", va="top", ha="left")

    return _save(fig, path_stem, formats)


# ===========================================================================
# Fig. 4 -- parameter sensitivity heatmaps
# ===========================================================================
def fig04_sensitivity(
    grid: pd.DataFrame,
    default_W: int = 45,
    default_M: int = 5,
    path_stem: str = "output/figures/fig04_sensitivity",
):
    """Six panels over the (W, M) space, matching the original layout."""
    set_style()
    panels = [
        ("n_pairs", "N valid pairs (lag $\\leq$ 0)", "RdYlBu"),
        ("n_origin_matched", "N origin events matched ($\\geq$1 upstream)", "RdYlBu"),
        ("mean_propagation_fraction", "Mean propagation fraction", "viridis"),
        ("median_propagation_fraction", "Median propagation fraction", "viridis"),
        ("median_lag", "Median lag (days)", "RdYlBu"),
        ("mean_chain_size", "Mean chain size", "viridis"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2))
    fig.suptitle("Sensitivity of propagation metrics to the search window W and minimum overlap M", fontsize=12, y=0.98)

    for ax, (col, title, cmap) in zip(axes.ravel(), panels):
        piv = grid.pivot(index="W", columns="M", values=col)
        im = ax.pcolormesh(piv.columns, piv.index, piv.values, cmap=cmap, shading="nearest")
        ax.plot(default_M, default_W, "+", color="white", ms=13, mew=2.2)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Minimum overlap M (days)")
        ax.set_ylabel("Search window W (days)")
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.ax.tick_params(labelsize=7)
        ax.text(0.97, 0.965, f"Default (W={default_W}, M={default_M})",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.2,
                bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="#9AA5B1", lw=0.5))

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, _save(fig, path_stem)


# ===========================================================================
# Fig. 5 -- event descriptor boxplots
# ===========================================================================
def fig05_event_summary(
    events: pd.DataFrame,
    path_stem: str = "output/figures/fig05_event_summary",
):
    """Duration, severity, intensity and volumetric severity on log axes."""
    set_style()
    panels = [
        ("duration", "Duration (days)", "(a)", "{:.1f}"),
        ("severity", "Severity (--)", "(b)", "{:.3g}"),
        ("intensity", "Intensity (day$^{-1}$)", "(c)", "{:.3g}"),
        ("severity_hm3", "Volumetric severity (hm$^3$)", "(d)", "{:.3g}"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 4.1))
    fig.suptitle(
        f"Summary statistics of the detected hydrological drought events "
        f"(n = {len(events):,})", fontsize=11, y=1.0,
    )

    for ax, (col, label, tag, fmt) in zip(axes, panels):
        v_all = events[col].replace([np.inf, -np.inf], np.nan).dropna()
        v = v_all[v_all > 0]          # log axis: zeros cannot be drawn
        bp = ax.boxplot(
            v, widths=0.55, patch_artist=True, showfliers=True,
            flierprops=dict(marker=".", ms=2, mfc="#7FB3D5", mec="none", alpha=0.45),
            medianprops=dict(color="black", lw=1.2),
            boxprops=dict(facecolor=PALETTE["box"], edgecolor="#3B6E9B", lw=0.8),
            whiskerprops=dict(color="#3B6E9B", lw=0.8),
            capprops=dict(color="#3B6E9B", lw=0.8),
        )
        ax.plot(1, v_all.mean(), "D", color=PALETTE["mean"], ms=5, zorder=5)
        ax.set_yscale("log")
        n0 = int((v_all <= 0).sum())
        if n0:
            ax.text(0.5, 0.01, f"{n0} events with zero deficit\nincluded in statistics, not drawn",
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=6)
        ax.set_ylabel(label)
        ax.set_xticks([])
        ax.text(0.03, 0.975, tag, transform=ax.transAxes, va="top",
                fontsize=10, fontweight="bold")
        _statbox(ax, v_all, loc="upper right", fmt=fmt)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig, _save(fig, path_stem)


# ===========================================================================
# Fig. 6 -- chain descriptor histograms
# ===========================================================================
def fig06_chain_distributions(
    chains: pd.DataFrame,
    path_stem: str = "output/figures/fig06_chain_distributions",
):
    set_style()
    panels = [
        ("chain_duration", "Duration (days)", "(a)", False, "{:.1f}"),
        ("chain_size", "Chain size (nodes)", "(b)", False, "{:.3g}"),
        ("propagation_fraction", "Propagation fraction", "(c)", False, "{:.3g}"),
        ("lag_mean", "Lag (days)", "(d)", False, "{:.3g}"),
        ("chain_severity", "Severity (--)", "(e)", True, "{:.3g}"),
        ("chain_severity_hm3", "Volumetric severity (hm$^3$)", "(f)", True, "{:.3g}"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.0))

    for ax, (col, label, tag, logx, fmt) in zip(axes.ravel(), panels):
        v = chains[col].replace([np.inf, -np.inf], np.nan).dropna()
        if col == "lag_mean":
            v = v.abs()                      # the original plots lag magnitude
        if logx:
            v = v[v > 0]
            bins = np.logspace(np.log10(v.min()), np.log10(v.max()), 28)
            ax.set_xscale("log")
        else:
            bins = 26
        ax.hist(v, bins=bins, color=PALETTE["box"], edgecolor="#3B6E9B", lw=0.5)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.text(0.03, 0.965, tag, transform=ax.transAxes, va="top",
                fontsize=10, fontweight="bold")
        _statbox(ax, v, loc="upper right", fmt=fmt)

    fig.tight_layout()
    return fig, _save(fig, path_stem)


# ===========================================================================
# Figs. 7-9 -- spatiotemporal chain figure
# ===========================================================================
def fig07_chain_spatiotemporal(
    chain_id: str,
    chains: pd.DataFrame,
    pairs: pd.DataFrame,
    ssi_df: pd.DataFrame,
    events: pd.DataFrame,
    nodes: pd.DataFrame,
    boundary_gdf=None,
    rivers_gdf=None,
    context_days: int = 30,
    path_stem: Optional[str] = None,
):
    """
    The paper's signature figure. Left: one SSI panel per node in the chain,
    stacked by earliest onset, with severity shading and per-event duration and
    lag annotated at the right margin. Right: the chain's spatial extent, with
    the origin node, implicated upstream nodes, connected but unimplicated
    nodes, and the remainder of the network distinguished.
    """
    set_style()
    row = chains.loc[chains["chain_id"] == chain_id].iloc[0]
    members = pairs.loc[
        (pairs["origin_node"].astype(str) + "_" + pairs["origin_event_id"].astype(str))
        == chain_id
    ].copy()

    origin_node = int(row["origin_node"])
    # The chain summary does not carry origin_event_id; chain_id encodes it as
    # "{origin_node}_{origin_event_id}", so recover it from there rather than
    # assuming a column that build_chain_summary drops.
    origin_event_id = int(str(chain_id).rsplit("_", 1)[1])
    order = members.sort_values("upstream_start")["upstream_node"].tolist()
    stack = order + [origin_node]

    lo = pd.Timestamp(row["chain_start"]) - pd.Timedelta(days=context_days)
    hi = pd.Timestamp(row["chain_end"]) + pd.Timedelta(days=context_days)

    n = len(stack)
    fig = plt.figure(figsize=(14.0, max(6.2, 0.46 * n + 2.2)))
    # wspace must leave room for the per-event lag/duration annotations that
    # sit just outside the right edge of the left panels.
    gs = fig.add_gridspec(1, 2, width_ratios=[2.05, 1.0], wspace=0.22)
    gs_left = gs[0, 0].subgridspec(n, 1, hspace=0.0)
    ax_map = fig.add_subplot(gs[0, 1])
    # Move the map's tick labels and axis title to the right so they cannot
    # collide with those annotations.
    ax_map.yaxis.set_label_position("right")
    ax_map.yaxis.tick_right()

    label = nodes.set_index("node_id")
    axes = []
    for i, nid in enumerate(stack):
        ax = fig.add_subplot(gs_left[i, 0])
        axes.append(ax)
        sub = ssi_df[(ssi_df["node_id"] == nid) & ssi_df["date"].between(lo, hi)].sort_values("date")
        ax.plot(sub["date"], sub["SSI"], color=PALETTE["daily_ssi"], lw=0.65)
        ax.axhline(BREAKS["moderate"], color="black", lw=0.4, ls="--")

        if nid == origin_node:
            ev = events[(events["node_id"] == nid) & (events["event_id"] == origin_event_id)]
        else:
            eid = members.loc[members["upstream_node"] == nid, "upstream_event_id"].iloc[0]
            ev = events[(events["node_id"] == nid) & (events["event_id"] == eid)]

        if not ev.empty:
            ev = ev.iloc[0]
            m = sub["date"].between(ev["start_date"], ev["end_date"])
            x, y = sub.loc[m, "date"], sub.loc[m, "SSI"]
            for key in ("moderate", "severe", "extreme"):
                ax.fill_between(x, y, BREAKS[key], where=y < BREAKS[key],
                                color=PALETTE[key], interpolate=True, lw=0)
            lag = 0 if nid == origin_node else int(
                members.loc[members["upstream_node"] == nid, "lag_days"].iloc[0]
            )
            ax.text(1.012, 0.5, f"L = {lag} d\nD = {int(ev['duration'])} d",
                    transform=ax.transAxes, va="center", fontsize=5.6)

        ax.set_xlim(lo, hi)
        ax.set_ylim(-3.2, 1.6)
        ax.set_yticks([-2, 0])
        ax.tick_params(axis="y", labelsize=5.5, length=2)
        name = (str(label.loc[nid, "label"]) if "label" in label.columns and nid in label.index
                else str(nid))
        tag = name + (" (origin)" if nid == origin_node else "")
        ax.text(0.005, 0.97, tag, transform=ax.transAxes, va="top", ha="left", fontsize=6.6,
                fontweight="bold" if nid == origin_node else "normal")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if i < n - 1:
            ax.set_xticks([])
            ax.spines["bottom"].set_visible(False)

    axes[0].set_title(
        f"Propagation chain {chain_id}  |  Origin "
        f"{label.loc[origin_node, 'label'] if 'label' in label.columns else origin_node}  |  "
        f"{pd.Timestamp(row['chain_start']):%d %b %Y} – {pd.Timestamp(row['chain_end']):%d %b %Y}\n"
        f"D$_c$ = {int(row['chain_duration'])} d | N$_c$ = {int(row['chain_size'])} nodes | "
        f"f$_p$ = {row['propagation_fraction']:.3f} | "
        f"$\\lambda$ = {row['lag_mean']:.1f} d | "
        f"S$_c$ = {row['chain_severity']:.1f} | V$_c$ = {row['chain_severity_hm3']:.1f} hm$^3$",
        fontsize=9, pad=12,
    )
    axes[n // 2].set_ylabel("SSI (--)", fontsize=8)

    # ---- right panel: spatial extent --------------------------------------
    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=ax_map, color="black", lw=1.0)
    if rivers_gdf is not None:
        rivers_gdf.plot(ax=ax_map, color="#8FB8D8", lw=0.35)

    implicated = set(order)
    connected = set(
        pairs.loc[pairs["origin_node"] == origin_node, "upstream_node"]
    ) - implicated
    other = set(nodes["node_id"]) - implicated - connected - {origin_node}

    for ids, colour, lbl, size in [
        (other, "#B9C2CB", f"Other nodes (n={len(other)})", 14),
        (connected, "#7F8C8D", f"Upstream not implicated (n={len(connected)})", 22),
        (implicated, "#F1C40F", f"Upstream implicated (n={len(implicated)})", 34),
        ({origin_node}, "#C0392B", "Origin node (n=1)", 60),
    ]:
        sel = nodes[nodes["node_id"].isin(ids)]
        if not sel.empty:
            ax_map.scatter(sel["lon"], sel["lat"], s=size, c=colour,
                           edgecolor="black", lw=0.3, label=lbl, zorder=4)

    for _, r in nodes[nodes["node_id"].isin(implicated | {origin_node})].iterrows():
        ax_map.annotate(str(r["label"]) if "label" in r.index else str(int(r["node_id"])),
                        (r["lon"], r["lat"]),
                        fontsize=4.8, xytext=(2, 2), textcoords="offset points")

    ax_map.set_title("Spatial extent of the chain", fontsize=9)
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.legend(loc="lower left", fontsize=6.2, framealpha=0.93)

    stem = path_stem or f"output/figures/fig07_chain_{chain_id}"
    return fig, _save(fig, stem)


# ===========================================================================
# Volta-only figures
# ===========================================================================
def fig10_transboundary(
    matrix: pd.DataFrame,
    chain_class: pd.DataFrame,
    path_stem: str = "output/figures/fig10_transboundary",
):
    """
    (a) country-to-country transmission matrix;
    (b) distribution of the foreign fraction across chains;
    (c) lead-time gain, the extra warning a foreign headwater signal provides
        over the nearest domestic one. Panel (c) is the operational payload of
        the paper for the Volta Basin Authority.
    """
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3))

    ax = axes[0]
    im = ax.imshow(matrix.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=7)
    ax.set_xlabel("Origin (downstream) state")
    ax.set_ylabel("Upstream state")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix.values[i, j]
            if v:
                ax.text(j, i, f"{v:g}", ha="center", va="center", fontsize=6,
                        color="white" if v > np.nanmax(matrix.values) * 0.55 else "black")
    fig.colorbar(im, ax=ax, pad=0.02, label="Matched event pairs")
    ax.set_title("(a) Transmission matrix", fontsize=9, loc="left")

    ax = axes[1]
    v = chain_class["foreign_fraction"].dropna()
    ax.hist(v, bins=20, color=PALETTE["box"], edgecolor="#3B6E9B", lw=0.5)
    ax.set_xlabel("Fraction of chain contributors outside the origin state")
    ax.set_ylabel("Count")
    ax.set_title("(b) Foreign contribution", fontsize=9, loc="left")
    _statbox(ax, v)

    ax = axes[2]
    g = chain_class["lead_time_gain_days"].dropna()
    ax.hist(g, bins=24, color="#CBE4C8", edgecolor="#4C8C4A", lw=0.5)
    ax.axvline(0, color="black", lw=1.0, ls="--")
    ax.set_xlabel("Lead-time gain from foreign headwaters (days)")
    ax.set_ylabel("Count")
    ax.set_title("(c) Transboundary early-warning value", fontsize=9, loc="left")
    _statbox(ax, g)

    fig.tight_layout()
    return fig, _save(fig, path_stem)


def fig11_validation(
    flow: pd.DataFrame,
    evt: pd.DataFrame,
    path_stem: str = "output/figures/fig11_validation",
):
    """GloFAS against GRDC: overall skill, low-flow skill, and event agreement."""
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))

    ax = axes[0]
    ax.scatter(flow["KGE"], flow.get("low_flow_KGE", pd.Series(dtype=float)),
               s=36, c="#3B6E9B", edgecolor="black", lw=0.4)
    ax.axhline(0.3, color="#C0392B", ls="--", lw=0.9)
    ax.axvline(0.5, color="#C0392B", ls="--", lw=0.9)
    ax.set_xlabel("KGE, full record")
    ax.set_ylabel("KGE, lower decile")
    ax.set_title("(a) Flow and low-flow skill", fontsize=9, loc="left")

    ax = axes[1]
    ax.scatter(evt["POD"], evt["FAR"], s=36, c="#4C8C4A", edgecolor="black", lw=0.4)
    ax.plot([0, 1], [0, 1], color="#9AA5B1", lw=0.7, ls=":")
    ax.set_xlabel("Probability of detection")
    ax.set_ylabel("False alarm ratio")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("(b) Event catalogue agreement", fontsize=9, loc="left")

    ax = axes[2]
    v = evt["onset_bias_days"].dropna()
    ax.hist(v, bins=18, color="#F3D5A5", edgecolor="#B9872F", lw=0.5)
    ax.axvline(0, color="black", ls="--", lw=1.0)
    ax.set_xlabel("Onset bias, reanalysis minus observed (days)")
    ax.set_ylabel("Count")
    ax.set_title("(c) Onset timing", fontsize=9, loc="left")
    _statbox(ax, v)

    fig.tight_layout()
    return fig, _save(fig, path_stem)
