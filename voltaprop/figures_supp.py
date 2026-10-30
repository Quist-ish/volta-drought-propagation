"""
Supplementary figures S1-S9, matching the published Supporting Information.

  figS1  Event-level sensitivity of chain size to M
  figS2  Case-study chains under varying M
  figS3  Lag versus along-channel distance, two panels
  figS4  Network-permutation null test, five statistics
  figS5  Detection sensitivity grid with the per-station bias check, six panels
  figS6  Meteorological context of the case studies
  figS7  Events that do not propagate, five panels
  figS8  Validation against an independent drought catalogue, three panels
  figS9  Homogeneity screening and reservoir proximity

Author: I. Quist (NUIST)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .figures import PALETTE, _save, _statbox, set_style

BOX = dict(
    patch_artist=True,
    medianprops=dict(color="black", lw=1.2),
    boxprops=dict(facecolor=PALETTE["box"], edgecolor="#3B6E9B", lw=0.8),
    whiskerprops=dict(color="#3B6E9B", lw=0.8),
    capprops=dict(color="#3B6E9B", lw=0.8),
    flierprops=dict(marker=".", ms=2.5, mfc="#7FB3D5", mec="none", alpha=0.5),
)


# ===========================================================================
def figS1_m_sensitivity(summary: pd.DataFrame, per_event: pd.DataFrame,
                        path_stem: str = "output/figures/figS1_m_sensitivity"):
    """Chain-size stability across M, with the flip rate and rank correlation."""
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))

    m_cols = [c for c in per_event.columns if c.startswith("M")]
    data = [per_event[c].dropna() for c in m_cols]
    axes[0].boxplot(data, widths=0.55, **BOX)
    axes[0].set_xticklabels([c[1:] for c in m_cols])
    axes[0].set_xlabel("Minimum overlap $M$ (days)")
    axes[0].set_ylabel("Chain size (upstream nodes)")
    axes[0].set_title("(a) Chain-size distribution", fontsize=9, loc="left")

    ax = axes[1]
    ax.plot(summary["M_days"], summary["spearman_rho_vs_reference"],
            "o-", color=PALETTE["daily_ssi"], lw=1.3, ms=5)
    ax.axhline(0.97, color="#C0392B", ls="--", lw=0.9)
    ax.text(summary["M_days"].max(), 0.971, "Ebro: $\\rho$ > 0.97 for M $\\leq$ 3",
            ha="right", va="bottom", fontsize=6.2, color="#C0392B")
    ax.set_xlabel("Minimum overlap $M$ (days)")
    ax.set_ylabel("Spearman $\\rho$ vs $M$ = 5")
    ax.set_ylim(min(0.5, float(summary["spearman_rho_vs_reference"].min(skipna=True) or 0.5)), 1.02)
    ax.set_title("(b) Rank stability", fontsize=9, loc="left")

    ax = axes[2]
    ax.bar(summary["M_days"], summary["flip_vs_reference_pct"],
           width=1.1, color=PALETTE["moderate"], edgecolor="#B9872F", lw=0.6)
    ax.axhline(2.2, color="#C0392B", ls="--", lw=0.9)
    ax.text(summary["M_days"].max(), 2.25, "Ebro: < 2.2% for M $\\leq$ 5",
            ha="right", va="bottom", fontsize=6.2, color="#C0392B")
    ax.set_xlabel("Minimum overlap $M$ (days)")
    ax.set_ylabel("Events changing classification (%)")
    ax.set_title("(c) Classification flips", fontsize=9, loc="left")

    fig.tight_layout()
    return fig, _save(fig, path_stem)


# ===========================================================================
def figS2_case_m_sensitivity(case_sens: pd.DataFrame,
                             path_stem: str = "output/figures/figS2_case_m_sensitivity"):
    """Per-case chain size against M, with membership similarity to M = 5."""
    set_style()
    cases = case_sens["chain_id"].unique()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0))

    for cid in cases:
        s = case_sens[case_sens["chain_id"] == cid].sort_values("M_days")
        axes[0].plot(s["M_days"], s["chain_size"], "o-", lw=1.3, ms=5, label=cid)
        axes[1].plot(s["M_days"], s["jaccard_vs_default"], "o-", lw=1.3, ms=5, label=cid)

    axes[0].set_xlabel("Minimum overlap $M$ (days)")
    axes[0].set_ylabel("Chain size (nodes)")
    axes[0].set_title("(a) Chain size by case study", fontsize=9, loc="left")
    axes[0].axvline(5, color="#C0392B", ls="--", lw=0.9)
    axes[0].legend(fontsize=6.5, title="Chain", title_fontsize=6.5)

    axes[1].set_xlabel("Minimum overlap $M$ (days)")
    axes[1].set_ylabel("Jaccard similarity of membership vs $M$ = 5")
    axes[1].set_ylim(0, 1.03)
    axes[1].axvline(5, color="#C0392B", ls="--", lw=0.9)
    axes[1].set_title("(b) Membership stability", fontsize=9, loc="left")

    fig.tight_layout()
    return fig, _save(fig, path_stem)


# ===========================================================================
def figS3_lag_vs_distance(pair_df: pd.DataFrame, stats_dict: dict,
                          regulated_nodes: Optional[Sequence[int]] = None,
                          path_stem: str = "output/figures/figS3_lag_vs_distance"):
    """(a) all matched pair-events with binned median; (b) one point per node pair."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))
    d = pair_df.dropna(subset=["distance_km", "lag_days"]).copy()
    d["abs_lag"] = d["lag_days"].abs()

    ax = axes[0]
    ax.scatter(d["distance_km"], d["abs_lag"], s=9, alpha=0.35,
               c=PALETTE["daily_ssi"], edgecolor="none")
    if len(d) > 40:
        bins = np.linspace(d["distance_km"].min(), d["distance_km"].max(), 13)
        idx = np.digitize(d["distance_km"], bins)
        xs, ys = [], []
        for b in range(1, len(bins)):
            sel = d[idx == b]
            if len(sel) >= 10:
                xs.append(0.5 * (bins[b - 1] + bins[b]))
                ys.append(sel["abs_lag"].median())
        if xs:
            ax.plot(xs, ys, "o-", color="#C0392B", lw=1.5, ms=5,
                    label="Binned median ($\\geq$10 pairs)")
            ax.legend(fontsize=6.8)
    ax.set_xlabel("Along-channel distance (km)")
    ax.set_ylabel("|Propagation lag| (days)")
    ax.set_title(f"(a) All matched pair-events (n = {len(d):,})", fontsize=9, loc="left")

    ax = axes[1]
    g = d.groupby(["upstream_node", "origin_node"]).agg(
        distance_km=("distance_km", "first"),
        abs_lag=("abs_lag", "mean"),
        n=("abs_lag", "size"),
    ).reset_index()
    reg = g["upstream_node"].isin(regulated_nodes or [])
    ax.scatter(g.loc[~reg, "distance_km"], g.loc[~reg, "abs_lag"],
               s=8 + 2.5 * g.loc[~reg, "n"], alpha=0.75, c="#3B6E9B",
               edgecolor="black", lw=0.3, label="Near-natural upstream")
    if reg.any():
        ax.scatter(g.loc[reg, "distance_km"], g.loc[reg, "abs_lag"],
                   s=8 + 2.5 * g.loc[reg, "n"], alpha=0.9, c="#C0392B",
                   edgecolor="black", lw=0.4, label="Reservoir-regulated upstream")
    ax.set_xlabel("Along-channel distance (km)")
    ax.set_ylabel("Mean |lag| (days)")
    ax.set_title(f"(b) One point per node pair (n = {len(g):,})", fontsize=9, loc="left")
    ax.legend(fontsize=6.5)

    txt = "\n".join(
        f"{k}: {v:.4g}" for k, v in stats_dict.items()
        if k in ("spearman_r", "spearman_p", "r_squared", "slope_days_per_km")
    )
    if txt:
        ax.text(0.97, 0.03, txt, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.3, family="DejaVu Sans Mono",
                bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="#9AA5B1", lw=0.5))

    fig.tight_layout()
    return fig, _save(fig, path_stem)


# ===========================================================================
def figS4_null_test(summary: pd.DataFrame, null: pd.DataFrame,
                    path_stem: str = "output/figures/figS4_null_test"):
    """
    Five panels, one per statistic: the null distribution with the observed
    value marked and the one-sided p-value annotated.

    Watch the location of the null mean, not only the p-value. In the Ebro the
    null connected fraction was already ~50%, because the permutation preserves
    genuine regional synchrony. Under the West African Monsoon it may sit
    higher still, and that baseline is itself a reportable result.
    """
    set_style()
    stats_order = [s for s in summary["statistic"] if s in null.columns]
    n = len(stats_order)
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 3.9))
    if n == 1:
        axes = [axes]

    labels = {
        "connected_fraction": "Connected fraction of origin events",
        "mean_chain_size": "Mean chain size",
        "mean_propagation_fraction": "Mean propagation fraction",
        "chain_count": "Chain count (raw)",
        "total_valid_pairs": "Total valid matched pairs (raw)",
    }

    for ax, stat in zip(axes, stats_order):
        row = summary[summary["statistic"] == stat].iloc[0]
        col = null[stat].dropna()
        ax.hist(col, bins=38, color=PALETTE["box"], edgecolor="#3B6E9B", lw=0.4)
        ax.axvline(row["observed"], color="#C0392B", lw=1.8)
        ax.set_xlabel(labels.get(stat, stat), fontsize=7.4)
        ax.set_ylabel("Frequency")
        sig = "" if row["significant_at_0.05"] else "  (n.s.)"
        ax.set_title(
            f"p = {row['p_value_one_sided']:.4g}{sig}\n"
            f"obs {row['observed']:.3g} vs null {row['null_mean']:.3g} "
            f"$\\pm$ {row['null_sd']:.3g}",
            fontsize=7.2,
        )
        ax.text(0.03, 0.96, f"{row['observed_percentile_in_null']:.0f}th pct",
                transform=ax.transAxes, va="top", fontsize=6.4, color="#C0392B")

    fig.suptitle(
        f"Network-permutation null test ({int(summary['n_permutations'].iloc[0]):,} permutations); "
        "the null preserves all genuine regional climatic synchrony",
        fontsize=9.5, y=1.02,
    )
    fig.tight_layout()
    return fig, _save(fig, path_stem)


# ===========================================================================
def figS5_detection_sensitivity(summary: pd.DataFrame,
                                path_stem: str = "output/figures/figS5_detection_sensitivity"):
    """Six panels: counts, duration, change, and the two station-bias checks."""
    set_style()
    panels = [
        ("n_events", "Event count", "(a)", "RdYlBu"),
        ("mean_duration", "Mean duration (days)", "(b)", "viridis"),
        ("pct_change_vs_default", "Change in event count (%)", "(c)", "RdBu_r"),
        ("rho_count_vs_default", "$\\rho$, per-node event count", "(e)", "viridis"),
        ("rho_severity_vs_default", "$\\rho$, per-node mean severity", "(f)", "viridis"),
    ]
    present = [p for p in panels if p[0] in summary.columns]
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.0))
    flat = axes.ravel()

    default = summary[summary["is_default"]]
    dg = int(default["pool_gap"].iloc[0]) if not default.empty else 10
    dd = int(default["min_duration"].iloc[0]) if not default.empty else 5

    for ax, (col, title, tag, cmap) in zip(flat, present):
        piv = summary.pivot(index="pool_gap", columns="min_duration", values=col)
        im = ax.pcolormesh(piv.columns, piv.index, piv.values, cmap=cmap, shading="nearest")
        ax.plot(dd, dg, "+", color="#C0392B", ms=14, mew=2.4)
        ax.set_xlabel("Minimum duration (days)")
        ax.set_ylabel("Pooling gap (days)")
        ax.set_title(f"{tag} {title}", fontsize=8.6, loc="left")
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.ax.tick_params(labelsize=6.5)
        for i, r in enumerate(piv.index):
            for j, c in enumerate(piv.columns):
                v = piv.values[i, j]
                if np.isfinite(v):
                    ax.text(c, r, f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}",
                            ha="center", va="center", fontsize=5.6)

    for ax in flat[len(present):]:
        ax.axis("off")
        rng = (summary["n_events"].min(), summary["n_events"].max())
        ax.text(
            0.02, 0.95,
            "Event count across the grid\n"
            f"  range: {rng[0]:,} to {rng[1]:,}\n"
            f"  published configuration: gap = {dg} d, min duration = {dd} d\n\n"
            "Ebro reference\n"
            "  1,834 to 5,155 events (-30% to +96% of 2,625)\n"
            "  severity rankings rho = 0.83-1.00 for 19 of 20 combinations\n\n"
            "What matters is panels (e) and (f): whether a threshold choice\n"
            "affects some nodes more than others. Stable rankings mean the\n"
            "published configuration sits in a stable region rather than at\n"
            "an unstable extreme.",
            transform=ax.transAxes, va="top", fontsize=6.6, family="DejaVu Sans Mono",
        )

    fig.tight_layout()
    return fig, _save(fig, path_stem)


# ===========================================================================
def figS6_meteorological_context(
    case_chain_ids: Sequence[str],
    chains: pd.DataFrame,
    ssi_df: pd.DataFrame,
    events: pd.DataFrame,
    spei: pd.DataFrame,
    path_stem: str = "output/figures/figS6_meteorological_context",
):
    """One row per case: daily SSI at the origin against SPEI-3 and SPEI-12."""
    set_style()
    n = len(case_chain_ids)
    fig, axes = plt.subplots(n, 1, figsize=(11.4, 3.0 * n), squeeze=False)

    for ax, cid in zip(axes[:, 0], case_chain_ids):
        row = chains.loc[chains["chain_id"] == cid]
        if row.empty:
            ax.axis("off")
            continue
        row = row.iloc[0]
        onode = int(row["origin_node"])
        cs, ce = pd.Timestamp(row["chain_start"]), pd.Timestamp(row["chain_end"])
        lo, hi = cs - pd.Timedelta(days=120), ce + pd.Timedelta(days=120)

        ax.axvspan(cs, ce, color="#E4E4E4", alpha=0.8, label="Network-wide chain window")

        s = ssi_df[(ssi_df["node_id"] == onode) & ssi_df["date"].between(lo, hi)]
        ax.plot(s["date"], s["SSI"], color=PALETTE["daily_ssi"], lw=0.9, label="Daily SSI (origin)")
        ax.axhline(-1.28, color="black", ls="--", lw=0.8)

        own = events[(events["node_id"] == onode)
                     & (events["start_date"] <= hi) & (events["end_date"] >= lo)]
        for e in own.itertuples(index=False):
            m = s["date"].between(e.start_date, e.end_date)
            ax.plot(s.loc[m, "date"], s.loc[m, "SSI"], color="#C0392B", lw=1.8)

        sp = spei[(spei["node_id"] == onode) & spei["date"].between(lo, hi)]
        if not sp.empty:
            ax2 = ax.twinx()
            if "SPEI_3" in sp:
                ax2.plot(sp["date"], sp["SPEI_3"], color="#E67E22", lw=1.0,
                         alpha=0.85, label="SPEI-3")
            if "SPEI_12" in sp:
                ax2.plot(sp["date"], sp["SPEI_12"], color="#7D3C98", lw=1.0,
                         alpha=0.85, label="SPEI-12")
            ax2.axhline(0, color="#999999", lw=0.5)
            ax2.set_ylabel("SPEI")
            ax2.legend(loc="lower right", fontsize=6.2)

        ax.set_ylabel("SSI")
        ax.set_title(
            f"{cid}  |  origin node {onode}  |  chain {cs:%b %Y} – {ce:%b %Y} "
            f"({int(row['chain_duration'])} d)  |  origin's own events in window: {len(own)}",
            fontsize=8.2, loc="left",
        )
        ax.legend(loc="lower left", fontsize=6.2)

    fig.tight_layout()
    return fig, _save(fig, path_stem)


# ===========================================================================
def figS7_non_propagating(origin_contrast: pd.DataFrame,
                          upstream_contrast: pd.DataFrame,
                          by_node: Optional[pd.DataFrame] = None,
                          path_stem: str = "output/figures/figS7_non_propagating"):
    """Isolated downstream events and non-propagating upstream events."""
    set_style()
    fig, axes = plt.subplots(1, 5, figsize=(15.0, 3.9))

    ax = axes[0]
    if by_node is not None and not by_node.empty:
        b = by_node.sort_values("pct_unconnected", ascending=False)
        ax.bar(range(len(b)), b["pct_unconnected"], color=PALETTE["box"],
               edgecolor="#3B6E9B", lw=0.5)
        ax.set_xticks(range(len(b)))
        ax.set_xticklabels(b["origin_node"].astype(str), rotation=90, fontsize=5.4)
    ax.set_xlabel("Origin node")
    ax.set_ylabel("Origin events without upstream link (%)")
    ax.set_title("(a)", fontsize=9, loc="left")

    for ax, df, col, lbl, tag in [
        (axes[1], origin_contrast, "mean_duration", "Duration (days)", "(b)"),
        (axes[2], origin_contrast, "mean_severity", "Severity (--)", "(c)"),
        (axes[3], upstream_contrast, "mean_duration", "Duration (days)", "(d)"),
        (axes[4], upstream_contrast, "mean_severity", "Severity (--)", "(e)"),
    ]:
        if df is None or df.empty or col not in df.columns:
            ax.axis("off")
            continue
        flag = df.columns[0]
        ax.bar(range(len(df)), df[col], color=[PALETTE["moderate"], PALETTE["box"]][: len(df)],
               edgecolor="#3B6E9B", lw=0.6)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([str(v) for v in df[flag]], fontsize=7)
        ax.set_xlabel(flag.replace("_", " "))
        ax.set_ylabel(lbl)
        ax.set_title(tag, fontsize=9, loc="left")

    fig.suptitle(
        "Events that do not propagate. Ebro reference: 602 of 1,558 upstream "
        "candidates (38.6%) never accepted; non-propagating events were about "
        "three to four times shorter and less severe.",
        fontsize=8, y=1.03,
    )
    fig.tight_layout()
    return fig, _save(fig, path_stem)


# ===========================================================================
def figS8_catalogue_validation(flagged: pd.DataFrame, summary: pd.DataFrame,
                               catalogue: pd.DataFrame,
                               cat_start_col: str = "start_date",
                               cat_end_col: str = "end_date",
                               path_stem: str = "output/figures/figS8_catalogue_validation"):
    """Timeline of chains against documented droughts, plus severity and size."""
    set_style()
    fig = plt.figure(figsize=(13.4, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.1, 1, 1], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])

    for e in catalogue.itertuples(index=False):
        ax.axvspan(pd.Timestamp(getattr(e, cat_start_col)),
                   pd.Timestamp(getattr(e, cat_end_col)),
                   color="#D5D8DC", alpha=0.85, zorder=1)

    f = flagged.copy()
    f["chain_start"] = pd.to_datetime(f["chain_start"])
    sz = 6 + 42 * (f["chain_severity"] / max(f["chain_severity"].max(), 1e-9))
    for flag, colour, lbl in [(True, "#E67E22", "Overlaps a documented drought"),
                              (False, "#3B6E9B", "No overlap")]:
        m = f["overlaps_documented"] == flag
        ax.scatter(f.loc[m, "chain_start"], f.loc[m, "chain_severity"],
                   s=sz[m], c=colour, alpha=0.75, edgecolor="black", lw=0.25,
                   label=lbl, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("Chain start")
    ax.set_ylabel("Chain severity (--)")
    ax.set_title("(a) Chains against documented droughts", fontsize=9, loc="left")
    ax.legend(fontsize=6.5, loc="upper left")

    for i, (col, lbl, tag) in enumerate(
        [("chain_severity", "Chain severity (--)", "(b)"),
         ("chain_size", "Chain size (nodes)", "(c)")]
    ):
        a = fig.add_subplot(gs[0, i + 1])
        groups, labels = [], []
        for flag, lbl2 in [(True, "Overlapping"), (False, "Non-overlapping")]:
            v = flagged.loc[flagged["overlaps_documented"] == flag, col].dropna()
            if len(v):
                groups.append(v)
                labels.append(f"{lbl2}\n(n={len(v)})")
        if groups:
            a.boxplot(groups, widths=0.55, **BOX)
            a.set_xticklabels(labels, fontsize=6.8)
        if col == "chain_severity":
            a.set_yscale("log")
        a.set_ylabel(lbl)
        a.set_title(tag, fontsize=9, loc="left")

    r = summary.iloc[0]
    fig.suptitle(
        f"{r['pct_chains_overlapping']:.1f}% of chains overlap a documented drought "
        f"(calendar coverage baseline {r['calendar_coverage_pct']:.1f}%); "
        f"severity ratio {r.get('severity_ratio', float('nan')):.2f}. "
        "Ebro: 78.7% against 42.8% coverage, severity ratio 3.4.",
        fontsize=8.2, y=1.02,
    )
    return fig, _save(fig, path_stem)


# ===========================================================================
def figS9_homogeneity_screening(table: pd.DataFrame, summary: pd.DataFrame,
                                path_stem: str = "output/figures/figS9_homogeneity"):
    """Distance to the nearest reservoir, flagged against retained."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))

    ax = axes[0]
    groups, labels = [], []
    for flag, lbl in [(True, "Flagged"), (False, "Retained")]:
        v = table.loc[table["flag"] == flag, "dist_nearest_reservoir_km"].dropna()
        if len(v):
            groups.append(v)
            labels.append(f"{lbl}\n(n={len(v)})")
    if groups:
        ax.boxplot(groups, widths=0.55, **BOX)
        ax.set_xticklabels(labels, fontsize=7.4)
    ax.axhline(5, color="#C0392B", ls="--", lw=0.9)
    ax.set_ylabel("Distance to nearest reservoir (km)")
    ax.set_title("(a) Reservoir proximity", fontsize=9, loc="left")

    ax = axes[1]
    groups, labels = [], []
    for flag, lbl in [(True, "Flagged"), (False, "Retained")]:
        v = table.loc[table["flag"] == flag, "uparea_km2"].dropna()
        if len(v):
            groups.append(v)
            labels.append(f"{lbl}\n(n={len(v)})")
    if groups:
        ax.boxplot(groups, widths=0.55, **BOX)
        ax.set_xticklabels(labels, fontsize=7.4)
    ax.set_yscale("log")
    ax.set_ylabel("Upstream area (km$^2$)")
    ax.set_title("(b) Catchment size, the confound to report", fontsize=9, loc="left")

    if not summary.empty and "p_value_one_sided" in summary.columns:
        r = summary.iloc[0]
        fig.suptitle(
            f"Median distance {r['median_dist_flagged_km']:.1f} km flagged vs "
            f"{r['median_dist_retained_km']:.1f} km retained, "
            f"Mann-Whitney one-sided p = {r['p_value_one_sided']:.4g}. "
            "Ebro: 5.0 vs 12.6 km, p = 0.005.",
            fontsize=8.2, y=1.02,
        )
    fig.tight_layout()
    return fig, _save(fig, path_stem)
