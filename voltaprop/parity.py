"""
Parity reporting against the published Ebro study.

Crespillo Lopez, A., Vicente-Serrano, S.M., Gimeno, L. (2026).
Tracking hydrological drought propagation in river networks: an event-based
framework. J. Hydrol. 680, 136544.

Every quantity that study reports has a Volta counterpart. This module
computes each one from the pipeline outputs and prints it beside the published
value, so the comparison is mechanical rather than a matter of the author
remembering to look. Where a quantity cannot be computed, the row says so
rather than being dropped, because a silently absent comparison is the one a
reviewer finds.

Two cautions are built in rather than left to the reader.

The study periods differ: the Ebro covers 1961-2020, sixty years, the Volta
1979-2024, forty-six. Raw counts are therefore not comparable and are marked
`scale: period`. Per-station rates and distributional statistics are.

The networks differ: 33 observed gauges against a constructed node set. Counts
that scale with network size are marked `scale: network`. A difference in
those rows is a difference in study design, not in hydrology, and must not be
interpreted as one.

    from voltaprop.parity import build_parity_table, format_parity_report
    tab = build_parity_table(events=ev, chains=ch, pairs=pr, null=nl)
    print(format_parity_report(tab))

Author: I. Quist (NUIST)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# The published reference values live in supplementary.py. Import them in a
# way that survives all three ways this file gets loaded: as part of the
# package, as a top-level module on sys.path, and pasted straight into a
# notebook cell, where __package__ is unset and a relative import raises
# "attempted relative import with no known parent package".
def _load_ebro_reference():
    try:                                    # normal: part of the package
        from .supplementary import EBRO_REFERENCE
        return EBRO_REFERENCE
    except ImportError:
        pass
    try:                                    # repo root on sys.path
        from voltaprop.supplementary import EBRO_REFERENCE
        return EBRO_REFERENCE
    except ImportError:
        pass
    try:                                    # sitting beside this file
        import importlib.util
        from pathlib import Path as _P
        here = _P(__file__).resolve().parent
        spec = importlib.util.spec_from_file_location(
            "_voltaprop_supplementary", here / "supplementary.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.EBRO_REFERENCE
    except Exception:
        pass
    raise ImportError(
        "Could not load EBRO_REFERENCE from voltaprop/supplementary.py.\n"
        "If you pasted this file into a notebook cell, import it instead:\n"
        "    import sys\n"
        "    sys.path.insert(0, '<path to>/volta_drought_propagation')\n"
        "    from voltaprop.parity import build_parity_table, "
        "format_parity_report"
    )


E = _load_ebro_reference()

# scale flags: how a row may legitimately differ for reasons of study design
FREE = ""              # directly comparable
PERIOD = "period"      # scales with record length (60 y vs 46 y)
NETWORK = "network"    # scales with number of nodes (33 gauges vs n nodes)


def _safe(fn, *a, **k):
    """Return fn(*a) or NaN, so one absent column cannot empty the table."""
    try:
        v = fn(*a, **k)
        return float(v) if v is not None and np.isfinite(float(v)) else np.nan
    except Exception:
        return np.nan


def _col(df: Optional[pd.DataFrame], name: str) -> Optional[pd.Series]:
    if df is None or name not in getattr(df, "columns", []):
        return None
    s = pd.to_numeric(df[name], errors="coerce").dropna()
    return s if len(s) else None


def build_parity_table(
    events: Optional[pd.DataFrame] = None,
    chains: Optional[pd.DataFrame] = None,
    pairs: Optional[pd.DataFrame] = None,
    null: Optional[pd.DataFrame] = None,
    nodes: Optional[pd.DataFrame] = None,
    n_origin_nodes: Optional[int] = None,
    period: str = "1979-2024",
    period_years: Optional[float] = None,
    supp: Optional[dict] = None,
) -> pd.DataFrame:
    """
    One row per published quantity, with the Volta value beside it.

    Every argument is optional. Pass what exists; the rest reports as
    'not computed' so the table always shows the full set of obligations.
    """
    rows = []

    def add(section, label, ebro, volta, unit="", scale=FREE, note=""):
        rows.append({
            "section": section, "quantity": label, "ebro": ebro,
            "volta": volta, "unit": unit, "scale": scale, "note": note,
        })

    n_nodes = len(nodes) if nodes is not None else np.nan
    n_ev = len(events) if events is not None else np.nan
    n_ch = len(chains) if chains is not None else np.nan

    # --- study frame -------------------------------------------------------
    ey = E["period"][1] - E["period"][0] + 1
    window = float(period.split("-")[1]) - float(period.split("-")[0]) + 1
    # vy is the record length per node used for rates; it equals the window
    # only when every node observes throughout, which the Volta's do not
    vy = period_years if period_years else window
    note = f"{ey:.0f} y vs {window:.0f} y window"
    if abs(vy - window) > 0.5:
        note += f"; mean defined record {vy:.1f} y per node, used for all rates"
    add("frame", "study period", f"{E['period'][0]}-{E['period'][1]}",
        period, "", FREE, note)
    add("frame", "stations with events", E["n_stations"], n_nodes, "stations", NETWORK,
        "33 screened gauges vs stations holding at least one event")
    add("frame", "origin locations", E["n_origin_stations"],
        n_origin_nodes if n_origin_nodes is not None else np.nan,
        "nodes", NETWORK)

    # --- 3.1 parameter sensitivity ----------------------------------------
    add("3.1", "parameter combinations", E["n_param_combinations"], 750,
        "", FREE, "same grid: W 0-120 step 5, M 1-30 step 1")
    add("3.1", "selected W", E["selected_W"], 45, "days")
    add("3.1", "selected M", E["selected_M"], 5, "days")
    supp = supp or {}
    rsi = supp.get("rsi")
    add("3.1", "metrics with RSI(W) = 0, of 6", 6,
        int((rsi["RSI_W"].abs() < 1e-9).sum()) if rsi is not None else np.nan,
        "", FREE, "identity, not evidence: with lag <= 0 and overlap >= 1 every candidate contains s0, so W cannot matter")
    add("3.1", "metrics with RSI(M) < 0.10, of 6", E["rsi_M_below_0p10_metrics"],
        int((rsi["RSI_M"] < 0.10).sum()) if rsi is not None else np.nan,
        "", FREE, "published: M=5 lies on a stable plateau")
    knee = supp.get("knee_M")
    add("3.1", "Kneedle elbow in M", E["selected_M"],
        knee if knee is not None else ("none found" if "knee_M" in supp else np.nan),
        "days", FREE)
    s5 = supp.get("s5")
    if s5 is not None:
        rng = f"{int(s5.n_events.min())}-{int(s5.n_events.max())}"
        lo = 100 * (s5.n_events.min() / s5.loc[s5.is_default, "n_events"].iloc[0] - 1)
        hi = 100 * (s5.n_events.max() / s5.loc[s5.is_default, "n_events"].iloc[0] - 1)
    add("3.1", "event catalogue range over configs",
        f"{E['s5_event_range'][0]}-{E['s5_event_range'][1]}",
        rng if s5 is not None else np.nan, "events", PERIOD)
    add("3.1", "catalogue range, % vs default", "-23 / +39",
        f"{lo:.0f} / +{hi:.0f}" if s5 is not None else np.nan, "", FREE,
        "record-length free; proportional sensitivity of catalogue size")
    add("S5", "station rank rho, severity, minimum", E["s5_spearman_severity"][0],
        float(s5.rho_severity_vs_default.min()) if s5 is not None else np.nan, "", FREE)
    add("S5", "station rank rho, count, minimum", E["s5_spearman_worst"][0],
        float(s5.rho_count_vs_default.min()) if s5 is not None else np.nan, "", FREE)
    s1 = supp.get("s1")
    if s1 is not None:
        low = s1[(s1.M_days < 5)]
    add("S1", "Spearman at M < 5, minimum", E["s1_spearman_at_M_le_5"],
        float(low.spearman_rho_vs_reference.min()) if s1 is not None else np.nan, "", FREE,
        "published: > 0.97")
    add("S1", "events reclassified at M < 5, max", E["s1_reclassified_fraction"],
        float(low.flip_vs_reference_pct.max()) / 100 if s1 is not None else np.nan, "", FREE,
        "published: < 2%")

    # --- 3.2 event catalogue ----------------------------------------------
    add("3.2", "drought events", E["n_events"], n_ev, "", PERIOD + "+" + NETWORK)
    per_station = (n_ev / n_nodes) if (events is not None and nodes is not None
                                       and len(nodes)) else np.nan
    add("3.2", "events per node", E["events_per_station"], per_station, "",
        PERIOD, f"raw count per node; Ebro {ey:.0f} y per station, Volta {vy:.1f} y per node")
    add("3.2", "events per node-year",
        E["events_per_station"] / ey,
        (per_station / vy) if np.isfinite(per_station) else np.nan, "",
        FREE, "record-length normalised; the directly comparable rate")

    d = _col(events, "duration_days") or _col(events, "duration")
    add("3.2", "event duration, mean", E["event_duration_mean"],
        _safe(lambda: d.mean()) if d is not None else np.nan, "days")
    add("3.2", "event duration, median", E["event_duration_median"],
        _safe(lambda: d.median()) if d is not None else np.nan, "days")
    add("3.2", "event duration, max", E["event_duration_max"],
        _safe(lambda: d.max()) if d is not None else np.nan, "days")

    s = _col(events, "severity")
    add("3.2", "event severity, mean", E["event_severity_mean"],
        _safe(lambda: s.mean()) if s is not None else np.nan)
    add("3.2", "event severity, median", E["event_severity_median"],
        _safe(lambda: s.median()) if s is not None else np.nan)

    it = _col(events, "intensity")
    add("3.2", "event intensity, mean", E["event_intensity_mean"],
        _safe(lambda: it.mean()) if it is not None else np.nan, "per day")
    add("3.2", "event intensity, max", E["event_intensity_max"],
        _safe(lambda: it.max()) if it is not None else np.nan, "per day")

    v = _col(events, "severity_hm3")
    add("3.2", "volumetric severity, mean", E["event_volsev_mean"],
        _safe(lambda: v.mean()) if v is not None else np.nan, "hm3")
    add("3.2", "volumetric severity, max", E["event_volsev_max"],
        _safe(lambda: v.max()) if v is not None else np.nan, "hm3")

    # --- 3.2 chains --------------------------------------------------------
    add("3.2", "propagation chains", E["n_chains"], n_ch, "",
        PERIOD + "+" + NETWORK)
    frac = (n_ch / n_ev) if (chains is not None and events is not None
                             and len(events)) else np.nan
    add("3.2", "chains as fraction of events", E["chains_as_fraction_of_events"],
        frac, "", FREE, "scale-free; a primary comparison")

    cd = _col(chains, "chain_duration")
    add("3.2", "chain duration, mean", E["chain_duration_mean"],
        _safe(lambda: cd.mean()) if cd is not None else np.nan, "days")
    add("3.2", "chain duration, median", E["chain_duration_median"],
        _safe(lambda: cd.median()) if cd is not None else np.nan, "days")
    add("3.2", "chain duration, max", E["chain_duration_max"],
        _safe(lambda: cd.max()) if cd is not None else np.nan, "days")

    cs = _col(chains, "chain_size")
    add("3.2", "chain size, mean", E["mean_chain_size"],
        _safe(lambda: cs.mean()) if cs is not None else np.nan, "nodes", NETWORK)
    add("3.2", "chain size, median", E["median_chain_size"],
        _safe(lambda: cs.median()) if cs is not None else np.nan, "nodes", NETWORK)
    add("3.2", "chain size, max", E["max_chain_size"],
        _safe(lambda: cs.max()) if cs is not None else np.nan, "nodes", NETWORK)

    pf = _col(chains, "propagation_fraction")
    add("3.2", "propagation fraction, mean", E["propagation_fraction_mean"],
        _safe(lambda: pf.mean()) if pf is not None else np.nan, "", FREE,
        ("set by topology when every chain has size two (pf = 1/N_avail); not a like-for-like comparison"
         if cs is not None and _safe(lambda: cs.max()) == 2 else "scale-free; a primary comparison"))

    lg = _col(chains, "lag_mean")
    add("3.2", "propagation lag, mean", E["lag_mean"],
        _safe(lambda: lg.mean()) if lg is not None else np.nan, "days")
    add("3.2", "propagation lag, median", E["lag_median"],
        _safe(lambda: lg.median()) if lg is not None else np.nan, "days")
    add("3.2", "propagation lag, IQR", E["lag_iqr"],
        _safe(lambda: lg.quantile(.75) - lg.quantile(.25)) if lg is not None
        else np.nan, "days")

    csev = _col(chains, "chain_severity")
    add("3.2", "chain severity, mean", E["chain_severity_mean"],
        _safe(lambda: csev.mean()) if csev is not None else np.nan)
    add("3.2", "chain severity, median", E["chain_severity_median"],
        _safe(lambda: csev.median()) if csev is not None else np.nan)

    cvol = _col(chains, "chain_severity_hm3")
    add("3.2", "chain volumetric severity, mean", E["chain_volsev_mean"],
        _safe(lambda: cvol.mean()) if cvol is not None else np.nan, "hm3")

    # --- null model, S4 ----------------------------------------------------
    def nullval(stat, col):
        if null is None or "statistic" not in getattr(null, "columns", []):
            return np.nan
        hit = null[null["statistic"].astype(str).str.contains(stat, case=False,
                                                              na=False)]
        return _safe(lambda: hit.iloc[0][col]) if len(hit) and col in null.columns \
            else np.nan

    add("S4", "connected fraction of origin events", E["connected_fraction"],
        nullval("connected", "observed"), "", FREE, "scale-free")
    add("S4", "null expectation", E["null_connected_fraction"],
        nullval("connected", "null_mean"), "")
    add("S4", "p, connected fraction", E["null_p_connected_fraction"],
        nullval("connected", "p_value"), "")
    add("S4", "p, mean chain size", E["null_p_mean_chain_size"],
        nullval("chain_size", "p_value"), "")
    add("S4", "valid upstream-downstream pairs", E["total_valid_pairs"],
        len(pairs) if pairs is not None else np.nan, "", PERIOD + "+" + NETWORK)

    # --- S7 unconnected and non-propagating --------------------------------
    oc, uc = supp.get("origin_contrast"), supp.get("upstream_contrast")
    def cget(df, flag_col, flag, col):
        if df is None: return np.nan
        r = df[df[flag_col].astype(str).str.lower() == str(flag).lower()]
        return float(r[col].iloc[0]) if len(r) else np.nan
    if oc is not None:
        n_un, n_co = cget(oc, "connected", False, "n"), cget(oc, "connected", True, "n")
    add("S7", "unconnected origin events, fraction", E["unconnected_fraction"],
        n_un / (n_un + n_co) if oc is not None else np.nan, "", FREE,
        "against all origin events; see opportunities for the fair denominator")
    add("S7", "isolated vs connected severity, ratio",
        E["connected_severity_mean"] / E["isolated_severity_mean"],
        cget(oc, "connected", True, "mean_severity") / cget(oc, "connected", False, "mean_severity")
        if oc is not None else np.nan, "", FREE, "direction and size of the contrast")
    if uc is not None:
        n_np, n_p = cget(uc, "accepted", False, "n"), cget(uc, "accepted", True, "n")
    add("S7", "non-propagating upstream, fraction", E["s7_nonpropagating_fraction"],
        n_np / (n_np + n_p) if uc is not None else np.nan, "", FREE, "scale-free")
    add("S7", "non-propagating duration, mean", E["nonprop_duration_mean"],
        cget(uc, "accepted", False, "mean_duration"), "days")
    add("S7", "propagating duration, mean", E["prop_duration_mean"],
        cget(uc, "accepted", True, "mean_duration"), "days")
    add("S7", "non-propagating severity, mean", E["nonprop_severity_mean"],
        cget(uc, "accepted", False, "mean_severity"))
    add("S7", "propagating severity, mean", E["prop_severity_mean"],
        cget(uc, "accepted", True, "mean_severity"))
    add("S7", "propagating / non-propagating severity",
        E["prop_severity_mean"] / E["nonprop_severity_mean"],
        cget(uc, "accepted", True, "mean_severity") / cget(uc, "accepted", False, "mean_severity")
        if uc is not None else np.nan, "", FREE, "how strongly the matcher favours substantial events")

    # --- S1.4 overlap relaxation -------------------------------------------
    s14 = supp.get("s14")
    def g(G, col):
        if s14 is None: return np.nan
        r = s14[s14.gap_tolerance_days == G]
        return float(r[col].iloc[0]) if len(r) else np.nan
    add("S1.4", "extra pairs at G=5 d, fraction", E["s1_4_extra_pairs_fraction"],
        g(5, "pct_more_than_accepted") / 100, "", FREE)
    add("S1.4", "connected fraction after relaxation", E["s1_4_connected_after"],
        g(5, "connected_fraction_after"), "")
    add("S1.4", "median severity, gap-admitted (G=45)", E["s1_4_median_sev_gap"],
        g(45, "median_severity_new"))
    add("S1.4", "median severity, overlap-accepted",
        E["s1_4_median_sev_accepted"], g(45, "median_severity_accepted"))
    add("S1.4", "accepted / gap-admitted severity", E["s1_4_median_sev_accepted"] / E["s1_4_median_sev_gap"],
        g(45, "median_severity_accepted") / g(45, "median_severity_new"), "", FREE,
        "published: overlap rule filters weak, coincidental pairs")

    # --- S8 independent catalogue ------------------------------------------
    add("S8", "chains overlapping an independent catalogue",
        E["s8_overlap_fraction"], np.nan, "", FREE,
        "Volta has no national catalogue of equivalent standing; see S8")

    # --- S9 ----------------------------------------------------------------
    s9 = supp.get("s9")
    add("S9", "stations flagged on homogeneity", E["s9_excluded"],
        int(s9["flag"].astype(bool).sum()) if s9 is not None else np.nan, "", NETWORK)
    add("S9", "fraction of network flagged", E["s9_excluded"] / (E["n_stations"] + E["s9_excluded"]),
        float(s9["flag"].astype(bool).mean()) if s9 is not None else np.nan, "", FREE,
        "published: 12 of 45 candidate stations removed")

    df = pd.DataFrame(rows)
    num = pd.to_numeric(df["ebro"], errors="coerce")
    vol = pd.to_numeric(df["volta"], errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        df["ratio"] = np.where((num != 0) & num.notna() & vol.notna(),
                               vol / num, np.nan)
    is_str = df["volta"].apply(lambda x: isinstance(x, str))
    df["status"] = np.where(vol.notna() | is_str, "computed", "not computed")
    return df


def format_parity_report(tab: pd.DataFrame, width: int = 96) -> str:
    """Readable fixed-width report, grouped by section."""
    out, done = [], 0
    out.append("=" * width)
    out.append("  PARITY AGAINST Crespillo Lopez et al. (2026), J. Hydrol. 680, 136544")
    out.append("=" * width)
    for sec, grp in tab.groupby("section", sort=False):
        out.append(f"\n[{sec}]")
        for _, r in grp.iterrows():
            e = r["ebro"]; v = r["volta"]

            def show(x):
                """Format a cell. Integers must not fall through to '--'."""
                if isinstance(x, str):
                    return x, True
                if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
                    return f"{int(x):,}", True
                if isinstance(x, (float, np.floating)) and np.isfinite(x):
                    s = f"{float(x):,.3f}".rstrip("0").rstrip(".")
                    return (s if s not in ("", "-") else "0"), True
                return "--", False

            es, _ = show(e)
            vs, ok = show(v)
            done += int(ok)
            rt = r["ratio"]
            rs = (f"  x{rt:,.2f}"
                  if ok and isinstance(rt, (float, np.floating)) and np.isfinite(rt)
                  else "")
            flag = f"  [{r['scale']}]" if r["scale"] else ""
            out.append(f"  {r['quantity'][:44]:<44} {es:>12} | {vs:>12}{rs}{flag}")
            if r["note"]:
                out.append(f"      {r['note']}")
    n = len(tab)
    out.append("\n" + "-" * width)
    out.append(f"  {done} of {n} quantities computed; {n - done} awaiting pipeline output.")
    out.append("  [period]  differs legitimately with record length per station")
    out.append("  [network] differs legitimately with network size, 33 gauges vs n nodes")
    out.append("  Rows without a flag are scale-free and are the real comparisons.")
    out.append("=" * width)
    return "\n".join(out)
