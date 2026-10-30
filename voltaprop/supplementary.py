"""
Supplementary analyses S1-S9, matching the published Supporting Information of
Crespillo Lopez, Vicente-Serrano and Gimeno (2026), J. Hydrol. 680, 136544.

Section correspondence, with the published Ebro result quoted so each Volta
run can be read against it:

  S1   Event-level sensitivity to the minimum overlap M (1, 3, 5, 7, 10, 15 d)
       Ebro: Spearman rho > 0.97 for M = 1 and 3; classification flips < 2.2%
  S1.4 Connectivity excluded by requiring overlap rather than precedence
       Ebro: +126 pairs (6.1%) at a 5-day gap tolerance; connected fraction
       56.6% -> 58.7%; median severity of gap-based candidates 4.2 against
       20.8 for accepted matches
  S2   Case-study chains under varying M
  S3   Lag versus along-channel distance                 [analysis.py]
       Ebro: Spearman rho = 0.14, p = 0.16 at station-pair level
  S4   Network-permutation null test, 1,999 permutations [analysis.py]
       Ebro: connected fraction 56.6% vs 50.3 +/- 3.1%, p = 0.020; mean chain
       size 4.95 vs 4.28 +/- 0.31, p = 0.012; total pairs 2,075 vs 1,578 +/-
       204, p = 0.004; two further statistics at the 84th and 88th percentiles
  S5   Detection sensitivity: pooling gap (0, 5, 7, 10, 15) x minimum duration
       (3, 5, 7, 10), 20 combinations, plus the per-station bias check
       Ebro: 1,834 to 5,155 events (-30% to +96% of 2,625); severity rankings
       rho = 0.83-1.00 for 19 of 20 combinations
  S6   Meteorological context of the case studies, and imputation sensitivity
  S7   Events that do not propagate                      [analysis.py]
       Ebro: 602 of 1,558 upstream candidates (38.6%) never accepted
  S8   Validation against an independent drought catalogue
       Ebro: 413 of 525 chains (78.7%) overlap a documented national drought
  S9   Stations excluded at the homogeneity screening stage, and their
       proximity to reservoirs
       Ebro: 12 of 45 excluded; median distance to nearest reservoir 5.0 km
       against 12.6 km retained, Mann-Whitney one-sided p = 0.005

Author: I. Quist (NUIST)
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .events import (
    EventConfig,
    detect_raw_events,
    filter_events,
    pool_events,
)
from .propagation import (
    PropagationConfig,
    build_node_arrays,
    run_matching_vectorised,
)

# Published Ebro values, carried here so every Volta table can be reported
# alongside its counterpart rather than in isolation.
EBRO_REFERENCE = {
    # ----------------------------------------------------------------------
    # Every quantity the published study reports, transcribed from
    # Crespillo Lopez et al. (2026), J. Hydrol. 680, 136544, and its
    # supplementary document. These are the parity targets: each Volta
    # result is printed beside its Ebro counterpart so a reader can see at
    # once which behaviours transfer and which do not.
    #
    # Transcribed and checked against the PDF and mmc1.docx on 12 Oct 2026.
    # The Ebro study period is 1961-2020 (60 years) against the Volta's
    # 1979-2024 (46 years); per-station rates are the comparable quantities,
    # not raw counts.
    # ----------------------------------------------------------------------

    # --- study frame -------------------------------------------------------
    "period": (1961, 2020),
    "n_stations": 33,
    "n_origin_stations": 12,

    # --- 3.1 parameter sensitivity ----------------------------------------
    "n_param_combinations": 750,          # W in [0,120] step 5 x M in [1,30]
    "rsi_W_all_metrics": 0.000,           # insensitive to W once M applied
    "rsi_M_below_0p10_metrics": 5,        # of 6
    "selected_W": 45,
    "selected_M": 5,
    "s1_spearman_at_M_le_5": 0.97,        # lower bound
    "s1_reclassified_fraction": 0.02,     # upper bound
    "s5_event_range": (2009, 3661),       # vs 2625 default; main text
    "s5_n_combinations": 20,
    "s5_spearman_severity": (0.83, 1.00), # 19 of 20 combinations
    "s5_spearman_worst": (0.30, 0.46),    # zero pooling, min duration >= 7 d

    # --- 3.2 event catalogue ----------------------------------------------
    "n_events": 2625,
    "events_per_station": 79.5,
    "event_duration_mean": 33.2,          # days
    "event_duration_median": 19.0,
    "event_duration_max": 613.0,
    "event_severity_mean": 12.95,
    "event_severity_median": 3.94,
    "event_intensity_mean": 0.293,        # per day
    "event_intensity_max": 3.47,
    "event_volsev_mean": 5.51,            # hm3
    "event_volsev_max": 362.26,

    # --- 3.2 chains --------------------------------------------------------
    "n_chains": 525,
    "chains_as_fraction_of_events": 0.20,
    "chain_duration_mean": 157.3,         # days
    "chain_duration_median": 109.0,
    "chain_duration_max": 692.0,
    "mean_chain_size": 4.95,              # text reports 5.0 stations
    "median_chain_size": 3.0,
    "max_chain_size": 24,
    "propagation_fraction_mean": 0.418,
    "propagation_fraction_range": (0.03, 1.00),
    "lag_mean": -33.37,                   # days, negative = upstream first
    "lag_median": -21.50,
    "lag_iqr": 34.80,
    "lag_max_abs": 399.0,
    "chain_severity_mean": 188.73,
    "chain_severity_median": 86.02,
    "chain_volsev_mean": 62.56,           # hm3
    "chain_volsev_max": 900.0,            # "exceeding 900"

    # --- null model, S4 ----------------------------------------------------
    "connected_fraction": 0.566,
    "null_connected_fraction": 0.503,
    "null_connected_sd": 0.031,
    "null_p_connected_fraction": 0.020,
    "null_p_mean_chain_size": 0.012,
    "total_valid_pairs": 2075,

    # --- unconnected and non-propagating, S7 -------------------------------
    "n_origin_events_with_upstream": 928,
    "n_origin_events_unconnected": 403,
    "unconnected_fraction": 0.434,
    "isolated_severity_mean": 11.2,
    "connected_severity_mean": 13.9,
    "isolated_duration_mean": 32.7,
    "connected_duration_mean": 35.1,
    "n_upstream_candidates": 1558,
    "n_upstream_nonpropagating": 602,
    "s7_nonpropagating_fraction": 0.386,
    "nonprop_duration_mean": 19.3,
    "prop_duration_mean": 54.3,
    "nonprop_severity_mean": 5.6,
    "prop_severity_mean": 24.1,

    # --- overlap relaxation, S1.4 ------------------------------------------
    "s1_4_extra_pairs_at_5d": 126,
    "s1_4_extra_pairs_fraction": 0.061,
    "s1_4_connected_after": 0.587,
    "s1_4_median_sev_gap": 4.2,
    "s1_4_median_sev_accepted": 20.8,

    # --- independent catalogue validation, S8 ------------------------------
    "s8_overlap_fraction": 0.787,         # 413 of 525 chains
    "s8_n_chains_overlapping": 413,
    "s8_n_national_events": 25,
    "s8_calendar_baseline": 0.428,
    "s8_reverse_fraction": 1.00,

    # --- homogeneity screening, S9 -----------------------------------------
    "s9_excluded": 12,
}


# ===========================================================================
# S1 -- event-level sensitivity to the minimum overlap parameter M
# ===========================================================================
def _chain_sizes_for_M(
    node_arr: Dict[int, dict],
    connectivity: Dict[int, List[int]],
    M: int,
    W: int = 45,
) -> pd.Series:
    """Chain size per origin event at a given M. Index: (origin_node, event_id)."""
    cfg = PropagationConfig(W_days=W, min_overlap_days=M, lag_max=0)
    pairs = run_matching_vectorised(node_arr, connectivity, cfg)

    # Every origin event that *could* have matched, so unmatched events enter
    # the comparison as zero rather than vanishing from the index.
    origins = [n for n, ch in connectivity.items() if ch and n in node_arr]
    index = pd.MultiIndex.from_tuples(
        [(n, int(e)) for n in origins for e in node_arr[n]["eid"]],
        names=["origin_node", "origin_event_id"],
    )
    sizes = pd.Series(0, index=index, dtype=int)

    if pairs:
        p = pd.DataFrame(
            pairs,
            columns=["origin_node", "origin_event_id", "upstream_node",
                     "upstream_event_id", "overlap_days", "lag_days"],
        )
        p = p[p["lag_days"] <= 0]
        if not p.empty:
            counted = (
                p.groupby(["origin_node", "origin_event_id"])["upstream_node"]
                .nunique()
            )
            sizes.loc[counted.index] = counted.values

    return sizes


def s1_event_level_m_sensitivity(
    events: pd.DataFrame,
    connectivity: Dict[int, List[int]],
    M_values: Sequence[int] = (1, 3, 5, 7, 10, 15),
    W: int = 45,
    reference_M: int = 5,
    node_col: str = "node_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Table S1 and Figure S1.

    Returns (summary, per_event). `summary` carries one row per M with the
    connected-event percentage, the proportion of events flipping between
    isolated and propagated relative to M = 5, the Spearman rank correlation of
    chain size against M = 5, and the mean change in chain size.

    The flip rate and the rank correlation are the two numbers that matter: the
    first says how many events change category, the second says whether the
    ordering of events by spatial reach survives. The Ebro reported rho > 0.97
    and flips below 2.2% for M <= 5.
    """
    node_arr = build_node_arrays(events, node_col)
    sizes = {M: _chain_sizes_for_M(node_arr, connectivity, M, W) for M in M_values}

    if reference_M not in sizes:
        sizes[reference_M] = _chain_sizes_for_M(node_arr, connectivity, reference_M, W)
    ref = sizes[reference_M]

    rows = []
    for M in M_values:
        s = sizes[M].reindex(ref.index).fillna(0)
        connected = s > 0
        ref_connected = ref > 0
        flips = (connected != ref_connected).mean()

        rho = np.nan
        if s.nunique() > 1 and ref.nunique() > 1:
            rho = float(stats.spearmanr(s.to_numpy(), ref.to_numpy()).statistic)

        rows.append(
            {
                "M_days": M,
                "n_origin_events": int(len(s)),
                "connected_events_pct": round(float(connected.mean() * 100), 2),
                "flip_vs_reference_pct": round(float(flips * 100), 2),
                "spearman_rho_vs_reference": round(rho, 4) if np.isfinite(rho) else np.nan,
                "mean_delta_chain": round(float((s - ref).mean()), 4),
                "mean_chain_size": round(float(s[connected].mean()), 3)
                if connected.any() else 0.0,
                "is_reference": M == reference_M,
            }
        )

    per_event = pd.DataFrame({f"M{M}": sizes[M] for M in M_values}).reset_index()
    return pd.DataFrame(rows), per_event


# ===========================================================================
# S1.4 -- connectivity excluded by requiring temporal overlap
# ===========================================================================
def s1_4_gap_tolerance_analysis(
    events: pd.DataFrame,
    connectivity: Dict[int, List[int]],
    gap_tolerances: Sequence[int] = (0, 5, 10, 15, 30, 45),
    W: int = 45,
    M: int = 5,
    node_col: str = "node_id",
) -> pd.DataFrame:
    """
    Table S3 of the published SI.

    The matcher requires real overlap rather than mere precedence. This asks
    what that excludes: for every (origin event, upstream node) pair with no
    accepted match, find the best zero-overlap candidate inside the search
    window, meaning an upstream event that ended shortly *before* the origin
    began, and report how many additional connections each gap tolerance would
    admit and how severe those events are.

    The Ebro finding was that relaxation admits weak signals: median severity
    of gap-based candidates was 4.2 against 20.8 for accepted matches, roughly
    a fivefold difference, across every tolerance tested. Reproducing that
    contrast is what justifies the overlap criterion as a conservative filter
    rather than an arbitrary one.
    """
    node_arr = build_node_arrays(events, node_col)
    cfg = PropagationConfig(W_days=W, min_overlap_days=M, lag_max=0)

    accepted = run_matching_vectorised(node_arr, connectivity, cfg)
    acc = pd.DataFrame(
        accepted,
        columns=["origin_node", "origin_event_id", "upstream_node",
                 "upstream_event_id", "overlap_days", "lag_days"],
    )
    acc = acc[acc["lag_days"] <= 0]
    accepted_keys = set(zip(acc["origin_node"], acc["origin_event_id"], acc["upstream_node"]))

    sev_lookup = events.set_index([node_col, "event_id"])["severity"]
    median_sev_accepted = float(
        sev_lookup.loc[
            list(zip(acc["upstream_node"], acc["upstream_event_id"]))
        ].median()
    ) if not acc.empty else np.nan

    origins = [n for n, ch in connectivity.items() if ch and n in node_arr]
    total_origin_events = sum(len(node_arr[n]["eid"]) for n in origins)
    base_connected = acc.groupby(["origin_node", "origin_event_id"]).ngroups

    # Collect, for every unaccepted (origin event, upstream node), the smallest
    # positive gap candidate: upstream_end < origin_start, within the window.
    candidates = []
    for origin in origins:
        o = node_arr[origin]
        for i in range(len(o["start"])):
            s_o = o["start"][i]
            e_id = int(o["eid"][i])
            w_start = s_o - W
            for up in [u for u in connectivity[origin] if u in node_arr]:
                if (origin, e_id, up) in accepted_keys:
                    continue
                u = node_arr[up]
                m = (u["end"] < s_o) & (u["end"] >= w_start)
                if not m.any():
                    continue
                gaps = s_o - u["end"][m]          # days between end and onset
                j = int(np.argmin(gaps))
                candidates.append(
                    {
                        "origin_node": origin,
                        "origin_event_id": e_id,
                        "upstream_node": up,
                        "upstream_event_id": int(u["eid"][m][j]),
                        "gap_days": int(gaps[j]),
                        "upstream_severity": float(u["sev"][m][j]),
                    }
                )

    cand = pd.DataFrame(candidates)

    rows = []
    for tol in gap_tolerances:
        if cand.empty:
            rows.append({"gap_tolerance_days": tol, "new_pairs": 0})
            continue
        sel = cand[cand["gap_days"] <= tol]
        newly = (
            pd.concat(
                [
                    acc[["origin_node", "origin_event_id"]],
                    sel[["origin_node", "origin_event_id"]],
                ]
            )
            .drop_duplicates()
            .shape[0]
        )
        rows.append(
            {
                "gap_tolerance_days": tol,
                "new_pairs": int(len(sel)),
                "pct_more_than_accepted": round(
                    float(100 * len(sel) / len(acc)), 2
                ) if len(acc) else np.nan,
                "connected_fraction_before": round(
                    float(base_connected / total_origin_events), 4
                ) if total_origin_events else np.nan,
                "connected_fraction_after": round(
                    float(newly / total_origin_events), 4
                ) if total_origin_events else np.nan,
                "median_severity_new": round(float(sel["upstream_severity"].median()), 3)
                if not sel.empty else np.nan,
                "median_severity_accepted": round(median_sev_accepted, 3),
                "severity_ratio": round(
                    float(median_sev_accepted / sel["upstream_severity"].median()), 2
                ) if not sel.empty and sel["upstream_severity"].median() > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows)


# ===========================================================================
# S2 -- case-study chains under varying M
# ===========================================================================
def s2_case_study_m_sensitivity(
    case_chain_ids: Sequence[str],
    events: pd.DataFrame,
    connectivity: Dict[int, List[int]],
    M_values: Sequence[int] = (1, 3, 5, 7, 10, 15),
    W: int = 45,
    node_col: str = "node_id",
) -> pd.DataFrame:
    """
    Figure S2: how each case study's chain membership changes with M.

    Reports, per case and per M, the chain size and which upstream nodes are
    retained, so the figure can show stability directly. The Ebro result was
    that large-scale events kept the same contributors across all M while
    spatially confined events lost connections as M rose.
    """
    node_arr = build_node_arrays(events, node_col)
    targets = [
        (int(c.rsplit("_", 1)[0]), int(c.rsplit("_", 1)[1])) for c in case_chain_ids
    ]

    rows = []
    for M in M_values:
        cfg = PropagationConfig(W_days=W, min_overlap_days=M, lag_max=0)
        pairs = run_matching_vectorised(node_arr, connectivity, cfg)
        p = pd.DataFrame(
            pairs,
            columns=["origin_node", "origin_event_id", "upstream_node",
                     "upstream_event_id", "overlap_days", "lag_days"],
        )
        p = p[p["lag_days"] <= 0]

        for (onode, oev), cid in zip(targets, case_chain_ids):
            sel = p[(p["origin_node"] == onode) & (p["origin_event_id"] == oev)]
            members = sorted(sel["upstream_node"].unique().tolist())
            rows.append(
                {
                    "chain_id": cid,
                    "M_days": M,
                    "chain_size": len(members) + 1,
                    "upstream_members": ",".join(map(str, members)),
                    "is_default": M == 5,
                }
            )

    out = pd.DataFrame(rows)
    # Jaccard similarity of membership against the default configuration.
    ref = {
        r.chain_id: set(r.upstream_members.split(",")) - {""}
        for r in out[out["is_default"]].itertuples(index=False)
    }
    out["jaccard_vs_default"] = [
        round(
            len(set(r.upstream_members.split(",")) - {""} & ref.get(r.chain_id, set()))
            / max(len(set(r.upstream_members.split(",")) - {""} | ref.get(r.chain_id, set())), 1),
            4,
        )
        for r in out.itertuples(index=False)
    ]
    return out


# ===========================================================================
# S5 -- detection sensitivity with the per-station bias check
# ===========================================================================
def s5_detection_sensitivity(
    ssi_df: pd.DataFrame,
    q_df: Optional[pd.DataFrame] = None,
    q_p10: Optional[pd.DataFrame] = None,
    pool_gaps: Sequence[int] = (0, 5, 7, 10, 15),
    min_durations: Sequence[int] = (3, 5, 7, 10),
    threshold: float = -1.28,
    default_gap: int = 10,
    default_duration: int = 5,
    node_col: str = "node_id",
    progress: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Figure S5 and Table S2, including the station-level homogeneity check that
    the published SI calls the question of practical relevance.

    The catalogue size moving with the thresholds is mechanical and expected.
    What matters is whether a given choice affects some nodes more than others,
    tested by the Spearman correlation across nodes between per-node event
    count (and, separately, per-node mean severity) and the values under the
    published configuration. The Ebro kept rho = 0.83-1.00 for 19 of 20
    combinations on severity, degrading only at zero pooling with a strict
    duration filter.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kw):
            return x

    prepared = {}
    for nid, sub in ssi_df.groupby(node_col):
        # unbroken calendar, so runs cannot jump gaps in the record
        sub = (sub.assign(date=pd.to_datetime(sub["date"]).dt.normalize())
                  .drop_duplicates("date").set_index("date").sort_index())
        sub = sub.reindex(pd.date_range(sub.index.min(), sub.index.max(), freq="D"))
        prepared[nid] = (
            sub["SSI"].reset_index(drop=True),
            pd.DatetimeIndex(sub.index),
            sub["SSI"].notna().to_numpy(),
        )

    combos = [(g, d) for g in pool_gaps for d in min_durations]
    iterator = tqdm(combos, desc="S5 detection sensitivity") if progress else combos

    per_node_count, per_node_sev, summary_rows = {}, {}, []

    for gap, dur in iterator:
        counts, sevs, durations, total = {}, {}, [], 0
        for nid, (ssi, dates, valid) in prepared.items():
            raw = detect_raw_events(ssi, threshold)
            pooled = pool_events(raw, dates, gap, valid, True)
            kept = filter_events(pooled, dates, dur)

            node_sev = []
            for s, e in kept:
                seg = ssi.iloc[s : e + 1]
                d = seg[seg < threshold]
                node_sev.append(float(np.sum(np.abs(threshold - d))))
                durations.append((dates[e] - dates[s]).days + 1)

            counts[nid] = len(kept)
            sevs[nid] = float(np.mean(node_sev)) if node_sev else 0.0
            total += len(kept)

        key = (gap, dur)
        per_node_count[key] = pd.Series(counts)
        per_node_sev[key] = pd.Series(sevs)
        summary_rows.append(
            {
                "pool_gap": gap,
                "min_duration": dur,
                "n_events": total,
                "mean_duration": round(float(np.mean(durations)), 2) if durations else np.nan,
                "is_default": gap == default_gap and dur == default_duration,
            }
        )

    summary = pd.DataFrame(summary_rows)
    base = (default_gap, default_duration)

    if base in per_node_count:
        n_default = int(summary.loc[summary["is_default"], "n_events"].iloc[0])
        summary["pct_change_vs_default"] = (
            100 * (summary["n_events"] - n_default) / n_default
        ).round(2)

        rc, rs = [], []
        for _, r in summary.iterrows():
            k = (int(r["pool_gap"]), int(r["min_duration"]))
            a, b = per_node_count[k], per_node_count[base]
            rc.append(
                round(float(stats.spearmanr(a.reindex(b.index), b).statistic), 4)
                if a.nunique() > 1 and b.nunique() > 1 else np.nan
            )
            a, b = per_node_sev[k], per_node_sev[base]
            rs.append(
                round(float(stats.spearmanr(a.reindex(b.index), b).statistic), 4)
                if a.nunique() > 1 and b.nunique() > 1 else np.nan
            )
        summary["rho_count_vs_default"] = rc
        summary["rho_severity_vs_default"] = rs

    per_node = pd.concat(
        {f"gap{g}_dur{d}": per_node_count[(g, d)] for g, d in combos}, axis=1
    ).reset_index(names=node_col)

    return summary, per_node


# ===========================================================================
# S6 -- meteorological context, and imputation sensitivity
# ===========================================================================
def s6_meteorological_context(
    case_chain_ids: Sequence[str],
    chains: pd.DataFrame,
    ssi_df: pd.DataFrame,
    events: pd.DataFrame,
    spei: pd.DataFrame,
    nodes: pd.DataFrame,
    node_col: str = "node_id",
) -> pd.DataFrame:
    """
    Figure S6: daily SSI at each case-study origin against SPEI at short
    (3-month) and long (12-month) accumulation, plus the comparison between the
    network-wide chain window and the origin node's own detected events.

    `spei` is expected in long form with columns [node_id, date, SPEI_3,
    SPEI_12], sampled at each node's coordinates. See
    docs/DATASET_REQUIREMENTS.md item 2.2 for which product to use; the Africa
    5 km product stops in 2016, so SPEIbase or a CHIRPS-derived index is needed
    for full coverage.

    The Ebro result worth reproducing: in Case 2 the origin was locally *wet*
    before onset and recorded six separate short events while the network-wide
    chain stayed continuously connected for 427 days. That mismatch is the
    strongest single argument for a network-aware framework over single-station
    monitoring, so test for it explicitly here.
    """
    rows = []
    for cid in case_chain_ids:
        row = chains.loc[chains["chain_id"] == cid]
        if row.empty:
            continue
        row = row.iloc[0]
        onode = int(row["origin_node"])
        cs, ce = pd.Timestamp(row["chain_start"]), pd.Timestamp(row["chain_end"])

        own = events[
            (events[node_col] == onode)
            & (events["start_date"] <= ce)
            & (events["end_date"] >= cs)
        ]

        sp = spei[(spei[node_col] == onode) & spei["date"].between(cs, ce)]
        onset = pd.Timestamp(row["origin_start"])
        pre = spei[
            (spei[node_col] == onode)
            & spei["date"].between(onset - pd.Timedelta(days=45), onset)
        ]

        rows.append(
            {
                "chain_id": cid,
                "origin_node": onode,
                "subbasin": nodes.set_index("node_id").loc[onode].get("subbasin"),
                "country": nodes.set_index("node_id").loc[onode].get("country"),
                "chain_start": cs,
                "chain_end": ce,
                "chain_duration": int(row["chain_duration"]),
                "n_own_events_in_window": int(len(own)),
                "own_event_days": int(own["duration"].sum()) if not own.empty else 0,
                "own_coverage_of_window": round(
                    float(own["duration"].sum() / row["chain_duration"]), 4
                ) if not own.empty and row["chain_duration"] else 0.0,
                "SPEI3_at_onset": round(float(pre["SPEI_3"].mean()), 3)
                if "SPEI_3" in pre and not pre.empty else np.nan,
                "SPEI12_at_onset": round(float(pre["SPEI_12"].mean()), 3)
                if "SPEI_12" in pre and not pre.empty else np.nan,
                "SPEI3_min_in_window": round(float(sp["SPEI_3"].min()), 3)
                if "SPEI_3" in sp and not sp.empty else np.nan,
                "locally_wet_at_onset": bool(pre["SPEI_3"].mean() > 0)
                if "SPEI_3" in pre and not pre.empty else None,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        # The diagnostic the Ebro Case 2 exemplifies: the chain is continuous
        # while the origin node's own record is fragmented and locally wet.
        out["network_adds_value"] = (
            (out["n_own_events_in_window"] > 1)
            | (out["own_coverage_of_window"] < 0.6)
            | (out["locally_wet_at_onset"] == True)   # noqa: E712
        )
    return out


def s6_imputation_sensitivity(
    node_id: int,
    q_observed: pd.Series,
    imputed_mask: pd.Series,
    chain_start,
    chain_end,
    ssi_cfg=None,
    event_cfg: EventConfig = EventConfig(),
) -> pd.DataFrame:
    """
    The second part of S6: recompute SSI with imputed days masked out of both
    the distribution fitting and the index, re-run detection, and report how
    far event timing and severity move.

    The Ebro reported onset unchanged, termination shifted by one day, duration
    -1.1% and peak date identical, with the severity drop attributable to the
    arithmetic removal of drought days rather than to index bias. Report the
    same quantities so the comparison is direct.

    For a GloFAS-primary Volta run this applies only to the observed series
    used in benchmarking, since the reanalysis has no imputed days. Say so in
    the supplementary rather than omitting the section.
    """
    from .ssi import SSIConfig, compute_ssi_node

    cfg = ssi_cfg or SSIConfig()

    def catalogue(series: pd.Series) -> pd.DataFrame:
        ssi, _ = compute_ssi_node(series, cfg)
        dates = pd.DatetimeIndex(series.index)
        s = ssi.reset_index(drop=True)
        raw = detect_raw_events(s, event_cfg.threshold_ssi)
        pooled = pool_events(raw, dates, event_cfg.pool_gap_days,
                             s.notna().to_numpy(), True)
        kept = filter_events(pooled, dates, event_cfg.min_duration_days)
        recs = []
        for a, b in kept:
            seg = s.iloc[a : b + 1]
            d = seg[seg < event_cfg.threshold_ssi]
            recs.append(
                {
                    "start_date": dates[a],
                    "end_date": dates[b],
                    "duration": (dates[b] - dates[a]).days + 1,
                    "severity": float(np.sum(np.abs(event_cfg.threshold_ssi - d))),
                    "peak_date": dates[seg.idxmin()],
                }
            )
        return pd.DataFrame(recs)

    full = catalogue(q_observed)
    masked_series = q_observed.copy()
    masked_series[imputed_mask.reindex(masked_series.index).fillna(False)] = np.nan
    masked = catalogue(masked_series)

    cs, ce = pd.Timestamp(chain_start), pd.Timestamp(chain_end)
    f = full[(full["start_date"] <= ce) & (full["end_date"] >= cs)]
    m = masked[(masked["start_date"] <= ce) & (masked["end_date"] >= cs)]

    if f.empty or m.empty:
        return pd.DataFrame([{"node_id": node_id, "note": "no event in window"}])

    f, m = f.iloc[0], m.iloc[0]
    return pd.DataFrame(
        [
            {
                "node_id": node_id,
                "pct_imputed_in_window": round(
                    float(imputed_mask.loc[cs:ce].mean() * 100), 2
                ),
                "onset_shift_days": int((m["start_date"] - f["start_date"]).days),
                "termination_shift_days": int((m["end_date"] - f["end_date"]).days),
                "duration_pct_change": round(
                    float(100 * (m["duration"] - f["duration"]) / f["duration"]), 2
                ),
                "peak_date_identical": bool(m["peak_date"] == f["peak_date"]),
                "severity_full": round(float(f["severity"]), 2),
                "severity_masked": round(float(m["severity"]), 2),
            }
        ]
    )


# ===========================================================================
# S8 -- validation against an independent drought catalogue
# ===========================================================================
def s8_independent_catalogue_validation(
    chains: pd.DataFrame,
    catalogue: pd.DataFrame,
    study_start: str,
    study_end: str,
    cat_start_col: str = "start_date",
    cat_end_col: str = "end_date",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Figure S8: do the reconstructed chains coincide with independently
    documented droughts?

    `catalogue` must carry one row per documented event with start and end
    dates. For the Volta there is no equivalent of the Spanish national
    catalogue used in the published study; see docs/DATASET_REQUIREMENTS.md
    Part 4. Use a CHIRPS-derived chronology, or EM-DAT entries for Ghana,
    Burkina Faso and Togo, and state in the text which was used and why it is
    only partially independent.

    Two outputs. The first flags each chain. The second is the decisive
    comparison the published study made: whether chains overlapping a
    documented drought are *more severe* than those that do not. Raw overlap
    percentage means little on its own, because the documented events may cover
    much of the calendar; the routine therefore also reports that calendar
    coverage as the baseline to beat. The Ebro reported 78.7% overlap against
    42.8% calendar coverage, with overlapping chains 3.4 times more severe.
    """
    cat = catalogue.copy()
    cat[cat_start_col] = pd.to_datetime(cat[cat_start_col])
    cat[cat_end_col] = pd.to_datetime(cat[cat_end_col])
    s0, s1 = pd.Timestamp(study_start), pd.Timestamp(study_end)
    cat = cat[(cat[cat_end_col] >= s0) & (cat[cat_start_col] <= s1)]

    ch = chains.copy()
    ch["chain_start"] = pd.to_datetime(ch["chain_start"])
    ch["chain_end"] = pd.to_datetime(ch["chain_end"])

    cs = cat[cat_start_col].values.astype("datetime64[D]").astype(np.int64)
    ce = cat[cat_end_col].values.astype("datetime64[D]").astype(np.int64)

    overlaps, which = [], []
    for a, b in zip(
        ch["chain_start"].values.astype("datetime64[D]").astype(np.int64),
        ch["chain_end"].values.astype("datetime64[D]").astype(np.int64),
    ):
        hit = (np.minimum(b, ce) - np.maximum(a, cs)) >= 0
        overlaps.append(bool(hit.any()))
        which.append(",".join(map(str, np.nonzero(hit)[0].tolist())))

    ch["overlaps_documented"] = overlaps
    ch["documented_event_idx"] = which

    # Calendar coverage of the documented events, the baseline to beat.
    days = np.zeros((s1 - s0).days + 1, dtype=bool)
    for a, b in zip(cat[cat_start_col], cat[cat_end_col]):
        i0 = max((a - s0).days, 0)
        i1 = min((b - s0).days, len(days) - 1)
        if i1 >= i0:
            days[i0 : i1 + 1] = True
    coverage = float(days.mean())

    g = ch.groupby("overlaps_documented")
    summary = pd.DataFrame(
        [
            {
                "n_chains": int(len(ch)),
                "n_documented_events": int(len(cat)),
                "pct_chains_overlapping": round(float(100 * ch["overlaps_documented"].mean()), 2),
                "calendar_coverage_pct": round(100 * coverage, 2),
                "excess_over_calendar_pp": round(
                    float(100 * ch["overlaps_documented"].mean() - 100 * coverage), 2
                ),
                "mean_severity_overlapping": round(
                    float(g.get_group(True)["chain_severity"].mean()), 2
                ) if True in g.groups else np.nan,
                "mean_severity_non_overlapping": round(
                    float(g.get_group(False)["chain_severity"].mean()), 2
                ) if False in g.groups else np.nan,
                "severity_ratio": round(
                    float(
                        g.get_group(True)["chain_severity"].mean()
                        / g.get_group(False)["chain_severity"].mean()
                    ), 2
                ) if True in g.groups and False in g.groups
                and g.get_group(False)["chain_severity"].mean() > 0 else np.nan,
                "mean_size_overlapping": round(
                    float(g.get_group(True)["chain_size"].mean()), 2
                ) if True in g.groups else np.nan,
                "mean_size_non_overlapping": round(
                    float(g.get_group(False)["chain_size"].mean()), 2
                ) if False in g.groups else np.nan,
                "mannwhitney_p_severity": float(
                    stats.mannwhitneyu(
                        g.get_group(True)["chain_severity"],
                        g.get_group(False)["chain_severity"],
                        alternative="greater",
                    ).pvalue
                ) if True in g.groups and False in g.groups
                and len(g.get_group(True)) > 5 and len(g.get_group(False)) > 5 else np.nan,
            }
        ]
    )
    return ch, summary


# ===========================================================================
# S9 -- homogeneity screening and reservoir proximity
# ===========================================================================
def s9_homogeneity_screening(
    ssi_df: pd.DataFrame,
    node_col: str = "node_id",
    split_fraction: float = 0.5,
    ks_alpha: float = 0.01,
    ks_effect_threshold: float = 0.15,
    mean_shift_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    The second screening stage of Section 2.1: nodes whose standardised series
    show abrupt regime shifts inconsistent with a stationary index.

    Three diagnostics per node, all reported rather than only the verdict, so
    the decision is auditable: a two-sample Kolmogorov-Smirnov comparison
    between the first and second halves of the record; the shift in mean SSI
    between halves, which should be near zero by construction; and a Pettitt
    change-point statistic on the annual mean.

    A note on why the flag is not driven by the KS p-value. Over a 46-year
    daily record each half holds roughly 8,000 values, and at that sample size
    the KS test detects differences far too small to matter: a statistic of
    0.05, meaning the two empirical distributions never diverge by more than
    five percentage points, returns p < 1e-10. Screening on significance alone
    would therefore flag almost every node, including perfectly stationary
    ones. The flag consequently requires an *effect size* large enough to
    matter (KS statistic above `ks_effect_threshold`, or a mean shift above
    `mean_shift_threshold`) and uses the p-value only as a secondary
    condition. Both thresholds are judgement calls and both are reported, so a
    reader can apply their own.

    The published study inspected flagged stations individually and excluded
    them only where no hydrological explanation could be found. Do the same.
    This routine proposes candidates; it does not decide. Record the decision
    and its reason for each flagged node in the supplementary table.
    """
    rows = []
    for nid, sub in ssi_df.groupby(node_col):
        s = sub.sort_values("date")
        v = s["SSI"].dropna()
        if len(v) < 500:
            rows.append({node_col: nid, "n_valid": int(len(v)), "flag": True,
                         "reason": "insufficient valid days"})
            continue

        cut = int(len(v) * split_fraction)
        a, b = v.iloc[:cut].to_numpy(), v.iloc[cut:].to_numpy()
        ks = stats.ks_2samp(a, b)
        shift = float(b.mean() - a.mean())

        annual = s.dropna(subset=["SSI"]).groupby(
            pd.DatetimeIndex(s.dropna(subset=["SSI"])["date"]).year
        )["SSI"].mean()
        pettitt_k, pettitt_year = _pettitt(annual.to_numpy())

        # Effect size first, significance only as a secondary condition.
        big_dist = (ks.statistic >= ks_effect_threshold) and (ks.pvalue < ks_alpha)
        big_mean = abs(shift) > mean_shift_threshold
        flag = bool(big_dist or big_mean)

        if big_mean and big_dist:
            reason = "distribution and mean shift"
        elif big_mean:
            reason = "mean shift"
        elif big_dist:
            reason = "distribution shift"
        elif ks.pvalue < ks_alpha:
            reason = "significant but negligible (effect size below threshold)"
        else:
            reason = ""

        rows.append(
            {
                node_col: nid,
                "n_valid": int(len(v)),
                "mean_ssi": round(float(v.mean()), 4),
                "std_ssi": round(float(v.std()), 4),
                "ks_statistic": round(float(ks.statistic), 4),
                "ks_pvalue": float(ks.pvalue),
                "ks_effect_exceeds_threshold": bool(ks.statistic >= ks_effect_threshold),
                "mean_shift_halves": round(shift, 4),
                "pettitt_K": round(float(pettitt_k), 2),
                "pettitt_year": int(annual.index[pettitt_year])
                if 0 <= pettitt_year < len(annual) else np.nan,
                "flag": flag,
                "reason": reason,
                "decision": "",      # fill in by inspection: retain or exclude
                "decision_reason": "",
            }
        )
    return pd.DataFrame(rows)


def _pettitt(x: np.ndarray) -> Tuple[float, int]:
    """Pettitt change-point statistic; returns (K, index of change point)."""
    n = len(x)
    if n < 5:
        return 0.0, -1
    u = np.zeros(n)
    for t in range(n):
        u[t] = np.sum(np.sign(x[t] - x)) if t else 0
    u = np.cumsum(u)
    k = int(np.argmax(np.abs(u)))
    return float(np.abs(u[k])), k


def s9_reservoir_proximity_test(
    screening: pd.DataFrame,
    nodes: pd.DataFrame,
    reservoirs: pd.DataFrame,
    node_col: str = "node_id",
    excluded_col: str = "flag",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Are the flagged nodes closer to reservoirs than the retained ones?

    The published study found median 5.0 km against 12.6 km with a one-sided
    Mann-Whitney p = 0.005, and read it as observational support for a
    regulation origin of the irregularities, while noting two caveats worth
    repeating: proximity is not causation, and excluded stations also drained
    smaller catchments, which are more sensitive to a given volume of
    regulation. Report the catchment-area contrast alongside the distance
    contrast so the second caveat is visible rather than buried.
    """
    from .network import haversine_km

    d = []
    for n in nodes.itertuples(index=False):
        dists = [
            haversine_km(n.lon, n.lat, r.lon, r.lat)
            for r in reservoirs.itertuples(index=False)
        ]
        d.append(
            {
                node_col: int(getattr(n, node_col)),
                "dist_nearest_reservoir_km": round(float(min(dists)), 3) if dists else np.nan,
                "nearest_reservoir": reservoirs.iloc[int(np.argmin(dists))]["name"]
                if dists else None,
                "uparea_km2": float(getattr(n, "uparea_km2", np.nan)),
            }
        )
    table = pd.DataFrame(d).merge(
        screening[[node_col, excluded_col]], on=node_col, how="left"
    )

    exc = table.loc[table[excluded_col] == True, "dist_nearest_reservoir_km"].dropna()   # noqa: E712
    ret = table.loc[table[excluded_col] == False, "dist_nearest_reservoir_km"].dropna()  # noqa: E712

    if len(exc) < 3 or len(ret) < 3:
        return table, pd.DataFrame([{"note": "too few nodes in one group",
                                     "n_flagged": len(exc), "n_retained": len(ret)}])

    u, p = stats.mannwhitneyu(exc, ret, alternative="less")
    ea = table.loc[table[excluded_col] == True, "uparea_km2"].dropna()    # noqa: E712
    ra = table.loc[table[excluded_col] == False, "uparea_km2"].dropna()   # noqa: E712

    summary = pd.DataFrame(
        [
            {
                "n_flagged": int(len(exc)),
                "n_retained": int(len(ret)),
                "median_dist_flagged_km": round(float(exc.median()), 2),
                "median_dist_retained_km": round(float(ret.median()), 2),
                "pct_flagged_within_5km": round(float(100 * (exc <= 5).mean()), 1),
                "pct_retained_within_5km": round(float(100 * (ret <= 5).mean()), 1),
                "mannwhitney_U": float(u),
                "p_value_one_sided": float(p),
                "median_uparea_flagged_km2": round(float(ea.median()), 1) if not ea.empty else np.nan,
                "median_uparea_retained_km2": round(float(ra.median()), 1) if not ra.empty else np.nan,
            }
        ]
    )
    return table, summary
