"""
Parameter sensitivity, null-model testing and lag diagnostics.

Ports of `06_ssi_propagation_sensitivity.ipynb` and of the peer-review
additions in `revision_analyses/` (S3 lag versus hydraulic distance, S4
network-permutation null test, S5 detection sensitivity, S7 non-propagating
events). These were added to the Ebro paper during review. Including them in
the Volta submission from the outset is the cheapest insurance available:
each one answers a question a hydrology reviewer will otherwise
raise, and answering it pre-emptively is worth a review round.

Author: I. Quist (NUIST)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .propagation import (
    PropagationConfig,
    build_node_arrays,
    run_matching_vectorised,
)


# ===========================================================================
# S1 / Section 3.1 -- parameter sensitivity over the (W, M) grid
# ===========================================================================
def sensitivity_grid(
    events: pd.DataFrame,
    connectivity: Dict[int, List[int]],
    W_values: Sequence[int] = tuple(range(0, 125, 5)),
    M_values: Sequence[int] = tuple(range(1, 31)),
    node_col: str = "node_id",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Recompute the propagation metrics across the full parameter space.
    With the Ebro grid this is 25 x 30 = 750 combinations; the vectorised
    matcher makes it tractable.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kw):
            return x

    node_arr = build_node_arrays(events, node_col)
    available = {n: sum(1 for u in ch if u in node_arr) for n, ch in connectivity.items()}

    rows = []
    combos = [(w, m) for w in W_values for m in M_values]
    iterator = tqdm(combos, desc="Sensitivity grid") if progress else combos

    for W, M in iterator:
        cfg = PropagationConfig(W_days=W, min_overlap_days=M, lag_max=0)
        pairs = run_matching_vectorised(node_arr, connectivity, cfg)
        if not pairs:
            rows.append({"W": W, "M": M, "n_pairs": 0, "n_origin_matched": 0,
                         "mean_propagation_fraction": np.nan,
                         "median_propagation_fraction": np.nan,
                         "median_lag": np.nan, "mean_chain_size": np.nan})
            continue

        p = pd.DataFrame(
            pairs,
            columns=["origin_node", "origin_event_id", "upstream_node",
                     "upstream_event_id", "overlap_days", "lag_days"],
        )
        p = p[p["lag_days"] <= 0]
        if p.empty:
            rows.append({"W": W, "M": M, "n_pairs": 0, "n_origin_matched": 0,
                         "mean_propagation_fraction": np.nan,
                         "median_propagation_fraction": np.nan,
                         "median_lag": np.nan, "mean_chain_size": np.nan})
            continue

        cs = (
            p.groupby(["origin_node", "origin_event_id"])["upstream_node"]
            .nunique()
            .rename("chain_size")
            .reset_index()
        )
        cs["avail"] = cs["origin_node"].map(available)
        cs["pf"] = cs["chain_size"] / cs["avail"]

        rows.append(
            {
                "W": W,
                "M": M,
                "n_pairs": int(len(p)),
                "n_origin_matched": int(len(cs)),
                "mean_propagation_fraction": round(float(cs["pf"].mean()), 4),
                "median_propagation_fraction": round(float(cs["pf"].median()), 4),
                "median_lag": float(p["lag_days"].median()),
                "mean_chain_size": round(float(cs["chain_size"].mean()), 3),
            }
        )

    return pd.DataFrame(rows)


def relative_sensitivity_index(grid: pd.DataFrame, metric: str, parameter: str) -> float:
    """
    RSI after Hamby (1994): the spread a metric shows across the range of one
    parameter, averaged over the levels of the other, normalised by the mean.
    Values near zero mean the metric is effectively insensitive.
    """
    other = "M" if parameter == "W" else "W"
    spreads = []
    for _, g in grid.groupby(other):
        v = g[metric].dropna()
        if len(v) > 1 and v.mean() != 0:
            spreads.append((v.max() - v.min()) / abs(v.mean()))
    return round(float(np.mean(spreads)), 4) if spreads else np.nan


def kneedle_elbow(x: Sequence[float], y: Sequence[float],
                  curve: str = "concave", direction: str = "increasing") -> Optional[float]:
    """Formal elbow detection (Satopaa et al., 2011); returns None if `kneed` is absent."""
    try:
        from kneed import KneeLocator
    except ImportError:
        return None
    kl = KneeLocator(list(x), list(y), curve=curve, direction=direction)
    return kl.knee


# ===========================================================================
# S4 -- network-permutation null test
# ===========================================================================
def permutation_null_test(
    events: pd.DataFrame,
    connectivity: Dict[int, List[int]],
    cfg: PropagationConfig = PropagationConfig(),
    n_permutations: int = 1999,
    seed: int = 42,
    node_col: str = "node_id",
    progress: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Does the reconstructed connectivity exceed what regional climatic
    synchrony alone would produce?

    Each node's complete event catalogue is reassigned at random to another
    node, holding the network topology and all event timing fixed. Any
    co-occurrence driven purely by the fact that the whole basin dries at once
    survives the shuffle; structure that depends on which node sits upstream of
    which does not. Three statistics are tested: the connected fraction of
    origin events, the mean chain size, and the mean propagation fraction.

    This matters more in the Volta than in the Ebro. West African rainfall is
    dominated by the West African Monsoon, which is coherent over hundreds of
    kilometres, so the null expectation for synchrony is high. Without this
    test, a reviewer is entitled to assume the chains are monsoon climatology
    rather than river-network transmission.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kw):
            return x

    rng = np.random.default_rng(seed)

    def statistics(node_arr) -> Dict[str, float]:
        pairs = run_matching_vectorised(node_arr, connectivity, cfg)
        origins = [n for n, ch in connectivity.items() if ch and n in node_arr]
        total_origin_events = sum(len(node_arr[n]["eid"]) for n in origins)
        zero = {
            # three primary, denominator-normalised statistics
            "connected_fraction": 0.0,
            "mean_chain_size": 0.0,
            "mean_propagation_fraction": 0.0,
            # two secondary, non-normalised statistics
            "chain_count": 0.0,
            "total_valid_pairs": 0.0,
        }
        if not pairs or total_origin_events == 0:
            return zero
        p = pd.DataFrame(
            pairs,
            columns=["origin_node", "origin_event_id", "upstream_node",
                     "upstream_event_id", "overlap_days", "lag_days"],
        )
        p = p[p["lag_days"] <= cfg.lag_max]
        if p.empty:
            return zero
        cs = (
            p.groupby(["origin_node", "origin_event_id"])["upstream_node"]
            .nunique().rename("chain_size").reset_index()
        )
        avail = {n: sum(1 for u in connectivity[n] if u in node_arr) for n in origins}
        cs["pf"] = cs["chain_size"] / cs["origin_node"].map(avail)
        return {
            "connected_fraction": len(cs) / total_origin_events,
            "mean_chain_size": float(cs["chain_size"].mean()),
            "mean_propagation_fraction": float(cs["pf"].mean()),
            "chain_count": float(len(cs)),
            "total_valid_pairs": float(len(p)),
        }

    observed_arr = build_node_arrays(events, node_col)
    observed = statistics(observed_arr)

    node_ids = list(observed_arr.keys())
    null_rows = []
    iterator = range(n_permutations)
    if progress:
        iterator = tqdm(iterator, desc="Permutation null")

    for _ in iterator:
        shuffled_ids = rng.permutation(node_ids)
        permuted = {new: observed_arr[old] for new, old in zip(node_ids, shuffled_ids)}
        null_rows.append(statistics(permuted))

    null = pd.DataFrame(null_rows)

    PRIMARY = {"connected_fraction", "mean_chain_size", "mean_propagation_fraction"}
    summary = []
    for k, obs in observed.items():
        col = null[k]
        p_val = float((np.sum(col >= obs) + 1) / (len(col) + 1))
        summary.append(
            {
                "statistic": k,
                "kind": "primary (normalised)" if k in PRIMARY else "secondary (raw)",
                "observed": round(obs, 4),
                "null_mean": round(float(col.mean()), 4),
                "null_sd": round(float(col.std(ddof=1)), 4),
                "null_p95": round(float(col.quantile(0.95)), 4),
                # Where the observed value sits in the null distribution. The
                # published study reported the two non-significant statistics
                # this way (84th and 88th percentiles) rather than by p-value
                # alone, which is more informative than a bare "not
                # significant"; do the same.
                "observed_percentile_in_null": round(
                    float(100 * (col < obs).mean()), 1
                ),
                "p_value_one_sided": round(p_val, 5),
                "significant_at_0.05": bool(p_val < 0.05),
                "n_permutations": n_permutations,
            }
        )

    order = ["connected_fraction", "mean_chain_size", "mean_propagation_fraction",
             "chain_count", "total_valid_pairs"]
    out = pd.DataFrame(summary)
    out["_o"] = out["statistic"].map({k: i for i, k in enumerate(order)})
    return out.sort_values("_o").drop(columns="_o").reset_index(drop=True), null


# ===========================================================================
# S3 -- propagation lag versus along-channel hydraulic distance
# ===========================================================================
def lag_vs_distance(
    pairs: pd.DataFrame,
    nodes: pd.DataFrame,
    flowdir,
    transform,
) -> Tuple[pd.DataFrame, dict]:
    """
    Test whether propagation lag scales with hydraulic distance.

    The Ebro result was that it does not, which the authors read as evidence
    that catchment storage and regulation, rather than flow routing, control
    propagation timing. Reproducing the test in the Volta is essential: the
    basin's far larger scale and its very large storage term, Lake Volta above
    all, make the storage interpretation more plausible still, but that must
    be demonstrated rather than assumed.
    """
    from .network import along_channel_distance_km

    recs = []
    for r in pairs.itertuples(index=False):
        d = along_channel_distance_km(
            int(r.upstream_node), int(r.origin_node), nodes, flowdir, transform
        )
        recs.append(
            {
                "origin_node": int(r.origin_node),
                "upstream_node": int(r.upstream_node),
                "lag_days": int(r.lag_days),
                "distance_km": d,
                "overlap_days": int(r.overlap_days),
            }
        )

    df = pd.DataFrame(recs)
    m = df["distance_km"].notna() & df["lag_days"].notna()
    if m.sum() < 30:
        return df, {"note": "insufficient connected pairs for regression"}

    x = df.loc[m, "distance_km"].to_numpy()
    y = df.loc[m, "lag_days"].to_numpy()
    lr = stats.linregress(x, y)
    sp = stats.spearmanr(x, y)

    return df, {
        "n_pairs": int(m.sum()),
        "pearson_r": round(float(lr.rvalue), 4),
        "r_squared": round(float(lr.rvalue ** 2), 4),
        "slope_days_per_km": round(float(lr.slope), 6),
        "slope_p_value": float(lr.pvalue),
        "spearman_r": round(float(sp.statistic), 4),
        "spearman_p": float(sp.pvalue),
    }


# ===========================================================================
# S5 -- detection-parameter sensitivity
# ===========================================================================
def detection_sensitivity(
    ssi_df: pd.DataFrame,
    pool_gaps: Sequence[int] = (0, 5, 10, 15, 20, 30),
    min_durations: Sequence[int] = (1, 3, 5, 7, 10, 15),
    threshold: float = -1.28,
    node_col: str = "node_id",
) -> pd.DataFrame:
    """
    How far does the event catalogue move with the pooling gap and the minimum
    duration? The Ebro catalogue ranged from 2,009 to 3,661 events against
    2,625 at the default. Report the equivalent range, and also report whether
    the node-level severity ranking is preserved, because that ranking is what
    the propagation analysis actually consumes.
    """
    from .events import (
        EventConfig,
        detect_raw_events,
        filter_events,
        pool_events,
    )

    base_rank = None
    rows = []
    for gap in pool_gaps:
        for dur in min_durations:
            total, per_node = 0, {}
            for nid, sub in ssi_df.groupby(node_col):
                sub = sub.sort_values("date")
                ssi = sub["SSI"].reset_index(drop=True)
                dates = pd.DatetimeIndex(sub["date"])
                valid = ssi.notna().to_numpy()
                raw = detect_raw_events(ssi, threshold)
                pooled = pool_events(raw, dates, gap, valid, True)
                kept = filter_events(pooled, dates, dur)
                total += len(kept)
                per_node[nid] = sum(
                    float(np.sum(np.abs(threshold - ssi.iloc[s : e + 1][ssi.iloc[s : e + 1] < threshold])))
                    for s, e in kept
                )

            rank = pd.Series(per_node).rank(ascending=False)
            if gap == 10 and dur == 5:
                base_rank = rank
            rho = (
                round(float(stats.spearmanr(rank, base_rank).statistic), 4)
                if base_rank is not None and len(rank) == len(base_rank)
                else np.nan
            )
            rows.append(
                {"pool_gap": gap, "min_duration": dur, "n_events": total,
                 "severity_rank_rho_vs_default": rho}
            )

    out = pd.DataFrame(rows)
    # Recompute rho against the default now that it is known
    if base_rank is not None:
        out["is_default"] = (out["pool_gap"] == 10) & (out["min_duration"] == 5)
    return out


# ===========================================================================
# S7 -- non-propagating events
# ===========================================================================
def non_propagating_events(
    events: pd.DataFrame,
    pairs: pd.DataFrame,
    connectivity: Dict[int, List[int]],
    node_col: str = "node_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Two complementary populations the source paper examined during review:
    origin events with upstream connectivity that nevertheless matched nothing,
    and upstream events that were candidates somewhere but were never accepted.

    Comparing their duration and severity against the connected population
    tests whether the matcher is preferentially selecting substantial, sustained
    droughts as propagation drivers, which is the physically expected behaviour.
    """
    matched_origin = set(zip(pairs["origin_node"], pairs["origin_event_id"]))
    matched_upstream = set(zip(pairs["upstream_node"], pairs["upstream_event_id"]))

    has_upstream = {n for n, ch in connectivity.items() if ch}
    origin_pool = events[events[node_col].isin(has_upstream)].copy()
    origin_pool["connected"] = [
        (n, e) in matched_origin
        for n, e in zip(origin_pool[node_col], origin_pool["event_id"])
    ]

    # The published population (Section 3.2, "1,558 upstream events that were,
    # at some point, temporally compatible candidates"): upstream events whose
    # interval intersects the search window [s0 - W, s0] of at least one origin
    # event downstream. An earlier version used every event at any upstream
    # station, which inflates the never-accepted fraction.
    W = 45
    cand = set()
    ev_by_node = {n: g for n, g in events.groupby(node_col)}
    for o, ups in connectivity.items():
        if o not in ev_by_node:
            continue
        for s0 in pd.to_datetime(ev_by_node[o]["start_date"]):
            for u in ups:
                g = ev_by_node.get(u)
                if g is None:
                    continue
                us, ue = pd.to_datetime(g["start_date"]), pd.to_datetime(g["end_date"])
                hit = g[(us <= s0) & (ue >= s0 - pd.Timedelta(days=W))]
                cand.update(zip(hit[node_col], hit["event_id"]))
    up_pool = events[[(n, e) in cand for n, e in zip(events[node_col], events["event_id"])]].copy()
    up_pool["accepted"] = [
        (n, e) in matched_upstream
        for n, e in zip(up_pool[node_col], up_pool["event_id"])
    ]

    def contrast(df: pd.DataFrame, flag: str) -> pd.DataFrame:
        g = df.groupby(flag).agg(
            n=("event_id", "size"),
            mean_duration=("duration", "mean"),
            median_duration=("duration", "median"),
            mean_severity=("severity", "mean"),
            median_severity=("severity", "median"),
        ).round(3).reset_index()
        a = df.loc[df[flag], "severity"]
        b = df.loc[~df[flag], "severity"]
        if len(a) > 10 and len(b) > 10:
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            g["mannwhitney_p_severity"] = float(p)
        return g

    return contrast(origin_pool, "connected"), contrast(up_pool, "accepted")
