"""
================================================================================
Surrogate / network-permutation null test for drought propagation chains
--------------------------------------------------------------------------------
Response analysis for Reviewer #3, comment 2 (synchrony vs. genuine propagation),
cross-referenced from R4.2 (Fig. S3) and R2.5.

QUESTION
--------
Could the reconstructed propagation chains simply reflect regionally synchronous
drought events (a common climatic forcing hitting many stations at once) rather
than genuine upstream-to-downstream transmission, given that the matching
criterion (temporal overlap + lag <= 0) does not by itself rule out coincidence?

NULL MODEL
----------
Network-label permutation: the 33 real event catalogues (exact dates, durations,
severities -- i.e. all real regional climatic synchrony is preserved intact) are
randomly re-assigned to the 33 station-ID "slots" of the FIXED river network
topology (upstream connectivity untouched). The propagation-matching algorithm
(Section 2.1.3, reproduced here verbatim in vectorised form) is re-run on each
random relabelling. This isolates exactly one thing: whether the TRUE
correspondence between a station's drought history and its true position in the
network produces more/larger chains than a random correspondence would, given
the same catalogues and the same topology.

This module provides:
  - build_station_arrays()      : fast per-station numpy arrays from the event catalogue
  - run_matching_vectorized()   : the propagation-matching algorithm (Sec. 2.1.3),
                                   numerically validated to be pair-for-pair IDENTICAL
                                   to the original notebook implementation
  - summarise()                 : the three primary test statistics for one run
  - permutation_null()          : runs N permutations and returns the null distribution

VALIDATION (see S4_permutation_null_test/01_permutation_null_test.py / the executed notebook for the printed proof)
  1. Vectorized algorithm reproduces the *exact* pair set of the original,
     unvectorized notebook code (set-equality over 2,075 tuples).
  2. The identity permutation (no shuffling) reproduces the published headline
     numbers exactly: 12 origin stations, 928 origin events, 525 chains, 56.6%.
================================================================================
"""
import numpy as np
import pandas as pd

W_DAYS      = 45   # search-window length (days), as in the published algorithm
MIN_OVERLAP = 5    # minimum real overlap (days) to qualify as a candidate
LAG_MAX     = 0    # retain only pairs with lag_days <= this value


# ------------------------------------------------------------------ data prep --
def build_station_arrays(events_df: pd.DataFrame) -> dict:
    """station_id -> dict of numpy arrays (ordinal-day ints, severity, event_id).

    Using integer ordinal days (instead of pandas Timestamps) is what makes the
    vectorised matching ~150x faster than the original per-row implementation,
    while giving bit-for-bit identical results (validated in S4_permutation_null_test/01_permutation_null_test.py).
    """
    out = {}
    for sid, grp in events_df.groupby("station_id"):
        grp = grp.sort_values("event_id")
        out[sid] = dict(
            start=grp["start_date"].values.astype("datetime64[D]").astype(np.int64),
            end=grp["end_date"].values.astype("datetime64[D]").astype(np.int64),
            sev=grp["severity"].values.astype(np.float64),
            eid=grp["event_id"].values.astype(np.int64),
        )
    return out


def parse_connectivity(conn_csv_path: str) -> dict:
    """station_id -> list of upstream station_ids, from the semicolon-separated
    upstream_connectivity.csv (unordered, no topology beyond membership)."""
    conn_raw = pd.read_csv(conn_csv_path, sep=";", dtype=str)
    conn_raw.columns = conn_raw.columns.str.strip()
    conn_raw["station_id"] = conn_raw["station_id"].str.strip().astype(int)
    conn_raw["upstream_chain"] = conn_raw["upstream_chain"].str.strip()

    def parse_chain(s):
        if pd.isna(s) or s == "":
            return []
        return [int(x.strip()) for x in s.split(",") if x.strip()]

    return {row["station_id"]: parse_chain(row["upstream_chain"])
            for _, row in conn_raw.iterrows()}


# ------------------------------------------------------- core matching engine --
def run_matching_vectorized(station_arr: dict, connectivity: dict) -> list:
    """Reproduces Section 2.1.3 exactly: asymmetric search window -> real-overlap
    filter -> deterministic 4-level tie-break -> lag. Returns ALL best-candidate
    matches (lag filter is applied separately by the caller, exactly as in the
    original notebook, where tie-breaking happens BEFORE the lag <= 0 filter).

    Returns a list of tuples: (origin_station, origin_event_id, upstream_station,
    upstream_event_id, lag_days).
    """
    origin_stations = [sid for sid, chain in connectivity.items()
                        if chain and sid in station_arr]
    all_pairs = []
    for origin_sid in origin_stations:
        upstream_sids = [s for s in connectivity[origin_sid] if s in station_arr]
        o = station_arr[origin_sid]
        n_o = len(o["start"])
        for i in range(n_o):
            s_o, e_o = o["start"][i], o["end"][i]
            w_start, w_end = s_o - W_DAYS, s_o
            for up_sid in upstream_sids:
                u = station_arr[up_sid]
                mask = (u["end"] >= w_start) & (u["start"] <= w_end)
                if not mask.any():
                    continue
                s2, e2, sv2, id2 = u["start"][mask], u["end"][mask], u["sev"][mask], u["eid"][mask]
                overlap = np.minimum(e_o, e2) - np.maximum(s_o, s2) + 1
                overlap = np.maximum(overlap, 0)
                keep = overlap >= MIN_OVERLAP
                if not keep.any():
                    continue
                ov, s3, sv3, id3 = overlap[keep], s2[keep], sv2[keep], id2[keep]
                absdiff = np.abs(s3 - s_o)
                # lexsort primary key is evaluated LAST:
                # overlap desc, severity desc, |start diff| asc, event_id asc
                order = np.lexsort((id3, absdiff, -sv3, -ov))
                b = order[0]
                lag = int(s3[b] - s_o)
                all_pairs.append((origin_sid, int(o["eid"][i]), up_sid, int(id3[b]), lag))
    return all_pairs


# --------------------------------------------------------------- summary stats --
def summarise(all_pairs: list, station_arr: dict, connectivity: dict) -> dict:
    """Computes the three primary, denominator-normalised test statistics from
    one matching run: connected_fraction, mean_chain_size (Nc = Nup + 1),
    mean_propagation_fraction (fp = Nup / Navail). Mirrors the definitions in
    Table 2 / Fig. 5 of the manuscript exactly.
    """
    if not all_pairs:
        return dict(n_origin_events=0, n_chains=0, connected_fraction=np.nan,
                     mean_chain_size=np.nan, mean_propagation_fraction=np.nan,
                     n_valid_pairs=0)

    df = pd.DataFrame(all_pairs, columns=["origin_station", "origin_event_id",
                                           "upstream_station", "upstream_event_id", "lag_days"])
    df_neg = df[df["lag_days"] <= LAG_MAX]

    origin_stations = [sid for sid, chain in connectivity.items()
                        if chain and sid in station_arr]
    n_origin_events = sum(len(station_arr[s]["eid"]) for s in origin_stations)

    if df_neg.empty:
        return dict(n_origin_events=n_origin_events, n_chains=0, connected_fraction=0.0,
                     mean_chain_size=np.nan, mean_propagation_fraction=np.nan,
                     n_valid_pairs=0)

    chain_size = (df_neg.groupby(["origin_station", "origin_event_id"])["upstream_station"]
                  .nunique().rename("chain_size").reset_index())

    upstream_available = {sid: sum(1 for s in connectivity[sid] if s in station_arr)
                           for sid in origin_stations}
    chain_size["n_upstream_with_events"] = chain_size["origin_station"].map(upstream_available)
    chain_size["propagation_fraction"] = chain_size["chain_size"] / chain_size["n_upstream_with_events"]
    chain_size["chain_size_Nc"] = chain_size["chain_size"] + 1   # Nc = Nup + 1 (Table 2)

    n_chains = len(chain_size)
    return dict(
        n_origin_events=n_origin_events,
        n_chains=n_chains,
        connected_fraction=100.0 * n_chains / n_origin_events,
        mean_chain_size=chain_size["chain_size_Nc"].mean(),
        mean_propagation_fraction=chain_size["propagation_fraction"].mean(),
        n_valid_pairs=len(df_neg),
    )


# ------------------------------------------------------------- permutation ------
def permutation_null(events_df: pd.DataFrame, connectivity: dict,
                      n_perm: int = 1999, seed: int = 20260713) -> pd.DataFrame:
    """Runs the network-label permutation null test.

    In each permutation, the 33 real event catalogues are randomly re-assigned
    to the 33 station-ID network slots (topology in `connectivity` unchanged).
    Returns a DataFrame with one row per permutation and the summary statistics.
    """
    station_arr = build_station_arrays(events_df)
    ids = np.array(sorted(station_arr.keys()))
    rng = np.random.default_rng(seed)

    rows = []
    for p in range(n_perm):
        perm_ids = rng.permutation(ids)
        permuted_arr = {ids[k]: station_arr[perm_ids[k]] for k in range(len(ids))}
        pairs = run_matching_vectorized(permuted_arr, connectivity)
        stats = summarise(pairs, permuted_arr, connectivity)
        stats["perm_id"] = p
        rows.append(stats)
    return pd.DataFrame(rows)
