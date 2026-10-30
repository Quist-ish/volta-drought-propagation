"""
Event-based drought propagation chains.

Vectorised port of `05b_ssi_propagation_vectorised.ipynb` (Crespillo Lopez
et al., 2026). The matching logic, the asymmetric search window, the real
overlap test, the four-level deterministic tie-break and the lag filter are
reproduced exactly; only the identifier column is renamed from `station_id`
to `node_id` to reflect that Volta nodes are virtual gauging points rather
than physical gauges.

The reference (row-wise) implementation is retained so that any Volta run can
assert bit-identical agreement between the two, exactly as the source
repository does. Keep `validate=True` for the production run reported in the
paper and switch it off only inside the sensitivity grid search.

Algorithm
---------
For every drought event at a node that has upstream neighbours (the origin
event), build the window [origin_start - W, origin_start]. Any upstream event
intersecting that window is a candidate. Candidates must share at least M days
of real overlap with the origin event. Where a single upstream node offers
several qualifying events, one is retained by ranking on overlap (desc),
severity (desc), absolute onset difference (asc) and event id (asc). The lag
is upstream_start - origin_start, and only pairs with lag <= 0 survive.

Author: I. Quist (NUIST)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class PropagationConfig:
    W_days: int = 45
    min_overlap_days: int = 5
    lag_max: int = 0


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------
def load_connectivity(path: str, sep: str = ";") -> Dict[int, List[int]]:
    """
    Read `upstream_connectivity.csv`: one row per node, with `upstream_chain`
    holding a comma-separated list of every hydraulically upstream node.
    """
    raw = pd.read_csv(path, sep=sep, dtype=str)
    raw.columns = raw.columns.str.strip()
    raw["node_id"] = raw["node_id"].str.strip().astype(int)
    raw["upstream_chain"] = raw["upstream_chain"].fillna("").str.strip()

    def parse(s: str) -> List[int]:
        return [int(x.strip()) for x in s.split(",") if x.strip()] if s else []

    return {r["node_id"]: parse(r["upstream_chain"]) for _, r in raw.iterrows()}


def build_node_arrays(events: pd.DataFrame, node_col: str = "node_id") -> Dict[int, dict]:
    """Per-node numpy arrays keyed on integer ordinal days, for fast matching."""
    out: Dict[int, dict] = {}
    for nid, grp in events.groupby(node_col):
        grp = grp.sort_values("event_id")
        out[nid] = dict(
            start=grp["start_date"].values.astype("datetime64[D]").astype(np.int64),
            end=grp["end_date"].values.astype("datetime64[D]").astype(np.int64),
            sev=grp["severity"].values.astype(np.float64),
            eid=grp["event_id"].values.astype(np.int64),
        )
    return out


# ---------------------------------------------------------------------------
# Vectorised matching
# ---------------------------------------------------------------------------
def run_matching_vectorised(
    node_arr: Dict[int, dict],
    connectivity: Dict[int, List[int]],
    cfg: PropagationConfig = PropagationConfig(),
) -> List[tuple]:
    """
    Best upstream candidate per (origin event, upstream node), before the lag
    filter. Returns tuples of
    (origin_node, origin_event_id, upstream_node, upstream_event_id,
     overlap_days, lag_days).
    """
    origin_nodes = [n for n, chain in connectivity.items() if chain and n in node_arr]
    pairs: List[tuple] = []

    for origin in origin_nodes:
        upstream_nodes = [u for u in connectivity[origin] if u in node_arr]
        o = node_arr[origin]

        for i in range(len(o["start"])):
            s_o, e_o = o["start"][i], o["end"][i]
            w_start, w_end = s_o - cfg.W_days, s_o

            for up in upstream_nodes:
                u = node_arr[up]
                mask = (u["end"] >= w_start) & (u["start"] <= w_end)
                if not mask.any():
                    continue

                s2, e2 = u["start"][mask], u["end"][mask]
                sv2, id2 = u["sev"][mask], u["eid"][mask]

                overlap = np.maximum(np.minimum(e_o, e2) - np.maximum(s_o, s2) + 1, 0)
                keep = overlap >= cfg.min_overlap_days
                if not keep.any():
                    continue

                ov, s3, sv3, id3 = overlap[keep], s2[keep], sv2[keep], id2[keep]
                absdiff = np.abs(s3 - s_o)
                # overlap desc, severity desc, |onset diff| asc, event_id asc
                b = np.lexsort((id3, absdiff, -sv3, -ov))[0]

                pairs.append(
                    (origin, int(o["eid"][i]), up, int(id3[b]), int(ov[b]), int(s3[b] - s_o))
                )

    return pairs


def run_matching_reference(
    events: pd.DataFrame,
    connectivity: Dict[int, List[int]],
    cfg: PropagationConfig = PropagationConfig(),
    node_col: str = "node_id",
) -> pd.DataFrame:
    """Row-wise reference implementation, kept solely to validate the above."""

    def real_overlap(s_o, e_o, s_u, e_u):
        return max(0, (min(e_o, e_u) - max(s_o, s_u)).days + 1)

    node_events = {n: g.reset_index(drop=True) for n, g in events.groupby(node_col)}
    W = pd.Timedelta(days=cfg.W_days)
    origin_nodes = [n for n, chain in connectivity.items() if chain and n in node_events]

    rows = []
    for origin in origin_nodes:
        upstream_nodes = [u for u in connectivity[origin] if u in node_events]
        for _, orig in node_events[origin].iterrows():
            s_o, e_o = orig["start_date"], orig["end_date"]
            w_start, w_end = s_o - W, s_o
            for up in upstream_nodes:
                ue = node_events[up]
                cand = ue[(ue["end_date"] >= w_start) & (ue["start_date"] <= w_end)].copy()
                if cand.empty:
                    continue
                cand["overlap_days"] = cand.apply(
                    lambda r: real_overlap(s_o, e_o, r["start_date"], r["end_date"]), axis=1
                )
                cand = cand[cand["overlap_days"] >= cfg.min_overlap_days]
                if cand.empty:
                    continue
                cand["abs_start_diff"] = (cand["start_date"] - s_o).abs().dt.days
                best = cand.sort_values(
                    ["overlap_days", "severity", "abs_start_diff", "event_id"],
                    ascending=[False, False, True, True],
                ).iloc[0]
                rows.append(
                    {
                        "origin_node": origin,
                        "origin_event_id": orig["event_id"],
                        "origin_start": s_o,
                        "origin_end": e_o,
                        "origin_duration": orig["duration"],
                        "origin_severity": orig["severity"],
                        "upstream_node": up,
                        "upstream_event_id": best["event_id"],
                        "upstream_start": best["start_date"],
                        "upstream_end": best["end_date"],
                        "upstream_duration": best["duration"],
                        "overlap_days": int(best["overlap_days"]),
                        "lag_days": (best["start_date"] - s_o).days,
                        "upstream_severity": best["severity"],
                    }
                )

    return (
        pd.DataFrame(rows)
        .sort_values(["origin_node", "origin_event_id", "lag_days", "upstream_node"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Pair table assembly
# ---------------------------------------------------------------------------
def build_pair_table(
    pairs: Sequence[tuple],
    events: pd.DataFrame,
    node_col: str = "node_id",
) -> pd.DataFrame:
    p = pd.DataFrame(
        pairs,
        columns=[
            "origin_node", "origin_event_id", "upstream_node",
            "upstream_event_id", "overlap_days", "lag_days",
        ],
    )
    ev = events.set_index([node_col, "event_id"])[
        ["start_date", "end_date", "duration", "severity"]
    ]
    o = ev.loc[list(zip(p["origin_node"], p["origin_event_id"]))].reset_index(drop=True)
    u = ev.loc[list(zip(p["upstream_node"], p["upstream_event_id"]))].reset_index(drop=True)

    out = pd.DataFrame(
        {
            "origin_node": p["origin_node"],
            "origin_event_id": p["origin_event_id"],
            "origin_start": o["start_date"],
            "origin_end": o["end_date"],
            "origin_duration": o["duration"],
            "origin_severity": o["severity"],
            "upstream_node": p["upstream_node"],
            "upstream_event_id": p["upstream_event_id"],
            "upstream_start": u["start_date"],
            "upstream_end": u["end_date"],
            "upstream_duration": u["duration"],
            "overlap_days": p["overlap_days"],
            "lag_days": p["lag_days"],
            "upstream_severity": u["severity"],
        }
    )
    return out.sort_values(
        ["origin_node", "origin_event_id", "lag_days", "upstream_node"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Chain metrics (Table 2 of the source paper)
# ---------------------------------------------------------------------------
def apply_lag_filter_and_chain_metrics(
    pair_table: pd.DataFrame,
    node_arr: Dict[int, dict],
    connectivity: Dict[int, List[int]],
    cfg: PropagationConfig = PropagationConfig(),
) -> pd.DataFrame:
    prop = pair_table[pair_table["lag_days"] <= cfg.lag_max].copy().reset_index(drop=True)

    chain_size = (
        prop.groupby(["origin_node", "origin_event_id"])["upstream_node"]
        .nunique()
        .rename("chain_size")
        .reset_index()
    )
    available = {
        n: sum(1 for u in chain if u in node_arr) for n, chain in connectivity.items()
    }
    chain_size["n_upstream_with_events"] = chain_size["origin_node"].map(available)
    chain_size["propagation_fraction"] = (
        chain_size["chain_size"] / chain_size["n_upstream_with_events"]
    ).round(4)

    prop = prop.merge(chain_size, on=["origin_node", "origin_event_id"], how="left")

    # Sanity checks, mirroring the source repository
    assert (prop["lag_days"] <= cfg.lag_max).all(), "lag filter failed"
    assert (prop["overlap_days"] >= cfg.min_overlap_days).all(), "overlap threshold violated"
    assert prop["propagation_fraction"].between(0, 1).all(), "propagation fraction out of [0,1]"
    assert not prop.duplicated(
        ["origin_node", "origin_event_id", "upstream_node"]
    ).any(), "duplicate (origin, event, upstream) triples"

    return prop


def build_chain_summary(
    prop: pd.DataFrame,
    events: pd.DataFrame,
    node_col: str = "node_id",
) -> pd.DataFrame:
    """One row per origin event. This is the primary analytical output."""
    prop = prop.copy()
    prop["chain_id"] = (
        prop["origin_node"].astype(str) + "_" + prop["origin_event_id"].astype(str)
    )

    hm3 = events[[node_col, "event_id", "severity_hm3"]]

    prop_hm3 = prop.merge(
        hm3.rename(
            columns={
                node_col: "upstream_node",
                "event_id": "upstream_event_id",
                "severity_hm3": "upstream_severity_hm3",
            }
        ),
        on=["upstream_node", "upstream_event_id"],
        how="left",
    )

    origin_fields = (
        prop.drop_duplicates("chain_id")[
            [
                "chain_id", "origin_node", "origin_event_id",
                "origin_start", "origin_end", "origin_severity",
                "chain_size", "n_upstream_with_events", "propagation_fraction",
            ]
        ]
        .merge(
            hm3.rename(
                columns={
                    node_col: "origin_node",
                    "event_id": "origin_event_id",
                    "severity_hm3": "origin_severity_hm3",
                }
            ),
            on=["origin_node", "origin_event_id"],
            how="left",
        )
    )

    agg = (
        prop_hm3.groupby("chain_id")
        .agg(
            chain_start=("upstream_start", "min"),
            upstream_end_max=("upstream_end", "max"),
            lag_mean=("lag_days", "mean"),
            lag_min=("lag_days", "min"),
            upstream_sev_sum=("upstream_severity", "sum"),
            upstream_hm3_sum=("upstream_severity_hm3", "sum"),
        )
        .reset_index()
    )

    chains = origin_fields.merge(agg, on="chain_id")
    chains["chain_end"] = chains[["upstream_end_max", "origin_end"]].max(axis=1)
    chains["chain_duration"] = (chains["chain_end"] - chains["chain_start"]).dt.days + 1
    chains["chain_size"] = chains["chain_size"] + 1          # include the origin node
    chains["chain_severity"] = (
        chains["upstream_sev_sum"] + chains["origin_severity"]
    ).round(4)
    chains["chain_severity_hm3"] = (
        chains["upstream_hm3_sum"] + chains["origin_severity_hm3"]
    ).round(4)
    chains["lag_mean"] = chains["lag_mean"].round(1)

    return (
        chains[
            [
                "chain_id", "origin_node", "origin_start", "origin_end",
                "chain_start", "chain_end", "chain_duration", "chain_size",
                "propagation_fraction", "lag_mean", "lag_min",
                "chain_severity", "chain_severity_hm3",
            ]
        ]
        .sort_values(["origin_node", "chain_start"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# End-to-end convenience wrapper
# ---------------------------------------------------------------------------
def run_propagation(
    events: pd.DataFrame,
    connectivity: Dict[int, List[int]],
    cfg: PropagationConfig = PropagationConfig(),
    node_col: str = "node_id",
    validate: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (pair_table_all, pair_table_lag_filtered, chain_summary).

    With `validate=True` the vectorised result is asserted identical to the
    row-wise reference, reproducing the check made in the source repository.
    """
    node_arr = build_node_arrays(events, node_col)
    pairs = run_matching_vectorised(node_arr, connectivity, cfg)
    pair_table = build_pair_table(pairs, events, node_col)

    if validate:
        ref = run_matching_reference(events, connectivity, cfg, node_col)
        pd.testing.assert_frame_equal(pair_table, ref)

    prop = apply_lag_filter_and_chain_metrics(pair_table, node_arr, connectivity, cfg)
    chains = build_chain_summary(prop, events, node_col)
    return pair_table, prop, chains
