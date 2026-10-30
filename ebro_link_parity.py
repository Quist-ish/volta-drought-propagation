#!/usr/bin/env python3
"""
Released Ebro link table against the reconstruction from the released Ebro catalogue
(Supplementary Section S9).

The original study released a per-link table of matched-pair counts and median absolute lags
(third_party/Code_v.1/revision_analyses/S3_lag_vs_distance/pairwise_lag_distance.csv) and the
event catalogue with the 101 directed gauge links (data/ebro/). Matching the catalogue with the
published rules (voltaprop.fastmatch, which reproduces the published basin totals) gives the
reconstructed pairs; this script compares the two per link.

Writes output_observational/tables/r5_ebro_link_parity.csv and r5_ebro_link_parity.json.
Author: I. Quist (NUIST)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from voltaprop import fastmatch as F  # noqa: E402

T = ROOT / "output_observational/tables"
REL = ROOT / "third_party/Code_v.1/revision_analyses/S3_lag_vs_distance/pairwise_lag_distance.csv"


def main():
    eb = pd.read_csv(ROOT / "data/ebro/ebro_events.csv", parse_dates=["start_date", "end_date"]).rename(
        columns={"station_id": "node_id"})
    ep = pd.read_csv(ROOT / "data/ebro/ebro_pairs_net_km.csv")
    conn = {int(s): [] for s in eb.node_id.unique()}
    for r in ep.itertuples():
        conn[int(r.origin_station)].append(int(r.upstream_station))
    pairs, _ = F.match(F.build_records(eb), conn)
    rec = (pairs.assign(abs_lag=pairs.lag.abs())
           .groupby(["upstream_node", "origin_node"]).agg(n=("lag", "size"), med=("abs_lag", "median")).reset_index()
           .rename(columns={"upstream_node": "upstream_station", "origin_node": "origin_station"}))
    rel = pd.read_csv(REL)[["upstream_station", "origin_station", "n_events", "med_abs_lag"]]
    m = rel.merge(rec, on=["upstream_station", "origin_station"], how="left").fillna({"n": 0})
    m["n_diff"] = m.n_events - m.n
    m["lag_diff"] = (m.med - m.med_abs_lag).abs()
    m = m[["upstream_station", "origin_station", "n_events", "n", "med_abs_lag", "med", "n_diff", "lag_diff"]]
    m.to_csv(T / "r5_ebro_link_parity.csv", index=False)
    cdiff = m.n_diff != 0
    ldiff = m.lag_diff > 0.5
    out = dict(links=len(m), published_pairs=int(m.n_events.sum()), reconstructed_pairs=int(len(pairs)),
               links_count_differ=int(cdiff.sum()), count_diff_abs_median=float(m.n_diff[cdiff].abs().median()),
               count_diff_abs_max=float(m.n_diff.abs().max()), links_lag_differ=int(ldiff.sum()),
               lag_diff_median=float(m.lag_diff[ldiff].median()), lag_diff_max=float(m.lag_diff.max()),
               links_either=int((cdiff | ldiff).sum()))
    (T / "r5_ebro_link_parity.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
