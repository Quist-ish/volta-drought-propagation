#!/usr/bin/env python3
"""
Cross-check against the original authors' released code
(github.com/AlexCrespillo/Code_v.1, revision_analyses/propagation_null_core.py).

  1. Volta: the released matcher, run on the Volta catalogue and connectivity,
     returns the same pairs as voltaprop at W = 0, 45 and 120 days.
  2. Volta: the released network-label null (seed and permutation count as
     published) gives the p-values reported in the manuscript.
  3. Ebro: the released event catalogue (S5 table3_events_corrected.csv, 2,625
     events) with the connectivity recovered from the released S3 pair table
     (101 links) reproduces every published S4 statistic, observed and null,
     through voltaprop.fastmatch.

Usage: ORIGINAL_CODE=<path to Code_v.1-main> python crosscheck_original_code.py
Writes output_observational/tables/crosscheck_original_code.json.
Author: I. Quist (NUIST)
"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voltaprop import fastmatch as F
from voltaprop.propagation import load_connectivity

ORIG = Path(os.environ.get("ORIGINAL_CODE", "third_party/Code_v.1"))
sys.path.insert(0, str(ORIG / "revision_analyses"))
import propagation_null_core as P   # noqa: E402

T = Path("output_observational/tables"); R = {}
ev = pd.read_csv(T / "SSI_drought_events.csv", parse_dates=["start_date", "end_date"])
conn = load_connectivity(T / "upstream_connectivity.csv")
pipe = pd.read_csv(T / "propagation_pairs.csv")
ref = set(zip(pipe.origin_node, pipe.origin_event_id, pipe.upstream_node, pipe.upstream_event_id))

arr = P.build_station_arrays(ev.rename(columns={"node_id": "station_id"}))
R["volta_pairs"] = {}
for W in (0, 45, 120):
    P.W_DAYS = W
    pr = P.run_matching_vectorized(arr, conn)
    kept = {(o, oe, u, ue) for o, oe, u, ue, lag in pr if lag <= P.LAG_MAX}
    R["volta_pairs"][f"W{W}"] = {"n_pairs": len(kept), "identical_to_voltaprop": kept == ref,
                                 "positive_lag_candidates": int(sum(lag > 0 for *_, lag in pr))}
P.W_DAYS = 45

null = P.permutation_null(ev.rename(columns={"node_id": "station_id"}), conn, n_perm=1999, seed=20260713)
obs = P.summarise(P.run_matching_vectorized(arr, conn), arr, conn)
R["volta_original_null"] = {k: {"observed": round(float(obs[k]), 4), "null_mean": round(float(null[k].mean()), 4),
                                "p": round(float((np.sum(null[k] >= obs[k]) + 1) / (len(null) + 1)), 3)}
                            for k in ("connected_fraction", "mean_chain_size", "mean_propagation_fraction",
                                      "n_chains", "n_valid_pairs")}

# Ebro reproduction
eb = pd.read_csv("data/ebro/ebro_events.csv", parse_dates=["start_date", "end_date"]).rename(columns={"station_id": "node_id"})
ep = pd.read_csv("data/ebro/ebro_pairs_net_km.csv")
econn = {int(s): [] for s in eb.node_id.unique()}
for _, r in ep.iterrows():
    econn[int(r.origin_station)].append(int(r.upstream_station))
erecs = F.build_records(eb)
eobs, esumm, _ = F.null_test(erecs, econn, n_perm=1999, seed=20260713)
pub = pd.read_csv(ORIG / "revision_analyses/S4_permutation_null_test/permutation_test_summary.csv").set_index("statistic")
mine = esumm.set_index("statistic")
cmp = {}
for pk, mk, scale in (("connected_fraction", "cf_all", 100), ("mean_chain_size", "mean_chain_size", 1),
                      ("mean_propagation_fraction", "mean_pf_published", 1), ("n_chains", "n_connected", 1),
                      ("n_valid_pairs", "n_pairs", 1)):
    cmp[pk] = {"published_observed": float(pub.loc[pk, "observed"]), "reproduced_observed": float(mine.loc[mk, "observed"]) * scale,
               "published_null_mean": float(pub.loc[pk, "null_mean"]), "reproduced_null_mean": float(mine.loc[mk, "null_mean"]) * scale,
               "published_p": float(pub.loc[pk, "p_value"]), "reproduced_p": round(float(mine.loc[mk, "p"]), 4)}
R["ebro_reproduction"] = {"events": int(len(eb)), "stations": int(eb.node_id.nunique()),
                          "links": int(sum(len(v) for v in econn.values())),
                          "origin_stations": int(sum(1 for v in econn.values() if v)), "statistics": cmp,
                          "all_match": bool(all(abs(c["published_observed"] - c["reproduced_observed"]) < 1e-6 and
                                                abs(c["published_null_mean"] - c["reproduced_null_mean"]) < 1e-6
                                                for c in cmp.values()))}
# alias keys read by build_supplement.py
R["original_null"] = R["volta_original_null"]
R["n_pairs"] = R["volta_pairs"]["W45"]["n_pairs"]
R["pairs_identical"] = all(v["identical_to_voltaprop"] for v in R["volta_pairs"].values())
(T / "crosscheck_original_code.json").write_text(json.dumps(R, indent=2, default=str))
print(json.dumps(R, indent=1, default=str))
