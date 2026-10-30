#!/usr/bin/env python3
"""
Monte Carlo precision of the availability-aware null on the observed network:
19,999 permutations at M = 3, 5 and 7 (seed 2026), and the spread of the 1,999-permutation
p-value across ten seeds at M = 5. Writes output_observational/tables/r6_pvalue_precision.json.
Author: I. Quist (NUIST)
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from voltaprop import fastmatch as F
from voltaprop.propagation import load_connectivity
T = Path("output_observational/tables")
ev = pd.read_csv(T / "SSI_drought_events.csv", parse_dates=["start_date", "end_date"])
conn = load_connectivity(T / "upstream_connectivity.csv")
ssi = pd.read_csv(T / "SSI_daily.csv", parse_dates=["date"])
recs = F.build_records(ev, avail=F.availability_from_daily(ssi), nodes=list(conn))
R = {"large": {}, "seeds_M5": {}}
for M in (3, 5, 7):
    _, s, _ = F.null_test(recs, conn, n_perm=19999, seed=2026, M=M)
    g = s.set_index("statistic")
    R["large"][f"M{M}"] = {k: {"p": round(float(g.loc[k, "p"]), 4), "null_mean": round(float(g.loc[k, "null_mean"]), 4)}
                          for k in ("cf_all", "cf_opp", "p_link")}
ps = {k: [] for k in ("cf_all", "cf_opp", "p_link")}
for sd in range(10):
    _, s, _ = F.null_test(recs, conn, n_perm=1999, seed=100 + sd, M=5)
    g = s.set_index("statistic")
    for k in ps:
        ps[k].append(float(g.loc[k, "p"]))
R["seeds_M5"] = {k: {"min": round(min(v), 4), "max": round(max(v), 4), "mean": round(float(np.mean(v)), 4)} for k, v in ps.items()}
(T / "r6_pvalue_precision.json").write_text(json.dumps(R, indent=2))
print(json.dumps(R, indent=1))
