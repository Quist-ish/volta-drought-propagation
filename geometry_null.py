#!/usr/bin/env python3
"""
Timing geometry of matched pairs against the availability-aware permutation null (26 Oct 2026).
Each matched pair is classed by onset lead (s0 - su) and recovery offset (eu - e0): near-simultaneous if the lead
is at most 1 day, relay if the upstream event ends first (eu < e0), nested otherwise. The null reassigns whole
station records (catalogue plus daily availability) to network slots, as in fastmatch.null_test.
Writes output_observational/tables/geometry_null.json. Author: I. Quist (NUIST)
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


def classes(p):
    lead = p["s0"] - p["su"]; off = p["eu"] - p["e0"]
    near = lead <= 1
    relay = (~near) & (off < 0)
    nested = (~near) & (off >= 0)
    return int(near.sum()), int(relay.sum()), int(nested.sum())


pairs, org = F.match(recs, conn, W=45, M=5, detail=False)
obs = classes(pairs)
assert sum(obs) == 10, obs
ids = np.array(sorted(recs.keys()))
rng = np.random.default_rng(2026)
rows = []
for _ in range(19999):
    perm = rng.permutation(ids)
    slot = {int(ids[i]): int(perm[i]) for i in range(len(ids))}
    p, _ = F.match(recs, conn, W=45, M=5, slot=slot, detail=False)
    rows.append(classes(p))
N = pd.DataFrame(rows, columns=["near", "relay", "nested"])
N["pairs"] = N.sum(axis=1)
tot = N[["near", "relay", "nested"]].sum()
res = {"observed": dict(zip(["near", "relay", "nested"], obs)),
       "null_mean_counts": {k: round(float(N[k].mean()), 3) for k in ("near", "relay", "nested", "pairs")},
       "null_pooled_share": {k: round(float(tot[k] / tot.sum()), 3) for k in ("near", "relay", "nested")},
       "p_relay_ge_obs": round(float((1 + (N.relay >= obs[1]).sum()) / (1 + len(N))), 4),
       "p_near_ge_obs": round(float((1 + (N.near >= obs[0]).sum()) / (1 + len(N))), 4),
       "p_nested_ge_obs": round(float((1 + (N.nested >= obs[2]).sum()) / (1 + len(N))), 4),
       "n_perm": len(N), "seed": 2026}
# share among permutations with exactly 10 pairs (conditional on the observed number of matches)
c10 = N[N.pairs == 10]
if len(c10):
    res["given_10_pairs"] = {"n": int(len(c10)), **{k: round(float(c10[k].mean()), 3) for k in ("near", "relay", "nested")}}
json.dump(res, open(T / "geometry_null.json", "w"), indent=1)
print(json.dumps(res, indent=1))
