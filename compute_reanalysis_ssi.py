#!/usr/bin/env python3
"""
Reanalysis arm, step 3: the daily SSI at every reanalysis reach, with the same
construction as the observed arm (six three-parameter families by L-moments,
minimum AIC, +/-15-day circular pooling, zero-flow mass, seasonal-validity
mask). Distributions are fitted on the common reference period 1961-2020 (the
published study's period) and applied to 1961-2025.

Writes output_reanalysis/tables/SSI_daily.parquet and SSI_fit_log.parquet.
Author: I. Quist (NUIST)
"""
import sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voltaprop.ssi import SSIConfig, compute_ssi_node

T = Path("output_reanalysis/tables"); T.mkdir(parents=True, exist_ok=True)
CFG = SSIConfig(reference_years=(1961, 2020))
q = pd.read_parquet("data/reanalysis/Q_daily.parquet")
q["date"] = pd.to_datetime(q.date)
q = q[(q.date >= "1961-01-01") & (q.date <= "2025-12-31")]


def one(n):
    s = q[q.node_id == n].set_index("date").Q.sort_index().astype(float)
    ssi, log = compute_ssi_node(s, CFG)
    lg = pd.DataFrame([{"node_id": n, "doy": d, "best": v.get("best"), "valid": v.get("valid"),
                        "p0": v.get("p0"), "reason": v.get("reason")} for d, v in log.items()])
    return pd.DataFrame({"node_id": n, "date": s.index, "Q": s.values, "SSI": ssi.values}), lg


if __name__ == "__main__":
    nodes = sorted(q.node_id.unique())
    t0 = time.time()
    with Pool(2) as p:
        res = p.map(one, nodes, chunksize=4)
    S = pd.concat([r[0] for r in res], ignore_index=True)
    L = pd.concat([r[1] for r in res], ignore_index=True)
    S.to_parquet(T / "SSI_daily.parquet", index=False); L.to_parquet(T / "SSI_fit_log.parquet", index=False)
    print(f"{len(nodes)} reaches in {time.time() - t0:.0f}s; defined share {S.SSI.notna().mean():.3f}; "
          f"invalid DOY share {1 - L.valid.mean():.3f}")
