#!/usr/bin/env python3
"""
Reanalysis arm, step 2: daily discharge from the GEOGLOWS v2 retrospective
simulation (ERA5 runoff routed on the TDX-Hydro network; open data on AWS,
s3://geoglows-v2/retrospective/daily.zarr; no account needed).

Reads the reaches that carry a node in any of the node networks built by
build_reanalysis_network.py, plus every reach matched to a GRDC gauge, for
1950-01-01 to the last full calendar year of the store.

Writes data/reanalysis/Q_daily.parquet (long: node_id, date, Q) and
data/reanalysis/fetch_meta.json.
Author: I. Quist (NUIST)
"""
import json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr

if os.path.exists("/root/.ccr/ca-bundle.crt"):                 # proxy certificate of the build machine only
    os.environ.setdefault("AWS_CA_BUNDLE", "/root/.ccr/ca-bundle.crt")
NET = Path("data/reanalysis")
STORE = "s3://geoglows-v2/retrospective/daily.zarr"
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

ids = set()
for tag in sorted(p.name for p in NET.iterdir() if p.is_dir()):
    ids |= set(pd.read_csv(NET / tag / "nodes.csv").node_id.astype(int))
    gm = pd.read_csv(NET / tag / "gauge_matches.csv").dropna(subset=["river_id"])
    ids |= set(gm.river_id.astype(int))
ids = sorted(ids)
log(f"{len(ids)} reaches requested")

ds = xr.open_zarr(STORE, storage_options={"anon": True, "client_kwargs": {"region_name": "us-west-2"}},
                  consolidated=True)
rid = ds.river_id.values
pos = pd.Series(np.arange(len(rid)), index=rid)
missing = [i for i in ids if i not in pos.index]
if missing:
    log(f"not in store: {missing}")
have = [i for i in ids if i in pos.index]
order = np.argsort(pos.loc[have].values)
have = [have[k] for k in order]
last = pd.Timestamp(ds.time.values[-1])
end = f"{last.year - 1}-12-31" if (last.month, last.day) != (12, 31) else str(last.date())
q = ds.Q.sel(time=slice("1950-01-01", end)).isel(river_id=pos.loc[have].values)
t0 = time.time()
arr = q.values
log(f"read {arr.shape} in {time.time() - t0:.0f}s")
out = pd.DataFrame(arr, index=pd.to_datetime(q.time.values), columns=have)
out.index.name = "date"
long = out.stack().rename("Q").reset_index().rename(columns={"level_1": "node_id"})
long["node_id"] = long.node_id.astype(np.int64)
long.to_parquet(NET / "Q_daily.parquet", index=False)
meta = {"store": STORE, "store_time_range": [str(pd.Timestamp(ds.time.values[0]).date()), str(last.date())],
        "period_read": ["1950-01-01", end], "reaches": len(have), "missing": missing,
        "negative_values": int((arr < 0).sum()), "nan_values": int(np.isnan(arr).sum()),
        "retrieved": time.strftime("%Y-%m-%d")}
(NET / "fetch_meta.json").write_text(json.dumps(meta, indent=2))
log(json.dumps(meta))
