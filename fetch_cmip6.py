#!/usr/bin/env python3
"""
Read monthly CMIP6 pr, tas (Amon) and mrro (Lmon) for the Volta region from the Pangeo
CMIP6 archive on Google Cloud (anonymous access) for historical, SSP2-4.5 and SSP5-8.5.

One realisation per modelling centre (the 16 sources below, first r1i1p1f* member that has
all nine variable-experiment combinations; CanESM5 and CAMS-CSM1-0 use the first complete
member). Subset: 6.5 W - 2.5 E, 4.5 - 15.5 N, 1950-2100.

Writes data/external/cmip6/subsets/<source>_<variable>_<experiment>.nc and fetch_log.csv.

Pinned retrieval: data/external/cmip6/cmip6_store_manifest.csv lists the Zarr store path and
catalogue version of each of the 144 subsets used in the study. When that file is present, the
exact stores it names are read, so a later run does not pick up newer dataset versions. Without
it, the latest catalogue version of each combination is used and the manifest is written.
The Pangeo catalogue (pangeo-cmip6.csv) is downloaded when it is not already present.
Author: I. Quist (NUIST)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import gcsfs
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent
D = ROOT / "data/external/cmip6"
OUT = D / "subsets"
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = ["ACCESS-CM2", "BCC-CSM2-MR", "CAMS-CSM1-0", "CESM2-WACCM", "CNRM-CM6-1", "CanESM5",
           "FIO-ESM-2-0", "GFDL-ESM4", "INM-CM5-0", "IPSL-CM6A-LR", "KACE-1-0-G", "MCM-UA-1-0",
           "MIROC6", "MPI-ESM1-2-HR", "MRI-ESM2-0", "UKESM1-0-LL"]
VARS = [("Amon", "pr"), ("Amon", "tas"), ("Lmon", "mrro")]
EXPS = ["historical", "ssp245", "ssp585"]
LON = (-6.5, 2.5)
LAT = (4.5, 15.5)


def subset(ds: xr.Dataset, var: str) -> xr.DataArray:
    da = ds[var]
    lonn = "lon" if "lon" in da.dims else ("longitude" if "longitude" in da.dims else None)
    latn = "lat" if "lat" in da.dims else ("latitude" if "latitude" in da.dims else None)
    if lonn is None or latn is None:
        raise ValueError(f"unexpected dims {da.dims}")
    da = da.rename({lonn: "lon", latn: "lat"})
    lon = da.lon
    if float(lon.max()) > 180:
        da = da.assign_coords(lon=(((lon + 180) % 360) - 180)).sortby("lon")
    da = da.sortby("lat")
    da = da.sel(lon=slice(LON[0] - 3, LON[1] + 3), lat=slice(LAT[0] - 3, LAT[1] + 3))
    return da


def to_monthly_index(da: xr.DataArray) -> xr.DataArray:
    t = da.indexes["time"]
    yrs = np.array([x.year for x in t]); mos = np.array([x.month for x in t])
    idx = pd.to_datetime({"year": yrs, "month": mos, "day": 1})
    return da.assign_coords(time=idx.values)


CATALOGUE_URL = "https://storage.googleapis.com/cmip6/pangeo-cmip6.csv"
MANIFEST = D / "cmip6_store_manifest.csv"


def main():
    if not (D / "pangeo-cmip6.csv").exists():
        print("downloading", CATALOGUE_URL, flush=True)
        pd.read_csv(CATALOGUE_URL).to_csv(D / "pangeo-cmip6.csv", index=False)
    cat = pd.read_csv(D / "pangeo-cmip6.csv")
    pinned = pd.read_csv(MANIFEST) if MANIFEST.exists() else None
    cand = pd.read_csv(D / "model_selection_candidates.csv").set_index("source")
    fs = gcsfs.GCSFileSystem(token="anon")
    log = []
    todo = sys.argv[1:] or SOURCES
    for src in todo:
        mem = cand.loc[src, "member"]
        for tab, var in VARS:
            for exp in EXPS:
                f = OUT / f"{src}_{var}_{exp}.nc"
                if f.exists():
                    continue
                r = cat[(cat.source_id == src) & (cat.member_id == mem) & (cat.table_id == tab)
                        & (cat.variable_id == var) & (cat.experiment_id == exp)].sort_values("version")
                if pinned is not None:
                    pin = pinned[(pinned.source == src) & (pinned.variable == var) & (pinned.experiment == exp)]
                    if len(pin):
                        r = r[r.zstore == pin.zstore.iloc[0]]
                if r.empty:
                    log.append((src, mem, var, exp, "missing", 0)); continue
                t0 = time.time()
                try:
                    ds = xr.open_zarr(fs.get_mapper(r.zstore.iloc[-1]), consolidated=True, use_cftime=True)
                    da = subset(ds, var)
                    yr0, yr1 = (1950, 2014) if exp == "historical" else (2015, 2100)
                    tt = da.indexes["time"]
                    keep = np.array([yr0 <= x.year <= yr1 for x in tt])
                    da = da.isel(time=np.flatnonzero(keep))
                    da = to_monthly_index(da).load()
                    da.attrs.update(source=src, member=mem, experiment=exp, zstore=r.zstore.iloc[-1],
                                    grid_label=r.grid_label.iloc[-1])
                    da.to_dataset(name=var).to_netcdf(f)
                    log.append((src, mem, var, exp, "ok", round(time.time() - t0, 1)))
                    print(src, var, exp, da.shape, round(time.time() - t0, 1), flush=True)
                except Exception as e:  # noqa: BLE001
                    log.append((src, mem, var, exp, f"error: {e}"[:200], round(time.time() - t0, 1)))
                    print("ERR", src, var, exp, e, flush=True)
    old = pd.read_csv(D / "fetch_log.csv") if (D / "fetch_log.csv").exists() else None
    new = pd.DataFrame(log, columns=["source", "member", "variable", "experiment", "status", "seconds"])
    (pd.concat([old, new]) if old is not None else new).to_csv(D / "fetch_log.csv", index=False)
    if pinned is None:
        rows = []
        for f in sorted(OUT.glob("*.nc")):
            with xr.open_dataset(f, decode_times=False) as d:
                v = list(d.data_vars)[0]; a = d[v].attrs
                ver = cat.loc[cat.zstore == a.get("zstore"), "version"]
                rows.append(dict(file=f.name, source=a.get("source"), member=a.get("member"), variable=v,
                                 experiment=a.get("experiment"), grid_label=a.get("grid_label"),
                                 version=int(ver.iloc[0]) if len(ver) else None, zstore=a.get("zstore"),
                                 n_months=d.sizes.get("time")))
        pd.DataFrame(rows).to_csv(MANIFEST, index=False)


if __name__ == "__main__":
    main()
