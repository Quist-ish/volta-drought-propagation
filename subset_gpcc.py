#!/usr/bin/env python3
"""
Subset of the GPCC Full Data Monthly Product v2022 at 0.5 deg (Schneider et al.,
2022; doi:10.5676/DWD_GPCC/FD_M_V2022_050) over the Volta Basin, 1951-2020.

Usage: python subset_gpcc.py <dir with full_data_monthly_v2022_YYYY_YYYY_05.nc.gz>
Writes data/external/gpcc_v2022_volta_05.nc (precip in mm/month, numgauge).
Author: I. Quist (NUIST)
"""
import gzip, shutil, sys, tempfile
from pathlib import Path
import xarray as xr

src = Path(sys.argv[1] if len(sys.argv) > 1 else "../ext/gpcc")
parts = []
with tempfile.TemporaryDirectory() as tmp:
    for f in sorted(src.glob("full_data_monthly_v2022_*_05.nc.gz")):
        if int(f.name.split("_")[4]) < 1951:
            continue
        nc = Path(tmp) / f.name[:-3]
        with gzip.open(f, "rb") as a, open(nc, "wb") as b:
            shutil.copyfileobj(a, b)
        ds = xr.open_dataset(nc)
        lat = slice(16, 4.5) if ds.lat[0] > ds.lat[-1] else slice(4.5, 16)
        parts.append(ds[["precip", "numgauge"]].sel(lat=lat, lon=slice(-6.5, 2.5)).load())
        ds.close()
out = xr.concat(parts, dim="time").sortby("time")
out.attrs.update(source="GPCC Full Data Monthly Product Version 2022 at 0.5 deg (Schneider et al., 2022)",
                 doi="10.5676/DWD_GPCC/FD_M_V2022_050", subset="lon -6.5 to 2.5, lat 4.5 to 16")
Path("data/external").mkdir(parents=True, exist_ok=True)
out.to_netcdf("data/external/gpcc_v2022_volta_05.nc")
print(out.sizes, str(out.time.values[0])[:10], str(out.time.values[-1])[:10])
