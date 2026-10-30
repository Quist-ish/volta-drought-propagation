#!/usr/bin/env python3
"""
Downloads the openly licensed inputs of the analysis (everything except the GRDC
discharge, which must be requested from the GRDC; see data/README.md).

  * GEOGLOWS v2 / TDX-Hydro hydrography for VPU 110 (streams and lakes), from the public
    bucket s3://geoglows-v2/hydrography/vpu=110/ (anonymous access, AWS us-west-2)
  * GPCC Full Data Monthly Product v2022 at 0.5 deg, 1951-2020 (Deutscher Wetterdienst)
  * Sahel precipitation index (JISAO/NOAA PSL)
  * Natural Earth 1:110m countries
  * NOAA ERSSTv5 monthly SST (NOAA PSL copy), from which the ATL3 index is computed
  * Monthly climate indices (NOAA PSL). The release ships the snapshots used in the study
    (retrieved 26 October 2026) in data/external/climate_indices/; they are downloaded only
    when missing, because the provider revises recent months.

Usage: python download_inputs.py [--dest external]
The GEOGLOWS daily discharge is read directly from the cloud by fetch_reanalysis.py.
Author: I. Quist (NUIST)
"""
import argparse, os, shutil, urllib.request, zipfile
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--dest", default="external"); a = ap.parse_args()
D = Path(a.dest); (D / "geoglows").mkdir(parents=True, exist_ok=True); (D / "gpcc").mkdir(exist_ok=True)
EXT = Path("data/external"); EXT.mkdir(parents=True, exist_ok=True)


def get(url, out):
    if Path(out).exists() and Path(out).stat().st_size > 0:
        return
    print("downloading", url)
    with urllib.request.urlopen(url) as r, open(out, "wb") as f:
        shutil.copyfileobj(r, f)


import s3fs                                                          # noqa: E402
fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-west-2"})
for f in ("streams_110.gpkg", "lakes_110.gpkg"):
    if not (D / "geoglows" / f).exists():
        print("downloading", f); fs.get(f"geoglows-v2/hydrography/vpu=110/{f}", str(D / "geoglows" / f))

for dec in ("1951_1960", "1961_1970", "1971_1980", "1981_1990", "1991_2000", "2001_2010", "2011_2020"):
    get(f"https://opendata.dwd.de/climate_environment/GPCC/full_data_monthly_v2022/05/"
        f"full_data_monthly_v2022_{dec}_05.nc.gz", D / "gpcc" / f"full_data_monthly_v2022_{dec}_05.nc.gz")

get("https://psl.noaa.gov/data/correlation/sahelrain.data", EXT / "sahelrain.data")

CI = EXT / "climate_indices"; CI.mkdir(exist_ok=True)
get("https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v5/sst.mnmean.nc", CI / "ersst_v5.nc")
for name, url in (("amo.txt", "https://psl.noaa.gov/data/correlation/amon.us.long.data"),
                  ("n34.txt", "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/nino34.long.anom.data"),
                  ("dmi.txt", "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data"),
                  ("tna.txt", "https://psl.noaa.gov/data/correlation/tna.data"),
                  ("tsa.txt", "https://psl.noaa.gov/data/correlation/tsa.data"),
                  ("pdo.txt", "https://psl.noaa.gov/data/correlation/pdo.data")):
    get(url, CI / name)

ne = EXT / "naturalearth_lowres"
if not (ne / "naturalearth_lowres.shp").exists():
    ne.mkdir(exist_ok=True)
    get("https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip", D / "ne110.zip")
    zipfile.ZipFile(D / "ne110.zip").extractall(ne)
    import geopandas as gpd
    g = gpd.read_file(ne / "ne_110m_admin_0_countries.shp")
    g[["NAME", "ISO_A3", "CONTINENT", "geometry"]].rename(
        columns={"NAME": "name", "ISO_A3": "iso_a3", "CONTINENT": "continent"}).to_file(ne / "naturalearth_lowres.shp")
print("inputs ready in", D, "and", EXT)
