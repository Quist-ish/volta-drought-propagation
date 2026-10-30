#!/usr/bin/env python3
"""
Volta reach table from the GEOGLOWS v2 / TDX-Hydro stream layer (VPU 110).

Usage: python export_reach_table.py <streams_110.gpkg>
Writes data/reanalysis/reaches_volta.parquet with LINKNO, DSLINKNO, strmOrder,
LengthGeodesicMeters, area_km2 (contributing area at the downstream end) and the
reach midpoint (lon, lat).
Author: I. Quist (NUIST)
"""
import sys
from pathlib import Path
import geopandas as gpd

VOLTA_TERMINAL = 140382813
src = sys.argv[1] if len(sys.argv) > 1 else "../ext/geoglows/streams_110.gpkg"
s = gpd.read_file(src, where=f"TerminalLink = {VOLTA_TERMINAL}")          # EPSG:3857
mid = gpd.GeoSeries(s.geometry.interpolate(0.5, normalized=True), crs=s.crs).to_crs(4326)
out = s[["LINKNO", "DSLINKNO", "strmOrder", "LengthGeodesicMeters"]].copy()
out["area_km2"] = s.DSContArea / 1e6
out["lon"], out["lat"] = mid.x.values, mid.y.values
Path("data/reanalysis").mkdir(parents=True, exist_ok=True)
out.to_parquet("data/reanalysis/reaches_volta.parquet", index=False)
print(f"{len(out)} reaches; largest contributing area {out.area_km2.max():,.0f} km2")
