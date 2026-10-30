#!/usr/bin/env python3
"""
Step 0: GRDC daily discharge files to the two input tables of the pipeline.

GRDC data cannot be redistributed. Request the daily discharge of the Volta stations
from the GRDC Data Portal (https://portal.grdc.bafg.de; station numbers in
data/grdc_station_list.csv), unpack the export, and run

    python prepare_grdc.py <folder with *_Q_Day.Cmd.txt files>

Writes data/grdc_stations_raw.csv (header metadata and record statistics, with
`in_basin` from the GRDC Volta basin polygon) and data/grdc_discharge_raw.csv
(long format: grdc_no, date, Q).
Author: I. Quist (NUIST)
"""
import sys
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voltaprop import grdc

st, q = grdc.read_folder(sys.argv[1])
basin = gpd.read_file("data/GRDC_VOLTA_POLITICAL_TRANSBOUNDARY_BASIN.shp").to_crs(4326).union_all()
st["in_basin"] = [basin.buffer(0.05).contains(Point(x, y)) for x, y in zip(st.lon, st.lat)]
st.to_csv("data/grdc_stations_raw.csv", index=False)
q.to_csv("data/grdc_discharge_raw.csv", index=False)
print(f"{len(st)} stations ({int(st.in_basin.sum())} in the basin), {len(q):,} daily values")
