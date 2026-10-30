#!/usr/bin/env python3
"""
Reanalysis node network on the TDX-Hydro reach topology (GEOGLOWS v2, VPU 110).

Domain: Volta reaches upstream of Akosombo dam, excluding reaches inside Lake
Volta. Channel reaches: contributing area >= --threshold-km2.

Nodes (Section 2.1.5):
  gauge       the 18 GRDC gauges, snapped to a reach (rules below);
  confluence  the reach immediately below every junction of two channel reaches,
              and every domain outlet (a channel reach entering Lake Volta);
  reach       on each channel segment above a confluence or outlet, a first node
              1.5 x merge-km upstream, then one every --spacing-km, stopping at
              the next junction or the channel head.
Candidates are accepted in order of priority (gauge, confluence, reach) and
contributing area; a non-gauge candidate closer than --merge-km (great-circle,
between reach midpoints) to an accepted node is dropped.

Gauge snapping, in order:
  1. reaches within 3 km whose contributing area agrees with the GRDC area
     within --area-tol (relative); score = area error + distance (km) / 10;
  2. otherwise the nearest reach within 3 km ("near, GRDC area inconsistent");
  3. otherwise the best-scoring area-consistent reach within 25 km;
  4. otherwise unmatched.

Connectivity: every node lying downstream along the reach topology of another
node lists it as upstream (the same format as the observed network).

Usage:
  python build_reanalysis_network.py --streams streams_110.gpkg --lakes lakes_110.gpkg \
      --threshold-km2 2500 --tag t2500
Writes data/reanalysis/<tag>/{nodes.csv, upstream_connectivity.csv, gauge_matches.csv,
pair_distance.csv, network_meta.json}.
Author: I. Quist (NUIST)
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point

ap = argparse.ArgumentParser()
ap.add_argument("--streams", default="../ext/geoglows/streams_110.gpkg")
ap.add_argument("--lakes", default="../ext/geoglows/lakes_110.gpkg")
ap.add_argument("--threshold-km2", type=float, default=1000.0)
ap.add_argument("--spacing-km", type=float, default=75.0)
ap.add_argument("--merge-km", type=float, default=15.0)
ap.add_argument("--area-tol", type=float, default=0.35)
ap.add_argument("--tag", default=None)
a = ap.parse_args()
tag = a.tag or f"t{int(a.threshold_km2)}"
OUT = Path("data/reanalysis") / tag; OUT.mkdir(parents=True, exist_ok=True)
AKOSOMBO = (0.0590, 6.2997)          # lon, lat of the dam
UTM = 32630

r = pd.read_parquet("data/reanalysis/reaches_volta.parquet")
r["LINKNO"] = r.LINKNO.astype(int); r["DSLINKNO"] = r.DSLINKNO.astype(int)
r = r.set_index("LINKNO")
r["len_km"] = r.LengthGeodesicMeters / 1000.0
ds = r.DSLINKNO.to_dict()
children = defaultdict(list)
for k, d in ds.items():
    if d in r.index:
        children[d].append(k)

# ---------------------------------------------------------------------------
# Domain: upstream of Akosombo, outside Lake Volta
# ---------------------------------------------------------------------------
d_ak = np.hypot((r.lon - AKOSOMBO[0]) * np.cos(np.radians(AKOSOMBO[1])), r.lat - AKOSOMBO[1]) * 111.2
ak = int(r[(d_ak < 8)].area_km2.idxmax())
up = set(); stack = [ak]
while stack:
    x = stack.pop()
    if x in up:
        continue
    up.add(x); stack.extend(children.get(x, []))
lakes = gpd.read_file(a.lakes).to_crs(UTM)
lake = lakes.loc[lakes.area.idxmax(), "geometry"]
mids = gpd.GeoSeries([Point(xy) for xy in zip(r.lon, r.lat)], index=r.index, crs=4326).to_crs(UTM)
in_lake = set(r.index[mids.within(lake).values]) & up
below = set()                      # reaches between the lake and the dam (downstream of a lake reach)
for k in in_lake:
    x = ds.get(k)
    while x in up and x not in below:
        if x not in in_lake:
            below.add(x)
        x = ds.get(x)
domain = up - in_lake - below
chan = {k for k in domain if r.at[k, "area_km2"] >= a.threshold_km2}

# ---------------------------------------------------------------------------
# Gauge snapping
# ---------------------------------------------------------------------------
st = pd.read_csv("data/grdc_stations_raw.csv")
gauges = pd.read_csv("output_observational/tables/upstream_connectivity.csv", sep=";").node_id.astype(int).tolist()
st = st[st.grdc_no.isin(gauges)].set_index("grdc_no")
streams = gpd.read_file(a.streams, where="TerminalLink = 140382813")[["LINKNO", "geometry"]].to_crs(UTM)
streams["LINKNO"] = streams.LINKNO.astype(int)
streams = streams.set_index("LINKNO")
rows = []
for g, s in st.iterrows():
    p = gpd.GeoSeries([Point(s.lon, s.lat)], crs=4326).to_crs(UTM).iloc[0]
    cand = streams[streams.geometry.distance(p) <= 25_000].copy()
    cand["dist_km"] = cand.geometry.distance(p) / 1000.0
    cand["area_tdx"] = r.loc[cand.index, "area_km2"].values
    cand["area_err"] = (cand.area_tdx - s.area_km2).abs() / s.area_km2
    cand["score"] = cand.area_err + cand.dist_km / 10.0
    near = cand[cand.dist_km <= 3.0]
    ok = near[near.area_err <= a.area_tol]
    if len(ok):
        b, rule = ok.score.idxmin(), "within 3 km, area consistent"
    elif len(near):
        b, rule = near.dist_km.idxmin(), "near, GRDC area inconsistent"
    elif len(cand[cand.area_err <= a.area_tol]):
        b, rule = cand[cand.area_err <= a.area_tol].score.idxmin(), "within 25 km, area consistent"
    else:
        b, rule = None, "unmatched"
    rows.append({"grdc_no": int(g), "station": s.station_name, "river_id": b,
                 "area_grdc": float(s.area_km2), "area_tdx": float(cand.loc[b, "area_tdx"]) if b else np.nan,
                 "dist_km": round(float(cand.loc[b, "dist_km"]), 2) if b else np.nan,
                 "area_err": round(float(cand.loc[b, "area_err"]), 3) if b else np.nan, "rule": rule})
gm = pd.DataFrame(rows)
gm.to_csv(OUT / "gauge_matches.csv", index=False)
greach = {int(x.river_id): int(x.grdc_no) for x in gm.itertuples() if pd.notna(x.river_id)}

# ---------------------------------------------------------------------------
# Candidate nodes
# ---------------------------------------------------------------------------
cands = {}                               # reach -> (priority, kind)
for k in greach:
    cands[k] = (0, "gauge")
cchild = {k: [c for c in children.get(k, []) if c in chan] for k in chan}
starts = []
for k in chan:
    if len(cchild[k]) >= 2 or ds.get(k) not in chan:     # junction or outlet
        cands.setdefault(k, (1, "confluence"))
        starts.append(k)
for j in starts:
    for c in cchild[j]:
        x, dist, nxt = c, 0.0, 1.5 * a.merge_km
        while True:
            dist += r.at[x, "len_km"]
            if dist >= nxt:
                cands.setdefault(x, (2, "reach")); nxt += a.spacing_km
            up_ = cchild.get(x, [])
            if len(up_) != 1:            # next junction (handled as its own start) or channel head
                break
            x = up_[0]
order = sorted(cands, key=lambda k: (cands[k][0], -r.at[k, "area_km2"]))
acc = []
for k in order:
    if cands[k][1] != "gauge" and acc:
        d = np.hypot((r.loc[acc, "lon"] - r.at[k, "lon"]) * np.cos(np.radians(r.at[k, "lat"])),
                     r.loc[acc, "lat"] - r.at[k, "lat"]) * 111.2
        if (d < a.merge_km).any():
            continue
    acc.append(k)
nodes = pd.DataFrame({"node_id": acc, "kind": [cands[k][1] for k in acc],
                      "grdc_no": [greach.get(k) for k in acc],
                      "lon": r.loc[acc, "lon"].values, "lat": r.loc[acc, "lat"].values,
                      "area_km2": r.loc[acc, "area_km2"].values})
countries = gpd.read_file("data/external/naturalearth_lowres/naturalearth_lowres.shp")
pts = gpd.GeoDataFrame(nodes, geometry=[Point(xy) for xy in zip(nodes.lon, nodes.lat)], crs=4326)
nodes["country"] = gpd.sjoin_nearest(pts.to_crs(UTM), countries[["name", "geometry"]].to_crs(UTM), how="left")\
    .groupby(level=0).name.first().values

# ---------------------------------------------------------------------------
# Connectivity and along-channel distances
# ---------------------------------------------------------------------------
nset = set(acc)
upstream = {k: [] for k in acc}; immediate = {}; prow = []
for k in acc:
    x, dist, first, seen = ds.get(k), 0.0, True, set()
    while x in r.index and x not in seen:
        seen.add(x); dist += r.at[x, "len_km"]
        if x in nset:
            upstream[x].append(k)
            euc = np.hypot((r.at[x, "lon"] - r.at[k, "lon"]) * np.cos(np.radians(r.at[k, "lat"])),
                           r.at[x, "lat"] - r.at[k, "lat"]) * 111.2
            prow.append({"origin": x, "upstream": k, "net_km": round(dist, 2), "euc_km": round(float(euc), 2)})
            if first:
                immediate[k] = x; first = False
        x = ds.get(x)
cn = pd.DataFrame({"node_id": acc, "n_upstream": [len(upstream[k]) for k in acc],
                   "immediate_downstream": [immediate.get(k) for k in acc],
                   "upstream_chain": [",".join(str(u) for u in sorted(upstream[k])) for k in acc]})
nodes.to_csv(OUT / "nodes.csv", index=False)
cn.to_csv(OUT / "upstream_connectivity.csv", sep=";", index=False)
pd.DataFrame(prow).to_csv(OUT / "pair_distance.csv", index=False)
imm = [p for p in prow if immediate.get(p["upstream"]) == p["origin"]]
meta = {"tag": tag, "threshold_km2": a.threshold_km2, "spacing_km": a.spacing_km, "merge_km": a.merge_km,
        "area_tol": a.area_tol, "akosombo_reach": ak, "domain_reaches": len(domain), "channel_reaches": len(chan),
        "lake_reaches_excluded": len(in_lake), "below_lake_excluded": len(below), "nodes": len(acc),
        "by_kind": nodes.kind.value_counts().to_dict(), "origin_nodes": int((cn.n_upstream > 0).sum()),
        "directed_links": int(cn.n_upstream.sum()),
        "immediate_link_km": {"median": round(float(np.median([p["net_km"] for p in imm])), 2),
                              "max": round(float(np.max([p["net_km"] for p in imm])), 2)} if imm else None,
        "countries": nodes.country.value_counts().to_dict(),
        "gauges_by_rule": gm.rule.value_counts().to_dict()}
(OUT / "network_meta.json").write_text(json.dumps(meta, indent=2, default=str))
print(json.dumps(meta, indent=1, default=str))
