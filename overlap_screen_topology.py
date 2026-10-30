#!/usr/bin/env python3
"""
Overlap sensitivity, the screened network, censoring and terrain-derived connectivity.

  1. M = 3 and M = 7 against the inherited M = 5: chains, opportunities,
     connected fractions, per-link probability, and an availability-aware
     network-label null (1,999 permutations) for each.
  2. The screened network: the four stations that pass the published screen on
     their own span, analysed as a stand-alone network.
  3. Censoring inside the chains: whether any connected pair involves an origin
     or upstream event bounded by missing data.
  4. Terrain-derived connectivity: the GRDC next-downstream pointers checked
     against the TDX-Hydro reach topology, including the Lawra-Dapola order.

Writes output_observational/tables/r6_*.csv and r6_numbers.json.
Author: I. Quist (NUIST)
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from voltaprop import fastmatch as F
from voltaprop.propagation import load_connectivity

T = Path("output_observational/tables"); R = {}
ev = pd.read_csv(T / "SSI_drought_events.csv", parse_dates=["start_date", "end_date"])
conn = load_connectivity(T / "upstream_connectivity.csv")
ssi = pd.read_csv(T / "SSI_daily.csv", parse_dates=["date"])
av = F.availability_from_daily(ssi)
recs = F.build_records(ev, avail=av, nodes=list(conn))
st = pd.read_csv("data/grdc_stations_raw.csv").set_index("grdc_no")
name = lambda n: st.loc[int(n), "station_name"].title()
rd = lambda v: round(v, 4) if isinstance(v, float) else v

# ---------------------------------------------------------------------------
# 1. M sensitivity with nulls
# ---------------------------------------------------------------------------
rows, sets = [], {}
for M in (3, 5, 7):
    obs, summ, _ = F.null_test(recs, conn, n_perm=1999, seed=42, M=M)
    g = summ.set_index("statistic")
    _, org = F.match(recs, conn, M=M)
    sets[M] = set(map(tuple, org.loc[org.connected, ["origin_node", "origin_eid"]].values))
    rows.append({"M": M, **{k: rd(v) for k, v in obs.items()},
                 **{f"p_{k}": round(float(g.loc[k, "p"]), 4) for k in ("cf_all", "cf_opp", "p_link")},
                 **{f"null_{k}": round(float(g.loc[k, "null_mean"]), 4) for k in ("cf_all", "cf_opp", "p_link")},
                 "connected_events": "; ".join(f"{name(o)} #{e}" for o, e in sorted(sets[M]))})
for r in rows:
    r["shared_with_M5"] = len(sets[r["M"]] & sets[5])
pd.DataFrame(rows).to_csv(T / "r6_m_sensitivity_null.csv", index=False)
R["m_sensitivity"] = rows

# ---------------------------------------------------------------------------
# 2. Screened network
# ---------------------------------------------------------------------------
qc = pd.read_csv(T / "qc_published_gate.csv")
passing = [int(x) for x in qc.loc[qc.pass_own.astype(bool), "node_id"]]
sub = {o: [u for u in conn.get(o, []) if u in passing] for o in passing}
srecs = {n: recs[n] for n in passing if n in recs}
sst = F.statistics(srecs, sub) if any(sub.values()) else None
R["screened_network"] = {"stations": [name(n) for n in passing], "rivers": [st.loc[n, "river"].title() for n in passing],
                         "directed_links": int(sum(len(v) for v in sub.values())),
                         "origin_stations": int(sum(1 for v in sub.values() if v)),
                         "events_at_screened_stations": int(ev.node_id.isin(passing).sum()),
                         "statistics": sst}

# ---------------------------------------------------------------------------
# 3. Censoring in chains
# ---------------------------------------------------------------------------
cen = pd.read_csv(T / "review_event_censoring.csv")
cflag = {(int(a), int(b)): c != "none" for a, b, c in zip(cen.node_id, cen.event_id, cen.bound)}
pairs, org = F.match(recs, conn)
pairs["origin_censored"] = [cflag[(o, e)] for o, e in zip(pairs.origin_node, pairs.origin_eid)]
pairs["upstream_censored"] = [cflag[(u, e)] for u, e in zip(pairs.upstream_node, pairs.upstream_eid)]
pairs["origin"] = pairs.origin_node.map(name); pairs["upstream"] = pairs.upstream_node.map(name)
pairs.to_csv(T / "r6_chain_censoring.csv", index=False)
opp = org[org.n_observing > 0].copy()
opp["origin_censored"] = [cflag[(o, e)] for o, e in zip(opp.origin_node, opp.origin_eid)]
both = pairs.origin_censored | pairs.upstream_censored
R["censoring_in_chains"] = {
    "pairs": len(pairs), "origin_censored": int(pairs.origin_censored.sum()),
    "upstream_censored": int(pairs.upstream_censored.sum()), "either": int(both.sum()),
    "which": pairs.loc[both, ["origin", "origin_eid", "upstream", "upstream_eid", "origin_censored",
                              "upstream_censored"]].to_dict(orient="records"),
    "events_censored_total": int(sum(cflag.values())), "events_total": len(cflag),
    "opportunities_with_censored_origin": int(opp.origin_censored.sum()), "opportunities": len(opp),
    "cf_opp_excluding_censored": round(float(opp.loc[~opp.origin_censored, "connected"].mean()), 4)
    if (~opp.origin_censored).any() else None,
    "n_opp_excluding_censored": int((~opp.origin_censored).sum())}

# ---------------------------------------------------------------------------
# 4. Terrain-derived connectivity (TDX-Hydro) against the GRDC pointers
# ---------------------------------------------------------------------------
rch = pd.read_parquet("data/reanalysis/reaches_volta.parquet").set_index("LINKNO")
ds = dict(zip(rch.index.astype(int), rch.DSLINKNO.astype(int)))
ln = dict(zip(rch.index.astype(int), rch.LengthGeodesicMeters / 1000.0))
gm = pd.read_csv("data/reanalysis/t2500/gauge_matches.csv")
g2r = {int(a): int(b) for a, b in zip(gm.grdc_no, gm.river_id) if pd.notna(b)}
loc_only = {}
for _, r in gm[gm.river_id.isna()].iterrows():
    # gauges whose GRDC area is inconsistent with the terrain: nearest reach >= 1,000 km2 within 2 km
    d = np.hypot((rch.lon - st.loc[r.grdc_no, "lon"]) * np.cos(np.radians(st.loc[r.grdc_no, "lat"])),
                 rch.lat - st.loc[r.grdc_no, "lat"]) * 111.2
    cand = d[(rch.area_km2 >= 1000)]
    j = int(cand.idxmin())
    loc_only[int(r.grdc_no)] = {"reach": j, "dist_km": round(float(cand.min()), 2),
                                "area_tdx": round(float(rch.loc[j, "area_km2"]), 0), "area_grdc": float(r.area_grdc)}
    g2r[int(r.grdc_no)] = j
r2g = {v: k for k, v in g2r.items()}
terr_up = {g: [] for g in g2r}; terr_next = {}
dist_next = {}
for g, rid in g2r.items():
    cur, acc, first, seen = ds.get(rid), 0.0, True, set()
    while cur is not None and cur in ds and cur not in seen:
        seen.add(cur); acc += ln.get(cur, 0.0)
        if cur in r2g:
            terr_up[r2g[cur]].append(g)
            if first:
                terr_next[g] = r2g[cur]; dist_next[g] = round(acc, 1); first = False
        cur = ds.get(cur)
grdc_up = {g: sorted(conn.get(g, [])) for g in g2r}
cmp = []
for g in sorted(g2r):
    a, b = set(grdc_up[g]), set(terr_up[g])
    cmp.append({"station": name(g), "grdc_upstream": ", ".join(name(x) for x in sorted(a)),
                "terrain_upstream": ", ".join(name(x) for x in sorted(b)),
                "agree": a == b, "terrain_next_downstream": name(terr_next[g]) if g in terr_next else "",
                "along_channel_km_to_next": dist_next.get(g)})
C = pd.DataFrame(cmp); C.to_csv(T / "r6_terrain_connectivity.csv", index=False)
links_g = {(u, o) for o, ups in grdc_up.items() for u in ups}
links_t = {(u, o) for o, ups in terr_up.items() for u in ups}
lw, dp = 1531050, 1931400
R["terrain_connectivity"] = {
    "gauges_compared": len(g2r), "grdc_links": len(links_g), "terrain_links": len(links_t),
    "links_in_both": len(links_g & links_t),
    "grdc_only": [f"{name(u)} -> {name(o)}" for u, o in sorted(links_g - links_t)],
    "terrain_only": [f"{name(u)} -> {name(o)}" for u, o in sorted(links_t - links_g)],
    "stations_agreeing": int(C.agree.sum()),
    "matched_by_location_only": {name(k): v for k, v in loc_only.items()},
    "lawra_dapola": {"terrain_order": "Lawra upstream of Dapola" if lw in terr_up.get(dp, []) else
                     ("Dapola upstream of Lawra" if dp in terr_up.get(lw, []) else "not connected"),
                     "along_channel_km": dist_next.get(lw) if terr_next.get(lw) == dp else dist_next.get(dp),
                     "grdc_pointer": "Lawra -> Dapola",
                     "grdc_area_dapola": float(st.loc[dp, "area_km2"]), "grdc_area_lawra": float(st.loc[lw, "area_km2"]),
                     "tdx_area_lawra": float(rch.loc[g2r[lw], "area_km2"].round(0)),
                     "tdx_area_dapola": float(rch.loc[g2r[dp], "area_km2"].round(0))}}
(T / "r6_numbers.json").write_text(json.dumps(R, indent=2, default=str))
print(json.dumps(R, indent=1, default=str))
