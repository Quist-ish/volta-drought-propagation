#!/usr/bin/env python3
"""
Chain-level corroboration with gauge-based precipitation (Sections 2.1.6, 2.2.10, 3.5).

Catchment precipitation: the GPCC Full Data Monthly v2022 0.5 deg cells holding
the midpoints of all TDX-Hydro reaches upstream of (and including) a gauge or
node reach, weighted by reach count. SPI-6: six-month totals standardised by
calendar month with a gamma distribution and a zero-precipitation mass,
1961-2020 (McKee et al., 1993).

Tests, for the observed gauges (1961-2007) and the main reanalysis network
(1961-2020):
  * events: share of onsets after a dry half-year (SPI-6 <= -0.5 in the event's own
    catchment, onset month) against the climatological share (one-sided binomial);
  * opportunities: driest SPI-6 among observing upstream catchments at the origin
    onset, connected vs unconnected (one-sided Mann-Whitney); share with both the
    origin and an observing upstream catchment dry (one-sided Fisher).

Writes output_rainfall/tables/{spi6_catchments.csv, rainfall_corroboration.csv,
opportunities_observed.csv, opportunities_reanalysis.csv} and gpcc_support.json.
Author: I. Quist (NUIST)
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np, pandas as pd, xarray as xr
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voltaprop import fastmatch as F
from voltaprop.propagation import load_connectivity

MAIN = "t2500"
T = Path("output_rainfall/tables"); T.mkdir(parents=True, exist_ok=True)
TO, TR = Path("output_observational/tables"), Path("output_reanalysis/tables")
DRY = -0.5

g = xr.open_dataset("data/external/gpcc_v2022_volta_05.nc")
pr = g.precip.sel(time=slice("1951-01-01", "2020-12-31")).load()
lat, lon = pr.lat.values, pr.lon.values

# GPCC gauge support over the basin, by decade
ng = g.numgauge.sel(lat=slice(15, 6) if lat[0] > lat[-1] else slice(6, 15), lon=slice(-5.5, 1.5))
dec = ng.groupby("time.year").mean().mean(["lat", "lon"]).to_series()
support = {f"{a}-{a + 9}": round(float(dec.loc[a:a + 9].mean()), 2) for a in range(1961, 2020, 10)}

# catchments
r = pd.read_parquet("data/reanalysis/reaches_volta.parquet").set_index("LINKNO")
children = defaultdict(list)
for k, d in r.DSLINKNO.items():
    children[int(d)].append(int(k))
gm = pd.read_csv(f"data/reanalysis/{MAIN}/gauge_matches.csv").dropna(subset=["river_id"])
g2r = dict(zip(gm.grdc_no.astype(int), gm.river_id.astype(int)))
nodes1 = pd.read_csv(f"data/reanalysis/{MAIN}/nodes.csv").node_id.astype(int).tolist()
targets = sorted(set(nodes1) | set(g2r.values()))


def upstream_reaches(k):
    out, stack = [], [k]
    while stack:
        x = stack.pop(); out.append(x); stack.extend(children.get(x, []))
    return out


def cell(v, grid):
    return int(np.abs(grid - v).argmin())


P = {}
for k in targets:
    ups = r.loc[upstream_reaches(k)]
    cells = pd.Series([(cell(a, lat), cell(b, lon)) for a, b in zip(ups.lat, ups.lon)]).value_counts()
    w = np.array(cells.values, float); w /= w.sum()
    P[k] = sum(wi * pr.isel(lat=i, lon=j).values for (i, j), wi in zip(cells.index, w))
PR = pd.DataFrame(P, index=pd.to_datetime(pr.time.values))


def spi(x: pd.Series, k=6, ref=(1961, 2020)):
    acc = x.rolling(k).sum()
    out = pd.Series(np.nan, index=x.index)
    for m in range(1, 13):
        sel = acc.index.month == m
        a = acc[sel]
        ref_ = a[(a.index.year >= ref[0]) & (a.index.year <= ref[1])].dropna()
        q0 = float((ref_ <= 0).mean()); pos = ref_[ref_ > 0]
        shape, loc_, scale = stats.gamma.fit(pos, floc=0)
        cdf = np.where(a > 0, q0 + (1 - q0) * stats.gamma.cdf(a, shape, loc=0, scale=scale), q0 / 2 if q0 > 0 else 1e-6)
        out[sel] = stats.norm.ppf(np.clip(cdf, 1e-6, 1 - 1e-6))
    return out


SPI = PR.apply(spi)
SPI = SPI[SPI.index.year >= 1961]
SPI.columns = [str(c) for c in SPI.columns]
SPI.to_csv(T / "spi6_catchments.csv")


def spi_at(key, dates):
    col = SPI[str(key)]
    m = pd.DatetimeIndex(pd.to_datetime(dates)).to_period("M").to_timestamp()
    return col.reindex(m).values


def evaluate(label, events, conn, key, avail, cols):
    ev = events.copy()
    vals = []
    for n, g_ in ev.groupby("node_id"):
        vals.append(pd.Series(spi_at(key(n), g_.start_date), index=g_.index))
    ev["spi"] = pd.concat(vals)
    e = ev.dropna(subset=["spi"])
    clim = float((SPI[cols] <= DRY).stack().mean())
    k_dry = int((e.spi <= DRY).sum())
    recs = F.build_records(events, avail=avail, nodes=list(conn))
    _, org = F.match(recs, conn)
    op = org[org.n_observing > 0].copy()
    op["origin_spi"] = [spi_at(key(o), [d])[0] for o, d in zip(op.origin_node, op.s0_date)]
    op["up_spi_min"] = [np.nanmin([spi_at(key(u), [d])[0] for u in obs]) if obs else np.nan
                        for obs, d in zip(op.observing, op.s0_date)]
    op = op.drop(columns=["observing"]).dropna(subset=["up_spi_min", "origin_spi"])
    c, u = op[op.connected], op[~op.connected]
    coh = (op.origin_spi <= DRY) & (op.up_spi_min <= DRY)
    t = np.array([[coh[op.connected].sum(), (~coh[op.connected]).sum()], [coh[~op.connected].sum(), (~coh[~op.connected]).sum()]])
    row = {"label": label, "events": len(e), "share_dry_onset": round(k_dry / len(e), 3), "climatology": round(clim, 3),
           "p_binom": float(stats.binomtest(k_dry, len(e), clim, alternative="greater").pvalue),
           "opportunities": len(op), "connected": len(c),
           "up_spi_connected_median": round(float(c.up_spi_min.median()), 3),
           "up_spi_unconnected_median": round(float(u.up_spi_min.median()), 3),
           "p_mw_less": float(stats.mannwhitneyu(c.up_spi_min, u.up_spi_min, alternative="less").pvalue),
           "coherent_share_connected": round(float(coh[op.connected].mean()), 3),
           "coherent_share_unconnected": round(float(coh[~op.connected].mean()), 3),
           "p_fisher_greater": float(stats.fisher_exact(t, alternative="greater")[1])}
    return row, op


rows = []
ev_o = pd.read_csv(TO / "SSI_drought_events.csv", parse_dates=["start_date", "end_date"])
conn_o = load_connectivity(TO / "upstream_connectivity.csv")
ssi_o = pd.read_csv(TO / "SSI_daily.csv", parse_dates=["date"])
row, op = evaluate("observed gauges 1961-2007", ev_o, conn_o, lambda n: g2r[int(n)],
                   F.availability_from_daily(ssi_o), [str(g2r[n]) for n in conn_o])
rows.append(row); op.to_csv(T / "opportunities_observed.csv", index=False)

rev = pd.read_csv(TR / "SSI_drought_events_1961_2020.csv", parse_dates=["start_date", "end_date"])
rconn = load_connectivity(f"data/reanalysis/{MAIN}/upstream_connectivity.csv")
rs = pd.read_parquet(TR / "SSI_daily.parquet", columns=["node_id", "date", "SSI"])
rs["date"] = pd.to_datetime(rs.date)
rs = rs[rs.node_id.isin(nodes1) & (rs.date.dt.year >= 1961) & (rs.date.dt.year <= 2020)]
row, op = evaluate(f"reanalysis {MAIN} 1961-2020", rev[rev.node_id.isin(nodes1)], rconn, lambda n: int(n),
                   F.availability_from_daily(rs), [str(n) for n in nodes1])
rows.append(row); op.to_csv(T / "opportunities_reanalysis.csv", index=False)

RC = pd.DataFrame(rows); RC.to_csv(T / "rainfall_corroboration.csv", index=False)
Path("output_rainfall/gpcc_support.json").write_text(json.dumps({"mean_gauges_per_cell": support}, indent=2))
print(RC.T.to_string()); print(support)
