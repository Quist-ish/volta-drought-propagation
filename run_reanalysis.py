#!/usr/bin/env python3
"""
Reanalysis arm, step 4: the framework on the GEOGLOWS v2 node networks.

  R1 main configuration: 2,500 km2 network, 1961-2020 (the published period), M = 5
  R2 node density: 1,000 and 5,000 km2 networks
  R3 data vintage and periods: 1961-2025, 1961-2007, 2008-2025
  R4 W x M grid (RSI, Kneedle) and M = 3, 7
  R5 detection sensitivity (pooling gap x minimum duration)
  R6 events that do not propagate (connected vs unconnected opportunities)
  R7 lag against along-channel distance
  R8 transboundary propagation
  R9 corroboration against the Sahel precipitation index (1961-2017)

Every configuration reports coverage, observing links, cf_all, cf_opp and the
per-link probability with calendar-year block-bootstrap intervals, and is
tested with the availability-aware network-label null.

Writes output_reanalysis/reanalysis_numbers.json and tables/*.csv.
Author: I. Quist (NUIST)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voltaprop import fastmatch as F
from voltaprop.events import EventConfig, detect_network_events
from voltaprop.propagation import load_connectivity
from voltaprop.analysis import relative_sensitivity_index, kneedle_elbow

MAIN, DENSITY = "t2500", ("t1000", "t5000")
OUT = Path("output_reanalysis"); T = OUT / "tables"; T.mkdir(parents=True, exist_ok=True)
NET = Path("data/reanalysis")
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
R = {"main_network": MAIN}

S_ALL = pd.read_parquet(T / "SSI_daily.parquet")
S_ALL["date"] = pd.to_datetime(S_ALL.date)


def network(tag):
    conn = load_connectivity(NET / tag / "upstream_connectivity.csv")
    nodes = pd.read_csv(NET / tag / "nodes.csv").set_index("node_id")
    return conn, nodes


def period_ssi(nodes, y0, y1):
    s = S_ALL[S_ALL.node_id.isin(nodes) & (S_ALL.date.dt.year >= y0) & (S_ALL.date.dt.year <= y1)]
    return s[["node_id", "date", "Q", "SSI"]]


def events_for(ssi, cfg=EventConfig()):
    return detect_network_events(ssi, cfg, node_col="node_id", progress=False)


def bootstrap(org, n=2000, seed=2026):
    opp = org[org.n_observing > 0].copy()
    opp["year"] = pd.to_datetime(opp.s0_date).dt.year
    years = np.array(sorted(opp.year.unique())); by = {y: g for y, g in opp.groupby("year")}
    rng = np.random.default_rng(seed); cf, pl = [], []
    for _ in range(n):
        g = pd.concat([by[y] for y in rng.choice(years, len(years), replace=True)])
        cf.append(g.connected.mean()); pl.append(F.link_probability_mle(g.n_observing.values, g.connected.values))
    q = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]
    return q(cf), q(pl)


def run_config(label, tag, y0, y1, M=5, n_perm=999, ev=None, keep=False):
    conn, nodes = network(tag)
    ssi = period_ssi(nodes.index, y0, y1)
    ev = events_for(ssi) if ev is None else ev
    av = F.availability_from_daily(ssi)
    recs = F.build_records(ev, avail=av, nodes=list(conn))
    pairs, org = F.match(recs, conn, M=M)
    obs, summ, null = F.null_test(recs, conn, n_perm=n_perm, seed=42, M=M)
    g = summ.set_index("statistic")
    cfi, pli = bootstrap(org)
    years = ssi.SSI.notna().sum() / 365.25
    row = {"label": label, "network": tag, "period": f"{y0}-{y1}", "M": M, "n_nodes": len(nodes),
           "n_links": int(sum(len(v) for v in conn.values())), "n_events": len(ev),
           "defined_node_years": round(float(years), 1), "event_rate": round(len(ev) / years, 3),
           **{k: obs[k] for k in ("n_origin_events", "n_opportunities", "n_connected", "n_pairs", "coverage",
                                  "mean_k_observing", "cf_all", "cf_opp", "p_link", "mean_chain_size",
                                  "mean_pf_published", "mean_pf_observing")},
           "cf_opp_ci": cfi, "p_link_ci": pli, "n_permutations": n_perm,
           **{f"null_{k}": float(g.loc[k, "null_mean"]) for k in ("cf_all", "cf_opp", "p_link", "mean_chain_size")},
           **{f"p_{k}": float(g.loc[k, "p"]) for k in ("cf_all", "cf_opp", "p_link", "mean_chain_size")}}
    log(f"  {label}: events={row['n_events']}, opp={row['n_opportunities']}, connected={row['n_connected']}, "
        f"cf_opp={row['cf_opp']:.4f}, p_link={row['p_link']:.4f} (null {row['null_p_link']:.4f}, p={row['p_p_link']}), "
        f"chain size={row['mean_chain_size']:.4f}")
    return (row, ev, ssi, conn, nodes, recs, pairs, org) if keep else row


rows = []
# ---------------------------------------------------------------------------
log("R1 main configuration")
main, EV, SSI, CONN, NODES, RECS, PAIRS, ORG = run_config(f"main {MAIN} 1961-2020 M5", MAIN, 1961, 2020,
                                                          n_perm=1999, keep=True)
rows.append(main)
EV.to_csv(T / "SSI_drought_events_1961_2020.csv", index=False)
ev_ix = EV.set_index(["node_id", "event_id"])
P = PAIRS.copy()
P["origin_start"], P["origin_end"] = P.s0_date, P.e0_date
P["upstream_start"], P["upstream_end"] = P.su_date, P.eu_date
P = P.drop(columns=["s0", "e0", "su", "eu", "s0_date", "e0_date", "su_date", "eu_date"])
P.to_csv(T / "propagation_pairs.csv", index=False)
O_ = ORG.drop(columns=["observing"]).copy(); O_["observing"] = [",".join(map(str, x)) for x in ORG.observing]
O_.to_csv(T / "origin_events_opportunities.csv", index=False)
ch = []
for (o, e), g in PAIRS.groupby(["origin_node", "origin_eid"]):
    s = min(g.s0.min(), g.su.min()); en = max(g.e0.max(), g.eu.max())
    sev = ev_ix.loc[(o, e), "severity"] + sum(ev_ix.loc[(u, ue), "severity"] for u, ue in zip(g.upstream_node, g.upstream_eid))
    na = int(ORG.loc[(ORG.origin_node == o) & (ORG.origin_eid == e), "n_avail"].iloc[0])
    ch.append({"origin_node": o, "origin_event_id": e, "chain_start": F.EPOCH + np.timedelta64(int(s), "D"),
               "chain_end": F.EPOCH + np.timedelta64(int(en), "D"), "chain_duration": int(en - s + 1),
               "chain_size": len(g) + 1, "propagation_fraction": len(g) / na, "mean_lag": float(g.lag.mean()),
               "chain_severity": round(float(sev), 4)})
CH = pd.DataFrame(ch); CH.to_csv(T / "Chains_final.csv", index=False)
R["main"] = main
dist = CH.chain_size.value_counts().sort_index()
R.update({"chains_n": len(CH), "chain_size_mean": round(float(CH.chain_size.mean()), 4),
          "chain_size_max": int(CH.chain_size.max()), "chain_size_distribution": {int(k): int(v) for k, v in dist.items()},
          "share_size2": round(float((CH.chain_size == 2).mean()), 4),
          "pf_median": round(float(CH.propagation_fraction.median()), 3),
          "lag_median_pairs": float(P.lag.median()), "lag_iqr_pairs": [float(P.lag.quantile(.25)), float(P.lag.quantile(.75))],
          "chain_duration_median": float(CH.chain_duration.median()),
          "event_duration_median": float(EV.duration.median()),
          "event_duration_iqr": [float(EV.duration.quantile(.25)), float(EV.duration.quantile(.75))]})

# ---------------------------------------------------------------------------
log("R2 node density")
for tag in DENSITY:
    rows.append(run_config(f"density {tag} 1961-2020 M5", tag, 1961, 2020))

log("R3 vintage and periods")
for y0, y1, lab in ((1961, 2025, "vintage"), (1961, 2007, "split"), (2008, 2025, "split")):
    rows.append(run_config(f"{lab} {MAIN} {y0}-{y1} M5", MAIN, y0, y1))

# ---------------------------------------------------------------------------
log("R4 W x M grid, M = 3 and 7")
grid = []
for W in range(0, 121, 5):
    for M in range(1, 31):
        st = F.statistics(RECS, CONN, W=W, M=M)
        grid.append({"W": W, "M": M, **{k: st[k] for k in ("n_pairs", "n_connected", "mean_chain_size",
                                                             "mean_pf_published", "cf_opp", "p_link")}})
G = pd.DataFrame(grid); G.to_csv(T / "sensitivity_grid.csv", index=False)
R["rsi"] = {f"{m}_{p}": relative_sensitivity_index(G, m, p) for m in ("n_pairs", "mean_chain_size", "mean_pf_published", "cf_opp")
            for p in ("W", "M")}
g45 = G[G.W == 45].sort_values("M")
R["kneedle_M_pairs"] = kneedle_elbow(g45.M, g45.n_pairs, curve="convex", direction="decreasing")
R["kneedle_M_chain_size"] = kneedle_elbow(g45.M, g45.mean_chain_size, curve="convex", direction="decreasing")
sets = {}
for M in (3, 5, 7):
    _, o = F.match(RECS, CONN, M=M)
    sets[M] = set(map(tuple, o.loc[o.connected, ["origin_node", "origin_eid"]].values))
for M in (3, 7):
    rows.append(run_config(f"M{M} {MAIN} 1961-2020", MAIN, 1961, 2020, M=M, ev=EV))
    R[f"chains_M{M}_jaccard_vs_M5"] = round(len(sets[M] & sets[5]) / len(sets[M] | sets[5]), 3)

# ---------------------------------------------------------------------------
log("R5 detection sensitivity")
base = EV.groupby("node_id").size()
s5 = []
for gap in (0, 5, 10, 15):
    for md in (3, 5, 7, 10):
        e = EV if (gap, md) == (10, 5) else events_for(SSI, EventConfig(pool_gap_days=gap, min_duration_days=md))
        st = F.statistics(F.build_records(e, avail=F.availability_from_daily(SSI), nodes=list(CONN)), CONN)
        cnt = e.groupby("node_id").size().reindex(base.index).fillna(0)
        s5.append({"pool_gap": gap, "min_duration": md, "n_events": len(e),
                   "pct_change": round(100 * (len(e) - len(EV)) / len(EV), 1),
                   "rho_count": round(float(stats.spearmanr(base, cnt)[0]), 3),
                   "cf_opp": round(st["cf_opp"], 3), "p_link": round(st["p_link"], 3),
                   "mean_chain_size": round(st["mean_chain_size"], 2)})
R["s5"] = s5; pd.DataFrame(s5).to_csv(T / "tableS5_detection_sensitivity.csv", index=False)

# ---------------------------------------------------------------------------
log("R6 non-propagating events")
op = ORG[ORG.n_observing > 0].merge(EV[["node_id", "event_id", "duration", "severity"]],
                                    left_on=["origin_node", "origin_eid"], right_on=["node_id", "event_id"])
c, u = op[op.connected], op[~op.connected]
R["s7"] = {k: {"connected_median": float(c[k].median()), "unconnected_median": float(u[k].median()),
               "p_mw_two_sided": float(stats.mannwhitneyu(c[k], u[k]).pvalue)} for k in ("duration", "severity")}

# ---------------------------------------------------------------------------
log("R7 lag against along-channel distance")
pdist = pd.read_csv(NET / MAIN / "pair_distance.csv")
lp = P.groupby(["origin_node", "upstream_node"]).lag.median().abs().rename("abs_lag").reset_index()
lp = lp.merge(pdist, left_on=["origin_node", "upstream_node"], right_on=["origin", "upstream"])
rho, p = stats.spearmanr(lp.net_km, lp.abs_lag)
R["s3"] = {"pairs": len(lp), "links": len(pdist), "rho_pair": round(float(rho), 3), "p_pair": float(p),
           "median_net_km": float(lp.net_km.median())}
lp.to_csv(T / "lag_vs_distance.csv", index=False)

# ---------------------------------------------------------------------------
log("R8 transboundary propagation")
cty = NODES.country.to_dict()
P["origin_country"] = P.origin_node.map(cty); P["upstream_country"] = P.upstream_node.map(cty)
xb = P[P.origin_country != P.upstream_country]
routes = (xb.upstream_country + "->" + xb.origin_country).value_counts()
gh = ORG[ORG.origin_node.map(cty) == "Ghana"]
foreign = {o for o in gh.origin_node.unique() if any(cty[u] != "Ghana" for u in CONN[o])}
ghf = gh[gh.origin_node.isin(foreign)]
fp = xb[xb.origin_country == "Ghana"].groupby(["origin_node", "origin_eid"]).lag.min().rename("lag").reset_index()
ghf = ghf.merge(fp, on=["origin_node", "origin_eid"], how="left")
lead = -ghf.lag.dropna()
R["transboundary"] = {"pairs_total": len(P), "pairs_cross_border": len(xb), "share_cross_border": round(len(xb) / len(P), 4),
                      "cross_border_by_route": routes.to_dict(),
                      "ghana_origin_events_with_foreign_upstream": len(ghf),
                      "share_with_foreign_precursor": round(float(ghf.lag.notna().mean()), 4),
                      "lead_days_median": float(lead.median()), "lead_days_iqr": [float(lead.quantile(.25)), float(lead.quantile(.75))],
                      "lead_ge_7d_share_of_events": round(float((lead >= 7).sum() / len(ghf)), 4),
                      "lead_ge_30d_share_of_events": round(float((lead >= 30).sum() / len(ghf)), 4)}

# ---------------------------------------------------------------------------
log("R9 Sahel corroboration")
raw = Path("data/external/sahelrain.data").read_text().splitlines()
recs_ = [[int(x[0])] + [float(v) for v in x[1:]] for x in (l.split() for l in raw[1:]) if len(x) == 13 and x[0].isdigit()]
sp = pd.DataFrame(recs_, columns=["year"] + list(range(1, 13))).set_index("year")
jj = sp[[6, 7, 8, 9, 10]].sum(axis=1); ylast = int(jj.index.max())
z = (jj - jj.loc[1961:2007].mean()) / jj.loc[1961:2007].std()
dry = set(z[z <= -0.5].index)
yrs = SSI[SSI.date.dt.year <= ylast].assign(y=lambda d: d.date.dt.year).groupby("y").SSI.apply(lambda s: s.notna().sum() / 365.25)
evy = EV.assign(y=EV.start_date.dt.year).query("y <= @ylast").groupby("y").size().reindex(yrs.index).fillna(0)
dm = yrs.index.isin(list(dry))
n1, t1, n0, t0 = evy[dm].sum(), yrs[dm].sum(), evy[~dm].sum(), yrs[~dm].sum()
CHs = CH[pd.to_datetime(CH.chain_start).dt.year <= ylast]
cd = CHs.apply(lambda r: any(y in dry for y in range(pd.Timestamp(r.chain_start).year, pd.Timestamp(r.chain_end).year + 1)), axis=1)
cover = float(t1 / (t1 + t0))
R["s8"] = {"years": [1961, ylast], "dry_years": sorted(int(y) for y in dry if y >= 1961), "rate_ratio": round(float((n1 / t1) / (n0 / t0)), 2),
           "p_rate_ratio": float(stats.binomtest(int(n1), int(n1 + n0), cover, alternative="greater").pvalue),
           "chains_in_dry_years": int(cd.sum()), "chains": len(CHs), "exposure_share_dry": round(cover, 3),
           "share_chains_dry": round(float(cd.mean()), 3),
           "p_chains": float(stats.binomtest(int(cd.sum()), len(CHs), cover, alternative="greater").pvalue)}

S = pd.DataFrame(rows); S.to_csv(T / "reanalysis_summary.csv", index=False)
R["summary"] = S.to_dict(orient="records")
(OUT / "reanalysis_numbers.json").write_text(json.dumps(R, indent=2, default=str))
log("done")
