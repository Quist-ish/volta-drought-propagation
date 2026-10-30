#!/usr/bin/env python3
"""
Reanalysis arm, step 5: benchmark against the 18 GRDC gauges, and the
observing-system experiment (OSSE).

Benchmark (observed days only; Section 2.2.7):
  discharge   KGE and its components (Gupta et al., 2009), log NSE, percent bias;
  index       Spearman rho of daily SSI, with the reanalysis index refitted on the
              reanalysis series masked to the observed days (co-masked) and with
              the full-period index; Cohen's kappa for the daily drought state;
  events      POD, FAR, CSI; a hit is a reanalysis event that overlaps an observed
              event or begins within 10 days of it.
Observed chains: whether the main reanalysis network holds the origin event and
the same upstream-downstream pair.

OSSE at the 18 gauge nodes, GRDC connectivity, 1961-2007:
  continuous  reanalysis index (fitted 1961-2020), no gaps;
  masked      the same index, kept only on days the gauge observed;
  refitted    index refitted on the observed days alone (as at the gauges).
Each is tested with the availability-aware null (1,999 permutations).

Writes output_reanalysis/benchmark_numbers.json and tables/benchmark_*.csv, osse_summary.csv.
Author: I. Quist (NUIST)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voltaprop import fastmatch as F
from voltaprop.ssi import SSIConfig, compute_ssi_node
from voltaprop.events import EventConfig, detect_network_events
from voltaprop.propagation import load_connectivity

MAIN = "t2500"
T = Path("output_reanalysis/tables"); TO = Path("output_observational/tables")
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
TAU = -1.28

gm = pd.read_csv(f"data/reanalysis/{MAIN}/gauge_matches.csv").dropna(subset=["river_id"])
g2r = {int(a): int(b) for a, b in zip(gm.grdc_no, gm.river_id)}
area_err = dict(zip(gm.grdc_no.astype(int), gm.area_err))
st = pd.read_csv("data/grdc_stations_raw.csv").set_index("grdc_no")
obs = pd.read_csv(TO / "SSI_daily.csv", parse_dates=["date"])
oev = pd.read_csv(TO / "SSI_drought_events.csv", parse_dates=["start_date", "end_date"])
conn_g = load_connectivity(TO / "upstream_connectivity.csv")
S = pd.read_parquet(T / "SSI_daily.parquet"); S["date"] = pd.to_datetime(S.date)
S = S[S.node_id.isin(set(g2r.values()))]


def kge(s, o):
    r = np.corrcoef(s, o)[0, 1]; a = s.std() / o.std(); b = s.mean() / o.mean()
    return {"KGE": 1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2), "r": r, "alpha": a, "beta": b}


def hits(oe, se, tol=10):
    """Observed events with a reanalysis counterpart, reanalysis events with an observed one, onset errors."""
    oh = np.zeros(len(oe), bool); sh = np.zeros(len(se), bool); err = []
    for i, (a, b) in enumerate(zip(oe.start_date, oe.end_date)):
        m = ((se.start_date <= b) & (se.end_date >= a)) | ((se.start_date - a).abs() <= pd.Timedelta(days=tol))
        if m.any():
            oh[i] = True; sh |= m.values
            d = (se.start_date[m] - a).dt.days
            err.append(int(d.iloc[np.argmin(np.abs(d.values))]))
    return oh, sh, err


rows, ERR, EVS = [], [], []
for g, rid in g2r.items():
    o = obs[obs.node_id == g].set_index("date")
    od = o[o.Q.notna() & (o.fill_tier == 0)]
    s = S[S.node_id == rid].set_index("date")
    d = od.index.intersection(s.index)
    qo, qs = od.loc[d, "Q"].values, s.loc[d, "Q"].values
    k = kge(qs, qo)
    eps = 0.01 * qo.mean()
    lo, ls = np.log(qo + eps), np.log(qs + eps)
    lnse = 1 - np.sum((ls - lo) ** 2) / np.sum((lo - lo.mean()) ** 2)
    # co-masked reanalysis index: the reanalysis series on observed days only, fitted as at the gauge
    qm = s.Q.where(s.index.isin(o.index[o.Q.notna()]))
    ssi_cm, _ = compute_ssi_node(qm.loc[o.index.min():o.index.max()], SSIConfig())
    both = o.SSI.notna() & ssi_cm.reindex(o.index).notna()
    a, b = o.SSI[both], ssi_cm.reindex(o.index)[both]
    full = s.SSI.reindex(o.index)
    bf = o.SSI.notna() & full.notna()
    sev = detect_network_events(pd.DataFrame({"node_id": g, "date": ssi_cm.index, "SSI": ssi_cm.values}),
                                EventConfig(), progress=False)
    EVS.append(sev)
    oe = oev[oev.node_id == g].reset_index(drop=True)
    oh, sh, err = hits(oe, sev); ERR += err
    rows.append({"station": st.loc[g, "station_name"], "grdc_no": g, "river_id": rid, "area_err": area_err[g],
                 "n_days_obs": len(d), **{f"KGE_{kk}": round(float(v), 3) for kk, v in k.items()},
                 "logNSE": round(float(lnse), 3), "pbias": round(float(100 * (qs.sum() - qo.sum()) / qo.sum()), 1),
                 "Q10_ratio": round(float(np.percentile(qs, 10) / max(np.percentile(qo, 10), 1e-6)), 3),
                 "rho_SSI_comasked": round(float(stats.spearmanr(a, b)[0]), 3) if len(a) > 10 else np.nan,
                 "rho_SSI_full": round(float(stats.spearmanr(o.SSI[bf], full[bf])[0]), 3) if bf.sum() > 10 else np.nan,
                 "kappa_drought_days": round(float(cohen_kappa_score(a < TAU, b < TAU)), 3) if len(a) > 10 else np.nan,
                 "n_obs": len(oe), "n_sim": len(sev), "hits_obs": int(oh.sum()), "hits_sim": int(sh.sum()),
                 "POD": round(oh.mean(), 3) if len(oe) else np.nan,
                 "FAR": round(1 - sh.mean(), 3) if len(sev) else np.nan,
                 "CSI": round(oh.sum() / (len(oe) + (len(sev) - sh.sum())), 3) if (len(oe) + len(sev)) else np.nan})
B = pd.DataFrame(rows); B.to_csv(T / "benchmark_stations.csv", index=False)
pooled_hits, n_obs, fa = B.hits_obs.sum(), B.n_obs.sum(), (B.n_sim - B.hits_sim).sum()
ERR = np.array(ERR)
BN = {"benchmark": {
    "gauges": len(B), "KGE_median": round(float(B.KGE_KGE.median()), 2),
    "KGE_range": [round(float(B.KGE_KGE.min()), 2), round(float(B.KGE_KGE.max()), 2)],
    "KGE_r_median": round(float(B.KGE_r.median()), 2), "KGE_beta_median": round(float(B.KGE_beta.median()), 2),
    "logNSE_median": round(float(B.logNSE.median()), 2), "pbias_median": round(float(B.pbias.median()), 1),
    "Q10_ratio_median": round(float(B.Q10_ratio.median()), 2),
    "rho_SSI_comasked_median": round(float(B.rho_SSI_comasked.median()), 2),
    "rho_SSI_full_median": round(float(B.rho_SSI_full.median()), 2),
    "kappa_median": round(float(B.kappa_drought_days.median()), 2),
    "pooled_POD": round(float(pooled_hits / n_obs), 2), "pooled_FAR": round(float(fa / B.n_sim.sum()), 2),
    "pooled_CSI": round(float(pooled_hits / (n_obs + fa)), 2),
    "onset_error_pooled": {"n": int(len(ERR)), "share_within_10d": round(float(np.mean(np.abs(ERR) <= 10)), 3),
                           "median": float(np.median(ERR)), "mae": round(float(np.mean(np.abs(ERR))), 1)}}}
log(json.dumps(BN["benchmark"]))

# ---------------------------------------------------------------------------
# Observed chains in the main reanalysis network
# ---------------------------------------------------------------------------
rp = pd.read_csv(T / "propagation_pairs.csv", parse_dates=["origin_start", "origin_end", "upstream_start", "upstream_end"])
rev = pd.read_csv(T / "SSI_drought_events_1961_2020.csv", parse_dates=["start_date", "end_date"])
op = pd.read_csv(TO / "propagation_pairs.csv", parse_dates=["origin_start", "origin_end"])
bc = []
for _, r in op.iterrows():
    ro, ru = g2r[r.origin_node], g2r[r.upstream_node]
    e = rev[(rev.node_id == ro) & (rev.start_date <= r.origin_end) & (rev.end_date >= r.origin_start)]
    q = rp[(rp.origin_node == ro) & (rp.upstream_node == ru) & (rp.origin_start <= r.origin_end) & (rp.origin_end >= r.origin_start)]
    bc.append({"origin": st.loc[r.origin_node, "station_name"], "upstream": st.loc[r.upstream_node, "station_name"],
               "origin_onset": r.origin_start.date().isoformat(), "observed_lag": int(r.lag_days),
               "reanalysis_origin_event": e.start_date.min().date().isoformat() if len(e) else "",
               "reanalysis_pair": "yes" if len(q) else "no", "reanalysis_lag": int(q.lag.iloc[0]) if len(q) else np.nan})
BC = pd.DataFrame(bc); BC.to_csv(T / "benchmark_chains.csv", index=False)
BN["chains_reproduced"] = {"observed_chains": len(BC), "origin_event_in_reanalysis": int((BC.reanalysis_origin_event != "").sum()),
                           "pair_in_reanalysis": int((BC.reanalysis_pair == "yes").sum())}

# ---------------------------------------------------------------------------
# OSSE
# ---------------------------------------------------------------------------
r2g = {v: k for k, v in g2r.items()}
conn = {g: list(u) for g, u in conn_g.items()}
cont = S[(S.date.dt.year >= 1961) & (S.date.dt.year <= 2007)].assign(node_id=lambda d: d.node_id.map(r2g))
obs_days = obs[obs.Q.notna()][["node_id", "date"]].assign(o=True)
masked = cont.merge(obs_days, on=["node_id", "date"], how="left")
masked.loc[masked.o.isna(), "SSI"] = np.nan
refit = []
for g in g2r:
    m = masked[masked.node_id == g].set_index("date")
    q = m.Q.where(m.o.notna())
    days = q.dropna().index
    if not len(days):
        continue
    s_, _ = compute_ssi_node(q.loc[days.min():days.max()], SSIConfig())
    refit.append(pd.DataFrame({"node_id": g, "date": s_.index, "SSI": s_.values}))
refit = pd.concat(refit)
osse = []
for name, df in (("continuous", cont[["node_id", "date", "SSI"]]),
                 ("masked to observed days", masked[["node_id", "date", "SSI"]]),
                 ("refitted on observed days", refit)):
    df = df[df.node_id.isin(conn)]
    keep = df.groupby("node_id").SSI.apply(lambda s: s.notna().any())
    df = df[df.node_id.isin(keep[keep].index)]
    ev = detect_network_events(df, EventConfig(), progress=False)
    recs = F.build_records(ev, avail=F.availability_from_daily(df), nodes=list(conn))
    o, summ, _ = F.null_test(recs, conn, n_perm=1999, seed=42)
    g = summ.set_index("statistic")
    osse.append({"scenario": name, "n_events": len(ev), **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()},
                 **{f"null_{k}": round(float(g.loc[k, "null_mean"]), 4) for k in ("cf_all", "cf_opp", "p_link")},
                 **{f"p_{k}": float(g.loc[k, "p"]) for k in ("cf_all", "cf_opp", "p_link")}})
    log(f"  OSSE {name}: {osse[-1]}")
pd.DataFrame(osse).to_csv(T / "osse_summary.csv", index=False)
BN["osse"] = osse
Path("output_reanalysis/benchmark_numbers.json").write_text(json.dumps(BN, indent=2, default=str))
log("done")
