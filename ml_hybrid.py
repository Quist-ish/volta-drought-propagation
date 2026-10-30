#!/usr/bin/env python3
"""
Observation-anchored reconstruction by random forest (Sections 2.2.8, 3.5).

Target: the observed daily SSI at each gauge. Predictors are taken at the same
gauge only (no upstream or downstream station enters any model):
  reanalysis SSI at the matched reach and its 7-, 30- and 90-day means; the 30-day
  change in log reanalysis discharge; day-of-year harmonics; the Sahel index of the
  current and previous monsoon season; log catchment area.
Skill: leave-one-year-out and leave-one-station-out cross-validation (pooled R2,
Spearman rho, Cohen's kappa for drought days), against the co-masked reanalysis
index on the same days. Event skill of the leave-one-year-out predictions.
Hybrid: predictions fill only the missing days inside each station's span (and,
for comparison, the whole of 1961-2007); propagation is recomputed with the
availability-aware null.

Writes output_ml/ml_numbers.json and tables/{rf_skill_by_station, hybrid_propagation}.csv.
Author: I. Quist (NUIST)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import cohen_kappa_score, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voltaprop import fastmatch as F
from voltaprop.ssi import SSIConfig, compute_ssi_node
from voltaprop.events import EventConfig, detect_network_events
from voltaprop.propagation import load_connectivity

MAIN, TAU = "t2500", -1.28
OUT = Path("output_ml"); T = OUT / "tables"; T.mkdir(parents=True, exist_ok=True)
TO, TR = Path("output_observational/tables"), Path("output_reanalysis/tables")
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
RF = dict(n_estimators=200, min_samples_leaf=20, max_features=0.5, n_jobs=2, random_state=42)

gm = pd.read_csv(f"data/reanalysis/{MAIN}/gauge_matches.csv").dropna(subset=["river_id"])
g2r = dict(zip(gm.grdc_no.astype(int), gm.river_id.astype(int)))
st = pd.read_csv("data/grdc_stations_raw.csv").set_index("grdc_no")
obs = pd.read_csv(TO / "SSI_daily.csv", parse_dates=["date"])
oev = pd.read_csv(TO / "SSI_drought_events.csv", parse_dates=["start_date", "end_date"])
conn = load_connectivity(TO / "upstream_connectivity.csv")
S = pd.read_parquet(TR / "SSI_daily.parquet"); S["date"] = pd.to_datetime(S.date)

raw = Path("data/external/sahelrain.data").read_text().splitlines()
rec = [[int(x[0])] + [float(v) for v in x[1:]] for x in (l.split() for l in raw[1:]) if len(x) == 13 and x[0].isdigit()]
sp = pd.DataFrame(rec, columns=["year"] + list(range(1, 13))).set_index("year")
jj = sp[[6, 7, 8, 9, 10]].sum(axis=1); z = (jj - jj.loc[1961:2007].mean()) / jj.loc[1961:2007].std()

FEATS = ["f_ssi_ra", "f_ssi_ra_7", "f_ssi_ra_30", "f_ssi_ra_90", "f_dlogq_30", "f_doy_sin", "f_doy_cos",
         "f_sahel_cur", "f_sahel_prev", "f_log_area"]
frames = []
for g, rid in g2r.items():
    s = S[(S.node_id == rid) & (S.date.dt.year >= 1960) & (S.date.dt.year <= 2007)].set_index("date").sort_index()
    f = pd.DataFrame(index=s.index)
    f["f_ssi_ra"] = s.SSI
    for w in (7, 30, 90):
        f[f"f_ssi_ra_{w}"] = s.SSI.rolling(w, min_periods=w // 2).mean()
    lq = np.log(s.Q + 0.01 * s.Q.mean())
    f["f_dlogq_30"] = lq - lq.shift(30)
    doy = f.index.dayofyear
    f["f_doy_sin"], f["f_doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    f["f_sahel_cur"] = z.reindex(f.index.year).values
    f["f_sahel_prev"] = z.reindex(f.index.year - 1).values
    f["f_log_area"] = np.log(st.loc[g, "area_km2"])
    o = obs[obs.node_id == g].set_index("date")
    f["y"] = o.SSI.reindex(f.index); f["q_obs"] = o.Q.reindex(f.index)
    f["node_id"] = g; f["year"] = f.index.year
    # co-masked reanalysis index (refitted on the reanalysis series on observed days)
    qm = s.Q.where(s.index.isin(o.index[o.Q.notna()]))
    if o.Q.notna().any():
        cm, _ = compute_ssi_node(qm.loc[o.index[o.Q.notna()].min():o.index[o.Q.notna()].max()], SSIConfig())
        f["ssi_comasked"] = cm.reindex(f.index)
    else:
        f["ssi_comasked"] = np.nan
    f["span"] = (f.index >= o.index.min()) & (f.index <= o.index.max())
    frames.append(f[f.index.year >= 1961])
D = pd.concat(frames).reset_index()
train = D.dropna(subset=["y"] + FEATS).reset_index(drop=True)
log(f"{len(train):,} training rows at {train.node_id.nunique()} gauges")


def cv(groups):
    pred = np.full(len(train), np.nan)
    for gval in np.unique(groups):
        te = groups == gval
        m = RandomForestRegressor(**RF).fit(train.loc[~te, FEATS], train.loc[~te, "y"])
        pred[te] = m.predict(train.loc[te, FEATS])
    return pred


def skill(y, p):
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    return {"n": int(ok.sum()), "r2": round(float(r2_score(y, p)), 4), "spearman": round(float(stats.spearmanr(y, p)[0]), 4),
            "kappa": round(float(cohen_kappa_score(y < TAU, p < TAU)), 4)}


log("leave-one-year-out"); train["p_loyo"] = cv(train.year.values)
log("leave-one-station-out"); train["p_loso"] = cv(train.node_id.values)
y = train.y.values
ML = {"n_training_rows": len(train), "skill": {"pooled_loyo": skill(y, train.p_loyo.values),
                                               "pooled_loso": skill(y, train.p_loso.values),
                                               "pooled_reanalysis_comasked_same_days": skill(y, train.ssi_comasked.values)}}
rows = []
for g, d in train.groupby("node_id"):
    s_ = lambda c: skill(d.y.values, d[c].values)
    a, b, c = s_("ssi_comasked"), s_("p_loyo"), s_("p_loso")
    rows.append({"station": st.loc[g, "station_name"], "node_id": g, "n_days": len(d),
                 "reanalysis_spearman": a["spearman"], "loyo_spearman": b["spearman"], "loso_spearman": c["spearman"],
                 "reanalysis_kappa": a["kappa"], "loyo_kappa": b["kappa"], "loso_kappa": c["kappa"]})
pd.DataFrame(rows).to_csv(T / "rf_skill_by_station.csv", index=False)

full = RandomForestRegressor(**RF).fit(train[FEATS], train.y)
imp = pd.Series(full.feature_importances_, index=FEATS).sort_values(ascending=False)
ML["feature_importance"] = {k: round(float(v), 3) for k, v in imp.items()}


def events_from(df, col):
    x = df[["node_id", "date", col]].rename(columns={col: "SSI"})
    return detect_network_events(x, EventConfig(), progress=False), x


def hit_stats(oe_all, se_all, tol=10):
    h_o = h_s = n_o = n_s = 0
    for g in oe_all.node_id.unique():
        oe, se = oe_all[oe_all.node_id == g], se_all[se_all.node_id == g]
        n_o += len(oe); n_s += len(se); hs = np.zeros(len(se), bool)
        for a, b in zip(oe.start_date, oe.end_date):
            m = ((se.start_date <= b) & (se.end_date >= a)) | ((se.start_date - a).abs() <= pd.Timedelta(days=tol))
            h_o += bool(m.any()); hs |= m.values
        h_s += int(hs.sum())
    return {"POD": round(h_o / n_o, 3) if n_o else np.nan, "FAR": round(1 - h_s / n_s, 3) if n_s else np.nan,
            "CSI": round(h_o / (n_o + n_s - h_s), 3) if (n_o + n_s) else np.nan, "n_obs": n_o, "n_pred": n_s}


cal = obs[["node_id", "date"]]
lo = cal.merge(train[["node_id", "date", "p_loyo"]], on=["node_id", "date"], how="left")
se_loyo, _ = events_from(lo, "p_loyo")
ML["event_skill_loyo"] = hit_stats(oev, se_loyo)

# ---------------------------------------------------------------------------
# Hybrid series and propagation
# ---------------------------------------------------------------------------
feat_ok = D[FEATS].notna().all(axis=1)
D["pred"] = np.nan
D.loc[feat_ok, "pred"] = full.predict(D.loc[feat_ok, FEATS])
missing = D.q_obs.isna()
D["h_span"] = np.where(D.y.notna(), D.y, np.where(missing & D.span, D.pred, np.nan))
D["h_full"] = np.where(D.y.notna(), D.y, np.where(missing, D.pred, np.nan))
ML["hybrid_share_observed"] = {"span": round(float(D.y.notna().sum() / D.h_span.notna().sum()), 3),
                               "full": round(float(D.y.notna().sum() / D.h_full.notna().sum()), 3)}
series = {"observed": obs[["node_id", "date", "SSI"]].rename(columns={"SSI": "s"}),
          "hybrid, within spans": D[["node_id", "date", "h_span"]].rename(columns={"h_span": "s"}),
          "hybrid, full 1961-2007": D[["node_id", "date", "h_full"]].rename(columns={"h_full": "s"})}
hyb, ORG = [], {}
for name, df in series.items():
    df = df.dropna(subset=["s"]) if name != "observed" else df
    ev, x = events_from(df, "s")
    recs = F.build_records(ev, avail=F.availability_from_daily(x), nodes=list(conn))
    _, org = F.match(recs, conn); ORG[name] = (org, ev)
    o, summ, _ = F.null_test(recs, conn, n_perm=1999, seed=42)
    gg = summ.set_index("statistic")
    hyb.append({"series": name, "n_events": len(ev), "n_origin_events": o["n_origin_events"],
                "n_opportunities": o["n_opportunities"], "n_connected": o["n_connected"], "cf_opp": round(o["cf_opp"], 3),
                "p_link": round(o["p_link"], 4), "p_cf_opp": round(float(gg.loc["cf_opp", "p"]), 4),
                "p_p_link": round(float(gg.loc["p_link", "p"]), 4)})
    log(f"  {hyb[-1]}")
pd.DataFrame(hyb).to_csv(T / "hybrid_propagation.csv", index=False)
ML["hybrid"] = hyb

oorg, _ = ORG["observed"]
un = oorg[(oorg.n_observing > 0) & ~oorg.connected]
horg, hev = ORG["hybrid, within spans"]
k = 0
for _, r_ in un.iterrows():
    h = horg[(horg.origin_node == r_.origin_node) & (horg.s0 <= r_.e0) & (horg.e0 >= r_.s0)]
    k += bool(h.connected.any())
ML["unconnected_opportunities"] = {"n": len(un), "connected_under_hybrid": k}
(OUT / "ml_numbers.json").write_text(json.dumps(ML, indent=2, default=str))
log(json.dumps({k_: ML[k_] for k_ in ("n_training_rows", "skill", "event_skill_loyo", "hybrid_share_observed",
                                       "unconnected_opportunities")}))
