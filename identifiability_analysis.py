#!/usr/bin/env python3
"""
Identifiability of propagation on sparse networks (Sections 2.2.4, 2.2.6, 3.3;
Appendix A, Propositions 3 and 4).

  1. Connectivity measures for the Ebro (released catalogue, 101 links) and the
     observed Volta: coverage, observing links, cf_all, cf_opp, per-link
     probability p (MLE), calendar-year block-bootstrap intervals, and the
     availability-aware network-label null (1,999 permutations).
  2. The per-link model by k, and the stratified comparison of the two basins
     within strata of equal k (Mantel-Haenszel odds ratio, Breslow-Day test).
  3. Reference-equivalent connected fractions.
  4. Ebro degradation: (a) thinning to k of 33 stations with complete records;
     (b) every retained catalogue cut to one window with a Volta-like record
     length. Each draw is tested with the availability-aware null (199
     permutations). Power is modelled by logistic regression of rejection on
     log(opportunities).
  5. The analytic sample size n* (Proposition 4).

Writes output_identifiability/identifiability_numbers.json and tables/*.csv.
Author: I. Quist (NUIST)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.contingency_tables import StratifiedTable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voltaprop import fastmatch as F
from voltaprop.propagation import load_connectivity

OUT = Path("output_identifiability"); T = OUT / "tables"; T.mkdir(parents=True, exist_ok=True)
TO = Path("output_observational/tables")
N_BOOT, N_PERM, N_PERM_DEG, N_DRAWS = 2000, 1999, 199, 40
SIZES = (5, 8, 11, 15, 21, 27)
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
R = {}

# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
eb = pd.read_csv("data/ebro/ebro_events.csv", parse_dates=["start_date", "end_date"]).rename(columns={"station_id": "node_id"})
ep = pd.read_csv("data/ebro/ebro_pairs_net_km.csv")
econn = {int(s): [] for s in eb.node_id.unique()}
for _, r in ep.iterrows():
    econn[int(r.origin_station)].append(int(r.upstream_station))
erecs = F.build_records(eb)

ev = pd.read_csv(TO / "SSI_drought_events.csv", parse_dates=["start_date", "end_date"])
vconn = load_connectivity(TO / "upstream_connectivity.csv")
ssi = pd.read_csv(TO / "SSI_daily.csv", parse_dates=["date"])
vrecs = F.build_records(ev, avail=F.availability_from_daily(ssi), nodes=list(vconn))
NETS = {"Ebro": (erecs, econn), "Volta observed": (vrecs, vconn)}


def opportunities(recs, conn):
    _, org = F.match(recs, conn)
    org = org.copy()
    org["year"] = pd.to_datetime(org.s0_date).dt.year
    return org


def block_bootstrap(org, n=N_BOOT, seed=2026):
    """Calendar-year block bootstrap of cf_opp and p over opportunities."""
    opp = org[org.n_observing > 0]
    years = np.array(sorted(opp.year.unique()))
    by = {y: g for y, g in opp.groupby("year")}
    rng = np.random.default_rng(seed)
    cf, pl = [], []
    for _ in range(n):
        g = pd.concat([by[y] for y in rng.choice(years, len(years), replace=True)])
        cf.append(g.connected.mean()); pl.append(F.link_probability_mle(g.n_observing.values, g.connected.values))
    q = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]
    return q(cf), q(pl), np.array(pl)


rows, ORG, PL_BOOT, NULLS = [], {}, {}, {}
for name, (recs, conn) in NETS.items():
    log(f"{name}: statistics, bootstrap, null")
    org = opportunities(recs, conn); ORG[name] = org
    cfi, pli, PL_BOOT[name] = block_bootstrap(org)
    obs, summ, null = F.null_test(recs, conn, n_perm=N_PERM, seed=42)
    NULLS[name] = null
    null.to_csv(T / ("null_ebro.csv" if name == "Ebro" else "null_volta_obs.csv"), index=False)
    g = summ.set_index("statistic")
    rows.append({"network": name, **{k: obs[k] for k in ("n_origin_events", "n_opportunities", "n_connected", "n_pairs",
                                                       "coverage", "mean_k_observing", "cf_all", "cf_opp", "p_link",
                                                       "mean_chain_size", "mean_pf_published")},
                 "cf_opp_ci": cfi, "p_link_ci": pli,
                 **{f"null_{k}": float(g.loc[k, "null_mean"]) for k in ("cf_all", "cf_opp", "p_link", "mean_chain_size")},
                 **{f"p_{k}": float(g.loc[k, "p"]) for k in ("cf_all", "cf_opp", "p_link", "mean_chain_size")}})
NS = pd.DataFrame(rows)
NS.to_csv(T / "networks_summary.csv", index=False)
R["networks"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]
nsi = NS.set_index("network")

# ---------------------------------------------------------------------------
# Per-link model by k, Proposition 3 check, stratified comparison
# ---------------------------------------------------------------------------
fit = []
for name in NETS:
    o = ORG[name]; o = o[o.n_observing > 0]; p = nsi.loc[name, "p_link"]
    for k, g in o.groupby("n_observing"):
        fit.append({"network": name, "k_observing": int(k), "n": len(g), "observed_cf": round(g.connected.mean(), 4),
                    "predicted": round(1 - (1 - p) ** k, 4)})
pd.DataFrame(fit).to_csv(T / "per_link_model_fit.csv", index=False)
R["proposition2_check"] = {}
for name in NETS:
    o = ORG[name]; op = o[o.n_observing > 0]; p = nsi.loc[name, "p_link"]; c = len(op) / len(o)
    pred = float(np.mean(1 - (1 - p) ** op.n_observing))
    R["proposition2_check"][name] = {"coverage": round(c, 4), "predicted_cf_opp": round(pred, 4),
                                     "observed_cf_opp": round(float(op.connected.mean()), 4),
                                     "predicted_cf_all": round(c * pred, 4), "observed_cf_all": round(float(o.connected.mean()), 4)}

vo = ORG["Volta observed"]; vo = vo[vo.n_observing > 0]
eo = ORG["Ebro"]; eo = eo[eo.n_observing > 0]
strata, tabs = [], []
for k in sorted(set(vo.n_observing) | set(eo.n_observing)):
    a, b = vo[vo.n_observing == k], eo[eo.n_observing == k]
    row = {"k": int(k), "volta_n": len(a), "volta_connected": int(a.connected.sum()),
           "volta_share": round(a.connected.mean(), 3) if len(a) else np.nan,
           "ebro_n": len(b), "ebro_connected": int(b.connected.sum()),
           "ebro_share": round(b.connected.mean(), 3) if len(b) else np.nan, "fisher_p": np.nan}
    if len(a) and len(b):
        t = np.array([[a.connected.sum(), len(a) - a.connected.sum()], [b.connected.sum(), len(b) - b.connected.sum()]])
        row["fisher_p"] = round(float(stats.fisher_exact(t)[1]), 3)
        tabs.append(t.astype(float))
    if len(a):
        strata.append(row)
SK = pd.DataFrame(strata); SK.to_csv(T / "stratified_by_k.csv", index=False)
st_ = StratifiedTable([t + 0.0 for t in tabs])
ci = st_.oddsratio_pooled_confint()
R["stratified"] = {"mh_odds_ratio_volta_vs_ebro": round(float(st_.oddsratio_pooled), 2),
                   "mh_or_ci": [round(float(ci[0]), 2), round(float(ci[1]), 2)],
                   "mh_test_p": round(float(st_.test_null_odds().pvalue), 2),
                   "breslow_day_p": round(float(st_.test_equal_odds().pvalue), 2),
                   "strata": [int(x) for x in SK.dropna(subset=["fisher_p"]).k],
                   "volta_opportunities_in_shared_strata": int(SK.dropna(subset=["fisher_p"]).volta_n.sum())}

# reference-equivalent connected fractions
gE = eo.n_observing.value_counts(normalize=True); gV = vo.n_observing.value_counts(normalize=True)
eq = lambda g, p: float(sum(w * (1 - (1 - p) ** k) for k, w in g.items()))
pv, pe = nsi.loc["Volta observed", "p_link"], nsi.loc["Ebro", "p_link"]
R["ebro_equivalent_cf_volta"] = round(eq(gE, pv), 3)
boot = [eq(gE, p) for p in PL_BOOT["Volta observed"] if np.isfinite(p)]
R["ebro_equivalent_cf_volta_ci"] = [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)]
R["volta_equivalent_cf_ebro"] = round(eq(gV, pe), 3)

# ---------------------------------------------------------------------------
# Ebro degradation
# ---------------------------------------------------------------------------
vlen = (ssi.groupby("node_id").SSI.apply(lambda s: int(s.notna().sum())))
vlen = vlen[vlen > 0].values
R["volta_defined_record_days"] = {"median": int(np.median(vlen)), "min": int(vlen.min()), "max": int(vlen.max())}
day0, day1 = int(F.to_day("1961-01-01")), int(F.to_day("2020-12-31"))
ids = sorted(econn)
rng = np.random.default_rng(2026)


def cut(rec, a, b):
    """A record restricted to [a, b]: events wholly inside kept, availability = the window."""
    keep = (rec.start >= a) & (rec.end <= b)
    mask = np.zeros(day1 - day0 + 1, dtype=bool); mask[a - day0: b - day0 + 1] = True
    return F.Record(rec.start[keep], rec.end[keep], rec.sev[keep], rec.eid[keep], F.mask_to_cum(mask), day0)


def one_draw(mode, n_st, draw):
    sub = sorted(rng.choice(ids, n_st, replace=False).tolist()) if n_st < len(ids) else ids
    conn = {o: [u for u in econn[o] if u in sub] for o in sub}
    if mode == "full":
        recs = {s: erecs[s] for s in sub if s in erecs}
    else:
        recs = {}
        for s in sub:
            L = int(rng.choice(vlen)); a = int(rng.integers(day0, day1 - L + 2)); recs[s] = cut(erecs[s], a, a + L - 1)
    obs, summ, _ = F.null_test(recs, conn, n_perm=N_PERM_DEG, seed=10_000 + draw)
    g = summ.set_index("statistic")
    return {"mode": mode, "n_stations": n_st, "draw": draw, **obs,
            **{f"null_{k}": float(g.loc[k, "null_mean"]) for k in ("cf_all", "cf_opp", "p_link")},
            **{f"p_{k}": float(g.loc[k, "p"]) for k in ("cf_all", "cf_opp", "p_link")}}


DEG, i = [], 0
for mode, sizes in (("full", SIZES + (33,)), ("volta_records", SIZES + (33,))):
    for n_st in sizes:
        nd = 1 if (mode == "full" and n_st == 33) else N_DRAWS
        for d in range(nd):
            DEG.append(one_draw(mode, n_st, i)); i += 1
        log(f"  {mode} {n_st} stations: {nd} draws")
D = pd.DataFrame(DEG); D.to_csv(T / "ebro_degradation.csv", index=False)

full = D[D["mode"] == "full"]
th = full.groupby("n_stations")[["coverage", "cf_all", "cf_opp", "mean_k_observing", "p_link"]].median().round(3).reset_index()
th.to_csv(T / "ebro_thinning_stability.csv", index=False)
R["thinning_full"] = th.to_dict(orient="records")
R["thinning_p_link_range_median"] = [float(th.p_link.min()), float(th.p_link.max())]
vr33 = D[(D["mode"] == "volta_records") & (D.n_stations == 33)]
R["ebro_volta_like_33"] = {"draws": len(vr33), "median_coverage": round(float(vr33.coverage.median()), 3),
                           "median_cf_all": round(float(vr33.cf_all.median()), 3),
                           "median_opportunities": float(vr33.n_opportunities.median()),
                           "reject_cf_all": int((vr33.p_cf_all < 0.05).sum()),
                           "reject_cf_opp": int((vr33.p_cf_opp < 0.05).sum()),
                           "reject_p_link": int((vr33.p_p_link < 0.05).sum())}

d = D[D.n_opportunities > 0].copy()
R["degraded_draws_total"] = len(D); R["degraded_draws_with_opportunities"] = len(d)
bins = [1, 10, 20, 40, 80, 160, 320, 640, 10_000]
d["opp_bin"] = pd.cut(d.n_opportunities, bins, right=False).astype(str)
pw = d.groupby("opp_bin", sort=False).agg(n=("n_opportunities", "size"), median_opp=("n_opportunities", "median"),
                                          power_cf_all=("p_cf_all", lambda x: (x < 0.05).mean()),
                                          power_cf_opp=("p_cf_opp", lambda x: (x < 0.05).mean()),
                                          power_p_link=("p_p_link", lambda x: (x < 0.05).mean()),
                                          median_cf_opp=("cf_opp", "median"), median_p_link=("p_link", "median")).reset_index()
pw = pw.sort_values("median_opp"); pw.to_csv(T / "ebro_power_by_opportunities.csv", index=False)
R["power_by_bin"] = pw.round(3).to_dict(orient="records")
R["N50"], R["N80"] = {}, {}
x = np.log(d.n_opportunities.values).reshape(-1, 1)
for col in ("p_cf_all", "p_cf_opp", "p_p_link"):
    y = (d[col].fillna(1) < 0.05).astype(int).values
    m = LogisticRegression(penalty=None).fit(x, y)
    b0, b1 = float(m.intercept_[0]), float(m.coef_[0, 0])
    R["N50"][col] = round(float(np.exp(-b0 / b1)), 0)
    R["N80"][col] = round(float(np.exp((np.log(4) - b0) / b1)), 0)
    R.setdefault("power_at", {})[col] = {str(n): round(float(m.predict_proba([[np.log(n)]])[0, 1]), 3) for n in (40, 928)}

# ---------------------------------------------------------------------------
# Analytic sample size (Proposition 4)
# ---------------------------------------------------------------------------
za, zb = stats.norm.ppf(0.95), stats.norm.ppf(0.80)
ab = []
for name in NETS:
    p0, p1 = float(nsi.loc[name, "null_cf_opp"]), float(nsi.loc[name, "cf_opp"])
    n = ((za * np.sqrt(p0 * (1 - p0)) + zb * np.sqrt(p1 * (1 - p1))) / (p1 - p0)) ** 2
    ab.append({"network": name, "null_cf_opp": round(p0, 4), "cf_opp": round(p1, 4), "excess": round(p1 - p0, 4),
               "n_opportunities": int(nsi.loc[name, "n_opportunities"]), "n_star_binomial": float(np.ceil(n))})
R["analytic_bound"] = ab
(OUT / "identifiability_numbers.json").write_text(json.dumps(R, indent=2, default=str))
log("done")
print(json.dumps({k: R[k] for k in ("networks", "stratified", "ebro_equivalent_cf_volta", "ebro_equivalent_cf_volta_ci",
                                    "volta_equivalent_cf_ebro", "N50", "N80", "analytic_bound", "ebro_volta_like_33",
                                    "degraded_draws_with_opportunities")}, indent=1, default=str))
