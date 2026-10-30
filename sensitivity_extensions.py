#!/usr/bin/env python3
"""
Sensitivity extensions of the main analyses.

  thin     Ebro thinning with complete records: draws holding at least one origin event, medians and
           10th-90th percentiles of origin events, cf, k-bar and p-hat, and the share of draws in which
           the availability-aware null rejects (Table S23; from ebro_degradation.csv).
  runoff   Runoff at fixed rainfall (main-text Eq. 3) with the decade term interacted with catchment
           latitude and with mean catchment rainfall, gauge-by-decade slopes, and wild cluster
           bootstrap (Webb weights) p-values, which stay valid with few clusters.
  amo      Event-rate association with the AMO on detrended series (linear trend in the model and
           AMO detrended before fitting) and an out-of-sample check (fit 1961-1990, score 1991-2007).
  tau      Matching at drought thresholds tau = -1.0, -1.28 (default) and -1.5.
  nofill   Matching on an index computed without the <= 15-day interpolation (filled days removed
           before the SSI is fitted).
  regime   Regime-conditional index: distribution parameters fitted separately for 1961-1969 and
           1970-2007, so that every gauge is standardised against the regime it was recorded in.
  cluster  Cluster bootstrap of p-hat, resampling downstream gauges with their origin events.
  cpt      Change-point and synchrony uncertainty under serial correlation: lag-1 autocorrelation,
           moving-block bootstrap of the Pettitt statistic and of the mean synchrony before and after 1972.

Variants tau, nofill and regime use the same matching (W = 45 d, M = 5 d), availability-aware null
(1,999 permutations, seed 42) and year-block bootstrap of p-hat as reference_equivalence_simulation.py part A.
Writes output_observational/tables/r9_<part>.json (collected in r9_numbers.json) and the CSVs named below.
Usage: python sensitivity_extensions.py [part ...]   (default: all parts)
Author: I. Quist (NUIST)
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from voltaprop import fastmatch as F  # noqa: E402
from voltaprop.events import EventConfig, detect_network_events  # noqa: E402
from voltaprop.propagation import load_connectivity  # noqa: E402
from voltaprop.ssi import SSIConfig, compute_ssi_network  # noqa: E402

T = ROOT / "output_observational/tables"
TI = ROOT / "output_identifiability/tables"
TC = ROOT / "output_climate/tables"
JS = T / "r9_numbers.json"


def save(key, val):
    """One JSON per part (parts can run in parallel); r9_numbers.json collects them."""
    json.dump(val, open(T / f"r9_{key}.json", "w"), indent=1, default=float)
    d = {f.stem[3:]: json.load(open(f)) for f in sorted(T.glob("r9_*.json")) if f.name != JS.name}
    json.dump(d, open(JS, "w"), indent=1, default=float)


# ------------------------------------------------------------------------------------------- thin
def part_thin():
    D = pd.read_csv(TI / "ebro_degradation.csv")
    f = D[D["mode"] == "full"]
    rows = []
    for n, x in f.groupby("n_stations"):
        h = x[x.n_origin_events > 0]
        q = lambda s, a: float(s.quantile(a))
        rows.append(dict(n_stations=int(n), draws=int(len(x)), draws_with_events=int(len(h)),
                         origin_median=float(h.n_origin_events.median()), origin_p10=q(h.n_origin_events, .1),
                         origin_p90=q(h.n_origin_events, .9), k_median=float(h.mean_k_observing.median()),
                         cf_median=float(h.cf_all.median()), cf_p10=q(h.cf_all, .1), cf_p90=q(h.cf_all, .9),
                         cf_equal_opp=bool(np.allclose(h.cf_all, h.cf_opp)),
                         coverage_min=float(h.coverage.min()),
                         p_median=float(h.p_link.median()), p_p10=q(h.p_link, .1), p_p90=q(h.p_link, .9),
                         reject_cf_pct=float(100 * (h.p_cf_all < 0.05).mean()),
                         reject_p_pct=float(100 * (h.p_p_link < 0.05).mean())))
    R = pd.DataFrame(rows)
    R.to_csv(TI / "ebro_thinning_summary.csv", index=False)
    print(R.round(3).to_string())
    save("thin", R.to_dict(orient="records"))


# ----------------------------------------------------------------------------------------- runoff
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def wild_cluster_p(d, formula, term, cluster="station", B=9999, seed=7):
    """Restricted wild cluster bootstrap-t (Webb six-point weights) for H0: term = 0."""
    rng = np.random.default_rng(seed)
    full = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d[cluster]})
    t_obs = full.params[term] / full.bse[term]
    restr = smf.ols(formula.replace(f" + {term}", ""), data=d).fit()
    yhat, u = restr.fittedvalues.values, restr.resid.values
    g = pd.factorize(d[cluster])[0]; G = g.max() + 1
    y, X = full.model.endog, full.model.exog
    names = full.model.exog_names; j = names.index(term)
    XtX_inv = np.linalg.pinv(X.T @ X)
    ts = np.empty(B)
    for b in range(B):
        w = rng.choice(WEBB, G)[g]
        yb = yhat + u * w
        beta = XtX_inv @ X.T @ yb
        e = yb - X @ beta
        meat = np.zeros((X.shape[1], X.shape[1]))
        for c in range(G):
            s = X[g == c].T @ e[g == c]
            meat += np.outer(s, s)
        n, k = X.shape
        adj = G / (G - 1) * (n - 1) / (n - k)
        V = adj * XtX_inv @ meat @ XtX_inv
        ts[b] = beta[j] / np.sqrt(V[j, j])
    return float(np.mean(np.abs(ts) >= abs(t_obs))), float(t_obs)


def part_runoff():
    df = pd.read_csv(TC / "runoff_coefficient_by_year.csv")
    d = df[(df.R_mm > 0) & (df.P_mm > 0)].copy()
    d["lnR"], d["lnP"], d["dec"] = np.log(d.R_mm), np.log(d.P_mm), (d.year - 1985) / 10
    meanP = d.groupby("station").P_mm.mean()
    d["Pbar"] = d.station.map(meanP)
    lat0, P0 = float(d.groupby("station").lat_c.first().mean()), float(meanP.mean())
    d["lat_s"] = d.lat_c - lat0                      # degrees from the panel-mean latitude
    d["Pbar_s"] = (d.Pbar - P0) / 100                # per 100 mm of mean annual rainfall
    out = {"gauges": int(d.station.nunique()), "gauge_years": int(len(d)), "lat_centre": lat0,
           "Pbar_centre_mm": P0}
    base = "lnR ~ C(station) + lnP + dec"
    f0 = smf.ols(base, data=d).fit(cov_type="cluster", cov_kwds={"groups": d.station})
    p_wcb, _ = wild_cluster_p(d, base, "dec")
    out["main"] = dict(change_pct=float(100 * (np.exp(f0.params.dec) - 1)), p_cluster=float(f0.pvalues.dec),
                       p_wild=p_wcb)
    reg = {"Yarugu", "Pwalagu", "Nawuni", "Sabari"}
    for lab, sub in (("north", d[d.lat_c > 10.5]), ("south", d[d.lat_c <= 10.5]), ("unregulated", d[~d.station.isin(reg)])):
        fsub = smf.ols(base, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub.station})
        out[lab] = dict(gauges=int(sub.station.nunique()), change_pct=float(100 * (np.exp(fsub.params.dec) - 1)),
                        p_cluster=float(fsub.pvalues.dec), p_wild=wild_cluster_p(sub.copy(), base, "dec")[0])
    for lab, var in (("latitude", "lat_s"), ("rainfall", "Pbar_s")):
        d["ix"] = d.dec * d[var]
        fm = base + " + ix"
        f = smf.ols(fm, data=d).fit(cov_type="cluster", cov_kwds={"groups": d.station})
        p_w, _ = wild_cluster_p(d, fm, "ix")
        b, se = f.params.ix, f.bse.ix
        out[lab] = dict(interaction=float(b), se=float(se), p_cluster=float(f.pvalues.ix), p_wild=p_w,
                        pct_points_per_unit=float(100 * (np.exp(f.params.dec + b) - np.exp(f.params.dec))),
                        change_at_centre_pct=float(100 * (np.exp(f.params.dec) - 1)))
        if lab == "latitude":
            for L in (9.0, 10.0, 11.0, 12.0, 13.0):
                c = f.params.dec + b * (L - lat0)
                v = f.cov_params().loc[["dec", "ix"], ["dec", "ix"]].values
                g = np.array([1.0, L - lat0]); s = float(np.sqrt(g @ v @ g))
                out[lab][f"change_at_{L:.0f}N"] = [float(100 * (np.exp(c) - 1)), float(100 * (np.exp(c - 1.96 * s) - 1)),
                                                     float(100 * (np.exp(c + 1.96 * s) - 1))]
    # gauge-by-decade slopes (common elasticity), with the spread and a heterogeneity test
    fs = smf.ols("lnR ~ C(station) + lnP + C(station):dec", data=d).fit()
    fr = smf.ols(base, data=d).fit()
    ft = sm.stats.anova_lm(fr, fs)
    slopes = {k.split("[")[1].split("]")[0].replace("T.", ""): float(100 * (np.exp(v) - 1))
              for k, v in fs.params.items() if k.endswith(":dec")}
    yrs = d.groupby("station").year.agg(["min", "max", "size"])
    S = pd.DataFrame({"station": list(slopes), "change_pct_per_decade": list(slopes.values())})
    S = S.merge(yrs, left_on="station", right_index=True)
    S["lat_c"] = S.station.map(d.groupby("station").lat_c.first())
    S["P_mean_mm"] = S.station.map(meanP)
    S["span_years"] = S["max"] - S["min"] + 1
    S.to_csv(TC / "runoff_gauge_slopes.csv", index=False)
    inf = S[(S["size"] >= 5) & (S.span_years >= 8)]
    from scipy.stats import spearmanr
    rl = spearmanr(inf.lat_c, inf.change_pct_per_decade); rp = spearmanr(inf.P_mean_mm, inf.change_pct_per_decade)
    out["gauge_slopes"] = dict(F=float(ft.F.iloc[1]), p_heterogeneity=float(ft["Pr(>F)"].iloc[1]),
                               df_num=float(ft.df_diff.iloc[1]), informative_gauges=int(len(inf)),
                               positive=int((inf.change_pct_per_decade > 0).sum()),
                               median=float(inf.change_pct_per_decade.median()),
                               spearman_lat=[float(rl.statistic), float(rl.pvalue)],
                               spearman_P=[float(rp.statistic), float(rp.pvalue)])
    # confounders of the north-south contrast: drainage area and storage signatures (exploratory, n = 9)
    sig = pd.read_csv(ROOT / "output_landscape/hydrological_signatures_grdc.csv")
    sig["station"] = sig.station_name.str.title()
    m = inf.merge(sig, on="station", how="left")
    out["gauge_slopes"]["spearman_with"] = {c: [float(spearmanr(m[c], m.change_pct_per_decade).statistic),
                                                float(spearmanr(m[c], m.change_pct_per_decade).pvalue)]
                                            for c in ("area_km2", "bfi", "zero_flow_frac", "tau_rec_d", "rb_flashiness")}
    out["gauge_slopes"]["rising_area_km2"] = [float(m[m.change_pct_per_decade > 0].area_km2.min()),
                                              float(m[m.change_pct_per_decade > 0].area_km2.max())]
    out["gauge_slopes"]["falling_area_km2"] = [float(m[m.change_pct_per_decade < 0].area_km2.min()),
                                               float(m[m.change_pct_per_decade < 0].area_km2.max())]
    sub = d[d.lat_c <= 10.5]
    out["south"]["usable_years_by_gauge"] = {k: int(v) for k, v in sub.groupby("station").size().items()}
    print(json.dumps(out, indent=1)); print(S.round(2).to_string())
    save("runoff", out)


# -------------------------------------------------------------------------------------------- amo
def part_amo():
    """AMO event-rate association: trend-adjusted, detrended and out-of-sample (quasi-Poisson, exposure offset)."""
    er = pd.read_csv(TC / "event_rate_by_year.csv", index_col=0)
    yr = er.index.values.astype(float); dec = (yr - 1984) / 10
    off = np.log(er.exposure.values); y = er.events.values
    z = (er.AMO - er.AMO.mean()) / er.AMO.std()
    fit = lambda X: sm.GLM(y, X, family=sm.families.Poisson(), offset=off).fit(scale="X2")
    rr = lambda m, j: [float(np.exp(m.params[j])), float(np.exp(m.params[j] - 1.96 * m.bse[j])),
                       float(np.exp(m.params[j] + 1.96 * m.bse[j])), float(m.pvalues[j])]
    out = {"years": int(len(er)), "first": int(yr.min()), "last": int(yr.max())}
    m0 = fit(sm.add_constant(z.values)); out["amo_only"] = rr(m0, 1); out["dispersion"] = float(m0.scale)
    m1 = fit(sm.add_constant(np.column_stack([z.values, dec]))); out["amo_with_trend"] = rr(m1, 1)
    out["trend_with_amo"] = rr(m1, 2)
    b = np.polyfit(yr, er.AMO.values, 1); res = er.AMO.values - np.polyval(b, yr)
    zr = (res - res.mean()) / res.std(ddof=1)
    m2 = fit(sm.add_constant(zr)); out["amo_detrended"] = rr(m2, 1)
    out["amo_trend_per_decade"] = float(10 * b[0]); out["amo_corr_with_year"] = float(np.corrcoef(yr, er.AMO)[0, 1])
    # out of sample: fit 1961-1990, score 1991-2007 by Poisson deviance against an intercept-only model
    tr, te = yr <= 1990, yr >= 1991
    def dev(mu, yy):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(yy > 0, yy * np.log(yy / mu), 0.0)
        return float(2 * np.sum(t - (yy - mu)))
    Xa = sm.add_constant(z.values)
    ma = sm.GLM(y[tr], Xa[tr], family=sm.families.Poisson(), offset=off[tr]).fit()
    mn = sm.GLM(y[tr], np.ones((tr.sum(), 1)), family=sm.families.Poisson(), offset=off[tr]).fit()
    mu_a = np.exp(Xa[te] @ ma.params + off[te]); mu_n = np.exp(mn.params[0] + off[te])
    mq = sm.GLM(y[tr], Xa[tr], family=sm.families.Poisson(), offset=off[tr]).fit(scale="X2")
    out["oos"] = dict(train_years=int(tr.sum()), test_years=int(te.sum()), events_test=int(y[te].sum()),
                      rr_train=float(np.exp(ma.params[1])), rr_train_ci=[float(np.exp(mq.params[1] - 1.96 * mq.bse[1])),
                                                                       float(np.exp(mq.params[1] + 1.96 * mq.bse[1]))],
                      p_train=float(mq.pvalues[1]), deviance_amo=dev(mu_a, y[te]), deviance_null=dev(mu_n, y[te]),
                      spearman_pred_obs=float(pd.Series(mu_a / np.exp(off[te])).corr(pd.Series(y[te] / np.exp(off[te])),
                                                                                       method="spearman")))
    print(json.dumps(out, indent=1)); save("amo", out)


# --------------------------------------------------------------------------------- event variants
def run_variant(name, s_df, e, rng_seed=2026):
    rng = np.random.default_rng(rng_seed)
    conn = load_connectivity(T / "upstream_connectivity.csv")
    nodes = set(conn) | {u for v in conn.values() for u in v}
    avail = F.availability_from_daily(s_df, col="SSI")
    recs = F.build_records(e, avail=avail, nodes=nodes)
    obs, summ, null = F.null_test(recs, conn, n_perm=1999, seed=42)
    s = summ.set_index("statistic")
    pairs, org = F.match(recs, conn, detail=True)
    o = org[org.n_observing >= 1].copy()
    o["year"] = pd.to_datetime(o.s0_date).dt.year
    years = o.year.unique()
    boots = [F.link_probability_mle(*(lambda d: (d.n_observing.values, d.connected.values))(
        pd.concat([o[o.year == y] for y in rng.choice(years, len(years), replace=True)]))) for _ in range(2000)]
    pk = set(zip(pairs.origin_node, pairs.upstream_node, pairs.origin_eid))
    res = dict(events=int(len(e)), origin_events=int(obs["n_origin_events"]), opportunities=int(obs["n_opportunities"]),
               connected=int(obs["n_connected"]), pairs=int(obs["n_pairs"]), coverage=float(obs["coverage"]),
               cf_all=float(obs["cf_all"]), cf_opp=float(obs["cf_opp"]), p_link=float(obs["p_link"]),
               p_link_ci=[float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))],
               null_cf_opp=float(s.loc["cf_opp", "null_mean"]), p_cf_opp=float(s.loc["cf_opp", "p"]),
               null_p_link=float(s.loc["p_link", "null_mean"]), p_p_link=float(s.loc["p_link", "p"]),
               p_cf_all=float(s.loc["cf_all", "p"]),
               links=sorted({f"{u}->{o_}" for o_, u, _ in pk}))
    print(name, {k: v for k, v in res.items() if k != "links"}, flush=True)
    return res


def base_inputs():
    ssi0 = pd.read_csv(T / "SSI_daily.csv", parse_dates=["date"])
    return ssi0


def recompute(qf, cal, cfg):
    s_df, _ = compute_ssi_network(qf, cfg, node_col="node_id", progress=False)
    return cal.merge(s_df[["node_id", "date", "SSI"]], on=["node_id", "date"], how="left")


def part_tau():
    ssi0 = base_inputs()
    s_df = ssi0[["node_id", "date", "SSI"]]
    res = {}
    for tau in (-1.0, -1.28, -1.5):
        e = detect_network_events(s_df, EventConfig(threshold_ssi=tau), node_col="node_id", progress=False)
        res[f"{tau:.2f}"] = run_variant(f"tau {tau}", s_df, e)
    save("tau", res)


def part_nofill():
    ssi0 = base_inputs()
    cal = ssi0[["node_id", "date"]]
    q = ssi0.loc[ssi0.Q.notna(), ["node_id", "date", "Q", "fill_tier"]]
    qn = q[q.fill_tier.fillna(0) == 0][["node_id", "date", "Q"]]
    out = {"filled_days_removed": int((q.fill_tier.fillna(0) > 0).sum())}
    s_def = recompute(q[["node_id", "date", "Q"]], cal, SSIConfig())          # check: reproduces the stored index
    m = ssi0[["node_id", "date", "SSI"]].merge(s_def, on=["node_id", "date"], suffixes=("_stored", "_new"))
    both = m.SSI_stored.notna() & m.SSI_new.notna()
    out["reproduction_max_abs_diff"] = float((m.SSI_stored[both] - m.SSI_new[both]).abs().max())
    out["reproduction_defined_days"] = [int(m.SSI_stored.notna().sum()), int(m.SSI_new.notna().sum())]
    s_nf = recompute(qn, cal, SSIConfig())
    e = detect_network_events(s_nf, EventConfig(), node_col="node_id", progress=False)
    out["no_fill"] = run_variant("no fill", s_nf, e)
    out["defined_days_no_fill"] = int(s_nf.SSI.notna().sum())
    save("nofill", out)


def part_regime():
    ssi0 = base_inputs()
    cal = ssi0[["node_id", "date"]]
    qf = ssi0.loc[ssi0.Q.notna(), ["node_id", "date", "Q"]]
    pre = qf[qf.date.dt.year <= 1969]
    span = pre.groupby("node_id").date.agg(lambda s: s.dt.year.nunique())
    pre_nodes = span[span >= 2].index.tolist()
    a = recompute(qf, cal, SSIConfig(reference_years=(1970, 2007)))
    b = recompute(qf[qf.node_id.isin(pre_nodes)], cal[cal.node_id.isin(pre_nodes)], SSIConfig(reference_years=(1961, 1969)))
    s = cal.merge(a, on=["node_id", "date"], how="left")
    s = s.merge(b.rename(columns={"SSI": "SSI_pre"}), on=["node_id", "date"], how="left")
    early = s.date.dt.year <= 1969
    s.loc[early, "SSI"] = s.loc[early, "SSI_pre"]
    s_df = s[["node_id", "date", "SSI"]]
    d = ssi0[["node_id", "date", "SSI"]].merge(s_df, on=["node_id", "date"], suffixes=("_own", "_reg"))
    both = d.SSI_own.notna() & d.SSI_reg.notna()
    diff = (d.SSI_reg - d.SSI_own)[both].groupby(d.node_id[both]).agg(lambda x: float(np.abs(x).max()))
    out = {"pre_1970_nodes": [int(n) for n in pre_nodes],
           "nodes_changed": [int(n) for n, v in diff.items() if v > 1e-6],
           "max_abs_change_by_changed_node": {int(n): float(v) for n, v in diff.items() if v > 1e-6}}
    e = detect_network_events(s_df, EventConfig(), node_col="node_id", progress=False)
    out["regime"] = run_variant("regime-conditional", s_df, e)
    save("regime", out)


# ---------------------------------------------------------------------------------------- cluster
def part_cluster():
    ssi0 = base_inputs()
    s_df = ssi0[["node_id", "date", "SSI"]]
    e = pd.read_csv(T / "SSI_drought_events.csv", parse_dates=["start_date", "end_date"])
    conn = load_connectivity(T / "upstream_connectivity.csv")
    nodes = set(conn) | {u for v in conn.values() for u in v}
    recs = F.build_records(e, avail=F.availability_from_daily(s_df, col="SSI"), nodes=nodes)
    _, org = F.match(recs, conn, detail=True)
    o = org[org.n_observing >= 1]
    gs = o.origin_node.unique()
    rng = np.random.default_rng(99)
    boots = []
    for _ in range(4999):
        pick = rng.choice(gs, len(gs), replace=True)
        dd = pd.concat([o[o.origin_node == g] for g in pick])
        boots.append(F.link_probability_mle(dd.n_observing.values, dd.connected.values))
    boots = np.array(boots)
    out = dict(downstream_gauges=int(len(gs)), opportunities_by_gauge={int(g): int((o.origin_node == g).sum()) for g in gs},
               connected_by_gauge={int(g): int(o[o.origin_node == g].connected.sum()) for g in gs},
               p_hat=float(F.link_probability_mle(o.n_observing.values, o.connected.values)),
               ci95=[float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))],
               share_zero=float(np.mean(boots == 0)))
    print(out); save("cluster", out)


# -------------------------------------------------------------------------------------------- cpt
def pettitt_k(x):
    n = len(x); r = pd.Series(x).rank().values
    u = np.array([2 * r[:t + 1].sum() - (t + 1) * (n + 1) for t in range(n - 1)])
    return float(np.abs(u).max()), int(np.abs(u).argmax())


def _block_p(x, K, L, rng, B=4999):
    nb = int(np.ceil(len(x) / L)); ks = np.empty(B)
    for i in range(B):
        st = rng.integers(0, len(x) - L + 1, nb)
        ks[i] = pettitt_k(np.concatenate([x[s:s + L] for s in st])[:len(x)])[0]
    return float((1 + np.sum(ks >= K)) / (1 + B))


def _block_mean_ci(x, L, rng, B=4999):
    nb = int(np.ceil(len(x) / L)); m = np.empty(B)
    for i in range(B):
        st = rng.integers(0, max(len(x) - L + 1, 1), nb)
        m[i] = np.concatenate([x[s:s + L] for s in st])[:len(x)].mean()
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def part_cpt():
    """Serial correlation in the change-point tests: lag-1 autocorrelation and moving-block bootstrap p-values."""
    rng = np.random.default_rng(5)
    P = pd.read_csv(TC / "precip_catchments_basin.csv", index_col=0, parse_dates=True)
    ann = P["basin"].groupby(P.index.year).sum().loc[1951:2020]
    S6 = pd.read_csv(TC / "spi6_catchments_basin.csv", index_col=0, parse_dates=True)
    cat = [c for c in S6.columns if c != "basin"]
    share = (S6[cat] <= -1).sum(axis=1) / S6[cat].notna().sum(axis=1)
    syn = share.groupby(share.index.year).mean().loc[1951:2020]
    out = {}
    for lab, ser in (("precipitation", ann), ("synchrony", syn)):
        x, yrs = ser.values.astype(float), ser.index.values
        K, t = pettitt_k(x)
        step = np.where(np.arange(len(x)) <= t, x[:t + 1].mean(), x[t + 1:].mean())
        r = x - step
        o = {"break_year": int(yrs[t]), "K": K, "mean_before": float(x[:t + 1].mean()), "mean_after": float(x[t + 1:].mean()),
             "lag1_raw": float(np.corrcoef(x[:-1], x[1:])[0, 1]), "lag1_after_step": float(np.corrcoef(r[:-1], r[1:])[0, 1])}
        for L in (3, 5, 8):
            o[f"block{L}_p"] = _block_p(x, K, L, rng)
        o["ci_before_block5"] = _block_mean_ci(x[:t + 1], 5, rng)
        o["ci_after_block5"] = _block_mean_ci(x[t + 1:], 5, rng)
        out[lab] = o
    print(json.dumps(out, indent=1)); save("cpt", out)


PARTS = {"thin": part_thin, "runoff": part_runoff, "amo": part_amo, "tau": part_tau, "nofill": part_nofill,
         "regime": part_regime, "cluster": part_cluster, "cpt": part_cpt}

if __name__ == "__main__":
    for p in (sys.argv[1:] or list(PARTS)):
        PARTS[p]()
