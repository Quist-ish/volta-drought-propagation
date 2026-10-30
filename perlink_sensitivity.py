#!/usr/bin/env python3
"""
Sensitivity of the per-link match probability p-hat on the observed Volta network
(Supplementary Table S21, Section S12 and Table S30).

Inputs (derived tables written by run_observational.py, published_design_checks.py,
zero_flow_sensitivity.py and identifiability_analysis.py):
  output_observational/tables/review_opportunities.csv   origin events and observing upstream gauges
  output_observational/tables/propagation_pairs.csv      accepted matches
  output_observational/tables/SSI_drought_events.csv     event severity
  output_identifiability/tables/networks_summary.csv     block-bootstrap interval of p-hat
  output_landscape/zero_flow_sensitivity.json            zero-flow-year deletion

Model: an origin event with k observing upstream gauges is matched with probability
1 - (1 - p)^k (independent-link working model, Eq. 17 of the article).

Writes output_identifiability/tables/perlink_sensitivity.csv (Table S21),
perlink_groups.csv (gauge groups) and perlink_severity.json (severity split, Section S12).
Author: I. Quist (NUIST)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize, minimize_scalar

ROOT = Path(__file__).resolve().parent
T = ROOT / "output_observational/tables"
OUT = ROOT / "output_identifiability/tables"
NAMES = {1431300: "Vonkoro", 1531100: "Bamboi", 1531450: "Nawuni", 1531550: "Pwalagu", 1731400: "Porga"}
SEED, N_BOOT = 2026, 3000


def loglik(p, k, y):
    if not 0 < p < 1:
        return -np.inf
    ln0 = np.log1p(-p)
    return float((y * np.log(-np.expm1(k * ln0)) + (1 - y) * k * ln0).sum())


def mle(k, y):
    return minimize_scalar(lambda p: -loglik(p, k, y), bounds=(1e-9, 0.999), method="bounded",
                           options={"xatol": 1e-11}).x


def posterior(k, y, a, b):
    x = np.linspace(1e-6, 1 - 1e-6, 50001)
    ln0 = np.log1p(-x[:, None])
    t = (y * np.log(-np.expm1(k * ln0)) + (1 - y) * k * ln0).sum(axis=1) + (a - 1) * np.log(x) + (b - 1) * np.log1p(-x)
    d = np.exp(t - t.max())
    c = np.r_[0, cumulative_trapezoid(d, x)]
    c /= c[-1]
    return [float(np.interp(q, c, x)) for q in (0.025, 0.5, 0.975)]


def cond_fit(k, y, z):
    """Per-link logit(p) = a + b z; returns (b, max log-likelihood)."""
    def nll(th):
        p = 1 / (1 + np.exp(-(th[0] + th[1] * z)))
        ln0 = np.log1p(-p)
        return -float((y * np.log(-np.expm1(k * ln0) + 1e-300) + (1 - y) * k * ln0).sum())
    r = minimize(nll, x0=[-1.6, 0.0], method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 4000})
    return r.x[1], -r.fun


def main():
    op = pd.read_csv(T / "review_opportunities.csv")
    pp = pd.read_csv(T / "propagation_pairs.csv")
    ev = pd.read_csv(T / "SSI_drought_events.csv")
    opp = op[op.n_observing >= 1].copy()
    matched = set(zip(pp.origin_node, pp.origin_event_id))
    opp["connected"] = [(a, b) in matched for a, b in zip(opp.node_id, opp.event_id)]
    opp = opp.merge(ev[["node_id", "event_id", "severity"]], on=["node_id", "event_id"])
    k = opp.n_observing.to_numpy(float); y = opp.connected.to_numpy(float)
    assert (len(opp), int(y.sum()), int(k.sum())) == (40, 10, 67), "unexpected opportunity set"

    ns = pd.read_csv(ROOT / "output_identifiability/tables/networks_summary.csv").set_index("network")
    ci = json.loads(ns.loc["Volta observed", "p_link_ci"])
    p0 = mle(k, y)
    rows = [dict(specification="Reference maximum likelihood", origin_events=40, connected="10", p_hat=p0,
                 interval=f"{ci[0]:.3f}-{ci[1]:.3f}", note="calendar-year block bootstrap (identifiability_analysis.py)")]
    loo = [mle(np.delete(k, i), np.delete(y, i)) for i in range(len(k))]
    con_loo = [int(np.delete(y, i).sum()) for i in range(len(k))]
    rows.append(dict(specification="Delete one origin event at a time", origin_events=39,
                     connected=f"{min(con_loo)}-{max(con_loo)}", p_hat=np.nan,
                     interval=f"{min(loo):.3f}-{max(loo):.3f}", note="40 refits; influence range"))
    for a, b in ((1, 1), (2, 2)):
        lo, med, hi = posterior(k, y, a, b)
        rows.append(dict(specification=f"Beta({a},{b}) prior on p", origin_events=40, connected="10", p_hat=med,
                         interval=f"{lo:.3f}-{hi:.3f}", note="posterior median and 95% credible interval"))
    for g, name in NAMES.items():
        keep = opp.node_id.to_numpy() != g
        rows.append(dict(specification=f"Withhold {name}", origin_events=int(keep.sum()), connected=str(int(y[keep].sum())),
                         p_hat=mle(k[keep], y[keep]), interval="", note="downstream-gauge deletion"))
    zf = json.loads((ROOT / "output_landscape/zero_flow_sensitivity.json").read_text())
    w = zf["without_zero_flow_years"]
    rows.append(dict(specification="Bamboi 1999-2007 and Pwalagu 1988-1996 unobserved", origin_events=int(w["n_opportunities"]),
                     connected=str(int(w["n_connected"])), p_hat=w["p_link"], interval="",
                     note=f"{174 - zf['events_removed']} events; cf_opp = {w['cf_opp']:.3f} (zero_flow_sensitivity.py)"))
    pd.DataFrame(rows).to_csv(OUT / "perlink_sensitivity.csv", index=False)

    grp = opp.groupby("node_id").agg(origin_events=("event_id", "size"), accepted_matches=("connected", "sum"),
                                     observing_link_records=("n_observing", "sum")).reset_index()
    grp.insert(1, "station", grp.node_id.map(NAMES))
    grp.to_csv(OUT / "perlink_groups.csv", index=False)

    # severity split (Section S12)
    med = float(opp.severity.median())
    hi = (opp.severity > med).to_numpy()
    tab = np.array([[int(((~hi) & (y == 0)).sum()), int(((~hi) & (y == 1)).sum())],
                    [int((hi & (y == 0)).sum()), int((hi & (y == 1)).sum())]])
    or_ev = (tab[1, 1] / tab[1, 0]) / (tab[0, 1] / tab[0, 0])
    se = np.sqrt((1 / tab).sum())
    lr_ev = 2 * (sum(n * np.log(n / r.sum()) for r in tab for n in r if n) -
                 sum(n * np.log(n / tab.sum()) for n in tab.sum(axis=0) if n))
    fisher_p = stats.fisher_exact(tab)[1]
    b, l1 = cond_fit(k, y, hi.astype(float))
    l0 = loglik(p0, k, y)
    lr = 2 * (l1 - l0)
    rng = np.random.default_rng(SEED)
    q = 1 - (1 - p0) ** k
    lr_null = []
    for _ in range(N_BOOT):
        yy = (rng.random(len(k)) < q).astype(float)
        if yy.sum() in (0, len(yy)):
            lr_null.append(0.0); continue
        lr_null.append(2 * (cond_fit(k, yy, hi.astype(float))[1] - loglik(mle(k, yy), k, yy)))
    sev = dict(origin_events=40, matches=10, downstream_gauges=int(opp.node_id.nunique()), median_severity=med,
               n_low=int((~hi).sum()), n_high=int(hi.sum()),
               low_unconnected_connected=tab[0].tolist(), high_unconnected_connected=tab[1].tolist(),
               mean_k_low=float(k[~hi].mean()), mean_k_high=float(k[hi].mean()),
               event_odds_ratio=float(or_ev), event_or_wald95=[float(np.exp(np.log(or_ev) - 1.96 * se)),
                                                               float(np.exp(np.log(or_ev) + 1.96 * se))],
               event_lr_p=float(stats.chi2.sf(lr_ev, 1)), fisher_p=float(fisher_p),
               p_hat_low=float(mle(k[~hi], y[~hi])), p_hat_high=float(mle(k[hi], y[hi])),
               conditional_or=float(np.exp(b)), conditional_lr_p=float(stats.chi2.sf(lr, 1)),
               null_bootstrap_p=float(np.mean(np.array(lr_null) >= lr - 1e-12)), n_boot=N_BOOT, seed=SEED)
    (OUT / "perlink_severity.json").write_text(json.dumps(sev, indent=1))

    # antecedent state (Table S30): mean non-imputed SSI at the origin gauge over the 30 days before
    # onset, for origin events with at least 20 such days; terciles and conditional p-hat with
    # 95% profile-likelihood intervals
    ssi = pd.read_csv(T / "SSI_daily.csv", parse_dates=["date"], usecols=["node_id", "date", "fill_tier", "SSI"])
    ssi = ssi[(ssi.fill_tier == 0) & ssi.SSI.notna()]
    D = {n: g.set_index("date").SSI for n, g in ssi.groupby("node_id")}
    ae = opp.merge(ev[["node_id", "event_id", "start_date", "end_date"]], on=["node_id", "event_id"])
    ae["start_date"] = pd.to_datetime(ae.start_date)
    d30, d1 = np.timedelta64(30, "D"), np.timedelta64(1, "D")
    pre = [D[r.node_id].loc[r.start_date - d30:r.start_date - d1] for r in ae.itertuples()]
    ae["ante"] = [p.mean() for p in pre]; ae["nok"] = [p.size for p in pre]
    ae = ae[ae.nok >= 20].copy()
    ae["terc"] = pd.qcut(ae.ante, 3, labels=["drier", "intermediate", "wetter"])
    ae.to_csv(OUT / "antecedent_events.csv", index=False)
    from scipy.optimize import brentq
    terc = []
    for t, sub in list(ae.groupby("terc", observed=True)) + [("all", ae)]:
        kk = sub.n_observing.to_numpy(float); yy = sub.connected.to_numpy(float)
        ph = mle(kk, yy); lmax = loglik(ph, kk, yy)
        f = lambda p: 2 * (lmax - loglik(p, kk, yy)) - 3.841
        lo = brentq(f, 1e-6, ph) if f(1e-6) > 0 else 0.0
        terc.append(dict(tercile=str(t), events=len(sub), matches=int(yy.sum()), observing_links=int(kk.sum()),
                         p_hat=ph, lo95=lo, hi95=brentq(f, ph, 1 - 1e-6)))
    pd.DataFrame(terc).to_csv(OUT / "antecedent_terciles.csv", index=False)
    print(pd.DataFrame(terc).round(3).to_string(index=False))
    print(pd.DataFrame(rows).drop(columns="note").to_string(index=False))
    print(grp.to_string(index=False))
    print(json.dumps(sev, indent=1))


if __name__ == "__main__":
    main()
