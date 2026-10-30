#!/usr/bin/env python3
"""
Observed storage-persistence test (exploratory): does median drought-event duration at a gauge
rise with catchment storage, indexed by observed signatures (Lyne-Hollick baseflow index and the
recession timescale)? Gauge-level outcome (n = 16 gauges with events), Spearman rank correlation,
10,000 permutations, 5,000 bootstrap resamples of gauges, partial correlation on log terrain drainage area
and years of record, sensitivity without flagged gauges. Writes output_landscape/storage_persistence.csv.
Caveat: signatures and events come from the same discharge records, so the test is not independent
of the flow series; it asks whether storage-dominated regimes produce longer standardised droughts.
Author: I. Quist (NUIST)
"""
import numpy as np, pandas as pd
from scipy import stats
rng = np.random.default_rng(2026)
sig = pd.read_csv("output_landscape/hydrological_signatures_grdc.csv")
ev = pd.read_csv("output_observational/tables/SSI_drought_events.csv")
cen = pd.read_csv("output_observational/tables/review_event_censoring.csv")
cens = cen[cen.bound.ne("none")][["node_id", "event_id"]]        # events bounded by missing data
ev_u = ev.merge(cens.assign(_c=1), on=["node_id", "event_id"], how="left")
ev_u = ev_u[ev_u._c.isna()]
g = ev.groupby("node_id").duration.median().rename("median_duration").to_frame()
g["median_duration_uncensored"] = ev_u.groupby("node_id").duration.median()
g["n_events"] = ev.groupby("node_id").size()
d = sig.set_index("grdc_no").join(g, how="inner")
# terrain (TDX-Hydro) drainage area: the catalogued areas at Dapola and Wiasi disagree with the terrain by 52% and 30%
d["log_area"] = np.log(d.tdx_area_km2)


def rank_test(x, y, n_perm=10000, n_boot=5000):
    ok = x.notna() & y.notna(); x, y = x[ok].values, y[ok].values
    rho = stats.spearmanr(x, y)[0]
    perm = np.array([stats.spearmanr(x, rng.permutation(y))[0] for _ in range(n_perm)])
    p = (1 + np.sum(np.abs(perm) >= abs(rho))) / (n_perm + 1)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if len(np.unique(x[i])) > 2 and len(np.unique(y[i])) > 2:
            bs.append(stats.spearmanr(x[i], y[i])[0])
    return rho, p, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5), len(x)


def partial_rank(x, y, Z):
    df = pd.concat([x, y, Z], axis=1).dropna().rank()
    rx = df.iloc[:, 0] - np.linalg.lstsq(np.c_[np.ones(len(df)), df.iloc[:, 2:]], df.iloc[:, 0], rcond=None)[0] @ np.c_[np.ones(len(df)), df.iloc[:, 2:]].T
    ry = df.iloc[:, 1] - np.linalg.lstsq(np.c_[np.ones(len(df)), df.iloc[:, 2:]], df.iloc[:, 1], rcond=None)[0] @ np.c_[np.ones(len(df)), df.iloc[:, 2:]].T
    return stats.pearsonr(rx, ry)[0], len(df)


rows = []
for subset, mask in (("all gauges with events", np.ones(len(d), bool)),
                     ("excluding flagged gauges", ~(d.flag_zero_artefact | d.flag_short_record).values)):
    dd = d[mask]
    for pred in ("bfi", "tau_rec_d", "zero_flow_frac"):
        for out in ("median_duration", "median_duration_uncensored"):
            rho, p, lo, hi, n = rank_test(dd[pred], dd[out])
            pr, npar = partial_rank(dd[pred], dd[out], dd[["log_area", "years_obs"]])
            rows.append({"subset": subset, "predictor": pred, "outcome": out, "n": n, "spearman_rho": round(rho, 3),
                         "ci95_low": round(lo, 3), "ci95_high": round(hi, 3), "perm_p_two_sided": round(p, 4),
                         "partial_rho_logarea_years": round(pr, 3), "n_partial": npar})
res = pd.DataFrame(rows)
prim = res[(res.subset == "all gauges with events") & (res.outcome == "median_duration")].copy()
order = prim.perm_p_two_sided.argsort().values; m = len(prim); adj = np.empty(m); run = 0
for r, i in enumerate(order):
    run = max(run, min(1, (m - r) * prim.perm_p_two_sided.values[i])); adj[i] = run
res.loc[prim.index, "holm_p_primary_family"] = np.round(adj, 4)
# smallest |rho| detectable with 80% power at alpha = 0.05 (Fisher z approximation)
for n in (16, 13):
    z = (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) / np.sqrt(n - 3) * 1.03   # Spearman variance inflation ~1.06
    print(f"n = {n}: detectable |rho| ~ {np.tanh(z):.2f}")
res.to_csv("output_landscape/storage_persistence.csv", index=False)
d[["station_name", "river", "area_km2", "years_obs", "bfi", "tau_rec_d", "zero_flow_frac", "median_duration",
   "median_duration_uncensored", "n_events", "flag_zero_artefact", "flag_short_record"]].to_csv(
    "output_landscape/storage_persistence_gauges.csv")
print(res.to_string(index=False))
