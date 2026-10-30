#!/usr/bin/env python3
"""
Year-displacement synchrony test and precision of the per-link estimate
(Supplementary Section S9 and Table S22; the precision table is not reported in the article).

S42  Binary phi of daily drought states (SSI < -1.28) on directed Volta gauge links, using
     non-imputed defined index values at both ends (fill tier 0; 29 February excluded), with the
     upstream series displaced by -1, 0 and +1 calendar years. A link qualifies for a given
     displacement when it has more than 365 paired valid days.
S43  The same contrast on the links with more than 365 days that are valid under all three
     displacements (common days), with the same-day advantage over the mean of the shifted
     controls and a link-resampling range (10,000 resamples).
S44  Precision of p-hat for hypothetical networks of 40-370 independent origin events, with k
     drawn from the observed distribution of observing upstream gauges and p = 0.162
     (1,200 Monte Carlo replicates each).

Inputs: output_observational/tables/SSI_daily.csv, upstream_connectivity.csv,
review_opportunities.csv. Writes output_identifiability/tables/year_shift_phi.csv,
year_shift_common.csv, year_shift_summary.json and fixed_k_precision.csv.
Author: I. Quist (NUIST)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent
T = ROOT / "output_observational/tables"
OUT = ROOT / "output_identifiability/tables"
THRESH, MIN_DAYS, SEED = -1.28, 365, 2026


def phi(a, b):
    a = np.asarray(a); b = np.asarray(b)
    tp = int(((a == 1) & (b == 1)).sum()); tn = int(((a == 0) & (b == 0)).sum())
    fp = int(((a == 1) & (b == 0)).sum()); fn = int(((a == 0) & (b == 1)).sum())
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return (tp * tn - fp * fn) / den if den > 0 else np.nan


def mle(k, y):
    def nll(p):
        ln0 = np.log1p(-p)
        return -float((y * np.log(-np.expm1(k * ln0)) + (1 - y) * k * ln0).sum())
    return minimize_scalar(nll, bounds=(1e-9, 0.999), method="bounded", options={"xatol": 1e-10}).x


def main():
    s = pd.read_csv(T / "SSI_daily.csv", parse_dates=["date"], usecols=["node_id", "date", "fill_tier", "SSI"])
    s = s[(s.fill_tier == 0) & s.SSI.notna() & ~((s.date.dt.month == 2) & (s.date.dt.day == 29))]
    conn = pd.read_csv(T / "upstream_connectivity.csv", sep=";")
    links = [(int(u), int(r.node_id)) for r in conn.itertuples() for u in str(r.upstream_chain).split(",")
             if u not in ("", "nan")]
    D = {n: g.set_index("date").SSI for n, g in s.groupby("node_id")}
    shifted = lambda n, k: D[n].set_axis(D[n].index + pd.DateOffset(years=k))

    rows, common = [], []
    for u, d in links:
        if u not in D or d not in D:
            continue
        for k in (-1, 0, 1):
            j = pd.concat([shifted(u, k).rename("u"), D[d].rename("d")], axis=1, join="inner").dropna()
            rows.append(dict(upstream=u, downstream=d, shift_years=k, paired_days=len(j),
                             phi=phi((j.u < THRESH).astype(int), (j.d < THRESH).astype(int))))
        idx = D[d].index
        for k in (-1, 0, 1):
            idx = idx.intersection(shifted(u, k).index)
        if len(idx) > MIN_DAYS:
            y = (D[d].loc[idx] < THRESH).astype(int)
            ph = {k: phi((shifted(u, k).loc[idx] < THRESH).astype(int), y) for k in (-1, 0, 1)}
            if all(np.isfinite(v) for v in ph.values()):
                common.append(dict(upstream=u, downstream=d, common_days=len(idx), phi_unshifted=ph[0],
                                   phi_minus1=ph[-1], phi_plus1=ph[1], advantage=ph[0] - (ph[-1] + ph[1]) / 2))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "year_shift_phi.csv", index=False)
    q = r[(r.paired_days > MIN_DAYS) & r.phi.notna()]
    s42 = q.groupby("shift_years").agg(qualified_links=("phi", "size"), paired_valid_days=("paired_days", "sum"),
                                         median_phi=("phi", "median")).reset_index()
    c = pd.DataFrame(common)
    c.to_csv(OUT / "year_shift_common.csv", index=False)
    rng = np.random.default_rng(SEED)
    adv = c.advantage.to_numpy()
    boot = np.array([rng.choice(adv, len(adv)).mean() for _ in range(10000)])
    summ = dict(by_shift=s42.to_dict("records"), common_links=len(c), median_advantage=float(np.median(adv)),
                mean_advantage=float(adv.mean()), resampling_2_5_97_5=[float(np.percentile(boot, 2.5)),
                                                                      float(np.percentile(boot, 97.5))],
                n_resamples=10000, seed=SEED)
    (OUT / "year_shift_summary.json").write_text(json.dumps(summ, indent=1))

    # precision of p-hat with k drawn from the observed opportunities
    op = pd.read_csv(T / "review_opportunities.csv")
    kobs = op.loc[op.n_observing >= 1, "n_observing"].to_numpy(float)
    rng = np.random.default_rng(SEED)
    prec = []
    for n in (40, 80, 160, 370):
        est = []
        for _ in range(1200):
            k = rng.choice(kobs, n)
            y = (rng.random(n) < 1 - (1 - 0.162) ** k).astype(float)
            est.append(mle(k, y))
        prec.append(dict(origins=n, replicates=1200, median_p_hat=float(np.median(est)),
                         p2_5=float(np.percentile(est, 2.5)), p97_5=float(np.percentile(est, 97.5))))
    pd.DataFrame(prec).to_csv(OUT / "fixed_k_precision.csv", index=False)
    print(s42.round(3).to_string(index=False)); print(c.round(3).to_string(index=False))
    print(json.dumps({k: v for k, v in summ.items() if k != "by_shift"}, indent=1))
    print(pd.DataFrame(prec).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
