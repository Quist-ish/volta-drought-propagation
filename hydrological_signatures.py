#!/usr/bin/env python3
"""
Catchment-storage and intermittency signatures of the 18 analysed GRDC gauges, computed from
observed daily discharge only (no gap filling), 1961-2007. Writes a derived table that can be
shared without the GRDC daily records:

    output_landscape/hydrological_signatures_grdc.csv

Signatures (all on observed days):
  zero_flow_frac   share of observed days with Q = 0
  bfi              baseflow index, Lyne-Hollick filter (alpha = 0.925, forward-backward-forward),
                   applied to unbroken runs of >= 60 observed days; sum(baseflow) / sum(Q)
  k_rec, tau_rec_d recession constant: median Q[t+1]/Q[t] over days inside recession limbs
                   (>= 7 consecutive declining days with Q > 0, first 2 days of each limb dropped);
                   tau = -1 / ln(k) in days
  q90_q50          Q exceeded 90% of the time divided by the median (zeros included)
  fdc_slope        (ln Q33 - ln Q66) / 0.33, exceedance percentiles of Q > 0
  rb_flashiness    Richards-Baker index, sum |Q[t] - Q[t-1]| / sum Q over consecutive observed days
  dry_wet_ratio    mean Q in December-April / mean Q in August-October
Flags: flag_zero_artefact (zero flow on catchments > 50,000 km2, possibly a recording artefact) and
flag_short_record (< 3 years of observed days).
Author: I. Quist (NUIST)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("output_landscape"); OUT.mkdir(exist_ok=True)
q = pd.read_csv("data/grdc_discharge_raw.csv", parse_dates=["date"])
st = pd.read_csv("data/grdc_stations_raw.csv").set_index("grdc_no")
gm = pd.read_csv("data/reanalysis/t2500/gauge_matches.csv").set_index("grdc_no")
gauges = sorted(gm.index.astype(int))
q = q[q.grdc_no.isin(gauges) & q.date.between("1961-01-01", "2007-12-31")].dropna(subset=["Q"])


def runs(dates):
    """Labels of unbroken daily runs."""
    gap = dates.diff().dt.days.fillna(1).ne(1)
    return gap.cumsum()


def lyne_hollick(x, a=0.925):
    def one(v, rev=False):
        v = v[::-1] if rev else v
        qf = np.zeros_like(v)
        for t in range(1, len(v)):
            qf[t] = a * qf[t - 1] + (1 + a) / 2 * (v[t] - v[t - 1])
            qf[t] = min(max(qf[t], 0.0), v[t])
        b = v - qf
        return b[::-1] if rev else b
    b = one(x); b = one(b, rev=True); b = one(b)
    return np.minimum(b, x)


rows = []
for g in gauges:
    s = q[q.grdc_no == g].sort_values("date")
    v = s.Q.values; d = s.date
    if len(s) < 365:
        rows.append({"grdc_no": g, "n_obs_days": len(s)}); continue
    lab = runs(d)
    # baseflow index on long unbroken runs
    bsum = qsum = 0.0
    for _, idx in s.groupby(lab.values).groups.items():
        seg = s.loc[idx, "Q"].values
        if len(seg) >= 60 and seg.sum() > 0:
            bsum += lyne_hollick(seg).sum(); qsum += seg.sum()
    # recession constant
    ratios = []
    for _, idx in s.groupby(lab.values).groups.items():
        seg = s.loc[idx, "Q"].values
        dec = np.r_[False, (seg[1:] < seg[:-1]) & (seg[1:] > 0)]
        i = 0
        while i < len(seg):
            if dec[i]:
                j = i
                while j < len(seg) and dec[j]:
                    j += 1
                if j - i >= 7:                                # limb of >= 7 declining days
                    limb = seg[i - 1:j]
                    r = limb[1:] / limb[:-1]
                    ratios.extend(r[2:])                      # drop the first two days after the peak
                i = j
            else:
                i += 1
    k = float(np.median(ratios)) if len(ratios) >= 20 else np.nan
    pos = v[v > 0]
    q33, q66 = (np.percentile(pos, 67), np.percentile(pos, 34)) if len(pos) > 100 else (np.nan, np.nan)
    cons = d.diff().dt.days.eq(1).values
    dq = np.abs(np.diff(v))[cons[1:]]
    m = d.dt.month
    wet = s.Q[m.isin([8, 9, 10]).values].mean(); dry = s.Q[m.isin([12, 1, 2, 3, 4]).values].mean()
    rows.append({
        "grdc_no": g, "n_obs_days": len(s), "years_obs": round(len(s) / 365.25, 1),
        "zero_flow_frac": round(float((v == 0).mean()), 4),
        "bfi": round(bsum / qsum, 3) if qsum > 0 else np.nan,
        "k_rec": round(k, 4) if np.isfinite(k) else np.nan,
        "tau_rec_d": round(-1 / np.log(k), 1) if np.isfinite(k) and 0 < k < 1 else np.nan,
        "q90_q50": round(np.percentile(v, 10) / np.percentile(v, 50), 4) if np.percentile(v, 50) > 0 else np.nan,
        "fdc_slope": round((np.log(q33) - np.log(q66)) / 0.33, 3) if np.isfinite(q33) and q66 > 0 else np.nan,
        "rb_flashiness": round(dq.sum() / v[1:][cons[1:]].sum(), 4) if v[1:][cons[1:]].sum() > 0 else np.nan,
        "dry_wet_ratio": round(dry / wet, 4) if wet > 0 else np.nan,
        "n_recession_days": len(ratios),
        "zero_first_year": int(d[v == 0].dt.year.min()) if (v == 0).any() else np.nan,
        "zero_last_year": int(d[v == 0].dt.year.max()) if (v == 0).any() else np.nan})

sig = pd.DataFrame(rows)
meta = st.loc[gauges, ["station_name", "river", "country", "lat", "lon", "area_km2"]].reset_index()
meta["tdx_river_id"] = gm.loc[gauges, "river_id"].values
meta["tdx_area_km2"] = gm.loc[gauges, "area_tdx"].round(0).values
out = meta.merge(sig, on="grdc_no")
# quality flags: zero flow on very large main-stem catchments is implausible and may be a recording
# artefact (Bamboi 1999-2007, Pwalagu 1988-1996); records under three years give unstable signatures
out["flag_zero_artefact"] = (out.area_km2 > 50000) & (out.zero_flow_frac > 0.05)
out["flag_short_record"] = out.years_obs.fillna(0) < 3
out.to_csv(OUT / "hydrological_signatures_grdc.csv", index=False)
json.dump({"period": "1961-2007", "source": "GRDC daily discharge, observed days only",
           "lyne_hollick_alpha": 0.925, "min_run_days_bfi": 60, "min_limb_days": 7},
          open(OUT / "hydrological_signatures_meta.json", "w"), indent=1)
print(out.to_string())
