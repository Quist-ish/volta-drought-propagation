#!/usr/bin/env python3
"""
Has the rainfall-runoff relation of the Volta catchments changed? (objective i)

For every gauge-year with at least 11 complete months (each with no more than 3 missing
days), annual runoff depth R (mm) = mean discharge x seconds / TDX-Hydro contributing area,
and annual catchment precipitation P (GPCC). Tests:
  * Mann-Kendall (Hamed-Rao) and Sen slope of the runoff coefficient R/P per gauge
    (gauges with >= 10 such years);
  * pooled fixed-effects regression ln R = a_g + b ln P + c (year - 1985)/10, with
    cluster-robust (gauge) standard errors: c is the change in runoff per decade at
    fixed rainfall, estimated for all gauges and for the Sudano-Sahelian (catchment
    centroid north of 10.5 N) and Sudanian-Guinean groups.
Writes output_climate/tables/runoff_coefficient_*.csv|json
Author: I. Quist (NUIST)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pymannkendall as mk
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent
TC = ROOT / "output_climate/tables"


def main():
    q = pd.read_csv(ROOT / "data/grdc_discharge_raw.csv", parse_dates=["date"]).dropna(subset=["Q"])
    q = q[(q.date >= "1961-01-01") & (q.date <= "2007-12-31")]
    gm = pd.read_csv(ROOT / "data/reanalysis/t2500/gauge_matches.csv").set_index("grdc_no")
    st = pd.read_csv(ROOT / "data/grdc_stations_raw.csv").set_index("grdc_no")
    P = pd.read_csv(TC / "precip_catchments_basin.csv", index_col=0, parse_dates=True)
    cells = json.load(open(TC / "catchment_gpcc_cells.json"))
    rows = []
    for g, d in q.groupby("grdc_no"):
        if str(g) not in P.columns or g not in gm.index:
            continue
        area = float(gm.loc[g, "area_tdx"])
        m = d.set_index("date").Q.resample("MS").agg(["mean", "count"])
        m = m[m["count"] >= m.index.days_in_month - 3]
        for y, mm in m.groupby(m.index.year):
            if len(mm) < 11:
                continue
            dim = np.asarray(mm.index.days_in_month, float); qbar = float((mm["mean"].values * dim).sum() / dim.sum())
            R = qbar * 365.25 * 86400 / (area * 1e6) * 1000
            Py = float(P[str(g)][P.index.year == y].sum())
            lat_c = np.average([c[0] for c in cells[str(g)]], weights=[c[2] for c in cells[str(g)]])
            rows.append(dict(grdc_no=g, station=st.loc[g, "station_name"].title(), year=y, R_mm=R, P_mm=Py,
                             rc=R / Py, lat_c=lat_c))
    df = pd.DataFrame(rows)
    df.to_csv(TC / "runoff_coefficient_by_year.csv", index=False)
    per = []
    for g, d in df.groupby("station"):
        if len(d) < 10:
            continue
        t = mk.hamed_rao_modification_test(d.sort_values("year").rc.values)
        yrs, rc = d.year.values, d.rc.values
        i, j = np.triu_indices(len(rc), 1)
        sen = np.median((rc[j] - rc[i]) / (yrs[j] - yrs[i])) * 10
        per.append(dict(station=g, years=len(d), first=int(yrs.min()), last=int(yrs.max()), rc_mean=float(rc.mean()),
                        sen_rc_per_decade=float(sen), mk_hr_p=float(t.p)))
    pd.DataFrame(per).to_csv(TC / "runoff_coefficient_trends.csv", index=False)
    d = df[(df.R_mm > 0) & (df.P_mm > 0)].copy()
    d["lnR"], d["lnP"], d["dec"] = np.log(d.R_mm), np.log(d.P_mm), (d.year - 1985) / 10
    out = {}
    for lab, sub in [("all", d), ("north of 10.5 N", d[d.lat_c > 10.5]), ("south of 10.5 N", d[d.lat_c <= 10.5])]:
        if sub.station.nunique() < 3:
            continue
        f = smf.ols("lnR ~ C(station) + lnP + dec", data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub.station})
        c, se = f.params["dec"], f.bse["dec"]
        out[lab] = dict(gauges=int(sub.station.nunique()), gauge_years=int(len(sub)),
                        elasticity_P=float(f.params["lnP"]), elasticity_se=float(f.bse["lnP"]),
                        runoff_change_pct_per_decade=float(100 * (np.exp(c) - 1)),
                        lo95=float(100 * (np.exp(c - 1.96 * se) - 1)), hi95=float(100 * (np.exp(c + 1.96 * se) - 1)),
                        p=float(f.pvalues["dec"]))
    # sensitivity: exclude gauges downstream of reservoirs built before 2007 (Bagre 1992, Kompienga 1988)
    reg = {"Yarugu", "Pwalagu", "Nawuni", "Sabari"}
    sub = d[~d.station.isin(reg)]
    f = smf.ols("lnR ~ C(station) + lnP + dec", data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub.station})
    c, se = f.params["dec"], f.bse["dec"]
    out["unregulated gauges"] = dict(gauges=int(sub.station.nunique()), gauge_years=int(len(sub)),
                                     runoff_change_pct_per_decade=float(100 * (np.exp(c) - 1)),
                                     lo95=float(100 * (np.exp(c - 1.96 * se) - 1)), hi95=float(100 * (np.exp(c + 1.96 * se) - 1)),
                                     p=float(f.pvalues["dec"]))
    loo = []
    for g in d.station.unique():
        sub = d[d.station != g]
        f = smf.ols("lnR ~ C(station) + lnP + dec", data=sub).fit()
        loo.append(100 * (np.exp(f.params["dec"]) - 1))
    out["all"]["leave_one_gauge_out_range"] = [float(min(loo)), float(max(loo))]
    json.dump(out, open(TC / "runoff_coefficient_regression.json", "w"), indent=1)
    print(pd.DataFrame(per).round(3).to_string()); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
