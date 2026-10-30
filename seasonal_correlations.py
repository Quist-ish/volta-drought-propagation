#!/usr/bin/env python3
"""
Seasonal correlations of June-October basin rainfall and drought synchrony with climate indices (objective ii; Supplementary Table S8).

Pearson r of annual June-October basin precipitation (GPCC) and of the June-October mean drought synchrony
(share of the 18 gauge catchments with SPI-6 <= -1) with seasonal means of six indices, for 1952-2020 and
1961-2007; effective sample size from lag-1 autocorrelations (Bretherton et al., 1999); Holm adjustment across
the six indices within each series and period; the same correlations after removing linear trends.
Inputs: output_climate/tables (from climate_trends_teleconnections.py).
Writes output_climate/tables/seasonal_index_correlations.csv
Author: I. Quist (NUIST)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

TC = Path(__file__).resolve().parent / "output_climate/tables"
SEAS = {"AMO": range(6, 11), "Nino34": [7, 8, 9], "ATL3": [6, 7, 8], "TNA_TSA": range(6, 11), "DMI": [7, 8, 9, 10],
        "PDO": range(1, 13)}


def r_neff(x, y):
    x, y = pd.Series(np.asarray(x, float)), pd.Series(np.asarray(y, float))
    r1x, r1y, n = x.autocorr(1), y.autocorr(1), len(x)
    ne = n * (1 - r1x * r1y) / (1 + r1x * r1y)
    r = float(np.corrcoef(x, y)[0, 1])
    t = r * np.sqrt(max(ne - 2, 1) / (1 - r * r))
    return r, ne, float(2 * stats.t.sf(abs(t), max(ne - 2, 1)))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def main():
    P = pd.read_csv(TC / "precip_catchments_basin.csv", index_col=0, parse_dates=True)
    idx = pd.read_csv(TC / "climate_indices_monthly.csv", index_col=0, parse_dates=True)
    S6 = pd.read_csv(TC / "spi6_catchments_basin.csv", index_col=0, parse_dates=True).drop(columns="basin")
    sync = (S6 <= -1).mean(axis=1)
    jj = P["basin"][P.index.month.isin(range(6, 11))]
    rain = jj.groupby(jj.index.year).sum()
    sj = sync[sync.index.month.isin(range(6, 11))]
    syn = sj.groupby(sj.index.year).mean()
    rows = []
    for k, mo in SEAS.items():
        s = idx[k][idx.index.month.isin(list(mo))]
        a = s.groupby(s.index.year).mean()
        for lab, ser in (("JJASO basin rain", rain), ("JJASO synchrony", syn)):
            for per, (y0, y1) in (("1952-2020", (1952, 2020)), ("1961-2007", (1961, 2007))):
                j = pd.concat([ser, a], axis=1).dropna()
                j = j[(j.index >= y0) & (j.index <= y1)]
                r, ne, p = r_neff(j.iloc[:, 0], j.iloc[:, 1])
                x = j.iloc[:, 0] - np.polyval(np.polyfit(j.index, j.iloc[:, 0], 1), j.index)
                y = j.iloc[:, 1] - np.polyval(np.polyfit(j.index, j.iloc[:, 1], 1), j.index)
                rd, _, pdt = r_neff(x, y)
                rows.append(dict(series=lab, index=k, period=per, n=len(j), r=r, n_eff=ne, p=p, r_detr=rd, p_detr=pdt))
    d = pd.DataFrame(rows)
    d["p_holm"] = np.nan
    for _, g in d.groupby(["series", "period"]):
        d.loc[g.index, "p_holm"] = holm(g.p.values)
    d.to_csv(TC / "seasonal_index_correlations.csv", index=False)
    print(d.round(3).to_string())


if __name__ == "__main__":
    main()
