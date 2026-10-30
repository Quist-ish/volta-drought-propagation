#!/usr/bin/env python3
"""
Trends, change points and climate teleconnections of Volta drought (objectives i and ii).

Inputs
  data/external/gpcc_v2022_volta_05.nc          GPCC Full Data Monthly v2022, 0.5 deg, 1951-2020
  data/external/sahelrain.data                    Sahel precipitation index (JISAO)
  data/external/climate_indices/*.txt, *.nc       NOAA PSL / CPC indices; ERSSTv5 (for ATL3)
  data/reanalysis/reaches_volta.parquet           TDX-Hydro reaches (catchment definition)
  output_observational/tables/*                   observed SSI, events, opportunities, pairs

Analyses
  1. Catchment and basin precipitation (GPCC); SPI-6 and SPI-12, gamma with zero mass,
     reference 1961-2014 (the CMIP6 historical overlap used in projection_cmip6.py).
  2. Homogeneity and change points (Pettitt; SNHT and Buishand range with permutation p),
     monotonic trend (Mann-Kendall with the Hamed-Rao variance correction, Sen slope),
     DWT-MK on the db4 approximation, BH-FDR across catchments.
  3. Baseline regime of each gauge's SSI reference: mean catchment SPI-12 over the years
     the gauge observed.
  4. Observed drought-event rate: Poisson / quasi-Poisson GLM with exposure offset on
     decade and on standardised seasonal climate indices (Holm across indices).
  5. Continuous wavelet power (Morlet), global wavelet spectrum against AR(1) red noise,
     cross-wavelet and wavelet coherence (Monte Carlo 95%) of basin SPI-12 and of the
     basin drought-synchrony index with AMO, Nino3.4, ATL3, TNA-TSA, DMI and PDO;
     lag correlations with Bretherton effective sample size.
  6. Drought synchrony S(t): share of the 18 gauge catchments with SPI-6 <= -1; S at the
     onset of connected and unconnected opportunities.

Writes output_climate/tables/*.csv|json and output_climate/figdata/*.npz
Author: I. Quist (NUIST)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pymannkendall as mk
import pywt
import statsmodels.api as sm
import xarray as xr
from scipy import stats

import pycwt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "output_climate"
TAB, FD = OUT / "tables", OUT / "figdata"
TAB.mkdir(parents=True, exist_ok=True)
FD.mkdir(parents=True, exist_ok=True)
CI = ROOT / "data/external/climate_indices"
TO = ROOT / "output_observational/tables"
REF = (1961, 2014)
RNG = np.random.default_rng(2026)
NPERM = 9999


# ------------------------------------------------------------------ climate indices
def read_psl(path: Path, missing=(-99.9, -99.99, -9.99, -999), floor=-90.0) -> pd.Series:
    """Parse a NOAA PSL 'year + 12 values' text file into a monthly series."""
    rows = []
    with open(path) as f:
        lines = f.read().splitlines()
    for ln in lines[1:]:
        p = ln.split()
        if len(p) != 13:
            continue
        try:
            y = int(p[0]); v = [float(x) for x in p[1:]]
        except ValueError:
            continue
        for m, x in enumerate(v, 1):
            rows.append((pd.Timestamp(y, m, 1), np.nan if any(abs(x - mm) < 1e-6 for mm in missing) or (floor is not None and x < floor) else x))
    s = pd.Series(dict(rows)).sort_index()
    return s


def atl3_from_ersst() -> pd.Series:
    d = xr.open_dataset(CI / "ersst_v5.nc")
    box = d.sst.sel(lat=slice(3, -3), lon=slice(340, 360))
    w = np.cos(np.deg2rad(box.lat))
    s = box.weighted(w).mean(["lat", "lon"]).to_series()
    s.index = pd.to_datetime(s.index)
    clim = s["1971":"2000"].groupby(s["1971":"2000"].index.month).mean()
    return s - clim.reindex(s.index.month).values


def load_indices() -> pd.DataFrame:
    idx = pd.DataFrame({
        "AMO": read_psl(CI / "amo.txt"),
        "Nino34": read_psl(CI / "n34.txt"),
        "TNA": read_psl(CI / "tna.txt"),
        "TSA": read_psl(CI / "tsa.txt"),
        "DMI": read_psl(CI / "dmi.txt"),
        "PDO": read_psl(CI / "pdo.txt"),
    })
    idx["ATL3"] = atl3_from_ersst()
    idx["TNA_TSA"] = idx.TNA - idx.TSA
    idx = idx.loc["1950-01-01":"2020-12-01", ["AMO", "Nino34", "ATL3", "TNA_TSA", "DMI", "PDO"]]
    return idx


# ------------------------------------------------------------------ precipitation
def catchment_precip():
    g = xr.open_dataset(ROOT / "data/external/gpcc_v2022_volta_05.nc")
    pr = g.precip.sel(time=slice("1951-01-01", "2020-12-31")).load()
    lat, lon = pr.lat.values, pr.lon.values
    r = pd.read_parquet(ROOT / "data/reanalysis/reaches_volta.parquet").set_index("LINKNO")
    children = defaultdict(list)
    for k, dsl in r.DSLINKNO.items():
        children[int(dsl)].append(int(k))
    gm = pd.read_csv(ROOT / "data/reanalysis/t2500/gauge_matches.csv").dropna(subset=["river_id"])

    def ups(k):
        out, stack = [], [k]
        while stack:
            x = stack.pop(); out.append(x); stack.extend(children.get(x, []))
        return out

    def cell(v, grid):
        return int(np.abs(grid - v).argmin())

    P, W = {}, {}
    for _, row in gm.iterrows():
        u = r.loc[[k for k in ups(int(row.river_id)) if k in r.index]]
        cells = pd.Series([(cell(a, lat), cell(b, lon)) for a, b in zip(u.lat, u.lon)]).value_counts()
        w = np.array(cells.values, float); w /= w.sum()
        P[int(row.grdc_no)] = sum(wi * pr.isel(lat=i, lon=j).values for (i, j), wi in zip(cells.index, w))
        W[int(row.grdc_no)] = [(float(lat[i]), float(lon[j]), float(wi)) for (i, j), wi in zip(cells.index, w)]
    # basin above Akosombo: GPCC cells inside the GRDC basin polygon, north of 6.5 N, cos(lat) weights
    import geopandas as gpd
    from shapely.geometry import Point
    poly = gpd.read_file(ROOT / "data/GRDC_VOLTA_POLITICAL_TRANSBOUNDARY_BASIN.shp").to_crs(4326).union_all()
    cells = []
    for i, a in enumerate(lat):
        for j, b in enumerate(lon):
            if a > 6.5 and poly.contains(Point(float(b), float(a))):
                cells.append((i, j, np.cos(np.deg2rad(a))))
    wsum = sum(c[2] for c in cells)
    P["basin"] = sum(c[2] / wsum * pr.isel(lat=c[0], lon=c[1]).values for c in cells)
    W["basin"] = [(float(lat[i]), float(lon[j]), float(w / wsum)) for i, j, w in cells]
    PR = pd.DataFrame(P, index=pd.to_datetime(pr.time.values))
    return PR, W


def spi(x: pd.Series, k=12, ref=REF) -> pd.Series:
    acc = x.rolling(k).sum()
    out = pd.Series(np.nan, index=x.index)
    for m in range(1, 13):
        sel = acc.index.month == m
        a = acc[sel]
        r_ = a[(a.index.year >= ref[0]) & (a.index.year <= ref[1])].dropna()
        q0 = float((r_ <= 0).mean()); pos = r_[r_ > 0]
        shape, _, scale = stats.gamma.fit(pos, floc=0)
        cdf = np.where(a > 0, q0 + (1 - q0) * stats.gamma.cdf(a, shape, loc=0, scale=scale), q0 / 2 if q0 > 0 else 1e-6)
        cdf = np.where(a.isna(), np.nan, cdf)
        out[sel] = stats.norm.ppf(np.clip(cdf, 1e-6, 1 - 1e-6))
    return out


# ------------------------------------------------------------------ homogeneity and trend
def pettitt(x):
    x = np.asarray(x, float); n = len(x)
    s = np.sign(x[None, :] - x[:, None])
    U = np.array([s[:t + 1, t + 1:].sum() for t in range(n - 1)])
    K = np.abs(U).max(); t = int(np.abs(U).argmax())
    p = min(1.0, 2 * np.exp(-6 * K ** 2 / (n ** 3 + n ** 2)))
    return t, K, p


def snht_stat(x):
    z = (x - x.mean()) / x.std(ddof=1); n = len(z)
    T = np.array([k * z[:k].mean() ** 2 + (n - k) * z[k:].mean() ** 2 for k in range(1, n)])
    return T.max(), int(T.argmax()) + 1


def buishand_stat(x):
    S = np.cumsum(x - x.mean()); s = x.std(ddof=1)
    return (S.max() - S.min()) / s / np.sqrt(len(x)), int(np.abs(S).argmax()) + 1


def perm_p(stat_fn, x, n=NPERM):
    obs = stat_fn(x)[0]
    cnt = sum(stat_fn(RNG.permutation(x))[0] >= obs for _ in range(n))
    return (1 + cnt) / (1 + n)


def sen_slope(t, x):
    t = np.asarray(t, float); x = np.asarray(x, float)
    i, j = np.triu_indices(len(x), 1)
    d = (x[j] - x[i]) / (t[j] - t[i])
    return float(np.median(d))


def trend_block(name, years, x, unit):
    years = np.asarray(years); x = np.asarray(x, float)
    ok = ~np.isnan(x); years, x = years[ok], x[ok]
    n = len(x)
    t_cp, K, p_pet = pettitt(x)
    T0, k_snht = snht_stat(x); p_snht = perm_p(snht_stat, x, 1999)
    Q, k_bu = buishand_stat(x); p_bu = perm_p(buishand_stat, x, 1999)
    mko = mk.original_test(x)
    mkh = mk.hamed_rao_modification_test(x)
    slope = sen_slope(years, x)
    # DWT-MK: db4 approximation at level 2
    lev = max(1, min(2, pywt.dwt_max_level(n, pywt.Wavelet("db4").dec_len)))
    coeffs = pywt.wavedec(x, "db4", level=lev, mode="periodization" if n % 4 == 0 else "symmetric")
    a_only = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    A = pywt.waverec(a_only, "db4", mode="periodization" if n % 4 == 0 else "symmetric")[:n]
    mka = mk.hamed_rao_modification_test(A)
    cp_year = int(years[t_cp])
    pre, post = x[: t_cp + 1], x[t_cp + 1:]
    return dict(series=name, unit=unit, n=n, first=int(years[0]), last=int(years[-1]),
                pettitt_year=cp_year, pettitt_K=float(K), pettitt_p=float(p_pet),
                snht_T0=float(T0), snht_year=int(years[k_snht - 1]), snht_p=float(p_snht),
                buishand_Q=float(Q), buishand_year=int(years[k_bu - 1]), buishand_p=float(p_bu),
                mean_before=float(pre.mean()), mean_after=float(post.mean()) if len(post) else np.nan,
                mk_tau=float(mko.Tau), mk_p=float(mko.p), mk_hr_z=float(mkh.z), mk_hr_p=float(mkh.p),
                sen_per_decade=float(slope * 10),
                dwt_mk_z=float(mka.z), dwt_mk_p=float(mka.p))


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for rank, i in enumerate(o[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k); q[i] = prev
    return q


# ------------------------------------------------------------------ wavelet helpers
def cwt_block(x, dt=1 / 12., dj=1 / 12., s0=2 / 12.):
    x = np.asarray(x, float)
    xs = (x - x.mean()) / x.std()
    N = len(xs)
    J = int(np.log2(N * dt / s0) / dj)
    mother = pycwt.Morlet(6)
    alpha = float(pd.Series(xs).autocorr(1))
    wave, scales, freqs, coi, _, _ = pycwt.cwt(xs, dt, dj, s0, J, mother)
    power = np.abs(wave) ** 2
    period = 1 / freqs
    signif, _ = pycwt.significance(1.0, dt, scales, 0, alpha, significance_level=0.95, wavelet=mother)
    sig95 = power / signif[:, None]
    glbl = power.mean(axis=1)
    dof = N - scales
    gsig, _ = pycwt.significance(1.0, dt, scales, 1, alpha, significance_level=0.95, dof=dof, wavelet=mother)
    return dict(power=power, period=period, coi=coi, sig95=sig95, gws=glbl, gws_sig=gsig, alpha=alpha)


def coherence_block(a, b, dt=1 / 12., dj=1 / 8., s0=2 / 12., mc=200, pmax=32.0):
    a = (np.asarray(a, float) - np.mean(a)) / np.std(a)
    b = (np.asarray(b, float) - np.mean(b)) / np.std(b)
    N = len(a)
    J = int(np.log2(pmax / 1.033 / s0) / dj)   # periods up to ~32 years
    # pycwt draws its red-noise surrogates from numpy's global generator; seed it so that the
    # Monte Carlo significance level, and hence the significant shares, are reproducible
    np.random.seed(2026)
    WCT, aWCT, coi, freq, sig = pycwt.wct(a, b, dt, dj=dj, s0=s0, J=J, sig=True, significance_level=0.95,
                                         wavelet="morlet", normalize=True, cache=False, progress=False, mc_count=mc)
    W12, cross_coi, freq2, sig_x = pycwt.xwt(a, b, dt, dj=dj, s0=s0, J=J, significance_level=0.95,
                                             wavelet="morlet", normalize=True)
    period = 1 / freq
    t = np.arange(N) * dt
    inside = period[:, None] <= coi[None, :]
    sigmask = (WCT >= sig[:, None]) & inside
    out = {}
    for lo, hi, lab in [(2, 8, "2-8 yr"), (8, 32, "8-32 yr")]:
        band = (period >= lo) & (period < hi)
        m_in = inside[band]
        if m_in.sum() == 0:
            out[lab] = dict(share_sig=np.nan, mean_coh=np.nan, phase_deg=np.nan, lag_months=np.nan)
            continue
        share = float(sigmask[band][m_in].mean()) if m_in.any() else np.nan
        mean_coh = float(WCT[band][m_in].mean())
        ph = aWCT[band][sigmask[band]]
        if ph.size:
            mphase = float(np.angle(np.mean(np.exp(1j * ph))))
            per = np.repeat(period[band][:, None], N, axis=1)[sigmask[band]]
            lag_m = float(np.median(mphase / (2 * np.pi) * per) * 12)
        else:
            mphase, lag_m = np.nan, np.nan
        out[lab] = dict(share_sig=share, mean_coh=mean_coh, phase_deg=float(np.degrees(mphase)) if ph.size else np.nan,
                        lag_months=lag_m)
    return out, dict(WCT=WCT, aWCT=aWCT, coi=coi, period=period, sig=sig, XWT=np.abs(W12), xsig=sig_x)


def lag_corr(x, y, maxlag=24):
    """r(x_t, y_{t-L}): positive L = index leads."""
    x = pd.Series(np.asarray(x, float)); y = pd.Series(np.asarray(y, float))
    r1x, r1y = x.autocorr(1), y.autocorr(1)
    out = []
    for L in range(-maxlag, maxlag + 1):
        yy = y.shift(L)
        ok = x.notna() & yy.notna()
        r = float(np.corrcoef(x[ok], yy[ok])[0, 1]); n = int(ok.sum())
        neff = n * (1 - r1x * r1y) / (1 + r1x * r1y)
        tval = r * np.sqrt(max(neff - 2, 1) / max(1 - r ** 2, 1e-9))
        p = 2 * stats.t.sf(abs(tval), max(neff - 2, 1))
        out.append((L, r, n, neff, p))
    return pd.DataFrame(out, columns=["lag_months", "r", "n", "n_eff", "p"])


# ------------------------------------------------------------------ main
def main():
    st = pd.read_csv(ROOT / "data/grdc_stations_raw.csv").set_index("grdc_no")
    names = st.station_name.to_dict()
    PR, W = catchment_precip()
    json.dump({str(k): v for k, v in W.items()}, open(TAB / "catchment_gpcc_cells.json", "w"))
    idx = load_indices()
    idx.to_csv(TAB / "climate_indices_monthly.csv")

    SPI12 = PR.apply(lambda c: spi(c, 12)); SPI6 = PR.apply(lambda c: spi(c, 6))
    SPI12.to_csv(TAB / "spi12_catchments_basin.csv"); SPI6.to_csv(TAB / "spi6_catchments_basin.csv")
    PR.to_csv(TAB / "precip_catchments_basin.csv")

    # ---------------- 1. trends and change points
    rows = []
    ann = PR.resample("YS").sum()
    jjaso = PR[PR.index.month.isin([6, 7, 8, 9, 10])].resample("YS").sum()
    yrs = ann.index.year.values
    gauge_cols = [c for c in PR.columns if c != "basin"]
    subb = {"Black Volta (Bamboi)": 1531100, "White Volta (Nawuni)": 1531450, "Oti (Sabari)": 1531800,
            "Pru (Prang)": 1531300}
    subb = {k: v for k, v in subb.items() if v in PR.columns}
    rows.append(trend_block("Basin annual precipitation", yrs, ann["basin"].values, "mm"))
    rows.append(trend_block("Basin June-October precipitation", yrs, jjaso["basin"].values, "mm"))
    d12 = (SPI12["basin"] <= -1).groupby(SPI12.index.year).sum()
    rows.append(trend_block("Basin months with SPI-12 <= -1", d12.index.values, d12.values.astype(float), "months/yr"))
    for lab, g in subb.items():
        rows.append(trend_block(f"{lab} annual precipitation", yrs, ann[g].values, "mm"))
    sahel = read_psl(ROOT / "data/external/sahelrain.data", missing=(), floor=None)
    sj = sahel[sahel.index.month.isin([6, 7, 8, 9, 10])].resample("YS").sum(min_count=5)
    sj = sj["1951":"2017"]
    rows.append(trend_block("Sahel index, June-October sum", sj.index.year.values, sj.values, "index units"))
    # drought synchrony
    S = (SPI6[gauge_cols] <= -1).mean(axis=1)
    S = S[S.index.year >= 1951].dropna()
    Sa = S.groupby(S.index.year).mean()
    rows.append(trend_block("Drought synchrony (share of 18 catchments with SPI-6 <= -1)", Sa.index.values, Sa.values, "share"))
    tr = pd.DataFrame(rows)
    # catchment-level field significance
    crow = []
    for g in gauge_cols:
        b = trend_block(str(g), yrs, ann[g].values, "mm")
        b["station"] = names.get(g, str(g)); crow.append(b)
    ct = pd.DataFrame(crow)
    ct["mk_hr_q_bh"] = bh(ct.mk_hr_p.values); ct["pettitt_q_bh"] = bh(ct.pettitt_p.values)
    tr.to_csv(TAB / "trend_changepoint_summary.csv", index=False)
    ct.to_csv(TAB / "trend_changepoint_catchments.csv", index=False)

    # segment trends around 1970 and 1990 for the basin
    seg = []
    for a, b_ in [(1951, 1970), (1971, 1990), (1991, 2020), (1961, 2007), (1951, 2020)]:
        s_ = ann["basin"][(yrs >= a) & (yrs <= b_)]
        t_ = mk.hamed_rao_modification_test(s_.values)
        seg.append(dict(period=f"{a}-{b_}", n=len(s_), mean_mm=float(s_.mean()),
                        sen_mm_per_decade=sen_slope(s_.index.year, s_.values) * 10, mk_hr_p=float(t_.p)))
    pd.DataFrame(seg).to_csv(TAB / "trend_segments_basin.csv", index=False)

    # ---------------- 2. gauge baseline regime: mean catchment SPI-12 over years with data
    sd = pd.read_csv(TO / "SSI_daily.csv", usecols=["node_id", "date", "SSI"], parse_dates=["date"])
    sd = sd.dropna(subset=["SSI"])
    base = []
    for g, d in sd.groupby("node_id"):
        if g not in SPI12.columns:
            continue
        yrs_obs = d.date.dt.year.value_counts()
        yrs_obs = yrs_obs[yrs_obs >= 60].index
        s12 = SPI12[g][SPI12.index.month == 12]
        s12.index = s12.index.year
        v = s12.reindex(sorted(yrs_obs)).dropna()
        base.append(dict(grdc_no=g, station=names.get(g, g), years_defined=len(v),
                         first=int(min(yrs_obs)) if len(yrs_obs) else None, last=int(max(yrs_obs)) if len(yrs_obs) else None,
                         mean_spi12_reference=float(v.mean()) if len(v) else np.nan,
                         share_years_before_1971=float((v.index <= 1970).mean()) if len(v) else np.nan))
    pd.DataFrame(base).sort_values("mean_spi12_reference").to_csv(TAB / "gauge_reference_regime.csv", index=False)

    # ---------------- 3. observed event rate: trend and climate covariates
    ev = pd.read_csv(TO / "SSI_drought_events.csv", parse_dates=["start_date"])
    cnt = ev.start_date.dt.year.value_counts().reindex(range(1961, 2008), fill_value=0).sort_index()
    expo = sd.groupby(sd.date.dt.year).size().reindex(range(1961, 2008), fill_value=0) / 365.25
    dfr = pd.DataFrame({"events": cnt, "exposure": expo})
    dfr = dfr[dfr.exposure > 0.5]
    seas = {
        "AMO": idx.AMO[idx.index.month.isin(range(6, 11))],
        "Nino34": idx.Nino34[idx.index.month.isin([7, 8, 9])],
        "ATL3": idx.ATL3[idx.index.month.isin([6, 7, 8])],
        "TNA_TSA": idx.TNA_TSA[idx.index.month.isin(range(6, 11))],
        "DMI": idx.DMI[idx.index.month.isin([7, 8, 9, 10])],
        "PDO": idx.PDO,
    }
    for k, s in seas.items():
        a = s.groupby(s.index.year).mean()
        dfr[k] = a.reindex(dfr.index).values
    glm_rows = []
    X = sm.add_constant(((dfr.index.values - 1984) / 10.0).astype(float))
    m = sm.GLM(dfr.events.values, X, family=sm.families.Poisson(), offset=np.log(dfr.exposure.values)).fit()
    disp = float(m.pearson_chi2 / m.df_resid)
    mq = sm.GLM(dfr.events.values, X, family=sm.families.Poisson(), offset=np.log(dfr.exposure.values)).fit(scale="X2")
    b, se = mq.params[1], mq.bse[1]
    glm_rows.append(dict(covariate="decade (trend)", rate_ratio=float(np.exp(b)), lo95=float(np.exp(b - 1.96 * se)),
                         hi95=float(np.exp(b + 1.96 * se)), p=float(mq.pvalues[1]), dispersion=disp, years=len(dfr)))
    ps = []
    for k in seas:
        z = (dfr[k] - dfr[k].mean()) / dfr[k].std()
        Xk = sm.add_constant(z.values)
        mk_ = sm.GLM(dfr.events.values, Xk, family=sm.families.Poisson(), offset=np.log(dfr.exposure.values)).fit(scale="X2")
        b, se = mk_.params[1], mk_.bse[1]
        glm_rows.append(dict(covariate=f"{k} (per SD)", rate_ratio=float(np.exp(b)), lo95=float(np.exp(b - 1.96 * se)),
                             hi95=float(np.exp(b + 1.96 * se)), p=float(mk_.pvalues[1]), dispersion=np.nan, years=len(dfr)))
        ps.append(float(mk_.pvalues[1]))
    g = pd.DataFrame(glm_rows)
    holm = np.minimum(1, np.maximum.accumulate(np.sort(ps) * (len(ps) - np.arange(len(ps)))))
    order = np.argsort(ps); hp = np.empty(len(ps)); hp[order] = holm
    g.loc[1:, "p_holm"] = hp
    g.to_csv(TAB / "event_rate_glm.csv", index=False)
    dfr.to_csv(TAB / "event_rate_by_year.csv")

    # ---------------- 4. wavelets: basin SPI-12 and synchrony with indices, 1951-2020
    b12 = SPI12["basin"].dropna()
    b12 = b12["1952-01-01":"2020-12-01"]
    I = idx.reindex(b12.index).interpolate(limit=2)
    cw = cwt_block(b12.values)
    np.savez_compressed(FD / "cwt_basin_spi12.npz", t=b12.index.year + (b12.index.month - 1) / 12, **cw)
    sum_rows, lagtabs = [], []
    for k in I.columns:
        res, arr = coherence_block(b12.values, I[k].values)
        np.savez_compressed(FD / f"wct_spi12_{k}.npz", t=b12.index.year + (b12.index.month - 1) / 12, **arr)
        for lab, v in res.items():
            sum_rows.append(dict(series="basin SPI-12", index=k, band=lab, **v))
        lt = lag_corr(b12.values, I[k].values); lt["index"] = k; lt["series"] = "basin SPI-12"; lagtabs.append(lt)
    S2 = S["1952-01-01":"2020-12-01"]
    S2 = S2.rolling(12, min_periods=12).mean().dropna()
    I2 = idx.reindex(S2.index).interpolate(limit=2)
    for k in I2.columns:
        res, arr = coherence_block(S2.values, I2[k].values)
        np.savez_compressed(FD / f"wct_sync_{k}.npz", t=S2.index.year + (S2.index.month - 1) / 12, **arr)
        for lab, v in res.items():
            sum_rows.append(dict(series="drought synchrony (12-month mean)", index=k, band=lab, **v))
        lt = lag_corr(S2.values, I2[k].values); lt["index"] = k; lt["series"] = "synchrony"; lagtabs.append(lt)
    pd.DataFrame(sum_rows).to_csv(TAB / "wavelet_coherence_summary.csv", index=False)
    pd.concat(lagtabs).to_csv(TAB / "lag_correlation.csv", index=False)
    # global wavelet spectrum peaks
    per, gws, gsig = cw["period"], cw["gws"], cw["gws_sig"]
    peaks = [dict(period_yr=float(per[i]), power=float(gws[i]), red_noise_95=float(gsig[i]), significant=bool(gws[i] > gsig[i]))
             for i in range(1, len(per) - 1) if gws[i] > gws[i - 1] and gws[i] > gws[i + 1]]
    pd.DataFrame(peaks).to_csv(TAB / "gws_peaks_basin_spi12.csv", index=False)

    # ---------------- 5. synchrony at the onset of opportunities
    opp = pd.read_csv(TO / "review_opportunities.csv")
    ev2 = ev.set_index(["node_id", "event_id"])
    pairs = pd.read_csv(TO / "propagation_pairs.csv")
    con = set(zip(pairs.origin_node, pairs.origin_event_id))
    srow = []
    for _, o in opp.iterrows():
        sdate = pd.Timestamp(ev2.loc[(o.node_id, o.event_id), "start_date"]).to_period("M").to_timestamp()
        srow.append(dict(node_id=o.node_id, event_id=o.event_id, n_observing=o.n_observing,
                         opportunity=o.n_observing > 0, connected=(o.node_id, o.event_id) in con,
                         sync=float(S.get(sdate, np.nan)),
                         **{k: float(idx[k].get(sdate, np.nan)) for k in idx.columns}))
    so = pd.DataFrame(srow)
    so.to_csv(TAB / "synchrony_at_onset.csv", index=False)
    oo = so[so.opportunity]
    mw = stats.mannwhitneyu(oo[oo.connected].sync, oo[~oo.connected].sync, alternative="greater")
    clim = S["1961":"2007"]
    mw2 = stats.mannwhitneyu(so.sync.dropna(), clim.values, alternative="greater")
    summ = dict(
        sync_clim_mean_1961_2007=float(clim.mean()),
        sync_origin_events_median=float(so.sync.median()),
        sync_connected_median=float(oo[oo.connected].sync.median()),
        sync_unconnected_median=float(oo[~oo.connected].sync.median()),
        mw_connected_vs_unconnected_p=float(mw.pvalue),
        mw_origin_vs_climatology_p=float(mw2.pvalue),
        n_connected=int(oo.connected.sum()), n_unconnected=int((~oo.connected).sum()),
        basin_spi12_ar1=cw["alpha"],
    )
    json.dump(summ, open(TAB / "synchrony_summary.json", "w"), indent=1)
    print(tr[["series", "pettitt_year", "pettitt_p", "snht_p", "buishand_p", "mk_hr_p", "sen_per_decade", "dwt_mk_p"]].to_string())
    print(g.to_string()); print(json.dumps(summ, indent=1))
    print(pd.DataFrame(sum_rows).to_string())


if __name__ == "__main__":
    main()
