#!/usr/bin/env python3
"""
Future hydrological drought risk in the Volta Basin (objective iv).

1. Catchment series from 16 CMIP6 models (one realisation per centre; historical 1950-2014,
   SSP2-4.5 and SSP5-8.5 2015-2100): pr and mrro (mm/month), tas (deg C), bilinear
   interpolation to the GPCC 0.5 deg cells of each catchment and the same weights.
2. Precipitation bias adjustment by quantile delta mapping (QDM; Cannon et al., 2015),
   multiplicative, per calendar month, observed reference GPCC 1961-2014, 30-year
   moving windows in the projection period.
3. Hybrid wavelet-machine learning transfer model per gauge: causal (a trous) Haar wavelet
   decomposition of monthly catchment precipitation (scales 1-64 months; only past and
   present values enter, so no information leaks from the future) plus two seasonal
   harmonics; ridge regression, random forest and XGBoost predict log monthly discharge;
   contiguous year-block cross-validation (5 folds); ensemble weights proportional to the
   inverse out-of-fold MSE.
4. Projected drought indices: SSI-12 of the simulated discharge (transfer model, BC pr),
   SPI-12 (BC pr) and SRI-12 (model total runoff, which carries each model's own
   evapotranspiration response to warming), each standardised against the same simulation's
   1985-2014 baseline. A temperature-only (Thornthwaite) SPEI was tested and dropped: with
   tas as the only driver of PET it labels every month of 2051-2080 as drought in some models,
   the known overstatement of drying by temperature-based PET.
5. Drought metrics for 1985-2014, 2021-2050 and 2051-2080: share of months below -1, -1.5
   and -2; drought events (runs below -1 of at least 3 months) per decade, mean duration.
6. Years of concurrent observation needed for 50% and 80% power of the availability-aware
   test under three monitoring designs, scaled by the projected change in drought frequency.

Writes output_projection/tables/*.csv|json and output_projection/figdata/*.
Author: I. Quist (NUIST)
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
CM = ROOT / "data/external/cmip6/subsets"
OUT = ROOT / "output_projection"
TAB, FD = OUT / "tables", OUT / "figdata"
TAB.mkdir(parents=True, exist_ok=True); FD.mkdir(parents=True, exist_ok=True)
TC = ROOT / "output_climate/tables"
SOURCES = ["ACCESS-CM2", "BCC-CSM2-MR", "CAMS-CSM1-0", "CESM2-WACCM", "CNRM-CM6-1", "CanESM5",
           "FIO-ESM-2-0", "GFDL-ESM4", "INM-CM5-0", "IPSL-CM6A-LR", "KACE-1-0-G", "MCM-UA-1-0",
           "MIROC6", "MPI-ESM1-2-HR", "MRI-ESM2-0", "UKESM1-0-LL"]
SCEN = ["ssp245", "ssp585"]
OBS_REF = (1961, 2014)
BASE = (1985, 2014)
PERIODS = {"1985-2014": (1985, 2014), "2021-2050": (2021, 2050), "2051-2080": (2051, 2080)}
SCALES = [1, 2, 4, 8, 16, 32, 64]
RNG = np.random.default_rng(2026)
MIN_MONTHS = 96
SKILL_RULE = dict(nse_log=0.5, anom_rho=0.4)   # stated before fitting


# ------------------------------------------------------------------ CMIP6 catchment series
def load_field(src, var, exp):
    f = CM / f"{src}_{var}_{exp}.nc"
    if not f.exists():
        return None
    da = xr.open_dataset(f)[var]
    da = da.assign_coords(time=pd.to_datetime(da.time.values))
    return da


def catchment_series(da, cells):
    lat = xr.DataArray([c[0] for c in cells], dims="pt")
    lon = xr.DataArray([c[1] for c in cells], dims="pt")
    w = xr.DataArray([c[2] for c in cells], dims="pt")
    v = da.interp(lat=lat, lon=lon, method="linear")
    if bool(v.isnull().any()):
        vn = da.sel(lat=lat, lon=lon, method="nearest")
        v = v.fillna(vn)
    return (v * w).sum("pt").to_series()


def model_catchments(src, cells):
    out = {}
    for var in ("pr", "tas", "mrro"):
        series = {}
        hist = load_field(src, var, "historical")
        if hist is None:
            continue
        for exp in SCEN:
            fut = load_field(src, var, exp)
            if fut is None:
                continue
            da = xr.concat([hist, fut], dim="time")
            df = pd.DataFrame({k: catchment_series(da, c) for k, c in cells.items()})
            if var in ("pr", "mrro"):
                df = df * 86400 * df.index.days_in_month.values[:, None]
            else:
                df = df - 273.15
            series[exp] = df.clip(lower=0) if var != "tas" else df
        out[var] = series
    return out


# ------------------------------------------------------------------ QDM
def qdm(obs: pd.Series, mod: pd.Series, ref=OBS_REF, thr=0.05, win=30):
    """Multiplicative quantile delta mapping per calendar month."""
    out = pd.Series(np.nan, index=mod.index)
    mod = mod.copy()
    small = mod < thr
    mod[small] = RNG.uniform(0, thr, small.sum())
    obs = obs.copy(); so = obs < thr; obs[so] = RNG.uniform(0, thr, so.sum())
    yrs = mod.index.year
    for m in range(1, 13):
        o = obs[(obs.index.month == m) & (obs.index.year >= ref[0]) & (obs.index.year <= ref[1])].values
        mh = mod[(mod.index.month == m) & (yrs >= ref[0]) & (yrs <= ref[1])].values
        qs = np.linspace(0.5 / len(mh), 1 - 0.5 / len(mh), len(mh))
        qo = np.quantile(o, qs); qmh = np.sort(mh)
        sel_m = mod.index.month == m
        # historical part: plain QM against the reference distribution
        for y0 in range(1950, 2101, 10):
            blk = sel_m & (yrs >= y0) & (yrs < y0 + 10)
            if not blk.any():
                continue
            if y0 + 10 <= 2015:
                wsel = sel_m & (yrs >= ref[0]) & (yrs <= ref[1])
            else:
                c = y0 + 5
                a, b = max(2015 - 5, c - win // 2), min(2100, c + win // 2 - 1)
                if b - a + 1 < win:
                    a = max(1985, b - win + 1)
                wsel = sel_m & (yrs >= a) & (yrs <= b)
            xw = np.sort(mod[wsel].values)
            x = mod[blk].values
            tau = np.clip((np.searchsorted(xw, x, side="right") - 0.5) / len(xw), qs[0], qs[-1])
            ref_mh = np.interp(tau, qs, qmh)
            delta = np.where(ref_mh > 0, x / ref_mh, 1.0)
            out[blk] = np.interp(tau, qs, qo) * delta
    out[out < thr] = 0.0
    return out


# ------------------------------------------------------------------ standardised indices
def std_gamma(x: pd.Series, k=12, base=BASE):
    acc = x.rolling(k).sum() if k > 1 else x
    out = pd.Series(np.nan, index=x.index)
    for m in range(1, 13):
        sel = acc.index.month == m
        a = acc[sel]
        r = a[(a.index.year >= base[0]) & (a.index.year <= base[1])].dropna()
        q0 = float((r <= 0).mean()); pos = r[r > 0]
        if len(pos) < 10:
            continue
        sh, _, sc = stats.gamma.fit(pos, floc=0)
        cdf = np.where(a > 0, q0 + (1 - q0) * stats.gamma.cdf(a, sh, loc=0, scale=sc), q0 / 2 if q0 > 0 else 1e-6)
        cdf = np.where(a.isna(), np.nan, cdf)
        out[sel] = stats.norm.ppf(np.clip(cdf, 1e-6, 1 - 1e-6))
    return out


def std_loglogistic(x: pd.Series, k=12, base=BASE):
    acc = x.rolling(k).sum()
    out = pd.Series(np.nan, index=x.index)
    for m in range(1, 13):
        sel = acc.index.month == m
        a = acc[sel]
        r = a[(a.index.year >= base[0]) & (a.index.year <= base[1])].dropna()
        try:
            c, loc, sc = stats.fisk.fit(r.values, loc=r.min() - 0.1 * r.std())
            cdf = stats.fisk.cdf(a, c, loc=loc, scale=sc)
            if not np.isfinite(cdf[~a.isna()]).all():
                raise ValueError
        except Exception:  # noqa: BLE001
            cdf = stats.norm.cdf((a - r.mean()) / r.std())
        out[sel] = stats.norm.ppf(np.clip(cdf, 1e-6, 1 - 1e-6))
    return out


def thornthwaite(t: pd.Series, lat: float):
    """Monthly PET (mm) by Thornthwaite (1948) from mean temperature (deg C)."""
    tt = t.clip(lower=0)
    tb = tt[(tt.index.year >= BASE[0]) & (tt.index.year <= BASE[1])]
    clim = tb.groupby(tb.index.month).mean()
    I = ((clim / 5) ** 1.514).sum()
    a = 6.75e-7 * I ** 3 - 7.71e-5 * I ** 2 + 1.792e-2 * I + 0.49239
    doy = np.array([15, 46, 75, 105, 135, 162, 198, 228, 258, 288, 318, 344])
    phi = np.deg2rad(lat)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(-np.tan(phi) * np.tan(dec))
    N = 24 / np.pi * ws
    m = tt.index.month.values - 1
    days = tt.index.days_in_month.values
    pet = 16 * (N[m] / 12) * (days / 30) * (10 * tt.values / I) ** a
    return pd.Series(pet, index=t.index)


def drought_metrics(s: pd.Series, period):
    x = s[(s.index.year >= period[0]) & (s.index.year <= period[1])].dropna()
    if len(x) == 0:
        return {}
    below = (x < -1).values
    runs, cur = [], 0
    for b in below:
        if b:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    ev = [r for r in runs if r >= 3]
    yrs = len(x) / 12
    return {"p_below_1": float((x < -1).mean()), "p_below_1_5": float((x < -1.5).mean()),
            "p_below_2": float((x < -2).mean()), "events_per_decade": 10 * len(ev) / yrs,
            "mean_duration_months": float(np.mean(ev)) if ev else 0.0}


# ------------------------------------------------------------------ wavelet features
def haar_atrous(p: pd.Series) -> pd.DataFrame:
    A = {j: p.rolling(s, min_periods=s).mean() for j, s in enumerate(SCALES)}
    f = {f"D{j}": A[j - 1] - A[j] for j in range(1, len(SCALES))}
    f[f"A{len(SCALES) - 1}"] = A[len(SCALES) - 1]
    f["A0"] = A[0]
    m = p.index.month.values
    f["s1"] = np.sin(2 * np.pi * m / 12); f["c1"] = np.cos(2 * np.pi * m / 12)
    f["s2"] = np.sin(4 * np.pi * m / 12); f["c2"] = np.cos(4 * np.pi * m / 12)
    return pd.DataFrame(f, index=p.index)


def models():
    return {
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25))),
        "rf": RandomForestRegressor(n_estimators=400, min_samples_leaf=5, max_features=0.5, random_state=2026, n_jobs=-1),
        "xgb": XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                            reg_lambda=1.0, random_state=2026, n_jobs=4),
    }


def observed_monthly():
    q = pd.read_csv(ROOT / "data/grdc_discharge_raw.csv", parse_dates=["date"]).dropna(subset=["Q"])
    q = q[(q.date >= "1961-01-01") & (q.date <= "2007-12-31")]
    out = {}
    for g, d in q.groupby("grdc_no"):
        s = d.set_index("date").Q
        m = s.resample("MS").agg(["mean", "count"])
        m = m[m["count"] >= m.index.days_in_month - 3]["mean"]
        out[int(g)] = m
    return out


def fit_transfer(PR_obs: pd.DataFrame, names: dict):
    obs = observed_monthly()
    skill, fitted = [], {}
    oof_store = {}
    for g, q in obs.items():
        if str(g) not in PR_obs.columns or len(q) < MIN_MONTHS:
            continue
        X = haar_atrous(PR_obs[str(g)]).reindex(q.index)
        ok = X.notna().all(axis=1)
        X, qq = X[ok], q[ok]
        eps = 0.01 * qq.mean()
        y = np.log(qq.values + eps)
        yrs = qq.index.year.values
        uy = np.unique(yrs)
        folds = np.array_split(uy, 5)
        oof = {k: np.full(len(y), np.nan) for k in models()}
        for fy in folds:
            te = np.isin(yrs, fy); tr = ~te
            for k, mdl in models().items():
                mdl.fit(X[tr], y[tr]); oof[k][te] = mdl.predict(X[te])
        mse = {k: float(np.mean((oof[k] - y) ** 2)) for k in oof}
        w = {k: (1 / v) / sum(1 / u for u in mse.values()) for k, v in mse.items()}
        ens = sum(w[k] * oof[k] for k in oof)
        clim = pd.Series(y, index=qq.index).groupby(qq.index.month).transform("mean").values
        def nse(p): return float(1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2))
        def anom_rho(p): return float(stats.spearmanr(p - clim, y - clim)[0])
        row = dict(grdc_no=g, station=names.get(g, g), months=int(len(y)), years=int(len(uy)))
        for k in list(oof) + ["ensemble"]:
            p = ens if k == "ensemble" else oof[k]
            row[f"nse_log_{k}"] = nse(p); row[f"anom_rho_{k}"] = anom_rho(p)
        row.update({f"w_{k}": w[k] for k in w})
        # 12-month mean flows on complete windows
        full = pd.Series(np.exp(y) - eps, index=qq.index).reindex(pd.date_range(qq.index.min(), qq.index.max(), freq="MS"))
        pe = pd.Series(np.exp(ens) - eps, index=qq.index).reindex(full.index)
        o12, p12 = full.rolling(12).mean(), pe.rolling(12).mean()
        okk = o12.notna() & p12.notna()
        row["q12_windows"] = int(okk.sum())
        row["q12_rho"] = float(stats.spearmanr(o12[okk], p12[okk])[0]) if okk.sum() > 12 else np.nan
        row["selected"] = bool(row["nse_log_ensemble"] >= SKILL_RULE["nse_log"] and row["anom_rho_ensemble"] >= SKILL_RULE["anom_rho"])
        skill.append(row)
        oof_store[g] = pd.DataFrame({"obs": y, "ens": ens, **{k: oof[k] for k in oof}}, index=qq.index)
        # final fit on all data
        fm = {k: m.fit(X, y) for k, m in models().items()}
        fitted[g] = dict(models=fm, w=w, eps=eps)
        print(row, flush=True)
    sk = pd.DataFrame(skill)
    sk.to_csv(TAB / "transfer_model_skill.csv", index=False)
    pd.concat({g: d for g, d in oof_store.items()}).to_csv(TAB / "transfer_model_oof.csv")
    return sk, fitted


def main():
    cells = {k: v for k, v in json.load(open(TC / "catchment_gpcc_cells.json")).items()}
    PR_obs = pd.read_csv(TC / "precip_catchments_basin.csv", index_col=0, parse_dates=True)
    PR_obs.columns = [str(c) for c in PR_obs.columns]
    st = pd.read_csv(ROOT / "data/grdc_stations_raw.csv").set_index("grdc_no")
    names = st.station_name.to_dict()
    sk, fitted = fit_transfer(PR_obs, names)
    sel = [int(g) for g in sk[sk.selected].grdc_no]
    keys = [str(g) for g in sel] + ["basin"]
    lat_c = {k: float(np.average([c[0] for c in cells[k]], weights=[c[2] for c in cells[k]])) for k in cells}
    rows = []
    ann = []
    for src in SOURCES:
        mc = model_catchments(src, {k: cells[k] for k in keys})
        if "pr" not in mc or not all(s in mc["pr"] for s in SCEN):
            print("skip", src); continue
        for scen in SCEN:
            P = mc["pr"][scen]
            P = P[~P.index.duplicated()]
            bc = pd.DataFrame({k: qdm(PR_obs[k].dropna(), P[k]) for k in keys})
            # annual basin precipitation change (bias-adjusted and raw)
            for lab, df in (("bias-adjusted", bc), ("raw", P)):
                a = df["basin"].resample("YS").sum()
                ann.append(dict(source=src, scenario=scen, kind=lab,
                                **{p: float(a[(a.index.year >= v[0]) & (a.index.year <= v[1])].mean()) for p, v in PERIODS.items()}))
            idx = {}
            for k in keys:
                idx[("SPI-12", k)] = std_gamma(bc[k], 12)
            if "mrro" in mc and scen in mc["mrro"]:
                R = mc["mrro"][scen][~mc["mrro"][scen].index.duplicated()]
                for k in keys:
                    idx[("SRI-12", k)] = std_gamma(R[k], 12)
            for g in sel:
                Xf = haar_atrous(bc[str(g)])
                okf = Xf.notna().all(axis=1)
                fm = fitted[g]
                preds = {k: pd.Series(m.predict(Xf[okf]), index=Xf.index[okf]) for k, m in fm["models"].items()}
                preds["ensemble"] = sum(fm["w"][k] * preds[k] for k in fm["w"])
                for k, p in preds.items():
                    qsim = (np.exp(p) - fm["eps"]).clip(lower=0)
                    q12 = qsim.rolling(12).mean()
                    idx[(f"SSI-12 ({k})", str(g))] = std_gamma(q12, 1)
            for (name, k), s in idx.items():
                for per, v in PERIODS.items():
                    rows.append(dict(source=src, scenario=scen, index=name, unit=k, period=per, **drought_metrics(s, v)))
        print("done", src, flush=True)
        pd.DataFrame(rows).to_csv(TAB / "projection_metrics_long.csv", index=False)
        pd.DataFrame(ann).to_csv(TAB / "projection_basin_precip.csv", index=False)
    summarise(names)


def summarise(names=None):
    df = pd.read_csv(TAB / "projection_metrics_long.csv")
    df["unit"] = df["unit"].astype(str)
    base = df[df.period == "1985-2014"].set_index(["source", "scenario", "index", "unit"])
    out = []
    for (scen, ind, unit, per), d in df[df.period != "1985-2014"].groupby(["scenario", "index", "unit", "period"]):
        d = d.set_index(["source", "scenario", "index", "unit"])
        b = base.reindex(d.index)
        for met in ("p_below_1", "p_below_1_5", "p_below_2", "events_per_decade", "mean_duration_months"):
            v, v0 = d[met].values, b[met].values
            ch = v - v0
            out.append(dict(scenario=scen, index=ind, unit=unit, period=per, metric=met, n_models=int(np.isfinite(ch).sum()),
                            baseline_median=float(np.nanmedian(v0)), future_median=float(np.nanmedian(v)),
                            change_median=float(np.nanmedian(ch)), change_p10=float(np.nanpercentile(ch, 10)),
                            change_p90=float(np.nanpercentile(ch, 90)),
                            share_models_increase=float(np.nanmean(ch > 0))))
    s = pd.DataFrame(out)
    s.to_csv(TAB / "projection_summary.csv", index=False)
    a = pd.read_csv(TAB / "projection_basin_precip.csv")
    a["chg_2021_2050_pct"] = 100 * (a["2021-2050"] / a["1985-2014"] - 1)
    a["chg_2051_2080_pct"] = 100 * (a["2051-2080"] / a["1985-2014"] - 1)
    a.groupby(["scenario", "kind"])[["chg_2021_2050_pct", "chg_2051_2080_pct"]].describe(percentiles=[.1, .5, .9]).to_csv(
        TAB / "projection_basin_precip_summary.csv")
    print(s[(s.unit == "basin") & (s.metric == "p_below_1")].to_string())


if __name__ == "__main__":
    if sys.argv[1:] == ["summarise"]:
        summarise()
    else:
        main()
