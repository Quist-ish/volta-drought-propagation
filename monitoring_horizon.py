#!/usr/bin/env python3
"""
Monitoring horizon (objective iv) and projection uncertainty partition.

Years of concurrent observation needed for 50% and 80% power of the availability-aware test
(N50 = 366, N80 = 1,737 opportunities; output_identifiability) under three designs:
  * the observed record pattern (40 opportunities in 46.1 years, Feb 1961 - Feb 2007);
  * the 18 current gauges with continuous records (observing-system experiment, continuous
    case: 465 opportunities, 1961-2007);
  * reanalysis node density, continuous (7,254 opportunities in the 129-node network, 1961-2020).
Rates are scaled by the projected ratio of drought-month shares (SSI-12 transfer ensemble,
mean of the selected gauges; future / 1985-2014) for each CMIP6 model.

Also partitions the spread of projected change in drought-month share between CMIP6 models
and transfer-model members (ridge, random forest, XGBoost) by a two-way decomposition.
Writes output_projection/tables/monitoring_horizon.json and projection_numbers.json.
Author: I. Quist (NUIST)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TP = ROOT / "output_projection/tables"
idn = json.load(open(ROOT / "output_identifiability/identifiability_numbers.json"))
N50, N80 = idn["N50"]["p_cf_opp"], idn["N80"]["p_cf_opp"]
osse = pd.read_csv(ROOT / "output_reanalysis/tables/osse_summary.csv").set_index("scenario")
net = pd.read_csv(ROOT / "output_identifiability/tables/networks_summary.csv").set_index("network")


def main():
    lg = pd.read_csv(TP / "projection_metrics_long.csv")
    lg = lg[~lg["index"].str.startswith("SPEI")]
    lg.to_csv(TP / "projection_metrics_long.csv", index=False)
    sk = pd.read_csv(TP / "transfer_model_skill.csv")
    sel = [str(g) for g in sk[sk.selected].grdc_no]
    ens = lg[(lg["index"] == "SSI-12 (ensemble)") & (lg.unit.astype(str).isin(sel))]
    piv = ens.groupby(["source", "scenario", "period"]).p_below_1.mean().unstack("period")
    rates = {
        "Observed record pattern, 18 gauges": 40 / 46.1,
        "18 gauges, continuous records": float(osse.loc["continuous", "n_opportunities"]) / 47.0,
        "Reanalysis-node density (129), continuous": float(0.983 * 7379) / 60.0,
    }
    out = {"N50": N50, "N80": N80, "designs": {}}
    for lab, r in rates.items():
        d = {"opportunities_per_year": r}
        for key, n in (("years_n50", N50), ("years_n80", N80)):
            v = {"historical": n / r}
            for sc in ("ssp245", "ssp585"):
                for per in ("2021-2050", "2051-2080"):
                    ratio = (piv.loc[(slice(None), sc), per] / piv.loc[(slice(None), sc), "1985-2014"]).values
                    ratio = np.clip(ratio, 1e-3, None)
                    yrs = n / (r * ratio)
                    tag = f"{sc}_{per.replace('-', '_')}"
                    v[f"{tag}_median"] = float(np.median(yrs))
                    v[f"{tag}_p10"] = float(np.percentile(yrs, 10))
                    v[f"{tag}_p90"] = float(np.percentile(yrs, 90))
                    v[f"{tag}_ratio_median"] = float(np.median(ratio))
            d[key] = v
        out["designs"][lab] = d
    json.dump(out, open(TP / "monitoring_horizon.json", "w"), indent=1)

    # uncertainty partition: change in drought-month share, SSI-12 members x GCM
    mem = lg[lg["index"].isin(["SSI-12 (ridge)", "SSI-12 (rf)", "SSI-12 (xgb)"]) & lg.unit.astype(str).isin(sel)]
    base = mem[mem.period == "1985-2014"].set_index(["source", "scenario", "index", "unit"]).p_below_1
    parts = {}
    for sc in ("ssp245", "ssp585"):
        for per in ("2021-2050", "2051-2080"):
            f = mem[(mem.period == per) & (mem.scenario == sc)].set_index(["source", "scenario", "index", "unit"]).p_below_1
            ch = (f - base.reindex(f.index)).reset_index()
            ch = ch.groupby(["source", "index"]).p_below_1.mean().unstack("index")
            gcm_var = float(ch.mean(axis=1).var(ddof=1))
            mem_var = float(ch.mean(axis=0).var(ddof=1))
            res_var = float((ch.sub(ch.mean(axis=1), axis=0).sub(ch.mean(axis=0), axis=1) + ch.values.mean()).var(ddof=1).mean())
            tot = gcm_var + mem_var + res_var
            parts[f"{sc} {per}"] = {"gcm_share": gcm_var / tot, "transfer_member_share": mem_var / tot, "interaction_share": res_var / tot}
    # headline numbers
    s = pd.read_csv(TP / "projection_summary.csv")
    s = s[~s["index"].str.startswith("SPEI")]
    s.to_csv(TP / "projection_summary.csv", index=False)
    b = pd.read_csv(TP / "projection_basin_precip.csv"); b = b[b.kind == "bias-adjusted"]
    prc = {}
    for sc in ("ssp245", "ssp585"):
        d = b[b.scenario == sc]
        for per in ("2021-2050", "2051-2080"):
            v = 100 * (d[per] / d["1985-2014"] - 1)
            prc[f"{sc} {per}"] = {"median": float(v.median()), "p10": float(v.quantile(.1)), "p90": float(v.quantile(.9)),
                                  "n_wetter": int((v > 0).sum()), "n": int(len(v))}
    json.dump({"basin_precip_change_pct": prc, "uncertainty_partition_ssi12": parts,
               "selected_gauges": sel}, open(TP / "projection_numbers.json", "w"), indent=1)
    print(json.dumps(out, indent=1)[:2500]); print(json.dumps(parts, indent=1)); print(json.dumps(prc, indent=1))


if __name__ == "__main__":
    main()
