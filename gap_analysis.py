#!/usr/bin/env python3
"""Corrected observational statistics, intervals, and the parity table."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
from voltaprop.propagation import load_connectivity
from voltaprop.parity import build_parity_table, format_parity_report

T = Path("output_observational/tables")
r = lambda f, **k: pd.read_csv(T / f, **k)
ev = r("SSI_drought_events.csv", parse_dates=["start_date", "end_date"])
ssi = r("SSI_daily.csv", parse_dates=["date"])
ch, pr = r("Chains_final.csv"), r("propagation_pairs.csv")
nl = r("null_summary.csv")
conn = load_connectivity(T / "upstream_connectivity.csv")
st = pd.read_csv("data/grdc_stations_raw.csv").set_index("grdc_no")
K = {}

# --- catalogue -------------------------------------------------------------
sy = ssi.SSI.notna().sum() / 365.25
K["stations"] = int(ssi.node_id.nunique()); K["stations_with_events"] = int(ev.node_id.nunique())
# SSI_daily is calendar-complete; Q is NaN on days still missing after tier-1 fill
qd = ssi.Q.notna()
K["obs"] = int((ssi.fill_tier == 0).sum()) if "fill_tier" in ssi else int(qd.sum())
K["filled_tier1"] = int((ssi.fill_tier == 1).sum()) if "fill_tier" in ssi else 0
K["defined_pct"] = round(100 * ssi.SSI[qd].notna().mean(), 1)
K["station_years"] = round(sy, 1); K["events"] = int(len(ev))
rate = len(ev) / sy
# exact (Garwood) Poisson interval on the count, scaled by exposure
lo_c = stats.chi2.ppf(0.025, 2 * len(ev)) / 2; hi_c = stats.chi2.ppf(0.975, 2 * (len(ev) + 1)) / 2
K["rate"] = round(rate, 3); K["rate_ci"] = [round(lo_c / sy, 3), round(hi_c / sy, 3)]
K["ebro_rate"] = round(2625 / (33 * 60), 3)
d = ev["duration"]; s = ev["severity"]
K.update(dur_mean=round(d.mean(), 1), dur_median=float(d.median()), dur_max=int(d.max()),
         sev_mean=round(s.mean(), 2), sev_median=round(s.median(), 2),
         int_mean=round(ev.intensity.mean(), 3), vol_mean=round(ev.severity_hm3.mean(), 2))

# --- opportunities ---------------------------------------------------------
obs = {n: set(g.loc[g.SSI.notna(), "date"].dt.normalize()) for n, g in ssi.groupby("node_id")}
rows = []
for _, e in ev.iterrows():
    ups = conn.get(int(e.node_id), [])
    if not ups: continue
    w = set(pd.date_range(e.start_date - pd.Timedelta(days=45), e.end_date, freq="D"))
    rows.append(sum(1 for u in ups if len(obs.get(u, set()) & w) >= 5))
opp = np.array(rows)
K["origin_events"] = int(len(opp)); K["no_upstream_observing"] = int((opp == 0).sum())
K["no_upstream_pct"] = round(100 * (opp == 0).mean(), 1); K["opportunities"] = int((opp > 0).sum())
k, n = len(ch), K["opportunities"]
lo, hi = stats.binomtest(k, n).proportion_ci(0.95, method="wilson")
K["cf_opp"] = round(k / n, 3); K["cf_opp_ci"] = [round(lo, 3), round(hi, 3)]
K["p_vs_ebro"] = float(stats.binomtest(k, n, 0.566, alternative="less").pvalue)
# power of the permutation test, from its own null distribution (normal
# approximation with the null SD), for the Ebro excess transported two ways
nlc = nl.set_index("statistic").loc["connected_fraction"]
mu0, sd0, crit = float(nlc.null_mean), float(nlc.null_sd), float(nlc.null_p95)
K["power_abs"] = round(float(1 - stats.norm.cdf((crit - (mu0 + 0.063)) / sd0)), 2)
K["power_rel"] = round(float(1 - stats.norm.cdf((crit - mu0 * 0.566 / 0.503) / sd0)), 2)
K["chain_sizes"] = sorted(ch.chain_size.unique().tolist())
nl2 = nl.set_index("statistic")
K["null_p"] = {i: round(float(nl2.loc[i, "p_value_one_sided"]), 3) for i in nl2.index}

# --- S3 proxy ----------------------------------------------------------------
def hav(a, b):
    la1, lo1, la2, lo2 = map(np.radians, [st.lat[a], st.lon[a], st.lat[b], st.lon[b]])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(h))
dist = [hav(o, u) for o, u in zip(pr.origin_node, pr.upstream_node)]
dA = [st.area_km2[o] - st.area_km2[u] for o, u in zip(pr.origin_node, pr.upstream_node)]
K["s3_dist"] = [round(x, 3) for x in stats.spearmanr(dist, pr.lag_days)]
K["s3_area"] = [round(x, 3) for x in stats.spearmanr(dA, pr.lag_days)]
K["s3_origins"] = pr.origin_node.value_counts().to_dict()

# --- S9 --------------------------------------------------------------------
s9 = r("tableS9_homogeneity.csv")
K["s9_flagged"] = int(s9["flag"].astype(bool).sum()); K["s9_n"] = int(len(s9))

json.dump(K, open(T / "gap_key_numbers.json", "w"), indent=2, default=str)
for kk, vv in K.items(): print(f"  {kk:<24} {vv}")

# --- parity ----------------------------------------------------------------
supp = {"rsi": r("sensitivity_rsi.csv"), "knee_M": None, "s5": r("tableS5_detection_sensitivity.csv"),
        "s1": r("tableS1_m_sensitivity.csv"), "s14": r("tableS1_4_gap_tolerance.csv"), "s9": s9,
        "origin_contrast": r("origin_contrast.csv"), "upstream_contrast": r("upstream_contrast.csv")}
tab = build_parity_table(events=ev, chains=ch, pairs=pr,
                         null=nl.rename(columns={"p_value_one_sided": "p_value"}),
                         nodes=pd.DataFrame({"node_id": sorted(ev.node_id.unique())}),
                         n_origin_nodes=sum(1 for v in conn.values() if v),
                         period="1961-2007", period_years=sy / ev.node_id.nunique(), supp=supp)
tab.to_csv(T / "parity_vs_ebro.csv", index=False)
(T / "parity_vs_ebro.txt").write_text(format_parity_report(tab))
print(f"\nparity: {(tab.status=='computed').sum()} of {len(tab)} computed")
