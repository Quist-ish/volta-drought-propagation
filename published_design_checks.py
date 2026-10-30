#!/usr/bin/env python3
"""
Checks against the published design of Crespillo López et al. (2026). Each block reproduces a quantity the published paper reports, or
tests a claim the manuscript makes, on the observed GRDC network.

Reads output_observational/tables (run_observational.py must have run first).
Writes review_*.csv and review_numbers.json to the same folder.

Author: I. Quist (NUIST)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voltaprop.ssi import SSIConfig, compute_ssi_network
from voltaprop.events import EventConfig, detect_network_events
from voltaprop.propagation import PropagationConfig, run_propagation, load_connectivity
from voltaprop.grdc import build_connectivity

T = Path("output_observational/tables")
R = {}
rng = np.random.default_rng(42)
rd = lambda x, k=2: None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), k)

ssi = pd.read_csv(T / "SSI_daily.csv", parse_dates=["date"])
ev = pd.read_csv(T / "SSI_drought_events.csv", parse_dates=["start_date", "end_date", "peak_date"])
ch = pd.read_csv(T / "Chains_final.csv", parse_dates=["origin_start", "origin_end", "chain_start", "chain_end"])
pr = pd.read_csv(T / "propagation_pairs.csv", parse_dates=["origin_start", "origin_end", "upstream_start", "upstream_end"])
conn = load_connectivity(T / "upstream_connectivity.csv")
st = pd.read_csv("data/grdc_stations_raw.csv")
sti = st.set_index("grdc_no")
W, M = 45, 5


def describe(x: pd.Series) -> dict:
    x = pd.Series(x).astype(float)
    return {"n": int(x.size), "mean": rd(x.mean(), 3), "median": rd(x.median(), 3),
            "q25": rd(x.quantile(.25), 3), "q75": rd(x.quantile(.75), 3),
            "iqr": rd(x.quantile(.75) - x.quantile(.25), 3), "min": rd(x.min(), 3), "max": rd(x.max(), 3)}


def observing_sets(ssi_df):
    return {n: set(g.loc[g.SSI.notna(), "date"]) for n, g in ssi_df.groupby("node_id")}


def opportunities(events, conn_, ssi_df):
    """Per origin event: upstream stations with >= M defined index days in
    [s0 - W, e0], the span over which a qualifying upstream event could lie."""
    obs = observing_sets(ssi_df)
    rows = []
    for _, e in events.iterrows():
        ups = conn_.get(int(e.node_id), [])
        if not ups:
            continue
        w = set(pd.date_range(e.start_date - pd.Timedelta(days=W), e.end_date, freq="D"))
        observing = [u for u in ups if len(obs.get(u, set()) & w) >= M]
        rows.append({"node_id": int(e.node_id), "event_id": int(e.event_id),
                     "n_upstream": len(ups), "n_observing": len(observing),
                     "observing": ",".join(map(str, observing))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Fig. 1b analogue: distribution of mean discharge across stations
# ---------------------------------------------------------------------------
obsq = ssi[ssi.fill_tier == 0]
mq = obsq.groupby("node_id").Q.mean()
R["fig1b_mean_discharge"] = {**describe(mq),
                             "min_station": sti.loc[mq.idxmin(), "station_name"],
                             "min_river": sti.loc[mq.idxmin(), "river"],
                             "max_station": sti.loc[mq.idxmax(), "station_name"],
                             "max_river": sti.loc[mq.idxmax(), "river"]}

# ---------------------------------------------------------------------------
# 2. Fig. 5 analogue: event descriptors in full
# ---------------------------------------------------------------------------
R["fig5_events"] = {k: describe(ev[c]) for k, c in
                    [("duration", "duration"), ("severity", "severity"),
                     ("intensity", "intensity"), ("volumetric_hm3", "severity_hm3")]}
per_st = ev.groupby("node_id").size()
R["events_per_station"] = {**describe(per_st),
                           "stations_without_events": sorted(set(ssi.node_id) - set(ev.node_id))}
qd = ssi[ssi.Q.notna()]
dshare = qd.groupby("node_id").SSI.apply(lambda s: s.notna().mean())
lat = sti.loc[dshare.index, "lat"]
rho, p = stats.spearmanr(lat, dshare)
R["defined_share_by_station"] = {"min": rd(dshare.min(), 3), "max": rd(dshare.max(), 3),
                                 "spearman_vs_latitude": [rd(rho, 3), rd(p, 4)]}
pd.DataFrame({"node_id": dshare.index, "station": sti.loc[dshare.index, "station_name"].values,
              "lat": lat.values, "defined_share": dshare.round(3).values,
              "n_events": per_st.reindex(dshare.index).fillna(0).astype(int).values}
             ).to_csv(T / "review_station_validity.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Event-rate uncertainty: exact Poisson and station-year block bootstrap
# ---------------------------------------------------------------------------
qd = qd.assign(year=qd.date.dt.year)
sy_tab = qd.groupby(["node_id", "year"]).SSI.apply(lambda s: s.notna().sum() / 365.25)
evy = ev.assign(year=ev.start_date.dt.year).groupby(["node_id", "year"]).size()
blocks = pd.DataFrame({"sy": sy_tab, "n": evy.reindex(sy_tab.index).fillna(0)})
blocks = blocks[blocks.sy > 0]
rate = blocks.n.sum() / blocks.sy.sum()
boots = []
idx = np.arange(len(blocks))
for _ in range(10000):
    b = blocks.iloc[rng.choice(idx, len(idx))]
    boots.append(b.n.sum() / b.sy.sum())
lo_c = stats.chi2.ppf(0.025, 2 * len(ev)) / 2
hi_c = stats.chi2.ppf(0.975, 2 * (len(ev) + 1)) / 2
R["event_rate"] = {"events": int(len(ev)), "station_years": rd(blocks.sy.sum(), 1),
                   "rate": rd(rate, 3), "exact_poisson_ci": [rd(lo_c / blocks.sy.sum(), 3),
                                                            rd(hi_c / blocks.sy.sum(), 3)],
                   "block_bootstrap_ci": [rd(np.percentile(boots, 2.5), 3),
                                          rd(np.percentile(boots, 97.5), 3)],
                   "ebro": 1.326,
                   "dispersion_index": rd(blocks.n.var() / blocks.n.mean(), 2)}

# ---------------------------------------------------------------------------
# 4. Fig. 6 analogue: chain descriptors in full, and propagation fraction
#    against the stations actually observing
# ---------------------------------------------------------------------------
R["fig6_chains"] = {k: describe(ch[c]) for k, c in
                    [("duration", "chain_duration"), ("size", "chain_size"),
                     ("propagation_fraction", "propagation_fraction"), ("lag", "lag_mean"),
                     ("severity", "chain_severity"), ("volumetric_hm3", "chain_severity_hm3")]}
R["chains_origin_stations"] = sorted(ch.origin_node.unique().tolist())
opp = opportunities(ev, conn, ssi)
opp.to_csv(T / "review_opportunities.csv", index=False)
cm = ch.merge(opp, left_on=["origin_node", ch.chain_id.str.split("_").str[1].astype(int)],
              right_on=["node_id", "event_id"], how="left")
cm["pf_observing"] = 1 / cm["n_observing"]
R["pf_against_observing"] = {"mean_catalogue_denominator": rd(ch.propagation_fraction.mean(), 3),
                             "mean_observing_denominator": rd(cm.pf_observing.mean(), 3),
                             "chains_with_single_observing_upstream": int((cm.n_observing == 1).sum()),
                             "note": "with every chain of size two, pf = 1/N_avail is fixed by topology"}
pd.DataFrame(cm[["chain_id", "propagation_fraction", "n_upstream", "n_observing", "pf_observing"]]
             ).to_csv(T / "review_pf_denominators.csv", index=False)

# ---------------------------------------------------------------------------
# 5. S7 on the fair denominator, with tests
# ---------------------------------------------------------------------------
connected = set(zip(pr.origin_node, pr.origin_event_id))
o = ev.merge(opp, on=["node_id", "event_id"])
o["connected"] = [(n, e) in connected for n, e in zip(o.node_id, o.event_id)]
oo = o[o.n_observing > 0]


def contrast(a, b):
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided") if len(a) and len(b) else (np.nan, np.nan)
    bs = []
    for _ in range(5000):
        bs.append(rng.choice(a, len(a)).mean() / rng.choice(b, len(b)).mean())
    return {"n_a": int(len(a)), "n_b": int(len(b)), "mean_a": rd(np.mean(a), 2), "mean_b": rd(np.mean(b), 2),
            "ratio": rd(np.mean(a) / np.mean(b), 2),
            "ratio_ci95": [rd(np.percentile(bs, 2.5), 2), rd(np.percentile(bs, 97.5), 2)],
            "mannwhitney_p": rd(p, 3)}


R["s7_isolated_vs_connected_all_origins"] = {
    "severity": contrast(o.loc[o.connected, "severity"].values, o.loc[~o.connected, "severity"].values),
    "duration": contrast(o.loc[o.connected, "duration"].values, o.loc[~o.connected, "duration"].values)}
R["s7_isolated_vs_connected_opportunities"] = {
    "severity": contrast(oo.loc[oo.connected, "severity"].values, oo.loc[~oo.connected, "severity"].values),
    "duration": contrast(oo.loc[oo.connected, "duration"].values, oo.loc[~oo.connected, "duration"].values),
    "unconnected": int((~oo.connected).sum()), "opportunities": int(len(oo))}
by_st = oo.groupby("node_id").connected.agg(["size", "sum"])
by_st["unconnected_pct"] = (100 * (1 - by_st["sum"] / by_st["size"])).round(1)
by_st.to_csv(T / "review_unconnected_by_station.csv")
R["s7_unconnected_by_station_pct"] = {str(k): float(v) for k, v in by_st.unconnected_pct.items()}

# published definition of the upstream population: events that intersect the
# search window [s0 - W, s0] of at least one origin event downstream
cand = set()
for _, e in ev.iterrows():
    for u in conn.get(int(e.node_id), []):
        ue = ev[ev.node_id == u]
        hit = ue[(ue.start_date <= e.start_date) & (ue.end_date >= e.start_date - pd.Timedelta(days=W))]
        cand.update(zip(hit.node_id, hit.event_id))
acc = set(zip(pr.upstream_node, pr.upstream_event_id))
up = ev[[(n, e) in cand for n, e in zip(ev.node_id, ev.event_id)]].copy()
up["accepted"] = [(n, e) in acc for n, e in zip(up.node_id, up.event_id)]
R["s7_upstream_candidates_published_definition"] = {
    "candidates": int(len(up)), "never_accepted": int((~up.accepted).sum()),
    "fraction_never_accepted": rd((~up.accepted).mean(), 3),
    "severity": contrast(up.loc[up.accepted, "severity"].values, up.loc[~up.accepted, "severity"].values),
    "duration": contrast(up.loc[up.accepted, "duration"].values, up.loc[~up.accepted, "duration"].values)}

# ---------------------------------------------------------------------------
# 6. S3: lag against two proxies, and against the mechanical bound
# ---------------------------------------------------------------------------
def hav(a, b):
    la1, lo1, la2, lo2 = map(np.radians, [sti.lat[a], sti.lon[a], sti.lat[b], sti.lon[b]])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(h))


p3 = pr.copy()
p3["dist_km"] = [hav(a, b) for a, b in zip(p3.origin_node, p3.upstream_node)]
p3["dA_km2"] = [sti.area_km2[a] - sti.area_km2[b] for a, b in zip(p3.origin_node, p3.upstream_node)]
p3["up_area_km2"] = [sti.area_km2[b] for b in p3.upstream_node]


def sp(x, y):
    r, p = stats.spearmanr(x, y)
    return [rd(r, 3), rd(p, 4)]


def partial_spearman(x, y, z):
    rx, ry, rz = (pd.Series(v).rank().values for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    r = np.corrcoef(ex, ey)[0, 1]
    n = len(x)
    t = r * np.sqrt((n - 3) / (1 - r ** 2))
    return [rd(r, 3), rd(2 * stats.t.sf(abs(t), n - 3), 4)]


nw = p3[p3.origin_node == 1531450]
R["s3"] = {"n_pairs": int(len(p3)),
           "lag_vs_distance": sp(p3.dist_km, p3.lag_days),
           "lag_vs_intervening_area": sp(p3.dA_km2, p3.lag_days),
           "bonferroni_alpha_two_proxies": 0.025,
           "lag_vs_upstream_duration": sp(p3.upstream_duration, p3.lag_days),
           "intervening_area_vs_upstream_duration": sp(p3.dA_km2, p3.upstream_duration),
           "lag_vs_area_partial_on_upstream_duration": partial_spearman(p3.dA_km2, p3.lag_days,
                                                                         p3.upstream_duration),
           "nawuni_only": {"n": int(len(nw)), "lag_vs_intervening_area": sp(nw.dA_km2, nw.lag_days)},
           "max_negative_lag_bound": "lag >= -(upstream_duration - M) by the overlap rule"}
p3.to_csv(T / "review_s3_pairs.csv", index=False)

# ---------------------------------------------------------------------------
# 7. The W identity: is RSI(W) = 0 forced by the algorithm?
# ---------------------------------------------------------------------------
grid = pd.read_csv(T / "sensitivity_grid.csv")
R["w_identity"] = {"distinct_n_pairs_across_W_within_any_M": int(grid.groupby("M").n_pairs.nunique().max()),
                   "argument": ("a candidate intersects [s0-W, s0], so s_u <= s0; overlap >= M >= 1 then "
                                "forces e_u >= s0, so the candidate contains s0 and lies in the window for "
                                "every W >= 0; the candidate set, and every metric, is independent of W")}

# ---------------------------------------------------------------------------
# 8. Seasonal-validity criterion: switch it off, vary it, restrict to season
# ---------------------------------------------------------------------------
qf = ssi.loc[ssi.Q.notna(), ["node_id", "date", "Q"]]
cal = ssi[["node_id", "date"]]
variants = {
    "default (40%, 0.10 m3/s)": SSIConfig(),
    "mask off": SSIConfig(seasonal_validity=False),
    "zero fraction 30%": SSIConfig(max_zero_fraction=0.30),
    "zero fraction 50%": SSIConfig(max_zero_fraction=0.50),
    "median floor 0.05": SSIConfig(min_pooled_median_q_m3s=0.05),
    "median floor 0.20": SSIConfig(min_pooled_median_q_m3s=0.20),
    "Apr-Oct only": SSIConfig(restrict_to_season=True),
}
rows = []
for name, cfg in variants.items():
    s_df, _ = compute_ssi_network(qf, cfg, node_col="node_id", progress=False)
    s_df = cal.merge(s_df[["node_id", "date", "SSI"]], on=["node_id", "date"], how="left")
    e = detect_network_events(s_df, EventConfig(), node_col="node_id", progress=False)
    e["severity_hm3"] = np.nan
    dry = e.start_date.dt.month.isin([11, 12, 1, 2, 3, 4])
    defined = s_df.SSI.notna().sum()
    _, prp, chs = run_propagation(e, conn, PropagationConfig(W_days=W, min_overlap_days=M, lag_max=0),
                                  node_col="node_id", validate=False)
    op = opportunities(e, conn, s_df)
    n_op = int((op.n_observing > 0).sum())
    rows.append({"variant": name,
                 "defined_pct": round(100 * defined / s_df.merge(qf, on=["node_id", "date"]).shape[0], 1),
                 "events": len(e), "rate": round(len(e) / (defined / 365.25), 3),
                 "events_onset_nov_apr": int(dry.sum()), "pct_onset_nov_apr": round(100 * dry.mean(), 1),
                 "mean_duration": round(e.duration.mean(), 1), "max_duration": int(e.duration.max()),
                 "origin_events": len(op), "opportunities": n_op, "chains": len(chs),
                 "cf_opportunities": round(len(chs) / n_op, 3) if n_op else np.nan,
                 "mean_pf": round(chs.propagation_fraction.mean(), 3) if len(chs) else np.nan,
                 "max_chain_size": int(chs.chain_size.max()) if len(chs) else 0})
    print(rows[-1], flush=True)
sv = pd.DataFrame(rows)
sv.to_csv(T / "review_seasonal_validity.csv", index=False)
R["seasonal_validity"] = sv.to_dict(orient="records")

# ---------------------------------------------------------------------------
# 9. Lawra-Dapola ordering: GRDC pointer against catchment area
# ---------------------------------------------------------------------------
have = sorted(ssi.node_id.unique())
st2 = st.copy()
st2.loc[st2.grdc_no == 1931400, "next_downstream"] = 1531050   # Dapola -> Lawra
st2.loc[st2.grdc_no == 1531050, "next_downstream"] = 1931475   # Lawra -> Noumbiel
conn2, _ = build_connectivity(st2, restrict_to=have)
_, prp2, ch2 = run_propagation(ev, conn2, PropagationConfig(W_days=W, min_overlap_days=M, lag_max=0),
                               node_col="node_id", validate=False)
op2 = opportunities(ev, conn2, ssi)
R["lawra_dapola_flip"] = {"km_apart": rd(hav(1531050, 1931400), 1),
                          "origin_stations": sum(1 for v in conn2.values() if v),
                          "origin_events": int(len(op2)),
                          "no_upstream_observing_pct": rd(100 * (op2.n_observing == 0).mean(), 1),
                          "opportunities": int((op2.n_observing > 0).sum()), "chains": int(len(ch2)),
                          "cf_opportunities": rd(len(ch2) / max(1, (op2.n_observing > 0).sum()), 3)}

# ---------------------------------------------------------------------------
# 10. Case studies: the extremes of the ten chains, with imputation share
# ---------------------------------------------------------------------------
cases = []
for _, c in ch.iterrows():
    on = int(c.origin_node)
    win = ssi[(ssi.node_id == on) & (ssi.date >= c.chain_start) & (ssi.date <= c.chain_end)]
    p = pr[(pr.origin_node == on) & (pr.origin_start == c.origin_start)].iloc[0]
    up_win = ssi[(ssi.node_id == p.upstream_node) & (ssi.date >= p.upstream_start) & (ssi.date <= p.upstream_end)]
    ev_o = ev[(ev.node_id == on) & (ev.start_date == c.origin_start)].iloc[0]
    ev_u = ev[(ev.node_id == p.upstream_node) & (ev.event_id == p.upstream_event_id)].iloc[0]
    cases.append({"chain_id": c.chain_id, "origin": sti.station_name[on], "origin_river": sti.river[on],
                  "upstream": sti.station_name[p.upstream_node], "upstream_river": sti.river[p.upstream_node],
                  "chain_start": c.chain_start.date(), "chain_end": c.chain_end.date(),
                  "chain_duration": int(c.chain_duration), "lag": int(p.lag_days),
                  "overlap": int(p.overlap_days), "chain_severity": rd(c.chain_severity, 2),
                  "chain_vol_hm3": rd(c.chain_severity_hm3, 1),
                  "origin_duration": int(ev_o.duration), "origin_peak_SSI": rd(ev_o.peak_SSI, 2),
                  "upstream_duration": int(ev_u.duration), "upstream_peak_SSI": rd(ev_u.peak_SSI, 2),
                  "pf": rd(c.propagation_fraction, 2),
                  "origin_days_missing_pct": rd(100 * win.Q.isna().mean(), 1),
                  "origin_days_tier1_pct": rd(100 * (win.fill_tier == 1).mean(), 1),
                  "upstream_days_tier1_pct": rd(100 * (up_win.fill_tier == 1).mean(), 1)})
cs = pd.DataFrame(cases)
cs.to_csv(T / "review_case_candidates.csv", index=False)
R["case_extremes"] = {"longest": cs.loc[cs.chain_duration.idxmax(), "chain_id"],
                      "most_severe": cs.loc[cs.chain_severity.idxmax(), "chain_id"],
                      "largest_volume": cs.loc[cs.chain_vol_hm3.idxmax(), "chain_id"],
                      "fastest_shortest": cs.loc[cs.chain_duration.idxmin(), "chain_id"]}

(T / "review_numbers.json").write_text(json.dumps(R, indent=2, default=str))
print(json.dumps(R, indent=1, default=str))
