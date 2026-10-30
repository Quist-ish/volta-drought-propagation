#!/usr/bin/env python3
"""
Sensitivity analyses of the index construction, the opportunity threshold and test power.

  A. Index construction: each of the six candidate families imposed alone; a
     +/-30-day pooling window (double the sample); a common reference period
     (1975-1992) shared by every station with data in it.
  B. Opportunity threshold: 1, 5, 10 and 20 defined upstream days.
  C. Simulation-based power of the permutation test, by injecting propagated
     upstream events into the real catalogue at controlled rates.
  D. Koppen-Geiger zone shares over the basin (Rubel et al., 2017 map, via kgcpy).
  E. Regional rainfall context and an external corroboration test against the
     NOAA/JISAO Sahel precipitation index (GPCC-based), the analogue of the
     original study's Sections S6 and S8.

Writes output_observational/tables/robust_*.csv and robust_numbers.json.
Author: I. Quist (NUIST)
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voltaprop.ssi import SSIConfig, compute_ssi_network, DISTRIBUTIONS
from voltaprop.events import EventConfig, detect_network_events
from voltaprop.propagation import PropagationConfig, run_propagation, load_connectivity
from voltaprop.analysis import permutation_null_test

ORIGINAL_CODE = os.path.join(os.environ.get("ORIGINAL_CODE", "third_party/Code_v.1"), "revision_analyses")
sys.path.insert(0, ORIGINAL_CODE)
import propagation_null_core as O   # the original authors' matcher, verified identical

T = Path("output_observational/tables")
EXT = Path(os.environ.get("EXT_DIR", "data/external"))
W, M = 45, 5
PC = PropagationConfig(W_days=W, min_overlap_days=M, lag_max=0)
OUT = {}
rng = np.random.default_rng(2026)
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

ssi0 = pd.read_csv(T / "SSI_daily.csv", parse_dates=["date"])
ev0 = pd.read_csv(T / "SSI_drought_events.csv", parse_dates=["start_date", "end_date"])
pr0 = pd.read_csv(T / "propagation_pairs.csv", parse_dates=["origin_start", "upstream_start"])
conn = load_connectivity(T / "upstream_connectivity.csv")
qf = ssi0.loc[ssi0.Q.notna(), ["node_id", "date", "Q"]]
cal = ssi0[["node_id", "date"]]


def opportunities(events, ssi_df, min_days=M):
    obs = {n: set(g.loc[g.SSI.notna(), "date"]) for n, g in ssi_df.groupby("node_id")}
    rows = []
    for _, e in events.iterrows():
        ups = conn.get(int(e.node_id), [])
        if not ups:
            continue
        w = set(pd.date_range(e.start_date - pd.Timedelta(days=W), e.end_date, freq="D"))
        observing = [u for u in ups if len(obs.get(u, set()) & w) >= min_days]
        rows.append({"node_id": int(e.node_id), "event_id": int(e.event_id),
                     "start": e.start_date, "end": e.end_date, "observing": observing})
    return pd.DataFrame(rows)


def overlap_share(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Share of events in `a` overlapped by at least one event at the same station in `b`."""
    hit = 0
    for n, g in a.groupby("node_id"):
        h = b[b.node_id == n]
        for _, e in g.iterrows():
            hit += bool(((h.start_date <= e.end_date) & (h.end_date >= e.start_date)).any())
    return hit / len(a) if len(a) else np.nan


# ---------------------------------------------------------------------------
# A. Index construction
# ---------------------------------------------------------------------------
log("A. index-construction variants")
variants = {"default (six families, AIC, +/-15 d, own record)": SSIConfig()}
for fam in DISTRIBUTIONS:
    variants[f"{fam} only"] = SSIConfig(distributions={fam: DISTRIBUTIONS[fam]})
variants["pooling +/-30 d"] = SSIConfig(doy_window_days=30)
variants["common reference 1975-1992"] = SSIConfig(reference_years=(1975, 1992))

base_counts = ev0.groupby("node_id").size()
default_chains = set(zip(pr0.origin_node, pr0.origin_start.dt.normalize(), pr0.upstream_node))
rows, vpairs = [], []
for name, cfg in variants.items():
    s_df, _ = compute_ssi_network(qf, cfg, node_col="node_id", progress=False)
    s_df = cal.merge(s_df[["node_id", "date", "SSI"]], on=["node_id", "date"], how="left")
    e = detect_network_events(s_df, EventConfig(), node_col="node_id", progress=False)
    e["severity_hm3"] = np.nan
    _, prp, chs = run_propagation(e, conn, PC, node_col="node_id", validate=False)
    if len(prp):
        vpairs.append(prp.assign(variant=name))
    op = opportunities(e, s_df)
    n_op = int((op.observing.str.len() > 0).sum())
    sy = s_df.SSI.notna().sum() / 365.25
    cnt = e.groupby("node_id").size().reindex(base_counts.index).fillna(0)
    rho = stats.spearmanr(base_counts, cnt)[0]
    keys = set(zip(prp.origin_node, pd.to_datetime(prp.origin_start).dt.normalize(), prp.upstream_node)) \
        if len(prp) else set()
    nl = permutation_null_test(e, conn, PC, n_permutations=999, seed=42, node_col="node_id", progress=False)
    nl = nl[0] if isinstance(nl, tuple) else nl
    p_cf = float(nl.set_index("statistic").loc["connected_fraction", "p_value_one_sided"])
    rows.append({"variant": name, "events": len(e), "rate": round(len(e) / sy, 3),
                 "defined_station_years": round(sy, 1),
                 "station_count_rho_vs_default": round(rho, 3),
                 "default_events_reproduced": round(overlap_share(ev0, e), 3),
                 "variant_events_in_default": round(overlap_share(e, ev0), 3),
                 "chains": len(chs), "default_chains_reproduced": len(keys & default_chains),
                 "max_chain_size": int(chs.chain_size.max()) if len(chs) else 0,
                 "opportunities": n_op, "cf_opportunities": round(len(chs) / n_op, 3) if n_op else np.nan,
                 "null_p_connected_fraction": round(p_cf, 3)})
    log(f"  {rows[-1]}")
rA = pd.DataFrame(rows)
rA.to_csv(T / "robust_index_variants.csv", index=False)
OUT["index_variants"] = rA.to_dict(orient="records")
VP = pd.concat(vpairs, ignore_index=True)
VP.to_csv(T / "robust_variant_pairs.csv", index=False)
# stability of each default chain: variants (default excluded) holding the same station pair
# with an origin onset within 10 days
VP["origin_start"] = pd.to_datetime(VP.origin_start)
others = [v for v in variants if not v.startswith("default")]
stab = []
for _, r in pr0.iterrows():
    k = 0
    for v in others:
        q = VP[(VP.variant == v) & (VP.origin_node == r.origin_node) & (VP.upstream_node == r.upstream_node)]
        k += bool(((q.origin_start - r.origin_start).abs() <= pd.Timedelta(days=10)).any())
    stab.append({"origin_node": int(r.origin_node), "upstream_node": int(r.upstream_node),
                 "origin_start": r.origin_start.date().isoformat(), "variants_reproducing_of_8": k})
pd.DataFrame(stab).to_csv(T / "robust_chain_stability.csv", index=False)
OUT["chain_stability"] = stab

# ---------------------------------------------------------------------------
# B. Opportunity threshold
# ---------------------------------------------------------------------------
log("B. opportunity threshold")
rowsB = []
for k in (1, 5, 10, 20):
    op = opportunities(ev0, ssi0, min_days=k)
    n_op = int((op.observing.str.len() > 0).sum())
    lo, hi = stats.binomtest(10, n_op).proportion_ci(0.95, method="wilson")
    rowsB.append({"min_defined_days": k, "origin_events": len(op), "opportunities": n_op,
                  "pct_without_observing_upstream": round(100 * (1 - n_op / len(op)), 1),
                  "cf_opportunities": round(10 / n_op, 3), "wilson_lo": round(lo, 3), "wilson_hi": round(hi, 3),
                  "p_below_ebro": float(stats.binomtest(10, n_op, 0.566, alternative="less").pvalue)})
rB = pd.DataFrame(rowsB)
rB.to_csv(T / "robust_opportunity_threshold.csv", index=False)
OUT["opportunity_threshold"] = rB.to_dict(orient="records")
print(rB.to_string(index=False))

# ---------------------------------------------------------------------------
# C. Simulation-based power
# ---------------------------------------------------------------------------
log("C. simulation-based power (injected propagation)")
op0 = opportunities(ev0, ssi0)
connected = set(zip(pr0.origin_node, pr0.origin_event_id))
cands = op0[(op0.observing.str.len() > 0) &
            ~op0.apply(lambda r: (r.node_id, r.event_id) in connected, axis=1)].reset_index(drop=True)
base = O.build_station_arrays(ev0.rename(columns={"node_id": "station_id"}))
for u in {u for ups in conn.values() for u in ups} | set(conn):
    base.setdefault(u, dict(start=np.array([], np.int64), end=np.array([], np.int64),
                            sev=np.array([], float), eid=np.array([], np.int64)))
up_sev = ev0.severity.values
day = lambda t: np.int64(np.datetime64(pd.Timestamp(t).date(), "D").astype(np.int64))
n_sim = int(os.environ.get("POWER_NSIM", 100)); n_perm = int(os.environ.get("POWER_NPERM", 299))


def p_value(arr):
    obs = O.summarise(O.run_matching_vectorized(arr, conn), arr, conn)
    ids = np.array(sorted(k for k, v in arr.items() if len(v["eid"])))
    null = []
    for _ in range(n_perm):
        perm = rng.permutation(ids)
        pa = {k: v for k, v in arr.items() if k not in ids}
        pa.update({ids[i]: arr[perm[i]] for i in range(len(ids))})
        null.append(O.summarise(O.run_matching_vectorized(pa, conn), pa, conn)["connected_fraction"])
    return obs, (1 + np.sum(np.array(null) >= obs["connected_fraction"])) / (1 + n_perm)


rowsC = []
for q in (0.0, 0.2, 0.4, 0.6, 0.8):
    rej, cfs = [], []
    for s in range(n_sim):
        arr = {k: {kk: vv.copy() for kk, vv in v.items()} for k, v in base.items()}
        for _, r in cands.iterrows():
            if rng.random() >= q:
                continue
            ups = list(r.observing); rng.shuffle(ups)
            s0 = day(r.start)
            for u in ups:
                L = int(rng.integers(0, 31)); D = L + M + int(rng.integers(0, 21))
                su, eu = s0 - L, s0 - L + D - 1
                a = arr[u]
                if ((a["start"] <= eu) & (a["end"] >= su)).any():
                    continue
                a["start"] = np.append(a["start"], su); a["end"] = np.append(a["end"], eu)
                a["sev"] = np.append(a["sev"], rng.choice(up_sev))
                a["eid"] = np.append(a["eid"], (a["eid"].max() + 1) if len(a["eid"]) else 1)
                break
        obs, p = p_value(arr)
        n_conn = obs["n_chains"]
        cfs.append(n_conn / len(op0[op0.observing.str.len() > 0])); rej.append(p < 0.05)
    rowsC.append({"injection_rate": q, "mean_cf_opportunities": round(float(np.mean(cfs)), 3),
                  "power": round(float(np.mean(rej)), 3), "n_sim": n_sim, "n_perm": n_perm})
    log(f"  {rowsC[-1]}")
rC = pd.DataFrame(rowsC)
rC.to_csv(T / "robust_power_simulation.csv", index=False)
OUT["power_simulation"] = rC.to_dict(orient="records")

# ---------------------------------------------------------------------------
# D. Koppen-Geiger shares
# ---------------------------------------------------------------------------
log("D. Koppen-Geiger zones")
try:
    import geopandas as gpd, kgcpy
    from shapely.geometry import Point
    basin = gpd.read_file("data/GRDC_VOLTA_POLITICAL_TRANSBOUNDARY_BASIN.shp").to_crs(4326).union_all()
    minx, miny, maxx, maxy = basin.bounds
    g = 0.05
    pts = [(y, x) for y in np.arange(miny + g / 2, maxy, g) for x in np.arange(minx + g / 2, maxx, g)
           if basin.contains(Point(x, y))]
    w = np.array([np.cos(np.radians(y)) for y, _ in pts])
    zones = np.array([kgcpy.lookupCZ(y, x) for y, x in pts])
    tab = (pd.DataFrame({"zone": zones, "w": w}).groupby("zone").w.sum() / w.sum() * 100).round(1)
    OUT["koppen_pct"] = tab.sort_values(ascending=False).to_dict()
    OUT["koppen_n_points"] = len(pts)
    print(tab.sort_values(ascending=False))
except Exception as exc:
    OUT["koppen_error"] = repr(exc); print(exc)

# ---------------------------------------------------------------------------
# E. Sahel precipitation index: context (S6 analogue) and corroboration (S8 analogue)
# ---------------------------------------------------------------------------
log("E. Sahel precipitation index")
raw = (EXT / "sahelrain.data").read_text().splitlines()
recs = []
for line in raw[1:]:
    parts = line.split()
    if len(parts) == 13 and parts[0].isdigit():
        recs.append([int(parts[0])] + [float(v) for v in parts[1:]])
sp = pd.DataFrame(recs, columns=["year"] + list(range(1, 13))).set_index("year")
jjaso = sp[[6, 7, 8, 9, 10]].sum(axis=1)
z = ((jjaso - jjaso.loc[1961:2007].mean()) / jjaso.loc[1961:2007].std()).loc[1961:2007]
dry = set(z[z <= -0.5].index)
OUT["sahel_dry_years_z_le_-0.5"] = sorted(int(y) for y in dry)
OUT["sahel_case_years_z"] = {int(y): round(float(z.loc[y]), 2) for y in (1976, 1983, 1990, 1997)}
OUT["sahel_case_years_rank_driest_of_47"] = {int(y): int((z <= z.loc[y]).sum()) for y in (1976, 1983, 1990, 1997)}
z.rename("sahel_jjaso_z").to_csv(T / "robust_sahel_jjaso_z.csv")

# events: rate in dry vs other years (station-year exposure), Poisson rate ratio
qd = ssi0[ssi0.Q.notna()].assign(year=lambda d: d.date.dt.year)
qd = qd[qd.year.between(1961, 2007)]
sy = qd.groupby("year").SSI.apply(lambda s: s.notna().sum() / 365.25)
evy = ev0.assign(year=ev0.start_date.dt.year).groupby("year").size().reindex(sy.index).fillna(0)
d_mask = sy.index.isin(list(dry))
n1, t1, n0, t0 = evy[d_mask].sum(), sy[d_mask].sum(), evy[~d_mask].sum(), sy[~d_mask].sum()
rr = (n1 / t1) / (n0 / t0)
p_rr = stats.binomtest(int(n1), int(n1 + n0), t1 / (t1 + t0), alternative="greater").pvalue
sev = ev0.assign(year=ev0.start_date.dt.year)
sev["dry"] = sev.year.isin(dry)
u_p = stats.mannwhitneyu(sev.loc[sev.dry, "severity"], sev.loc[~sev.dry, "severity"], alternative="greater").pvalue
# chains overlapping a dry rainy season, against calendar coverage of defined chain-capable time
ch = pd.read_csv(T / "Chains_final.csv", parse_dates=["chain_start", "chain_end"])
ch["dry"] = ch.apply(lambda r: any(y in dry for y in range(r.chain_start.year, r.chain_end.year + 1)), axis=1)
cover = float(t1 / (t1 + t0))
OUT["s8_analogue"] = {
    "dry_years_n": len(dry), "exposure_share_in_dry_years": round(cover, 3),
    "event_rate_dry": round(n1 / t1, 3), "event_rate_other": round(n0 / t0, 3),
    "rate_ratio": round(float(rr), 2), "p_rate_ratio_one_sided": float(p_rr),
    "event_share_in_dry_years": round(float(n1 / (n1 + n0)), 3),
    "severity_dry_vs_other_median": [round(float(sev.loc[sev.dry, 'severity'].median()), 2),
                                     round(float(sev.loc[~sev.dry, 'severity'].median()), 2)],
    "severity_mannwhitney_p_one_sided": float(u_p),
    "chains_in_dry_years": int(ch.dry.sum()), "chains": int(len(ch)),
    "p_chains_binomial_vs_coverage": float(stats.binomtest(int(ch.dry.sum()), len(ch), cover,
                                                           alternative="greater").pvalue)}
print(json.dumps(OUT["s8_analogue"], indent=1))

(T / "robust_numbers.json").write_text(json.dumps(OUT, indent=2, default=str))
log("done")
