#!/usr/bin/env python3
"""
Common reference period, equivalence with the Ebro, synthetic truth and channel distance.

  A. Common reference period (1975-1992) as a co-primary index: coverage, cf_opp, p-hat
     and the availability-aware permutation null (1,999 permutations).
  B. Equivalence of Volta and Ebro per-link connectivity: Mantel-Haenszel odds ratio
     over the shared k strata with the Robins-Breslow-Greenland variance; two one-sided
     tests (TOST) at alpha = 0.05 for margins of 2 and 3 (odds ratio within 1/m to m).
  C. Synthetic-truth observing experiment: with the per-link probability fixed at
     p = 0.10, 0.16 and 0.25, outcomes are simulated on the 104 observed origin events
     under (i) continuous records (k = mapped upstream stations) and (ii) the observed
     availability (k = observing stations); cf_all, cf_opp and the bias and 95% coverage of
     the maximum-likelihood p-hat are summarised over 20,000 replicates.
  D. Lag against mapped along-channel distance (TDX-Hydro) for the ten matched pairs.

Writes output_observational/tables/r5_*.csv|json
Author: I. Quist (NUIST)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from voltaprop import fastmatch as F  # noqa: E402
from voltaprop.events import EventConfig, detect_network_events  # noqa: E402
from voltaprop.propagation import load_connectivity  # noqa: E402
from voltaprop.ssi import SSIConfig, compute_ssi_network  # noqa: E402

T = ROOT / "output_observational/tables"
TI = ROOT / "output_identifiability/tables"
OUT = {}
rng = np.random.default_rng(2026)


def part_a():
    rng = np.random.default_rng(2026)      # each part has its own generator, so results do not depend on run order
    ssi0 = pd.read_csv(T / "SSI_daily.csv", parse_dates=["date"])
    conn = load_connectivity(T / "upstream_connectivity.csv")
    qf = ssi0.loc[ssi0.Q.notna(), ["node_id", "date", "Q"]]
    cal = ssi0[["node_id", "date"]]
    res = {}
    for name, cfg in [("own record (default)", SSIConfig()), ("common reference 1975-1992", SSIConfig(reference_years=(1975, 1992)))]:
        if name.startswith("own"):
            s_df = ssi0[["node_id", "date", "SSI"]]
            e = pd.read_csv(T / "SSI_drought_events.csv", parse_dates=["start_date", "end_date"])
        else:
            s_df, _ = compute_ssi_network(qf, cfg, node_col="node_id", progress=False)
            s_df = cal.merge(s_df[["node_id", "date", "SSI"]], on=["node_id", "date"], how="left")
            e = detect_network_events(s_df, EventConfig(), node_col="node_id", progress=False)
        nodes = set(conn) | {u for v in conn.values() for u in v}
        avail = F.availability_from_daily(s_df, col="SSI")
        recs = F.build_records(e, avail=avail, nodes=nodes)
        obs, summ, null = F.null_test(recs, conn, n_perm=1999, seed=42)
        s = summ.set_index("statistic")
        # year-block bootstrap of p-hat
        pairs, org = F.match(recs, conn, detail=True)
        org = org[org.n_observing >= 1].copy()
        org["year"] = pd.to_datetime(org.s0_date).dt.year
        years = org.year.unique()
        boots = []
        for _ in range(2000):
            yy = rng.choice(years, len(years), replace=True)
            d = pd.concat([org[org.year == y] for y in yy])
            boots.append(F.link_probability_mle(d.n_observing.values, d.connected.values))
        res[name] = dict(events=int(len(e)), origin_events=int(obs["n_origin_events"]),
                         opportunities=int(obs["n_opportunities"]), connected=int(obs["n_connected"]),
                         coverage=float(obs["coverage"]), cf_all=float(obs["cf_all"]), cf_opp=float(obs["cf_opp"]),
                         p_link=float(obs["p_link"]), p_link_ci=[float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))],
                         null_cf_opp=float(s.loc["cf_opp", "null_mean"]), p_cf_opp=float(s.loc["cf_opp", "p"]),
                         null_p_link=float(s.loc["p_link", "null_mean"]), p_p_link=float(s.loc["p_link", "p"]),
                         p_cf_all=float(s.loc["cf_all", "p"]))
        print(name, res[name], flush=True)
    OUT["A_common_reference"] = res


def mh_rgb(a, b, c, d):
    """Mantel-Haenszel OR with Robins-Breslow-Greenland variance of ln OR."""
    a, b, c, d = map(lambda v: np.asarray(v, float), (a, b, c, d))
    n = a + b + c + d
    R = a * d / n; S = b * c / n
    P = (a + d) / n; Q = (b + c) / n
    orr = R.sum() / S.sum()
    var = (P * R).sum() / (2 * R.sum() ** 2) + ((P * S + Q * R).sum()) / (2 * R.sum() * S.sum()) + (Q * S).sum() / (2 * S.sum() ** 2)
    return orr, np.sqrt(var)


def part_b():
    st = pd.read_csv(TI / "stratified_by_k.csv")
    st = st[(st.volta_n > 0) & (st.ebro_n > 0)]
    a = st.volta_connected; b = st.volta_n - st.volta_connected
    c = st.ebro_connected; d = st.ebro_n - st.ebro_connected
    orr, se = mh_rgb(a, b, c, d)
    lo95, hi95 = np.exp(np.log(orr) - 1.96 * se), np.exp(np.log(orr) + 1.96 * se)
    lo90, hi90 = np.exp(np.log(orr) - 1.645 * se), np.exp(np.log(orr) + 1.645 * se)
    out = dict(strata=st.k.tolist(), volta_opportunities=int(st.volta_n.sum()), OR_MH=float(orr), se_lnOR=float(se),
               ci95=[float(lo95), float(hi95)], ci90=[float(lo90), float(hi90)])
    for m in (1.5, 2.0, 3.0):
        z1 = (np.log(orr) - np.log(1 / m)) / se
        z2 = (np.log(m) - np.log(orr)) / se
        p_tost = float(max(stats.norm.sf(z1), stats.norm.sf(z2)))
        out[f"tost_margin_{m}"] = dict(p=p_tost, equivalent=bool(p_tost < 0.05))
    # smallest symmetric margin for which equivalence holds at alpha = 0.05
    out["smallest_margin_equivalent"] = float(max(1 / lo90, hi90))
    OUT["B_equivalence"] = out
    print(out, flush=True)


def part_c():
    rng = np.random.default_rng(2026)
    opp = pd.read_csv(T / "review_opportunities.csv")
    k_cont = opp.n_upstream.values.astype(int)
    k_obs = opp.n_observing.values.astype(int)
    rows = []
    R = 20000
    for p in (0.10, 0.16, 0.25):
        cf_c, cf_m, cfo_m, ph, cover = [], [], [], [], []
        for _ in range(R):
            con_c = rng.random(len(k_cont)) < 1 - (1 - p) ** k_cont
            con_m = rng.random(len(k_obs)) < 1 - (1 - p) ** k_obs
            cf_c.append(con_c.mean()); cf_m.append(con_m.mean())
            op = k_obs >= 1
            cfo_m.append(con_m[op].mean())
            ph.append(F.link_probability_mle(k_obs[op], con_m[op])) if _ < 2000 else None
        ph = np.array(ph)
        rows.append(dict(p_true=p, cf_all_continuous=float(np.mean(cf_c)), cf_all_observed=float(np.mean(cf_m)),
                         cf_opp_observed=float(np.mean(cfo_m)), ratio_cf_all=float(np.mean(cf_m) / np.mean(cf_c)),
                         p_hat_median=float(np.median(ph)), p_hat_bias=float(np.mean(ph) - p),
                         p_hat_2_5=float(np.percentile(ph, 2.5)), p_hat_97_5=float(np.percentile(ph, 97.5))))
    df = pd.DataFrame(rows)
    df.to_csv(T / "r5_synthetic_truth.csv", index=False)
    OUT["C_synthetic_truth"] = df.to_dict(orient="records")
    print(df.to_string(), flush=True)


def part_d():
    tc = pd.read_csv(T / "r6_terrain_connectivity.csv").set_index("station")
    st = pd.read_csv(ROOT / "data/grdc_stations_raw.csv").set_index("grdc_no")
    name = st.station_name.str.title().to_dict()

    def along(up, down):
        km, cur = 0.0, up
        for _ in range(10):
            nxt = tc.loc[cur, "terrain_next_downstream"]
            if not isinstance(nxt, str):
                return np.nan
            km += float(tc.loc[cur, "along_channel_km_to_next"])
            if nxt == down:
                return km
            cur = nxt
        return np.nan

    pr = pd.read_csv(T / "propagation_pairs.csv")
    pr["origin"] = pr.origin_node.map(name); pr["upstream"] = pr.upstream_node.map(name)
    fix = {"Lama Kara": "Lama Kara"}
    pr["channel_km"] = [along(fix.get(u, u), fix.get(o, o)) for u, o in zip(pr.upstream, pr.origin)]
    rho, p = stats.spearmanr(pr.channel_km, pr.lag_days)
    rho_d, p_d = stats.spearmanr(pr.upstream_duration, pr.lag_days)
    pr[["origin", "upstream", "origin_start", "lag_days", "overlap_days", "upstream_duration", "channel_km"]].to_csv(
        T / "r5_pairs_channel_distance.csv", index=False)
    OUT["D_lag_distance"] = dict(rho_channel=float(rho), p_channel=float(p), n=int(len(pr)),
                                 channel_km_range=[float(pr.channel_km.min()), float(pr.channel_km.max())],
                                 rho_upstream_duration=float(rho_d), p_upstream_duration=float(p_d))
    print(OUT["D_lag_distance"], flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["b", "c", "d", "a"]
    for w in which:
        {"a": part_a, "b": part_b, "c": part_c, "d": part_d}[w]()
    f = T / "r5_numbers.json"
    old = json.load(open(f)) if f.exists() else {}
    old.update(OUT)
    json.dump(old, open(f, "w"), indent=1, default=float)
