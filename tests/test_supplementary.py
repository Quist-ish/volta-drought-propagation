"""
Exercise every supplementary analysis S1-S9 and its figure on synthetic data.

This verifies the code paths run and return the expected shapes; it says
nothing about the science. Run with:  python -I tests/test_supplementary.py
"""

import tempfile
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voltaprop import figures_supp as FS  # noqa: E402
from voltaprop import supplementary as S  # noqa: E402
from voltaprop.analysis import non_propagating_events, permutation_null_test  # noqa: E402
from voltaprop.propagation import PropagationConfig, run_propagation  # noqa: E402

rng = np.random.default_rng(11)
OUT = tempfile.mkdtemp(prefix="voltaprop_supptest_")
Path(OUT).mkdir(parents=True, exist_ok=True)

# A 12-node network: 1-6 headwaters, 7-9 mid, 10-11 lower, 12 outlet.
CONNECTIVITY = {
    1: [], 2: [], 3: [], 4: [], 5: [], 6: [],
    7: [1, 2], 8: [3, 4], 9: [5, 6],
    10: [1, 2, 3, 4, 7, 8],
    11: [5, 6, 9],
    12: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}
DEPTH = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 1, 9: 1, 10: 2, 11: 2, 12: 3}


def synth_events(n_episodes: int = 40) -> pd.DataFrame:
    """Event catalogue with a genuine upstream-first signal plus noise."""
    recs = []
    counter = {n: 0 for n in CONNECTIVITY}

    # Coherent basin-wide episodes, onset delayed with network depth.
    for _ in range(n_episodes):
        t0 = pd.Timestamp("1979-01-01") + pd.Timedelta(days=int(rng.integers(0, 16000)))
        dur = int(rng.integers(60, 240))
        for n in CONNECTIVITY:
            if rng.random() < 0.25:
                continue
            start = t0 + pd.Timedelta(days=DEPTH[n] * 14 + int(rng.integers(0, 10)))
            d = max(10, dur + int(rng.integers(-30, 30)))
            counter[n] += 1
            recs.append({
                "node_id": n, "event_id": counter[n], "start_date": start,
                "end_date": start + pd.Timedelta(days=d), "duration": d,
                "severity": float(rng.lognormal(2.6, 0.8)),
                "intensity": float(rng.lognormal(-1.3, 0.5)),
                "peak_SSI": -float(rng.lognormal(0.6, 0.3)),
                "peak_date": start + pd.Timedelta(days=d // 2),
                "n_drought_days": d,
                "severity_hm3": float(rng.lognormal(1.6, 1.6)),
            })

    # Independent short noise events.
    for n in CONNECTIVITY:
        for _ in range(int(rng.integers(20, 45))):
            start = pd.Timestamp("1979-01-01") + pd.Timedelta(days=int(rng.integers(0, 16400)))
            d = int(rng.integers(6, 30))
            counter[n] += 1
            recs.append({
                "node_id": n, "event_id": counter[n], "start_date": start,
                "end_date": start + pd.Timedelta(days=d), "duration": d,
                "severity": float(rng.lognormal(0.7, 0.7)),
                "intensity": float(rng.lognormal(-1.6, 0.5)),
                "peak_SSI": -float(rng.lognormal(0.35, 0.25)),
                "peak_date": start + pd.Timedelta(days=d // 2),
                "n_drought_days": d,
                "severity_hm3": float(rng.lognormal(0.3, 1.4)),
            })

    df = pd.DataFrame(recs).sort_values(["node_id", "start_date"]).reset_index(drop=True)
    df["event_id"] = df.groupby("node_id").cumcount() + 1
    return df


def main() -> int:
    events = synth_events()
    print(f"Synthetic catalogue: {len(events):,} events, {events.node_id.nunique()} nodes")

    _, pairs, chains = run_propagation(events, CONNECTIVITY, PropagationConfig(),
                                       validate=False)
    print(f"Pairs {len(pairs):,} | chains {len(chains):,}")

    # ---- S1 -------------------------------------------------------------
    s1, s1_per_event = S.s1_event_level_m_sensitivity(events, CONNECTIVITY)
    print("\nS1 event-level M sensitivity")
    print(s1[["M_days", "connected_events_pct", "flip_vs_reference_pct",
              "spearman_rho_vs_reference", "mean_delta_chain"]].to_string(index=False))
    assert s1.loc[s1.M_days == 5, "flip_vs_reference_pct"].iloc[0] == 0.0
    assert abs(s1.loc[s1.M_days == 5, "spearman_rho_vs_reference"].iloc[0] - 1.0) < 1e-9
    FS.figS1_m_sensitivity(s1, s1_per_event, path_stem=f"{OUT}/figS1")

    # ---- S1.4 -----------------------------------------------------------
    s14 = S.s1_4_gap_tolerance_analysis(events, CONNECTIVITY)
    print("\nS1.4 gap tolerance")
    print(s14.to_string(index=False))
    assert s14["new_pairs"].is_monotonic_increasing

    # ---- S2 -------------------------------------------------------------
    cases = chains.nlargest(3, "chain_size")["chain_id"].tolist()
    s2 = S.s2_case_study_m_sensitivity(cases, events, CONNECTIVITY)
    print(f"\nS2 case sensitivity ({len(cases)} cases x "
          f"{s2.M_days.nunique()} M values)")
    print(s2[s2.M_days.isin([1, 5, 15])][
        ["chain_id", "M_days", "chain_size", "jaccard_vs_default"]].to_string(index=False))
    FS.figS2_case_m_sensitivity(s2, path_stem=f"{OUT}/figS2")

    # ---- S4 -------------------------------------------------------------
    nullsum, nulldist = permutation_null_test(
        events, CONNECTIVITY, PropagationConfig(), n_permutations=60, progress=False
    )
    print("\nS4 null test (60 permutations for speed; use 1,999 in production)")
    print(nullsum[["statistic", "kind", "observed", "null_mean",
                   "observed_percentile_in_null", "p_value_one_sided"]].to_string(index=False))
    assert len(nullsum) == 5, "S4 must report five statistics"
    FS.figS4_null_test(nullsum, nulldist, path_stem=f"{OUT}/figS4")

    # ---- S5 -------------------------------------------------------------
    dates = pd.date_range("1990-01-01", "2010-12-31", freq="D")
    ssi_df = pd.concat([
        pd.DataFrame({
            "node_id": n, "date": dates,
            "SSI": pd.Series(rng.normal(0, 1, len(dates)))
                     .rolling(20, min_periods=1).mean().to_numpy() * 2.3,
        }) for n in range(1, 9)
    ], ignore_index=True)
    s5, s5_per_node = S.s5_detection_sensitivity(ssi_df, progress=False)
    print("\nS5 detection sensitivity (selected rows)")
    print(s5[s5.min_duration == 5][
        ["pool_gap", "min_duration", "n_events", "pct_change_vs_default",
         "rho_count_vs_default", "rho_severity_vs_default"]].to_string(index=False))
    assert s5["is_default"].sum() == 1
    assert abs(s5.loc[s5.is_default, "rho_severity_vs_default"].iloc[0] - 1.0) < 1e-9
    FS.figS5_detection_sensitivity(s5, path_stem=f"{OUT}/figS5")

    # ---- S6 -------------------------------------------------------------
    spei = pd.concat([
        pd.DataFrame({
            "node_id": n, "date": dates,
            "SPEI_3": pd.Series(rng.normal(0, 1, len(dates)))
                        .rolling(90, min_periods=1).mean().to_numpy() * 2.0,
            "SPEI_12": pd.Series(rng.normal(0, 1, len(dates)))
                         .rolling(365, min_periods=1).mean().to_numpy() * 3.0,
        }) for n in range(1, 13)
    ], ignore_index=True)
    nodes = pd.DataFrame({
        "node_id": list(CONNECTIVITY), "lon": rng.uniform(-4, 1, 12),
        "lat": rng.uniform(6, 14, 12), "subbasin": "White Volta",
        "country": "Ghana", "uparea_km2": rng.uniform(1e3, 1e5, 12),
    })
    in_window = [c for c in cases
                 if pd.Timestamp(chains.set_index("chain_id").loc[c, "chain_start"]) >= dates[0]
                 and pd.Timestamp(chains.set_index("chain_id").loc[c, "chain_end"]) <= dates[-1]]
    probe = in_window or cases[:1]
    s6 = S.s6_meteorological_context(probe, chains, ssi_df, events, spei, nodes)
    print("\nS6 meteorological context")
    print(s6[["chain_id", "n_own_events_in_window", "own_coverage_of_window",
              "SPEI3_at_onset", "network_adds_value"]].to_string(index=False))
    FS.figS6_meteorological_context(probe, chains, ssi_df, events, spei,
                                    path_stem=f"{OUT}/figS6")

    # ---- S7 -------------------------------------------------------------
    oc, uc = non_propagating_events(events, pairs, CONNECTIVITY)
    print("\nS7 non-propagating events")
    print(oc.to_string(index=False))
    print(uc.to_string(index=False))
    by_node = (
        events[events.node_id.isin([n for n, c in CONNECTIVITY.items() if c])]
        .assign(connected=lambda d: [
            (n, e) in set(zip(pairs.origin_node, pairs.origin_event_id))
            for n, e in zip(d.node_id, d.event_id)])
        .groupby("node_id")["connected"]
        .apply(lambda s: 100 * (~s).mean())
        .rename("pct_unconnected").reset_index()
        .rename(columns={"node_id": "origin_node"})
    )
    FS.figS7_non_propagating(oc, uc, by_node, path_stem=f"{OUT}/figS7")

    # ---- S8 -------------------------------------------------------------
    cat = pd.DataFrame({
        "event_id": range(1, 13),
        "start_date": pd.to_datetime([f"{y}-06-01" for y in range(1980, 2016, 3)]),
        "end_date": pd.to_datetime([f"{y + 1}-05-31" for y in range(1980, 2016, 3)]),
    })
    flagged, s8 = S.s8_independent_catalogue_validation(
        chains, cat, "1979-01-01", "2024-12-31")
    print("\nS8 catalogue validation")
    print(s8.T.to_string())
    assert 0 <= s8["pct_chains_overlapping"].iloc[0] <= 100
    FS.figS8_catalogue_validation(flagged, s8, cat, path_stem=f"{OUT}/figS8")

    # ---- S9 -------------------------------------------------------------
    screen = S.s9_homogeneity_screening(ssi_df)
    print("\nS9 homogeneity screening")
    print(screen[["node_id", "ks_statistic", "ks_pvalue", "mean_shift_halves",
                  "pettitt_year", "flag"]].to_string(index=False))
    res = pd.DataFrame({
        "name": ["Akosombo", "Bui", "Bagre", "Kompienga"],
        "lon": rng.uniform(-4, 1, 4), "lat": rng.uniform(6, 14, 4),
    })
    tbl, s9 = S.s9_reservoir_proximity_test(screen, nodes[nodes.node_id <= 8], res)
    print(s9.to_string(index=False))
    FS.figS9_homogeneity_screening(tbl, s9, path_stem=f"{OUT}/figS9")

    print("\nAll supplementary analyses S1-S9 ran and all assertions passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
