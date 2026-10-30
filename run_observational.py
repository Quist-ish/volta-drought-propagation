#!/usr/bin/env python3
"""
Observational run: the full framework on the observed GRDC network.

This is the analysis the data in hand actually supports. It is not a
substitute for the GloFAS run, which will carry a denser node network, but it
is real observed discharge from the Volta and it produces real numbers.

Study period 1961-2007, matching the published study's 1961 start and bounded
at the end by the last observation in the GRDC holdings.

The index is fitted on each station's full record within that window, which
maximises the day-of-year climatology sample; propagation is then restricted
by the data itself to periods where connected stations observe concurrently.

Author: I. Quist (NUIST)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voltaprop.ssi import SSIConfig, compute_ssi_network
from voltaprop.events import (EventConfig, detect_network_events,
                              compute_doy_q_threshold, add_volumetric_severity)
from voltaprop.propagation import PropagationConfig, run_propagation, build_chain_summary
from voltaprop.grdc import build_connectivity
from voltaprop.analysis import permutation_null_test, non_propagating_events
from voltaprop.gapfill import fill_short_gaps, published_qc_gate, donor_tier_assessment

START, END = "1961-01-01", "2007-12-31"
OUT = Path("output_observational"); OUT.mkdir(exist_ok=True)
T = OUT / "tables"; T.mkdir(exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main() -> int:
    log("loading observed discharge")
    q = pd.read_csv("data/grdc_discharge_raw.csv", parse_dates=["date"])
    q = q.rename(columns={"grdc_no": "node_id"})
    q = q[(q["date"] >= START) & (q["date"] <= END)].copy()
    q = q.dropna(subset=["Q"])
    q = q[q["Q"] >= 0]

    # Restrict to stations inside the Volta Basin polygon. The GRDC download
    # also carries coastal and neighbouring basins (Ankobra, Ayensu, Mono,
    # Pra, Sio, Tanoe); an earlier run omitted this filter and 114 of its 288
    # events came from outside the basin. See docs/GAP_ANALYSIS.md.
    st = pd.read_csv("data/grdc_stations_raw.csv")
    inside = set(st.loc[st["in_basin"].astype(bool), "grdc_no"])
    dropped = sorted(set(q.node_id) - inside)
    q = q[q.node_id.isin(inside)].copy()
    log(f"  dropped {len(dropped)} stations outside the basin: "
        + ", ".join(st.set_index("grdc_no").loc[dropped, "river"].astype(str)))
    log(f"  {len(q):,} observations, {q.node_id.nunique()} stations, "
        f"{q.date.min().date()} to {q.date.max().date()}")
    have = sorted(q.node_id.unique())

    # Section 2.1 of the published study: station screen, then gap filling.
    qc = published_qc_gate(q, START, END)
    qc.to_csv(T / "qc_published_gate.csv", index=False)
    log(f"  published QC gate: {int(qc.pass_window.sum())} of {len(qc)} pass over the "
        f"common window; {int(qc.pass_own.sum())} pass on own span")
    full, fsum = fill_short_gaps(q, max_gap=15)
    fsum.to_csv(T / "gapfill_tier1_summary.csv", index=False)
    log(f"  tier-1 interpolation (<= 15 d): {int(fsum.filled_tier1.sum()):,} days filled; "
        f"{int(fsum.missing_after.sum()):,} days remain missing within station spans")

    conn, ctab = build_connectivity(st, restrict_to=have)
    t2, t2d = donor_tier_assessment(full, st, conn)
    t2.to_csv(T / "gapfill_tier2_assessment.csv", index=False)
    t2d.to_csv(T / "gapfill_tier2_donors.csv", index=False)
    log(f"  tiers 2-3 not applied; tier 2 could fill {int(t2.tier2_fillable.sum()):,} days, "
        f"tier 3 would supply {int(t2.tier3_remainder.sum()):,}")
    q = full.loc[full.Q.notna(), ["node_id", "date", "Q"]].copy()
    n_origin = sum(1 for n, up in conn.items() if up)
    log(f"  connectivity: {n_origin} stations have upstream neighbours; "
        f"{sum(len(v) for v in conn.values())} directed links")
    ctab.to_csv(T / "upstream_connectivity.csv", sep=";", index=False)

    log("computing SSI (seasonal-validity mask on; this is the slow stage)")
    t0 = time.perf_counter()
    scfg = SSIConfig()
    ssi_df, ssi_log = compute_ssi_network(q, scfg, node_col="node_id", progress=False)
    log(f"  SSI done in {time.perf_counter()-t0:.0f}s; "
        f"{ssi_df['SSI'].notna().sum():,} of {len(ssi_df):,} days defined "
        f"({100*ssi_df['SSI'].notna().mean():.1f}% of non-missing days)")
    # write the index on an unbroken calendar, with the fill tier carried along
    ssi_df = (full[["node_id", "date", "Q", "fill_tier"]]
              .merge(ssi_df[["node_id", "date", "SSI"]], on=["node_id", "date"], how="left"))
    ssi_df.to_csv(T / "SSI_daily.csv", index=False)
    ssi_log.to_csv(T / "SSI_fit_log.csv", index=False)

    log("detecting drought events")
    ecfg = EventConfig()
    ev = detect_network_events(ssi_df, ecfg, node_col="node_id")
    log(f"  {len(ev):,} events across {ev.node_id.nunique()} stations")

    log("adding volumetric severity")
    qp10 = compute_doy_q_threshold(q, ecfg, node_col="node_id")
    ev = add_volumetric_severity(ev, q, qp10, ecfg, node_col="node_id")
    ev.to_csv(T / "SSI_drought_events.csv", index=False)

    log("reconstructing propagation chains")
    pcfg = PropagationConfig(W_days=45, min_overlap_days=5, lag_max=0)
    pair_table, prop, chains = run_propagation(ev, conn, pcfg, node_col="node_id")
    pair_table.to_csv(T / "propagation_pairs.csv", index=False)
    chains.to_csv(T / "Chains_final.csv", index=False)
    log(f"  {len(prop):,} retained pairs; {len(chains):,} chains")

    log("network-permutation null test (1,999 permutations)")
    t0 = time.perf_counter()
    nullres = permutation_null_test(ev, conn, pcfg, n_permutations=1999,
                                    node_col="node_id", seed=42)
    nl = nullres[0] if isinstance(nullres, tuple) else nullres
    nl.to_csv(T / "null_summary.csv", index=False)
    log(f"  null test done in {time.perf_counter()-t0:.0f}s")

    log("non-propagating contrast")
    oc, uc = non_propagating_events(ev, prop, conn, node_col="node_id")
    oc.to_csv(T / "origin_contrast.csv", index=False)
    uc.to_csv(T / "upstream_contrast.csv", index=False)

    meta = {"period": [START, END], "n_obs": int(len(q)),
            "n_observed": int((full.fill_tier == 0).sum()),
            "n_filled_tier1": int((full.fill_tier == 1).sum()),
            "n_stations": int(q.node_id.nunique()),
            "n_origin_stations": int(n_origin),
            "n_events": int(len(ev)), "n_pairs": int(len(prop)),
            "n_chains": int(len(chains))}
    (T / "run_meta.json").write_text(json.dumps(meta, indent=2))
    log("done: " + json.dumps(meta))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
