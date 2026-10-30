"""
End-to-end smoke test on synthetic data.

Builds a small artificial network with a known upstream-to-downstream drought
signal, runs the full pipeline, and asserts that the framework recovers it.
This does not validate the science; it validates that the code executes and
that the matcher behaves as specified before anyone points it at real Volta
data. Run with:  python -I tests/test_pipeline_synthetic.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voltaprop.events import (  # noqa: E402
    EventConfig,
    add_volumetric_severity,
    compute_doy_q_threshold,
    detect_network_events,
)
from voltaprop.propagation import PropagationConfig, run_propagation  # noqa: E402
from voltaprop.ssi import SSIConfig, compute_ssi_network, validity_summary  # noqa: E402

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Synthetic basin: 1 and 2 are headwaters, 3 is mid-network, 4 is the outlet.
# ---------------------------------------------------------------------------
CONNECTIVITY = {1: [], 2: [], 3: [1, 2], 4: [1, 2, 3]}
DATES = pd.date_range("1979-01-01", "2024-12-31", freq="D")
N = len(DATES)
DOY = DATES.dayofyear.to_numpy()

# Monomodal West African regime: a single wet season peaking in late August.
seasonal = 40 + 220 * np.exp(-0.5 * ((DOY - 240) / 42.0) ** 2)

# Injected drought episodes, each starting earlier upstream than downstream.
EPISODES = [("1983-05-15", 170), ("1998-06-01", 150), ("2015-05-20", 160)]
ONSET_OFFSET = {1: 0, 2: 6, 3: 18, 4: 32}   # days after the episode start


def build_series(node_id: int) -> pd.DataFrame:
    scale = {1: 0.28, 2: 0.34, 3: 0.72, 4: 1.0}[node_id]
    q = seasonal * scale * np.exp(rng.normal(0, 0.22, N))

    for start, length in EPISODES:
        s = pd.Timestamp(start) + pd.Timedelta(days=ONSET_OFFSET[node_id])
        m = (DATES >= s) & (DATES < s + pd.Timedelta(days=length))
        q[m] *= 0.22

    # Headwaters go dry in the Sahelian dry season; the outlet stays perennial.
    if node_id in (1, 2):
        q[np.isin(DOY, np.r_[1:91, 336:367])] = 0.0

    return pd.DataFrame({"node_id": node_id, "date": DATES, "Q": np.maximum(q, 0.0)})


def main() -> int:
    q_df = pd.concat([build_series(n) for n in CONNECTIVITY], ignore_index=True)
    print(f"Synthetic discharge: {len(q_df):,} rows, {q_df['node_id'].nunique()} nodes")

    ssi_df, log_df = compute_ssi_network(q_df, SSIConfig(), progress=False)
    vs = validity_summary(log_df)
    print("\nSeasonal-validity mask (share of the calendar on which SSI is defined):")
    print(vs[["node_id", "valid_fraction", "mean_zero_fraction"]].to_string(index=False))

    assert vs.loc[vs.node_id.isin([1, 2]), "valid_fraction"].max() < 0.90, \
        "ephemeral headwaters should be partly masked"
    assert vs.loc[vs.node_id == 4, "valid_fraction"].iloc[0] > 0.95, \
        "the perennial outlet should be almost entirely valid"

    ecfg = EventConfig()
    events = detect_network_events(ssi_df, ecfg, progress=False)
    print(f"\nEvents detected: {len(events):,}")
    print(events.groupby("node_id").size().rename("n_events").to_string())

    q_p10 = compute_doy_q_threshold(q_df, ecfg)
    events = add_volumetric_severity(events, q_df, q_p10, ecfg)
    assert events["severity_hm3"].notna().all(), "volumetric severity has gaps"

    pairs_all, pairs, chains = run_propagation(
        events, CONNECTIVITY, PropagationConfig(), validate=True
    )
    print(f"\nPairs before lag filter : {len(pairs_all):,}")
    print(f"Pairs after lag filter  : {len(pairs):,}")
    print(f"Chains                  : {len(chains):,}")

    assert len(chains) > 0, "no chains reconstructed from a signal that was injected"
    assert (pairs["lag_days"] <= 0).all()
    assert (pairs["overlap_days"] >= 5).all()
    assert chains["propagation_fraction"].between(0, 1).all()

    outlet = chains[chains["origin_node"] == 4]
    assert not outlet.empty, "the outlet reconstructed no chain"
    assert (outlet["lag_mean"] <= 0).all(), "upstream onset must not follow the outlet"

    # The injected episodes must be recovered with a genuinely negative lag.
    recovered = 0
    for start, _ in EPISODES:
        s0 = pd.Timestamp(start)
        hit = outlet[
            (outlet["chain_start"] >= s0 - pd.Timedelta(days=45))
            & (outlet["chain_start"] <= s0 + pd.Timedelta(days=75))
            & (outlet["lag_mean"] < 0)
        ]
        if not hit.empty:
            recovered += 1
            print(
                f"  episode {start}: chain {hit.iloc[0]['chain_id']}, "
                f"lag_mean {hit.iloc[0]['lag_mean']:.1f} d, "
                f"size {int(hit.iloc[0]['chain_size'])}"
            )
    print(f"\nInjected episodes recovered with negative lag: {recovered}/{len(EPISODES)}")
    assert recovered == len(EPISODES), "the matcher failed to recover an injected signal"

    print("\nChain summary at the outlet:")
    print(
        outlet[["chain_id", "chain_start", "chain_duration", "chain_size",
                "propagation_fraction", "lag_mean", "chain_severity"]].to_string(index=False)
    )

    print("\nRecovered mean lag by origin node (days):")
    print(pairs.groupby("origin_node")["lag_days"].agg(["count", "mean", "median"]).round(1))

    print("\nAll assertions passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
