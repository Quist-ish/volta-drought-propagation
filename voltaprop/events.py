"""
Drought event detection and characterisation.

Faithful port of `04_ssi_drought_events.ipynb` (Crespillo Lopez et al., 2026).
Run theory is applied to the daily SSI series: days below tau = -1.28 are
flagged, adjacent runs separated by at most 10 non-drought days are pooled,
and pooled events shorter than 5 days are discarded. Each surviving event is
described by start and end date, duration, SSI-based severity, intensity,
peak SSI with its date, and a volumetric severity in hm3 computed against a
DOY-varying tenth-percentile discharge threshold.

One Volta-specific behaviour is added: days on which SSI is undefined
(masked by the seasonal-validity rule in `ssi.py`) are treated as non-drought
and, additionally, may not be absorbed into an event by pooling. Without that
second rule a short wet-season deficit on either side of the dry season would
be welded into a single spurious multi-month "event" spanning months in which
the index does not exist.

Author: I. Quist (NUIST)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class EventConfig:
    threshold_ssi: float = -1.28      # P10 of the standard normal
    pool_gap_days: int = 10
    min_duration_days: int = 5
    volumetric_percentile: int = 10
    doy_window_days: int = 15
    m3s_to_hm3_per_day: float = 86_400 / 1e6     # 0.0864
    # [VOLTA] forbid pooling across stretches where SSI is undefined
    block_pooling_across_invalid: bool = True


# ---------------------------------------------------------------------------
# Step 1 -- runs of consecutive drought days
# ---------------------------------------------------------------------------
def detect_raw_events(ssi: pd.Series, threshold: float) -> List[Tuple[int, int]]:
    """Runs of consecutive days with SSI < threshold. NaN counts as non-drought."""
    below = (ssi < threshold).to_numpy()
    events: List[Tuple[int, int]] = []
    in_event = False
    start = 0
    for i, flag in enumerate(below):
        if flag and not in_event:
            in_event, start = True, i
        elif not flag and in_event:
            in_event = False
            events.append((start, i - 1))
    if in_event:
        events.append((start, len(below) - 1))
    return events


# ---------------------------------------------------------------------------
# Step 2 -- pool short inter-event gaps
# ---------------------------------------------------------------------------
def pool_events(
    events: Sequence[Tuple[int, int]],
    dates: pd.DatetimeIndex,
    gap_days: int,
    valid_mask: np.ndarray | None = None,
    block_across_invalid: bool = True,
) -> List[Tuple[int, int]]:
    """
    Merge consecutive events separated by at most `gap_days` non-drought days.

    `valid_mask` is a boolean array, True where SSI is defined. When
    `block_across_invalid` is set, a gap containing any undefined day is never
    bridged, so events on opposite sides of the dry season stay separate.
    """
    if not events:
        return []

    pooled = [list(events[0])]
    for start, end in events[1:]:
        prev_end = pooled[-1][1]
        gap = (dates[start] - dates[prev_end]).days - 1

        bridgeable = gap <= gap_days
        if bridgeable and block_across_invalid and valid_mask is not None:
            if not valid_mask[prev_end + 1 : start].all():
                bridgeable = False

        if bridgeable:
            pooled[-1][1] = end
        else:
            pooled.append([start, end])

    return [tuple(e) for e in pooled]


# ---------------------------------------------------------------------------
# Step 3 -- minimum duration filter
# ---------------------------------------------------------------------------
def filter_events(
    events: Sequence[Tuple[int, int]], dates: pd.DatetimeIndex, min_days: int
) -> List[Tuple[int, int]]:
    return [(s, e) for s, e in events if (dates[e] - dates[s]).days + 1 >= min_days]


# ---------------------------------------------------------------------------
# Step 4 -- event descriptors (Table 1 of the source paper)
# ---------------------------------------------------------------------------
def compute_event_stats(
    events: Sequence[Tuple[int, int]],
    ssi: pd.Series,
    dates: pd.DatetimeIndex,
    node_id,
    threshold: float,
) -> List[dict]:
    records = []
    for event_id, (start, end) in enumerate(events, start=1):
        event_ssi = ssi.iloc[start : end + 1]
        duration = (dates[end] - dates[start]).days + 1

        drought_ssi = event_ssi[event_ssi < threshold]
        severity = float(np.sum(np.abs(threshold - drought_ssi)))
        intensity = severity / duration

        peak_pos = event_ssi.idxmin()
        peak_ssi = float(event_ssi.min())
        peak_date = dates[peak_pos]

        records.append(
            {
                "node_id": node_id,
                "event_id": event_id,
                "start_date": dates[start],
                "end_date": dates[end],
                "duration": duration,
                "severity": round(severity, 4),
                "intensity": round(intensity, 4),
                "peak_SSI": round(peak_ssi, 4),
                "peak_date": peak_date,
                "n_drought_days": int(len(drought_ssi)),
            }
        )
    return records


# ---------------------------------------------------------------------------
# Detection across the whole network
# ---------------------------------------------------------------------------
def detect_network_events(
    ssi_df: pd.DataFrame,
    cfg: EventConfig = EventConfig(),
    node_col: str = "node_id",
    progress: bool = True,
) -> pd.DataFrame:
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kw):
            return x

    all_records: List[dict] = []
    nodes = ssi_df[node_col].unique()
    iterator = tqdm(nodes, desc="Detecting drought events") if progress else nodes

    for node in iterator:
        sub = ssi_df.loc[ssi_df[node_col] == node, ["date", "SSI"]].copy()
        sub["date"] = pd.to_datetime(sub["date"]).dt.normalize()
        sub = sub.drop_duplicates("date").set_index("date").sort_index()
        # A series holding only observed days lets a run of drought days jump
        # across a gap in the record. Reindexing to an unbroken calendar makes
        # every missing day NaN, which counts as non-drought and, with
        # block_pooling_across_invalid, cannot be bridged by pooling either.
        sub = sub.reindex(pd.date_range(sub.index.min(), sub.index.max(), freq="D"))
        ssi = sub["SSI"].reset_index(drop=True)
        dates = pd.DatetimeIndex(sub.index)
        valid_mask = ssi.notna().to_numpy()

        raw = detect_raw_events(ssi, cfg.threshold_ssi)
        pooled = pool_events(
            raw, dates, cfg.pool_gap_days, valid_mask, cfg.block_pooling_across_invalid
        )
        kept = filter_events(pooled, dates, cfg.min_duration_days)
        all_records.extend(compute_event_stats(kept, ssi, dates, node, cfg.threshold_ssi))

    cols = [
        "node_id", "event_id", "start_date", "end_date", "duration",
        "severity", "intensity", "peak_SSI", "peak_date", "n_drought_days",
    ]
    return pd.DataFrame(all_records, columns=cols)


# ---------------------------------------------------------------------------
# Volumetric severity
# ---------------------------------------------------------------------------
def compute_doy_q_threshold(
    q_df: pd.DataFrame,
    cfg: EventConfig = EventConfig(),
    node_col: str = "node_id",
    q_col: str = "Q",
) -> pd.DataFrame:
    """
    DOY-varying tenth-percentile discharge per node, pooled with the same
    circular +/-15-day window used for the SSI fit.
    """
    q = q_df[[node_col, "date", q_col]].copy()
    q["doy"] = pd.DatetimeIndex(q["date"]).dayofyear

    records = []
    for node, grp in q.groupby(node_col):
        vals = grp[q_col].to_numpy()
        doys = grp["doy"].to_numpy()
        for doy in range(1, 367):
            lo, hi = doy - cfg.doy_window_days, doy + cfg.doy_window_days
            if lo < 1:
                mask = (doys >= lo + 366) | (doys <= hi)
            elif hi > 366:
                mask = (doys >= lo) | (doys <= hi - 366)
            else:
                mask = (doys >= lo) & (doys <= hi)
            pool = vals[mask]
            pool = pool[~np.isnan(pool)]
            p10 = float(np.percentile(pool, cfg.volumetric_percentile)) if pool.size else np.nan
            records.append({node_col: node, "doy": doy, "Q_p10": p10})

    return pd.DataFrame(records)


def add_volumetric_severity(
    events_df: pd.DataFrame,
    q_df: pd.DataFrame,
    q_p10: pd.DataFrame,
    cfg: EventConfig = EventConfig(),
    node_col: str = "node_id",
    q_col: str = "Q",
) -> pd.DataFrame:
    """
    Cumulative discharge deficit below the DOY P10 threshold over each event
    window, in hm3. Event windows are taken from the SSI catalogue; no
    re-detection occurs in discharge space.
    """
    q = q_df[[node_col, "date", q_col]].copy()
    q["doy"] = pd.DatetimeIndex(q["date"]).dayofyear
    lookup = q.merge(q_p10, on=[node_col, "doy"], how="left")

    by_node = {n: g.set_index("date").sort_index() for n, g in lookup.groupby(node_col)}

    volumes = []
    for row in events_df.itertuples(index=False):
        sub = by_node.get(getattr(row, node_col))
        if sub is None:
            volumes.append(np.nan)
            continue
        window = sub.loc[row.start_date : row.end_date]
        deficit = np.maximum(0.0, window["Q_p10"].to_numpy() - window[q_col].to_numpy())
        volumes.append(round(float(np.nansum(deficit)) * cfg.m3s_to_hm3_per_day, 4))

    out = events_df.copy()
    out["severity_hm3"] = volumes
    return out
