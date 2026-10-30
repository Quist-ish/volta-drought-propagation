"""
Quality control and gap filling for observed discharge, following Section 2.1
of Crespillo López et al. (2026).

The published workflow screens stations, then fills gaps in three tiers of
decreasing fidelity: (1) log-space linear interpolation for gaps of 15 days
or fewer; (2) donor-based log-regression on the ten nearest stations, where a
donor shares at least 1,500 concurrent observations and a log-space Pearson
correlation above 0.75; (3) the station's day-of-year climatological median.
In the Ebro the three tiers together touched 1.41 per cent of the record.

On the Volta network the published QC gate admits no station over 1961-2007,
and tiers 2 and 3 would have to manufacture a large share of the record. Tier
2 is also unsafe in a propagation study when it is extensive: a downstream
series reconstructed by regression on an upstream donor is partly a function
of that donor, so any drought it acquires is matched to the donor by
construction. This module therefore applies tier 1 only, and reports how much
of the record tiers 2 and 3 would have had to supply, so the decision is
auditable.

Author: I. Quist (NUIST)
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Inclusive (start, end) index pairs of True runs."""
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def calendar_complete(q: pd.DataFrame, node_col: str = "node_id",
                      q_col: str = "Q") -> pd.DataFrame:
    """Reindex every node to an unbroken daily calendar from its first to its
    last observation. Missing days carry NaN."""
    out = []
    for node, g in q.groupby(node_col):
        s = (g.assign(date=pd.to_datetime(g["date"]).dt.normalize())
               .drop_duplicates("date").set_index("date")[q_col].sort_index())
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
        out.append(pd.DataFrame({node_col: node, "date": s.index, q_col: s.values}))
    return pd.concat(out, ignore_index=True)


def fill_short_gaps(q: pd.DataFrame, max_gap: int = 15, node_col: str = "node_id",
                    q_col: str = "Q") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tier 1 of the published hierarchy. Interior NaN runs of length <= max_gap,
    bounded by observations on both sides, are filled by linear interpolation
    in log discharge. Where either bounding value is zero the interpolation is
    done in discharge space, since log(0) is undefined and a zero-bounded gap in
    an intermittent channel should not be bridged by a positive curve.

    Returns the calendar-complete table with a `fill_tier` column (0 observed,
    1 interpolated, NaN still missing) and a per-node summary.
    """
    full = calendar_complete(q, node_col, q_col)
    full["fill_tier"] = np.where(full[q_col].notna(), 0.0, np.nan)
    rows = []
    for node, idx in full.groupby(node_col).groups.items():
        idx = np.asarray(idx)
        v = full.loc[idx, q_col].to_numpy(dtype=float).copy()
        miss = np.isnan(v)
        filled = 0
        for s, e in _runs(miss):
            n = e - s + 1
            if s == 0 or e == len(v) - 1 or n > max_gap:
                continue
            a, b = v[s - 1], v[e + 1]
            t = np.arange(1, n + 1) / (n + 1)
            if a > 0 and b > 0:
                v[s:e + 1] = np.exp(np.log(a) + t * (np.log(b) - np.log(a)))
            else:
                v[s:e + 1] = a + t * (b - a)
            filled += n
        full.loc[idx, q_col] = v
        tier = full.loc[idx, "fill_tier"].to_numpy().copy()
        tier[miss & ~np.isnan(v)] = 1.0
        full.loc[idx, "fill_tier"] = tier
        rows.append({node_col: node, "span_days": len(v), "observed": int((~miss).sum()),
                     "missing_before": int(miss.sum()), "filled_tier1": filled,
                     "missing_after": int(np.isnan(v).sum())})
    summ = pd.DataFrame(rows)
    summ["pct_missing_before"] = (100 * summ.missing_before / summ.span_days).round(2)
    summ["pct_filled_tier1"] = (100 * summ.filled_tier1 / summ.span_days).round(2)
    summ["pct_missing_after"] = (100 * summ.missing_after / summ.span_days).round(2)
    return full, summ


def published_qc_gate(q: pd.DataFrame, start: str, end: str, node_col: str = "node_id",
                      q_col: str = "Q", max_pct_missing: float = 30.0,
                      max_gap_days: int = 365, max_zero_run: int = 120) -> pd.DataFrame:
    """
    The station screen of Section 2.1, evaluated two ways. 'window' measures
    missingness and the longest gap against the full common analysis period,
    which is how the published screen reduced 295 candidate records to 45.
    'own span' measures them against the station's first-to-last observation
    only, a more lenient reading offered so the result does not hinge on
    interpretation.
    """
    cal = pd.date_range(start, end, freq="D")
    rows = []
    for node, g in q.groupby(node_col):
        s = (g.assign(date=pd.to_datetime(g["date"]).dt.normalize())
               .drop_duplicates("date").set_index("date")[q_col].reindex(cal))
        obs = s.notna().to_numpy()
        zero = (s == 0).to_numpy()
        first, last = (np.argmax(obs), len(obs) - 1 - np.argmax(obs[::-1])) if obs.any() else (0, -1)
        own = obs[first:last + 1]
        gap_w = max((e - s_ + 1 for s_, e in _runs(~obs)), default=0)
        gap_o = max((e - s_ + 1 for s_, e in _runs(~own)), default=0)
        zr = max((e - s_ + 1 for s_, e in _runs(zero)), default=0)
        r = {node_col: node, "first": cal[first].date() if obs.any() else None,
             "last": cal[last].date() if obs.any() else None,
             "n_obs": int(obs.sum()),
             "pct_missing_window": round(100 * (1 - obs.mean()), 1),
             "max_gap_window": int(gap_w),
             "pct_missing_own": round(100 * (1 - own.mean()), 1) if own.size else 100.0,
             "max_gap_own": int(gap_o), "max_zero_run": int(zr)}
        r["pass_window"] = bool(r["pct_missing_window"] <= max_pct_missing and gap_w <= max_gap_days
                               and zr <= max_zero_run)
        r["pass_own"] = bool(r["pct_missing_own"] <= max_pct_missing and gap_o <= max_gap_days
                            and zr <= max_zero_run)
        r["pass_own_no_zero_rule"] = bool(r["pct_missing_own"] <= max_pct_missing
                                         and gap_o <= max_gap_days)
        rows.append(r)
    return pd.DataFrame(rows)


def _hav(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.radians, [lat1, lon1, lat2, lon2])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


def donor_tier_assessment(full: pd.DataFrame, stations: pd.DataFrame,
                          connectivity: Dict[int, List[int]], node_col: str = "node_id",
                          q_col: str = "Q", n_nearest: int = 10, min_overlap: int = 1500,
                          min_r: float = 0.75) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    What tiers 2 and 3 would have had to do, without doing it. For each target
    station, donors are the ten nearest stations sharing >= min_overlap
    concurrent positive observations with log-space Pearson r > min_r. Reports
    the share of still-missing days a qualifying donor could fill, the share
    left for climatological medians, and whether each qualifying donor is
    hydraulically connected to the target, which is the contamination risk.
    """
    st = stations.set_index("grdc_no")
    wide = full.pivot(index="date", columns=node_col, values=q_col)
    logq = np.log(wide.where(wide > 0))
    nodes = list(wide.columns)
    donors_rows, summ = [], []
    for t in nodes:
        d = {u: _hav(st.lat[t], st.lon[t], st.lat[u], st.lon[u]) for u in nodes if u != t}
        near = sorted(d, key=d.get)[:n_nearest]
        qual = []
        for u in near:
            both = logq[[t, u]].dropna()
            if len(both) < min_overlap:
                continue
            r = float(np.corrcoef(both[t], both[u])[0, 1])
            linked = (u in connectivity.get(t, [])) or (t in connectivity.get(u, []))
            donors_rows.append({"target": t, "donor": u, "km": round(d[u], 1),
                                "n_concurrent": len(both), "r_log": round(r, 3),
                                "qualifies": r > min_r, "hydraulically_linked": linked})
            if r > min_r:
                qual.append(u)
        miss = wide[t].isna()
        span = wide[t].loc[wide[t].first_valid_index():wide[t].last_valid_index()]
        miss = span.isna()
        can = miss & (wide.loc[span.index, qual].notna().any(axis=1) if qual
                      else pd.Series(False, index=span.index))
        summ.append({node_col: t, "missing_after_tier1": int(miss.sum()),
                     "n_qualifying_donors": len(qual),
                     "n_qualifying_linked": sum(1 for u in qual if (u in connectivity.get(t, []))
                                                or (t in connectivity.get(u, []))),
                     "tier2_fillable": int(can.sum()),
                     "tier3_remainder": int(miss.sum() - can.sum()),
                     "span_days": len(span)})
    return pd.DataFrame(summ), pd.DataFrame(donors_rows)
