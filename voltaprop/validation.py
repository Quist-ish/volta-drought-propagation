"""
Benchmarking reanalysis discharge against observed gauges.

Crespillo Lopez et al. (2026) closed their Discussion by noting that moving
the framework onto gridded or reanalysis discharge "would require a dedicated
assessment of the additional uncertainty such products introduce -- largest
for low flows and in arid and semi-arid regions". The Volta is precisely that
case, so this assessment is not an appendix here; it is a condition of the
paper being believable, and it should appear in the main text.

Three levels of comparison, in increasing relevance to the actual claim:

  1. Flow-space skill -- KGE and its decomposition, log-NSE, percent bias.
     Standard, but a model can score well on these while misplacing every
     low-flow episode, so this level alone settles nothing.
  2. Low-flow skill -- the same metrics restricted to the lower decile, plus
     bias in the DOY-varying Q10 threshold that volumetric severity depends on.
  3. Index- and event-space agreement -- rank correlation between GRDC-derived
     and GloFAS-derived SSI, and hit rate, false-alarm ratio and onset-timing
     error between the two event catalogues.

Level 3 is what the propagation results actually rest on. A reviewer who sees
only Level 1 will, with justification, ask for Level 3.

Author: I. Quist (NUIST)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Flow-space metrics
# ---------------------------------------------------------------------------
def kge(sim: np.ndarray, obs: np.ndarray, decompose: bool = True) -> Dict[str, float]:
    """Kling-Gupta efficiency (Gupta et al., 2009) with its three components."""
    m = np.isfinite(sim) & np.isfinite(obs)
    s, o = sim[m], obs[m]
    if s.size < 30 or o.std() == 0:
        return {"KGE": np.nan, "r": np.nan, "alpha": np.nan, "beta": np.nan, "n": int(s.size)}
    r = float(np.corrcoef(s, o)[0, 1])
    alpha = float(s.std(ddof=1) / o.std(ddof=1))
    beta = float(s.mean() / o.mean()) if o.mean() != 0 else np.nan
    val = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    out = {"KGE": round(float(val), 4), "n": int(s.size)}
    if decompose:
        out.update({"r": round(r, 4), "alpha": round(alpha, 4), "beta": round(beta, 4)})
    return out


def nse_log(sim: np.ndarray, obs: np.ndarray, epsilon: Optional[float] = None) -> float:
    """Nash-Sutcliffe on log-transformed flows, which weights low flows."""
    m = np.isfinite(sim) & np.isfinite(obs)
    s, o = sim[m], obs[m]
    if s.size < 30:
        return np.nan
    if epsilon is None:
        positive = o[o > 0]
        epsilon = float(np.percentile(positive, 1) / 100) if positive.size else 1e-3
    ls, lo = np.log(s + epsilon), np.log(o + epsilon)
    denom = np.sum((lo - lo.mean()) ** 2)
    return round(float(1 - np.sum((ls - lo) ** 2) / denom), 4) if denom > 0 else np.nan


def pbias(sim: np.ndarray, obs: np.ndarray) -> float:
    m = np.isfinite(sim) & np.isfinite(obs)
    s, o = sim[m], obs[m]
    return round(float(100 * (s.sum() - o.sum()) / o.sum()), 3) if o.sum() != 0 else np.nan


# ---------------------------------------------------------------------------
# Level 1 and 2: flow and low-flow skill
# ---------------------------------------------------------------------------
def compare_discharge(
    obs_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    node_col: str = "node_id",
    q_col: str = "Q",
    low_flow_quantile: float = 0.10,
) -> pd.DataFrame:
    """Per-node skill of the reanalysis against the observed record."""
    rows = []
    merged = obs_df.merge(
        sim_df, on=[node_col, "date"], suffixes=("_obs", "_sim"), how="inner"
    )

    for nid, g in merged.groupby(node_col):
        o = g[f"{q_col}_obs"].to_numpy(dtype=float)
        s = g[f"{q_col}_sim"].to_numpy(dtype=float)
        rec = {node_col: nid, "n_days": len(g)}
        rec.update(kge(s, o))
        rec["NSE_log"] = nse_log(s, o)
        rec["pbias"] = pbias(s, o)

        valid = np.isfinite(o) & np.isfinite(s)
        if valid.sum() > 100:
            rec["r_spearman"] = round(
                float(stats.spearmanr(s[valid], o[valid]).statistic), 4
            )
            thr = float(np.nanquantile(o[valid], low_flow_quantile))
            low = valid & (o <= thr)
            if low.sum() > 30:
                rec["low_flow_KGE"] = kge(s[low], o[low], decompose=False)["KGE"]
                rec["low_flow_pbias"] = pbias(s[low], o[low])
            q10_o = float(np.nanquantile(o[valid], 0.10))
            q10_s = float(np.nanquantile(s[valid], 0.10))
            rec["Q10_obs"] = round(q10_o, 4)
            rec["Q10_sim"] = round(q10_s, 4)
            rec["Q10_rel_bias"] = (
                round(float((q10_s - q10_o) / q10_o), 4) if q10_o > 0 else np.nan
            )
        rows.append(rec)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Level 3: index and event agreement
# ---------------------------------------------------------------------------
def compare_ssi(
    ssi_obs: pd.DataFrame, ssi_sim: pd.DataFrame, node_col: str = "node_id"
) -> pd.DataFrame:
    """Rank agreement between the two SSI series, overall and in the dry tail."""
    merged = ssi_obs.merge(
        ssi_sim, on=[node_col, "date"], suffixes=("_obs", "_sim"), how="inner"
    )
    rows = []
    for nid, g in merged.groupby(node_col):
        a, b = g["SSI_obs"].to_numpy(), g["SSI_sim"].to_numpy()
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 100:
            rows.append({node_col: nid, "n": int(m.sum())})
            continue
        dry = m & (a < -1.0)
        rows.append(
            {
                node_col: nid,
                "n": int(m.sum()),
                "pearson_r": round(float(np.corrcoef(a[m], b[m])[0, 1]), 4),
                "spearman_r": round(float(stats.spearmanr(a[m], b[m]).statistic), 4),
                "rmse": round(float(np.sqrt(np.mean((a[m] - b[m]) ** 2))), 4),
                "mean_bias": round(float(np.mean(b[m] - a[m])), 4),
                "spearman_r_dry_tail": round(
                    float(stats.spearmanr(a[dry], b[dry]).statistic), 4
                ) if dry.sum() > 50 else np.nan,
                "n_dry": int(dry.sum()),
            }
        )
    return pd.DataFrame(rows)


def compare_event_catalogues(
    events_obs: pd.DataFrame,
    events_sim: pd.DataFrame,
    node_col: str = "node_id",
    tolerance_days: int = 10,
) -> pd.DataFrame:
    """
    Contingency agreement between the observed and reanalysis event
    catalogues. A simulated event counts as a hit when it overlaps an observed
    event, or when its onset falls within `tolerance_days` of one.

    POD  = hits / observed events
    FAR  = false alarms / simulated events
    CSI  = hits / (hits + misses + false alarms)
    """
    rows = []
    nodes = sorted(set(events_obs[node_col]) & set(events_sim[node_col]))

    for nid in nodes:
        o = events_obs[events_obs[node_col] == nid]
        s = events_sim[events_sim[node_col] == nid]
        if o.empty or s.empty:
            rows.append({node_col: nid, "n_obs": len(o), "n_sim": len(s)})
            continue

        os_ = o["start_date"].values.astype("datetime64[D]").astype(np.int64)
        oe_ = o["end_date"].values.astype("datetime64[D]").astype(np.int64)
        ss_ = s["start_date"].values.astype("datetime64[D]").astype(np.int64)
        se_ = s["end_date"].values.astype("datetime64[D]").astype(np.int64)

        matched_obs, matched_sim, onset_err = set(), set(), []
        for i in range(len(os_)):
            overlap = np.minimum(oe_[i], se_) - np.maximum(os_[i], ss_) + 1
            near = np.abs(ss_ - os_[i]) <= tolerance_days
            hit = np.nonzero((overlap > 0) | near)[0]
            if hit.size:
                j = hit[int(np.argmin(np.abs(ss_[hit] - os_[i])))]
                matched_obs.add(i)
                matched_sim.add(int(j))
                onset_err.append(int(ss_[j] - os_[i]))

        hits = len(matched_obs)
        misses = len(os_) - hits
        false_alarms = len(ss_) - len(matched_sim)
        denom = hits + misses + false_alarms

        rows.append(
            {
                node_col: nid,
                "n_obs": len(os_),
                "n_sim": len(ss_),
                "hits": hits,
                "misses": misses,
                "false_alarms": false_alarms,
                "POD": round(hits / len(os_), 4) if len(os_) else np.nan,
                "FAR": round(false_alarms / len(ss_), 4) if len(ss_) else np.nan,
                "CSI": round(hits / denom, 4) if denom else np.nan,
                "onset_bias_days": round(float(np.mean(onset_err)), 2) if onset_err else np.nan,
                "onset_mae_days": round(float(np.mean(np.abs(onset_err))), 2) if onset_err else np.nan,
            }
        )

    return pd.DataFrame(rows)


def validation_verdict(
    flow: pd.DataFrame, ssi: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    """
    Single table combining the three levels, with a per-node recommendation.

    The rule is deliberately conservative. A node is marked `use` only when the
    reanalysis reproduces both the low-flow regime and the event catalogue;
    where it reproduces the hydrograph but not the events, the node is marked
    `caution` and should carry a visible flag in the figures rather than being
    quietly retained.
    """
    out = flow.merge(ssi, on="node_id", how="outer", suffixes=("", "_ssi")).merge(
        events, on="node_id", how="outer", suffixes=("", "_evt")
    )

    def verdict(r) -> str:
        ok_flow = (r.get("KGE", np.nan) or np.nan) >= 0.5
        ok_low = (r.get("low_flow_KGE", np.nan) or np.nan) >= 0.3
        ok_evt = ((r.get("POD", np.nan) or np.nan) >= 0.5) and (
            (r.get("FAR", np.nan) or np.nan) <= 0.5
        )
        if ok_flow and ok_low and ok_evt:
            return "use"
        if ok_flow and not ok_evt:
            return "caution"
        if not ok_flow:
            return "exclude_from_validation_claims"
        return "caution"

    out["verdict"] = out.apply(verdict, axis=1)
    return out
