"""
Exact, vectorised event matcher with observing-opportunity accounting, and the
availability-aware network-label permutation null.

The matcher implements Section 2.2.3 (after Crespillo López et al., 2026). It is
exact, not an approximation, because of two properties proved in Appendix A
(Proposition 1):

  P1  every retained upstream candidate contains the origin onset s0
      (s_u <= s0 <= e_u), for any window W >= 0 and overlap M >= 1;
  P2  events at one station are disjoint, so at most one upstream event per
      station can contain s0, and the four-level tie-break is never invoked.

For each origin event the matcher therefore looks up, at each upstream station,
the last event starting on or before s0 (one `searchsorted`), and keeps it if it
is still running at s0 and overlaps the origin event by at least M days.

Beyond the original matcher it records, for each origin event, how many
upstream stations were *observing*: holding at least MIN_DEFINED defined index
days within [s0 - W, e0], the span in which a qualifying upstream event could
lie. An origin event with at least one observing upstream station is an
opportunity. The per-link connection probability p is the maximum-likelihood
estimate under P(connected | k) = 1 - (1 - p)^k (Appendix A, Proposition 3).

The null reassigns whole station records (event catalogue plus daily
availability) to network slots with the topology fixed. With records that carry
no availability mask it reproduces the published null of the original code
exactly, including its random stream (numpy default_rng, permutation of the
sorted station identifiers).

Author: I. Quist (NUIST)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

EPOCH = np.datetime64("1970-01-01", "D")
MIN_DEFINED = 5
NULL_STATS = ("cf_all", "cf_opp", "p_link", "mean_chain_size", "n_connected", "n_pairs",
              "n_opportunities", "coverage", "mean_pf_published")


def to_day(x) -> np.ndarray:
    """Dates (scalar, array, Series) to integer days since 1970-01-01."""
    a = np.asarray(pd.to_datetime(x)).astype("datetime64[D]")
    return (a - EPOCH).astype(np.int64)


@dataclass
class Record:
    """One station's event catalogue (sorted by start) and optional availability."""
    start: np.ndarray
    end: np.ndarray
    sev: np.ndarray
    eid: np.ndarray
    cum: Optional[np.ndarray] = None   # cumulative count of defined days, length L + 1
    d0: Optional[int] = None           # day number of the first element of the mask

    @property
    def n_events(self) -> int:
        return int(len(self.start))

    def defined_days(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Number of defined days in the inclusive spans [a, b] (vectorised)."""
        a = np.asarray(a, dtype=np.int64); b = np.asarray(b, dtype=np.int64)
        if self.cum is None:
            return np.maximum(b - a + 1, 0)
        L = len(self.cum) - 1
        ia = np.clip(a - self.d0, 0, L)
        ib = np.clip(b - self.d0 + 1, 0, L)
        return np.maximum(self.cum[ib] - self.cum[ia], 0)


def mask_to_cum(mask: np.ndarray) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(np.asarray(mask, dtype=np.int64))])


def availability_from_daily(df: pd.DataFrame, col: str = "SSI", node_col: str = "node_id",
                            date_col: str = "date") -> Dict[int, Tuple[int, np.ndarray]]:
    """node -> (d0, cum) from a long daily table; a day is defined when `col` is not NaN."""
    out = {}
    for n, g in df[[node_col, date_col, col]].groupby(node_col):
        d = to_day(g[date_col].values)
        d0, d1 = int(d.min()), int(d.max())
        mask = np.zeros(d1 - d0 + 1, dtype=bool)
        mask[d - d0] = g[col].notna().values
        out[int(n)] = (d0, mask_to_cum(mask))
    return out


def build_records(events: pd.DataFrame, avail: Optional[Dict[int, Tuple[int, np.ndarray]]] = None,
                  nodes: Optional[Iterable[int]] = None, node_col: str = "node_id") -> Dict[int, Record]:
    """Records for every node with events, plus any node listed in `nodes` (possibly without events)."""
    recs: Dict[int, Record] = {}
    for n, g in events.groupby(node_col):
        g = g.sort_values("start_date")
        recs[int(n)] = Record(to_day(g.start_date.values), to_day(g.end_date.values),
                              g.severity.values.astype(float), g.event_id.values.astype(np.int64))
    for n in (nodes or []):
        n = int(n)
        if n not in recs:
            e = np.array([], dtype=np.int64)
            recs[n] = Record(e, e.copy(), np.array([], float), e.copy())
    if avail is not None:
        for n, r in recs.items():
            if n in avail:
                r.d0, r.cum = avail[n][0], avail[n][1]
            else:                                   # no daily series: never observing
                r.d0, r.cum = 0, np.zeros(1, dtype=np.int64)
    return recs


def link_probability_mle(k: np.ndarray, connected: np.ndarray) -> float:
    """MLE of p in P(connected | k) = 1 - (1 - p)^k over opportunities (k >= 1)."""
    k = np.asarray(k, dtype=float); c = np.asarray(connected, dtype=bool)
    if len(k) == 0:
        return np.nan
    if c.all():
        return 1.0
    if not c.any():
        return 0.0

    def nll(p):
        q = 1.0 - p
        return -(np.log1p(-q ** k[c]).sum() + (k[~c] * np.log(q)).sum())

    return float(minimize_scalar(nll, bounds=(1e-9, 1 - 1e-9), method="bounded",
                                 options={"xatol": 1e-10}).x)


def match(recs: Dict[int, Record], conn: Dict[int, List[int]], W: int = 45, M: int = 5,
          slot: Optional[Dict[int, int]] = None, detail: bool = True):
    """
    Match every origin event against its upstream stations.

    `slot` maps a network position to the station whose record occupies it
    (identity when None). Returns (pairs, origins) as DataFrames when
    `detail` is True, or a dict of arrays used by the null when False.
    """
    get = (lambda n: recs.get(n)) if slot is None else (lambda n: recs.get(slot.get(n, n)) if n in slot else recs.get(n))
    P = {k: [] for k in ("origin_node", "origin_eid", "upstream_node", "upstream_eid", "overlap", "lag",
                         "s0", "e0", "su", "eu")}
    O = {k: [] for k in ("origin_node", "origin_eid", "s0", "e0", "n_upstream", "n_avail", "n_observing",
                         "n_matched", "observing")}
    for o, ups in conn.items():
        ro = get(o)
        if not ups or ro is None or ro.n_events == 0:
            continue
        s0, e0 = ro.start, ro.end
        n = len(s0)
        k_obs = np.zeros(n, dtype=np.int64); k_mat = np.zeros(n, dtype=np.int64)
        n_avail = 0
        obs_lists = [[] for _ in range(n)] if detail else None
        for u in ups:
            ru = get(u)
            if ru is None:
                continue
            if ru.n_events > 0:
                n_avail += 1
            dd = ru.defined_days(s0 - W, e0)
            obs = dd >= MIN_DEFINED
            k_obs += obs
            if detail:
                for i in np.flatnonzero(obs):
                    obs_lists[i].append(u)
            if ru.n_events == 0:
                continue
            j = np.searchsorted(ru.start, s0, side="right") - 1
            ok = j >= 0
            jj = np.where(ok, j, 0)
            eu = ru.end[jj]
            ov = np.minimum(e0, eu) - s0 + 1
            hit = ok & (eu >= s0) & (ov >= M)
            if not hit.any():
                continue
            k_mat += hit
            idx = np.flatnonzero(hit)
            P["origin_node"].append(np.full(len(idx), o)); P["origin_eid"].append(ro.eid[idx])
            P["upstream_node"].append(np.full(len(idx), u)); P["upstream_eid"].append(ru.eid[jj[idx]])
            P["overlap"].append(ov[idx]); P["lag"].append(ru.start[jj[idx]] - s0[idx])
            P["s0"].append(s0[idx]); P["e0"].append(e0[idx])
            P["su"].append(ru.start[jj[idx]]); P["eu"].append(eu[idx])
        O["origin_node"].append(np.full(n, o)); O["origin_eid"].append(ro.eid)
        O["s0"].append(s0); O["e0"].append(e0)
        O["n_upstream"].append(np.full(n, len(ups))); O["n_avail"].append(np.full(n, n_avail))
        O["n_observing"].append(k_obs); O["n_matched"].append(k_mat)
        if detail:
            O["observing"].extend(obs_lists)
    cat = lambda v, dt=np.int64: np.concatenate(v) if v else np.array([], dtype=dt)
    pairs = {k: cat(v) for k, v in P.items()}
    org = {k: cat(v) for k, v in O.items() if k != "observing"}
    org["connected"] = org["n_matched"] > 0
    if not detail:
        return pairs, org
    pdf = pd.DataFrame(pairs)
    odf = pd.DataFrame(org)
    odf["observing"] = O["observing"]
    for df, cols in ((pdf, ("s0", "e0", "su", "eu")), (odf, ("s0", "e0"))):
        for c in cols:
            df[c + "_date"] = EPOCH + df[c].values.astype("timedelta64[D]") if len(df) else pd.Series(dtype="datetime64[ns]")
    return pdf, odf


def _stats(pairs: dict, org: dict) -> dict:
    n_org = len(org["n_matched"])
    k = org["n_observing"]; conn = org["connected"]
    opp = k >= 1
    n_opp = int(opp.sum()); n_con = int(conn.sum())
    cs = org["n_matched"][conn] + 1
    pf = org["n_matched"][conn] / np.maximum(org["n_avail"][conn], 1)
    pfo = org["n_matched"][conn] / np.maximum(k[conn], 1)
    return {
        "n_origin_events": n_org, "n_opportunities": n_opp, "n_connected": n_con,
        "n_pairs": int(len(pairs["lag"])),
        "coverage": n_opp / n_org if n_org else np.nan,
        "cf_all": n_con / n_org if n_org else np.nan,
        "cf_opp": n_con / n_opp if n_opp else np.nan,
        "mean_k_observing": float(k[opp].mean()) if n_opp else np.nan,
        "p_link": link_probability_mle(k[opp], conn[opp]) if n_opp else np.nan,
        "mean_chain_size": float(cs.mean()) if n_con else np.nan,
        "mean_pf_published": float(pf.mean()) if n_con else np.nan,
        "mean_pf_observing": float(pfo.mean()) if n_con else np.nan,
    }


def statistics(recs: Dict[int, Record], conn: Dict[int, List[int]], W: int = 45, M: int = 5,
               slot: Optional[Dict[int, int]] = None) -> dict:
    pairs, org = match(recs, conn, W=W, M=M, slot=slot, detail=False)
    return _stats(pairs, org)


def null_test(recs: Dict[int, Record], conn: Dict[int, List[int]], n_perm: int = 1999, seed: int = 42,
              W: int = 45, M: int = 5, ids: Optional[Iterable[int]] = None):
    """
    Network-label permutation null. Whole records (catalogue plus availability)
    are reassigned to network slots; the topology is fixed. By default every
    station holding a record is permuted, as in the original code.

    Returns (observed dict, summary DataFrame, null DataFrame).
    """
    obs = statistics(recs, conn, W=W, M=M)
    ids = np.array(sorted(recs.keys()) if ids is None else sorted(ids))
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_perm):
        perm = rng.permutation(ids)
        slot = {int(ids[i]): int(perm[i]) for i in range(len(ids))}
        rows.append(statistics(recs, conn, W=W, M=M, slot=slot))
    null = pd.DataFrame(rows)
    summ = []
    for s in NULL_STATS:
        v = null[s].dropna().values
        o = obs[s]
        summ.append({"statistic": s, "observed": o, "null_mean": float(v.mean()) if len(v) else np.nan,
                     "null_sd": float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                     "null_p95": float(np.percentile(v, 95)) if len(v) else np.nan,
                     "p": float((1 + np.sum(v >= o)) / (1 + len(v))) if len(v) and np.isfinite(o) else np.nan,
                     "percentile": float(100 * np.mean(v < o)) if len(v) and np.isfinite(o) else np.nan,
                     "n_valid": int(len(v))})
    return obs, pd.DataFrame(summ), null
