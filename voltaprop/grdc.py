"""
Reader for GRDC station data files (`*_Q_Day.Cmd.txt`).

The GRDC export format carries, in its comment header, everything the
framework needs to build the network: station identity, river, country,
coordinates, catchment area, and, decisively, the **next downstream station**.
That last field makes the drainage topology explicit, so connectivity can be
assembled by following the downstream pointers rather than by tracing D8 flow
directions across a DEM. This removes the single most error-prone step of the
Ebro workflow, where connectivity was built by hand in a GIS.

Header fields parsed (names as GRDC writes them):

    GRDC-No., River, Station, Country, Latitude (DD), Longitude (DD),
    Catchment area (km2), Altitude (m ASL), Next downstream station,
    Owner of original data, Unit of measure, Time series, No. of years,
    Data lines

Encoding: the files are DOS-ASCII and the degree and superscript characters
in "km²" and "m³/s" are latin-1, so they are read with latin-1 and the unit
string is normalised rather than matched literally.

Author: I. Quist (NUIST)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

MISSING = -999.0

_HEADER_MAP = {
    "GRDC-No.": "grdc_no",
    "River": "river",
    "Station": "station_name",
    "Country": "country_code",
    "Latitude (DD)": "lat",
    "Longitude (DD)": "lon",
    "Catchment area (km": "area_km2",          # trailing unit char varies
    "Altitude (m ASL)": "altitude_m",
    "Next downstream station": "next_downstream",
    "Owner of original data": "owner",
    "Unit of measure": "unit_raw",
    "Time series": "period_raw",
    "No. of years": "n_years_header",
    "Data lines": "n_data_lines",
    "Data Set Content": "content",
}

COUNTRY_NAMES = {
    "GH": "Ghana", "BF": "Burkina Faso", "TG": "Togo", "TO": "Togo",
    "ML": "Mali", "BJ": "Benin", "CI": "Cote d'Ivoire", "NE": "Niger",
}

# River name to Volta sub-basin. GRDC river strings are upper case and vary,
# so matching is on substrings.
_SUBBASIN_RULES = [
    ("BLACK VOLTA", "Black Volta"), ("MOUHOUN", "Black Volta"),
    ("SOUROU", "Black Volta"), ("BOUGOURIBA", "Black Volta"),
    ("WHITE VOLTA", "White Volta"), ("NAKAMBE", "White Volta"),
    ("NAKANBE", "White Volta"), ("RED VOLTA", "White Volta"),
    ("NAZINON", "White Volta"), ("SISSILI", "White Volta"),
    ("KULPAWN", "White Volta"), ("MORAGO", "White Volta"),
    ("OTI", "Oti"), ("PENDJARI", "Oti"), ("KARA", "Oti"),
    ("KOUMONGOU", "Oti"), ("KOMPIENGA", "Oti"), ("OUALE", "Oti"),
    ("VOLTA", "Lower Volta"),           # bare "VOLTA" only after the above
    ("DAKA", "Lower Volta"), ("AFRAM", "Lower Volta"), ("PRU", "Lower Volta"),
    ("SENE", "Lower Volta"), ("OBOSUM", "Lower Volta"),
]


def assign_subbasin(river: Optional[str]) -> Optional[str]:
    if not river:
        return None
    r = str(river).upper()
    for key, sub in _SUBBASIN_RULES:
        if key in r:
            return sub
    return None


def _parse_header(lines: Sequence[str]) -> dict:
    out: dict = {}
    for ln in lines:
        if not ln.startswith("#"):
            break
        body = ln.lstrip("#").strip()
        if ":" not in body:
            continue
        key, _, val = body.partition(":")
        key, val = key.strip(), val.strip()
        for pat, name in _HEADER_MAP.items():
            if key.startswith(pat):
                out[name] = val
                break
    return out


def _to_float(v) -> float:
    try:
        f = float(str(v).strip())
        return np.nan if f == MISSING else f
    except Exception:
        return np.nan


def read_station(path: str | Path) -> Tuple[dict, pd.DataFrame]:
    """
    Read one GRDC file. Returns (metadata dict, daily DataFrame).

    The DataFrame has columns [date, Q] with missing values (-999) as NaN and
    a continuous daily index spanning first to last record, so gaps are
    explicit rather than implied by absent rows. The quality-control stage
    depends on that.
    """
    path = Path(path)
    raw = path.read_text(encoding="latin-1", errors="replace").splitlines()

    meta = _parse_header(raw)
    meta["file"] = path.name

    for k in ("lat", "lon", "area_km2", "altitude_m"):
        meta[k] = _to_float(meta.get(k))
    for k in ("grdc_no", "next_downstream", "n_years_header", "n_data_lines"):
        try:
            meta[k] = int(float(str(meta.get(k, "")).strip()))
        except Exception:
            meta[k] = None
    if meta.get("next_downstream") in (0, -999, -9999):
        meta["next_downstream"] = None

    meta["country"] = COUNTRY_NAMES.get(str(meta.get("country_code", "")).strip(),
                                        meta.get("country_code"))
    meta["subbasin"] = assign_subbasin(meta.get("river"))

    unit = str(meta.get("unit_raw", "")).lower()
    # "m³/s" arrives with a latin-1 superscript; normalise by stripping it.
    meta["unit"] = "m3/s" if re.sub(r"[^a-z/]", "", unit) in ("m/s", "ms") else unit

    # --- data block ---
    try:
        start = next(i for i, ln in enumerate(raw) if ln.strip().startswith("YYYY-MM-DD")) + 1
    except StopIteration:
        return meta, pd.DataFrame(columns=["date", "Q"])

    dates, vals = [], []
    for ln in raw[start:]:
        if not ln.strip():
            continue
        parts = ln.split(";")
        if len(parts) < 3:
            continue
        dates.append(parts[0].strip())
        vals.append(_to_float(parts[-1]))

    df = pd.DataFrame({"date": pd.to_datetime(dates, errors="coerce"), "Q": vals})
    df = df.dropna(subset=["date"]).sort_values("date")

    if not df.empty:
        full = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
        df = df.set_index("date").reindex(full).rename_axis("date").reset_index()

    return meta, df


def read_folder(
    folder: str | Path, pattern: str = "*_Q_Day.Cmd.txt"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read every GRDC file in a folder.

    Returns (stations, discharge). `stations` is one row per station with the
    parsed header plus observed record statistics; `discharge` is long format
    [grdc_no, date, Q].
    """
    files = sorted(Path(folder).glob(pattern))
    metas, frames = [], []

    for f in files:
        meta, df = read_station(f)
        if meta.get("grdc_no") is None:
            continue
        if df.empty:
            meta.update(first_date=pd.NaT, last_date=pd.NaT, n_days=0,
                        n_obs=0, pct_nan=np.nan, mean_Q=np.nan)
            metas.append(meta)
            continue

        df.insert(0, "grdc_no", meta["grdc_no"])
        frames.append(df)

        q = df["Q"]
        meta.update(
            first_date=df["date"].min(), last_date=df["date"].max(),
            n_days=int(len(df)), n_obs=int(q.notna().sum()),
            pct_nan=float(q.isna().mean()),
            pct_zero=float((q == 0).mean()),
            mean_Q=float(q.mean()), max_Q=float(q.max()),
            min_Q=float(q.min()),
        )
        metas.append(meta)

    cols = ["grdc_no", "station_name", "river", "subbasin", "country",
            "country_code", "lat", "lon", "area_km2", "altitude_m",
            "next_downstream", "unit", "owner", "period_raw",
            "first_date", "last_date", "n_days", "n_obs", "pct_nan",
            "pct_zero", "mean_Q", "min_Q", "max_Q", "file"]
    stations = pd.DataFrame(metas)
    stations = stations.reindex(columns=[c for c in cols if c in stations.columns])
    stations = stations.sort_values("grdc_no").reset_index(drop=True)

    discharge = (pd.concat(frames, ignore_index=True)
                 if frames else pd.DataFrame(columns=["grdc_no", "date", "Q"]))
    return stations, discharge


# ---------------------------------------------------------------------------
# Connectivity from the downstream pointers
# ---------------------------------------------------------------------------
def build_connectivity(
    stations: pd.DataFrame, restrict_to: Optional[Sequence[int]] = None
) -> Tuple[Dict[int, List[int]], pd.DataFrame]:
    """
    Assemble the full upstream set for every station by following
    `next_downstream` pointers.

    A pointer may reference a station that is absent from the retained set,
    either because GRDC does not hold it or because it failed quality control.
    Such a gap would silently sever a chain, so the walk passes *through*
    missing stations: from each station we follow the pointer chain to the
    outlet and record every retained station encountered. Station A is
    upstream of station B whenever B lies on A's downstream path.

    Returns (connectivity dict, table). The table is written in the
    semicolon-separated form the propagation module expects, with `node_id`
    as the column name so the rest of the pipeline is unchanged.
    """
    nxt = {
        int(r.grdc_no): (int(r.next_downstream)
                         if r.next_downstream is not None and not pd.isna(r.next_downstream)
                         else None)
        for r in stations.itertuples(index=False)
    }
    keep = set(int(x) for x in (restrict_to if restrict_to is not None
                                else stations["grdc_no"]))

    upstream: Dict[int, List[int]] = {n: [] for n in keep}
    immediate: Dict[int, Optional[int]] = {n: None for n in keep}

    for n in keep:
        seen, cur, first = set(), nxt.get(n), True
        while cur is not None and cur not in seen:
            seen.add(cur)
            if cur in keep:
                upstream[cur].append(n)
                if first:
                    immediate[n] = cur
                    first = False
            cur = nxt.get(cur)

    rows = []
    for n in sorted(keep):
        ups = sorted(set(upstream[n]))
        rows.append({"node_id": n, "n_upstream": len(ups),
                     "immediate_downstream": immediate[n],
                     "upstream_chain": ",".join(map(str, ups))})

    table = pd.DataFrame(rows)
    conn = {int(r.node_id): [int(x) for x in r.upstream_chain.split(",") if x]
            for r in table.itertuples(index=False)}
    return conn, table


# ---------------------------------------------------------------------------
# Quality control, following the Ebro gate
# ---------------------------------------------------------------------------
def quality_control(
    stations: pd.DataFrame,
    discharge: pd.DataFrame,
    start: str,
    end: str,
    max_pct_nan: float = 0.30,
    max_nan_run_days: int = 365,
    max_zero_run_days: int = 240,
    min_years: int = 10,
) -> pd.DataFrame:
    """
    Apply the Ebro screening thresholds over a fixed analysis window.

    The Ebro gate is at most 30 per cent missing days, no gap beyond 365 days,
    and no zero-flow run beyond 120 days. The zero-flow threshold is relaxed
    to 240 days here, because extended zero flow is a climatological feature
    of the Sahelian Volta rather than a sensor failure; intermittency is
    handled instead by the seasonal-validity rule at the standardisation
    stage. Every criterion is reported separately so the screening can be
    audited and re-cut at a different threshold.
    """
    def max_run(mask: np.ndarray) -> int:
        best = run = 0
        for v in mask:
            run = run + 1 if v else 0
            best = max(best, run)
        return best

    s0, s1 = pd.Timestamp(start), pd.Timestamp(end)
    span_days = (s1 - s0).days + 1
    out = []

    for st in stations.itertuples(index=False):
        g = discharge[(discharge.grdc_no == st.grdc_no)
                      & discharge.date.between(s0, s1)]
        rec = {"grdc_no": st.grdc_no, "station_name": st.station_name,
               "river": st.river, "subbasin": st.subbasin,
               "country": st.country, "area_km2": st.area_km2}

        if g.empty:
            rec.update(n_days_in_window=0, pct_nan=1.0, max_nan_run=span_days,
                       max_zero_run=0, n_years=0, passes=False,
                       fail_reason="no data in window")
            out.append(rec)
            continue

        full = pd.date_range(s0, s1, freq="D")
        q = g.set_index("date")["Q"].reindex(full)
        isnan = q.isna().to_numpy()
        iszero = (q == 0).to_numpy()
        years = q.dropna().index.year.nunique()

        pct_nan = float(isnan.mean())
        nan_run = max_run(isnan)
        zero_run = max_run(iszero)

        fails = []
        if pct_nan > max_pct_nan:
            fails.append(f"missing {pct_nan:.0%} > {max_pct_nan:.0%}")
        if nan_run > max_nan_run_days:
            fails.append(f"gap {nan_run} d > {max_nan_run_days} d")
        if zero_run > max_zero_run_days:
            fails.append(f"zero-run {zero_run} d > {max_zero_run_days} d")
        if years < min_years:
            fails.append(f"{years} yr < {min_years} yr")

        rec.update(
            n_days_in_window=int(q.notna().sum()), pct_nan=round(pct_nan, 4),
            max_nan_run=int(nan_run), max_zero_run=int(zero_run),
            pct_zero=round(float(iszero.mean()), 4), n_years=int(years),
            mean_Q=round(float(q.mean()), 3) if q.notna().any() else np.nan,
            passes=not fails, fail_reason="; ".join(fails),
        )
        out.append(rec)

    return pd.DataFrame(out).sort_values(
        ["passes", "n_days_in_window"], ascending=[False, False]
    ).reset_index(drop=True)
