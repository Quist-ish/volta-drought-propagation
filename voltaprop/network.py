"""
River network, virtual gauging nodes and upstream connectivity.

This module has no counterpart in the Ebro repository. Crespillo Lopez et al.
(2026) derived their connectivity table interactively in QGIS from 33 physical
CEDEX gauges, and stated in their Discussion that scripting this step was a
known outstanding task. The Volta has no comparable gauge network, so the node
set must be constructed on the MERIT Hydro drainage structure and the
connectivity derived programmatically. Doing so also closes the gap the source
paper identified.

Node placement follows three rules, applied in order:
  1. every real gauge (GRDC, Hydrological Services Department Ghana, DGRE
     Burkina Faso) that survives quality control, snapped to the derived
     channel;
  2. every confluence of two channels above the drainage-area threshold;
  3. regular spacing along long un-branched reaches, so that headwater-to-
     outlet chains are not dominated by a handful of junctions.

Connectivity is then obtained by tracing D8 flow directions downstream from
each node and recording every node encountered, which yields, for each node,
the complete set of nodes hydraulically upstream of it.

MERIT Hydro D8 encoding: 1 E, 2 SE, 4 S, 8 SW, 16 W, 32 NW, 64 N, 128 NE;
0 river mouth; -1 inland depression; 255 ocean.

Author: I. Quist (NUIST)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# D8 code -> (row offset, col offset)
D8_OFFSETS: Dict[int, Tuple[int, int]] = {
    1: (0, 1),     # E
    2: (1, 1),     # SE
    4: (1, 0),     # S
    8: (1, -1),    # SW
    16: (0, -1),   # W
    32: (-1, -1),  # NW
    64: (-1, 0),   # N
    128: (-1, 1),  # NE
}
SINK_CODES = {0, -1, 255}


@dataclass
class NetworkConfig:
    min_upstream_area_km2: float = 1000.0
    place_at_real_gauges: bool = True
    place_at_confluences: bool = True
    place_at_basin_outlet: bool = True
    place_along_channel: bool = True
    along_channel_spacing_km: float = 75.0
    node_merge_distance_km: float = 15.0
    target_n_nodes: int = 40
    max_n_nodes: int = 60
    gauge_snap_tolerance_km: float = 5.0


# ---------------------------------------------------------------------------
# Raster loading
# ---------------------------------------------------------------------------
def load_hydro_rasters(
    flowdir_path: str,
    uparea_path: str,
    boundary_path: Optional[str] = None,
):
    """
    Read MERIT Hydro flow direction and upstream area, optionally clipped to
    the basin boundary. Returns (flowdir, uparea, transform, crs).

    `uparea` is expected in km2. MERIT Hydro distributes it that way; if your
    copy is in m2, divide before calling.
    """
    import rasterio
    from rasterio.mask import mask as rio_mask

    if boundary_path is not None:
        import geopandas as gpd

        bnd = gpd.read_file(boundary_path)
        with rasterio.open(flowdir_path) as src:
            bnd = bnd.to_crs(src.crs)
            fd, transform = rio_mask(src, bnd.geometry, crop=True, filled=True, nodata=255)
            crs = src.crs
        with rasterio.open(uparea_path) as src:
            ua, _ = rio_mask(src, bnd.geometry, crop=True, filled=True, nodata=-9999.0)
        return fd[0].astype(np.int16), ua[0].astype(np.float32), transform, crs

    with rasterio.open(flowdir_path) as src:
        fd, transform, crs = src.read(1).astype(np.int16), src.transform, src.crs
    with rasterio.open(uparea_path) as src:
        ua = src.read(1).astype(np.float32)
    return fd, ua, transform, crs


# ---------------------------------------------------------------------------
# Channel network and confluences
# ---------------------------------------------------------------------------
def extract_channel_mask(uparea: np.ndarray, cfg: NetworkConfig) -> np.ndarray:
    return (uparea >= cfg.min_upstream_area_km2) & np.isfinite(uparea)


def downstream_cell(r: int, c: int, flowdir: np.ndarray) -> Optional[Tuple[int, int]]:
    code = int(flowdir[r, c])
    if code in SINK_CODES or code not in D8_OFFSETS:
        return None
    dr, dc = D8_OFFSETS[code]
    nr, nc = r + dr, c + dc
    if 0 <= nr < flowdir.shape[0] and 0 <= nc < flowdir.shape[1]:
        return nr, nc
    return None


def find_confluences(flowdir: np.ndarray, channel: np.ndarray) -> List[Tuple[int, int]]:
    """Channel cells receiving flow from two or more channel cells."""
    inflow = np.zeros(flowdir.shape, dtype=np.int8)
    rows, cols = np.nonzero(channel)
    for r, c in zip(rows, cols):
        d = downstream_cell(r, c, flowdir)
        if d is not None and channel[d]:
            inflow[d] = min(inflow[d] + 1, 127)
    rr, cc = np.nonzero((inflow >= 2) & channel)
    return list(zip(rr.tolist(), cc.tolist()))


def find_outlet(uparea: np.ndarray, channel: np.ndarray) -> Tuple[int, int]:
    masked = np.where(channel, uparea, -np.inf)
    return np.unravel_index(int(np.argmax(masked)), uparea.shape)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def rowcol_to_lonlat(r: int, c: int, transform) -> Tuple[float, float]:
    import rasterio

    lon, lat = rasterio.transform.xy(transform, r, c, offset="center")
    return float(lon), float(lat)


def haversine_km(lon1, lat1, lon2, lat2) -> float:
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def trace_downstream(
    r: int, c: int, flowdir: np.ndarray, max_steps: int = 10_000_000
) -> List[Tuple[int, int]]:
    """Ordered list of cells from (r, c) to the basin outlet, exclusive of start."""
    path: List[Tuple[int, int]] = []
    seen = set()
    cur = (r, c)
    for _ in range(max_steps):
        nxt = downstream_cell(cur[0], cur[1], flowdir)
        if nxt is None or nxt in seen:
            break
        path.append(nxt)
        seen.add(nxt)
        cur = nxt
    return path


# ---------------------------------------------------------------------------
# Gauge snapping
# ---------------------------------------------------------------------------
def snap_gauges_to_channel(
    gauges: pd.DataFrame,
    channel: np.ndarray,
    uparea: np.ndarray,
    transform,
    cfg: NetworkConfig,
    lon_col: str = "lon",
    lat_col: str = "lat",
) -> pd.DataFrame:
    """
    Move each gauge onto the nearest channel cell within the snap tolerance,
    preferring the cell whose upstream area best matches the reported
    catchment area where that attribute is supplied.
    """
    import rasterio

    rows, cols = np.nonzero(channel)
    ch_lon = np.empty(rows.size)
    ch_lat = np.empty(rows.size)
    for i, (r, c) in enumerate(zip(rows, cols)):
        ch_lon[i], ch_lat[i] = rowcol_to_lonlat(r, c, transform)

    out = []
    for g in gauges.itertuples(index=False):
        glon, glat = getattr(g, lon_col), getattr(g, lat_col)
        d = np.array([haversine_km(glon, glat, lo, la) for lo, la in zip(ch_lon, ch_lat)])
        within = np.nonzero(d <= cfg.gauge_snap_tolerance_km)[0]
        if within.size == 0:
            out.append({**g._asdict(), "snapped": False, "row": np.nan, "col": np.nan,
                        "snap_dist_km": np.nan, "uparea_km2": np.nan})
            continue

        reported = getattr(g, "area_km2", np.nan)
        if np.isfinite(reported) and reported > 0:
            cand_area = uparea[rows[within], cols[within]]
            pick = within[int(np.argmin(np.abs(cand_area - reported) / reported))]
        else:
            pick = within[int(np.argmin(d[within]))]

        out.append(
            {
                **g._asdict(),
                "snapped": True,
                "row": int(rows[pick]),
                "col": int(cols[pick]),
                "snap_dist_km": round(float(d[pick]), 3),
                "uparea_km2": float(uparea[rows[pick], cols[pick]]),
            }
        )
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Node construction
# ---------------------------------------------------------------------------
def build_nodes(
    flowdir: np.ndarray,
    uparea: np.ndarray,
    transform,
    cfg: NetworkConfig,
    gauges: Optional[pd.DataFrame] = None,
    subbasin_lookup=None,
    country_lookup=None,
) -> pd.DataFrame:
    """
    Assemble the node table.

    `subbasin_lookup` and `country_lookup` are optional callables taking
    (lon, lat) and returning a label; supply them from the basin and admin
    shapefiles so that every node carries its sub-basin and riparian state,
    which the transboundary analysis requires.
    """
    channel = extract_channel_mask(uparea, cfg)
    candidates: List[dict] = []

    if cfg.place_at_real_gauges and gauges is not None:
        for g in gauges.loc[gauges.get("snapped", True)].itertuples(index=False):
            candidates.append(
                {
                    "row": int(g.row),
                    "col": int(g.col),
                    "kind": "gauge",
                    "priority": 0,
                    "source_name": getattr(g, "name", None),
                    "grdc_no": getattr(g, "grdc_no", None),
                }
            )

    if cfg.place_at_basin_outlet:
        r, c = find_outlet(uparea, channel)
        candidates.append({"row": int(r), "col": int(c), "kind": "outlet",
                           "priority": 0, "source_name": "Basin outlet", "grdc_no": None})

    if cfg.place_at_confluences:
        for r, c in find_confluences(flowdir, channel):
            candidates.append({"row": r, "col": c, "kind": "confluence",
                               "priority": 1, "source_name": None, "grdc_no": None})

    if cfg.place_along_channel:
        for r, c in _spaced_channel_cells(flowdir, channel, uparea, transform, cfg):
            candidates.append({"row": r, "col": c, "kind": "reach",
                               "priority": 2, "source_name": None, "grdc_no": None})

    nodes = pd.DataFrame(candidates)
    if nodes.empty:
        raise RuntimeError("No candidate nodes produced; check the drainage-area threshold.")

    nodes["lon"], nodes["lat"] = zip(
        *[rowcol_to_lonlat(r, c, transform) for r, c in zip(nodes["row"], nodes["col"])]
    )
    nodes["uparea_km2"] = uparea[nodes["row"].to_numpy(), nodes["col"].to_numpy()]

    nodes = _deduplicate(nodes, cfg)
    nodes = _trim_to_target(nodes, cfg)

    nodes = nodes.sort_values("uparea_km2", ascending=False).reset_index(drop=True)
    nodes.insert(0, "node_id", np.arange(1, len(nodes) + 1))

    if subbasin_lookup is not None:
        nodes["subbasin"] = [subbasin_lookup(lo, la) for lo, la in zip(nodes["lon"], nodes["lat"])]
    if country_lookup is not None:
        nodes["country"] = [country_lookup(lo, la) for lo, la in zip(nodes["lon"], nodes["lat"])]

    return nodes


def _spaced_channel_cells(flowdir, channel, uparea, transform, cfg) -> List[Tuple[int, int]]:
    """Walk downstream from each headwater, dropping a node every N km."""
    inflow = np.zeros(flowdir.shape, dtype=np.int8)
    rows, cols = np.nonzero(channel)
    for r, c in zip(rows, cols):
        d = downstream_cell(r, c, flowdir)
        if d is not None and channel[d]:
            inflow[d] = min(inflow[d] + 1, 127)

    heads = [(int(r), int(c)) for r, c in zip(rows, cols) if inflow[r, c] == 0]
    placed: List[Tuple[int, int]] = []
    visited = set()

    for hr, hc in heads:
        acc = 0.0
        cur = (hr, hc)
        plon, plat = rowcol_to_lonlat(hr, hc, transform)
        for nxt in trace_downstream(hr, hc, flowdir):
            if not channel[nxt]:
                break
            nlon, nlat = rowcol_to_lonlat(nxt[0], nxt[1], transform)
            acc += haversine_km(plon, plat, nlon, nlat)
            plon, plat = nlon, nlat
            if acc >= cfg.along_channel_spacing_km:
                if nxt not in visited:
                    placed.append(nxt)
                    visited.add(nxt)
                acc = 0.0
            cur = nxt
            if cur in visited:
                break
    return placed


def _deduplicate(nodes: pd.DataFrame, cfg: NetworkConfig) -> pd.DataFrame:
    """Drop lower-priority nodes within the merge distance of a kept node."""
    nodes = nodes.sort_values(["priority", "uparea_km2"], ascending=[True, False])
    kept: List[dict] = []
    for n in nodes.to_dict("records"):
        if any(
            haversine_km(n["lon"], n["lat"], k["lon"], k["lat"]) < cfg.node_merge_distance_km
            for k in kept
        ):
            continue
        kept.append(n)
    return pd.DataFrame(kept)


def _trim_to_target(nodes: pd.DataFrame, cfg: NetworkConfig) -> pd.DataFrame:
    if len(nodes) <= cfg.max_n_nodes:
        return nodes
    keep = nodes[nodes["priority"] <= 1]
    if len(keep) >= cfg.target_n_nodes:
        return keep.nlargest(min(len(keep), cfg.max_n_nodes), "uparea_km2")
    extra = (
        nodes[nodes["priority"] == 2]
        .nlargest(cfg.target_n_nodes - len(keep), "uparea_km2")
    )
    return pd.concat([keep, extra], ignore_index=True)


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------
def derive_connectivity(
    nodes: pd.DataFrame, flowdir: np.ndarray
) -> Tuple[Dict[int, List[int]], pd.DataFrame]:
    """
    For each node, the full set of nodes hydraulically upstream of it.

    Obtained by tracing downstream from every node and recording which other
    nodes lie on its path; node A is upstream of node B exactly when B appears
    on A's downstream path.
    """
    cell_to_node = {
        (int(r), int(c)): int(nid)
        for r, c, nid in zip(nodes["row"], nodes["col"], nodes["node_id"])
    }

    upstream: Dict[int, List[int]] = {int(n): [] for n in nodes["node_id"]}
    downstream_of: Dict[int, Optional[int]] = {int(n): None for n in nodes["node_id"]}

    for r, c, nid in zip(nodes["row"], nodes["col"], nodes["node_id"]):
        nid = int(nid)
        first = True
        for cell in trace_downstream(int(r), int(c), flowdir):
            if cell in cell_to_node:
                other = cell_to_node[cell]
                if other == nid:
                    continue
                upstream[other].append(nid)
                if first:
                    downstream_of[nid] = other
                    first = False

    records = []
    for nid in sorted(upstream):
        ups = sorted(set(upstream[nid]))
        records.append(
            {
                "node_id": nid,
                "n_upstream": len(ups),
                "immediate_downstream": downstream_of[nid],
                "upstream_chain": ",".join(str(u) for u in ups),
            }
        )

    table = pd.DataFrame(records)
    conn = {int(r.node_id): [int(x) for x in r.upstream_chain.split(",") if x]
            for r in table.itertuples(index=False)}
    return conn, table


def along_channel_distance_km(
    node_a: int, node_b: int, nodes: pd.DataFrame, flowdir: np.ndarray, transform
) -> float:
    """
    Hydraulic (along-channel) distance from an upstream node to a downstream
    node, used for the lag-versus-distance analysis (Supplementary S3 in the
    source paper). Returns NaN where the two are not hydraulically connected.
    """
    idx = nodes.set_index("node_id")
    ra, ca = int(idx.loc[node_a, "row"]), int(idx.loc[node_a, "col"])
    rb, cb = int(idx.loc[node_b, "row"]), int(idx.loc[node_b, "col"])

    dist = 0.0
    plon, plat = rowcol_to_lonlat(ra, ca, transform)
    for cell in trace_downstream(ra, ca, flowdir):
        nlon, nlat = rowcol_to_lonlat(cell[0], cell[1], transform)
        dist += haversine_km(plon, plat, nlon, nlat)
        plon, plat = nlon, nlat
        if cell == (rb, cb):
            return round(dist, 3)
    return float("nan")


# ---------------------------------------------------------------------------
# Discharge extraction at nodes
# ---------------------------------------------------------------------------
def open_discharge_grids(path: str, discharge_var: Optional[str] = None):
    """
    Open the GloFAS discharge grids, whether GRIB or NetCDF, file or directory.

    `fetch_glofas.py` writes one GRIB per year because a failure should cost
    one year rather than the whole retrieval, so the ordinary case here is a
    directory of 46 files. A single file still works.

    GRIB2 is the format the EWDS form emits by default; its NetCDF4 option is
    labelled experimental. cfgrib reads GRIB through xarray but leaves the
    ECMWF dimension furniture in place (`step`, `surface`, `number`, and
    `valid_time` alongside `time`), so this normalises all of that to a plain
    (time, latitude, longitude) array before anything downstream sees it.

    Returns an xarray.DataArray sorted and de-duplicated on time.
    """
    import xarray as xr

    p = Path(path)
    if p.is_dir():
        files = sorted(
            [f for f in p.iterdir()
             if f.suffix.lower() in (".grib", ".grib2", ".grb", ".grb2", ".nc")]
        )
        if not files:
            raise FileNotFoundError(f"no GRIB or NetCDF files in {p}")
    else:
        files = [p]

    arrays = []
    for f in files:
        if f.suffix.lower() in (".grib", ".grib2", ".grb", ".grb2"):
            try:
                ds = xr.open_dataset(f, engine="cfgrib",
                                     backend_kwargs={"indexpath": ""})
            except Exception as e:
                raise RuntimeError(
                    f"could not read {f.name} as GRIB. cfgrib and its eccodes "
                    f"binary are required:  conda install -c conda-forge cfgrib "
                    f"eccodes   (original error: {e})"
                ) from e
        else:
            ds = xr.open_dataset(f)

        var = discharge_var
        if var is None or var not in ds.data_vars:
            cands = [v for v in ds.data_vars if "dis" in v.lower()]
            if not cands:
                cands = list(ds.data_vars)
            if not cands:
                raise ValueError(f"no data variables in {f.name}")
            var = cands[0]
        da = ds[var]

        # cfgrib leaves valid_time as the true timestamp when step is present.
        if "valid_time" in da.coords and "time" in da.dims:
            vt = da["valid_time"]
            if vt.ndim == 1 and vt.sizes.get("time", -1) == da.sizes["time"]:
                da = da.assign_coords(time=vt.values)
        for junk in ("step", "surface", "number", "heightAboveGround"):
            if junk in da.dims and da.sizes[junk] == 1:
                da = da.squeeze(junk, drop=True)
            elif junk in da.coords and da[junk].ndim == 0:
                da = da.drop_vars(junk)
        if "valid_time" in da.coords and "valid_time" not in da.dims:
            da = da.drop_vars("valid_time")
        arrays.append(da)

    out = arrays[0] if len(arrays) == 1 else xr.concat(arrays, dim="time")
    out = out.sortby("time")
    _, keep = np.unique(out["time"].values, return_index=True)
    if len(keep) != out.sizes["time"]:
        out = out.isel(time=np.sort(keep))
    return out


def extract_glofas_at_nodes(
    nodes: pd.DataFrame,
    glofas_path: str,
    start: str,
    end: str,
    discharge_var: Optional[str] = "dis24",
    uparea_match_tolerance: float = 0.25,
    glofas_uparea_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Pull a daily discharge series for every node from GloFAS.

    The nearest GloFAS cell is not always the right one: at 0.05 degrees the
    neighbouring cell may sit on a different river. Where a GloFAS upstream-area
    file is supplied, the routine searches a 3x3 neighbourhood and takes the
    cell whose upstream area is closest to the MERIT value, rejecting the node
    if the relative difference exceeds `uparea_match_tolerance`. Record the
    rejected nodes; they belong in the Supplementary Material.
    """
    import xarray as xr

    da = open_discharge_grids(glofas_path, discharge_var).sel(
        time=slice(start, end))

    ua_glofas = None
    if glofas_uparea_path is not None:
        ua_glofas = xr.open_dataset(glofas_uparea_path)
        ua_name = list(ua_glofas.data_vars)[0]
        ua_glofas = ua_glofas[ua_name]

    frames, rejected = [], []
    for n in nodes.itertuples(index=False):
        if ua_glofas is None:
            series = da.sel(longitude=n.lon, latitude=n.lat, method="nearest")
            chosen_dlon = chosen_dlat = 0.0
        else:
            best, best_err = None, np.inf
            for dlon in (-0.05, 0.0, 0.05):
                for dlat in (-0.05, 0.0, 0.05):
                    try:
                        a = float(
                            ua_glofas.sel(
                                longitude=n.lon + dlon, latitude=n.lat + dlat, method="nearest"
                            )
                        )
                    except Exception:
                        continue
                    a_km2 = a / 1e6 if a > 1e6 else a   # m2 -> km2 if needed
                    err = abs(a_km2 - n.uparea_km2) / max(n.uparea_km2, 1.0)
                    if err < best_err:
                        best, best_err = (dlon, dlat), err
            if best is None or best_err > uparea_match_tolerance:
                rejected.append({"node_id": n.node_id, "rel_area_error": best_err})
                continue
            chosen_dlon, chosen_dlat = best
            series = da.sel(
                longitude=n.lon + chosen_dlon, latitude=n.lat + chosen_dlat, method="nearest"
            )

        frames.append(
            pd.DataFrame(
                {
                    "node_id": n.node_id,
                    "date": pd.DatetimeIndex(series["time"].values).normalize(),
                    "Q": series.values.astype(float),
                }
            )
        )

    if rejected:
        pd.DataFrame(rejected).to_csv("data/nodes_rejected_uparea_mismatch.csv", index=False)

    return pd.concat(frames, ignore_index=True)
