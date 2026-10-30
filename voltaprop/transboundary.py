"""
Transboundary and regulation diagnostics.

These analyses have no Ebro counterpart and carry the Volta paper's claim to
originality. The Ebro sits inside one state; the Volta drains six, and roughly
two-thirds of the water that reaches Ghana's Akosombo reservoir is generated
in Burkina Faso, Mali and Cote d'Ivoire. An event-based propagation catalogue
therefore answers a question the Ebro application could not pose: when a
drought is registered at a downstream node in one country, where upstream, and
in whose territory, did the signal originate, and how long beforehand?

Two blocks are provided:

  1. Transboundary attribution -- classify every chain by the riparian states
     it spans, build the country-to-country transmission matrix, and compute
     the upstream-area-weighted contribution of each state to each chain.

  2. Regulation diagnostics -- the source paper found that the node immediately
     below the Yesa reservoir showed an anomalously long propagation lag,
     uncorrelated with hydraulic distance, and read that as an observational
     signature of reservoir operation. The Volta offers a stronger test: Bui
     was commissioned in 2013, inside the study period, so the same reach can
     be compared against itself before and after impoundment.

Author: I. Quist (NUIST)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


# ===========================================================================
# 1. TRANSBOUNDARY ATTRIBUTION
# ===========================================================================
def tag_pairs_with_geography(
    pairs: pd.DataFrame, nodes: pd.DataFrame
) -> pd.DataFrame:
    """Attach country, sub-basin and upstream area to both ends of each pair."""
    cols = ["node_id", "country", "subbasin", "uparea_km2"]
    meta = nodes[cols]

    out = pairs.merge(
        meta.add_prefix("origin_").rename(columns={"origin_node_id": "origin_node"}),
        on="origin_node",
        how="left",
    ).merge(
        meta.add_prefix("upstream_").rename(columns={"upstream_node_id": "upstream_node"}),
        on="upstream_node",
        how="left",
    )
    out["is_transboundary"] = out["origin_country"] != out["upstream_country"]
    out["country_pair"] = out["upstream_country"] + " -> " + out["origin_country"]
    return out


def country_transmission_matrix(
    tagged_pairs: pd.DataFrame, normalise: str = "none"
) -> pd.DataFrame:
    """
    Counts of matched upstream-downstream event pairs by (upstream country,
    origin country). `normalise` accepts 'none', 'row' or 'total'.
    """
    m = pd.crosstab(tagged_pairs["upstream_country"], tagged_pairs["origin_country"])
    if normalise == "row":
        m = m.div(m.sum(axis=1).replace(0, np.nan), axis=0).round(4)
    elif normalise == "total":
        m = (m / m.to_numpy().sum()).round(4)
    return m


def classify_chains(
    chains: pd.DataFrame, tagged_pairs: pd.DataFrame
) -> pd.DataFrame:
    """
    One row per chain, with the set of contributing states, a transboundary
    flag, the share of matched upstream nodes lying outside the origin state,
    and the mean lag split by domestic and foreign contributors. The last
    quantity is the operationally useful one: it states how much earlier a
    foreign headwater signal arrives than a domestic one.
    """
    tp = tagged_pairs.copy()
    tp["chain_id"] = tp["origin_node"].astype(str) + "_" + tp["origin_event_id"].astype(str)

    def agg(g: pd.DataFrame) -> pd.Series:
        foreign = g["is_transboundary"]
        return pd.Series(
            {
                "origin_country": g["origin_country"].iloc[0],
                "origin_subbasin": g["origin_subbasin"].iloc[0],
                "contributing_countries": ",".join(sorted(g["upstream_country"].dropna().unique())),
                "n_countries": int(g["upstream_country"].nunique()),
                "is_transboundary": bool(foreign.any()),
                "foreign_fraction": round(float(foreign.mean()), 4),
                "lag_mean_domestic": round(float(g.loc[~foreign, "lag_days"].mean()), 2)
                if (~foreign).any() else np.nan,
                "lag_mean_foreign": round(float(g.loc[foreign, "lag_days"].mean()), 2)
                if foreign.any() else np.nan,
                "severity_domestic": round(float(g.loc[~foreign, "upstream_severity"].sum()), 4),
                "severity_foreign": round(float(g.loc[foreign, "upstream_severity"].sum()), 4),
            }
        )

    summary = tp.groupby("chain_id").apply(agg, include_groups=False).reset_index()
    out = chains.merge(summary, on="chain_id", how="left")
    out["lead_time_gain_days"] = (
        out["lag_mean_domestic"] - out["lag_mean_foreign"]
    ).round(2)
    return out


def area_weighted_country_contribution(
    tagged_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Share of each chain's upstream drainage area contributed by each state,
    restricted to nodes actually matched into that chain. This separates a
    state that contributes many small headwater nodes from one contributing a
    single large tributary.
    """
    tp = tagged_pairs.copy()
    tp["chain_id"] = tp["origin_node"].astype(str) + "_" + tp["origin_event_id"].astype(str)

    g = (
        tp.groupby(["chain_id", "upstream_country"])["upstream_uparea_km2"]
        .sum()
        .rename("area_km2")
        .reset_index()
    )
    total = g.groupby("chain_id")["area_km2"].transform("sum")
    g["area_share"] = (g["area_km2"] / total).round(4)
    return g


def transboundary_significance(
    tagged_pairs: pd.DataFrame, n_permutations: int = 2000, seed: int = 42
) -> dict:
    """
    Is the observed share of transboundary matches higher than the share that
    would arise if drought events were merely synchronous across the region?

    The country labels of upstream nodes are permuted while the network
    topology and all event timing are held fixed, mirroring the logic of the
    network-permutation null test in Supplementary Section S4 of the source
    paper. A one-sided empirical p-value is returned.
    """
    rng = np.random.default_rng(seed)
    observed = float(tagged_pairs["is_transboundary"].mean())

    node_country = (
        tagged_pairs[["upstream_node", "upstream_country"]]
        .drop_duplicates()
        .set_index("upstream_node")["upstream_country"]
    )
    labels = node_country.to_numpy()
    origin_country = tagged_pairs["origin_country"].to_numpy()
    upstream_nodes = tagged_pairs["upstream_node"].to_numpy()
    index = {n: i for i, n in enumerate(node_country.index)}
    positions = np.array([index[n] for n in upstream_nodes])

    null = np.empty(n_permutations)
    for k in range(n_permutations):
        shuffled = rng.permutation(labels)
        null[k] = float(np.mean(shuffled[positions] != origin_country))

    p = float((np.sum(null >= observed) + 1) / (n_permutations + 1))
    return {
        "observed_transboundary_fraction": round(observed, 4),
        "null_mean": round(float(null.mean()), 4),
        "null_sd": round(float(null.std(ddof=1)), 4),
        "p_value_one_sided": round(p, 5),
        "n_permutations": n_permutations,
    }


# ===========================================================================
# 2. REGULATION DIAGNOSTICS
# ===========================================================================
def flag_regulated_nodes(
    nodes: pd.DataFrame,
    reservoirs: Sequence[dict],
    flowdir: np.ndarray,
    transform,
    influence_km: float = 150.0,
) -> pd.DataFrame:
    """
    Mark every node lying on the channel within `influence_km` downstream of a
    listed impoundment, recording the dam and its commissioning year.
    """
    from .network import haversine_km, rowcol_to_lonlat, trace_downstream

    out = nodes.copy()
    out["regulated_by"] = None
    out["reservoir_distance_km"] = np.nan
    out["reservoir_year"] = np.nan

    node_cells = {
        (int(r), int(c)): int(n)
        for r, c, n in zip(nodes["row"], nodes["col"], nodes["node_id"])
    }

    for res in reservoirs:
        if "row" not in res or "col" not in res:
            continue  # caller must snap dams to the grid first
        r0, c0 = int(res["row"]), int(res["col"])
        plon, plat = rowcol_to_lonlat(r0, c0, transform)
        dist = 0.0
        for cell in trace_downstream(r0, c0, flowdir):
            nlon, nlat = rowcol_to_lonlat(cell[0], cell[1], transform)
            dist += haversine_km(plon, plat, nlon, nlat)
            plon, plat = nlon, nlat
            if dist > influence_km:
                break
            if cell in node_cells:
                nid = node_cells[cell]
                m = out["node_id"] == nid
                closer = out.loc[m, "reservoir_distance_km"].isna().all() or (
                    dist < out.loc[m, "reservoir_distance_km"].iloc[0]
                )
                if closer:
                    out.loc[m, "regulated_by"] = res["name"]
                    out.loc[m, "reservoir_distance_km"] = round(dist, 2)
                    out.loc[m, "reservoir_year"] = res.get("year", np.nan)

    out["is_regulated"] = out["regulated_by"].notna()
    return out


def regulation_lag_contrast(
    pairs: pd.DataFrame, nodes: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare propagation lag at regulated against near-natural origin nodes.
    Mann-Whitney is used rather than a t-test because the lag distribution is
    strongly skewed; the rank-biserial correlation is reported as an effect
    size, since a p-value alone says nothing about magnitude.
    """
    reg = nodes[["node_id", "is_regulated", "regulated_by"]].rename(
        columns={"node_id": "origin_node"}
    )
    df = pairs.merge(reg, on="origin_node", how="left")

    a = df.loc[df["is_regulated"] == True, "lag_days"].dropna()       # noqa: E712
    b = df.loc[df["is_regulated"] == False, "lag_days"].dropna()      # noqa: E712
    if len(a) < 10 or len(b) < 10:
        return pd.DataFrame([{"note": "insufficient sample for contrast",
                              "n_regulated": len(a), "n_natural": len(b)}])

    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    rbc = 1 - (2 * u) / (len(a) * len(b))

    return pd.DataFrame(
        [
            {
                "n_regulated_pairs": len(a),
                "n_natural_pairs": len(b),
                "median_lag_regulated": round(float(a.median()), 2),
                "median_lag_natural": round(float(b.median()), 2),
                "mean_lag_regulated": round(float(a.mean()), 2),
                "mean_lag_natural": round(float(b.mean()), 2),
                "mannwhitney_U": float(u),
                "p_value": float(p),
                "rank_biserial_r": round(float(rbc), 4),
            }
        ]
    )


def impoundment_breakpoint_test(
    pairs: pd.DataFrame,
    nodes: pd.DataFrame,
    reservoir_name: str = "Bui",
    commission_year: int = 2013,
    buffer_years: int = 1,
) -> pd.DataFrame:
    """
    Before-and-after test at nodes below a dam commissioned mid-record.

    Bui (2013) is the natural candidate in the Volta: the same reach is
    observed under near-natural and regulated conditions within one homogeneous
    discharge product, which removes the between-site confounding that limits
    the equivalent Ebro analysis. Years within `buffer_years` of commissioning
    are dropped to exclude reservoir filling.

    Interpret with care. A shift in lag across 2013 is consistent with
    regulation but is not proof of it: the 2010s were also wetter than the
    1980s across much of West Africa, and GloFAS does not assimilate Bui's
    operating rules, so part of any signal may be model structure rather than
    river behaviour. Say so in the manuscript.
    """
    affected = nodes.loc[nodes["regulated_by"] == reservoir_name, "node_id"].tolist()
    if not affected:
        return pd.DataFrame([{"note": f"no nodes flagged below {reservoir_name}"}])

    df = pairs[pairs["origin_node"].isin(affected)].copy()
    df["year"] = pd.DatetimeIndex(df["origin_start"]).year

    before = df.loc[df["year"] < commission_year - buffer_years, "lag_days"].dropna()
    after = df.loc[df["year"] > commission_year + buffer_years, "lag_days"].dropna()

    if len(before) < 10 or len(after) < 10:
        return pd.DataFrame(
            [{"note": "insufficient sample either side of commissioning",
              "n_before": len(before), "n_after": len(after)}]
        )

    u, p = stats.mannwhitneyu(before, after, alternative="two-sided")
    return pd.DataFrame(
        [
            {
                "reservoir": reservoir_name,
                "commission_year": commission_year,
                "n_nodes_affected": len(affected),
                "n_pairs_before": len(before),
                "n_pairs_after": len(after),
                "median_lag_before": round(float(before.median()), 2),
                "median_lag_after": round(float(after.median()), 2),
                "delta_median_days": round(float(after.median() - before.median()), 2),
                "mannwhitney_U": float(u),
                "p_value": float(p),
            }
        ]
    )
