"""
Standardised Streamflow Index (SSI) for the Transboundary Volta Basin.

Faithful port of `03_ssi_transformation.ipynb` from Crespillo Lopez et al.
(2026), Journal of Hydrology 680, 136544, with one substantive addition: a
seasonal-validity mask required by the Volta's monomodal, partly ephemeral
regime. Everything else (candidate distribution set, circular DOY +/-15-day
pooling, Shapiro-Wilks selection, zero-flow probability-mass adjustment)
reproduces the original exactly.

Why the mask is necessary
-------------------------
In the Ebro, every gauge carries flow on every day of the year, so a
distribution can be fitted to each DOY pool and the resulting index is
interpretable throughout the calendar. Across the Sahelian White Volta and
upper Black Volta, channels are dry for five to seven months in a normal
year. Fitting a distribution to a pool that is 80 per cent zeros produces an
index that flags "drought" on days when the absence of flow is the
climatological norm, inflating the event catalogue with meaningless episodes
and corrupting every downstream propagation statistic. The mask sets SSI to
NaN wherever the pooled sample cannot support a meaningful standardisation.
Masked days can neither open nor sustain a drought event.

Author: I. Quist (NUIST)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro
import lmoments3.distr as _ld

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Candidate distributions. Identical set and order to the Ebro implementation.
# ---------------------------------------------------------------------------
# Candidate families, parameter estimator and selection criterion follow
# Vicente-Serrano et al. (2025, Int. J. Climatol. 45, e8906), which is the
# method the Ebro study cites and does not itself restate. Six three-parameter
# families, fitted by L-moments, selected by AIC. Earlier versions of this
# module fitted Gamma/LogNormal/Weibull/PearsonIII/LogLogistic/GEV by maximum
# likelihood and selected on the Shapiro-Wilks W of the transformed series;
# all three of those choices departed from the published method and were
# corrected on 12 October 2026. See docs/METHOD_CONFORMANCE.md.
DISTRIBUTIONS: Dict[str, object] = {
    "GEV": _ld.gev,            # generalised extreme value
    "GLO": _ld.glo,            # generalised logistic (log-logistic family)
    "GPA": _ld.gpa,            # generalised Pareto
    "LNO": _ld.gno,            # generalised normal, the 3-parameter log-normal
    "PE3": _ld.pe3,            # Pearson type III
    "WEI": _ld.wei,            # Weibull
}


@dataclass
class SSIConfig:
    """Parameters controlling the SSI transformation."""

    doy_window_days: int = 15
    min_pool_size: int = 30
    min_positive_for_fit: int = 10
    zero_flow_adjustment: bool = True

    # --- Volta seasonal-validity mask -------------------------------------
    seasonal_validity: bool = True
    max_zero_fraction: float = 0.40
    min_pooled_median_q_m3s: float = 0.10

    # --- Coarse alternative used as a sensitivity test ---------------------
    restrict_to_season: bool = False
    season_months: Tuple[int, ...] = (4, 5, 6, 7, 8, 9, 10)

    # --- Common reference period (sensitivity; None = each node's own record)
    reference_years: Optional[Tuple[int, int]] = None

    distributions: Dict[str, object] = field(default_factory=lambda: dict(DISTRIBUTIONS))


# ---------------------------------------------------------------------------
# Helpers (unchanged from the Ebro code)
# ---------------------------------------------------------------------------
def doy_circular_distance(doy_array: np.ndarray, target_doy: int) -> np.ndarray:
    """Circular distance on a 365-day cycle, so the window wraps at year end."""
    diff = np.abs(doy_array.astype(int) - int(target_doy))
    return np.minimum(diff, 365 - diff)


def fit_and_score(dist, values: np.ndarray
                  ) -> Tuple[Optional[dict], float, float]:
    """
    Fit `dist` to `values` by L-moments and score it by AIC.

    Returns (params, aic, shapiro_w). On failure, (None, +inf, nan), so a
    failed family is never selected under a minimum-AIC rule.

    The log-likelihood is CDF-based rather than density-based, following
    Vicente-Serrano et al. (2025), who compare empirical against theoretical
    interval probabilities. This is the maximum-product-of-spacings
    construction: with the sample sorted, the likelihood contribution of each
    gap is the theoretical probability mass the fitted CDF places in it,

        logL = sum_i log( F(x_(i+1)) - F(x_(i)) )

    taken over the n+1 spacings formed by appending F = 0 and F = 1. It
    rewards a CDF that reproduces the empirical spacing of the order
    statistics and is better behaved than the density likelihood for
    heavy-tailed and bounded-support families, which matters here because GPA
    and WEI both have a boundary near the low-flow tail the index depends on.

    AIC = -2 logL + 2k, with k = 3 for every candidate, so the criterion
    reduces to a likelihood comparison; the penalty is retained so the
    expression stays correct if a two-parameter family is ever added.

    The Shapiro-Wilks W of the transformed series is computed and returned as
    a diagnostic only. It no longer selects the family. Vicente-Serrano et al.
    use it to assess the standardised output, and it is logged per node and
    day of year so the Supplementary Material can report normality of the
    fitted index.
    """
    try:
        params = dist.lmom_fit(values)
        if not params or not all(np.isfinite(list(params.values()))):
            return None, np.inf, np.nan

        xs = np.sort(np.asarray(values, dtype=float))
        F = dist.cdf(xs, **params)
        if not np.all(np.isfinite(F)):
            return None, np.inf, np.nan
        F = np.clip(F, 0.0, 1.0)

        # spacings, including the two boundary gaps
        spac = np.diff(np.concatenate(([0.0], F, [1.0])))
        spac = np.clip(spac, 1e-12, None)
        logL = float(np.sum(np.log(spac)))
        if not np.isfinite(logL):
            return None, np.inf, np.nan
        k = len(params)
        aic = -2.0 * logL + 2.0 * k

        # diagnostic only
        pit = np.clip(dist.cdf(np.asarray(values, dtype=float), **params),
                      1e-6, 1 - 1e-6)
        z = stats.norm.ppf(pit)
        w = np.nan
        if np.all(np.isfinite(z)):
            zz = z
            if len(zz) > 5000:
                zz = np.random.default_rng(42).choice(zz, 5000, replace=False)
            if len(zz) >= 3:
                w = float(shapiro(zz)[0])

        return params, float(aic), w
    except Exception:
        return None, np.inf, np.nan


# ---------------------------------------------------------------------------
# Seasonal-validity test [VOLTA]
# ---------------------------------------------------------------------------
def pool_is_valid(pool_vals: np.ndarray, cfg: SSIConfig) -> Tuple[bool, str]:
    """
    Decide whether a DOY pool can support a meaningful standardisation.

    Returns (is_valid, reason). The reason string is logged so the
    Supplementary Material can report exactly why each node-DOY was masked.
    """
    if len(pool_vals) < cfg.min_pool_size:
        return False, "pool_too_small"

    if not cfg.seasonal_validity:
        return True, "ok"

    zero_fraction = float(np.mean(pool_vals == 0))
    if zero_fraction > cfg.max_zero_fraction:
        return False, "zero_dominated"

    if float(np.median(pool_vals)) < cfg.min_pooled_median_q_m3s:
        return False, "median_below_floor"

    if np.count_nonzero(pool_vals > 0) < cfg.min_positive_for_fit:
        return False, "too_few_positive"

    return True, "ok"


# ---------------------------------------------------------------------------
# Core per-node computation
# ---------------------------------------------------------------------------
def compute_ssi_node(
    series: pd.Series,
    cfg: SSIConfig = SSIConfig(),
) -> Tuple[pd.Series, Dict[int, dict]]:
    """
    Compute daily SSI for one node.

    Parameters
    ----------
    series : pd.Series
        Daily discharge (m3 s-1) with a DatetimeIndex. NaNs permitted.
    cfg : SSIConfig

    Returns
    -------
    ssi : pd.Series aligned to the input index
    log : dict keyed by DOY, holding the selected family, all W statistics,
          the zero fraction, and the validity verdict.
    """
    doy_arr = series.index.dayofyear.values
    ssi = pd.Series(np.nan, index=series.index, name="SSI", dtype=float)
    log: Dict[int, dict] = {}

    for d in sorted(np.unique(doy_arr)):

        # (a) circular DOY pool across all years
        mask_pool = doy_circular_distance(doy_arr, d) <= cfg.doy_window_days
        if cfg.reference_years is not None:
            yrs = series.index.year.values
            mask_pool &= (yrs >= cfg.reference_years[0]) & (yrs <= cfg.reference_years[1])
        pool_vals = series[mask_pool].dropna().values

        valid, reason = pool_is_valid(pool_vals, cfg)
        if not valid:
            log[int(d)] = {
                "best": None,
                "W": {},
                "p0": float(np.mean(pool_vals == 0)) if len(pool_vals) else np.nan,
                "n_pool": int(len(pool_vals)),
                "valid": False,
                "reason": reason,
            }
            continue

        # (b) zero-flow probability mass; fit on positive values only
        p0 = float(np.mean(pool_vals == 0))
        fit_vals = pool_vals[pool_vals > 0] if p0 > 0 else pool_vals

        # (c) fit all six families by L-moments, select the minimum AIC
        best_name, best_params, best_aic = None, None, np.inf
        aic_scores: Dict[str, float] = {}
        sw_scores: Dict[str, float] = {}
        for name, dist in cfg.distributions.items():
            params, aic, w = fit_and_score(dist, fit_vals)
            aic_scores[name] = round(aic, 4) if np.isfinite(aic) else np.nan
            sw_scores[name] = round(w, 5) if np.isfinite(w) else np.nan
            if params is not None and aic < best_aic:
                best_aic, best_name, best_params = aic, name, params

        log[int(d)] = {
            "best": best_name,
            "AIC": aic_scores,          # selection criterion
            "best_AIC": round(best_aic, 4) if np.isfinite(best_aic) else np.nan,
            "W": sw_scores,             # diagnostic only, not used to select
            "best_W": sw_scores.get(best_name, np.nan),
            "p0": p0,
            "n_pool": int(len(pool_vals)),
            "valid": best_params is not None,
            "reason": "ok" if best_params is not None else "all_fits_failed",
        }
        if best_params is None:
            continue

        # (d) transform the actual observations for this DOY
        mask_doy = doy_arr == d
        actual = series[mask_doy].dropna()
        if actual.empty:
            continue

        best_dist = cfg.distributions[best_name]
        # lmoments3 fits return a parameter dict, not a positional tuple
        p = best_dist.cdf(actual.values, **best_params)

        if cfg.zero_flow_adjustment and p0 > 0:
            p = np.where(
                actual.values == 0,
                p0 / 2.0,                    # midpoint of the zero mass
                p0 + (1.0 - p0) * p,         # shifted CDF for positive flows
            )

        p = np.clip(p, 1e-6, 1 - 1e-6)
        ssi.loc[actual.index] = stats.norm.ppf(p)

    # Optional coarse seasonal restriction (sensitivity test only)
    if cfg.restrict_to_season:
        off_season = ~series.index.month.isin(cfg.season_months)
        ssi[off_season] = np.nan

    return ssi, log


def compute_ssi_network(
    df: pd.DataFrame,
    cfg: SSIConfig = SSIConfig(),
    node_col: str = "node_id",
    date_col: str = "date",
    q_col: str = "Q",
    progress: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run `compute_ssi_node` across every node in a long-format discharge table.

    Returns
    -------
    ssi_df : long-format table [node_id, date, Q, SSI]
    log_df : diagnostic table [node_id, doy, best_dist, p0, n_pool, valid,
             reason, W_<family> ...] -- report the `valid` column per node in
             the Supplementary Material.
    """
    try:
        from tqdm import tqdm
    except ImportError:  # keep the module importable without tqdm
        def tqdm(x, **kw):
            return x

    frames, log_rows = [], []
    nodes = df[node_col].unique()
    iterator = tqdm(nodes, desc="Computing SSI") if progress else nodes

    for node in iterator:
        sub = (
            df.loc[df[node_col] == node]
            .set_index(date_col)[q_col]
            .sort_index()
        )
        ssi, log = compute_ssi_node(sub, cfg)

        frames.append(
            pd.DataFrame(
                {node_col: node, date_col: sub.index, "Q": sub.values, "SSI": ssi.values}
            )
        )
        for doy, info in log.items():
            row = {
                node_col: node,
                "doy": doy,
                "best_dist": info["best"],
                "p0": info["p0"],
                "n_pool": info["n_pool"],
                "valid": info["valid"],
                "reason": info["reason"],
            }
            row.update({f"W_{k}": v for k, v in info["W"].items()})
            log_rows.append(row)

    ssi_df = pd.concat(frames, ignore_index=True)[[node_col, date_col, "Q", "SSI"]]
    log_df = pd.DataFrame(log_rows)
    return ssi_df, log_df


def validity_summary(log_df: pd.DataFrame, node_col: str = "node_id") -> pd.DataFrame:
    """
    Per-node share of the calendar on which SSI is defined. This table belongs
    in the Supplementary Material: it is the honest statement of how much of
    the Volta year the index can speak to, and it is the first thing a
    hydrological reviewer will look for.
    """
    out = (
        log_df.groupby(node_col)
        .agg(
            n_doy=("doy", "nunique"),
            n_valid=("valid", "sum"),
            mean_zero_fraction=("p0", "mean"),
        )
        .reset_index()
    )
    out["valid_fraction"] = (out["n_valid"] / out["n_doy"]).round(4)
    reasons = (
        log_df.loc[~log_df["valid"]]
        .groupby([node_col, "reason"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    return out.merge(reasons, on=node_col, how="left").fillna(0)
