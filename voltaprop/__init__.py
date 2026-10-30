"""
voltaprop -- event-based tracking of hydrological drought propagation on a
sparsely gauged river network (transboundary Volta Basin).

Implements the framework of Crespillo Lopez, A., Vicente-Serrano, S.M. and
Gimeno, L. (2026), "Tracking hydrological drought propagation in river
networks: an event-based framework", Journal of Hydrology 680, 136544,
https://doi.org/10.1016/j.jhydrol.2026.136544 (code: MIT licence,
https://doi.org/10.5281/zenodo.22755834), with the additions this study needs.

  grdc           GRDC daily files, station metadata and next-downstream connectivity
  gapfill        the published screening gate and gap-filling tiers
  ssi            Standardised Streamflow Index (six families, L-moments, minimum AIC,
                 +/-15-day pooling, zero-flow mass) with a seasonal-validity mask
  events         run-theory event detection on an unbroken calendar
  propagation    reference matcher and chain construction (Section 2.2.3)
  fastmatch      exact vectorised matcher with observing-opportunity accounting,
                 per-link probability and the availability-aware permutation null
  analysis       W x M sensitivity, published-design permutation null, lag relations
  supplementary  the supplementary analyses S1-S9 of the original study
  parity         every published quantity beside its Volta counterpart
  figures, figures_supp   figures in the layout of the original study

Author: Ishmeal Quist, Nanjing University of Information Science and
Technology.
"""

__version__ = "1.0.0"
__all__ = ["grdc", "gapfill", "ssi", "events", "propagation", "fastmatch", "analysis",
           "supplementary", "parity", "figures", "figures_supp"]

from . import (  # noqa: E402,F401
    analysis,
    events,
    figures,
    figures_supp,
    network,
    propagation,
    ssi,
    supplementary,
    transboundary,
    validation,
)
