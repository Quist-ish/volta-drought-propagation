# Vendored files from the original framework

These files are copied unchanged from the code released with

> Crespillo López, A., Vicente-Serrano, S.M., Gimeno, L., 2026. Tracking hydrological
> drought propagation in river networks: an event-based framework. Journal of Hydrology
> 680, 136544. https://doi.org/10.1016/j.jhydrol.2026.136544

Code: Crespillo López, A., 2026. Code_v.1 (version v2.0). Zenodo.
https://doi.org/10.5281/zenodo.22755834. MIT licence (see `LICENSE` in this folder).

They are used to cross-check this study's matcher and null test against the original
implementation (`crosscheck_original_code.py`, `robustness_analyses.py`) and as the Ebro
reference catalogue (`data/ebro/`). Set `ORIGINAL_CODE` to a full copy of the released code
to use that instead.

| File | Used for |
|---|---|
| `revision_analyses/propagation_null_core.py` | original matcher and network-label null |
| `revision_analyses/S4_permutation_null_test/permutation_test_summary.csv` | published Ebro null-test values |
| `revision_analyses/S5_detection_sensitivity/table3_events_corrected.csv` | Ebro event catalogue (2,625 events) |
| `revision_analyses/S3_lag_vs_distance/pairwise_lag_distance.csv` | Ebro gauge pairs (101 links) |
| `revision_analyses/S8_national_catalogue/chains_national_validation.csv` | Ebro chain sizes (Fig. 11b) |
