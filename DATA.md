# Data: sources, licences and what this repository contains

The release contains code, redistributable inputs and derived products. It contains no daily or
monthly discharge series from the GRDC. Every input that is not shipped is either
downloaded by the scripts or must be requested from its custodian, as listed below.

## Inputs

| Input | Used for | In this repository | How to obtain | Terms |
|---|---|---|---|---|
| GRDC daily discharge, 18 Volta stations, 1961–2007 | runoff at fixed rainfall (objective i), drought events and matching (iii), transfer model (iv) | No | Request the stations in `data/grdc_station_list.csv` from the GRDC Data Portal (https://portal.grdc.bafg.de); run `python prepare_grdc.py <export folder>`. The study used the export of 7 June 2026. | GRDC data policy: no redistribution of the data |
| GRDC station catalogue fields and record statistics | station table, maps | `data/grdc_stations_raw.csv`, `data/grdc_station_list.csv`, `data/grdc_qc_summary.csv` | written by `prepare_grdc.py` | GRDC; summary statistics only |
| Volta Basin boundary | maps, catchment rainfall | `data/GRDC_VOLTA_POLITICAL_TRANSBOUNDARY_BASIN.*` | GRDC Major River Basins of the World (GRDC, 2020) | GRDC; attribution |
| Ebro event catalogue (2,625 events) and 101 gauge links | Ebro comparison (objective iii) | `data/ebro/`, `third_party/Code_v.1/` | Crespillo López (2026), https://doi.org/10.5281/zenodo.22755834 | MIT licence |
| GEOGLOWS v2 hydrography (TDX-Hydro, VPU 110) | reanalysis node networks, channel distances | derived networks in `data/reanalysis/t1000`, `t2500`, `t5000`; reach table `data/reanalysis/reaches_volta.parquet` | `download_inputs.py` (anonymous S3, `s3://geoglows-v2/hydrography/`) | CC BY 4.0 (Hales et al., 2025) |
| GEOGLOWS v2 retrospective daily discharge | reanalysis network, benchmark, observing-system experiment | No (21 MB); derived index and event tables are shipped | `fetch_reanalysis.py` (anonymous, `s3://geoglows-v2/retrospective/daily.zarr`); provenance in `data/reanalysis/fetch_meta.json` | CC BY 4.0 |
| GPCC Full Data Monthly v2022, 0.5°, 1951–2020 | catchment rainfall, SPI (objectives i–iii), bias adjustment (iv) | No; derived catchment series in `output_climate/tables/` and `output_rainfall/tables/` | `download_inputs.py`, then `subset_gpcc.py` | DWD terms of use; cite Schneider et al. (2022), https://doi.org/10.5676/DWD_GPCC/FD_M_V2022_050 |
| Sahel precipitation index | trend tests and rainfall corroboration (objectives i, iii) | `data/external/sahelrain.data` | `download_inputs.py` (NOAA PSL) | JISAO (2019), https://doi.org/10.6069/H5MW2F2Q |
| Monthly climate indices: AMO (Kaplan SST, unsmoothed), Niño 3.4, DMI, TNA, TSA, PDO | teleconnections (objective ii) | `data/external/climate_indices/*.txt`, snapshots retrieved 26 October 2026 | NOAA PSL; URLs in `download_inputs.py` | US Government work |
| NOAA ERSSTv5 monthly SST | ATL3 index (objective ii) | No (161 MB); the ATL3 series is in `output_climate/tables/climate_indices_monthly.csv` | `download_inputs.py` (NOAA PSL copy, file dated 3 October 2026) | NOAA; no constraints on use |
| CMIP6 monthly pr, tas, mrro; 16 models; historical, SSP2-4.5, SSP5-8.5 | projections (objective iv) | No (156 MB); store paths and versions of all 144 subsets in `data/external/cmip6/cmip6_store_manifest.csv` | `fetch_cmip6.py` (Pangeo archive on Google Cloud, anonymous; pinned to the manifest) | CMIP6 terms of use of each modelling group |
| Natural Earth 1:110m countries | maps | `data/external/naturalearth_lowres/` | `download_inputs.py` | public domain |
| Köppen–Geiger classification (Rubel et al., 2017) | Fig. 1, climate shares | `figures_pub/cache_koppen.npz` (classes at basin grid points) | `kgcpy` | as distributed with kgcpy |
| HydroATLAS v1.0 upstream attributes (soil water content, aridity index, slope) | Table S29, Fig. S18 | derived table `figures_pub/supp_tables/TableS37.csv` | separate GIS workflow (see `MANIFEST.md`) | HydroSHEDS terms; attribution (Linke et al., 2019) |

## Derived products in this repository

All derived tables are in `output_*/`; `MANIFEST.md` lists which table or figure each one supports.
For the GRDC records, the release contains only quantities from which the daily series cannot be
reconstructed:

* the daily Standardised Streamflow Index (`output_observational/tables/SSI_daily.csv`, with the
  gap-filling tier of each day); the discharge column of the analysis table has been removed, and the
  fitted distribution parameters are not included (`SSI_fit_log.csv` holds only the selected family
  and goodness-of-fit statistics for each day-of-year pool);
* drought events, matched pairs, chains, opportunities and null-test outputs;
* annual runoff depth and runoff coefficient per gauge-year (`runoff_coefficient_by_year.csv`),
  hydrological signatures and station summary statistics (record length, mean, minimum and maximum
  discharge, station mean discharge in `figures_pub/cache_station_mean_q.csv`);
* out-of-fold predictions of the transfer model (`transfer_model_oof.csv`), without the observed
  monthly series, and the skill scores.

A leak check run on the release compares every shipped text table with the raw GRDC records
(station, date and value in any common format) and finds no match.

## References

Crespillo López, A., 2026. Tracking hydrological drought propagation in river networks: an
event-based framework (code), version v2.0. Zenodo. https://doi.org/10.5281/zenodo.22755834

GRDC, 2020. Major River Basins of the World. Global Runoff Data Centre, Federal Institute of
Hydrology (BfG), Koblenz.

Hales, R.C., Nelson, E.J., Souffront, M., Gutierrez, A.L., Prudhomme, C., Kopp, S., Ames, D.P.,
Williams, G.P., Jones, N.L., 2025. Advancing global hydrologic modeling with the GEOGloWS ECMWF
streamflow service. J. Flood Risk Manag. 18 (1), e12859. https://doi.org/10.1111/jfr3.12859

Huang, B., Thorne, P.W., Banzon, V.F., Boyer, T., Chepurin, G., Lawrimore, J.H., Menne, M.J.,
Smith, T.M., Vose, R.S., Zhang, H.-M., 2017. Extended Reconstructed Sea Surface Temperature,
version 5 (ERSSTv5): upgrades, validations, and intercomparisons. J. Clim. 30, 8179–8205.
https://doi.org/10.1175/JCLI-D-16-0836.1

JISAO, 2019. Sahel precipitation index. University of Washington; distributed by NOAA PSL.
https://doi.org/10.6069/H5MW2F2Q

Linke, S., Lehner, B., Ouellet Dallaire, C., Ariwi, J., Grill, G., Anand, M., Beames, P.,
Burchard-Levine, V., Maxwell, S., Moidu, H., Tan, F., Thieme, M., 2019. Global hydro-environmental
sub-basin and river reach characteristics at high spatial resolution. Sci. Data 6, 283.
https://doi.org/10.1038/s41597-019-0300-6

Rubel, F., Brugger, K., Haslinger, K., Auer, I., 2017. The climate of the European Alps: shift of
very high resolution Köppen–Geiger climate zones 1800–2100. Meteorol. Z. 26, 115–125.
https://doi.org/10.1127/metz/2016/0816

Schneider, U., Hänsel, S., Finger, P., Rustemeier, E., Ziese, M., 2022. GPCC Full Data Monthly
Product Version 2022 at 0.5°: Monthly land-surface precipitation from rain-gauges built on
GTS-based and historical data. Deutscher Wetterdienst. https://doi.org/10.5676/DWD_GPCC/FD_M_V2022_050
