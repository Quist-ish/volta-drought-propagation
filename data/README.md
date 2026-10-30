# data/

Redistributable inputs and metadata. `../DATA.md` lists every input of the study, its source and
terms, and what is not shipped.

| Path | Content |
|---|---|
| `grdc_station_list.csv` | GRDC numbers, names, rivers and coordinates of the stations to request |
| `grdc_stations_raw.csv` | catalogue fields and record statistics of the stations in the GRDC export (written by `prepare_grdc.py`) |
| `grdc_qc_summary.csv`, `pair_concurrency.csv` | record screening and concurrent-observation summaries |
| `GRDC_VOLTA_POLITICAL_TRANSBOUNDARY_BASIN.*` | basin boundary (GRDC Major River Basins of the World) |
| `ebro/` | Ebro event catalogue and the 101 gauge links (Crespillo López, 2026; MIT) |
| `reanalysis/` | TDX-Hydro reach table and the reanalysis node networks (1,000, 2,500 and 5,000 km² thresholds); `fetch_meta.json` records the GEOGLOWS retrieval |
| `external/sahelrain.data` | Sahel precipitation index (JISAO, NOAA PSL) |
| `external/climate_indices/` | NOAA PSL monthly indices used in the study (snapshots of 26 October 2026) |
| `external/cmip6/` | CMIP6 model and member selection, retrieval log, and the store path and version of each of the 144 subsets |
| `external/naturalearth_lowres/` | country borders (Natural Earth, public domain) |

**GRDC daily discharge is not in this folder.** Request daily discharge for the stations in
`grdc_station_list.csv` from the GRDC Data Portal (https://portal.grdc.bafg.de), unpack the
export and run `python prepare_grdc.py <folder with the *_Q_Day.Cmd.txt files>`, which writes
`grdc_discharge_raw.csv` (grdc_no, date, Q). That file is listed in `.gitignore` and must not be
committed or shared.
