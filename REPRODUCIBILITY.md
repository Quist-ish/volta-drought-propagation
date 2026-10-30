# Reproducing the study

This document is for the editor, the reviewers and anyone re-running the analysis of
*Regime shift, rising runoff efficiency and sparse gauging: limits to testing drought propagation in a Sahel–savanna landscape*. It describes three levels of checking, from a few minutes
without any restricted data to the full run from raw inputs.

## 1. Environment

```bash
conda env create -f environment.yml && conda activate voltaprop
# or, with Python 3.13 installed:
pip install -r requirements.txt
```

All results were produced with Python 3.13.16 and the versions pinned in `requirements.txt` on
Linux (x86-64). The pinned set resolves from PyPI as listed.

## 2. Level 1: check the reported numbers (about one minute, no restricted data)

```bash
python verify_manuscript_numbers.py
```

The script reads each checked number of the Abstract, Results and Discussion, and of the main
supplementary tables, from the derived tables shipped in `output_*/`, and compares it with the
value printed in the article at the printed precision. It writes
`verification/verification_report.md` and `.csv` and exits with status 1 if any check fails.
`MANIFEST.md` lists, for every table and figure, the derived file it is built from and the script
that writes that file.

## 3. Level 2: rerun the analyses that need no restricted data (minutes)

```bash
python -I tests/test_pipeline_synthetic.py   # SSI, events, matching, chains, null on a synthetic basin
python -I tests/test_supplementary.py        # supplementary analyses on synthetic data
python -I tests/test_fastmatch.py            # vectorised matcher = reference matcher (shipped catalogue)
python perlink_sensitivity.py                # Table S21, Section S12, Table S30
python year_shift_precision.py               # Section S9, Table S22
python ebro_link_parity.py                   # Section S9 (released Ebro files only)
python seasonal_correlations.py              # Table S8 (shipped catchment rainfall and indices)
jupyter nbconvert --to notebook --execute notebooks/key_results.ipynb --output key_results.ipynb
(cd figures_pub && python figs_climate_projection.py regime tele proj && python fig10_11.py)   # figures from shipped tables
```

These steps regenerate their tables from the shipped derived inputs and leave the numbers checked
by Level 1 unchanged. Every figure except Figs. S5 and S19 can be redrawn this way.

## 4. Level 3: full run from the raw inputs

```bash
./reproduce.sh <folder with the GRDC *_Q_Day.Cmd.txt files>
```

Inputs: GRDC daily discharge must be requested from the GRDC (see `DATA.md`); all other inputs are
open and are downloaded by `download_inputs.py`, `fetch_reanalysis.py` and `fetch_cmip6.py`, which
need network access (roughly 1 GB in total).

| Stage | Scripts | Article | Run time |
|---|---|---|---|
| 0. Inputs | `prepare_grdc.py`, `download_inputs.py` | Section 2.2 | depends on the connection |
| 1. Observed network: quality control, gap filling, SSI, events, matching, sensitivity, nulls, parity with the original study | `run_observational*.py`, `gap_analysis.py`, `published_design_checks.py`, `crosscheck_original_code.py`, `robustness_analyses.py`, `geometry_null.py`, `zero_flow_sensitivity.py`, `render_fig01_availability.py` | Sections 2.2, 2.3.1–2.3.4, 3.3–3.4 | stages 0–4 together: about 3 h on two cores |
| 2. Reanalysis networks and discharge | `export_reach_table.py`, `build_reanalysis_network.py`, `fetch_reanalysis.py`, `compute_reanalysis_ssi.py` | Section 2.2.2 | (included above) |
| 3. Storage signatures, identifiability, reanalysis framework, benchmark, observing-system experiment | `hydrological_signatures.py`, `storage_persistence_test.py`, `overlap_screen_topology.py`, `pvalue_precision.py`, `identifiability_analysis.py`, `run_reanalysis.py`, `benchmark_osse.py` | Sections 3.5–3.6 | (included above) |
| 4. Rainfall corroboration and reconstruction | `subset_gpcc.py`, `rainfall_corroboration.py`, `ml_hybrid.py` | Sections 3.3, 3.5 | (included above) |
| 5. Regime shift, teleconnections, equivalence and stability (objectives i–iii) | `climate_trends_teleconnections.py`, `seasonal_correlations.py`, `runoff_coefficient.py`, `reference_equivalence_simulation.py`, `perlink_sensitivity.py`, `year_shift_precision.py`, `ebro_link_parity.py` | Sections 3.1–3.4 | about 15 min (wavelet significance 14 min) |
| 6. Projections and monitoring horizon (objective iv) | `fetch_cmip6.py`, `projection_cmip6.py`, `monitoring_horizon.py` | Section 3.7 | about 13 min (CMIP6 retrieval 12 min) |
| 7. Figures | `figures_pub/*.py`, `collect_final_figures.py` | all figures | about 2.5 min |
| 8. Check | `verify_manuscript_numbers.py` | all checked numbers | about 1 s |

Stages 1–6 overwrite the derived tables shipped in `output_*/`; stage 8 compares the regenerated
tables with the article.

## 5. Determinism and expected tolerances

Every random procedure is seeded:

| Procedure | Seed |
|---|---|
| Availability-aware and published-design permutation nulls (1,999 permutations) | 42; 20260713 where the original study's seed is reproduced |
| Null p-values with 19,999 permutations; bootstraps; Ebro thinning and degradation draws | 2026 |
| Random-forest reconstruction (`ml_hybrid.py`) | 42 |
| Change-point permutation tests (1,999 permutations) | 2026 |
| Wavelet-coherence significance (200 AR(1) surrogate pairs; numpy global generator, reseeded for each pair of series) | 2026 |
| Common-reference bootstrap and synthetic-truth experiment (`reference_equivalence_simulation.py`, each part with its own generator) | 2026 |
| Transfer model (random forest, XGBoost) and other resampling in `projection_cmip6.py` | 2026 |
| Per-link sensitivity, year-displacement resampling and precision simulation | 2026 |

Rerunning stage 5 and stage 6 (except the CMIP6 download) on the release machine returned byte-identical
tables, with one exception: the out-of-fold transfer-model predictions differ in the 15th decimal
place because of multithreaded floating-point summation, and the skill table and every projection
derived from them are identical.

Expected tolerances when the environment differs:

* The SSI distribution family selected for a few day-of-year pools can change with another SciPy
  release; use the pinned versions to reproduce the 174 events exactly.
* Monte Carlo quantities are reproduced exactly with the shipped seeds, and
  `verify_manuscript_numbers.py` checks every reported number at its printed precision.
* Wavelet-coherence shares depend on the surrogate realisation; across three realisations they varied
  by up to 2 percentage points for basin SPI-12 and up to 7 points for synchrony at 8–32 years. The
  article reports the seeded realisation.
* External inputs are pinned where the provider allows it: CMIP6 stores by path and version
  (`data/external/cmip6/cmip6_store_manifest.csv`); NOAA PSL indices as shipped snapshots; ERSSTv5 by
  the file date (3 October 2026; later files revise recent months); GEOGLOWS v2 by retrieval date
  (`data/reanalysis/fetch_meta.json`, 14 October 2026); GRDC by the export date (7 June 2026).

## 6. What is not regenerated by this release

`MANIFEST.md` marks each table by status. One supplementary table, S29, combines storage-signature
correlations regenerated by `storage_persistence_test.py` with HydroATLAS attribute correlations
drawn from a derived table of an earlier, separate attribute-extraction workflow
(`figures_pub/supp_tables/TableS37.csv`); the extraction script is not part of this release. No
other table or figure depends on a separate workflow.

## 7. Restricted data

The release contains no GRDC daily or monthly discharge. Derived products that are shipped
(standardised indices without fitted parameters, events, link tables, annual runoff coefficients,
signatures and station summaries) do not allow the daily records to be reconstructed; `DATA.md`
lists them. Before the release was packaged, every shipped text table was compared with the raw GRDC
records (station, date and value in five numeric formats) and no record was found.
