# volta-drought-propagation

Code and derived data for

> Quist, I., Evi, M., Owusu, A.B., Quist, B.N.Y. Regime shift, rising runoff efficiency and sparse gauging: limits to testing drought propagation in a Sahel–savanna landscape.
> Submitted to *CATENA*.

Release v2.0.0. The archived version of this repository has the DOI given in `CITATION.cff`
once it is minted on Zenodo.

## What the study asks, and where each answer is computed

The study covers the Volta Basin above Akosombo: 18 GRDC gauges with daily discharge for
1961–2007, GPCC rainfall for 1951–2020 and 16 CMIP6 models for 2015–2100. It has four objectives.

| Objective | Main scripts | Derived tables | Article |
|---|---|---|---|
| (i) Date the shifts in rainfall, drought synchrony and runoff at fixed rainfall, and trace their consequence for the drought-index reference | `climate_trends_teleconnections.py`, `runoff_coefficient.py` | `output_climate/` | Section 3.1, Fig. 3, Tables S5–S7 |
| (ii) Identify the ocean–atmosphere modes that pace basin drought and its synchrony | `climate_trends_teleconnections.py`, `seasonal_correlations.py` | `output_climate/` | Section 3.2, Fig. 4, Tables S8–S10, Figs. S6–S8 |
| (iii) Establish what the gauge network can test, whether matches exceed synchrony, and whether per-link connectivity is equivalent to the Ebro's | `run_observational.py` and the scripts of stages 1–5 of `reproduce.sh` | `output_observational/`, `output_identifiability/`, `output_reanalysis/`, `output_rainfall/`, `output_landscape/`, `output_ml/` | Sections 3.3–3.6, Figs. 5–7, Table 2, Tables S11–S30 |
| (iv) Project drought risk with bias-adjusted CMIP6 output and the monitoring needed to test propagation | `fetch_cmip6.py`, `projection_cmip6.py`, `monitoring_horizon.py` | `output_projection/` | Section 3.7, Fig. 8, Tables S31–S34 |

`MANIFEST.md` maps every figure and table of the article and the Supplementary Material to the
file it is built from and the script that writes that file. `verify_manuscript_numbers.py`
checks the reported numbers one by one against those files.

## Quick start (no restricted data needed)

```bash
conda env create -f environment.yml && conda activate voltaprop    # or: pip install -r requirements.txt
python verify_manuscript_numbers.py          # every checked number (245) against the shipped tables
python -I tests/test_pipeline_synthetic.py   # SSI, events, matching, chains and null on a synthetic basin
python -I tests/test_supplementary.py        # supplementary analyses on synthetic data
python -I tests/test_fastmatch.py            # vectorised matcher = reference matcher on the shipped catalogue
jupyter nbconvert --to notebook --execute notebooks/key_results.ipynb --output key_results.ipynb
```

The figure scripts in `figures_pub/` redraw every figure except Figs. S5 and S19 from the shipped
tables; those two plot discharge and need the GRDC records (see `MANIFEST.md`).

## Continuous integration

Every push and pull request runs `.github/workflows/verify.yml` on GitHub: it checks the SHA-256 sum
of every shipped file against `CHECKSUMS.sha256`, recomputes all checked numbers of the article and
the Supplementary Material from the shipped tables (`verify_manuscript_numbers.py`, report attached
to the run), and runs the three test scripts. A green run means that the repository, the derived
tables and the article agree.

## Full reproduction

```bash
./reproduce.sh <folder with the GRDC *_Q_Day.Cmd.txt files>
```

`reproduce.sh` runs the whole analysis in order, from the raw inputs to the figures, and ends with
`verify_manuscript_numbers.py`. The GRDC daily discharge must be requested from the Global Runoff
Data Centre; every other input is open and is downloaded by the scripts (`DATA.md`). Network
access is needed for the downloads. `REPRODUCIBILITY.md` describes the stages, run times, seeds and
the expected tolerances.

## Repository layout

```
reproduce.sh                  the whole pipeline, stage by stage
verify_manuscript_numbers.py  reported numbers against the derived tables (report in verification/)
*.py                          one script per analysis stage (listed in reproduce.sh and MANIFEST.md)
voltaprop/                    package: GRDC reader, quality control, gap filling, SSI, events,
                              matching (propagation.py, fastmatch.py), nulls, parity checks
config/volta_config.yaml      analysis parameters (Supplementary Table S35)
data/                         redistributable inputs and metadata (DATA.md lists what is not shipped)
output_*/                     derived tables used in the article; no raw daily discharge
figures_pub/                  figure scripts, shared style, small caches, supplementary-table exports
figures/                      the figures under their article numbers (vector PDF)
manifest/                     tables.csv and figures.csv: provenance of every table and figure
notebooks/key_results.ipynb   the central matching results recomputed from the shipped tables
tests/                        synthetic-data tests and the matcher equivalence test
third_party/Code_v.1          files from the original Ebro study's release (MIT licence)
```

## Data and licences

Code is released under the MIT licence (`LICENSE`); derived data, tables and figures under
CC BY 4.0 (`LICENSE-DATA`). Third-party inputs keep their own terms, listed in `DATA.md`. GRDC daily
discharge is not redistributed: the release contains no daily or monthly discharge series, only derived quantities (standardised indices, event and link tables, annual runoff
coefficients, signatures and summary statistics).

## Citation

Cite the article and this archive (`CITATION.cff`). The event-matching framework is that of
Crespillo López, Vicente-Serrano and Gimeno (2026), *Journal of Hydrology* 680, 136544
(https://doi.org/10.1016/j.jhydrol.2026.136544); their code and catalogue are archived at
https://doi.org/10.5281/zenodo.22755834.

Contact: Ishmeal Quist, Nanjing University of Information Science and Technology
(quistishmeal@nuist.edu.cn).
