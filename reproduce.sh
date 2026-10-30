#!/usr/bin/env bash
# Reproduces every table, figure and number of the study from the raw inputs.
#
#   ./reproduce.sh <folder with the GRDC *_Q_Day.Cmd.txt files>
#
# Runs from the repository root; the permutation nulls dominate the run time (several hours
# on two cores). Network access is needed for the open inputs (download_inputs.py,
# fetch_reanalysis.py, fetch_cmip6.py). GRDC daily discharge cannot be redistributed; see DATA.md.
# Stages 1-6 overwrite the derived tables shipped in output_*/; stage 8 checks them against the
# values reported in the article.
set -euo pipefail
GRDC_DIR=${1:?"usage: ./reproduce.sh <GRDC export folder>"}
EXT=${EXT:-external}
export ORIGINAL_CODE=${ORIGINAL_CODE:-third_party/Code_v.1}
export MPLBACKEND=Agg

# 0. inputs: GRDC tables (local, not redistributed) and open inputs
python prepare_grdc.py "$GRDC_DIR"
python download_inputs.py --dest "$EXT"

# 1. observed network (GRDC): quality control, gap filling, SSI, events, matching, sensitivity,
#    null tests, published-paper parity and supplementary analyses
python run_observational.py
python run_observational_supplement.py
python gap_analysis.py
python published_design_checks.py
python crosscheck_original_code.py
python robustness_analyses.py
python geometry_null.py                   # onset-recovery classes of matched pairs against the permutation null
python zero_flow_sensitivity.py           # per-link estimate without the Bamboi/Pwalagu zero-flow years
python render_fig01_availability.py       # Koppen-Geiger shares of the basin (fig01_koppen_shares.json)

# 2. reanalysis node networks (GEOGLOWS v2 / TDX-Hydro) and discharge
python export_reach_table.py "$EXT/geoglows/streams_110.gpkg"
for t in 1000 2500 5000; do
  python build_reanalysis_network.py --streams "$EXT/geoglows/streams_110.gpkg" \
      --lakes "$EXT/geoglows/lakes_110.gpkg" --threshold-km2 $t --tag t$t
done
python fetch_reanalysis.py
python compute_reanalysis_ssi.py

# 3. storage signatures; observed-network checks, identifiability, reanalysis framework,
#    benchmark and observing-system experiment
python hydrological_signatures.py         # storage and intermittency signatures (derived, shareable)
python storage_persistence_test.py        # storage signatures against drought duration
python overlap_screen_topology.py
python pvalue_precision.py
python identifiability_analysis.py
python run_reanalysis.py
python benchmark_osse.py

# 4. rainfall corroboration and machine-learning reconstruction
python subset_gpcc.py "$EXT/gpcc"
python rainfall_corroboration.py
python ml_hybrid.py

# 5. trends and change points (objective i), teleconnections (ii), equivalence, common reference,
#    synthetic truth and channel distance (iii)
python climate_trends_teleconnections.py  # GPCC SPI, Pettitt/SNHT/Buishand, MK-HR, wavelets, event-rate GLM
python seasonal_correlations.py           # June-October rainfall and synchrony against seasonal indices
python runoff_coefficient.py              # runoff at fixed rainfall (fixed effects, gauge-clustered errors)
python reference_equivalence_simulation.py  # common 1975-1992 reference, TOST equivalence, synthetic truth, channel distance
python perlink_sensitivity.py             # stability, severity split and antecedent terciles of the per-link estimate
python year_shift_precision.py            # year-displacement synchrony test and precision of the per-link estimate
python ebro_link_parity.py                # released Ebro link table against the reconstruction from the catalogue
python sensitivity_extensions.py          # thinning summary, runoff interactions, AMO checks, threshold/no-fill/regime variants

# 6. CMIP6 projections and monitoring horizon (objective iv); the Pangeo archive is read anonymously,
#    pinned to the stores listed in data/external/cmip6/cmip6_store_manifest.csv
python fetch_cmip6.py
python projection_cmip6.py
python monitoring_horizon.py
python projection_post1970.py             # transfer model trained on 1970-2007 only (sensitivity)

# 7. publication figures (vector PDF, 1000 dpi TIFF, 600 dpi PNG in figures_pub/out),
#    then copied under their article numbers to figures/
(cd figures_pub && for f in graphical_abstract.py fig01_study_area.py fig02_framework_redraw.py fig02_event_example.py fig04_05_06.py \
    fig07_09_chains.py fig10_11.py figS_a.py figS_b.py figS_matching_rules.py fig05_landscape_panels.py; do
  python "$f"; done
  python figs_climate_projection.py regime tele proj wtcall syncwtc lag transfer projmodels
  python collect_final_figures.py)

# 8. compare the regenerated tables with the values reported in the article
python verify_manuscript_numbers.py
echo "done"
