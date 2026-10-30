# Publication figures

Run from this folder after the analysis stages of `../reproduce.sh` (or with the derived tables
shipped in `../output_*/`):

    for f in fig01_study_area.py fig02_framework.py fig02_event_example.py fig04_05_06.py \
             fig07_09_chains.py fig10_11.py figS_a.py figS_b.py figS_matching_rules.py \
             fig05_landscape_panels.py; do python "$f"; done
    python figs_climate_projection.py regime tele proj wtcall syncwtc lag transfer projmodels
    python collect_final_figures.py

Each script writes to `out/` a vector PDF (fonts embedded), a 1000 dpi LZW TIFF and a 600 dpi PNG.
File names in `out/` (and the figure numbers in the scripts' comments) keep the numbering of the
first draft; `collect_final_figures.py` copies the PDFs to `../figures/` under their numbers in the
article (its table `FIGURES`, also in `../MANIFEST.md`). Some scripts also draw figures of earlier
drafts that the article no longer contains; those are not copied.

Widths follow Elsevier artwork sizes (90, 140 or 190 mm); lettering is 6–7.5 pt Liberation Sans
(Arial metrics), at least 6 pt. Shared style and colour roles are in `pubstyle.py`: Ebro blue, Volta
orange and reanalysis aqua (distinguishable under the common colour-vision deficiencies), one
single-hue ramp for drought severity and one blue ramp for magnitude.

Distribution and uncertainty marks are in `pubviz.py`: raincloud plots (half density, one dot per
observation or model, median; Allen et al., 2021, Wellcome Open Research 4, 63), Wilkinson dot plots and
gradient intervals whose opacity follows the confidence curve of the estimate (Kay, IEEE Trans. Vis. Comput. Graph., VIS 2023,
the ggdist paper). No figure uses dashed or dotted lines: reference levels are solid hairlines or shaded
regions, and `pubstyle.save` refuses to export a figure that contains a dashed or dotted line.

Without the GRDC records, `fig02_event_example.py` (Fig. S5) stops with a message and
`figs_climate_projection.py transfer` (Fig. S19) is skipped, because both plot discharge; every other figure is
drawn from the shipped tables. `fig01_study_area.py` then reads the station means from
`cache_station_mean_q.csv`.

Auxiliary files

| File | Content |
|---|---|
| `cache_koppen.npz` | Köppen–Geiger classes at the basin grid points (kgcpy; Rubel et al., 2017) |
| `cache_station_mean_q.csv` | mean discharge of the 18 stations over non-imputed days (Fig. 1b) |
| `cache_s1.csv`, `cache_s4_*.csv` | overlap sensitivity and 1,999-permutation published-design null; recomputed by `figS_a.py` when missing |
| `supp_tables/` | `TableS37.csv` (HydroATLAS attribute correlations, from an earlier separate workflow) and `TableS64.csv` (storage-signature correlations, the same values as `../output_landscape/storage_persistence.csv`), read by `fig05_landscape_panels.py s19` for Fig. S18; `Fig05_chain_geometry_data.csv`, the data behind Fig. 5, written by `fig05_landscape_panels.py fig5` |
