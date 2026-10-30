#!/usr/bin/env python3
"""
Copy the figures drawn in out/ to ../figures/ under their numbers in the article and the
Supplementary Material (vector PDF).

File names in out/ keep the numbering of the first draft; FIGURES below is the map to the
final numbers (also listed in MANIFEST.md).
Author: I. Quist (NUIST)
"""
import shutil
from pathlib import Path

H = Path(__file__).resolve().parent
OUT, STATIC, DEST = H / "out", H / "static", H.parent / "figures"

# final number, final file name, file written by the figure scripts, generating script
FIGURES = [
    ("Fig. 1", "Fig01_study_area", "Fig01_study_area", "fig01_study_area.py"),
    ("Fig. 2", "Fig02_framework", "Fig02_framework_redraw", "fig02_framework_redraw.py"),
    ("Fig. 3", "Fig03_regime_shift", "Fig03_regime_shift", "figs_climate_projection.py regime"),
    ("Fig. 4", "Fig04_teleconnections", "Fig04_teleconnections", "figs_climate_projection.py tele"),
    ("Fig. 5", "Fig05_chain_geometry", "Fig05_chain_geometry", "fig05_landscape_panels.py fig5"),
    ("Fig. 6", "Fig06_identifiability", "Fig10_identifiability", "fig10_11.py"),
    ("Fig. 7", "Fig07_reanalysis", "Fig11_reanalysis", "fig10_11.py"),
    ("Fig. 8", "Fig08_projection", "Fig10_projection", "figs_climate_projection.py proj"),
    ("Fig. S1", "FigS01_availability", "FigS12_availability", "figS_a.py"),
    ("Fig. S2", "FigS02_homogeneity", "FigS10_homogeneity", "figS_a.py"),
    ("Fig. S3", "FigS03_detection_sensitivity", "FigS06_detection_sensitivity", "figS_a.py"),
    ("Fig. S4", "FigS04_event_summary", "Fig05_event_summary", "fig04_05_06.py"),
    ("Fig. S5", "FigS05_event_example", "Fig02_event_example", "fig02_event_example.py"),
    ("Fig. S6", "FigS06_wtc_all", "FigS_wtc_all", "figs_climate_projection.py wtcall"),
    ("Fig. S7", "FigS07_sync_wtc", "FigS_sync_wtc", "figs_climate_projection.py syncwtc"),
    ("Fig. S8", "FigS08_lagcorr", "FigS_lagcorr", "figs_climate_projection.py lag"),
    ("Fig. S9", "FigS09_matching_rules", "FigS_matching_rules", "figS_matching_rules.py"),
    ("Fig. S10", "FigS10_lag_relations", "FigS03_lag_relations", "figS_a.py"),
    ("Fig. S11", "FigS11_M_sensitivity", "FigS01_M_sensitivity", "figS_a.py"),
    ("Fig. S12", "FigS12_WM_sensitivity", "Fig04_sensitivity", "fig04_05_06.py"),
    ("Fig. S13", "FigS13_case1_bamboi", "Fig07_case1_bamboi", "fig07_09_chains.py"),
    ("Fig. S14", "FigS14_case2_nawuni", "Fig08_case2_nawuni", "fig07_09_chains.py"),
    ("Fig. S15", "FigS15_case3_porga", "Fig09_case3_porga", "fig07_09_chains.py"),
    ("Fig. S16", "FigS16_rainfall", "FigS14_rainfall", "figS_b.py"),
    ("Fig. S17", "FigS17_opportunities", "FigS11_opportunities", "figS_a.py"),
    ("Fig. S18", "FigS18_landscape_power", "FigS19_landscape_power", "fig05_landscape_panels.py s19"),
    ("Fig. S19", "FigS19_transfer", "FigS_transfer", "figs_climate_projection.py transfer"),
    ("Fig. S20", "FigS20_projmodels", "FigS_projmodels", "figs_climate_projection.py projmodels"),
    ("Graphical abstract", "GraphicalAbstract", "GraphicalAbstract", "graphical_abstract.py"),
]


def main():
    DEST.mkdir(exist_ok=True)
    missing = []
    for num, final, src, _ in FIGURES:
        for d, ext in ((OUT, ".pdf"), (STATIC, ".png"), (OUT, ".png")):
            f = d / f"{src}{ext}"
            if f.exists():
                shutil.copy2(f, DEST / f"{final}{ext}")
                break
        else:
            missing.append(f"{num} ({src})")
    print(f"{len(FIGURES) - len(missing)} of {len(FIGURES)} figures copied to {DEST}")
    if missing:
        print("missing:", ", ".join(missing))


if __name__ == "__main__":
    main()
