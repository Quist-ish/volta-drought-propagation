#!/usr/bin/env python3
"""
Remaining published analyses on the observed network: Section 3.1 parameter
sensitivity (750-point W x M grid, RSI, Kneedle), S1 event-level M
sensitivity, S1.4 gap tolerance, S5 detection sensitivity and S9 homogeneity.
Reads the outputs of run_observational.py.
"""
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from voltaprop.propagation import load_connectivity
from voltaprop import analysis as A, supplementary as S

T = Path("output_observational/tables")
ev = pd.read_csv(T / "SSI_drought_events.csv", parse_dates=["start_date", "end_date", "peak_date"])
ssi = pd.read_csv(T / "SSI_daily.csv", parse_dates=["date"])
conn = load_connectivity(T / "upstream_connectivity.csv")
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

log("3.1 sensitivity grid, 25 W x 30 M = 750 combinations")
grid = A.sensitivity_grid(ev, conn, node_col="node_id", progress=False)
grid.to_csv(T / "sensitivity_grid.csv", index=False)
log(f"  grid {grid.shape}, columns {list(grid.columns)}")
metrics = [c for c in grid.columns if c not in ("W", "M")]
rsi = []
for m in metrics:
    rsi.append({"metric": m,
                "RSI_W": A.relative_sensitivity_index(grid, m, "W"),
                "RSI_M": A.relative_sensitivity_index(grid, m, "M")})
rsi = pd.DataFrame(rsi); rsi.to_csv(T / "sensitivity_rsi.csv", index=False)
print(rsi.round(3).to_string(index=False))
g45 = grid[grid.W == 45].sort_values("M")
knees = {}
for m in metrics:
    try:
        y = g45[m].values.astype(float)
        if np.nanmax(y) - np.nanmin(y) > 0:
            knees[m] = A.kneedle_elbow(g45.M.values, y, curve="convex", direction="decreasing")
    except Exception as e:
        knees[m] = None
log(f"  Kneedle elbow in M at W=45: {knees}")
json.dump({k: (None if v is None else float(v)) for k, v in knees.items()},
          open(T / "sensitivity_knee.json", "w"), indent=2)

log("S1 event-level M sensitivity")
s1, s1pe = S.s1_event_level_m_sensitivity(ev, conn)
s1.to_csv(T / "tableS1_m_sensitivity.csv", index=False); print(s1.round(3).to_string(index=False))

log("S1.4 gap tolerance")
s14 = S.s1_4_gap_tolerance_analysis(ev, conn)
s14.to_csv(T / "tableS1_4_gap_tolerance.csv", index=False); print(s14.round(3).to_string(index=False))

log("S5 detection sensitivity")
s5, s5pn = S.s5_detection_sensitivity(ssi, progress=False)
s5.to_csv(T / "tableS5_detection_sensitivity.csv", index=False); print(s5.round(3).to_string(index=False))

log("S9 homogeneity screening")
s9 = S.s9_homogeneity_screening(ssi)
s9.to_csv(T / "tableS9_homogeneity.csv", index=False)
print(s9.round(3).to_string(index=False))
log("done")
