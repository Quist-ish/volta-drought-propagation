#!/usr/bin/env python3
"""
Sensitivity of the observed per-link estimate to the zero-flow years at Bamboi (1531100,
1999-2007) and Pwalagu (1531550, 1988-1996), where zero flow on catchments above 60,000 km2
may be a recording artefact. Those station-days are treated as unobserved (SSI set to missing,
events starting in them dropped); everything else is unchanged.
Writes output_landscape/zero_flow_sensitivity.json.
Author: I. Quist (NUIST)
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from voltaprop import fastmatch as F
from voltaprop.propagation import load_connectivity

TO = Path("output_observational/tables")
ev = pd.read_csv(TO / "SSI_drought_events.csv", parse_dates=["start_date", "end_date"])
ssi = pd.read_csv(TO / "SSI_daily.csv", parse_dates=["date"])
conn = load_connectivity(TO / "upstream_connectivity.csv")
DROP = {1531100: (1999, 2007), 1531550: (1988, 1996)}


def run(ev, ssi):
    recs = F.build_records(ev, avail=F.availability_from_daily(ssi), nodes=list(conn))
    st = F.statistics(recs, conn)
    return {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in st.items()
            if k in ("cf_all", "cf_opp", "p_link", "n_connected", "n_pairs", "n_opportunities", "n_origins")}


base = run(ev, ssi)
ev2, ssi2 = ev.copy(), ssi.copy()
for g, (a, b) in DROP.items():
    m = (ssi2.node_id == g) & ssi2.date.dt.year.between(a, b)
    ssi2.loc[m, "SSI"] = np.nan
    ev2 = ev2[~((ev2.node_id == g) & ev2.start_date.dt.year.between(a, b))]
sens = run(ev2, ssi2)
out = {"dropped": {str(k): v for k, v in DROP.items()}, "events_removed": int(len(ev) - len(ev2)),
       "baseline": base, "without_zero_flow_years": sens}
Path("output_landscape").mkdir(exist_ok=True)
json.dump(out, open("output_landscape/zero_flow_sensitivity.json", "w"), indent=1)
print(json.dumps(out, indent=1))
