"""The vectorised matcher returns exactly the pipeline's pairs (Proposition 1)."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voltaprop import fastmatch as F
from voltaprop.propagation import PropagationConfig, run_propagation, load_connectivity

T = Path(__file__).resolve().parents[1] / "output_observational" / "tables"


def test_fastmatch_equals_pipeline():
    ev = pd.read_csv(T / "SSI_drought_events.csv", parse_dates=["start_date", "end_date"])
    conn = load_connectivity(T / "upstream_connectivity.csv")
    _, prop, _ = run_propagation(ev, conn, PropagationConfig(45, 5, 0), validate=False)
    for W in (0, 45, 120):
        pairs, _ = F.match(F.build_records(ev), conn, W=W, M=5)
        assert sorted(zip(pairs.origin_node, pairs.origin_eid, pairs.upstream_node, pairs.upstream_eid)) == \
            sorted(zip(prop.origin_node, prop.origin_event_id, prop.upstream_node, prop.upstream_event_id))


if __name__ == "__main__":
    test_fastmatch_equals_pipeline(); print("fastmatch: identical pairs at W = 0, 45, 120")
