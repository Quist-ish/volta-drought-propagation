#!/usr/bin/env python3
"""
Sensitivity of the transfer model and SSI-12 projections to the 1969 regime shift.

Reruns projection_cmip6.py with the transfer model trained only on months from 1970 onwards, so that
no gauge is fitted across the rainfall-runoff shift of Section 3.1 (Porga and Tiele are the only
projected gauges with pre-1970 records). Bias adjustment, CMIP6 inputs, skill rule and summaries
are unchanged. Writes output_projection/post1970/*.
Author: I. Quist (NUIST)
"""
import projection_cmip6 as PC

PC.TAB = PC.OUT / "post1970"
PC.TAB.mkdir(parents=True, exist_ok=True)
_obs = PC.observed_monthly
PC.observed_monthly = lambda: {g: m[m.index.year >= 1970] for g, m in _obs().items()}

if __name__ == "__main__":
    PC.main()
