#!/usr/bin/env python
"""Run the Operation Greens replication pipeline in order.

    python run_all.py                 full pipeline (raw data required in 01-data/)
    python run_all.py --figures-only  figures from the shipped derived files
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "02-analysis"
RAW = ROOT / "01-data" / "archive (4)" / "parquet"
GADM = ROOT / "01-data" / "geo" / "gadm41_IND_2.json"

FULL = [
    "volatility_analysis.py",
    "temporal_spatial_analysis.py",
    "dml_operation_greens.py",
    "table4_clustered_se.py",
    "revision_analysis.py",
    "visualizations.py",
    "figures_v2.py",
]
FIGURES = ["visualizations.py", "figures_v2.py"]


def main() -> int:
    figures_only = "--figures-only" in sys.argv[1:]
    steps = FIGURES if figures_only else FULL
    if not GADM.exists():
        sys.exit(f"Missing {GADM.relative_to(ROOT)} - see 01-data/README.md")
    if not figures_only:
        missing = [y for y in range(2013, 2025) if not (RAW / f"{y}.parquet").exists()]
        if missing:
            sys.exit(f"Missing raw parquet for {missing} - see 01-data/README.md")
    (ROOT / "03-drafts" / "figures").mkdir(parents=True, exist_ok=True)
    for script in steps:
        print(f"\n=== {script} ===", flush=True)
        rc = subprocess.call([sys.executable, script], cwd=ANALYSIS)
        if rc != 0:
            print(f"{script} failed with exit code {rc}")
            return rc
    print("\nAll steps completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
