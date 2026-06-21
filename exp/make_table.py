"""
Build aggregated comparison table from summary_final.csv.

Reads exp/summary_final.csv (produced by make_final_sum.py) and writes
exp/table.csv with per-(instance, scale, algorithm) statistics:

  min_objective          — best single-run objective
  mean±std_objective     — mean ± std over all runs
  min_runtime_s          — fastest run (seconds)
  mean±std_runtime_s     — mean ± std runtime

Usage:
  python exp/make_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("pandas is required: pip install pandas", file=sys.stderr)
    raise

# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH  = SCRIPT_DIR / "summary_final.csv"
OUT_PATH   = SCRIPT_DIR / "table.csv"

# ---------------------------------------------------------------------------


def mean_std(s: "pd.Series") -> str:
    return f"{s.mean():.4f} ± {s.std():.4f}"


def main() -> None:
    if not DATA_PATH.exists():
        print(f"[ERROR] summary_final.csv not found at {DATA_PATH}")
        print("Run make_final_sum.py first.")
        return

    df = pd.read_csv(DATA_PATH)

    # Ensure numeric types
    df["fitness"]    = pd.to_numeric(df["fitness"],    errors="coerce")
    df["runtime_s"]  = pd.to_numeric(df["runtime_s"],  errors="coerce")
    df["run_num"]    = pd.to_numeric(df["run_num"],     errors="coerce")

    agg = (
        df.groupby(["instance", "scale", "algorithm"])
        .agg(
            total_run        = ("run_num",  "max"),
            min_fitness      = ("fitness",  "min"),
            mean_std_fitness = ("fitness",  mean_std),
            min_runtime_s    = ("runtime_s", "min"),
            mean_std_runtime_s = ("runtime_s", mean_std),
        )
        .reset_index()
    )

    agg.columns = [
        "instance", "scale", "algorithm",
        "total_run",
        "min_fitness",      "mean±std_fitness",
        "min_runtime_s",    "mean±std_runtime_s",
    ]

    # Sort by scale (ascending) then instance then algorithm
    agg = agg.sort_values(["scale", "instance", "algorithm"]).reset_index(drop=True)

    agg.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(agg)} rows → {OUT_PATH}")
    print()
    print(agg.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
