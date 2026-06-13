"""
Check progress of Experiment 3 (exp3_solution_quality.py).

Checks two sources:
  1. Convergence CSV files: exp/results/exp3/{instance}/{algo}_run{r:02d}.csv
  2. Summary file:          exp/results/exp3/summary.csv

Output:
  - Progress overview
  - Detail per instance × algorithm
  - Discrepancy report (in summary but missing CSV, or vice-versa)
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — must match exp3_solution_quality.py
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

INSTANCES_DIR = ROOT / "data" / "instances"
ALGORITHMS = ["VND", "VNS", "LNS", "GA", "MA", "IMMA", "LCB-IMMA"]
RUNS: int = 20

# Accept both "result" and "results" directory names
_CANDIDATES = [
    ROOT / "exp" / "result"  / "exp3",
    ROOT / "exp" / "results" / "exp3",
]
RESULT_DIR = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[1])

# ---------------------------------------------------------------------------


def _collect_instances() -> list[str]:
    """Return sorted list of instance stem names from data/instances/."""
    return sorted(f.stem for f in INSTANCES_DIR.glob("MTLARP*.dat"))


def _conv_filename(algo: str, run: int) -> str:
    return f"{algo.replace('-', '_')}_run{run:02d}.csv"


def _load_summary_done() -> set[tuple[str, str, int]]:
    """Read summary.csv → set of (instance, algorithm, run_num) completed."""
    done: set[tuple[str, str, int]] = set()
    summary = RESULT_DIR / "summary.csv"
    if not summary.exists():
        return done
    with summary.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                done.add((row["instance"], row["algorithm"], int(row["run_num"])))
            except (KeyError, ValueError):
                pass
    return done


def _runs_to_ranges(runs: list[int]) -> str:
    """Convert [1,2,3,5,6,10] → '01-03, 05-06, 10'."""
    if not runs:
        return ""
    runs = sorted(runs)
    parts = []
    start = end = runs[0]
    for r in runs[1:]:
        if r == end + 1:
            end = r
        else:
            parts.append(f"{start:02d}-{end:02d}" if start != end else f"{start:02d}")
            start = end = r
    parts.append(f"{start:02d}-{end:02d}" if start != end else f"{start:02d}")
    return ", ".join(parts)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    instances = _collect_instances()
    if not instances:
        print("No MTLARP instances found in data/instances/!")
        return

    total_expected = len(instances) * len(ALGORITHMS) * RUNS

    # --- check convergence CSV files (primary source) ---
    file_done:    set[tuple[str, str, int]] = set()
    file_missing: list[tuple[str, str, int]] = []

    for inst_name in instances:
        inst_dir = RESULT_DIR / inst_name
        for algo in ALGORITHMS:
            for run in range(1, RUNS + 1):
                csv_path = inst_dir / _conv_filename(algo, run)
                key = (inst_name, algo, run)
                if csv_path.exists() and csv_path.stat().st_size > 0:
                    file_done.add(key)
                else:
                    file_missing.append(key)

    summary_done = _load_summary_done()
    in_summary_not_file = summary_done - file_done
    in_file_not_summary = file_done - summary_done

    # --- print overview ---
    sep = "=" * 72
    print(sep)
    print("  LCB-IMMA — EXPERIMENT 3 PROGRESS CHECK")
    print(sep)
    print(f"  Result dir  : {RESULT_DIR}")
    print(f"  Instances   : {len(instances)}")
    print(f"  Algorithms  : {ALGORITHMS}")
    print(f"  Runs/algo   : {RUNS}")
    print(f"  Total needed: {total_expected}")
    print()
    pct = 100.0 * len(file_done) / total_expected if total_expected else 0.0
    status = "COMPLETE" if not file_missing else "IN PROGRESS"
    print(f"  Status              : {status}")
    print(f"  Completed (CSV)     : {len(file_done):4d} / {total_expected}  ({pct:.1f}%)")
    print(f"  Missing (CSV)       : {len(file_missing):4d}")
    print(f"  In summary.csv      : {len(summary_done):4d}")
    if in_summary_not_file:
        print(f"  [!] In summary but CSV missing : {len(in_summary_not_file)}")
    if in_file_not_summary:
        print(f"  [!] Has CSV but not in summary : {len(in_file_not_summary)}")

    # --- per-instance detail table ---
    print()
    print(sep)
    print("  DETAIL: INSTANCE × ALGORITHM")
    print(sep)
    col_w = 10
    header = f"  {'Instance':<32}" + "".join(f"{a:>{col_w}}" for a in ALGORITHMS) + f"  {'Total':>8}"
    print(header)
    print("  " + "-" * (32 + col_w * len(ALGORITHMS) + 10))

    all_complete = True
    for inst_name in instances:
        row_parts = [f"  {inst_name:<32}"]
        row_done = 0
        for algo in ALGORITHMS:
            done_count = sum(
                1 for r in range(1, RUNS + 1) if (inst_name, algo, r) in file_done
            )
            row_done += done_count
            cell = "[OK]" if done_count == RUNS else f"{done_count}/{RUNS}"
            if done_count < RUNS:
                all_complete = False
            row_parts.append(f"{cell:>{col_w}}")
        total_cell = f"{row_done}/{RUNS * len(ALGORITHMS)}"
        row_parts.append(f"  {total_cell:>8}")
        print("".join(row_parts))

    if all_complete:
        print()
        print("  >>> All experiment runs are complete! <<<")

    # --- missing-run detail ---
    if file_missing:
        print()
        print(sep)
        print("  MISSING RUNS")
        print(sep)
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for inst_name, algo, run in sorted(file_missing):
            groups[(inst_name, algo)].append(run)
        current_inst = None
        for (inst_name, algo), runs in sorted(groups.items()):
            if inst_name != current_inst:
                print(f"\n  [{inst_name}]")
                current_inst = inst_name
            ranges = _runs_to_ranges(runs)
            print(f"    {algo:<12}  {len(runs):>2} missing  →  run {ranges}")
        print(f"\n  Total missing: {len(file_missing)}")

    # --- discrepancy report ---
    if in_summary_not_file:
        print()
        print(sep)
        print("  [!] IN SUMMARY.CSV BUT CSV FILE MISSING")
        print(sep)
        for inst_name, algo, run in sorted(in_summary_not_file):
            print(f"    {inst_name:<32}  {algo:<12}  run{run:02d}")


if __name__ == "__main__":
    main()
