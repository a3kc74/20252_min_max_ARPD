"""
Build summary_final.csv from convergence CSVs in exp/results/exp3/.

For each file {instance}/{algo}_run{r:02d}.csv:
  - Find the row with the smallest fitness (best GHG emission)
  - Tie-break: smallest time
  - Write one output row to summary_final.csv

Convergence CSVs have columns: generation, time, best_solution, GHG, fitness.

Output: exp/summary_final.csv  (same directory as this script)
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

_EXP3_CANDIDATES = [
    ROOT / "exp" / "result"  / "exp3",
    ROOT / "exp" / "results" / "exp3",
]
EXP3_DIR = next((p for p in _EXP3_CANDIDATES if p.exists()), _EXP3_CANDIDATES[1])

OUTPUT = Path(__file__).resolve().parent / "summary_final.csv"

# Mapping: safe filename prefix → canonical algorithm name
# Safe names use underscores (hyphens are replaced in exp3 by algo.replace("-", "_"))
ALGO_MAP: dict[str, str] = {
    "VND":      "VND",
    "VNS":      "VNS",
    "LNS":      "LNS",
    "GA":       "GA",
    "MA":       "MA",
    "IMMA":     "IMMA",
    "LCB_IMMA": "LCB-IMMA",
}

_SCALE_RE = re.compile(r"^MTLARP(\d+)_")

OUTPUT_FIELDS = [
    "instance", "scale", "algorithm", "run_num",
    "fitness", "is_feasible", "final_generation", "runtime_s",
]

# ---------------------------------------------------------------------------


def _get_scale(inst_name: str) -> int:
    """Extract the first integer from 'MTLARP{P}_{D}_{K}_{i}' → P."""
    m = _SCALE_RE.match(inst_name)
    return int(m.group(1)) if m else 0


def _parse_filename(filename: str) -> tuple[str, int] | None:
    """
    Parse '{algo_safe}_run{r:02d}.csv' → (algo_canonical, run_num).
    Uses rfind("_run") so 'LCB_IMMA_run01' is handled correctly.
    """
    stem = filename[:-4] if filename.endswith(".csv") else filename
    idx = stem.rfind("_run")
    if idx == -1:
        return None
    algo_safe = stem[:idx]
    run_str   = stem[idx + 4:]
    if algo_safe not in ALGO_MAP:
        return None
    try:
        run_num = int(run_str)
    except ValueError:
        return None
    return ALGO_MAP[algo_safe], run_num


def _best_row(path: Path) -> dict | None:
    """
    Read a convergence CSV and return the best row (min fitness,
    tie-broken by min time).  Returns None if the file is empty
    or cannot be parsed.
    """
    rows: list[dict] = []
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    rows.append({
                        "generation": int(row.get("generation", 0)),
                        "time":       float(row.get("time", 0.0)),
                        "fitness":    float(row["fitness"]),
                    })
                except (KeyError, ValueError):
                    continue
    except OSError as e:
        print(f"[WARN] Cannot read {path}: {e}", file=sys.stderr)
        return None

    if not rows:
        return None

    min_fit    = min(r["fitness"] for r in rows)
    candidates = [r for r in rows if r["fitness"] == min_fit]
    return min(candidates, key=lambda r: r["time"])


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not EXP3_DIR.exists():
        print(f"Exp3 directory not found: {EXP3_DIR}")
        return

    rows_out: list[dict] = []
    skipped = 0

    for inst_dir in sorted(EXP3_DIR.iterdir()):
        if not inst_dir.is_dir():
            continue
        inst_name = inst_dir.name
        scale     = _get_scale(inst_name)

        for csv_file in sorted(inst_dir.glob("*.csv")):
            parsed = _parse_filename(csv_file.name)
            if parsed is None:
                continue
            algo, run_num = parsed

            best = _best_row(csv_file)
            if best is None:
                print(f"[WARN] Empty/unreadable, skipping: {csv_file.relative_to(ROOT)}")
                skipped += 1
                continue

            rows_out.append({
                "instance":         inst_name,
                "scale":            scale,
                "algorithm":        algo,
                "run_num":          run_num,
                "fitness":          f"{best['fitness']:.6f}",
                "is_feasible":      "",
                "final_generation": best["generation"],
                "runtime_s":        f"{best['time']:.2f}",
            })

    # Sort by instance → algorithm → run_num
    rows_out.sort(key=lambda r: (r["instance"], r["algorithm"], int(r["run_num"])))

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows → {OUTPUT}")
    if skipped:
        print(f"Skipped {skipped} empty/unreadable file(s).")


if __name__ == "__main__":
    main()
