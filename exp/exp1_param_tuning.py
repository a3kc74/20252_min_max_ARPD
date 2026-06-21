"""
Experiment 1: LCB-IMMA Parameter Tuning — one parameter at a time.

Tests each LCB-IMMA constructor parameter independently while keeping all
other parameters at their defaults.  Termination: T_wall (wall-clock budget)
per run so comparisons across values are fair.

Parameter specs and default values are read from:
    exp/param_tunning_lcbimma.csv   (via lcbimma_params.py)

Parameters tested (as of current CSV):
  No. 01  alpha       LinUCB exploration coefficient           [0.5, 1.0, 2.0]
  No. 02  n_pop       Population size (4 islands × n_pop//4)  [20, 40, 60]
  No. 03  tau_base    Periodic migration interval (epochs)     [5, 10, 20]
  No. 04  delta_stag  Stagnation window triggering migration   [10, 20, 30]
  No. 05  lambda_pen  Reward penalty for infeasible migration  [0.05, 0.10, 0.20]
  No. 06  theta_0     Initial migration-filter threshold       [0.3, 0.5, 0.7]
  No. 07  theta_1     Final migration-filter threshold         [0.0, 0.1, 0.2]
  No. 08  T_0         Initial Metropolis temperature           [0.5, 1.0, 2.0]
  No. 09  rho         Geometric decay rate (temp + filter)     [0.95, 0.99, 0.999]
  No. 10  ils_iter    Inner ILS iterations for island I2       [5, 10, 20]

Output:
  exp/results/exp1/p{no:02d}_{name}/v{i:02d}_run{r:02d}.csv
  exp/results/exp1/summary.csv
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
EXP_DIR = Path(__file__).resolve().parent          # exp/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP_DIR))                   # make lcbimma_params importable

# Load parameter specs from CSV (single source of truth)
from lcbimma_params import DEFAULTS as _CSV_DEFAULTS, PARAM_SPECS  # noqa: E402

from mm_mt_dlarp.algorithms import LCBIMMASolver, SolverConfig  # noqa: E402
from mm_mt_dlarp.discretize import build_discrete_instance, initial_breakpoints  # noqa: E402
from mm_mt_dlarp.parser import parse_instance  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration — edit here
# ---------------------------------------------------------------------------

# Use a medium-scale instance for tuning (MTLARP5_5_5_1: 250 vertices, 15 req. edges, K=3)
INSTANCE_PATH = ROOT / "data" / "instances" / "MTLARP4_4_3_1.dat"

T_WALL_DEFAULT: float = 100    # wall-clock budget per run during tuning (s)
NUM_TRUCKS: int = 3             # all instances have 3 DEPOTS → use all

RUNS_PER_VALUE: int = 1

OUTPUT_DIR = ROOT / "exp" / "results" / "exp1"
OVERWRITE: bool = False
VERBOSE: bool = False

# Default LCB-IMMA parameters — from "Default for others" column of CSV.
# T_wall is a run-budget setting (not in CSV); appended here.
DEFAULTS: dict = {**_CSV_DEFAULTS, "T_wall": T_WALL_DEFAULT}

# ---------------------------------------------------------------------------

CONV_FIELDS = ["generation", "time", "best_solution", "objective", "fitness"]
SUMMARY_FIELDS = [
    "param_no", "param_name", "value_idx", "value",
    "run_num", "fitness", "is_feasible", "final_generation", "runtime_s",
]


def _solution_repr(sol) -> str:
    """Routing string: L{v}:[e+/-,...]|[...]||L{v}:...
    Each L is a truck (launch vertex), each [...] is one flight,
    +/- is traversal direction, || separates trucks.
    """
    parts = []
    for lv in sorted(sol.flights_by_launch):
        flights = sol.flights_by_launch[lv]
        flight_strs = []
        for f in flights:
            if not f.tasks:
                continue
            tasks_str = ",".join(
                f"{t.edge_id}+" if t.forward else f"{t.edge_id}-"
                for t in f.tasks
            )
            flight_strs.append(f"[{tasks_str}]")
        if flight_strs:
            parts.append(f"L{lv}:" + "|".join(flight_strs))
    return "||".join(parts) if parts else "empty"


def _write_conv_log(writer, conv_log) -> None:
    for gen, t, sol in conv_log:
        ghg = sum(sol.makespan_by_launch.values()) if sol.makespan_by_launch else sol.objective
        writer.writerow({
            "generation":    gen,
            "time":          f"{t:.4f}",
            "best_solution": _solution_repr(sol),
            "objective":           f"{ghg:.6f}",
            "fitness":       f"{sol.objective:.6f}",
        })


def _build_solver(instance_path: str, seed: int, lcb_kwargs: dict) -> LCBIMMASolver:
    raw = parse_instance(instance_path)
    base = raw.depot_vertices[0]
    discrete = build_discrete_instance(
        raw,
        base_vertex=base,
        launch_vertices=raw.depot_vertices,
        selected_breakpoints=initial_breakpoints(raw),
    )
    config = SolverConfig(
        num_trucks=NUM_TRUCKS,
        base_vertex=base,
        seed=seed,
        verbose=VERBOSE,
    )
    return LCBIMMASolver(discrete, config, **lcb_kwargs)


def _run_one(inst_path: str, lcb_kwargs: dict, seed: int, out_csv: Path) -> dict:
    solver = _build_solver(inst_path, seed, lcb_kwargs)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    solution, _ = solver.solve()
    runtime = time.perf_counter() - t0

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CONV_FIELDS)
        writer.writeheader()
        _write_conv_log(writer, solver.convergence_log)

    final_gen = solver.convergence_log[-1][0] if solver.convergence_log else 0
    is_feasible = solver.is_feasible_solution(solution)
    return {
        "fitness":          solution.objective,
        "is_feasible":      is_feasible,
        "final_generation": final_gen,
        "runtime_s":        runtime,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inst_path = str(INSTANCE_PATH.resolve())

    total_runs = sum(len(vs) for _, _, _, vs in PARAM_SPECS) * RUNS_PER_VALUE
    print(f"Instance    : {INSTANCE_PATH.stem}")
    print(f"T_wall      : {T_WALL_DEFAULT}s per run")
    print(f"Runs/value  : {RUNS_PER_VALUE}")
    print(f"Total runs  : {total_runs}")
    print(f"Output dir  : {OUTPUT_DIR}")
    print(f"Params CSV  : param_tunning_lcbimma.csv ({len(PARAM_SPECS)} parameters)")

    summary_path = OUTPUT_DIR / "summary.csv"
    new_header = not summary_path.exists() or summary_path.stat().st_size == 0
    with summary_path.open("a", newline="", encoding="utf-8") as sf:
        sw = csv.DictWriter(sf, fieldnames=SUMMARY_FIELDS)
        if new_header:
            sw.writeheader()

        for (param_no, param_name, kwarg_key, test_values) in PARAM_SPECS:
            param_dir = OUTPUT_DIR / f"p{param_no:02d}_{param_name}"
            param_dir.mkdir(parents=True, exist_ok=True)

            for vi, value in enumerate(test_values, start=1):
                # Build kwargs: all defaults, override the tested key
                lcb_kwargs = dict(DEFAULTS)
                lcb_kwargs[kwarg_key] = value

                for run_num in range(1, RUNS_PER_VALUE + 1):
                    out_csv = param_dir / f"v{vi:02d}_run{run_num:02d}.csv"
                    if out_csv.exists() and not OVERWRITE:
                        print(
                            f"  Skip p{param_no:02d} {param_name} "
                            f"v{vi} val={value} run{run_num}"
                        )
                        continue

                    result = _run_one(
                        inst_path, lcb_kwargs, seed=run_num - 1, out_csv=out_csv
                    )

                    sw.writerow({
                        "param_no":        param_no,
                        "param_name":      param_name,
                        "value_idx":       vi,
                        "value":           value,
                        "run_num":         run_num,
                        "fitness":         f"{result['fitness']:.6f}",
                        "is_feasible":     result["is_feasible"],
                        "final_generation": result["final_generation"],
                        "runtime_s":       f"{result['runtime_s']:.2f}",
                    })
                    sf.flush()

                    print(
                        f"  p{param_no:02d} {param_name:<12} "
                        f"v{vi}/{len(test_values)} val={value!s:<10} "
                        f"run{run_num}/{RUNS_PER_VALUE} "
                        f"| fitness={result['fitness']:.4f} "
                        f"| feasible={result['is_feasible']} "
                        f"| gen={result['final_generation']} "
                        f"| {result['runtime_s']:.1f}s"
                    )

    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
