"""
Experiment 3: Solution Quality Comparison.

Algorithms compared (7 total):
  Baselines (single-solution, no population):
    VND     — Variable Neighbourhood Descent (no time cap)
    VNS     — Variable Neighbourhood Search  (no time cap)
    LNS     — Large Neighbourhood Search     (no time cap)

  Population-based ablation (isolates the benefit of each LCB-IMMA component):
    GA      — Standard Genetic Algorithm, no local search        (T_wall budget)
    MA      — Memetic Algorithm = GA + VND local search          (T_wall budget)
    IMMA    — Island-Model MA, 4 islands, RANDOM migration       (T_wall budget)
    LCB-IMMA— Island-Model MA, 4 islands, LinUCB migration       (T_wall budget)

Ablation ladder: GA < MA < IMMA < LCB-IMMA
  GA    vs MA      → benefit of local-search hybridisation
  MA    vs IMMA    → benefit of the island model
  IMMA  vs LCB-IMMA→ benefit of the adaptive LinUCB controller

All MTLARP benchmark instances × RUNS independent runs × 7 algorithms.
VND / VNS / LNS run to natural completion (no T_wall cap).
GA / MA / IMMA / LCB-IMMA use a wall-clock budget scaled by instance scale.

Scale → T_wall mapping (seconds):
  P = 4  → 30 s    (MTLARP4_*)
  P = 5  → 60 s    (MTLARP5_*)
  P = 6  → 120 s   (MTLARP6_*)
  P = 7  → 180 s   (MTLARP7_*)
  P = 8  → 240 s   (MTLARP8_*)
  P = 10 → 300 s   (MTLARP10_*)

Output:
  exp/results/exp3/summary.csv
    — one row per (instance, algorithm, run)
  exp/results/exp3/{instance_name}/{algo}_run{r:02d}.csv
    — convergence log: generation, time, best_solution, objective, fitness
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

from lcbimma_params import DEFAULTS as _LCB_DEFAULTS  # noqa: E402

from mm_mt_dlarp.algorithms import (  # noqa: E402
    GASolver, IMMASolver, LCBIMMASolver, LNSSolver,
    MASolver, SolverConfig, VNDSolver, VNSSolver,
)
from mm_mt_dlarp.discretize import build_discrete_instance, initial_breakpoints  # noqa: E402
from mm_mt_dlarp.parser import parse_instance  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration — edit here
# ---------------------------------------------------------------------------

INSTANCES_DIR = ROOT / "data" / "instances"
NUM_TRUCKS: int = 3      # all MTLARP instances have exactly 3 DEPOTS

# Wall-clock budget for population-based algorithms keyed by scale integer
# (first number in the MTLARP filename, which proxies for problem complexity).
TIME_LIMITS_POP: dict[int, float] = {
    4:   100,
    5:   600,
    6:  1200,
    7:  1800,
    8:  2400,
    10: 3000,
}
DEFAULT_TIME_LIMIT_POP: float = 120.0

# 7 algorithms: 3 single-solution baselines + 4 population-based (ablation ladder)
ALGORITHMS = ["VND", "VNS", "LNS", "GA", "MA", "IMMA", "LCB-IMMA"]
RUNS: int = 1

OUTPUT_DIR = ROOT / "exp" / "results" / "exp3"
OVERWRITE: bool = False
VERBOSE: bool = False

SAVE_CONVERGENCE_LOG: bool = True

# LCB-IMMA / IMMA hyperparameters — loaded from "Default for others" column of
# param_tunning_lcbimma.csv via lcbimma_params.py.
# After Experiment 1 (param tuning), update that CSV column and re-run.
# T_wall is NOT here; it is added per-instance in _build_solver().
LCB_KWARGS: dict = dict(_LCB_DEFAULTS)   # shared by IMMA and LCB-IMMA

# GA / MA hyperparameters (mirror LCB_KWARGS where applicable)
GA_KWARGS: dict = dict(
    n_pop=LCB_KWARGS["n_pop"],   # same total population as LCB-IMMA / IMMA
    tournament_k=3,
    elite_frac=0.1,
    mutation_rate=0.3,           # same as LCBIMMASolver._mutate rate
)
# MASolver inherits from GASolver → same kwargs

# VNS / LNS default constructor kwargs
VNS_KWARGS: dict = dict(k_max=4, max_iter=100)
LNS_KWARGS: dict = dict(destroy_frac=0.3, max_iter=200)

# ---------------------------------------------------------------------------

SUMMARY_FIELDS = [
    "instance", "scale", "algorithm", "run_num",
    "fitness", "is_feasible", "final_generation", "runtime_s",
]
CONV_FIELDS = ["generation", "time", "best_solution", "objective", "fitness"]


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


def _write_conv_log(writer, conv_log, fallback_sol, fallback_runtime) -> None:
    if conv_log:
        for gen, t, sol in conv_log:
            ghg = sum(sol.makespan_by_launch.values()) if sol.makespan_by_launch else sol.objective
            writer.writerow({
                "generation":    gen,
                "time":          f"{t:.4f}",
                "best_solution": _solution_repr(sol),
                "objective":           f"{ghg:.6f}",
                "fitness":       f"{sol.objective:.6f}",
            })
    else:
        ghg = sum(fallback_sol.makespan_by_launch.values()) if fallback_sol.makespan_by_launch else fallback_sol.objective
        writer.writerow({
            "generation":    0,
            "time":          f"{fallback_runtime:.4f}",
            "best_solution": _solution_repr(fallback_sol),
            "objective":           f"{ghg:.6f}",
            "fitness":       f"{fallback_sol.objective:.6f}",
        })


def _instance_scale(stem: str) -> int:
    """Parse first integer from 'MTLARP{P}_{D}_{K}_{i}' → P (scale proxy)."""
    try:
        return int(stem.replace("MTLARP", "").split("_")[0])
    except (ValueError, IndexError):
        return 0


def _collect_instances() -> list[tuple[int, str]]:
    """Return sorted (scale, inst_path) pairs for all MTLARP*.dat files."""
    entries = []
    for f in INSTANCES_DIR.glob("MTLARP*.dat"):
        scale = _instance_scale(f.stem)
        entries.append((scale, str(f)))
    return sorted(entries, key=lambda x: (x[0], x[1]))


def _build_solver(inst_path: str, algo_name: str, seed: int, t_wall: float):
    """Construct the appropriate solver for the given algorithm.

    Population-based algorithms (GA, MA, IMMA, LCB-IMMA) receive a T_wall
    budget; single-solution algorithms (VND, VNS, LNS) run to completion.
    """
    raw = parse_instance(inst_path)
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

    # ── Single-solution baselines (no T_wall) ────────────────────────────────
    if algo_name == "VND":
        return VNDSolver(discrete, config)
    if algo_name == "VNS":
        return VNSSolver(discrete, config, **VNS_KWARGS)
    if algo_name == "LNS":
        return LNSSolver(discrete, config, **LNS_KWARGS)

    # ── Population-based ablation ladder (all use T_wall) ────────────────────
    if algo_name == "GA":
        return GASolver(discrete, config, **{**GA_KWARGS, "T_wall": t_wall})
    if algo_name == "MA":
        return MASolver(discrete, config, **{**GA_KWARGS, "T_wall": t_wall})
    if algo_name == "IMMA":
        return IMMASolver(discrete, config, **{**LCB_KWARGS, "T_wall": t_wall})
    if algo_name == "LCB-IMMA":
        return LCBIMMASolver(discrete, config, **{**LCB_KWARGS, "T_wall": t_wall})

    raise ValueError(f"Unknown algorithm: {algo_name}")


def _run_one(
    inst_path: str,
    algo_name: str,
    seed: int,
    t_wall: float,
    conv_csv: Path | None,
) -> dict:
    solver = _build_solver(inst_path, algo_name, seed, t_wall)

    t0 = time.perf_counter()
    solution, _ = solver.solve()
    runtime = time.perf_counter() - t0

    is_feasible = solver.is_feasible_solution(solution)

    conv_log = getattr(solver, "convergence_log", None)
    final_gen = conv_log[-1][0] if conv_log else 0

    if conv_csv is not None:
        conv_csv.parent.mkdir(parents=True, exist_ok=True)
        with conv_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CONV_FIELDS)
            writer.writeheader()
            _write_conv_log(writer, conv_log, solution, runtime)

    return {
        "fitness":          solution.objective,
        "is_feasible":      is_feasible,
        "final_generation": final_gen,
        "runtime_s":        runtime,
    }


def _load_done_set(summary_path: Path) -> set[tuple[str, str, int]]:
    done: set[tuple[str, str, int]] = set()
    if not summary_path.exists():
        return done
    with summary_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                done.add((row["instance"], row["algorithm"], int(row["run_num"])))
            except (KeyError, ValueError):
                pass
    return done


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_instances = _collect_instances()
    total = len(all_instances) * len(ALGORITHMS) * RUNS
    print(f"Instances   : {len(all_instances)}")
    print(f"Algorithms  : {ALGORITHMS}")
    print(f"Runs each   : {RUNS}")
    print(f"Total runs  : {total}")
    print(f"Output dir  : {OUTPUT_DIR}")

    summary_path = OUTPUT_DIR / "summary.csv"
    new_header = not summary_path.exists() or summary_path.stat().st_size == 0
    done_set = set() if OVERWRITE else _load_done_set(summary_path)

    with summary_path.open("a", newline="", encoding="utf-8") as sf:
        sw = csv.DictWriter(sf, fieldnames=SUMMARY_FIELDS)
        if new_header:
            sw.writeheader()

        done = len(done_set)
        for scale, inst_path in all_instances:
            inst_name = Path(inst_path).stem
            t_wall = TIME_LIMITS_POP.get(scale, DEFAULT_TIME_LIMIT_POP)

            inst_dir = OUTPUT_DIR / inst_name
            if SAVE_CONVERGENCE_LOG:
                inst_dir.mkdir(parents=True, exist_ok=True)

            for algo_name in ALGORITHMS:
                for run_num in range(1, RUNS + 1):
                    if (inst_name, algo_name, run_num) in done_set:
                        continue

                    conv_csv = None
                    if SAVE_CONVERGENCE_LOG:
                        safe_name = algo_name.replace("-", "_")
                        conv_csv = inst_dir / f"{safe_name}_run{run_num:02d}.csv"

                    result = _run_one(
                        inst_path,
                        algo_name,
                        seed=run_num - 1,
                        t_wall=t_wall,
                        conv_csv=conv_csv,
                    )

                    sw.writerow({
                        "instance":         inst_name,
                        "scale":            scale,
                        "algorithm":        algo_name,
                        "run_num":          run_num,
                        "fitness":          f"{result['fitness']:.6f}",
                        "is_feasible":      result["is_feasible"],
                        "final_generation": result["final_generation"],
                        "runtime_s":        f"{result['runtime_s']:.2f}",
                    })
                    sf.flush()
                    done_set.add((inst_name, algo_name, run_num))
                    done += 1

                    print(
                        f"  [{done:4d}/{total}] {inst_name:<28} {algo_name:<10} "
                        f"run{run_num:02d}/{RUNS} "
                        f"| fitness={result['fitness']:.4f} "
                        f"| feasible={result['is_feasible']} "
                        f"| {result['runtime_s']:.1f}s"
                    )

    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
