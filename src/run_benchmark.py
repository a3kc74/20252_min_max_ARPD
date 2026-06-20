#!/usr/bin/env python3
"""Benchmark the implemented MM-MT-dLARP heuristics on one instance.

The benchmark intentionally treats each algorithm as a black box and calls only
its public ``solve()`` method after constructing the solver object.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

# Allow running this file directly from ``src/`` or from the repository root.
# Prefer the repository-root package over the legacy src/mm_mt_dlarp namespace.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mm_mt_dlarp.algorithms import LCBIMMASolver, LNSSolver, SolverConfig, VNDSolver, VNSSolver
from mm_mt_dlarp.discretize import build_discrete_instance, initial_breakpoints
from mm_mt_dlarp.models import DiscreteInstance
from mm_mt_dlarp.parser import parse_instance


@dataclass(frozen=True)
class BenchmarkRow:
    Algorithm: str
    Instance: str
    Base: int
    Flight_Limit: float
    Trucks_Used: int
    Trucks_Submitted: int
    Objective_Type: str
    Objective: float
    Paper_Makespan: float
    GHG_Makespan: float
    Total_GHG: float
    Time: float
    Flight_Optimizer: str
    Convergence_Iterations: str
    Convergence_Count: int
    Convergence_First_Objective: str
    Convergence_Final_Objective: str
    Convergence_Improvement: str
    Convergence_Improvement_Percent: str

    def as_csv_row(self) -> dict:
        return {
            "Algorithm": self.Algorithm,
            "Instance": self.Instance,
            "Base": self.Base,
            "Flight Limit": _format_number(self.Flight_Limit),
            "Trucks Used": self.Trucks_Used,
            "Trucks Submitted": self.Trucks_Submitted,
            "Objective Type": self.Objective_Type,
            "Objective": _format_number(self.Objective),
            "Paper Makespan": _format_number(self.Paper_Makespan),
            "GHG Makespan": _format_number(self.GHG_Makespan),
            "Total GHG": _format_number(self.Total_GHG),
            "Time": _format_number(self.Time),
            "Flight Optimizer": self.Flight_Optimizer,
            "Convergence Iterations": self.Convergence_Iterations,
            "Convergence Count": self.Convergence_Count,
            "Convergence First Objective": self.Convergence_First_Objective,
            "Convergence Final Objective": self.Convergence_Final_Objective,
            "Convergence Improvement": self.Convergence_Improvement,
            "Convergence Improvement Percent": self.Convergence_Improvement_Percent,
        }


BASELINE_ALGORITHMS: Tuple[Tuple[str, Type], ...] = (
    ("vnd", VNDSolver),
    ("vns", VNSSolver),
    ("lns", LNSSolver),
    ("lcb_imma", LCBIMMASolver),
)

ALGORITHM_NAMES: Tuple[str, ...] = tuple(name for name, _ in BASELINE_ALGORITHMS)


def _format_number(value: float) -> str:
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def _parse_algorithms(raw: str) -> List[str]:
    requested = [item.strip().lower() for item in raw.split(",") if item.strip()]
    valid = set(ALGORITHM_NAMES)
    unknown = [name for name in requested if name not in valid]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown algorithm(s): {', '.join(unknown)}; valid choices: {', '.join(sorted(valid))}"
        )
    return requested


def _clone_config(config_template: SolverConfig, seed: int, verbose: bool) -> SolverConfig:
    return SolverConfig(
        num_trucks=config_template.num_trucks,
        flight_limit=config_template.flight_limit,
        seed=seed,
        base_vertex=config_template.base_vertex,
        max_construction_solutions=config_template.max_construction_solutions,
        fast_construction_target=config_template.fast_construction_target,
        fast_construction_seconds=config_template.fast_construction_seconds,
        max_construction_seconds=config_template.max_construction_seconds,
        max_construction_attempts=config_template.max_construction_attempts,
        max_stall_attempts=config_template.max_stall_attempts,
        nmax_destroy=config_template.nmax_destroy,
        itmax_destroy=config_template.itmax_destroy,
        lmax_exchange=config_template.lmax_exchange,
        exact_flight_threshold=config_template.exact_flight_threshold,
        flight_optimizer=config_template.flight_optimizer,
        bc_time_limit=config_template.bc_time_limit,
        bc_mip_gap=config_template.bc_mip_gap,
        bc_enable_connectivity_cuts=config_template.bc_enable_connectivity_cuts,
        bc_enable_r_odd_cuts=config_template.bc_enable_r_odd_cuts,
        bc_enable_advanced_cuts=config_template.bc_enable_advanced_cuts,
        bc_cut_at_fractional_nodes=config_template.bc_cut_at_fractional_nodes,
        bc_cut_at_integer_solutions=config_template.bc_cut_at_integer_solutions,
        bc_max_cuts_per_round=config_template.bc_max_cuts_per_round,
        bc_tailoff_window=config_template.bc_tailoff_window,
        bc_tailoff_tol=config_template.bc_tailoff_tol,
        split_top_k=config_template.split_top_k,
        time_limit_seconds=config_template.time_limit_seconds,
        search_deadline=config_template.search_deadline,
        objective_type=config_template.objective_type,
        emission_truck=config_template.emission_truck,
        emission_drone_cruise=config_template.emission_drone_cruise,
        emission_drone_vt=config_template.emission_drone_vt,
        emission_epsilon=config_template.emission_epsilon,
        verbose=verbose,
    )


def build_instance(instance_file: Path, num_trucks: int, flight_limit: Optional[float], base_vertex: Optional[int]) -> Tuple[DiscreteInstance, SolverConfig]:
    raw = parse_instance(instance_file)
    base = base_vertex if base_vertex is not None else raw.depot_vertices[0]
    discrete = build_discrete_instance(
        raw,
        base_vertex=base,
        launch_vertices=raw.depot_vertices,
        selected_breakpoints=initial_breakpoints(raw),
    )
    config = SolverConfig(
        num_trucks=num_trucks,
        flight_limit=flight_limit,
        base_vertex=base,
    )
    return discrete, config


def _valid_convergence_entries(solver: Any) -> List[Tuple[Any, Any, Any]]:
    """Return convergence entries that include iteration, elapsed time, and solution."""
    convergence_log = getattr(solver, "convergence_log", None)
    if not convergence_log:
        return []

    entries: List[Tuple[Any, Any, Any]] = []
    for entry in convergence_log:
        if len(entry) < 3:
            continue
        entries.append((entry[0], entry[1], entry[2]))
    return entries


def _convergence_entry_metadata(entry: Any) -> Dict[str, Any]:
    """Return optional metadata stored in a convergence entry."""
    if len(entry) >= 4 and isinstance(entry[3], dict):
        return entry[3]
    return {}


def _format_convergence_iterations(solver: Any) -> str:
    """Return convergence iteration indices recorded by a solver.

    Solvers usually record convergence as tuples of
    ``(iteration, elapsed, solution_snapshot)``. LNS additionally stores
    metadata as ``(iteration, elapsed, solution_snapshot, metadata)`` so the
    trace can show the local iteration inside the specific candidate that
    improved the global incumbent.
    """
    convergence_log = getattr(solver, "convergence_log", None)
    if not convergence_log:
        return ""

    formatted: List[str] = []
    for entry in convergence_log:
        if len(entry) < 3:
            continue
        metadata = _convergence_entry_metadata(entry)
        candidate = metadata.get("candidate")
        phase = metadata.get("phase")
        local_iteration = metadata.get("local_iteration", entry[0])

        if candidate is not None:
            formatted.append(f"candidate{candidate}:{phase or 'iter'}@{local_iteration}")
        elif phase:
            formatted.append(f"{phase}@{local_iteration}")
        else:
            formatted.append(str(entry[0]))

    return ";".join(formatted)


def _convergence_analysis(solver: Any) -> Dict[str, Any]:
    """Summarize convergence into table-friendly metrics.

    The metrics are generic enough for VND, VNS, and LNS. For LNS, the
    iteration values can include candidate offsets because LNSSolver records
    improvements across multiple starting candidates.
    """
    entries = _valid_convergence_entries(solver)
    if not entries:
        return {
            "count": 0,
            "first_objective": "",
            "final_objective": "",
            "improvement": "",
            "improvement_percent": "",
        }

    first_objective = getattr(entries[0][2], "objective", float("nan"))
    best_objective = min(
        getattr(solution, "objective", float("nan"))
        for _, _, solution in entries
    )
    improvement = first_objective - best_objective

    if first_objective and not math.isnan(first_objective):
        improvement_percent = 100.0 * improvement / first_objective
    else:
        improvement_percent = float("nan")

    return {
        "count": len(entries),
        "first_objective": _format_number(first_objective),
        "final_objective": _format_number(best_objective),
        "improvement": _format_number(improvement),
        "improvement_percent": _format_number(improvement_percent),
    }


def _print_convergence_log(name: str, solver: Any) -> None:
    """Print convergence records captured during a solver run."""
    convergence_log = getattr(solver, "convergence_log", None)
    if not convergence_log:
        print(f"[{name}] convergence: <empty>")
        return

    print(f"[{name}] convergence:")
    for entry in convergence_log:
        if len(entry) < 3:
            print(f"  iteration={entry[0] if entry else '?'}")
            continue
        iteration, elapsed, solution = entry[:3]
        metadata = _convergence_entry_metadata(entry)
        objective = getattr(solution, "objective", float("nan"))

        if metadata:
            candidate = metadata.get("candidate")
            phase = metadata.get("phase", "unknown")
            local_iteration = metadata.get("local_iteration", iteration)
            candidate_label = "global" if candidate is None else f"candidate={candidate}"
            print(
                f"  phase={phase} "
                f"{candidate_label} "
                f"local_iteration={local_iteration} "
                f"time={_format_number(elapsed)}s "
                f"objective={_format_number(objective)}"
            )
        else:
            print(
                f"  iteration={iteration} "
                f"time={_format_number(elapsed)}s "
                f"objective={_format_number(objective)}"
            )


def run_one_algorithm(
    name: str,
    solver_cls: Type,
    instance: DiscreteInstance,
    config_template: SolverConfig,
    seed: int,
    verbose: bool,
    solver_kwargs: Optional[Dict[str, Any]] = None,
) -> BenchmarkRow:
    # Each algorithm gets its own fresh instance object and config so mutations
    # inside one solve() call cannot leak into the next algorithm.
    discrete = build_discrete_instance(
        instance.raw,
        base_vertex=instance.base_vertex,
        launch_vertices=instance.launch_vertices,
        selected_breakpoints=initial_breakpoints(instance.raw),
    )
    config = _clone_config(config_template, seed, verbose)
    config.time_limit_seconds = None

    solver = solver_cls(discrete, config, **(solver_kwargs or {}))
    started = time.perf_counter()
    solution, final_instance = solver.solve()  # The benchmark uses only the public solve() call.
    elapsed = time.perf_counter() - started
    _print_convergence_log(name, solver)
    convergence = _convergence_analysis(solver)

    return BenchmarkRow(
        Algorithm=name,
        Instance=instance.raw.name,
        Base=final_instance.base_vertex,
        Flight_Limit=solver.L,
        Trucks_Used=len(solution.selected_launches),
        Trucks_Submitted=config.num_trucks,
        Objective_Type=config.objective_type,
        Objective=solution.objective,
        Paper_Makespan=solution.paper_makespan,
        GHG_Makespan=solution.ghg_makespan,
        Total_GHG=solution.total_ghg,
        Time=elapsed,
        Flight_Optimizer=config.flight_optimizer,
        Convergence_Iterations=_format_convergence_iterations(solver),
        Convergence_Count=convergence["count"],
        Convergence_First_Objective=convergence["first_objective"],
        Convergence_Final_Objective=convergence["final_objective"],
        Convergence_Improvement=convergence["improvement"],
        Convergence_Improvement_Percent=convergence["improvement_percent"],
    )


def write_output(rows: Sequence[BenchmarkRow], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".json":
        output.write_text(
            json.dumps([row.as_csv_row() for row in rows], indent=2),
            encoding="utf-8",
        )
        return
    if suffix in {".md", ".markdown"}:
        output.write_text(render_markdown(rows) + "\n", encoding="utf-8")
        return

    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "Algorithm",
                "Instance",
                "Base",
                "Flight Limit",
                "Trucks Used",
                "Trucks Submitted",
                "Objective Type",
                "Objective",
                "Paper Makespan",
                "GHG Makespan",
                "Total GHG",
                "Time",
                "Flight Optimizer",
                "Convergence Iterations",
                "Convergence Count",
                "Convergence First Objective",
                "Convergence Final Objective",
                "Convergence Improvement",
                "Convergence Improvement Percent",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_row())


def render_markdown(rows: Sequence[BenchmarkRow]) -> str:
    headers = [
        "Algorithm",
        "Instance",
        "Base",
        "Flight Limit",
        "Trucks Used",
        "Trucks Submitted",
        "Objective Type",
        "Objective",
        "Paper Makespan",
        "GHG Makespan",
        "Total GHG",
        "Time",
        "Flight Optimizer",
        "Convergence Iterations",
        "Convergence Count",
        "Convergence First Objective",
        "Convergence Final Objective",
        "Convergence Improvement",
        "Convergence Improvement Percent",
    ]
    body = []
    for row in rows:
        payload = row.as_csv_row()
        body.append([str(payload[h]) for h in headers])

    widths = [len(h) for h in headers]
    for cells in body:
        for i, cell in enumerate(cells):
            widths[i] = max(widths[i], len(cell))

    def line(cells: Sequence[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)) + " |"

    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    return "\n".join([line(headers), sep, *(line(cells) for cells in body)])


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a side-by-side benchmark of baseline solvers."
    )
    parser.add_argument("instance_file", type=Path, help="Path to an MM-MT-dLARP .dat instance file")
    parser.add_argument("--num-trucks", type=int, required=True, help="Number of trucks P submitted to every solver")
    parser.add_argument("--flight-limit", type=float, default=None, help="Flight limit L; defaults to the solver estimate")
    parser.add_argument("--base-vertex", type=int, default=None, help="Central truck depot vertex; defaults to first DEPOTS entry")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed; each algorithm receives this same seed")
    parser.add_argument("--output", type=Path, default=Path("benchmark.csv"), help="Output table path: .csv, .json, or .md")
    parser.add_argument(
        "--algorithms",
        type=_parse_algorithms,
        default=["vnd", "vns", "lns", "lcb_imma"],
        help="Comma-separated subset/order. Baselines: vnd,vns,lns,lcb_imma.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable solver progress logs")
    parser.add_argument("--objective-type", choices=["minmax_ghg", "paper_makespan", "total_ghg"], default="minmax_ghg")
    parser.add_argument("--flight-optimizer", choices=["bc", "dp", "auto"], default="bc")
    parser.add_argument("--bc-time-limit", type=float, default=None)
    parser.add_argument("--bc-mip-gap", type=float, default=0.0)
    parser.add_argument("--bc-enable-connectivity-cuts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bc-enable-r-odd-cuts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bc-enable-advanced-cuts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bc-cut-at-fractional-nodes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bc-cut-at-integer-solutions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bc-max-cuts-per-round", type=int, default=200)

    # Practical knobs for faster or deeper benchmark runs. Defaults mirror SolverConfig.
    parser.add_argument("--max-construction-solutions", type=int, default=20)
    parser.add_argument("--fast-construction-target", type=int, default=300)
    parser.add_argument("--fast-construction-seconds", type=float, default=30.0)
    parser.add_argument("--max-construction-seconds", type=float, default=600.0)
    parser.add_argument("--max-construction-attempts", type=int, default=2000)
    parser.add_argument("--max-stall-attempts", type=int, default=250)
    parser.add_argument("--split-top-k", type=int, default=10)
    parser.add_argument(
        "--component-time-limit",
        type=float,
        default=9.5,
        help="Wall-clock time budget in seconds for each improvement-component benchmark row; default leaves a small guard for a 10s target; use 0 to disable.",
    )
    parser.add_argument("--vns-k-max", type=int, default=4, help="VNSSolver k_max constructor argument")
    parser.add_argument("--vns-max-iter", type=int, default=100, help="VNSSolver max_iter constructor argument")
    parser.add_argument("--lns-destroy-frac", type=float, default=0.3, help="LNSSolver destroy_frac constructor argument")
    parser.add_argument("--lns-max-iter", type=int, default=200, help="LNSSolver max_iter constructor argument")
    # LCB-IMMA knobs
    parser.add_argument("--lcb-n-pop", type=int, default=40, help="LCBIMMASolver population size")
    parser.add_argument("--lcb-t-wall", type=float, default=60.0, help="LCBIMMASolver wall-clock budget (s)")
    parser.add_argument("--lcb-tau-base", type=int, default=10, help="LCBIMMASolver periodic migration interval (epochs)")
    parser.add_argument("--lcb-alpha", type=float, default=1.0, help="LCBIMMASolver LinUCB exploration coefficient")
    parser.add_argument("--lcb-rho", type=float, default=0.99, help="LCBIMMASolver annealing decay rate")
    parser.add_argument("--lcb-T0", type=float, default=1.0, help="LCBIMMASolver initial Metropolis temperature")
    # Emission factor overrides
    parser.add_argument("--emission-truck", type=float, default=1.0, help="F_t: truck GHG emission factor")
    parser.add_argument("--emission-drone-cruise", type=float, default=1.0, help="F_d: drone cruise emission factor")
    parser.add_argument("--emission-drone-vt", type=float, default=0.0, help="F_d^vt: drone vertical takeoff/landing emission per flight")
    args = parser.parse_args(argv)

    instance, config = build_instance(args.instance_file, args.num_trucks, args.flight_limit, args.base_vertex)
    config.seed = args.seed
    config.max_construction_solutions = args.max_construction_solutions
    config.fast_construction_target = args.fast_construction_target
    config.fast_construction_seconds = args.fast_construction_seconds
    config.max_construction_seconds = args.max_construction_seconds
    config.max_construction_attempts = args.max_construction_attempts
    config.max_stall_attempts = args.max_stall_attempts
    config.split_top_k = args.split_top_k
    config.time_limit_seconds = args.component_time_limit if args.component_time_limit > 0 else None
    config.verbose = not args.quiet
    config.objective_type = args.objective_type
    config.flight_optimizer = args.flight_optimizer
    config.bc_time_limit = args.bc_time_limit
    config.bc_mip_gap = args.bc_mip_gap
    config.bc_enable_connectivity_cuts = args.bc_enable_connectivity_cuts
    config.bc_enable_r_odd_cuts = args.bc_enable_r_odd_cuts
    config.bc_enable_advanced_cuts = args.bc_enable_advanced_cuts
    config.bc_cut_at_fractional_nodes = args.bc_cut_at_fractional_nodes
    config.bc_cut_at_integer_solutions = args.bc_cut_at_integer_solutions
    config.bc_max_cuts_per_round = args.bc_max_cuts_per_round
    config.emission_truck = args.emission_truck
    config.emission_drone_cruise = args.emission_drone_cruise
    config.emission_drone_vt = args.emission_drone_vt

    selected_baselines = dict(BASELINE_ALGORITHMS)
    per_algorithm_kwargs: Dict[str, Dict[str, Any]] = {
        "vnd": {},
        "vns": {"k_max": args.vns_k_max, "max_iter": args.vns_max_iter},
        "lns": {"destroy_frac": args.lns_destroy_frac, "max_iter": args.lns_max_iter},
        "lcb_imma": {
            "n_pop": args.lcb_n_pop,
            "T_wall": args.lcb_t_wall,
            "tau_base": args.lcb_tau_base,
            "alpha": args.lcb_alpha,
            "rho": args.lcb_rho,
            "T_0": args.lcb_T0,
        },
    }

    rows: List[BenchmarkRow] = []
    for name in args.algorithms:
        row = run_one_algorithm(
            name=name,
            solver_cls=selected_baselines[name],
            instance=instance,
            config_template=config,
            seed=args.seed,
            verbose=not args.quiet,
            solver_kwargs=per_algorithm_kwargs[name],
        )
        rows.append(row)
        print(f"[{name}] objective={_format_number(row.Objective)} trucks_used={row.Trucks_Used} time={_format_number(row.Time)}s")

    print(render_markdown(rows))
    write_output(rows, args.output)
    print(f"Wrote benchmark table to {args.output.resolve()}")


if __name__ == "__main__":
    main()
