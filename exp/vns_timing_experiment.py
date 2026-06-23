#!/usr/bin/env python3
"""Measure VNSSolver phase timings for one MM-MT-dLARP instance.

This experiment runs VNS once and reports:
- total wall-clock solve time,
- total time spent in adaptive shaking,
- total time spent in adaptive VND,
- final splitting_phase wall-clock time.

The timers are implemented by subclassing VNSSolver, so the solver logic is not
changed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

# Allow running from repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mm_mt_dlarp.algorithms import SolverConfig, VNSSolver
from mm_mt_dlarp.discretize import build_discrete_instance, initial_breakpoints
from mm_mt_dlarp.models import DiscreteInstance, Solution
from mm_mt_dlarp.parser import parse_instance


class TimedVNSSolver(VNSSolver):
    """VNSSolver with cumulative wall-clock timers around selected phases."""

    def __init__(self, instance: DiscreteInstance, config: SolverConfig, **kwargs) -> None:
        super().__init__(instance, config, **kwargs)
        self.timing = {
            "adaptive_shake_seconds": 0.0,
            "adaptive_shake_calls": 0,
            "adaptive_vnd_seconds": 0.0,
            "adaptive_vnd_calls": 0,
            "splitting_phase_seconds": 0.0,
            "splitting_phase_calls": 0,
        }

    def _adaptive_shake(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return super()._adaptive_shake(*args, **kwargs)
        finally:
            self.timing["adaptive_shake_seconds"] += time.perf_counter() - started
            self.timing["adaptive_shake_calls"] += 1

    def _adaptive_vnd(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return super()._adaptive_vnd(*args, **kwargs)
        finally:
            self.timing["adaptive_vnd_seconds"] += time.perf_counter() - started
            self.timing["adaptive_vnd_calls"] += 1

    def splitting_phase(self, base_solution: Solution) -> Tuple[Solution, DiscreteInstance]:
        started = time.perf_counter()
        try:
            return super().splitting_phase(base_solution)
        finally:
            self.timing["splitting_phase_seconds"] += time.perf_counter() - started
            self.timing["splitting_phase_calls"] += 1


def build_instance(
    instance_file: Path,
    num_trucks: int,
    flight_limit: Optional[float],
    base_vertex: Optional[int],
) -> Tuple[DiscreteInstance, SolverConfig]:
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
        seed=0,
        base_vertex=base,
        flight_optimizer="bc",
        bc_max_cuts_per_round=200,
        time_limit_seconds=None,
        verbose=True,
    )
    return discrete, config


def format_seconds(value: float) -> str:
    return f"{value:.6f}"


def render_markdown(payload: dict) -> str:
    rows = [
        ("Instance", payload["instance"]),
        ("Num trucks", str(payload["num_trucks"])),
        ("Flight limit", f'{payload["flight_limit"]:.6f}'),
        ("Objective", f'{payload["objective"]:.6f}'),
        ("Paper makespan", f'{payload["paper_makespan"]:.6f}'),
        ("Trucks used", str(payload["trucks_used"])),
        ("Total solve time (s)", format_seconds(payload["solve_seconds"])),
        ("Adaptive shake total (s)", format_seconds(payload["adaptive_shake_seconds"])),
        ("Adaptive shake calls", str(payload["adaptive_shake_calls"])),
        ("Adaptive VND total (s)", format_seconds(payload["adaptive_vnd_seconds"])),
        ("Adaptive VND calls", str(payload["adaptive_vnd_calls"])),
        ("Splitting phase final time (s)", format_seconds(payload["splitting_phase_seconds"])),
        ("Splitting phase calls", str(payload["splitting_phase_calls"])),
        (
            "Adaptive VND outside splitting estimate (s)",
            format_seconds(payload["adaptive_vnd_outside_splitting_estimate_seconds"]),
        ),
    ]

    width = max(len(name) for name, _ in rows)
    body = ["# VNS Timing Experiment", ""]
    body.append("| Metric | Value |")
    body.append("| --- | ---: |")
    for name, value in rows:
        body.append(f"| {name.ljust(width)} | {value} |")

    body.extend(
        [
            "",
            "## Notes",
            "",
            "- `Adaptive VND total` is cumulative over every `_adaptive_vnd(...)` call, including calls made inside `splitting_phase`.",
            "- `Splitting phase final time` is inclusive wall-clock time for the final `splitting_phase(...)` call.",
            "- `Adaptive VND outside splitting estimate` is computed as `max(0, adaptive_vnd_total - splitting_phase_time)`; it is an estimate because splitting time also contains conversion/refinement overhead.",
        ]
    )
    return "\n".join(body) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run VNS and measure internal phase timings.")
    parser.add_argument(
        "instance_file",
        type=Path,
        nargs="?",
        default=Path("data/instances/MTLARP4_4_6_2.dat"),
    )
    parser.add_argument("--num-trucks", type=int, default=2)
    parser.add_argument("--flight-limit", type=float, default=903.0)
    parser.add_argument("--base-vertex", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vns-k-max", type=int, default=4)
    parser.add_argument("--vns-max-iter", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/vns_timing_MTLARP4_4_6_2_P2_L903.md"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("docs/vns_timing_MTLARP4_4_6_2_P2_L903.json"),
    )
    args = parser.parse_args(argv)

    instance, config = build_instance(
        args.instance_file,
        num_trucks=args.num_trucks,
        flight_limit=args.flight_limit,
        base_vertex=args.base_vertex,
    )
    config.seed = args.seed

    solver = TimedVNSSolver(instance, config, k_max=args.vns_k_max, max_iter=args.vns_max_iter)

    started = time.perf_counter()
    solution, final_instance = solver.solve()
    solve_seconds = time.perf_counter() - started

    splitting_phase_seconds = solver.timing["splitting_phase_seconds"]
    adaptive_vnd_seconds = solver.timing["adaptive_vnd_seconds"]

    payload = {
        "instance": instance.raw.name,
        "base": final_instance.base_vertex,
        "num_trucks": args.num_trucks,
        "flight_limit": solver.L,
        "objective": solution.objective,
        "paper_makespan": solution.paper_makespan,
        "trucks_used": len(solution.selected_launches),
        "solve_seconds": solve_seconds,
        "adaptive_shake_seconds": solver.timing["adaptive_shake_seconds"],
        "adaptive_shake_calls": solver.timing["adaptive_shake_calls"],
        "adaptive_vnd_seconds": adaptive_vnd_seconds,
        "adaptive_vnd_calls": solver.timing["adaptive_vnd_calls"],
        "splitting_phase_seconds": splitting_phase_seconds,
        "splitting_phase_calls": solver.timing["splitting_phase_calls"],
        "adaptive_vnd_outside_splitting_estimate_seconds": max(
            0.0,
            adaptive_vnd_seconds - splitting_phase_seconds,
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(payload), encoding="utf-8")

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(render_markdown(payload))
    print(f"Wrote markdown timing report to {args.output.resolve()}")
    print(f"Wrote JSON timing report to {args.json_output.resolve()}")


if __name__ == "__main__":
    main()
