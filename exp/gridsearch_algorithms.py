#!/usr/bin/env python3
"""Grid search VND, VNS, and LNS parameters on one MM-MT-dLARP instance.

The experiment reuses ``src.run_benchmark.run_one_algorithm`` so every trial is
reported with the same objective, timing, and convergence fields as the regular
benchmark runner. Rankings minimize paper makespan first and wall-clock time
second.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mm_mt_dlarp.algorithms import LNSSolver, VNDSolver, VNSSolver
from src.run_benchmark import BenchmarkRow, build_instance, render_markdown, run_one_algorithm

SOLVER_CLASSES = {
    "vnd": VNDSolver,
    "vns": VNSSolver,
    "lns": LNSSolver,
}


@dataclass(frozen=True)
class GridTrial:
    algorithm: str
    seed: int
    config_overrides: Dict[str, Any]
    solver_kwargs: Dict[str, Any]
    param_label: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "seed": self.seed,
            "config_overrides": dict(self.config_overrides),
            "solver_kwargs": dict(self.solver_kwargs),
            "param_label": self.param_label,
        }


def parse_csv_ints(raw: str) -> List[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return [int(value) for value in values]


def parse_csv_floats(raw: str) -> List[float]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one number")
    return [float(value) for value in values]


def parse_algorithms(raw: str) -> List[str]:
    algorithms = [item.strip().lower() for item in raw.split(",") if item.strip()]
    unknown = [name for name in algorithms if name not in SOLVER_CLASSES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown algorithm(s): {', '.join(unknown)}; valid choices: {', '.join(SOLVER_CLASSES)}"
        )
    if not algorithms:
        raise argparse.ArgumentTypeError("expected at least one algorithm")
    return algorithms


def _label(parts: Mapping[str, Any]) -> str:
    return ";".join(f"{key}={value}" for key, value in parts.items())


def build_grid(
    *,
    algorithms: Sequence[str],
    seeds: Sequence[int],
    vns_k_max_values: Sequence[int],
    vns_max_iter_values: Sequence[int],
    lns_destroy_frac_values: Sequence[float],
    lns_max_iter_values: Sequence[int],
    vnd_split_top_k_values: Sequence[int],
) -> List[Dict[str, Any]]:
    trials: List[GridTrial] = []

    for algorithm in algorithms:
        if algorithm == "vns":
            for seed, k_max, max_iter in product(seeds, vns_k_max_values, vns_max_iter_values):
                params = {"vns_k_max": k_max, "vns_max_iter": max_iter}
                trials.append(
                    GridTrial(
                        algorithm="vns",
                        seed=seed,
                        config_overrides={},
                        solver_kwargs={"k_max": k_max, "max_iter": max_iter},
                        param_label=_label(params),
                    )
                )
        elif algorithm == "lns":
            for seed, destroy_frac, max_iter in product(seeds, lns_destroy_frac_values, lns_max_iter_values):
                params = {"lns_destroy_frac": destroy_frac, "lns_max_iter": max_iter}
                trials.append(
                    GridTrial(
                        algorithm="lns",
                        seed=seed,
                        config_overrides={},
                        solver_kwargs={"destroy_frac": destroy_frac, "max_iter": max_iter},
                        param_label=_label(params),
                    )
                )
        elif algorithm == "vnd":
            for seed, split_top_k in product(seeds, vnd_split_top_k_values):
                params = {"vnd_split_top_k": split_top_k}
                trials.append(
                    GridTrial(
                        algorithm="vnd",
                        seed=seed,
                        config_overrides={"split_top_k": split_top_k},
                        solver_kwargs={},
                        param_label=_label(params),
                    )
                )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    return [trial.as_dict() for trial in trials]


def _apply_overrides(config: Any, overrides: Mapping[str, Any]) -> None:
    for name, value in overrides.items():
        setattr(config, name, value)


def _row_payload(row: BenchmarkRow, trial: Mapping[str, Any], trial_index: int) -> Dict[str, Any]:
    payload = row.as_csv_row()
    payload.update(
        {
            "Trial": trial_index,
            "Seed": trial["seed"],
            "Parameters": trial["param_label"],
        }
    )
    return payload


def select_best_rows(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    best: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        algorithm = str(row["Algorithm"])
        current = best.get(algorithm)
        if current is None or (float(row["Objective"]), float(row["Time"])) < (
            float(current["Objective"]),
            float(current["Time"]),
        ):
            best[algorithm] = row
    return best


def _write_csv(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    fieldnames = ["Trial", "Algorithm", "Seed", "Parameters"] + [
        key for key in rows[0].keys() if key not in {"Trial", "Algorithm", "Seed", "Parameters"}
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(list(rows), indent=2), encoding="utf-8")


def _render_markdown_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    headers = [
        "Algorithm",
        "Seed",
        "Parameters",
        "Objective",
        "Paper Makespan",
        "Time",
        "Trucks Used",
        "Convergence Iterations",
    ]
    body = [[str(row.get(header, "")) for header in headers] for row in rows]
    widths = [len(header) for header in headers]
    for cells in body:
        for idx, cell in enumerate(cells):
            widths[idx] = max(widths[idx], len(cell))

    def line(cells: Sequence[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(cells)) + " |"

    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([line(headers), sep, *(line(cells) for cells in body)])


def _write_summary(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    best = select_best_rows(rows)
    ordered = [best[name] for name in ("vns", "lns", "vnd") if name in best]
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Grid Search Best Parameters", "", "Ranked by objective first, then runtime.", ""]
    lines.append(_render_markdown_rows(ordered))
    lines.append("")
    lines.append("## All Trials")
    lines.append("")
    lines.append(_render_markdown_rows(rows))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_gridsearch(args: argparse.Namespace) -> List[Dict[str, Any]]:
    instance, base_config = build_instance(
        args.instance_file,
        num_trucks=args.num_trucks,
        flight_limit=args.flight_limit,
        base_vertex=args.base_vertex,
    )
    base_config.flight_optimizer = args.flight_optimizer
    base_config.bc_time_limit = args.bc_time_limit
    base_config.bc_mip_gap = args.bc_mip_gap
    base_config.bc_enable_connectivity_cuts = args.bc_enable_connectivity_cuts
    base_config.bc_enable_r_odd_cuts = args.bc_enable_r_odd_cuts
    base_config.bc_enable_advanced_cuts = args.bc_enable_advanced_cuts
    base_config.bc_cut_at_fractional_nodes = args.bc_cut_at_fractional_nodes
    base_config.bc_cut_at_integer_solutions = args.bc_cut_at_integer_solutions
    base_config.bc_max_cuts_per_round = args.bc_max_cuts_per_round
    base_config.max_construction_solutions = args.max_construction_solutions
    base_config.fast_construction_target = args.fast_construction_target
    base_config.fast_construction_seconds = args.fast_construction_seconds
    base_config.max_construction_seconds = args.max_construction_seconds
    base_config.max_construction_attempts = args.max_construction_attempts
    base_config.max_stall_attempts = args.max_stall_attempts
    base_config.time_limit_seconds = args.component_time_limit if args.component_time_limit > 0 else None
    base_config.verbose = not args.quiet

    trials = build_grid(
        algorithms=args.algorithms,
        seeds=args.seeds,
        vns_k_max_values=args.vns_k_max_values,
        vns_max_iter_values=args.vns_max_iter_values,
        lns_destroy_frac_values=args.lns_destroy_frac_values,
        lns_max_iter_values=args.lns_max_iter_values,
        vnd_split_top_k_values=args.vnd_split_top_k_values,
    )

    rows: List[Dict[str, Any]] = []
    for idx, trial in enumerate(trials, start=1):
        config = deepcopy(base_config)
        _apply_overrides(config, trial["config_overrides"])
        row = run_one_algorithm(
            name=trial["algorithm"],
            solver_cls=SOLVER_CLASSES[trial["algorithm"]],
            instance=instance,
            config_template=config,
            seed=trial["seed"],
            verbose=not args.quiet,
            solver_kwargs=trial["solver_kwargs"],
        )
        payload = _row_payload(row, trial, idx)
        rows.append(payload)
        print(
            f"[{idx}/{len(trials)}] {payload['Algorithm']} seed={payload['Seed']} "
            f"params={payload['Parameters']} objective={payload['Objective']} time={payload['Time']}s"
        )

    _write_csv(rows, args.output_csv)
    _write_json(rows, args.output_json)
    _write_summary(rows, args.output_summary)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grid search VNS, LNS, and VND parameters on one instance.")
    parser.add_argument("instance_file", type=Path, help="Path to an MM-MT-dLARP .dat instance file")
    parser.add_argument("--num-trucks", type=int, required=True)
    parser.add_argument("--flight-limit", type=float, required=True)
    parser.add_argument("--base-vertex", type=int, default=None)
    parser.add_argument("--algorithms", type=parse_algorithms, default=["vns", "lns", "vnd"])
    parser.add_argument("--seeds", type=parse_csv_ints, default=[0])
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--vns-k-max-values", type=parse_csv_ints, default=[2, 3, 4])
    parser.add_argument("--vns-max-iter-values", type=parse_csv_ints, default=[50])
    parser.add_argument("--lns-destroy-frac-values", type=parse_csv_floats, default=[0.2, 0.3, 0.4])
    parser.add_argument("--lns-max-iter-values", type=parse_csv_ints, default=[100, 200])
    parser.add_argument("--vnd-split-top-k-values", type=parse_csv_ints, default=[5, 10, 15])

    parser.add_argument("--flight-optimizer", choices=["bc", "dp", "auto"], default="bc")
    parser.add_argument("--bc-time-limit", type=float, default=None)
    parser.add_argument("--bc-mip-gap", type=float, default=0.0)
    parser.add_argument("--bc-enable-connectivity-cuts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bc-enable-r-odd-cuts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bc-enable-advanced-cuts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bc-cut-at-fractional-nodes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bc-cut-at-integer-solutions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bc-max-cuts-per-round", type=int, default=200)

    parser.add_argument("--max-construction-solutions", type=int, default=20)
    parser.add_argument("--fast-construction-target", type=int, default=300)
    parser.add_argument("--fast-construction-seconds", type=float, default=30.0)
    parser.add_argument("--max-construction-seconds", type=float, default=600.0)
    parser.add_argument("--max-construction-attempts", type=int, default=2000)
    parser.add_argument("--max-stall-attempts", type=int, default=250)
    parser.add_argument("--component-time-limit", type=float, default=9.5)

    parser.add_argument("--output-csv", type=Path, default=Path("docs/gridsearch_results.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("docs/gridsearch_results.json"))
    parser.add_argument("--output-summary", type=Path, default=Path("docs/gridsearch_summary.md"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    rows = run_gridsearch(args)
    print(render_markdown([])) if not rows else None
    print(f"Wrote grid rows to {args.output_csv.resolve()}")
    print(f"Wrote grid JSON to {args.output_json.resolve()}")
    print(f"Wrote best-parameter summary to {args.output_summary.resolve()}")


if __name__ == "__main__":
    main()
