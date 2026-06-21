#!/usr/bin/env python3
"""Run VNS/LNS/VND benchmark over multiple datasets and seeds.

Each trial delegates to ``src.run_benchmark.run_one_algorithm`` so result fields
match the normal benchmark output. The summary groups by dataset configuration
and algorithm, reporting mean and sample standard deviation for objective and
runtime across seeds.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mm_mt_dlarp.algorithms import LNSSolver, VNDSolver, VNSSolver
from src.run_benchmark import build_instance, run_one_algorithm

DEFAULT_DATASETS = """
MTLARP6_6_3_1.dat,2,1361
MTLARP6_6_3_2.dat,2,1088
MTLARP6_6_4_1.dat,2,2027
MTLARP6_6_4_1.dat,3,2027
MTLARP6_6_4_2.dat,2,1699
MTLARP6_6_4_2.dat,3,1699
MTLARP6_6_5_1.dat,2,961
MTLARP6_6_5_1.dat,3,961
MTLARP6_6_5_1.dat,4,961
MTLARP6_6_5_2.dat,2,1551
MTLARP6_6_5_2.dat,3,1551
MTLARP6_6_5_2.dat,4,1551
MTLARP6_6_6_1.dat,2,1219
MTLARP6_6_6_1.dat,3,1219
MTLARP6_6_6_1.dat,4,1219
MTLARP6_6_6_1.dat,5,1219
MTLARP6_6_6_2.dat,2,1015
MTLARP6_6_6_2.dat,3,1015
MTLARP6_6_6_2.dat,4,1015
MTLARP6_6_6_2.dat,5,1015
"""

SOLVER_CLASSES = {
    "vns": VNSSolver,
    "lns": LNSSolver,
    "vnd": VNDSolver,
}

DEFAULT_ALGORITHMS = ["vns", "lns", "vnd"]
DEFAULT_SEEDS = [0, 1, 2]


@dataclass(frozen=True)
class DatasetConfig:
    instance_name: str
    num_trucks: int
    flight_limit: float


@dataclass(frozen=True)
class BenchmarkTrial:
    dataset: DatasetConfig
    algorithm: str
    seed: int


def parse_csv_ints(raw: str) -> List[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return [int(value) for value in values]


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


def parse_dataset_configs(raw: str) -> List[DatasetConfig]:
    configs: List[DatasetConfig] = []
    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Invalid dataset config at line {line_number}: {raw_line!r}")
        instance_name, num_trucks, flight_limit = parts
        configs.append(DatasetConfig(instance_name, int(num_trucks), float(flight_limit)))
    if not configs:
        raise ValueError("No dataset configs provided")
    return configs


def load_dataset_configs(path: Optional[Path]) -> List[DatasetConfig]:
    if path is None:
        return parse_dataset_configs(DEFAULT_DATASETS)
    return parse_dataset_configs(path.read_text(encoding="utf-8"))


def build_trials(
    configs: Sequence[DatasetConfig],
    *,
    algorithms: Sequence[str],
    seeds: Sequence[int],
) -> List[BenchmarkTrial]:
    return [
        BenchmarkTrial(dataset=config, algorithm=algorithm, seed=seed)
        for config in configs
        for algorithm in algorithms
        for seed in seeds
    ]


def sample_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _format_number(value: float) -> str:
    return f"{value:.6f}"


def _float(row: Mapping[str, Any], key: str) -> float:
    value = row[key]
    return float(value)


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    grouped: Dict[Tuple[str, int, str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["Instance"]),
            int(row["Trucks Submitted"]),
            str(row["Flight Limit"]),
            str(row["Algorithm"]),
        )
        grouped.setdefault(key, []).append(row)

    summary: List[Dict[str, str]] = []
    for key in sorted(grouped):
        instance, trucks, flight_limit, algorithm = key
        group = grouped[key]
        objectives = [_float(row, "Objective") for row in group]
        times = [_float(row, "Time") for row in group]
        paper_makespans = [_float(row, "Paper Makespan") for row in group]
        summary.append(
            {
                "Instance": instance,
                "Trucks Submitted": str(trucks),
                "Flight Limit": flight_limit,
                "Algorithm": algorithm,
                "Runs": str(len(group)),
                "Objective Mean": _format_number(sum(objectives) / len(objectives)),
                "Objective Std": _format_number(sample_std(objectives)),
                "Paper Makespan Mean": _format_number(sum(paper_makespans) / len(paper_makespans)),
                "Paper Makespan Std": _format_number(sample_std(paper_makespans)),
                "Time Mean": _format_number(sum(times) / len(times)),
                "Time Std": _format_number(sample_std(times)),
                "Seeds": ";".join(str(row["Seed"]) for row in sorted(group, key=lambda item: int(item["Seed"]))),
            }
        )
    return sorted(summary, key=lambda row: (row["Instance"], int(row["Trucks Submitted"]), float(row["Flight Limit"]), row["Algorithm"]))


def _write_csv(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(list(rows), indent=2), encoding="utf-8")


def _render_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    body = [[str(row.get(header, "")) for header in headers] for row in rows]
    widths = [len(header) for header in headers]
    for cells in body:
        for idx, cell in enumerate(cells):
            widths[idx] = max(widths[idx], len(cell))

    def line(cells: Sequence[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(cells)) + " |"

    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([line(headers), sep, *(line(cells) for cells in body)]) + "\n"


def _write_markdown(rows: Sequence[Mapping[str, Any]], output: Path, title: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"# {title}\n\n" + _render_markdown(rows), encoding="utf-8")


def _solver_kwargs(algorithm: str, args: argparse.Namespace) -> Dict[str, Any]:
    if algorithm == "vns":
        return {"k_max": args.vns_k_max, "max_iter": args.vns_max_iter}
    if algorithm == "lns":
        return {"destroy_frac": args.lns_destroy_frac, "max_iter": args.lns_max_iter}
    return {}


def _row_payload(row: Any, trial: BenchmarkTrial, trial_index: int, params: str) -> Dict[str, Any]:
    payload = row.as_csv_row()
    return {
        "Trial": trial_index,
        "Seed": trial.seed,
        "Parameters": params,
        **payload,
    }


def _algorithm_params(algorithm: str, args: argparse.Namespace) -> str:
    if algorithm == "vns":
        return f"vns_k_max={args.vns_k_max};vns_max_iter={args.vns_max_iter}"
    if algorithm == "lns":
        return f"lns_destroy_frac={args.lns_destroy_frac};lns_max_iter={args.lns_max_iter}"
    return "vnd_default"


def run_experiment(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    configs = load_dataset_configs(args.dataset_config)
    trials = build_trials(configs, algorithms=args.algorithms, seeds=args.seeds)

    instance_cache: Dict[Tuple[str, int, float], Tuple[Any, Any]] = {}
    raw_rows: List[Dict[str, Any]] = []

    for idx, trial in enumerate(trials, start=1):
        instance_path = args.instance_dir / trial.dataset.instance_name
        cache_key = (trial.dataset.instance_name, trial.dataset.num_trucks, trial.dataset.flight_limit)
        if cache_key not in instance_cache:
            instance_cache[cache_key] = build_instance(
                instance_path,
                num_trucks=trial.dataset.num_trucks,
                flight_limit=trial.dataset.flight_limit,
                base_vertex=args.base_vertex,
            )
        instance, config_template = instance_cache[cache_key]
        config = deepcopy(config_template)
        config.flight_optimizer = args.flight_optimizer
        config.bc_time_limit = args.bc_time_limit
        config.bc_mip_gap = args.bc_mip_gap
        config.bc_enable_connectivity_cuts = args.bc_enable_connectivity_cuts
        config.bc_enable_r_odd_cuts = args.bc_enable_r_odd_cuts
        config.bc_enable_advanced_cuts = args.bc_enable_advanced_cuts
        config.bc_cut_at_fractional_nodes = args.bc_cut_at_fractional_nodes
        config.bc_cut_at_integer_solutions = args.bc_cut_at_integer_solutions
        config.bc_max_cuts_per_round = args.bc_max_cuts_per_round
        config.max_construction_solutions = args.max_construction_solutions
        config.fast_construction_target = args.fast_construction_target
        config.fast_construction_seconds = args.fast_construction_seconds
        config.max_construction_seconds = args.max_construction_seconds
        config.max_construction_attempts = args.max_construction_attempts
        config.max_stall_attempts = args.max_stall_attempts
        config.split_top_k = args.split_top_k
        config.time_limit_seconds = args.component_time_limit if args.component_time_limit > 0 else None
        config.verbose = not args.quiet

        params = _algorithm_params(trial.algorithm, args)
        row = run_one_algorithm(
            name=trial.algorithm,
            solver_cls=SOLVER_CLASSES[trial.algorithm],
            instance=instance,
            config_template=config,
            seed=trial.seed,
            verbose=not args.quiet,
            solver_kwargs=_solver_kwargs(trial.algorithm, args),
        )
        payload = _row_payload(row, trial, idx, params)
        raw_rows.append(payload)
        print(
            f"[{idx}/{len(trials)}] {trial.dataset.instance_name} P={trial.dataset.num_trucks} "
            f"L={trial.dataset.flight_limit:g} {trial.algorithm} seed={trial.seed} "
            f"objective={payload['Objective']} time={payload['Time']}s"
        )

    summary_rows = aggregate_rows(raw_rows)
    _write_csv(raw_rows, args.output_raw_csv)
    _write_json(raw_rows, args.output_raw_json)
    _write_markdown(raw_rows, args.output_raw_md, "Multi-seed Benchmark Raw Results")
    _write_csv(summary_rows, args.output_summary_csv)
    _write_json(summary_rows, args.output_summary_json)
    _write_markdown(summary_rows, args.output_summary_md, "Multi-seed Benchmark Mean/Std Summary")
    return raw_rows, summary_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VNS/LNS/VND on multiple datasets and seeds, then aggregate mean/std.")
    parser.add_argument("--instance-dir", type=Path, default=Path("data/instances"))
    parser.add_argument("--dataset-config", type=Path, default=None, help="Optional CSV-like file: instance,num_trucks,flight_limit")
    parser.add_argument("--algorithms", type=parse_algorithms, default=DEFAULT_ALGORITHMS)
    parser.add_argument("--seeds", type=parse_csv_ints, default=DEFAULT_SEEDS)
    parser.add_argument("--base-vertex", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--lns-destroy-frac", type=float, default=0.4)
    parser.add_argument("--lns-max-iter", type=int, default=200)
    parser.add_argument("--vns-k-max", type=int, default=2)
    parser.add_argument("--vns-max-iter", type=int, default=100)

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
    parser.add_argument("--split-top-k", type=int, default=10)
    parser.add_argument("--component-time-limit", type=float, default=9.5)

    parser.add_argument("--output-raw-csv", type=Path, default=Path("docs/multiseed_raw.csv"))
    parser.add_argument("--output-raw-json", type=Path, default=Path("docs/multiseed_raw.json"))
    parser.add_argument("--output-raw-md", type=Path, default=Path("docs/multiseed_raw.md"))
    parser.add_argument("--output-summary-csv", type=Path, default=Path("docs/multiseed_summary.csv"))
    parser.add_argument("--output-summary-json", type=Path, default=Path("docs/multiseed_summary.json"))
    parser.add_argument("--output-summary-md", type=Path, default=Path("docs/multiseed_summary.md"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    raw_rows, summary_rows = run_experiment(args)
    print(f"Wrote {len(raw_rows)} raw rows to {args.output_raw_csv.resolve()}")
    print(f"Wrote {len(summary_rows)} summary rows to {args.output_summary_csv.resolve()}")
    print(f"Wrote Markdown summary to {args.output_summary_md.resolve()}")


if __name__ == "__main__":
    main()
