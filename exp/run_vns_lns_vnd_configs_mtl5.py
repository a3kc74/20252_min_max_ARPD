from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from exp.run_vns_lns_vnd_configs import FIELDS, _write_csv, _write_xlsx  # noqa: E402
from src.run_benchmark import build_instance, run_one_algorithm  # noqa: E402
from mm_mt_dlarp.algorithms import LNSSolver, VNDSolver, VNSSolver  # noqa: E402


CONFIGS: list[tuple[str, int, float]] = [
    ("MTLARP5_5_3_1.dat", 2, 1469),
    ("MTLARP5_5_3_2.dat", 2, 1368),
    ("MTLARP5_5_4_1.dat", 2, 1159),
    ("MTLARP5_5_4_1.dat", 3, 1159),
    ("MTLARP5_5_4_2.dat", 2, 1010),
    ("MTLARP5_5_4_2.dat", 3, 1010),
    ("MTLARP5_5_5_1.dat", 2, 955),
    ("MTLARP5_5_5_1.dat", 3, 955),
    ("MTLARP5_5_5_1.dat", 4, 955),
    ("MTLARP5_5_5_2.dat", 2, 988),
    ("MTLARP5_5_5_2.dat", 3, 988),
    ("MTLARP5_5_5_2.dat", 4, 988),
    ("MTLARP5_5_6_1.dat", 2, 783),
    ("MTLARP5_5_6_1.dat", 3, 783),
    ("MTLARP5_5_6_1.dat", 4, 783),
    ("MTLARP5_5_6_1.dat", 5, 783),
    ("MTLARP5_5_6_2.dat", 2, 1168),
    ("MTLARP5_5_6_2.dat", 3, 1168),
    ("MTLARP5_5_6_2.dat", 4, 1168),
    ("MTLARP5_5_6_2.dat", 5, 1168),
]

ALGORITHMS = [
    ("vns", VNSSolver, {"k_max": 4, "max_iter": 100}),
    ("lns", LNSSolver, {"destroy_frac": 0.3, "max_iter": 500}),
    ("vnd", VNDSolver, {}),
]


def main() -> None:
    output_dir = ROOT / "exp" / "results" / "vns_lns_vnd_configs_mtl5"
    output_csv = output_dir / "results.csv"
    output_xlsx = output_dir / "results.xlsx"

    rows: list[dict] = []
    total = len(CONFIGS) * len(ALGORITHMS)
    done = 0

    for instance_name, p, l_value in CONFIGS:
        instance_path = ROOT / "data" / "instances" / instance_name
        instance, config = build_instance(instance_path, p, l_value, base_vertex=None)
        config.objective_type = "minmax_ghg"
        config.flight_optimizer = "bc"
        config.verbose = False

        for algorithm_name, solver_cls, kwargs in ALGORITHMS:
            done += 1
            print(f"[{done:02d}/{total}] {algorithm_name.upper():<3} {instance_name} P={p} L={l_value}", flush=True)
            try:
                bench_row = run_one_algorithm(
                    name=algorithm_name,
                    solver_cls=solver_cls,
                    instance=instance,
                    config_template=config,
                    seed=0,
                    verbose=False,
                    solver_kwargs=kwargs,
                )
                row = bench_row.as_csv_row()
                row.update(
                    {
                        "P": p,
                        "L": l_value,
                        "Seed": 0,
                        "Status": "ok",
                        "Error": "",
                    }
                )
                print(
                    f"       objective={row['Objective']} paper_makespan={row['Paper Makespan']} "
                    f"total_ghg={row['Total GHG']} time={row['Time']}s",
                    flush=True,
                )
            except Exception as exc:
                row = {
                    "Algorithm": algorithm_name,
                    "Instance": instance_name,
                    "P": p,
                    "L": l_value,
                    "Base": "",
                    "Flight Limit": l_value,
                    "Trucks Used": "",
                    "Trucks Submitted": p,
                    "Objective Type": config.objective_type,
                    "Objective": "",
                    "Paper Makespan": "",
                    "GHG Makespan": "",
                    "Total GHG": "",
                    "Time": "",
                    "Flight Optimizer": config.flight_optimizer,
                    "Seed": 0,
                    "Status": "error",
                    "Error": repr(exc),
                }
                print(f"       ERROR: {exc!r}", flush=True)

            rows.append(row)
            _write_csv(rows, output_csv)
            _write_xlsx(rows, output_xlsx)

    print(f"Wrote CSV : {output_csv}")
    print(f"Wrote XLSX: {output_xlsx}")


if __name__ == "__main__":
    started = time.perf_counter()
    main()
    print(f"Total wall time: {time.perf_counter() - started:.2f}s")