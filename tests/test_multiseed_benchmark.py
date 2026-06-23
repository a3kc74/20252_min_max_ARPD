import math
import unittest
from pathlib import Path

from exp.multiseed_benchmark import (
    DatasetConfig,
    aggregate_rows,
    build_trials,
    parse_dataset_configs,
    sample_std,
)


class MultiSeedBenchmarkTests(unittest.TestCase):
    def test_parse_dataset_configs_from_text(self):
        configs = parse_dataset_configs("""
        MTLARP4_4_3_1.dat,2,847
        MTLARP4_4_3_2.dat, 2, 1033
        """)

        self.assertEqual(
            configs,
            [
                DatasetConfig("MTLARP4_4_3_1.dat", 2, 847.0),
                DatasetConfig("MTLARP4_4_3_2.dat", 2, 1033.0),
            ],
        )

    def test_build_trials_expands_datasets_algorithms_and_seeds(self):
        configs = [DatasetConfig("a.dat", 2, 10.0), DatasetConfig("b.dat", 3, 20.0)]
        trials = build_trials(configs, algorithms=["vns", "lns", "vnd"], seeds=[0, 1, 2])

        self.assertEqual(len(trials), 18)
        self.assertEqual(trials[0].dataset.instance_name, "a.dat")
        self.assertEqual(trials[0].algorithm, "vns")
        self.assertEqual(trials[0].seed, 0)
        self.assertEqual(trials[-1].dataset.instance_name, "b.dat")
        self.assertEqual(trials[-1].algorithm, "vnd")
        self.assertEqual(trials[-1].seed, 2)

    def test_sample_std(self):
        self.assertEqual(sample_std([5.0]), 0.0)
        self.assertTrue(math.isclose(sample_std([1.0, 2.0, 3.0]), 1.0))

    def test_aggregate_rows_groups_by_dataset_and_algorithm(self):
        rows = [
            {"Instance": "a.dat", "Trucks Submitted": 2, "Flight Limit": "10.000000", "Algorithm": "vns", "Objective": "10.000000", "Time": "1.000000", "Paper Makespan": "10.000000", "Seed": 0},
            {"Instance": "a.dat", "Trucks Submitted": 2, "Flight Limit": "10.000000", "Algorithm": "vns", "Objective": "12.000000", "Time": "3.000000", "Paper Makespan": "12.000000", "Seed": 1},
            {"Instance": "a.dat", "Trucks Submitted": 2, "Flight Limit": "10.000000", "Algorithm": "lns", "Objective": "8.000000", "Time": "2.000000", "Paper Makespan": "8.000000", "Seed": 0},
        ]

        summary = aggregate_rows(rows)

        self.assertEqual(len(summary), 2)
        self.assertEqual(summary[0]["Algorithm"], "lns")
        self.assertEqual(summary[0]["Objective Mean"], "8.000000")
        self.assertEqual(summary[0]["Objective Std"], "0.000000")
        self.assertEqual(summary[1]["Algorithm"], "vns")
        self.assertEqual(summary[1]["Objective Mean"], "11.000000")
        self.assertEqual(summary[1]["Objective Std"], "1.414214")
        self.assertEqual(summary[1]["Time Mean"], "2.000000")


if __name__ == "__main__":
    unittest.main()
