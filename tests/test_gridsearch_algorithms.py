import unittest

from exp.gridsearch_algorithms import build_grid, parse_csv_floats, parse_csv_ints, select_best_rows


class GridSearchAlgorithmsTests(unittest.TestCase):
    def test_parse_csv_helpers(self):
        self.assertEqual(parse_csv_ints("1, 2,3"), [1, 2, 3])
        self.assertEqual(parse_csv_floats("0.1, .25"), [0.1, 0.25])

    def test_build_grid_contains_expected_algorithm_specific_params(self):
        grid = build_grid(
            algorithms=["vns", "lns", "vnd"],
            seeds=[0, 1],
            vns_k_max_values=[2, 3],
            vns_max_iter_values=[50],
            lns_destroy_frac_values=[0.2, 0.3],
            lns_max_iter_values=[100],
            vnd_split_top_k_values=[5],
        )

        self.assertEqual(len(grid), 10)
        self.assertIn(
            {
                "algorithm": "vns",
                "seed": 1,
                "config_overrides": {},
                "solver_kwargs": {"k_max": 3, "max_iter": 50},
                "param_label": "vns_k_max=3;vns_max_iter=50",
            },
            grid,
        )
        self.assertIn(
            {
                "algorithm": "lns",
                "seed": 0,
                "config_overrides": {},
                "solver_kwargs": {"destroy_frac": 0.2, "max_iter": 100},
                "param_label": "lns_destroy_frac=0.2;lns_max_iter=100",
            },
            grid,
        )
        self.assertIn(
            {
                "algorithm": "vnd",
                "seed": 0,
                "config_overrides": {"split_top_k": 5},
                "solver_kwargs": {},
                "param_label": "vnd_split_top_k=5",
            },
            grid,
        )

    def test_select_best_rows_prefers_objective_then_time(self):
        rows = [
            {"Algorithm": "vns", "Objective": 10.0, "Time": 4.0},
            {"Algorithm": "vns", "Objective": 9.0, "Time": 9.0},
            {"Algorithm": "lns", "Objective": 8.0, "Time": 5.0},
            {"Algorithm": "lns", "Objective": 8.0, "Time": 3.0},
        ]

        best = select_best_rows(rows)

        self.assertEqual(best["vns"]["Objective"], 9.0)
        self.assertEqual(best["lns"]["Time"], 3.0)


if __name__ == "__main__":
    unittest.main()
