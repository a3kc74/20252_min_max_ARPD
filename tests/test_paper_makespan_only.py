import unittest
from dataclasses import fields

from mm_mt_dlarp.algorithms.base import MatheuristicBase, SolverConfig
from mm_mt_dlarp.models import DiscreteInstance, Flight, Instance, OriginalLine, RequiredEdge, Solution, Task, Vertex
from src.run_benchmark import BenchmarkRow, render_markdown


class PaperMakespanOnlyTests(unittest.TestCase):
    def _solver(self):
        raw = Instance(
            name="tiny",
            original_vertex_count=2,
            total_vertex_count=2,
            depot_vertices=(0,),
            vertices={0: Vertex(0, 0.0, 0.0), 1: Vertex(1, 3.0, 4.0)},
            lines=(OriginalLine(1, 0, 1, 7.0, (0, 1), (7.0,)),),
        )
        edge = RequiredEdge("1:0-1", 1, 0, 1, 0, 1, 7.0)
        instance = DiscreteInstance(
            raw=raw,
            base_vertex=0,
            launch_vertices=(0,),
            selected_breakpoints={1: (0, 1)},
            required_edges=(edge,),
            edge_by_id={edge.edge_id: edge},
            edge_by_key={edge.key(): edge},
        )
        return MatheuristicBase(instance, SolverConfig(num_trucks=1, flight_limit=100.0, verbose=False))

    def test_solver_config_has_no_objective_selector(self):
        config_fields = {field.name for field in fields(SolverConfig)}
        self.assertEqual({"objective"} & config_fields, set())
        self.assertFalse(any(name.endswith("_type") for name in config_fields))

    def test_evaluate_uses_paper_makespan_as_only_objective(self):
        solver = self._solver()
        solution = Solution(
            selected_launches=[0],
            flights_by_launch={0: [Flight(0, [Task("1:0-1")])]},
        )

        evaluated = solver.evaluate(solution)

        self.assertEqual(evaluated.objective, evaluated.paper_makespan)
        self.assertEqual(evaluated.makespan_by_launch, {0: evaluated.paper_makespan})
        solution_fields = {field.name for field in fields(Solution)}
        self.assertEqual({"objective", "paper_makespan"} & solution_fields, {"objective", "paper_makespan"})
        self.assertFalse(any(name.startswith("ghg") or name.startswith("total") for name in solution_fields))

    def test_benchmark_output_contains_only_paper_makespan_columns(self):
        row = BenchmarkRow(
            Algorithm="vnd",
            Instance="tiny.dat",
            Base=0,
            Flight_Limit=100.0,
            Trucks_Used=1,
            Trucks_Submitted=1,
            Objective=12.0,
            Paper_Makespan=12.0,
            Time=0.1,
            Flight_Optimizer="bc",
            Convergence_Iterations="initial@0;final@1",
            Convergence_Count=2,
            Convergence_First_Objective="12.000000",
            Convergence_Final_Objective="12.000000",
            Convergence_Improvement="0.000000",
            Convergence_Improvement_Percent="0.000000",
        )

        rendered = render_markdown([row])
        header = rendered.splitlines()[0]

        self.assertIn("Paper Makespan", header)
        self.assertIn("Objective", header)
        self.assertNotIn("Type", header)
        self.assertNotIn("Total", header)


if __name__ == "__main__":
    unittest.main()
