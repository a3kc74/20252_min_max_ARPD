"""LNSSolver — Large Neighborhood Search for MM-MT-dLARP.

LNS repeatedly destroys a large portion of the current solution
(more tasks than VND's destroy-and-repair) and repairs it with a
greedy best-insertion procedure, accepting the result by a configurable
acceptance criterion.

References:
    Shaw (1998) — Using Constraint Programming and Local Search Methods
                  to Solve Vehicle Routing Problems.
    Paper Section 4.2 for the repair/insertion logic reused here.
"""
from __future__ import annotations

import math
import random
import time
from typing import Callable, List, Optional, Tuple

from ..discretize import (
    add_midpoints_to_all_intervals,
    build_discrete_instance,
    convert_solution_between_instances,
    convert_solution_to_refined_instance,
    detect_used_midpoints,
    initial_breakpoints,
    map_parent_children,
    refine_breakpoints_from_used_midpoints,
)
from ..models import DiscreteInstance, Solution, Task
from .base import MatheuristicBase, SolverConfig


# ---------------------------------------------------------------------------
# Acceptance criteria
# ---------------------------------------------------------------------------

def accept_improving(candidate: Solution, current: Solution, **_) -> bool:
    """Accept only strict improvements (greedy / hill-climbing)."""
    return candidate.objective + 1e-9 < current.objective


def accept_simulated_annealing(
    candidate: Solution,
    current: Solution,
    temperature: float = 100.0,
    rng: Optional[random.Random] = None,
    **_,
) -> bool:
    """Accept improvements always; accept worse solutions with probability exp(-Δ/T)."""
    delta = candidate.objective - current.objective
    if delta < 0:
        return True
    if temperature <= 0:
        return False
    rng = rng or random
    return rng.random() < math.exp(-delta / temperature)


# ---------------------------------------------------------------------------
# LNS solver
# ---------------------------------------------------------------------------

class LNSSolver(MatheuristicBase):
    """Large Neighborhood Search for MM-MT-dLARP.

    Parameters
    ----------
    instance     : DiscreteInstance
    config       : SolverConfig
    destroy_frac : float
        Fraction of tasks to remove per destroy step (default 0.3).
    max_iter     : int
        Number of LNS outer iterations.
    accept       : callable
        Acceptance criterion; signature ``(candidate, current, **kw) -> bool``.
        Defaults to ``accept_improving``.
    """

    def __init__(
        self,
        instance: DiscreteInstance,
        config: SolverConfig,
        destroy_frac: float = 0.3,
        max_iter: int = 200,
        accept: Callable = accept_improving,
    ) -> None:
        super().__init__(instance, config)
        self.destroy_frac = destroy_frac
        self.max_iter = max_iter
        self.accept = accept

    # ------------------------------------------------------------------
    # Destroy operator
    # ------------------------------------------------------------------
    def _destroy(self, solution: Solution) -> Tuple[Solution, List[Task]]:
        """Remove a random fraction of tasks from the solution."""
        candidate = solution.clone()
        all_tasks: List[Tuple[int, int, int, Task]] = []
        for launch, flights in candidate.flights_by_launch.items():
            for f_idx, flight in enumerate(flights):
                for pos, task in enumerate(flight.tasks):
                    all_tasks.append((launch, f_idx, pos, task))

        n_remove = max(1, int(len(all_tasks) * self.destroy_frac))
        selected = sorted(
            self.rng.sample(all_tasks, k=min(n_remove, len(all_tasks))),
            key=lambda x: (x[1], x[2]),
            reverse=True,
        )
        removed: List[Task] = []
        for launch, f_idx, pos, task in selected:
            removed.append(candidate.flights_by_launch[launch][f_idx].tasks.pop(pos))
        self.evaluate(candidate)
        return candidate, removed

    # ------------------------------------------------------------------
    # Repair operator  (greedy best-insertion)
    # ------------------------------------------------------------------
    def _repair(self, solution: Solution, removed: List[Task]) -> Optional[Solution]:
        """Reinsert removed tasks at the best feasible position."""
        current = solution
        for task in removed:
            best_sol = None
            best_obj = float("inf")
            for launch, f_idx, pos, oriented in self.all_possible_insertions(current, task):
                inserted = self.insert_task(current, launch, f_idx, pos, oriented)
                if inserted is not None and inserted.objective < best_obj:
                    best_sol = inserted
                    best_obj = inserted.objective
            if best_sol is None:
                return None
            current = best_sol
        return current

    # ------------------------------------------------------------------
    # Splitting phase  (same coarse-to-fine mechanism as VNDSolver)
    # ------------------------------------------------------------------
    def splitting_phase(self, base_solution: Solution) -> Tuple[Solution, DiscreteInstance]:
        """Coarse-to-fine discretization loop reused by LNS top candidates."""
        if self.time_limit_reached():
            return base_solution, self.instance

        best_solution = base_solution
        best_instance = self.instance
        current_breakpoints = dict(self.instance.selected_breakpoints)
        current_instance = self.instance

        expanded_breakpoints = add_midpoints_to_all_intervals(current_instance.raw, current_breakpoints)
        if expanded_breakpoints == current_breakpoints:
            return best_solution, best_instance

        refined_instance = build_discrete_instance(
            current_instance.raw,
            base_vertex=current_instance.base_vertex,
            launch_vertices=current_instance.launch_vertices,
            selected_breakpoints=expanded_breakpoints,
        )
        parent_children = map_parent_children(current_breakpoints, expanded_breakpoints)
        converted = convert_solution_to_refined_instance(
            base_solution, current_instance, refined_instance, parent_children
        )

        refined_solver = LNSSolver(
            refined_instance,
            self.config,
            destroy_frac=self.destroy_frac,
            max_iter=self.max_iter,
            accept=self.accept,
        )
        refined_solution = refined_solver.vnd_improvement(
            converted,
            optimize_bottleneck=False,
        )

        if refined_solution.objective + 1e-9 >= best_solution.objective:
            return best_solution, best_instance

        best_solution = refined_solution
        best_instance = refined_instance
        current_instance = refined_instance
        current_solution = refined_solution
        current_breakpoints = expanded_breakpoints
        self.instance = refined_instance
        self._distance_cache.clear()

        while True:
            if self.time_limit_reached():
                break
            used = detect_used_midpoints(current_solution, current_instance)
            if not any(used.values()):
                break

            next_breakpoints = refine_breakpoints_from_used_midpoints(
                current_instance.raw, current_breakpoints, used
            )
            if next_breakpoints == current_breakpoints:
                break

            next_instance = build_discrete_instance(
                current_instance.raw,
                base_vertex=current_instance.base_vertex,
                launch_vertices=current_instance.launch_vertices,
                selected_breakpoints=next_breakpoints,
            )
            next_solution = convert_solution_between_instances(
                current_solution, current_instance, next_instance
            )

            solver = LNSSolver(
                next_instance,
                self.config,
                destroy_frac=self.destroy_frac,
                max_iter=self.max_iter,
                accept=self.accept,
            )
            improved_solution = solver.vnd_improvement(
                next_solution,
                optimize_bottleneck=False,
            )

            if improved_solution.objective + 1e-9 >= best_solution.objective:
                break

            best_solution = improved_solution
            best_instance = next_instance
            current_instance = next_instance
            current_solution = improved_solution
            current_breakpoints = next_breakpoints
            self.instance = next_instance
            self._distance_cache.clear()

        return best_solution, best_instance

    def _run_lns_from(
        self,
        start_solution: Solution,
        started: float,
        candidate_idx: int,
        global_best: Solution,
    ) -> Solution:
        """Run the LNS main loop from one initial candidate.

        Convergence entries are recorded only when this candidate improves the
        global incumbent. The stored iteration is the local iteration inside the
        current candidate, and the optional metadata identifies the candidate.
        """
        iteration_searches = (
            "intraroute_move",
            # "zero_to_l_exchange",
        )
        current = start_solution
        best = current
        incumbent = global_best

        for iteration in range(1, self.max_iter + 1):
            if self.time_limit_reached():
                break

            destroyed, removed = self._destroy(current)
            repaired = self._repair(destroyed, removed)
            if repaired is None:
                continue

            repaired = self.vnd_improvement(
                repaired,
                local_searches=iteration_searches,
                optimize_bottleneck=False,
            )

            if self.accept(repaired, current, rng=self.rng):
                current = repaired
                if current.objective + 1e-9 < best.objective:
                    best = current
                    if best.objective + 1e-9 < incumbent.objective:
                        incumbent = best
                        self.convergence_log.append(
                            (
                                iteration,
                                time.time() - started,
                                best.clone(),
                                {
                                    "phase": "lns",
                                    "candidate": candidate_idx + 1,
                                    "local_iteration": iteration,
                                },
                            )
                        )

        return best

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def solve(self, initial: Optional[Solution] = None) -> Tuple[Solution, DiscreteInstance]:
        started = time.time()

        if initial is None:
            pool = self.generate_initial_pool()
            improved_pool = []
            for sol in pool:
                if self.time_limit_reached() and improved_pool:
                    break
                # Pool initialization keeps the full 4-operator VND, but skips bottleneck optimization.
                improved_pool.append(self.vnd_improvement(sol, optimize_bottleneck=False))
        else:
            improved_pool = [self.vnd_improvement(initial, optimize_bottleneck=False)]

        improved_pool.sort(key=lambda s: s.objective)
        top = improved_pool[: min(self.config.split_top_k, len(improved_pool))]
        best = min(improved_pool, key=lambda s: s.objective)
        best_instance = self.instance

        elapsed_init = time.time() - started
        self.convergence_log: list = [
            (
                0,
                elapsed_init,
                best.clone(),
                {
                    "phase": "initial_pool",
                    "candidate": None,
                    "local_iteration": 0,
                },
            )
        ]

        self._log(f"solve | initial pool size={len(improved_pool)}")
        self._log(f"solve | running LNS from top {len(top)} initial candidates")

        lns_candidates: List[Solution] = []
        for idx, sol in enumerate(top):
            if self.time_limit_reached():
                break
            candidate_best = self._run_lns_from(sol, started, idx, best)
            final = self.vnd_improvement(candidate_best)
            if final.objective + 1e-9 < candidate_best.objective:
                candidate_best = final

            lns_candidates.append(candidate_best)
            if candidate_best.objective + 1e-9 < best.objective:
                best = candidate_best
                self.convergence_log.append(
                    (
                        self.max_iter,
                        time.time() - started,
                        best.clone(),
                        {
                            "phase": "candidate_vnd",
                            "candidate": idx + 1,
                            "local_iteration": self.max_iter,
                        },
                    )
                )

        split_candidates = sorted(lns_candidates or top, key=lambda s: s.objective)[
            : min(self.config.split_top_k, len(lns_candidates or top))
        ]

        self._log(f"solve | applying splitting phase to top {len(split_candidates)} LNS candidates")
        for idx, sol in enumerate(split_candidates):
            if self.time_limit_reached():
                break

            self.instance = build_discrete_instance(
                self.instance.raw,
                base_vertex=self.instance.base_vertex,
                launch_vertices=self.instance.launch_vertices,
                selected_breakpoints=initial_breakpoints(self.instance.raw),
            )
            self._distance_cache.clear()

            split_candidate, split_instance = self.splitting_phase(sol)
            self._log(f"solve | split candidate {idx + 1}/{len(split_candidates)} obj={self._fmt_obj(split_candidate.objective)}")

            if split_candidate.objective + 1e-9 < best.objective:
                best = split_candidate
                best_instance = split_instance
                self.convergence_log.append(
                    (
                        idx + 1,
                        time.time() - started,
                        best.clone(),
                        {
                            "phase": "splitting",
                            "candidate": idx + 1,
                            "local_iteration": idx + 1,
                        },
                    )
                )

        self.instance = best_instance
        self._distance_cache.clear()

        final = self.evaluate(best)
        return final, best_instance
