"""VNDSolver — shared matheuristic pipeline with configurable improvement.

Extends MatheuristicBase with:
  - splitting_phase(): coarse-to-fine discretization loop (paper Section 4.3)
  - solve():           full orchestration (pool → improvement → split)
"""
from __future__ import annotations

import time
from typing import Tuple

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
from ..models import DiscreteInstance, Solution
from .base import MatheuristicBase, SolverConfig


class VNDSolver(MatheuristicBase):
    """Full matheuristic solver with a configurable improvement component."""

    # ------------------------------------------------------------------
    # Splitting phase  (paper Section 4.3)
    # ------------------------------------------------------------------
    def splitting_phase(self, base_solution: Solution) -> Tuple[Solution, DiscreteInstance]:
        """Coarse-to-fine discretization loop.

        Step 1 – add midpoint to every required edge, convert, improve.
        Step 2+ – for each used midpoint add midpoints to incident edges;
                   remove unused midpoints; improve.  Repeat until no improvement.
        """
        if self.time_limit_reached():
            return base_solution, self.instance

        best_solution = base_solution
        best_instance = self.instance
        current_breakpoints = dict(self.instance.selected_breakpoints)
        current_instance = self.instance

        # Step 1: expand ALL edges with their midpoints
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

        refined_solver = VNDSolver(refined_instance, self.config)
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

        # Steps 2+: detect used midpoints → add adjacent midpoints → improve; repeat
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

            solver = VNDSolver(next_instance, self.config)
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

    # ------------------------------------------------------------------
    # Top-level solve
    # ------------------------------------------------------------------
    def solve(self) -> Tuple[Solution, DiscreteInstance]:
        self._log("solve | starting matheuristic solve")
        started = time.time()
        previous_deadline = self.config.search_deadline
        if self.config.time_limit_seconds is not None:
            self.config.search_deadline = started + self.config.time_limit_seconds
        # -----------------------------------------------------------
        try:
            pool = self.generate_initial_pool()
        finally:
            if previous_deadline is not None and self.config.time_limit_seconds is None:
                self.config.search_deadline = previous_deadline
        self._log(f"solve | initial pool size={len(pool)}")

        # -----------------------------------------------------------
        improved_pool = []
        for sol in pool:
            if self.time_limit_reached() and improved_pool:
                break
            improved_pool.append(self.vnd_improvement(sol))
        self._log("solve | finished VND on initial pool")

        # -----------------------------------------------------------
        improved_pool.sort(key=lambda s: s.objective)
        top = improved_pool[: min(self.config.split_top_k, len(improved_pool))]
        best = min(improved_pool, key=lambda s: s.objective)
        best_instance = self.instance
        self.best_initial_objective = best.objective
        self.initial_improvement_elapsed = time.time() - started
        self.convergence_log: list = [(0, self.initial_improvement_elapsed, best.clone())]

        self._log(f"solve | best initial objective={self._fmt_obj(best.objective)}")

        for idx, sol in enumerate(top):
            if self.time_limit_reached():
                break
            self._log(f"solve | splitting candidate {idx + 1}/{len(top)}")

            self.instance = build_discrete_instance(
                self.instance.raw,
                base_vertex=self.instance.base_vertex,
                launch_vertices=self.instance.launch_vertices,
                selected_breakpoints=initial_breakpoints(self.instance.raw),
            )
            self._distance_cache.clear()

            split_candidate, split_instance = self.splitting_phase(sol)
            self._log(f"solve | split candidate obj={self._fmt_obj(split_candidate.objective)}")

            if split_candidate.objective < best.objective:
                self._log("solve | new best found after splitting")
                best = split_candidate
                best_instance = split_instance

        self.instance = best_instance
        self._distance_cache.clear()

        final = self.evaluate(best)
        elapsed = time.time() - started
        if self.convergence_log[-1][0] == 0:
            self.convergence_log.append((1, elapsed, final.clone()))
        self._log(f"solve | finished | final objective={self._fmt_obj(final.objective)}")

        self.config.search_deadline = previous_deadline
        return final, best_instance
