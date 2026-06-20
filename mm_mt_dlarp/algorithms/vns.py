"""VNSSolver — Variable Neighborhood Search for MM-MT-dLARP.

Extends MatheuristicBase with shaking to escape local optima.

This implementation uses a VNS/VND hybrid:
1. build or receive an initial solution,
2. improve it with a light VND,
3. repeatedly perturb (shake) the incumbent,
4. re-optimize the shaken solution with light VND,
5. accept strict improvements or controlled record-to-record moves,
6. polish the best solution with the full VND at the end.

References:
    Hansen & Mladenović (2001) — Variable Neighborhood Search.
    Paper Section 4.2 for the neighborhood definitions reused here.
"""
from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

from ..models import DiscreteInstance, Solution, Task
from .base import MatheuristicBase, SolverConfig


class VNSSolver(MatheuristicBase):
    """VNS: shaking + VND local search to escape local optima.

    Parameters
    ----------
    instance : DiscreteInstance
    config   : SolverConfig
    k_max    : int
        Maximum neighborhood index before restarting from k=1.
    max_iter : int
        Total number of VNS outer iterations.
    initial_deviation:
        Initial record-to-record acceptance tolerance as a ratio of the current
        incumbent objective.
    min_deviation:
        Lower bound for the adaptive acceptance tolerance.
    max_deviation:
        Upper bound for the adaptive acceptance tolerance.
    deviation_decay:
        Multiplier applied after a global-best improvement.
    deviation_growth:
        Multiplier applied after a non-improving outer iteration.
    stagnation_patience:
        Number of consecutive outer iterations without a global-best
        improvement before restarting from the global best with a strong shake.
    """

    def __init__(
        self,
        instance: DiscreteInstance,
        config: SolverConfig,
        k_max: int = 4,
        max_iter: int = 100,
        initial_deviation: float = 0.01,
        min_deviation: float = 0.001,
        max_deviation: float = 0.05,
        deviation_decay: float = 0.90,
        deviation_growth: float = 1.10,
        stagnation_patience: int = 30,
    ) -> None:
        super().__init__(instance, config)
        self.k_max = max(1, k_max)
        self.max_iter = max_iter
        self.initial_deviation = max(0.0, initial_deviation)
        self.min_deviation = max(0.0, min_deviation)
        self.max_deviation = max(self.min_deviation, max_deviation)
        self.deviation_decay = deviation_decay
        self.deviation_growth = deviation_growth
        self.stagnation_patience = max(1, stagnation_patience)

    # ------------------------------------------------------------------
    # Shaking  (random perturbation to escape local optima)
    # ------------------------------------------------------------------
    def _task_locations(self, solution: Solution) -> List[Tuple[int, int, int, Task]]:
        """Return all task locations as (launch, flight_idx, pos, task)."""
        locations: List[Tuple[int, int, int, Task]] = []
        for launch, flights in solution.flights_by_launch.items():
            for flight_idx, flight in enumerate(flights):
                for pos, task in enumerate(flight.tasks):
                    locations.append((launch, flight_idx, pos, task))
        return locations

    def _remove_task_at(
        self,
        solution: Solution,
        launch: int,
        flight_idx: int,
        pos: int,
    ) -> Tuple[Optional[Solution], Optional[Task]]:
        """Remove one task from a cloned solution and re-evaluate it."""
        candidate = solution.clone()
        flights = candidate.flights_by_launch.get(launch, [])
        if flight_idx >= len(flights) or pos >= len(flights[flight_idx].tasks):
            return None, None

        removed = flights[flight_idx].tasks.pop(pos)
        if not flights[flight_idx].tasks:
            flights.pop(flight_idx)

        return self.evaluate(candidate), removed

    def _random_feasible_reinsert(self, partial: Solution, task: Task) -> Optional[Solution]:
        """Reinsert a removed task into a random feasible position.

        Unlike ``destroy_and_repair`` this does not require an improving move.
        It samples feasible insertions and therefore can create a worse-but-valid
        perturbed solution for VNS.
        """
        options = self.all_possible_insertions(partial, task)
        self.rng.shuffle(options)
        for launch, flight_idx, pos, oriented in options:
            inserted = self.insert_task(partial, launch, flight_idx, pos, oriented)
            if inserted is not None:
                return inserted
        return None

    def _random_task_relocate(self, solution: Solution, moves: int) -> Solution:
        """Apply random task relocation moves while preserving feasibility."""
        current = solution.clone()
        for _ in range(max(1, moves)):
            locations = self._task_locations(current)
            if not locations:
                break

            launch, flight_idx, pos, _ = self.rng.choice(locations)
            partial, removed = self._remove_task_at(current, launch, flight_idx, pos)
            if partial is None or removed is None:
                continue

            reinserted = self._random_feasible_reinsert(partial, removed)
            if reinserted is not None and self.is_feasible_solution(reinserted):
                current = reinserted

        return self.evaluate(current)

    def _bottleneck_task_relocate(self, solution: Solution, moves: int) -> Solution:
        """Relocate tasks sampled from the current bottleneck launch."""
        current = solution.clone()
        for _ in range(max(1, moves)):
            bottleneck = self.bottleneck_launch(current)
            if bottleneck is None:
                break

            locations: List[Tuple[int, int, int, Task]] = []
            for flight_idx, flight in enumerate(current.flights_by_launch.get(bottleneck, [])):
                for pos, task in enumerate(flight.tasks):
                    locations.append((bottleneck, flight_idx, pos, task))

            if not locations:
                break

            launch, flight_idx, pos, _ = self.rng.choice(locations)
            partial, removed = self._remove_task_at(current, launch, flight_idx, pos)
            if partial is None or removed is None:
                continue

            reinserted = self._random_feasible_reinsert(partial, removed)
            if reinserted is not None and self.is_feasible_solution(reinserted):
                current = reinserted

        return self.evaluate(current)

    def _destroy_repair_perturb(self, solution: Solution, moves: int) -> Solution:
        """Use existing destroy-and-repair if it improves, otherwise fall back.

        ``destroy_and_repair`` is an improvement neighbourhood in
        ``MatheuristicBase`` and returns ``None`` for non-improving candidates.
        For shaking, VNS still needs real movement on plateaus/local optima, so
        the fallback relocation operators guarantee perturbation opportunities.
        """
        current = solution.clone()
        for _ in range(max(1, moves)):
            repaired = self.destroy_and_repair(current)
            if repaired is not None:
                current = repaired
            else:
                current = self._random_task_relocate(current, 1)
        return self.evaluate(current)

    def _shake(self, solution: Solution, k: int, no_improve_iters: int) -> Solution:
        """Apply a real perturbation whose strength adapts to stagnation.

        The shaking strength increases with both neighbourhood index ``k`` and
        stagnation. The operator also rotates by ``k`` so consecutive
        neighbourhoods are structurally different rather than repeated calls to
        one improving-only neighbourhood.
        """
        stagnation_boost = 1 + no_improve_iters // max(1, self.stagnation_patience // 3)
        strength = max(1, k * stagnation_boost)

        operator_idx = (k - 1) % 3
        if operator_idx == 0:
            return self._random_task_relocate(solution, strength)
        if operator_idx == 1:
            return self._bottleneck_task_relocate(solution, strength)
        return self._destroy_repair_perturb(solution, strength)

    # ------------------------------------------------------------------
    # Acceptance and restart logic
    # ------------------------------------------------------------------
    def _accepted_by_record_to_record(
        self,
        candidate: Solution,
        current: Solution,
        best: Solution,
        deviation: float,
    ) -> bool:
        """Return whether candidate should replace current.

        Always accept strict incumbent improvements. Otherwise, accept candidates
        that remain within an adaptive deviation from the global-best record.
        This enables controlled diversification without drifting too far from
        the best known solution.
        """
        if candidate.objective + 1e-9 < current.objective:
            return True

        reference = best.objective if math.isfinite(best.objective) else current.objective
        threshold = reference * (1.0 + deviation)
        return candidate.objective <= threshold + 1e-9

    def _restart_from_best(self, best: Solution, no_improve_iters: int) -> Solution:
        """Restart around the global best using a strong shake and light repair."""
        strong_k = max(self.k_max, self.k_max + no_improve_iters // self.stagnation_patience)
        restarted = self._shake(best, strong_k, no_improve_iters)
        return restarted

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def solve(self, initial: Optional[Solution] = None) -> Tuple[Solution, DiscreteInstance]:
        started = time.time()

        light_vnd = ("intraroute_move", "zero_to_l_exchange")
        if initial is None:
            pool = self.generate_initial_pool()
            current = min(
                (
                    self.vnd_improvement(
                        s,
                        local_searches=light_vnd,
                        optimize_bottleneck=False,
                    )
                    for s in pool
                ),
                key=lambda s: s.objective,
            )
        else:
            current = self.vnd_improvement(
                initial,
                local_searches=light_vnd,
                optimize_bottleneck=False,
            )

        best = current
        elapsed_init = time.time() - started
        self.convergence_log: list = [(0, elapsed_init, best.clone())]

        deviation = min(self.max_deviation, max(self.min_deviation, self.initial_deviation))
        no_improve_iters = 0
        iteration = 0

        while iteration < self.max_iter and time.time() < self._deadline:
            improved_best_this_iter = False
            accepted_this_iter = False
            k = 1

            while k <= self.k_max and time.time() < self._deadline:
                shaken = self._shake(current, k, no_improve_iters)
                improved = self.vnd_improvement(
                    shaken,
                    local_searches=light_vnd,
                    optimize_bottleneck=False,
                )

                if improved.objective + 1e-9 < best.objective:
                    best = improved
                    current = improved
                    self.convergence_log.append(
                        (iteration + 1, time.time() - started, best.clone())
                    )
                    deviation = max(self.min_deviation, deviation * self.deviation_decay)
                    no_improve_iters = 0
                    improved_best_this_iter = True
                    accepted_this_iter = True
                    k = 1
                    continue

                if self._accepted_by_record_to_record(improved, current, best, deviation):
                    current = improved
                    accepted_this_iter = True
                    # Record-to-record accepts may be non-improving. Move to
                    # the next neighbourhood to avoid cycling forever on
                    # accepted plateau/worse candidates.
                    k += 1
                else:
                    k += 1

            iteration += 1

            if improved_best_this_iter:
                continue

            no_improve_iters += 1
            deviation = min(self.max_deviation, deviation * self.deviation_growth)

            if no_improve_iters >= self.stagnation_patience:
                current = self._restart_from_best(best, no_improve_iters)
                current = self.vnd_improvement(
                    current,
                    local_searches=light_vnd,
                    optimize_bottleneck=False,
                )
                if current.objective + 1e-9 < best.objective:
                    best = current
                    self.convergence_log.append(
                        (iteration, time.time() - started, best.clone())
                    )
                no_improve_iters = 0
                deviation = min(self.max_deviation, max(self.min_deviation, self.initial_deviation))
            elif not accepted_this_iter:
                # Keep the search anchored when every neighbourhood rejects.
                current = best.clone()

        final = self.vnd_improvement(best)
        if final.objective + 1e-9 < best.objective:
            best = final
            self.convergence_log.append((iteration, time.time() - started, best.clone()))

        elapsed = time.time() - started
        if self.convergence_log[-1][0] != iteration:
            self.convergence_log.append((iteration, elapsed, best.clone()))
        return best, self.instance