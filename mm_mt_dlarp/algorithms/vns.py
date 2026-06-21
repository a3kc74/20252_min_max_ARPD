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
from typing import List, Optional, Sequence, Tuple

from ..discretize import (
    add_midpoints_to_all_intervals,
    build_discrete_instance,
    convert_solution_between_instances,
    convert_solution_to_refined_instance,
    detect_used_midpoints,
    map_parent_children,
    refine_breakpoints_from_used_midpoints,
)
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
        improvement before increasing diversification pressure.
    restart_patience:
        Number of consecutive outer iterations without a global-best
        improvement before restarting from elite memory or a fresh pool.
    elite_size:
        Maximum number of high-quality distinct solutions stored in elite memory.
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
        restart_patience: Optional[int] = None,
        elite_size: int = 5,
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
        self.restart_patience = max(
            self.stagnation_patience,
            restart_patience if restart_patience is not None else self.stagnation_patience,
        )
        self.elite_size = max(1, elite_size)
        self.elite_pool: List[Solution] = []
        self.shake_operators = (
            "random_task_relocate",
            "bottleneck_task_relocate",
            "destroy_repair_perturb",
        )
        self.operator_stats = {
            name: {
                "uses": 0,
                "accepted": 0,
                "best_improvements": 0,
                "current_improvements": 0,
                "feasible": 0,
                "reward": 0.0,
            }
            for name in self.shake_operators
        }
        self.adaptive_log: List[dict] = []
        self.vnd_operator_stats = {
            name: {"uses": 0, "improvements": 0, "reward": 0.0}
            for name in (
                "intraroute_move",
                "destroy_and_repair",
                "zero_to_l_exchange",
                "l1_l2_exchange",
            )
        }

    # ------------------------------------------------------------------
    # Adaptive local search ordering
    # ------------------------------------------------------------------
    def _ordered_local_searches(
        self,
        base_order: Optional[Sequence[str]] = None,
        exploration_rotation: int = 0,
    ) -> Tuple[str, ...]:
        """Return local-search operators ordered by observed VNS reward.

        Operators with no history keep their original position. Once rewards are
        available, the order favours neighbourhoods that have produced stronger
        objective reductions in previous VNS calls while preserving deterministic
        tie-breaking through the base order.
        """
        if base_order is None:
            base_order = (
                "intraroute_move",
                "destroy_and_repair",
                "zero_to_l_exchange",
                "l1_l2_exchange",
            )

        indexed = list(enumerate(base_order))

        def score(item: Tuple[int, str]) -> Tuple[float, int]:
            idx, name = item
            stats = self.vnd_operator_stats.get(name, {"uses": 0, "reward": 0.0})
            uses = stats["uses"]
            if uses == 0:
                return (0.0, -idx)
            return (stats["reward"] / uses, -idx)

        ordered = [name for _, name in sorted(indexed, key=score, reverse=True)]
        if ordered and exploration_rotation:
            shift = exploration_rotation % len(ordered)
            ordered = ordered[shift:] + ordered[:shift]
        return tuple(ordered)

    def _adaptive_vnd(
        self,
        solution: Solution,
        base_order: Optional[Sequence[str]] = None,
        optimize_bottleneck: bool = False,
        exploration_rotation: int = 0,
    ) -> Solution:
        """Run VND with adaptive ordering and update per-operator rewards.

        Because ``MatheuristicBase.vnd_improvement`` executes a sequence as a
        black box, each operator is tried individually in the adaptive order.
        Successful moves reset the neighbourhood scan, matching VND semantics.
        """
        order = self._ordered_local_searches(base_order, exploration_rotation)
        current = self.evaluate(solution.clone())
        k = 0
        while k < len(order):
            if time.time() >= self._deadline:
                return current
            name = order[k]
            before_obj = current.objective
            stats = self.vnd_operator_stats.setdefault(
                name, {"uses": 0, "improvements": 0, "reward": 0.0}
            )
            stats["uses"] += 1
            improved = self.vnd_improvement(
                current,
                local_searches=(name,),
                optimize_bottleneck=False,
            )
            gain = before_obj - improved.objective
            if gain > 1e-9:
                stats["improvements"] += 1
                stats["reward"] += gain / max(1.0, abs(before_obj))
                current = improved
                k = 0
            else:
                k += 1

        if optimize_bottleneck:
            return self.vnd_improvement(current, local_searches=order, optimize_bottleneck=True)
        return current

    # ------------------------------------------------------------------
    # Splitting phase  (coarse-to-fine breakpoint refinement)
    # ------------------------------------------------------------------
    def splitting_phase(self, base_solution: Solution) -> Tuple[Solution, DiscreteInstance]:
        """Refine discretization around useful breakpoints for VNS.

        Mirrors the VND solver's splitting phase: first add all edge midpoints,
        then repeatedly keep only refinements around used midpoints. Each refined
        instance is improved by adaptive light VND so VNS can compete fairly with
        the standalone VND/LNS pipelines on breakpoint refinement.
        """
        if self.time_limit_reached() or time.time() >= self._deadline:
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

        previous_instance = self.instance
        previous_cache = self._distance_cache
        self.instance = refined_instance
        self._distance_cache = {}
        refined_solution = self._adaptive_vnd(
            converted,
            base_order=("intraroute_move", "zero_to_l_exchange", "destroy_and_repair"),
            optimize_bottleneck=False,
        )

        if refined_solution.objective + 1e-9 >= best_solution.objective:
            self.instance = previous_instance
            self._distance_cache = previous_cache
            return best_solution, best_instance

        best_solution = refined_solution
        best_instance = refined_instance
        current_instance = refined_instance
        current_solution = refined_solution
        current_breakpoints = expanded_breakpoints

        while True:
            if self.time_limit_reached() or time.time() >= self._deadline:
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

            self.instance = next_instance
            self._distance_cache = {}
            improved_solution = self._adaptive_vnd(
                next_solution,
                base_order=("intraroute_move", "zero_to_l_exchange", "destroy_and_repair"),
                optimize_bottleneck=False,
            )

            if improved_solution.objective + 1e-9 >= best_solution.objective:
                break

            best_solution = improved_solution
            best_instance = next_instance
            current_instance = next_instance
            current_solution = improved_solution
            current_breakpoints = next_breakpoints

        self.instance = best_instance
        self._distance_cache = {}
        return best_solution, best_instance

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

    def _shake_strength(self, k: int, no_improve_iters: int) -> int:
        patience_unit = max(1, self.stagnation_patience // 3)
        boost = 1 + int(math.log1p(no_improve_iters / patience_unit))
        return max(1, k * boost)

    def _apply_shake_operator(self, solution: Solution, operator_name: str, strength: int) -> Solution:
        """Apply one named shaking operator."""
        if operator_name == "random_task_relocate":
            return self._random_task_relocate(solution, strength)
        if operator_name == "bottleneck_task_relocate":
            return self._bottleneck_task_relocate(solution, strength)
        if operator_name == "destroy_repair_perturb":
            return self._destroy_repair_perturb(solution, strength)
        raise ValueError(f"Unknown VNS shaking operator: {operator_name}")

    def _operator_score(self, operator_name: str, total_uses: int) -> float:
        """UCB-style score balancing operator quality and exploration."""
        stats = self.operator_stats[operator_name]
        uses = stats["uses"]
        if uses == 0:
            return float("inf")
        average_reward = stats["reward"] / uses
        exploration = math.sqrt(2.0 * math.log(max(2, total_uses)) / uses)
        return average_reward + exploration

    def _select_shake_operator(self, k: int) -> str:
        """Select a shaking operator using UCB with deterministic warm-up.

        The neighbourhood index still influences early exploration: until every
        operator has been tried, ``k`` rotates through the operator list. After
        that, UCB gives more probability to operators that historically produce
        accepted moves and improvements while keeping an exploration bonus.
        """
        total_uses = sum(stats["uses"] for stats in self.operator_stats.values())
        if total_uses < len(self.shake_operators):
            return self.shake_operators[(k - 1) % len(self.shake_operators)]

        return max(
            self.shake_operators,
            key=lambda operator_name: self._operator_score(operator_name, total_uses),
        )

    def _adaptive_shake(
        self,
        solution: Solution,
        k: int,
        no_improve_iters: int,
    ) -> Tuple[Solution, str, int]:
        """Apply an adaptively selected perturbation operator."""
        strength = self._shake_strength(k, no_improve_iters)
        operator_name = self._select_shake_operator(k)
        self.operator_stats[operator_name]["uses"] += 1
        return self._apply_shake_operator(solution, operator_name, strength), operator_name, strength

    def _shake(self, solution: Solution, k: int, no_improve_iters: int) -> Solution:
        """Apply a real perturbation using the adaptive operator selector."""
        shaken, _, _ = self._adaptive_shake(solution, k, no_improve_iters)
        return shaken

    def _reward_operator(
        self,
        operator_name: str,
        before: Solution,
        improved: Solution,
        current: Solution,
        best: Solution,
        accepted: bool,
        improved_best: bool,
        iteration: int,
        k: int,
        strength: int,
        deviation: float,
        no_improve_iters: int,
    ) -> None:
        """Update operator statistics and append an adaptive-search event.

        Reward intentionally combines several signals, not only global-best
        improvement, to avoid starving operators that are useful for feasible
        diversification or incumbent improvement.
        """
        feasible = self.is_feasible_solution(improved)
        current_improvement = improved.objective + 1e-9 < current.objective
        plateau_secondary_improvement = (
            abs(improved.objective - current.objective) <= 1e-9
            and improved.paper_makespan + 1e-9 < current.paper_makespan
        )

        reward = 0.0
        if feasible:
            reward += 0.05
        if accepted:
            reward += 0.25
        if current_improvement:
            reward += 1.0
        if plateau_secondary_improvement:
            reward += 0.25
        if improved_best:
            reward += 3.0

        stats = self.operator_stats[operator_name]
        if feasible:
            stats["feasible"] += 1
        if accepted:
            stats["accepted"] += 1
        if current_improvement:
            stats["current_improvements"] += 1
        if improved_best:
            stats["best_improvements"] += 1
        stats["reward"] += reward

        self.adaptive_log.append(
            {
                "iteration": iteration,
                "k": k,
                "operator": operator_name,
                "strength": strength,
                "reward": reward,
                "accepted": accepted,
                "improved_best": improved_best,
                "current_improvement": current_improvement,
                "feasible": feasible,
                "deviation": deviation,
                "no_improve_iters": no_improve_iters,
                "before_objective": before.objective,
                "candidate_objective": improved.objective,
                "current_objective": current.objective,
                "best_objective": best.objective,
            }
        )

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

    def _solution_signature(self, solution: Solution) -> Tuple[Tuple[int, Tuple[Tuple[Tuple[int, int], ...], ...]], ...]:
        """Return a route-level signature used to keep elite memory diverse."""
        signature = []
        for launch in sorted(solution.flights_by_launch):
            flights = []
            for flight in solution.flights_by_launch[launch]:
                tasks = tuple((task.edge_id, task.forward) for task in flight.tasks)
                flights.append(tasks)
            signature.append((launch, tuple(flights)))
        return tuple(signature)

    def _add_to_elite(self, solution: Solution) -> None:
        """Store a distinct high-quality solution in bounded elite memory."""
        candidate = solution.clone()
        candidate_signature = self._solution_signature(candidate)

        for idx, elite in enumerate(self.elite_pool):
            if self._solution_signature(elite) == candidate_signature:
                if candidate.objective + 1e-9 < elite.objective:
                    self.elite_pool[idx] = candidate
                    self.elite_pool.sort(key=lambda s: s.objective)
                return

        self.elite_pool.append(candidate)
        self.elite_pool.sort(key=lambda s: s.objective)
        if len(self.elite_pool) > self.elite_size:
            self.elite_pool = self.elite_pool[: self.elite_size]

    def _restart_from_best(self, best: Solution, no_improve_iters: int) -> Solution:
        """Restart around the global best using a strong shake and light repair."""
        strong_k = max(self.k_max, self.k_max + no_improve_iters // self.stagnation_patience)
        restarted = self._shake(best, strong_k, no_improve_iters)
        return restarted

    def _restart_from_elite_or_new_pool(
        self,
        best: Solution,
        no_improve_iters: int,
        light_vnd: Tuple[str, ...],
    ) -> Tuple[Solution, str]:
        """Restart from elite memory when possible, otherwise from a fresh pool.

        The restart source follows the design in ``docs/vns_improve.md``:
        prefer a stored elite solution with strong shaking, but occasionally use
        a newly constructed solution plus light VND to diversify further.
        """
        use_new_pool = not self.elite_pool or self.rng.random() < 0.25

        if use_new_pool:
            pool = self.generate_initial_pool()
            if pool:
                restarted = min(
                    (
                        self._adaptive_vnd(
                            s,
                            base_order=light_vnd,
                            optimize_bottleneck=False,
                        )
                        for s in pool
                    ),
                    key=lambda s: s.objective,
                )
                return restarted, "new_pool"

        elite = self.rng.choice(self.elite_pool) if self.elite_pool else best
        restarted = self._shake(elite, self.k_max + 1, no_improve_iters)
        restarted = self._adaptive_vnd(
            restarted,
            base_order=light_vnd,
            optimize_bottleneck=False,
        )
        return restarted, "elite"

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def solve(self, initial: Optional[Solution] = None) -> Tuple[Solution, DiscreteInstance]:
        started = time.time()

        light_vnd = ("intraroute_move", "zero_to_l_exchange")
        initial_vnd = (
            "intraroute_move",
            "destroy_and_repair",
            "zero_to_l_exchange",
            "l1_l2_exchange",
        )
        if initial is None:
            pool = self.generate_initial_pool()
            current = min(
                (
                    self.vnd_improvement(
                        s,
                        local_searches=initial_vnd,
                        optimize_bottleneck=False,
                    )
                    for s in pool
                ),
                key=lambda s: s.objective,
            )
        else:
            current = self.vnd_improvement(
                initial,
                local_searches=initial_vnd,
                optimize_bottleneck=False,
            )

        best = current
        self.elite_pool = []
        self.operator_stats = {
            name: {
                "uses": 0,
                "accepted": 0,
                "best_improvements": 0,
                "current_improvements": 0,
                "feasible": 0,
                "reward": 0.0,
            }
            for name in self.shake_operators
        }
        self.adaptive_log = []
        self._add_to_elite(best)
        elapsed_init = time.time() - started
        self.convergence_log: list = [
            (0, elapsed_init, best.clone(), {"phase": "initial", "local_iteration": 0})
        ]

        deviation = min(self.max_deviation, max(self.min_deviation, self.initial_deviation))
        no_improve_iters = 0
        iteration = 0

        while iteration < self.max_iter and time.time() < self._deadline:
            improved_best_this_iter = False
            accepted_this_iter = False
            k = 1

            while k <= self.k_max and time.time() < self._deadline:
                before_shake = current
                shaken, operator_name, strength = self._adaptive_shake(current, k, no_improve_iters)
                improved = self._adaptive_vnd(
                    shaken,
                    base_order=light_vnd,
                    optimize_bottleneck=False,
                    exploration_rotation=k - 1,
                )

                improves_best = improved.objective + 1e-9 < best.objective
                accepted = improves_best or self._accepted_by_record_to_record(
                    improved,
                    current,
                    best,
                    deviation,
                )
                self._reward_operator(
                    operator_name=operator_name,
                    before=before_shake,
                    improved=improved,
                    current=current,
                    best=best,
                    accepted=accepted,
                    improved_best=improves_best,
                    iteration=iteration + 1,
                    k=k,
                    strength=strength,
                    deviation=deviation,
                    no_improve_iters=no_improve_iters,
                )

                if improves_best:
                    best = improved
                    current = improved
                    self._add_to_elite(best)
                    self.convergence_log.append(
                        (
                            iteration + 1,
                            time.time() - started,
                            best.clone(),
                            {
                                "phase": "improvement",
                                "local_iteration": iteration + 1,
                                "operator": operator_name,
                                "operator_reward": self.adaptive_log[-1]["reward"],
                            },
                        )
                    )
                    deviation = max(self.min_deviation, deviation * self.deviation_decay)
                    no_improve_iters = 0
                    improved_best_this_iter = True
                    accepted_this_iter = True
                    k = 1
                    continue

                if accepted:
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

            if no_improve_iters >= self.restart_patience:
                current, restart_source = self._restart_from_elite_or_new_pool(
                    best,
                    no_improve_iters,
                    light_vnd,
                )
                self.convergence_log.append(
                    (
                        iteration,
                        time.time() - started,
                        best.clone(),
                        {
                            "phase": f"restart_{restart_source}",
                            "local_iteration": iteration,
                            "elite_size": len(self.elite_pool),
                            "operator_stats": self.operator_stats,
                        },
                    )
                )
                if current.objective + 1e-9 < best.objective:
                    best = current
                    self._add_to_elite(best)
                    self.convergence_log.append(
                        (
                            iteration,
                            time.time() - started,
                            best.clone(),
                            {"phase": "restart_improvement", "local_iteration": iteration},
                        )
                    )
                no_improve_iters = 0
                deviation = min(self.max_deviation, max(self.min_deviation, self.initial_deviation))
            elif no_improve_iters >= self.stagnation_patience:
                current = self._restart_from_best(best, no_improve_iters)
                current = self._adaptive_vnd(
                    current,
                    base_order=light_vnd,
                    optimize_bottleneck=False,
                )
                if current.objective + 1e-9 < best.objective:
                    best = current
                    self._add_to_elite(best)
                    self.convergence_log.append(
                        (
                            iteration,
                            time.time() - started,
                            best.clone(),
                            {"phase": "stagnation_improvement", "local_iteration": iteration},
                        )
                    )
                deviation = min(self.max_deviation, max(self.min_deviation, self.initial_deviation))
            elif not accepted_this_iter:
                # Keep the search anchored when every neighbourhood rejects.
                current = best.clone()

        final = self._adaptive_vnd(best, optimize_bottleneck=True)
        if final.objective + 1e-9 < best.objective:
            best = final
            self._add_to_elite(best)
            self.convergence_log.append(
                (
                    iteration,
                    time.time() - started,
                    best.clone(),
                    {"phase": "final_vnd", "local_iteration": iteration},
                )
            )

        split_solution, split_instance = self.splitting_phase(best)
        if split_solution.objective + 1e-9 < best.objective:
            best = split_solution
            self.instance = split_instance
            self._distance_cache.clear()
            self._add_to_elite(best)
            self.convergence_log.append(
                (
                    iteration,
                    time.time() - started,
                    best.clone(),
                    {"phase": "final_splitting", "local_iteration": iteration},
                )
            )

        elapsed = time.time() - started
        if self.convergence_log[-1][0] != iteration:
            self.convergence_log.append(
                (
                    iteration,
                    elapsed,
                    best.clone(),
                    {
                        "phase": "final",
                        "local_iteration": iteration,
                        "elite_size": len(self.elite_pool),
                        "operator_stats": self.operator_stats,
                        "vnd_operator_stats": self.vnd_operator_stats,
                    },
                )
            )
        return best, self.instance
