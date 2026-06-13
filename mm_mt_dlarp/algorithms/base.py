"""Core solver infrastructure for MM-MT-dLARP.

SolverConfig    — hyperparameters shared by all solvers.
MatheuristicBase — geometry, feasibility, construction, neighborhoods, VND.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..grp import BranchCutSolver, BranchCutStats, build_grp_from_flight, reconstruct_flight
from ..models import DiscreteInstance, Flight, RequiredEdge, Solution, Task


@dataclass
class SolverConfig:
    num_trucks: int
    flight_limit: Optional[float] = None
    seed: int = 0
    base_vertex: Optional[int] = None
    max_construction_solutions: int = 20
    fast_construction_target: int = 300
    fast_construction_seconds: float = 30.0
    max_construction_seconds: float = 600.0
    max_construction_attempts: int = 2000
    max_stall_attempts: int = 250
    nmax_destroy: int = 5
    itmax_destroy: int = 30
    lmax_exchange: int = 3
    exact_flight_threshold: int = 12
    flight_optimizer: str = "bc"
    bc_time_limit: Optional[float] = None
    bc_mip_gap: float = 0.0
    bc_enable_connectivity_cuts: bool = True
    bc_enable_r_odd_cuts: bool = True
    bc_enable_advanced_cuts: bool = False
    bc_cut_at_fractional_nodes: bool = True
    bc_cut_at_integer_solutions: bool = True
    bc_max_cuts_per_round: int = 50
    bc_tailoff_window: int = 5
    bc_tailoff_tol: float = 5e-7
    split_top_k: int = 10
    objective_type: str = "minmax_ghg"
    time_limit_seconds: Optional[float] = None
    search_deadline: Optional[float] = None
    # GHG emission factors (paper Section 3.1)
    emission_truck: float = 1.0        # F_t  (g/km or any consistent unit)
    emission_drone_cruise: float = 1.0  # F_d  (g/km)
    emission_drone_vt: float = 0.0      # F_d^vt (g/flight, fixed per takeoff-landing cycle)
    emission_epsilon: float = 1e-6      # ε for Eq. 4 (avoids division by zero at depot)
    verbose: bool = True


def accept_improving(candidate: Solution, current: Solution, **_) -> bool:
    """Accept only strict improvements."""
    return candidate.objective + 1e-9 < current.objective


class MatheuristicBase:
    """Geometry, feasibility, construction, neighborhoods, and improvements.

    This class contains the algorithmic machinery used by the VND solver.
    It does *not* include the splitting phase or the top-level solve() —
    those live in VNDSolver.
    """

    def __init__(self, instance: DiscreteInstance, config: SolverConfig) -> None:
        self.instance = instance
        self.config = config
        self.rng = random.Random(config.seed)
        if config.flight_limit is None:
            self.config.flight_limit = self.estimate_default_flight_limit(instance)
        self.L = float(self.config.flight_limit)
        self._distance_cache: Dict[Tuple[int, int], float] = {}
        self._grp_cache: Dict[Tuple, "Flight"] = {}
        self.bc_stats = BranchCutStats()
        self._deadline: float = float("inf")  # set by outer solver to enforce T_wall
        self._log(
            f"initialized | trucks={self.config.num_trucks} "
            f"L={self.L:.3f} seed={self.config.seed} "
            f"launches={len(self.instance.launch_vertices)} "
            f"edges={len(self.instance.required_edges)}"
        )

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(f"[{self.__class__.__name__}] {message}")

    def _fmt_obj(self, value: float) -> str:
        return "inf" if math.isinf(value) else f"{value:.3f}"

    def time_limit_reached(self) -> bool:
        return self.config.search_deadline is not None and time.time() >= self.config.search_deadline

    # ------------------------------------------------------------------
    # Geometry and costs
    # ------------------------------------------------------------------
    def distance(self, v1: int, v2: int) -> float:
        key = (v1, v2) if v1 <= v2 else (v2, v1)
        if key in self._distance_cache:
            return self._distance_cache[key]
        a = self.instance.raw.vertices[v1]
        b = self.instance.raw.vertices[v2]
        d = math.hypot(a.x - b.x, a.y - b.y)
        self._distance_cache[key] = d
        return d

    def _grp_cache_key(self, flight: Flight) -> tuple:
        """Cache key for GRP optimisation: (launch_vertex, frozenset of edge_ids)."""
        return (flight.launch_vertex, frozenset(t.edge_id for t in flight.tasks))

    def truck_cost(self, launch: int) -> float:
        return 2.0 * self.distance(self.instance.base_vertex, launch)

    def truck_emission(self, launch: int) -> float:
        """GHG emission for round-trip truck travel to *launch* (Eq. 5)."""
        return self.config.emission_truck * self.truck_cost(launch)

    def flight_emission(self, flight: Flight) -> float:
        """GHG emission for one drone flight (Eq. 7).

        Includes fixed vertical takeoff+landing term (2·F_d^vt) and
        cruise term (F_d · range_cost).  Returns 0 for empty flights.
        """
        if not flight.tasks:
            return 0.0
        return (2.0 * self.config.emission_drone_vt
                + self.config.emission_drone_cruise * self.flight_cost(flight))

    def edge_distance_to_launch(self, edge: RequiredEdge, launch: int) -> float:
        return min(self.distance(launch, edge.start_vertex), self.distance(launch, edge.end_vertex))

    def task_start(self, task: Task) -> int:
        edge = self.instance.edge_by_id[task.edge_id]
        return edge.start_vertex if task.forward else edge.end_vertex

    def task_end(self, task: Task) -> int:
        edge = self.instance.edge_by_id[task.edge_id]
        return edge.end_vertex if task.forward else edge.start_vertex

    def flight_cost(self, flight: Flight) -> float:
        if not flight.tasks:
            return 0.0
        total = 0.0
        current = flight.launch_vertex
        for task in flight.tasks:
            edge = self.instance.edge_by_id[task.edge_id]
            start_v = self.task_start(task)
            end_v = self.task_end(task)
            total += self.distance(current, start_v)
            total += edge.service_cost
            current = end_v
        total += self.distance(current, flight.launch_vertex)
        return total

    def route_cost_for_launch(self, solution: Solution, launch: int) -> float:
        """Original paper route cost for one launching point: truck + drone flight costs."""
        flights = solution.flights_by_launch.get(launch, [])
        if not any(f.tasks for f in flights):
            return 0.0
        return self.truck_cost(launch) + sum(self.flight_cost(f) for f in flights if f.tasks)

    def route_emission_for_launch(self, solution: Solution, launch: int) -> float:
        """Total GHG emission for one launching point (truck + all drone flights)."""
        flights = solution.flights_by_launch.get(launch, [])
        if not any(f.tasks for f in flights):
            return 0.0
        return self.truck_emission(launch) + sum(self.flight_emission(f) for f in flights if f.tasks)

    def _objective_from_metrics(
        self,
        paper_by_launch: Dict[int, float],
        ghg_by_launch: Dict[int, float],
        total_ghg: float,
    ) -> float:
        objective_type = self.config.objective_type.lower()
        if objective_type == "paper_makespan":
            return max(paper_by_launch.values()) if paper_by_launch else float("inf")
        if objective_type == "minmax_ghg":
            return max(ghg_by_launch.values()) if ghg_by_launch else float("inf")
        if objective_type == "total_ghg":
            return total_ghg if paper_by_launch else float("inf")
        raise ValueError(f"Unknown objective_type: {self.config.objective_type}")

    def evaluate(self, solution: Solution) -> Solution:
        """Evaluate solution and set the configured objective.

        Supported objectives:
        - minmax_ghg: max_d [F_t·2c_0d + sum_k (2F_d^vt + F_d·range_cost_dk)]
        - paper_makespan: max_d [2c_0d + sum_k range_cost_dk], the original MM-MT-dLARP objective
        - total_ghg: sum_d [F_t·2c_0d + sum_k (2F_d^vt + F_d·range_cost_dk)]
        """
        pruned: Dict[int, List[Flight]] = {}
        selected: List[int] = []
        paper_by_launch: Dict[int, float] = {}
        ghg_by_launch: Dict[int, float] = {}
        flight_costs: Dict[Tuple[int, int], float] = {}
        total_ghg = 0.0
        for launch in list(solution.flights_by_launch.keys()):
            nonempty_flights = [f for f in solution.flights_by_launch[launch] if f.tasks]
            if not nonempty_flights:
                continue
            pruned[launch] = nonempty_flights
            selected.append(launch)

            paper_total = self.truck_cost(launch)
            ghg_total = self.truck_emission(launch)
            for idx, flight in enumerate(nonempty_flights):
                c = self.flight_cost(flight)
                flight_costs[(launch, idx)] = c
                paper_total += c
                ghg_total += self.flight_emission(flight)

            paper_by_launch[launch] = paper_total
            ghg_by_launch[launch] = ghg_total
            total_ghg += ghg_total

        solution.flights_by_launch = pruned
        solution.selected_launches = sorted(selected)
        solution.makespan_by_launch = paper_by_launch if self.config.objective_type.lower() == "paper_makespan" else ghg_by_launch
        solution.flight_costs = flight_costs
        solution.paper_makespan = max(paper_by_launch.values()) if selected else float("inf")
        solution.ghg_makespan = max(ghg_by_launch.values()) if selected else float("inf")
        solution.total_ghg = total_ghg if selected else float("inf")
        solution.objective = self._objective_from_metrics(paper_by_launch, ghg_by_launch, solution.total_ghg)
        return solution

    # ------------------------------------------------------------------
    # Feasibility
    # ------------------------------------------------------------------
    def is_feasible_flight(self, flight: Flight) -> bool:
        """Range constraint (Eq. 8): 2·F_d^vt + range_cost <= L."""
        if not flight.tasks:
            return True
        return 2.0 * self.config.emission_drone_vt + self.flight_cost(flight) <= self.L + 1e-9

    def is_feasible_solution(self, solution: Solution) -> bool:
        used_edges = set()
        for flights in solution.flights_by_launch.values():
            for flight in flights:
                if not self.is_feasible_flight(flight):
                    return False
                for task in flight.tasks:
                    if task.edge_id in used_edges:
                        return False
                    used_edges.add(task.edge_id)
        all_edges = {edge.edge_id for edge in self.instance.required_edges}
        return used_edges == all_edges and len(solution.selected_launches) <= self.config.num_trucks

    def bottleneck_launch(self, solution: Solution) -> Optional[int]:
        if not solution.makespan_by_launch:
            return None
        return max(solution.makespan_by_launch, key=lambda d: solution.makespan_by_launch[d])

    @staticmethod
    def estimate_default_flight_limit(instance: DiscreteInstance) -> float:
        def dist(v1: int, v2: int) -> float:
            a = instance.raw.vertices[v1]
            b = instance.raw.vertices[v2]
            return math.hypot(a.x - b.x, a.y - b.y)

        values = []
        for edge in instance.required_edges:
            best = min(
                min(dist(launch, edge.start_vertex), dist(launch, edge.end_vertex))
                + edge.service_cost
                + min(dist(edge.start_vertex, launch), dist(edge.end_vertex, launch))
                for launch in instance.launch_vertices
            )
            values.append(best)
        lmin = max(values) if values else 0.0
        L = 1.5 * lmin
        return L if L > 0 else 1.0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _weighted_sample_without_replacement(
        self, population: List, weights: List[float], k: int
    ) -> List:
        """Weighted sampling without replacement (used for emission-guided launch selection)."""
        if k >= len(population):
            return list(population)
        remaining_pop = list(population)
        remaining_w = list(weights)
        selected: List = []
        for _ in range(k):
            total = sum(remaining_w)
            if total <= 0:
                idx = self.rng.randrange(len(remaining_pop))
            else:
                r = self.rng.random() * total
                cumul = 0.0
                idx = len(remaining_pop) - 1
                for i, w in enumerate(remaining_w):
                    cumul += w
                    if cumul >= r:
                        idx = i
                        break
            selected.append(remaining_pop.pop(idx))
            remaining_w.pop(idx)
        return selected

    def construct_initial_solution(self) -> Solution:
        all_launches = list(self.instance.launch_vertices)
        k = min(self.config.num_trucks, len(all_launches))

        # Emission-guided launch selection: p_d ∝ 1 / (F_t · 2·c_{0d} + ε)  (Eq. 4)
        weights = [
            1.0 / max(
                self.config.emission_truck * self.truck_cost(d) + self.config.emission_epsilon,
                1e-12,
            )
            for d in all_launches
        ]
        launches = self._weighted_sample_without_replacement(all_launches, weights, k)
        self._log(f"construction | selected launches={launches}")

        assignments: Dict[int, List[RequiredEdge]] = {d: [] for d in launches}
        for edge in self.instance.required_edges:
            weights = [1.0 / max(self.edge_distance_to_launch(edge, d), 1e-6) for d in launches]
            chosen = self.rng.choices(launches, weights=weights, k=1)[0]
            assignments[chosen].append(edge)

        flights_by_launch: Dict[int, List[Flight]] = {d: [] for d in launches}
        for launch, edges in assignments.items():
            giant = self.build_giant_tour(launch, edges)
            flights_by_launch[launch] = self.split_giant_tour(launch, giant)
        sol = Solution(selected_launches=launches, flights_by_launch=flights_by_launch)
        return self.evaluate(sol)

    def build_giant_tour(self, launch: int, edges: Sequence[RequiredEdge]) -> List[Task]:
        remaining = list(edges)
        giant: List[Task] = []
        current = launch
        while remaining:
            best_idx = -1
            best_dist = float("inf")
            best_forward = True
            for idx, edge in enumerate(remaining):
                d_start = self.distance(current, edge.start_vertex)
                d_end = self.distance(current, edge.end_vertex)
                if d_start <= d_end:
                    dist, forward = d_start, True
                else:
                    dist, forward = d_end, False
                if dist < best_dist:
                    best_idx = idx
                    best_dist = dist
                    best_forward = forward
            edge = remaining.pop(best_idx)
            task = Task(edge.edge_id, best_forward)
            giant.append(task)
            current = self.task_end(task)
        return giant

    def split_giant_tour(self, launch: int, giant: Sequence[Task]) -> List[Flight]:
        flights: List[Flight] = []
        current_flight = Flight(launch, [])
        for task in giant:
            candidate = Flight(launch, current_flight.tasks + [task])
            if self.is_feasible_flight(candidate):
                current_flight.tasks.append(task)
            else:
                if not current_flight.tasks:
                    # L is assumed large enough for a single edge; keep and mark infeasible later
                    current_flight.tasks.append(task)
                    flights.append(current_flight)
                    current_flight = Flight(launch, [])
                else:
                    flights.append(current_flight)
                    current_flight = Flight(launch, [task])
        if current_flight.tasks:
            flights.append(current_flight)
        return flights

    def generate_initial_pool(self) -> List[Solution]:
        start = time.time()
        pool: Dict[Tuple, Solution] = {}
        attempts = 0
        stall = 0
        while True:
            if self.time_limit_reached():
                break
            elapsed = time.time() - start
            if elapsed >= self.config.max_construction_seconds:
                break
            if attempts >= self.config.max_construction_attempts:
                break
            if stall >= self.config.max_stall_attempts and pool:
                break
            if len(pool) >= self.config.max_construction_solutions:
                break
            if elapsed < self.config.fast_construction_seconds and len(pool) >= self.config.fast_construction_target:
                break
            attempts += 1
            sol = self.construct_initial_solution()
            if not self.is_feasible_solution(sol):
                stall += 1
                continue
            sig = sol.normalized_signature()
            if sig in pool:
                stall += 1
            else:
                pool[sig] = sol
                stall = 0
        if not pool:
            raise RuntimeError("Failed to build any feasible initial solution")
        return list(pool.values())

    # ------------------------------------------------------------------
    # Exact flight optimization
    # ------------------------------------------------------------------
    def exact_optimize_flight(self, flight: Flight) -> Flight:
        optimizer = self.config.flight_optimizer.lower()
        if optimizer == "dp":
            return self.dp_optimize_flight(flight)
        if optimizer == "bc":
            return self.branch_cut_optimize_flight(flight)
        if optimizer == "auto":
            if len(flight.tasks) <= self.config.exact_flight_threshold:
                return self.branch_cut_optimize_flight(flight)
            return self.intraroute_single_flight(flight)
        raise ValueError(f"Unknown flight_optimizer: {self.config.flight_optimizer}")

    def dp_optimize_flight(self, flight: Flight) -> Flight:
        n = len(flight.tasks)
        if n == 0:
            return Flight(flight.launch_vertex, [])
        if n > self.config.exact_flight_threshold:
            return self.intraroute_single_flight(flight)

        key = self._grp_cache_key(flight)
        if key in self._grp_cache:
            cached = self._grp_cache[key]
            if self.is_feasible_flight(cached):
                return cached

        edges = [self.instance.edge_by_id[t.edge_id] for t in flight.tasks]
        start_v = [[edge.start_vertex, edge.end_vertex] for edge in edges]
        exit_v = [[edge.end_vertex, edge.start_vertex] for edge in edges]

        dp: Dict[Tuple[int, int, int], float] = {}
        parent: Dict[Tuple[int, int, int], Optional[Tuple[int, int, int]]] = {}
        for i, edge in enumerate(edges):
            for o in (0, 1):
                mask = 1 << i
                dp[(mask, i, o)] = self.distance(flight.launch_vertex, start_v[i][o]) + edge.service_cost
                parent[(mask, i, o)] = None

        for mask in range(1, 1 << n):
            for last in range(n):
                for o in (0, 1):
                    key = (mask, last, o)
                    if key not in dp:
                        continue
                    current_cost = dp[key]
                    for nxt in range(n):
                        if mask & (1 << nxt):
                            continue
                        next_mask = mask | (1 << nxt)
                        for no in (0, 1):
                            cand = current_cost + self.distance(exit_v[last][o], start_v[nxt][no]) + edges[nxt].service_cost
                            nkey = (next_mask, nxt, no)
                            if cand < dp.get(nkey, float("inf")):
                                dp[nkey] = cand
                                parent[nkey] = key

        full_mask = (1 << n) - 1
        best_key: Optional[Tuple[int, int, int]] = None
        best_cost = float("inf")
        for last in range(n):
            for o in (0, 1):
                key = (full_mask, last, o)
                if key not in dp:
                    continue
                total = dp[key] + self.distance(exit_v[last][o], flight.launch_vertex)
                if total < best_cost:
                    best_cost = total
                    best_key = key

        if best_key is None:
            return self.intraroute_single_flight(flight)

        order: List[Tuple[int, int]] = []
        cursor = best_key
        while cursor is not None:
            _, idx, o = cursor
            order.append((idx, o))
            cursor = parent[cursor]
        order.reverse()
        tasks = [Task(edges[idx].edge_id, forward=(o == 0)) for idx, o in order]
        candidate = Flight(flight.launch_vertex, tasks)
        if self.is_feasible_flight(candidate):
            self._grp_cache[key] = candidate
            return candidate
        return self.intraroute_single_flight(flight)

    def branch_cut_optimize_flight(self, flight: Flight) -> Flight:
        n = len(flight.tasks)
        if n == 0:
            return Flight(flight.launch_vertex, [])

        key = self._grp_cache_key(flight)
        if key in self._grp_cache:
            cached = self._grp_cache[key]
            if self.is_feasible_flight(cached):
                return cached

        grp = build_grp_from_flight(self.instance, flight)
        bc_time_limit = self.config.bc_time_limit
        if bc_time_limit is None and not math.isinf(self._deadline):
            bc_time_limit = max(1.0, self._deadline - time.time())
        solver = BranchCutSolver(
            grp,
            time_limit=bc_time_limit,
            mip_gap=self.config.bc_mip_gap,
            enable_connectivity_cuts=self.config.bc_enable_connectivity_cuts,
            enable_r_odd_cuts=self.config.bc_enable_r_odd_cuts,
            enable_advanced_cuts=self.config.bc_enable_advanced_cuts,
            cut_at_fractional_nodes=self.config.bc_cut_at_fractional_nodes,
            cut_at_integer_solutions=self.config.bc_cut_at_integer_solutions,
            max_cuts_per_round=self.config.bc_max_cuts_per_round,
        )
        result = solver.solve()
        self.bc_stats.merge(result.stats)
        if not result.optimal:
            if n <= self.config.exact_flight_threshold:
                return self.dp_optimize_flight(flight)
            return self.intraroute_single_flight(flight)

        candidate = reconstruct_flight(grp, result)
        if not self._same_required_task_multiset(flight, candidate):
            if n <= self.config.exact_flight_threshold:
                return self.dp_optimize_flight(flight)
            return self.intraroute_single_flight(flight)
        if not self.is_feasible_flight(candidate):
            return self.intraroute_single_flight(flight)
        if self.flight_cost(candidate) <= self.flight_cost(flight) + 1e-9:
            self._grp_cache[key] = candidate
            return candidate
        return flight

    def _same_required_task_multiset(self, left: Flight, right: Flight) -> bool:
        counts: Dict[str, int] = {}
        for task in left.tasks:
            counts[task.edge_id] = counts.get(task.edge_id, 0) + 1
        for task in right.tasks:
            counts[task.edge_id] = counts.get(task.edge_id, 0) - 1
        return all(value == 0 for value in counts.values())

    # ------------------------------------------------------------------
    # Neighborhoods
    # ------------------------------------------------------------------
    def intraroute_single_flight(self, flight: Flight) -> Flight:
        if len(flight.tasks) <= 1:
            return Flight(flight.launch_vertex, list(flight.tasks))
        improved = True
        best = Flight(flight.launch_vertex, list(flight.tasks))
        while improved:
            improved = False
            current_cost = self.flight_cost(best)
            for idx in range(len(best.tasks)):
                base_tasks = list(best.tasks)
                task = base_tasks.pop(idx)
                best_local = None
                best_local_cost = current_cost
                for pos in range(len(base_tasks) + 1):
                    for forward in (True, False):
                        cand_tasks = list(base_tasks)
                        cand_tasks.insert(pos, Task(task.edge_id, forward))
                        cand = Flight(flight.launch_vertex, cand_tasks)
                        cost = self.flight_cost(cand)
                        if cost + 1e-9 < best_local_cost and cost <= self.L + 1e-9:
                            best_local = cand
                            best_local_cost = cost
                if best_local is not None:
                    best = best_local
                    improved = True
                    break
        return best

    def intraroute_move(self, solution: Solution) -> Optional[Solution]:
        candidate = solution.clone()
        improved = False
        for launch, flights in candidate.flights_by_launch.items():
            for idx, flight in enumerate(flights):
                optimized = self.intraroute_single_flight(flight)
                if self.flight_cost(optimized) + 1e-9 < self.flight_cost(flight):
                    flights[idx] = optimized
                    improved = True
        candidate = self.evaluate(candidate)
        if improved and candidate.objective + 1e-9 < solution.objective:
            return candidate
        return None

    def all_possible_insertions(self, solution: Solution, task: Task) -> List[Tuple[int, int, int, Task]]:
        options: List[Tuple[int, int, int, Task]] = []
        for launch, flights in solution.flights_by_launch.items():
            for f_idx, flight in enumerate(flights):
                for pos in range(len(flight.tasks) + 1):
                    for forward in (True, False):
                        options.append((launch, f_idx, pos, Task(task.edge_id, forward)))
            options.append((launch, len(flights), 0, Task(task.edge_id, True)))
            options.append((launch, len(flights), 0, Task(task.edge_id, False)))
        return options

    def insert_task(self, solution: Solution, launch: int, flight_idx: int, pos: int, task: Task) -> Optional[Solution]:
        candidate = solution.clone()
        flights = candidate.flights_by_launch.setdefault(launch, [])
        if flight_idx == len(flights):
            flights.append(Flight(launch, [task]))
        else:
            flights[flight_idx].tasks.insert(pos, task)
        candidate = self.evaluate(candidate)
        if all(self.is_feasible_flight(f) for fs in candidate.flights_by_launch.values() for f in fs):
            return candidate
        return None

    def destroy_and_repair(self, solution: Solution) -> Optional[Solution]:
        bottleneck = self.bottleneck_launch(solution)
        if bottleneck is None:
            return None
        current = solution
        non_improving = 0
        while non_improving < self.config.itmax_destroy:
            candidate = current.clone()
            locations: List[Tuple[int, int, int, Task]] = []
            for f_idx, flight in enumerate(candidate.flights_by_launch.get(bottleneck, [])):
                for pos, task in enumerate(flight.tasks):
                    locations.append((bottleneck, f_idx, pos, task))
            if not locations:
                return None
            n_remove = self.rng.randint(1, min(self.config.nmax_destroy, len(locations)))
            selected = sorted(
                self.rng.sample(locations, k=n_remove),
                key=lambda x: (x[1], x[2]),
                reverse=True,
            )
            removed: List[Task] = []
            for _, f_idx, pos, task in selected:
                removed.append(candidate.flights_by_launch[bottleneck][f_idx].tasks.pop(pos))
            self.evaluate(candidate)
            removed.reverse()
            feasible = True
            for task in removed:
                best_sol = None
                best_obj = float("inf")
                for launch, f_idx, pos, oriented in self.all_possible_insertions(candidate, task):
                    inserted = self.insert_task(candidate, launch, f_idx, pos, oriented)
                    if inserted is not None and inserted.objective < best_obj:
                        best_sol = inserted
                        best_obj = inserted.objective
                if best_sol is None:
                    feasible = False
                    break
                candidate = best_sol
            if feasible and candidate.objective + 1e-9 < current.objective:
                current = candidate
                non_improving = 0
            else:
                non_improving += 1
        return current if current.objective + 1e-9 < solution.objective else None

    def zero_to_l_exchange(self, solution: Solution) -> Optional[Solution]:
        bottleneck = self.bottleneck_launch(solution)
        if bottleneck is None:
            return None
        l = 1
        current = solution
        while l <= self.config.lmax_exchange:
            improved = False
            donor_flights = current.flights_by_launch.get(bottleneck, [])
            for df_idx, donor in enumerate(donor_flights):
                if len(donor.tasks) < l:
                    continue
                for start in range(len(donor.tasks) - l + 1):
                    chain = donor.tasks[start:start + l]
                    for launch, flights in current.flights_by_launch.items():
                        for rf_idx in range(len(flights) + 1):
                            if launch == bottleneck and rf_idx == df_idx:
                                continue
                            recipient_positions = [0]
                            if rf_idx < len(flights):
                                recipient_positions = list(range(len(flights[rf_idx].tasks) + 1))
                            for pos in recipient_positions:
                                cand = current.clone()
                                cand.flights_by_launch[bottleneck][df_idx].tasks = (
                                    cand.flights_by_launch[bottleneck][df_idx].tasks[:start]
                                    + cand.flights_by_launch[bottleneck][df_idx].tasks[start + l:]
                                )
                                flights2 = cand.flights_by_launch.setdefault(launch, [])
                                if rf_idx == len(flights2):
                                    flights2.append(Flight(launch, list(chain)))
                                else:
                                    flights2[rf_idx].tasks[pos:pos] = list(chain)
                                cand = self.evaluate(cand)
                                if self.is_feasible_solution(cand) and cand.objective + 1e-9 < current.objective:
                                    current = cand
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                l = 1
            else:
                l += 1
        return current if current.objective + 1e-9 < solution.objective else None

    def l1_l2_exchange(self, solution: Solution) -> Optional[Solution]:
        bottleneck = self.bottleneck_launch(solution)
        if bottleneck is None:
            return None
        l1 = l2 = 1
        current = solution
        while l1 <= self.config.lmax_exchange:
            improved = False
            donor_flights = current.flights_by_launch.get(bottleneck, [])
            for df_idx, donor in enumerate(donor_flights):
                if len(donor.tasks) < l1:
                    continue
                for ds in range(len(donor.tasks) - l1 + 1):
                    chain1 = donor.tasks[ds:ds + l1]
                    for launch, flights in current.flights_by_launch.items():
                        for rf_idx, recip in enumerate(flights):
                            if launch == bottleneck and rf_idx == df_idx:
                                continue
                            if len(recip.tasks) < l2:
                                continue
                            for rs in range(len(recip.tasks) - l2 + 1):
                                chain2 = recip.tasks[rs:rs + l2]
                                cand = current.clone()
                                cand.flights_by_launch[bottleneck][df_idx].tasks = (
                                    cand.flights_by_launch[bottleneck][df_idx].tasks[:ds]
                                    + list(chain2)
                                    + cand.flights_by_launch[bottleneck][df_idx].tasks[ds + l1:]
                                )
                                cand.flights_by_launch[launch][rf_idx].tasks = (
                                    cand.flights_by_launch[launch][rf_idx].tasks[:rs]
                                    + list(chain1)
                                    + cand.flights_by_launch[launch][rf_idx].tasks[rs + l2:]
                                )
                                cand = self.evaluate(cand)
                                if self.is_feasible_solution(cand) and cand.objective + 1e-9 < current.objective:
                                    current = cand
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                l1 = l2 = 1
                continue
            l2 += 1
            if l2 > self.config.lmax_exchange:
                l1 += 1
                l2 = 1
        return current if current.objective + 1e-9 < solution.objective else None

    def optimize_bottleneck_flights(self, solution: Solution) -> Optional[Solution]:
        current = solution.clone()
        improved_any = False
        while True:
            if time.time() >= self._deadline:
                break
            bottleneck = self.bottleneck_launch(current)
            if bottleneck is None:
                break
            changed = False
            for idx, flight in enumerate(current.flights_by_launch[bottleneck]):
                if time.time() >= self._deadline:
                    break
                opt = self.exact_optimize_flight(flight)
                if self.flight_cost(opt) + 1e-9 < self.flight_cost(flight) and self.is_feasible_flight(opt):
                    current.flights_by_launch[bottleneck][idx] = opt
                    changed = True
                    improved_any = True
            current = self.evaluate(current)
            if not changed:
                break
        return current if improved_any and current.objective + 1e-9 < solution.objective else None

    # ------------------------------------------------------------------
    # Island-specific operators (Sec. 3.2.3)
    # ------------------------------------------------------------------
    def ils_improvement(self, solution: Solution, n_iter: int = 20) -> Solution:
        """Island I2: ILS with emission-guided ruin-and-recreate perturbation."""
        current = self.vnd_improvement(solution)
        best = current
        for _ in range(n_iter):
            if time.time() >= self._deadline:
                break
            perturbed = self.destroy_and_repair(current)
            if perturbed is None:
                perturbed = current
            improved = self.vnd_improvement(perturbed)
            if improved.objective + 1e-9 < current.objective:
                current = improved
                if current.objective + 1e-9 < best.objective:
                    best = current
        return best

    def dp_bottleneck_improvement(self, solution: Solution) -> Solution:
        """Island I3: DP-based flight optimiser on the highest-emission launching point."""
        current = solution.clone()
        improved_any = False
        while True:
            bottleneck = self.bottleneck_launch(current)
            if bottleneck is None:
                break
            changed = False
            for idx, flight in enumerate(current.flights_by_launch.get(bottleneck, [])):
                if not flight.tasks:
                    continue
                opt = self.dp_optimize_flight(flight)
                if (self.flight_cost(opt) + 1e-9 < self.flight_cost(flight)
                        and self.is_feasible_flight(opt)):
                    current.flights_by_launch[bottleneck][idx] = opt
                    changed = True
                    improved_any = True
            current = self.evaluate(current)
            if not changed:
                break
        if improved_any and current.objective + 1e-9 < solution.objective:
            return current
        return solution

    def greedy_repair_emission_edge(self, solution: Solution) -> Solution:
        """Island I4: greedy repair targeting the highest per-km emission required edge."""
        current = solution.clone()

        # Find required edge with highest service_cost / edge_length ratio
        best_edge_id: Optional[str] = None
        best_ratio = -1.0
        for flights in current.flights_by_launch.values():
            for flight in flights:
                for task in flight.tasks:
                    edge = self.instance.edge_by_id[task.edge_id]
                    length = self.distance(edge.start_vertex, edge.end_vertex)
                    ratio = edge.service_cost / max(length, 1e-12)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_edge_id = task.edge_id

        if best_edge_id is None:
            return solution

        # Remove target edge from its current flight
        removed_task: Optional[Task] = None
        for flights in current.flights_by_launch.values():
            for flight in flights:
                for pos, task in enumerate(flight.tasks):
                    if task.edge_id == best_edge_id:
                        removed_task = flight.tasks.pop(pos)
                        break
                if removed_task is not None:
                    break
            if removed_task is not None:
                break

        if removed_task is None:
            return solution

        self.evaluate(current)

        # Re-insert at the best feasible position
        best_sol: Optional[Solution] = None
        best_obj = float("inf")
        for launch, f_idx, pos, oriented in self.all_possible_insertions(current, removed_task):
            inserted = self.insert_task(current, launch, f_idx, pos, oriented)
            if inserted is not None and inserted.objective < best_obj:
                best_sol = inserted
                best_obj = inserted.objective

        return best_sol if best_sol is not None else solution

    def vnd_improvement(
        self,
        solution: Solution,
        local_searches: Optional[Sequence[str]] = None,
        optimize_bottleneck: bool = True,
    ) -> Solution:
        """VND improvement with configurable local-search operators.

        Parameters
        ----------
        solution:
            Starting solution to improve.
        local_searches:
            Ordered local-search operator names. Supported names are
            ``"intraroute_move"``, ``"destroy_and_repair"``,
            ``"zero_to_l_exchange"``, and ``"l1_l2_exchange"``. If omitted,
            the full VND neighbourhood list from the paper is used.
        optimize_bottleneck:
            If true, run exact bottleneck-flight optimisation after VND reaches
            a local optimum, then restart VND if that exact step improves.
        """
        operator_map: Dict[str, Callable[[Solution], Optional[Solution]]] = {
            "intraroute_move": self.intraroute_move,
            "destroy_and_repair": self.destroy_and_repair,
            "zero_to_l_exchange": self.zero_to_l_exchange,
            "l1_l2_exchange": self.l1_l2_exchange,
        }
        if local_searches is None:
            local_searches = (
                "intraroute_move",
                "destroy_and_repair",
                "zero_to_l_exchange",
                "l1_l2_exchange",
            )

        neighborhoods: List[Callable[[Solution], Optional[Solution]]] = []
        for name in local_searches:
            try:
                neighborhoods.append(operator_map[name])
            except KeyError as exc:
                valid = ", ".join(operator_map)
                raise ValueError(f"Unknown VND local search '{name}'. Valid values: {valid}") from exc

        current = self.evaluate(solution.clone())
        k = 0
        while k < len(neighborhoods):
            if time.time() >= self._deadline:
                return current
            improved = neighborhoods[k](current)
            if improved is not None and improved.objective + 1e-9 < current.objective:
                current = self.evaluate(improved)
                k = 0
            else:
                k += 1

        if optimize_bottleneck:
            if time.time() >= self._deadline:
                return current
            opt = self.optimize_bottleneck_flights(current)
            if opt is not None and opt.objective + 1e-9 < current.objective:
                return self.vnd_improvement(
                    opt,
                    local_searches=local_searches,
                    optimize_bottleneck=optimize_bottleneck,
                )
        return current
