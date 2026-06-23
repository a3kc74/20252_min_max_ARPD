"""LCBIMMASolver — Linear Contextual Bandit Island-Model Memetic Algorithm.

Implements the four-stage LCB-IMMA pipeline (paper Section 3):

  Stage 1 – Lazy GRP lower-bound cache (reuses base-class GRP machinery).
  Stage 2 – Emission-guided population initialisation across four islands.
  Stage 3 – Island evolution + two-stage LinUCB migration controller.
  Stage 4 – Return best feasible solution.

Four islands, each with a distinct local-search operator:
  I0 (I1 in paper) – VND with four nested neighbourhoods.
  I1 (I2 in paper) – ILS with cost-based ruin-and-recreate.
  I2 (I3 in paper) – DP-based flight optimiser on the highest-cost launch.
  I3 (I4 in paper) – Greedy repair targeting the highest service-cost density edge.

The two-stage LinUCB bandit (alpha-coefficient ridge regression) adaptively
selects (source island, destination island, migration granularity) at every
stagnation-driven or periodic epoch.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

from ..models import DiscreteInstance, Flight, Solution, Task
from .base import MatheuristicBase, SolverConfig

# ── context-vector dimension (phi ∈ R^7) ─────────────────────────────────────
_CTX_DIM = 7
_NUM_ISLANDS = 4
_NUM_GRAN = 3   # granularity levels: single edge / single flight / all flights at launch


def _np():
    """Lazy numpy import (avoids hard dependency at module level)."""
    import numpy as np  # noqa: PLC0415
    return np


# ─────────────────────────────────────────────────────────────────────────────
# LinUCB helper (operates on plain lists so numpy can stay optional at import)
# ─────────────────────────────────────────────────────────────────────────────

class _LinUCBBandit:
    """Single LinUCB bandit: n_arms arms, d-dimensional context."""

    def __init__(self, n_arms: int, d: int, alpha: float) -> None:
        np = _np()
        self.alpha = alpha
        self.A = [np.eye(d) for _ in range(n_arms)]   # gram matrices
        self.b = [np.zeros(d) for _ in range(n_arms)]  # reward accumulators
        self.theta = [np.zeros(d) for _ in range(n_arms)]

    def select(self, phi) -> int:
        """Return arm index with highest UCB score."""
        np = _np()
        phi = np.asarray(phi, dtype=float)
        best, best_score = 0, -math.inf
        for i, (A, theta) in enumerate(zip(self.A, self.theta)):
            A_inv = np.linalg.inv(A)
            score = float(theta @ phi) + self.alpha * math.sqrt(float(phi @ A_inv @ phi))
            if score > best_score:
                best_score = score
                best = i
        return best

    def update(self, arm: int, phi, reward: float) -> None:
        """Ridge-regression update (Eq. 17)."""
        np = _np()
        phi = np.asarray(phi, dtype=float)
        self.A[arm] = self.A[arm] + np.outer(phi, phi)
        self.b[arm] = self.b[arm] + reward * phi
        self.theta[arm] = np.linalg.solve(self.A[arm], self.b[arm])


# ─────────────────────────────────────────────────────────────────────────────
# LCBIMMASolver
# ─────────────────────────────────────────────────────────────────────────────

class LCBIMMASolver(MatheuristicBase):
    """LCB-IMMA: four-island memetic algorithm with LinUCB migration control.

    Parameters
    ----------
    instance, config : standard solver arguments.
    n_pop            : total population size; each of the four islands gets n_pop//4.
    tau_base         : periodic migration interval (epochs).
    delta_stag       : stagnation window (epochs without improvement) that triggers migration.
    alpha            : LinUCB exploration coefficient.
    lambda_pen       : reward penalty for infeasible migration outcomes.
    theta_0, theta_1 : annealing migration-filter thresholds (theta_0 > theta_1 >= 0).
    T_0              : initial Metropolis acceptance temperature.
    rho              : geometric decay factor for temperature and migration threshold.
    T_wall           : wall-clock time budget in seconds.
    ils_iter         : inner ILS iterations for island I1.
    """

    def __init__(
        self,
        instance: DiscreteInstance,
        config: SolverConfig,
        *,
        n_pop: int = 40,
        tau_base: int = 10,
        delta_stag: int = 20,
        alpha: float = 1.0,
        lambda_pen: float = 0.1,
        theta_0: float = 0.5,
        theta_1: float = 0.0,
        T_0: float = 1.0,
        rho: float = 0.99,
        T_wall: float = 60.0,
        ils_iter: int = 10,
    ) -> None:
        super().__init__(instance, config)
        self.n_pop = n_pop
        self.n_isl = max(1, n_pop // _NUM_ISLANDS)
        self.tau_base = tau_base
        self.delta_stag = delta_stag
        self.alpha = alpha
        self.lambda_pen = lambda_pen
        self.theta_0 = theta_0
        self.theta_1 = theta_1
        self.T_0 = T_0
        self.rho = rho
        self.T_wall = T_wall
        self.ils_iter = ils_iter

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 2: population initialisation
    # ──────────────────────────────────────────────────────────────────────────

    def _init_islands(self) -> List[List[Solution]]:
        """Generate N_pop feasible solutions and distribute evenly across islands."""
        pool: Dict[Tuple, Solution] = {}
        attempts = stall = 0
        while (len(pool) < self.n_pop
               and attempts < self.config.max_construction_attempts
               and (stall < self.config.max_stall_attempts or not pool)):
            sol = self.construct_initial_solution()
            attempts += 1
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
            raise RuntimeError("LCB-IMMA: failed to generate any feasible initial solution")

        all_sols = list(pool.values())
        # Pad to n_pop by cloning existing solutions
        while len(all_sols) < self.n_pop:
            all_sols.append(self.rng.choice(all_sols).clone())

        return [all_sols[i * self.n_isl:(i + 1) * self.n_isl] for i in range(_NUM_ISLANDS)]

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 3 helpers
    # ──────────────────────────────────────────────────────────────────────────

    # --- diversity -----------------------------------------------------------

    def _edge_assignment(self, sol: Solution) -> Dict[str, Tuple[int, int]]:
        """Map edge_id → (launch_vertex, flight_index)."""
        asgn: Dict[str, Tuple[int, int]] = {}
        for launch, flights in sol.flights_by_launch.items():
            for f_idx, flight in enumerate(flights):
                for task in flight.tasks:
                    asgn[task.edge_id] = (launch, f_idx)
        return asgn

    def _hamming(self, s1: Solution, s2: Solution) -> int:
        """Count required edges with different (launch, flight) assignments."""
        a1, a2 = self._edge_assignment(s1), self._edge_assignment(s2)
        return sum(1 for eid in (e.edge_id for e in self.instance.required_edges)
                   if a1.get(eid) != a2.get(eid))

    def _diversity(self, islands: List[List[Solution]]) -> float:
        """Average pairwise Hamming distance over best-of-island solutions (Eq. 9)."""
        bests = [min(isl, key=lambda s: s.objective) for isl in islands if isl]
        n = len(bests)
        if n < 2:
            return 0.0
        n_edges = len(self.instance.required_edges)
        if n_edges == 0:
            return 0.0
        total = sum(self._hamming(bests[i], bests[j])
                    for i in range(n) for j in range(i + 1, n))
        n_pairs = n * (n - 1) // 2
        return total / n_pairs / n_edges  # normalised to [0, 1]

    # --- context vector ------------------------------------------------------

    def _context(
        self,
        islands: List[List[Solution]],
        f_star: float,
        prev_f_star: float,
        stag: int,
        tau: int,
        T_tau: float,
        f0_bar: float,
    ):
        """Build phi(s) ∈ R^7 (Eq. 10)."""
        np = _np()

        div = self._diversity(islands)

        bests = [min(isl, key=lambda s: s.objective).objective for isl in islands if isl]
        f_avg_isl = (sum(bests) / len(bests) / max(f0_bar, 1e-12)) if bests else 1.0

        if not math.isinf(prev_f_star) and prev_f_star > 1e-12:
            delta_rel = (prev_f_star - f_star) / prev_f_star
        else:
            delta_rel = 0.0

        stag_norm = stag / max(self.delta_stag, 1)
        tau_norm = min(tau / max(self.tau_base * 50, 1), 1.0)
        T_ratio = T_tau / max(self.T_0, 1e-12)

        return np.array([div, f_avg_isl, delta_rel, stag_norm, tau_norm, T_ratio, 1.0])

    # --- local search per island ---------------------------------------------

    def _local_search(self, island_idx: int, sol: Solution) -> Solution:
        if island_idx == 0:
            return self.vnd_improvement(sol)
        if island_idx == 1:
            return self.ils_improvement(sol, n_iter=self.ils_iter)
        if island_idx == 2:
            return self.dp_bottleneck_improvement(sol)
        return self.greedy_repair_cost_edge(sol)

    # --- crossover & mutation ------------------------------------------------

    def _crossover(self, pop: List[Solution]) -> List[Solution]:
        """Donate a random flight from one parent into another."""
        if len(pop) < 2:
            return list(pop)
        new_pop = list(pop)
        for _ in range(max(1, len(pop) // 2)):
            i1, i2 = self.rng.sample(range(len(pop)), 2)
            donor = pop[i1]
            if not donor.flights_by_launch:
                continue
            launch = self.rng.choice(list(donor.flights_by_launch.keys()))
            nonempty = [f for f in donor.flights_by_launch[launch] if f.tasks]
            if not nonempty:
                continue
            flight = self.rng.choice(nonempty)
            task_ids = {t.edge_id for t in flight.tasks}

            recipient = pop[i2].clone()
            # Remove donated tasks from recipient
            for fl in [f for flist in recipient.flights_by_launch.values() for f in flist]:
                fl.tasks = [t for t in fl.tasks if t.edge_id not in task_ids]
            self.evaluate(recipient)

            # Re-insert donated tasks
            ok = True
            for task in flight.tasks:
                best_sol: Optional[Solution] = None
                best_obj = float("inf")
                for lv, fi, pos, oriented in self.all_possible_insertions(recipient, task):
                    ins = self.insert_task(recipient, lv, fi, pos, oriented)
                    if ins is not None and ins.objective < best_obj:
                        best_sol, best_obj = ins, ins.objective
                if best_sol is None:
                    ok = False
                    break
                recipient = best_sol

            if ok and self.is_feasible_solution(recipient) and recipient.objective < pop[i2].objective:
                new_pop[i2] = recipient
        return new_pop

    def _mutate(self, pop: List[Solution]) -> List[Solution]:
        """Random single-task relocation mutation (30 % rate)."""
        new_pop: List[Solution] = []
        for sol in pop:
            if self.rng.random() < 0.3:
                mutated = self.destroy_and_repair(sol)
                if mutated is not None and self.is_feasible_solution(mutated):
                    new_pop.append(mutated)
                    continue
            new_pop.append(sol)
        return new_pop

    @staticmethod
    def _elitism(pop: List[Solution]) -> List[Solution]:
        """Sort population by fitness (ascending); full population is retained.

        There is no truncation — all individuals survive. Sorting ensures the
        best individuals seed crossover and migration in subsequent epochs
        (selection pressure via quality ordering, not elimination).
        """
        return sorted(pop, key=lambda s: s.objective)

    # --- segment extraction --------------------------------------------------

    def _extract_segments(
        self, island_pop: List[Solution], granularity: int
    ) -> List[Tuple[int, List[Task]]]:
        """Return (launch_vertex, task-list) pairs at the requested granularity.

        Each segment carries its associated launch vertex so that the migration
        score can be computed from the correct departure point.
        """
        if not island_pop:
            return []
        best = min(island_pop, key=lambda s: s.objective)
        segs: List[Tuple[int, List[Task]]] = []
        if granularity == 0:          # g1: individual required edges
            for launch, flights in best.flights_by_launch.items():
                for flight in flights:
                    for task in flight.tasks:
                        segs.append((launch, [task]))
        elif granularity == 1:        # g2: individual flights
            for launch, flights in best.flights_by_launch.items():
                for flight in flights:
                    if flight.tasks:
                        segs.append((launch, list(flight.tasks)))
        else:                         # g3: all flights from one launching point
            for launch, flights in best.flights_by_launch.items():
                tasks = [t for f in flights for t in f.tasks]
                if tasks:
                    segs.append((launch, tasks))
        return segs

    # --- migration filter ----------------------------------------------------

    def _migration_score(
        self, launch_vertex: int, seg_tasks: List[Task], f_avg_source: float
    ) -> float:
        """mv(π) = (f_avg(I_i*) − f(π)) / f_avg(I_i*)  (Eq. 13).

        f(π) is approximated as the open-path deadheading cost of the segment,
        traversed from the actual launch vertex (drone departure point).
        """
        if f_avg_source <= 0:
            return 0.0
        seg_cost = 0.0
        current = launch_vertex          # start from the actual launch vertex
        for task in seg_tasks:
            edge = self.instance.edge_by_id[task.edge_id]
            sv = edge.start_vertex if task.forward else edge.end_vertex
            ev = edge.end_vertex if task.forward else edge.start_vertex
            seg_cost += self.distance(current, sv) + edge.service_cost
            current = ev
        return (f_avg_source - seg_cost) / f_avg_source

    # --- inject segment into destination population --------------------------

    def _inject_segment(
        self,
        dest_pop: List[Solution],
        seg_tasks: List[Task],
        f_avg_dest: float,
        T_tau: float,
    ) -> bool:
        """Try to integrate *seg_tasks* into a clone of a random destination individual.

        Returns True if at least one feasible migrant was produced and inserted.
        """
        base = self.rng.choice(dest_pop).clone()
        seg_ids = {t.edge_id for t in seg_tasks}

        # Remove duplicate tasks from the base clone
        for flights in base.flights_by_launch.values():
            for f in flights:
                f.tasks = [t for t in f.tasks if t.edge_id not in seg_ids]
        self.evaluate(base)

        pi = base
        for task in seg_tasks:
            best_sol: Optional[Solution] = None
            best_obj = float("inf")
            for lv, fi, pos, oriented in self.all_possible_insertions(pi, task):
                ins = self.insert_task(pi, lv, fi, pos, oriented)
                if ins is not None and ins.objective < best_obj:
                    best_sol, best_obj = ins, ins.objective
            if best_sol is None:
                return False
            pi = best_sol

        if not self.is_feasible_solution(pi):
            return False

        dest_best = min(dest_pop, key=lambda s: s.objective)
        worst_idx = max(range(len(dest_pop)), key=lambda i: dest_pop[i].objective)

        if pi.objective + 1e-9 < dest_best.objective and pi.objective < f_avg_dest:
            dest_pop[worst_idx] = pi
        elif self.rng.random() < math.exp(
            -(pi.objective - f_avg_dest) / max(T_tau, 1e-12)
        ):  # Metropolis (Eq. 15)
            rand_idx = self.rng.randrange(len(dest_pop))
            dest_pop[rand_idx] = pi
        else:
            return False
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # Public solve()
    # ──────────────────────────────────────────────────────────────────────────

    def solve(self) -> Tuple[Solution, DiscreteInstance]:
        """Run the full LCB-IMMA pipeline and return (best_solution, instance)."""
        np = _np()
        started = time.time()
        self._deadline = started + self.T_wall

        # ── Stage 2: initialise population ───────────────────────────────────
        islands = self._init_islands()
        all_init = [s for isl in islands for s in isl]
        f0_bar = float(np.mean([s.objective for s in all_init]))
        self._log(
            f"LCB-IMMA | init | n_pop={len(all_init)} "
            f"n_isl={self.n_isl} f0_bar={self._fmt_obj(f0_bar)}"
        )

        # ── Bandits ───────────────────────────────────────────────────────────
        # B1: 4 arms — select source island
        b1 = _LinUCBBandit(_NUM_ISLANDS, _CTX_DIM, self.alpha)
        # B2[i]: 9 arms — select (destination island, granularity) for each source i
        n_arms2 = (_NUM_ISLANDS - 1) * _NUM_GRAN
        b2 = [_LinUCBBandit(n_arms2, _CTX_DIM, self.alpha) for _ in range(_NUM_ISLANDS)]

        # ── Tracking ─────────────────────────────────────────────────────────
        best_sol = min(all_init, key=lambda s: s.objective)
        f_star = best_sol.objective
        prev_f_star = f_star
        stag = 0
        tau = 0
        self.convergence_log: list = [(0, 0.0, best_sol.clone())]  # (epoch, elapsed_s, Solution)

        # ── Stage 3: main loop ────────────────────────────────────────────────
        while time.time() - started < self.T_wall:
            tau += 1
            T_tau = self.T_0 * (self.rho ** tau)

            # Per-island evolution (equal wall-clock budget approximated by round-robin)
            for i in range(_NUM_ISLANDS):
                if not islands[i]:
                    continue
                evolved = []
                for s in islands[i]:
                    if time.time() >= self._deadline:
                        evolved.append(s)
                    else:
                        evolved.append(self._local_search(i, s))
                islands[i] = evolved
                islands[i] = self._crossover(islands[i])
                islands[i] = self._mutate(islands[i])
                islands[i] = self._elitism(islands[i])

            # Update global best
            epoch_best = min(
                (min(isl, key=lambda s: s.objective) for isl in islands if isl),
                key=lambda s: s.objective,
            )
            if epoch_best.objective + 1e-9 < f_star:
                best_sol = epoch_best
                f_star = epoch_best.objective
                stag = 0
                self.convergence_log.append((tau, time.time() - started, best_sol.clone()))
                self._log(f"LCB-IMMA | tau={tau} | new best={self._fmt_obj(f_star)}")
            else:
                stag += 1

            # ── Migration trigger ─────────────────────────────────────────────
            if stag >= self.delta_stag or tau % self.tau_base == 0:
                phi = self._context(islands, f_star, prev_f_star, stag, tau, T_tau, f0_bar)
                prev_f_star = f_star

                # Stage-1 bandit: choose source island
                i_star = b1.select(phi)

                # Stage-2 bandit: choose (destination, granularity)
                arm2 = b2[i_star].select(phi)
                j_idx = arm2 // _NUM_GRAN
                g_star = arm2 % _NUM_GRAN
                dests = [j for j in range(_NUM_ISLANDS) if j != i_star]
                j_star = dests[j_idx]

                dest_pop = islands[j_star]
                if not dest_pop:
                    stag = 0
                    continue

                f_before = min(dest_pop, key=lambda s: s.objective).objective
                f_avg_source = float(
                    np.mean([s.objective for s in islands[i_star]])
                ) if islands[i_star] else 1.0
                f_avg_dest = float(np.mean([s.objective for s in dest_pop]))

                # Migration threshold (Eq. 14)
                theta_mv = self.theta_1 + (self.theta_0 - self.theta_1) * (self.rho ** tau)

                segs = self._extract_segments(islands[i_star], g_star)
                feasible_migration = False
                for launch_v, seg_tasks in segs:
                    mv = self._migration_score(launch_v, seg_tasks, f_avg_source)
                    if mv < theta_mv:
                        continue
                    if self._inject_segment(dest_pop, seg_tasks, f_avg_dest, T_tau):
                        feasible_migration = True

                # Reward (Eq. 16)
                f_after = min(dest_pop, key=lambda s: s.objective).objective
                if feasible_migration and not math.isinf(f_before) and f_before > 1e-12:
                    r_tau = (f_before - f_after) / f_before
                else:
                    r_tau = -self.lambda_pen

                # Update bandits (Eq. 17)
                b1.update(i_star, phi, r_tau)
                b2[i_star].update(arm2, phi, r_tau)

                stag = 0

        elapsed = time.time() - started
        self._log(f"LCB-IMMA | done | f*={self._fmt_obj(f_star)} elapsed={elapsed:.1f}s")
        if self.convergence_log[-1][0] != tau:
            self.convergence_log.append((tau, elapsed, best_sol.clone()))
        return best_sol, self.instance
