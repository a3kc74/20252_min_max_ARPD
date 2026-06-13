"""GASolver — Standard Genetic Algorithm for MM-MT-dLARP.

Serves as the weakest comparison baseline for LCB-IMMA.  GASolver uses
NO local search; comparing it with MASolver isolates the benefit of local
search hybridisation.

Operators
---------
* Selection  : binary / k-way tournament (default k = 3).
* Crossover  : emission-guided flight crossover — a random non-empty flight
               from the tournament-winner donor is re-inserted into a clone
               of the recipient parent; tasks already in the recipient that
               overlap with the donated flight are first removed.
* Mutation   : random single-task relocation (destroy-and-repair, rate 0.3).
* Survivor   : (μ + λ) elitism — top n_pop individuals survive every epoch.

Termination : wall-clock time budget T_wall (seconds).
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from ..models import DiscreteInstance, Flight, Solution, Task
from .base import MatheuristicBase, SolverConfig


class GASolver(MatheuristicBase):
    """Standard Genetic Algorithm — no local search.

    Parameters
    ----------
    instance      : DiscreteInstance
    config        : SolverConfig
    n_pop         : int   — population size (default 40).
    tournament_k  : int   — tournament pool size for parent selection (default 3).
    elite_frac    : float — fraction of population guaranteed to survive (default 0.1).
    mutation_rate : float — per-individual mutation probability (default 0.3).
    T_wall        : float — wall-clock time budget in seconds (default 60.0).
    """

    def __init__(
        self,
        instance: DiscreteInstance,
        config: SolverConfig,
        *,
        n_pop: int = 40,
        tournament_k: int = 3,
        elite_frac: float = 0.1,
        mutation_rate: float = 0.3,
        T_wall: float = 60.0,
    ) -> None:
        super().__init__(instance, config)
        self.n_pop = n_pop
        self.tournament_k = tournament_k
        self.elite_frac = elite_frac
        self.mutation_rate = mutation_rate
        self.T_wall = T_wall

    # ──────────────────────────────────────────────────────────────────────────
    # Population initialisation
    # ──────────────────────────────────────────────────────────────────────────

    def _init_population(self) -> List[Solution]:
        """Generate n_pop distinct feasible solutions."""
        pool: Dict[Tuple, Solution] = {}
        attempts = stall = 0
        while (
            len(pool) < self.n_pop
            and attempts < self.config.max_construction_attempts
            and (stall < self.config.max_stall_attempts or not pool)
        ):
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
            raise RuntimeError("GASolver: failed to generate any feasible initial solution")

        pop = list(pool.values())
        # Pad to exactly n_pop by cloning existing solutions
        while len(pop) < self.n_pop:
            pop.append(self.rng.choice(pop).clone())
        return pop

    # ──────────────────────────────────────────────────────────────────────────
    # Selection
    # ──────────────────────────────────────────────────────────────────────────

    def _tournament_select(self, pop: List[Solution]) -> Solution:
        """Return the best individual from a random tournament of size k."""
        k = min(self.tournament_k, len(pop))
        return min(self.rng.sample(pop, k), key=lambda s: s.objective)

    # ──────────────────────────────────────────────────────────────────────────
    # Crossover — emission-guided flight donation
    # ──────────────────────────────────────────────────────────────────────────

    def _crossover(self, pop: List[Solution]) -> List[Solution]:
        """Produce len(pop) offspring via emission-guided flight crossover.

        For each offspring:
          1. Select donor and recipient via tournament (two independent draws).
          2. Pick a random non-empty flight from the donor.
          3. Clone the recipient; remove all tasks that appear in the donated flight.
          4. Re-insert donated tasks at the globally best feasible position.
          5. If the child is feasible, add it to the offspring pool.
        """
        if len(pop) < 2:
            return list(pop)

        offspring: List[Solution] = []
        for _ in range(len(pop)):
            donor     = self._tournament_select(pop)
            recipient = self._tournament_select(pop)

            if not donor.flights_by_launch:
                offspring.append(recipient.clone())
                continue

            launch   = self.rng.choice(list(donor.flights_by_launch.keys()))
            nonempty = [f for f in donor.flights_by_launch[launch] if f.tasks]
            if not nonempty:
                offspring.append(recipient.clone())
                continue

            flight   = self.rng.choice(nonempty)
            task_ids = {t.edge_id for t in flight.tasks}

            # Build child from recipient, removing duplicated tasks
            child = recipient.clone()
            for fl in [f for flist in child.flights_by_launch.values() for f in flist]:
                fl.tasks = [t for t in fl.tasks if t.edge_id not in task_ids]
            self.evaluate(child)

            # Re-insert donated tasks at best feasible positions
            ok = True
            for task in flight.tasks:
                best_sol: Optional[Solution] = None
                best_obj = float("inf")
                for lv, fi, pos, oriented in self.all_possible_insertions(child, task):
                    ins = self.insert_task(child, lv, fi, pos, oriented)
                    if ins is not None and ins.objective < best_obj:
                        best_sol, best_obj = ins, ins.objective
                if best_sol is None:
                    ok = False
                    break
                child = best_sol

            if ok and self.is_feasible_solution(child):
                offspring.append(child)
            else:
                offspring.append(recipient.clone())   # fallback: keep recipient

        return offspring

    # ──────────────────────────────────────────────────────────────────────────
    # Mutation — random single-task relocation
    # ──────────────────────────────────────────────────────────────────────────

    def _mutate(self, pop: List[Solution]) -> List[Solution]:
        """Apply random task-relocation mutation with probability mutation_rate."""
        new_pop: List[Solution] = []
        for sol in pop:
            if self.rng.random() < self.mutation_rate:
                mutated = self.destroy_and_repair(sol)
                if mutated is not None and self.is_feasible_solution(mutated):
                    new_pop.append(mutated)
                    continue
            new_pop.append(sol)
        return new_pop

    # ──────────────────────────────────────────────────────────────────────────
    # Survivor selection — (μ + λ) elitism
    # ──────────────────────────────────────────────────────────────────────────

    def _select_survivors(
        self, parents: List[Solution], offspring: List[Solution]
    ) -> List[Solution]:
        """Merge parents + offspring, keep top n_pop (elitism)."""
        combined = parents + offspring
        combined.sort(key=lambda s: s.objective)
        return combined[: self.n_pop]

    # ──────────────────────────────────────────────────────────────────────────
    # Public solve()
    # ──────────────────────────────────────────────────────────────────────────

    def solve(self) -> Tuple[Solution, DiscreteInstance]:
        """Run the GA and return (best_solution, instance)."""
        started = time.time()

        pop = self._init_population()
        best_sol = min(pop, key=lambda s: s.objective)
        f_star   = best_sol.objective
        gen      = 0
        self.convergence_log: list = [(0, 0.0, best_sol.clone())]
        self._log(
            f"GA | init | n_pop={len(pop)} "
            f"best_init={self._fmt_obj(f_star)}"
        )

        while time.time() - started < self.T_wall:
            gen += 1

            offspring = self._crossover(pop)
            offspring = self._mutate(offspring)
            pop       = self._select_survivors(pop, offspring)

            epoch_best = pop[0]  # list is sorted by _select_survivors
            if epoch_best.objective + 1e-9 < f_star:
                best_sol = epoch_best
                f_star   = epoch_best.objective
                self.convergence_log.append((gen, time.time() - started, best_sol.clone()))
                self._log(f"GA | gen={gen} | new best={self._fmt_obj(f_star)}")

        elapsed = time.time() - started
        if self.convergence_log[-1][0] != gen:
            self.convergence_log.append((gen, elapsed, best_sol.clone()))
        self._log(
            f"GA | done | f*={self._fmt_obj(f_star)} "
            f"gen={gen} elapsed={elapsed:.1f}s"
        )
        return best_sol, self.instance
