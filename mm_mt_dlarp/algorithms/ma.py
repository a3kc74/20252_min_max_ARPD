"""MASolver — Memetic Algorithm for MM-MT-dLARP.

MA = GA + VND local search applied to every new offspring.
Comparing GASolver (no LS) vs MASolver (LS) isolates the benefit of
local-search hybridisation; comparing MASolver vs IMMASolver isolates the
benefit of the island model.

The local-search operator used is VND (four nested neighbourhoods), the same
component as in island I0 of LCB-IMMA, ensuring a fair comparison.

Algorithm sketch
----------------
1. Initialise population of n_pop feasible solutions; apply VND to each.
2. While wall-clock < T_wall:
   a. Produce n_pop offspring via tournament selection + cost-based crossover.
   b. Apply random task-relocation mutation (rate mutation_rate).
   c. Apply VND local search to each offspring.
   d. (μ + λ) survivor selection: keep top n_pop from parents ∪ offspring.
3. Return global best.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from ..models import DiscreteInstance, Solution
from .base import MatheuristicBase, SolverConfig
from .ga import GASolver   # reuse crossover / mutation / survivor helpers


class MASolver(GASolver):
    """Memetic Algorithm: tournament-GA with VND local search on offspring.

    Inherits all operators from GASolver (crossover, mutation, survivor
    selection, tournament selection) and adds a VND improvement step applied
    to every newly created offspring before survivor selection.

    Parameters
    ----------
    instance      : DiscreteInstance
    config        : SolverConfig
    n_pop         : int   — population size (default 40).
    tournament_k  : int   — tournament pool size (default 3).
    elite_frac    : float — elite survival fraction (default 0.1).
    mutation_rate : float — per-individual mutation probability (default 0.3).
    T_wall        : float — wall-clock time budget in seconds (default 60.0).
    """

    # __init__ is fully inherited from GASolver; no extra parameters needed.

    # ──────────────────────────────────────────────────────────────────────────
    # Local search step — applied to offspring only
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_local_search(self, pop: List[Solution]) -> List[Solution]:
        """Apply VND improvement to every individual in *pop*."""
        return [self.vnd_improvement(sol) for sol in pop]

    # ──────────────────────────────────────────────────────────────────────────
    # Public solve()
    # ──────────────────────────────────────────────────────────────────────────

    def solve(self) -> Tuple[Solution, DiscreteInstance]:
        """Run the Memetic Algorithm and return (best_solution, instance)."""
        started = time.time()

        # Step 1: initialise population and apply VND to every individual
        raw_pop = self._init_population()
        pop     = self._apply_local_search(raw_pop)
        pop.sort(key=lambda s: s.objective)

        best_sol     = pop[0]
        f_star       = best_sol.objective
        gen          = 0
        elapsed_init = time.time() - started
        self.convergence_log: list = [(0, elapsed_init, best_sol.clone())]
        self._log(
            f"MA | init (after VND) | n_pop={len(pop)} "
            f"best_init={self._fmt_obj(f_star)}"
        )

        while time.time() - started < self.T_wall:
            gen += 1

            # GA operators
            offspring = self._crossover(pop)
            offspring = self._mutate(offspring)

            # Memetic step: local search on all new offspring
            offspring = self._apply_local_search(offspring)

            # (μ + λ) survivor selection
            pop = self._select_survivors(pop, offspring)

            epoch_best = pop[0]
            if epoch_best.objective + 1e-9 < f_star:
                best_sol = epoch_best
                f_star   = epoch_best.objective
                self.convergence_log.append((gen, time.time() - started, best_sol.clone()))
                self._log(f"MA | gen={gen} | new best={self._fmt_obj(f_star)}")

        elapsed = time.time() - started
        if self.convergence_log[-1][0] != gen:
            self.convergence_log.append((gen, elapsed, best_sol.clone()))
        self._log(
            f"MA | done | f*={self._fmt_obj(f_star)} "
            f"gen={gen} elapsed={elapsed:.1f}s"
        )
        return best_sol, self.instance
