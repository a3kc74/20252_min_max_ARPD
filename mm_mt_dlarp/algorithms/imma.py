"""IMMASolver — Island-Model Memetic Algorithm with random migration.

IMMA is an ablation variant of LCB-IMMA that is *identical* in every respect
except the migration controller:

  LCB-IMMA : two-stage LinUCB bandit selects (source island, destination,
              granularity) adaptively from a context vector.
  IMMA     : (source island, destination, granularity) are chosen UNIFORMLY
              AT RANDOM — no bandit, no context vector, no reward signal.

Comparing IMMA vs LCB-IMMA isolates the benefit of the adaptive LinUCB
controller while holding everything else constant:
  - same 4-island structure (VND / ILS / DP / Greedy)
  - same crossover, mutation, and elitism operators
  - same migration filter (decaying threshold θ_mv)
  - same Metropolis acceptance criterion
  - same stagnation / periodic migration trigger
  - same wall-clock budget

All island helper methods are inherited from LCBIMMASolver.  Only solve()
is overridden to swap bandit arm selection for uniform random selection.
"""
from __future__ import annotations

import math
import time
from typing import List, Tuple

from ..models import DiscreteInstance, Solution
from .base import SolverConfig
from .lcb_imma import LCBIMMASolver, _NUM_GRAN, _NUM_ISLANDS


class IMMASolver(LCBIMMASolver):
    """Island-Model Memetic Algorithm — random migration (no LinUCB bandit).

    Parameters
    ----------
    instance, config : standard solver arguments (passed to MatheuristicBase).
    n_pop            : total population size; each island gets n_pop // 4.
    tau_base         : periodic migration interval (epochs).
    delta_stag       : stagnation window (epochs without improvement) that
                       triggers migration.
    lambda_pen       : reward penalty logged for infeasible migration
                       (kept for API compatibility; not used to update a bandit).
    theta_0, theta_1 : annealing migration-filter thresholds (θ_0 > θ_1 ≥ 0).
    T_0              : initial Metropolis acceptance temperature.
    rho              : geometric decay factor for temperature and filter threshold.
    T_wall           : wall-clock time budget in seconds.
    ils_iter         : inner ILS iterations for island I1 (I2 in paper).

    Note: ``alpha`` (LinUCB exploration coefficient) is accepted for API
    compatibility with LCBIMMASolver but is ignored.
    """

    # __init__ is fully inherited from LCBIMMASolver.
    # alpha is accepted but unused; all other parameters apply unchanged.

    # ──────────────────────────────────────────────────────────────────────────
    # Public solve()  —  identical to LCBIMMASolver.solve() but with the
    # two-stage LinUCB replaced by uniform random arm selection.
    # ──────────────────────────────────────────────────────────────────────────

    def solve(self) -> Tuple[Solution, DiscreteInstance]:   # noqa: C901
        """Run IMMA (random-migration island MA) and return (best_solution, instance)."""
        started = time.time()

        # ── Stage 2: initialise population ───────────────────────────────────
        islands = self._init_islands()
        all_init = [s for isl in islands for s in isl]
        self._log(
            f"IMMA | init | n_pop={len(all_init)} "
            f"n_isl={self.n_isl}"
        )

        # ── Tracking ─────────────────────────────────────────────────────────
        best_sol = min(all_init, key=lambda s: s.objective)
        f_star   = best_sol.objective
        stag     = 0
        tau      = 0
        self.convergence_log: list = [(0, 0.0, best_sol.clone())]

        # ── Stage 3: main loop ────────────────────────────────────────────────
        while time.time() - started < self.T_wall:
            tau  += 1
            T_tau = self.T_0 * (self.rho ** tau)

            # Per-island evolution (with deadline check between local-search calls)
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
                f_star   = epoch_best.objective
                stag     = 0
                self.convergence_log.append((tau, time.time() - started, best_sol.clone()))
                self._log(f"IMMA | tau={tau} | new best={self._fmt_obj(f_star)}")
            else:
                stag += 1

            # ── Migration trigger ─────────────────────────────────────────────
            if stag >= self.delta_stag or tau % self.tau_base == 0:

                # ── RANDOM arm selection (replaces LinUCB) ────────────────────
                i_star = self.rng.randrange(_NUM_ISLANDS)

                dests  = [j for j in range(_NUM_ISLANDS) if j != i_star]
                j_star = self.rng.choice(dests)
                g_star = self.rng.randrange(_NUM_GRAN)
                # ─────────────────────────────────────────────────────────────

                dest_pop = islands[j_star]
                if not dest_pop:
                    stag = 0
                    continue

                f_avg_source = float(
                    sum(s.objective for s in islands[i_star]) / len(islands[i_star])
                ) if islands[i_star] else 1.0
                f_avg_dest = float(
                    sum(s.objective for s in dest_pop) / len(dest_pop)
                )

                # Migration filter: same decaying threshold as LCB-IMMA (Eq. 14)
                theta_mv = self.theta_1 + (self.theta_0 - self.theta_1) * (self.rho ** tau)

                segs = self._extract_segments(islands[i_star], g_star)
                for launch_v, seg_tasks in segs:
                    mv = self._migration_score(launch_v, seg_tasks, f_avg_source)
                    if mv < theta_mv:
                        continue
                    self._inject_segment(dest_pop, seg_tasks, f_avg_dest, T_tau)

                stag = 0

        elapsed = time.time() - started
        if self.convergence_log[-1][0] != tau:
            self.convergence_log.append((tau, elapsed, best_sol.clone()))
        self._log(
            f"IMMA | done | f*={self._fmt_obj(f_star)} "
            f"elapsed={elapsed:.1f}s"
        )
        return best_sol, self.instance
