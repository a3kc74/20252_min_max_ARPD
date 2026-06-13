"""VNSSolver — Variable Neighborhood Search for MM-MT-dLARP.

Extends MatheuristicBase with shaking to escape local optima.
After each local search, if no improvement is found, shakes the
solution (random perturbation) before restarting.

References:
    Hansen & Mladenović (2001) — Variable Neighborhood Search.
    Paper Section 4.2 for the neighborhood definitions reused here.
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

from ..models import DiscreteInstance, Solution
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
    """

    def __init__(
        self,
        instance: DiscreteInstance,
        config: SolverConfig,
        k_max: int = 4,
        max_iter: int = 100,
    ) -> None:
        super().__init__(instance, config)
        self.k_max = k_max
        self.max_iter = max_iter

    # ------------------------------------------------------------------
    # Shaking  (random perturbation to escape local optima)
    # ------------------------------------------------------------------
    def _shake(self, solution: Solution, k: int) -> Solution:
        """Apply k random destroy-and-repair moves as perturbation."""
        current = solution.clone()
        for _ in range(k):
            result = self.destroy_and_repair(current)
            if result is not None:
                current = result
        return current

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
        iteration = 0

        while iteration < self.max_iter:
            k = 1
            while k <= self.k_max:
                shaken = self._shake(current, k)
                improved = self.vnd_improvement(
                    shaken,
                    local_searches=light_vnd,
                    optimize_bottleneck=False,
                )

                if improved.objective + 1e-9 < current.objective:
                    current = improved
                    if current.objective + 1e-9 < best.objective:
                        best = current
                        self.convergence_log.append(
                            (iteration + 1, time.time() - started, best.clone())
                        )
                    k = 1
                else:
                    k += 1

            iteration += 1
        
        final = self.vnd_improvement(best)
        if final.objective + 1e-9 < best.objective:
            best = final
            self.convergence_log.append((iteration, time.time() - started, best.clone()))

        elapsed = time.time() - started
        if self.convergence_log[-1][0] != iteration:
            self.convergence_log.append((iteration, elapsed, best.clone()))
        return best, self.instance
