"""Backward-compatibility shim.

All solver logic has moved to mm_mt_dlarp/algorithms/:
  base.py  — SolverConfig, MatheuristicBase
  vnd.py   — VNDSolver  (splitting phase + full solve)
  vns.py   — VNSSolver
  lns.py   — LNSSolver
"""
from .algorithms.base import MatheuristicBase, SolverConfig
from .algorithms.vnd import VNDSolver

# MatheuristicSolver is the full solver (VND + splitting phase).
MatheuristicSolver = VNDSolver

__all__ = ["SolverConfig", "MatheuristicBase", "MatheuristicSolver", "VNDSolver"]
