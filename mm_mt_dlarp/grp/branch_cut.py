from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from ..models import Task
from .cuts import ArcKey, compute_rsets, cut_arcs, separate_connectivity_cuts, separate_r_odd_cuts
from .models import GRPInstance


@dataclass
class BranchCutStats:
    flights_solved: int = 0
    flights_optimal: int = 0
    cuts_added: int = 0
    connectivity_cuts: int = 0
    r_odd_cuts: int = 0
    advanced_cuts: int = 0
    lazy_cuts: int = 0
    user_cuts: int = 0
    nodes: int = 0
    lp_iterations: int = 0
    objective_bound: float = float("inf")
    mip_gap: float = float("inf")
    total_time: float = 0.0
    status: str = ""

    def merge(self, other: "BranchCutStats") -> None:
        self.flights_solved += other.flights_solved
        self.flights_optimal += other.flights_optimal
        self.cuts_added += other.cuts_added
        self.connectivity_cuts += other.connectivity_cuts
        self.r_odd_cuts += other.r_odd_cuts
        self.advanced_cuts += other.advanced_cuts
        self.lazy_cuts += other.lazy_cuts
        self.user_cuts += other.user_cuts
        self.nodes += other.nodes
        self.lp_iterations += other.lp_iterations
        self.objective_bound = other.objective_bound
        self.mip_gap = other.mip_gap
        self.total_time += other.total_time
        self.status = other.status or self.status


@dataclass(frozen=True)
class BranchCutResult:
    tasks: Tuple[Task, ...]
    objective: float
    optimal: bool
    status: str
    x_values: Dict[ArcKey, int]
    stats: BranchCutStats


class BranchCutSolver:
    """Gurobi branch-and-cut solver for one GRP flight instance."""

    def __init__(
        self,
        instance: GRPInstance,
        *,
        time_limit: Optional[float] = None,
        mip_gap: float = 0.0,
        enable_connectivity_cuts: bool = True,
        enable_r_odd_cuts: bool = True,
        enable_advanced_cuts: bool = False,
        cut_at_fractional_nodes: bool = True,
        cut_at_integer_solutions: bool = True,
        max_cuts_per_round: int = 50,
    ) -> None:
        self.instance = instance
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.enable_connectivity_cuts = enable_connectivity_cuts
        self.enable_r_odd_cuts = enable_r_odd_cuts
        self.enable_advanced_cuts = enable_advanced_cuts
        self.cut_at_fractional_nodes = cut_at_fractional_nodes
        self.cut_at_integer_solutions = cut_at_integer_solutions
        self.max_cuts_per_round = max_cuts_per_round

        self.model = None
        self.x = {}
        self.stats = BranchCutStats(flights_solved=1)
        self._seen_lazy_cuts: set[Tuple[str, frozenset[int], float]] = set()
        self._seen_user_cuts: set[Tuple[str, frozenset[int], float]] = set()

    def solve(self) -> BranchCutResult:
        started = time.perf_counter()
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError:
            self.stats.status = "NO_GUROBI"
            self.stats.total_time = time.perf_counter() - started
            return BranchCutResult((), float("inf"), False, self.stats.status, {}, self.stats)

        if not self.instance.required_edges:
            self.stats.flights_optimal = 1
            self.stats.status = "OPTIMAL"
            self.stats.objective_bound = 0.0
            self.stats.mip_gap = 0.0
            self.stats.total_time = time.perf_counter() - started
            return BranchCutResult((), 0.0, True, self.stats.status, {}, self.stats)

        self.model = gp.Model("single_flight_grp_bc")
        self.model.Params.OutputFlag = 0
        self.model.Params.LazyConstraints = 1
        self.model.Params.PreCrush = 1
        if self.time_limit is not None:
            self.model.Params.TimeLimit = self.time_limit
        if self.mip_gap is not None:
            self.model.Params.MIPGap = self.mip_gap

        self._build_variables(GRB)
        self._add_required_service_constraints()
        self._add_flow_constraints()
        self._add_initial_connectivity_constraints()
        self.model.ModelSense = GRB.MINIMIZE

        self.model.optimize(self._callback)

        status_name = self._status_name(self.model.Status, GRB)
        self.stats.status = status_name
        self.stats.nodes = int(getattr(self.model, "NodeCount", 0))
        self.stats.lp_iterations = int(getattr(self.model, "IterCount", 0))
        if self.model.SolCount > 0:
            self.stats.objective_bound = float(self.model.ObjBound)
            objective = float(self.model.ObjVal)
            if abs(objective) <= 1e-12:
                self.stats.mip_gap = 0.0
            else:
                self.stats.mip_gap = abs(objective - self.model.ObjBound) / max(abs(objective), 1e-12)
            x_values = self._extract_x_values()
        else:
            self.stats.objective_bound = float("inf")
            self.stats.mip_gap = float("inf")
            objective = float("inf")
            x_values = {}

        optimal = self.model.Status == GRB.OPTIMAL
        if optimal:
            self.stats.flights_optimal = 1
            self.stats.mip_gap = 0.0
        self.stats.total_time = time.perf_counter() - started
        return BranchCutResult((), objective, optimal, status_name, x_values, self.stats)

    def _build_variables(self, grb) -> None:
        assert self.model is not None
        for edge in self.instance.edges:
            self.x[(edge.edge_id, edge.u, edge.v)] = self.model.addVar(
                lb=0.0,
                vtype=grb.INTEGER,
                obj=edge.cost_uv,
                name=f"x[{edge.edge_id},{edge.u},{edge.v}]",
            )
            self.x[(edge.edge_id, edge.v, edge.u)] = self.model.addVar(
                lb=0.0,
                vtype=grb.INTEGER,
                obj=edge.cost_vu,
                name=f"x[{edge.edge_id},{edge.v},{edge.u}]",
            )
        self.model.update()

    def _add_required_service_constraints(self) -> None:
        assert self.model is not None
        for edge in self.instance.required_edges:
            self.model.addConstr(
                self.x[(edge.edge_id, edge.u, edge.v)] + self.x[(edge.edge_id, edge.v, edge.u)] == 1,
                name=f"service[{edge.edge_id}]",
            )

    def _add_flow_constraints(self) -> None:
        assert self.model is not None
        for vertex in self.instance.vertices:
            outgoing = []
            incoming = []
            for edge in self.instance.edges:
                if edge.u == vertex:
                    outgoing.append(self.x[(edge.edge_id, edge.u, edge.v)])
                    incoming.append(self.x[(edge.edge_id, edge.v, edge.u)])
                elif edge.v == vertex:
                    outgoing.append(self.x[(edge.edge_id, edge.v, edge.u)])
                    incoming.append(self.x[(edge.edge_id, edge.u, edge.v)])
            self.model.addConstr(sum(outgoing) - sum(incoming) == 0, name=f"flow[{vertex}]")

    def _add_initial_connectivity_constraints(self) -> None:
        if not self.enable_connectivity_cuts:
            return
        assert self.model is not None
        required_vertices = {
            vertex
            for edge in self.instance.required_edges
            for vertex in (edge.u, edge.v)
        }
        launch_cut = set(required_vertices)
        if self.instance.launch_vertex in launch_cut:
            launch_cut.remove(self.instance.launch_vertex)
        if launch_cut:
            arcs = cut_arcs(self.instance, launch_cut)
            if arcs:
                self.model.addConstr(sum(self.x[arc] for arc in arcs) >= 1, name="initial_launch_conn")

        rset_data = compute_rsets(self.instance)
        all_vertices = set(self.instance.vertices)
        for rset in rset_data.rsets:
            subset = set(rset)
            if self.instance.launch_vertex in subset or subset == all_vertices:
                continue
            arcs = cut_arcs(self.instance, subset)
            if arcs:
                self.model.addConstr(sum(self.x[arc] for arc in arcs) >= 1, name=f"initial_conn[{len(subset)}]")

    def _callback(self, model, where) -> None:
        from gurobipy import GRB

        if where == GRB.Callback.MIPSOL and self.cut_at_integer_solutions:
            values = {arc: model.cbGetSolution(var) for arc, var in self.x.items()}
            self._add_callback_cuts(model, values, lazy=True)
            return

        if where == GRB.Callback.MIPNODE and self.cut_at_fractional_nodes:
            if model.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
                return
            values = {arc: model.cbGetNodeRel(var) for arc, var in self.x.items()}
            self._add_callback_cuts(model, values, lazy=False)

    def _add_callback_cuts(self, model, values: Dict[ArcKey, float], *, lazy: bool) -> None:
        if self.enable_connectivity_cuts:
            cuts = separate_connectivity_cuts(
                self.instance,
                values,
                max_cuts=self.max_cuts_per_round,
            )
            for cut in cuts:
                self._submit_cut(model, cut.arcs, cut.rhs, cut.kind, cut.vertices, lazy=lazy)

        if self.enable_r_odd_cuts:
            remaining = max(0, self.max_cuts_per_round - self.stats.cuts_added)
            if remaining:
                cuts = separate_r_odd_cuts(
                    self.instance,
                    values,
                    max_cuts=remaining,
                )
                for cut in cuts:
                    self._submit_cut(model, cut.arcs, cut.rhs, cut.kind, cut.vertices, lazy=lazy)

        if self.enable_advanced_cuts:
            # Hook for K-C, HC, zigzag, and path-bridge separators.
            pass

    def _submit_cut(
        self,
        model,
        arcs: Tuple[ArcKey, ...],
        rhs: float,
        kind: str,
        vertices: frozenset[int],
        *,
        lazy: bool,
    ) -> None:
        signature = (kind, vertices, rhs)
        seen = self._seen_lazy_cuts if lazy else self._seen_user_cuts
        if signature in seen:
            return
        seen.add(signature)
        expr = sum(self.x[arc] for arc in arcs)
        if lazy:
            model.cbLazy(expr >= rhs)
            self.stats.lazy_cuts += 1
        else:
            model.cbCut(expr >= rhs)
            self.stats.user_cuts += 1
        self.stats.cuts_added += 1
        if kind == "connectivity":
            self.stats.connectivity_cuts += 1
        elif kind == "r_odd":
            self.stats.r_odd_cuts += 1
        else:
            self.stats.advanced_cuts += 1

    def _extract_x_values(self) -> Dict[ArcKey, int]:
        values: Dict[ArcKey, int] = {}
        for arc, var in self.x.items():
            value = int(round(var.X))
            if value > 0:
                values[arc] = value
        return values

    @staticmethod
    def _status_name(status: int, grb) -> str:
        names = {
            grb.OPTIMAL: "OPTIMAL",
            grb.INFEASIBLE: "INFEASIBLE",
            grb.INF_OR_UNBD: "INF_OR_UNBD",
            grb.UNBOUNDED: "UNBOUNDED",
            grb.CUTOFF: "CUTOFF",
            grb.ITERATION_LIMIT: "ITERATION_LIMIT",
            grb.NODE_LIMIT: "NODE_LIMIT",
            grb.TIME_LIMIT: "TIME_LIMIT",
            grb.SOLUTION_LIMIT: "SOLUTION_LIMIT",
            grb.INTERRUPTED: "INTERRUPTED",
            grb.NUMERIC: "NUMERIC",
            grb.SUBOPTIMAL: "SUBOPTIMAL",
        }
        return names.get(status, str(status))
