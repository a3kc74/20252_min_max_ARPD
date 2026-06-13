from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..models import Flight, Task
from .branch_cut import BranchCutResult
from .models import GRPEdge, GRPInstance


def reconstruct_flight(instance: GRPInstance, result: BranchCutResult) -> Flight:
    if result.x_values:
        return _reconstruct_from_x_values(instance, result)
    return Flight(instance.launch_vertex, list(result.tasks))


def _reconstruct_from_x_values(instance: GRPInstance, result: BranchCutResult) -> Flight:
    edge_by_id: Dict[str, GRPEdge] = {edge.edge_id: edge for edge in instance.edges}
    adjacency: Dict[int, List[Tuple[str, int, int]]] = {vertex: [] for vertex in instance.vertices}
    for (edge_id, u, v), count in result.x_values.items():
        for _ in range(count):
            adjacency.setdefault(u, []).append((edge_id, u, v))

    stack: List[Tuple[int, Optional[Tuple[str, int, int]]]] = [(instance.launch_vertex, None)]
    circuit: List[Optional[Tuple[str, int, int]]] = []
    while stack:
        vertex, _ = stack[-1]
        if adjacency.get(vertex):
            edge_id, u, v = adjacency[vertex].pop()
            stack.append((v, (edge_id, u, v)))
        else:
            _, incoming = stack.pop()
            circuit.append(incoming)

    ordered_arcs = [arc for arc in reversed(circuit) if arc is not None]
    tasks: List[Task] = []
    for edge_id, u, v in ordered_arcs:
        edge = edge_by_id[edge_id]
        if not edge.required:
            continue
        source_edge_id = edge.source_task_edge_id or edge.edge_id
        tasks.append(Task(source_edge_id, forward=(u == edge.u and v == edge.v)))
    return Flight(instance.launch_vertex, tasks)
