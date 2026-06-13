from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from .models import GRPInstance

ArcKey = Tuple[str, int, int]


@dataclass(frozen=True)
class RSetData:
    rsets: Tuple[frozenset[int], ...]
    vertex_to_rset: Dict[int, int]
    required_degree: Dict[int, int]


@dataclass(frozen=True)
class Cut:
    vertices: frozenset[int]
    arcs: Tuple[ArcKey, ...]
    rhs: float
    kind: str
    violation: float


def compute_rsets(instance: GRPInstance) -> RSetData:
    adjacency: Dict[int, Set[int]] = {vertex: set() for vertex in instance.vertices}
    required_degree: Dict[int, int] = {vertex: 0 for vertex in instance.vertices}
    required_vertices: Set[int] = set()
    for edge in instance.required_edges:
        adjacency[edge.u].add(edge.v)
        adjacency[edge.v].add(edge.u)
        required_degree[edge.u] += 1
        required_degree[edge.v] += 1
        required_vertices.add(edge.u)
        required_vertices.add(edge.v)

    seen: Set[int] = set()
    rsets: List[frozenset[int]] = []
    for vertex in sorted(required_vertices):
        if vertex in seen:
            continue
        stack = [vertex]
        component: Set[int] = set()
        seen.add(vertex)
        while stack:
            current = stack.pop()
            component.add(current)
            for nxt in adjacency[current]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        rsets.append(frozenset(component))

    if instance.launch_vertex not in required_vertices:
        rsets.append(frozenset({instance.launch_vertex}))

    vertex_to_rset: Dict[int, int] = {}
    for idx, rset in enumerate(rsets):
        for vertex in rset:
            vertex_to_rset[vertex] = idx
    return RSetData(tuple(rsets), vertex_to_rset, required_degree)


def cut_arcs(instance: GRPInstance, subset: Set[int], *, both_directions: bool = False) -> Tuple[ArcKey, ...]:
    arcs: List[ArcKey] = []
    for edge in instance.edges:
        u_inside = edge.u in subset
        v_inside = edge.v in subset
        if u_inside == v_inside:
            continue
        if u_inside:
            arcs.append((edge.edge_id, edge.u, edge.v))
            if both_directions:
                arcs.append((edge.edge_id, edge.v, edge.u))
        else:
            arcs.append((edge.edge_id, edge.v, edge.u))
            if both_directions:
                arcs.append((edge.edge_id, edge.u, edge.v))
    return tuple(arcs)


def separate_connectivity_cuts(
    instance: GRPInstance,
    values: Dict[ArcKey, float],
    *,
    epsilons: Sequence[float] = (1e-6, 0.25, 0.5),
    max_cuts: int = 50,
) -> List[Cut]:
    cuts: List[Cut] = []
    required_vertices = {
        vertex
        for edge in instance.required_edges
        for vertex in (edge.u, edge.v)
    }
    relevant_vertices = set(required_vertices)
    relevant_vertices.add(instance.launch_vertex)

    for epsilon in epsilons:
        components = _support_components(instance, values, epsilon)
        for component in components:
            if instance.launch_vertex in component:
                continue
            if not (component & relevant_vertices):
                continue
            arcs = cut_arcs(instance, component)
            lhs = sum(values.get(arc, 0.0) for arc in arcs)
            violation = 1.0 - lhs
            if violation > 1e-6:
                cuts.append(Cut(frozenset(component), arcs, 1.0, "connectivity", violation))
                if len(cuts) >= max_cuts:
                    return select_diverse_cuts(cuts, max_cuts=max_cuts)
    return select_diverse_cuts(cuts, max_cuts=max_cuts)


def separate_r_odd_cuts(
    instance: GRPInstance,
    values: Dict[ArcKey, float],
    *,
    epsilons: Sequence[float] = (1e-6, 0.25, 0.5),
    max_cuts: int = 50,
) -> List[Cut]:
    cuts: List[Cut] = []
    for epsilon in epsilons:
        components = _support_components(instance, values, epsilon, subtract_required=True)
        for component in components:
            required_crossing = [
                edge for edge in instance.required_edges
                if (edge.u in component) != (edge.v in component)
            ]
            if len(required_crossing) % 2 == 0:
                continue
            arcs = cut_arcs(instance, component, both_directions=True)
            rhs = float(len(required_crossing) + 1)
            lhs = sum(values.get(arc, 0.0) for arc in arcs)
            violation = rhs - lhs
            if violation > 1e-6:
                cuts.append(Cut(frozenset(component), arcs, rhs, "r_odd", violation))
                if len(cuts) >= max_cuts:
                    return select_diverse_cuts(cuts, max_cuts=max_cuts)
    return select_diverse_cuts(cuts, max_cuts=max_cuts)


def select_diverse_cuts(
    cuts: Iterable[Cut],
    *,
    similarity_threshold: float = 0.9,
    max_cuts: int = 50,
) -> List[Cut]:
    selected: List[Cut] = []
    for cut in sorted(cuts, key=lambda item: item.violation, reverse=True):
        if all(_jaccard(cut.vertices, other.vertices) <= similarity_threshold for other in selected):
            selected.append(cut)
            if len(selected) >= max_cuts:
                break
    return selected


def _support_components(
    instance: GRPInstance,
    values: Dict[ArcKey, float],
    epsilon: float,
    *,
    subtract_required: bool = False,
) -> List[Set[int]]:
    adjacency: Dict[int, Set[int]] = {vertex: set() for vertex in instance.vertices}
    for edge in instance.edges:
        weight = (
            values.get((edge.edge_id, edge.u, edge.v), 0.0)
            + values.get((edge.edge_id, edge.v, edge.u), 0.0)
        )
        if subtract_required and edge.required:
            weight -= 1.0
        if weight > epsilon:
            adjacency[edge.u].add(edge.v)
            adjacency[edge.v].add(edge.u)

    components: List[Set[int]] = []
    seen: Set[int] = set()
    for vertex in instance.vertices:
        if vertex in seen:
            continue
        stack = [vertex]
        component: Set[int] = set()
        seen.add(vertex)
        while stack:
            current = stack.pop()
            component.add(current)
            for nxt in adjacency[current]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(component)
    return components


def _jaccard(left: Set[int] | frozenset[int], right: Set[int] | frozenset[int]) -> float:
    union = len(left | right)
    if union == 0:
        return 0.0
    return len(left & right) / union
