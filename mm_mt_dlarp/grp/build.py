from __future__ import annotations

import math
from typing import List, Set

from ..models import DiscreteInstance, Flight
from .models import GRPEdge, GRPInstance


def _distance(discrete: DiscreteInstance, v1: int, v2: int) -> float:
    a = discrete.raw.vertices[v1]
    b = discrete.raw.vertices[v2]
    return math.hypot(a.x - b.x, a.y - b.y)


def build_grp_from_flight(discrete: DiscreteInstance, flight: Flight) -> GRPInstance:
    """Build the induced GRP graph for one UAV flight.

    Required service edges represent tasks. Non-required edges form the complete
    deadheading graph on the launch vertex plus all endpoints of required edges.
    """
    vertex_set: Set[int] = {flight.launch_vertex}
    edges: List[GRPEdge] = []

    for idx, task in enumerate(flight.tasks):
        required = discrete.edge_by_id[task.edge_id]
        vertex_set.add(required.start_vertex)
        vertex_set.add(required.end_vertex)
        edges.append(
            GRPEdge(
                edge_id=f"service:{idx}:{required.edge_id}",
                u=required.start_vertex,
                v=required.end_vertex,
                cost_uv=required.service_cost,
                cost_vu=required.service_cost,
                required=True,
                kind="required_service",
                source_task_edge_id=required.edge_id,
            )
        )

    vertices = tuple(sorted(vertex_set))
    for i, u in enumerate(vertices):
        for v in vertices[i + 1:]:
            cost = _distance(discrete, u, v)
            edges.append(
                GRPEdge(
                    edge_id=f"deadhead:{u}:{v}",
                    u=u,
                    v=v,
                    cost_uv=cost,
                    cost_vu=cost,
                    required=False,
                    kind="deadhead",
                )
            )

    return GRPInstance(
        launch_vertex=flight.launch_vertex,
        vertices=vertices,
        edges=tuple(edges),
    )
