from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple


@dataclass(frozen=True)
class GRPEdge:
    edge_id: str
    u: int
    v: int
    cost_uv: float
    cost_vu: float
    required: bool
    kind: Literal["required_service", "deadhead"]
    source_task_edge_id: Optional[str] = None


@dataclass(frozen=True)
class GRPInstance:
    launch_vertex: int
    vertices: Tuple[int, ...]
    edges: Tuple[GRPEdge, ...]

    @property
    def required_edges(self) -> Tuple[GRPEdge, ...]:
        return tuple(edge for edge in self.edges if edge.required)
