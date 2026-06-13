from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass(frozen=True)
class Vertex:
    vid: int
    x: float
    y: float


@dataclass(frozen=True)
class OriginalLine:
    """Fine-grained polygonal-chain line from the instance file.

    chain_vertices includes the two endpoints and all intermediate split vertices,
    in traversal order.
    segment_costs[k] is the service cost of the segment from chain_vertices[k]
    to chain_vertices[k + 1].
    """

    line_id: int
    start_vertex: int
    end_vertex: int
    total_service_cost: float
    chain_vertices: Tuple[int, ...]
    segment_costs: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.chain_vertices) < 2:
            raise ValueError("Each line must contain at least two vertices")
        if len(self.segment_costs) != len(self.chain_vertices) - 1:
            raise ValueError("segment_costs length must be len(chain_vertices) - 1")

    @property
    def last_index(self) -> int:
        return len(self.chain_vertices) - 1

    def service_cost_between(self, start_index: int, end_index: int) -> float:
        if not (0 <= start_index < end_index <= self.last_index):
            raise ValueError("Invalid line interval")
        return float(sum(self.segment_costs[start_index:end_index]))

    def midpoint_index(self, start_index: int, end_index: int) -> Optional[int]:
        """Return a fine-grained vertex index near half the service cost.

        If no interior vertex exists, return None.
        """
        if end_index - start_index <= 1:
            return None
        total = self.service_cost_between(start_index, end_index)
        target = total / 2.0
        acc = 0.0
        best_idx: Optional[int] = None
        best_gap = float("inf")
        for idx in range(start_index + 1, end_index):
            acc += self.segment_costs[idx - 1]
            gap = abs(acc - target)
            if gap < best_gap:
                best_gap = gap
                best_idx = idx
        return best_idx


@dataclass(frozen=True)
class Instance:
    name: str
    original_vertex_count: int
    total_vertex_count: int
    depot_vertices: Tuple[int, ...]
    vertices: Dict[int, Vertex]
    lines: Tuple[OriginalLine, ...]


@dataclass(frozen=True)
class RequiredEdge:
    """
    An edge in the discrete problem, representing a segment of an original line
    between two consecutive breakpoints.
    """
    edge_id: str
    line_id: int
    start_index: int
    end_index: int
    start_vertex: int
    end_vertex: int
    service_cost: float

    def key(self) -> Tuple[int, int, int]:
        return (self.line_id, self.start_index, self.end_index)


@dataclass(frozen=True)
class DiscreteInstance:
    raw: Instance
    base_vertex: int
    launch_vertices: Tuple[int, ...]
    selected_breakpoints: Dict[int, Tuple[int, ...]]
    required_edges: Tuple[RequiredEdge, ...]
    edge_by_id: Dict[str, RequiredEdge]
    edge_by_key: Dict[Tuple[int, int, int], RequiredEdge]
    children_of_coarse_edge: Dict[Tuple[int, int, int], Tuple[Tuple[int, int, int], ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class Task:
    edge_id: str
    forward: bool = True  # True if the UAV flies from start_vertex to end_vertex, False otherwise


@dataclass
class Flight:
    launch_vertex: int
    tasks: List[Task] = field(default_factory=list)


@dataclass
class Solution:
    selected_launches: List[int]
    flights_by_launch: Dict[int, List[Flight]]
    objective: float = float("inf")
    makespan_by_launch: Dict[int, float] = field(default_factory=dict)
    flight_costs: Dict[Tuple[int, int], float] = field(default_factory=dict)
    paper_makespan: float = float("inf")
    ghg_makespan: float = float("inf")
    total_ghg: float = float("inf")

    def clone(self) -> "Solution":
        copied = {
            d: [Flight(f.launch_vertex, list(f.tasks)) for f in flights]
            for d, flights in self.flights_by_launch.items()
        }
        return Solution(
            selected_launches=list(self.selected_launches),
            flights_by_launch=copied,
            objective=self.objective,
            makespan_by_launch=dict(self.makespan_by_launch),
            flight_costs=dict(self.flight_costs),
            paper_makespan=self.paper_makespan,
            ghg_makespan=self.ghg_makespan,
            total_ghg=self.total_ghg,
        )

    def normalized_signature(self) -> Tuple:
        """Return a canonical signature for the solution independent of launch vertex IDs.

        This allows comparing solutions that use the same set of launches but with
        different launch vertex IDs (e.g., if launches were renumbered).
        """
        blocks = []
        for launch in sorted(self.flights_by_launch):
            flights = []
            for flight in self.flights_by_launch[launch]:
                if not flight.tasks:
                    continue
                flights.append(tuple((t.edge_id, t.forward) for t in flight.tasks))
            if flights:
                blocks.append((launch, tuple(flights)))
        return tuple(blocks)
