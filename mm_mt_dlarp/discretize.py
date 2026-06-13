from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .models import DiscreteInstance, Instance, RequiredEdge, Solution, Task


def initial_breakpoints(instance: Instance) -> Dict[int, Tuple[int, ...]]:
    """Initialize breakpoints for all lines to include only endpoints.
    
    Args:
        instance: The original instance.
    
    Returns:
        A dictionary mapping each line ID to a tuple of breakpoints containing
        only the start and end indices of the line.
    """
    return {line.line_id: (0, line.last_index) for line in instance.lines}


def build_discrete_instance(
    instance: Instance,
    base_vertex: int,
    launch_vertices: Sequence[int],
    selected_breakpoints: Dict[int, Sequence[int]],
) -> DiscreteInstance:
    """
    Build a discrete instance from an instance and selected breakpoints.

    Args:
        instance: The original instance.
        base_vertex: The base vertex ID (unused in this implementation).
        launch_vertices: A sequence of launch vertex IDs.
        selected_breakpoints: A dictionary mapping each line ID to a sequence of
            breakpoint indices (including endpoints).

    Returns:
        A DiscreteInstance representing the discretized problem.
    """
    required_edges: List[RequiredEdge] = []
    edge_by_id: Dict[str, RequiredEdge] = {}
    edge_by_key: Dict[Tuple[int, int, int], RequiredEdge] = {}
    for line in instance.lines:
        breaks = sorted(set(int(v) for v in selected_breakpoints[line.line_id]))
        if breaks[0] != 0 or breaks[-1] != line.last_index:
            raise ValueError(f"Line {line.line_id} breakpoints must include endpoints")
        for start_index, end_index in zip(breaks[:-1], breaks[1:]):
            edge = RequiredEdge(
                edge_id=f"L{line.line_id}_{start_index}_{end_index}",
                line_id=line.line_id,
                start_index=start_index,
                end_index=end_index,
                start_vertex=line.chain_vertices[start_index],
                end_vertex=line.chain_vertices[end_index],
                service_cost=line.service_cost_between(start_index, end_index),
            )
            required_edges.append(edge)
            edge_by_id[edge.edge_id] = edge
            edge_by_key[edge.key()] = edge

    return DiscreteInstance(
        raw=instance,
        base_vertex=base_vertex,
        launch_vertices=tuple(launch_vertices),
        selected_breakpoints={k: tuple(sorted(set(v))) for k, v in selected_breakpoints.items()},
        required_edges=tuple(required_edges),
        edge_by_id=edge_by_id,
        edge_by_key=edge_by_key,
    )


def add_midpoints_to_all_intervals(
    instance: Instance,
    current_breakpoints: Dict[int, Sequence[int]],
) -> Dict[int, Tuple[int, ...]]:
    """
    Add midpoint breakpoints to all intervals in the current breakpoints.

    Args:
        instance: The original instance.
        current_breakpoints: A dictionary mapping each line ID to a sequence of
            breakpoint indices (including endpoints).

    Returns:
        A dictionary mapping each line ID to a tuple of breakpoints including
        midpoints of all intervals.
    """
    updated: Dict[int, Tuple[int, ...]] = {}
    for line in instance.lines:
        breaks = sorted(set(int(v) for v in current_breakpoints[line.line_id]))
        new_breaks: Set[int] = set(breaks)
        for a, b in zip(breaks[:-1], breaks[1:]):
            mid = line.midpoint_index(a, b)
            if mid is not None:
                new_breaks.add(mid)
        updated[line.line_id] = tuple(sorted(new_breaks))
    return updated


def map_parent_children(
    old_breakpoints: Dict[int, Sequence[int]],
    new_breakpoints: Dict[int, Sequence[int]],
) -> Dict[Tuple[int, int, int], Tuple[Tuple[int, int, int], ...]]:
    """
    Map parent edges to their children edges across refinement levels.

    Args:
        old_breakpoints: The coarser breakpoints.
        new_breakpoints: The finer breakpoints.

    Returns:
        A dictionary mapping each parent edge (line_id, start_index, end_index) to
        a tuple of child edges (line_id, child_start_index, child_end_index).
    """
    mapping: Dict[Tuple[int, int, int], Tuple[Tuple[int, int, int], ...]] = {}
    for line_id, old_bps in old_breakpoints.items():
        old_sorted = sorted(set(int(v) for v in old_bps))
        new_sorted = sorted(set(int(v) for v in new_breakpoints[line_id]))
        for a, b in zip(old_sorted[:-1], old_sorted[1:]):
            child_points = [idx for idx in new_sorted if a <= idx <= b]
            children = [(u, v) for u, v in zip(child_points[:-1], child_points[1:])]
            mapping[(line_id, a, b)] = tuple((line_id, x, y) for x, y in children)
    return mapping


def convert_solution_to_refined_instance(
    solution: Solution,
    old_instance: DiscreteInstance,
    new_instance: DiscreteInstance,
    parent_children: Dict[Tuple[int, int, int], Tuple[Tuple[int, int, int], ...]],
) -> Solution:
    """
    Convert a solution from the old instance to the new instance.

    This function reconstructs the flights for the new instance based on the flights
    in the old instance and the mapping between parent and child edges.

    Args:
        solution: The solution in the old instance.
        old_instance: The old discrete instance.
        new_instance: The new discrete instance.
        parent_children: Mapping from parent edges to their children edges.

    Returns:
        A new solution in the new instance.
    """
    converted = solution.clone()
    converted.flight_costs.clear()
    converted.makespan_by_launch.clear()
    converted.objective = float("inf")
    for launch, flights in converted.flights_by_launch.items():
        for flight in flights:
            new_tasks: List[Task] = []
            for task in flight.tasks:
                edge = old_instance.edge_by_id[task.edge_id]
                child_keys = list(parent_children[edge.key()])
                if not task.forward:
                    child_keys = list(reversed(child_keys))
                for child_key in child_keys:
                    child = new_instance.edge_by_key[child_key]
                    forward = True
                    if not task.forward:
                        forward = False
                    new_tasks.append(Task(child.edge_id, forward))
            flight.tasks = new_tasks
    return converted


def detect_used_midpoints(solution: Solution, instance: DiscreteInstance) -> Dict[int, Set[int]]:
    """Return selected non-endpoint breakpoints that are used as entry/exit points.

    A midpoint is considered unused when its two incident child edges are traversed
    consecutively in the same flight in a consistent direction, meaning the route
    simply passes through the midpoint as an internal point.
    """
    used: Dict[int, Set[int]] = {line.line_id: set() for line in instance.raw.lines}
    adjacency_ok: Set[Tuple[int, int]] = set()

    for flights in solution.flights_by_launch.values():
        for flight in flights:
            seq = [instance.edge_by_id[t.edge_id] for t in flight.tasks]
            dirs = [t.forward for t in flight.tasks]
            for idx in range(len(seq) - 1):
                left = seq[idx]
                right = seq[idx + 1]
                lf = dirs[idx]
                rf = dirs[idx + 1]
                if left.line_id != right.line_id:
                    continue
                if left.end_index == right.start_index and lf and rf:
                    adjacency_ok.add((left.line_id, left.end_index))
                elif right.end_index == left.start_index and (not lf) and (not rf):
                    adjacency_ok.add((left.line_id, left.start_index))

    for line in instance.raw.lines:
        breaks = sorted(instance.selected_breakpoints[line.line_id])
        for mid in breaks[1:-1]:
            if (line.line_id, mid) not in adjacency_ok:
                used[line.line_id].add(mid)
    return used


def refine_breakpoints_from_used_midpoints(
    instance: Instance,
    current_breakpoints: Dict[int, Sequence[int]],
    used_midpoints: Dict[int, Set[int]],
) -> Dict[int, Tuple[int, ...]]:
    refined: Dict[int, Tuple[int, ...]] = {}
    for line in instance.lines:
        breaks = sorted(set(int(v) for v in current_breakpoints[line.line_id]))
        keep: Set[int] = {0, line.last_index}
        used = set(used_midpoints.get(line.line_id, set()))
        keep.update(used)
        for mid in used:
            pos = breaks.index(mid)
            left = breaks[pos - 1]
            right = breaks[pos + 1]
            new_left = line.midpoint_index(left, mid)
            new_right = line.midpoint_index(mid, right)
            if new_left is not None:
                keep.add(new_left)
            if new_right is not None:
                keep.add(new_right)
        refined[line.line_id] = tuple(sorted(keep))
    return refined

def convert_solution_between_instances(
    solution: Solution,
    old_instance: DiscreteInstance,
    new_instance: DiscreteInstance,
) -> Solution:
    """Port a solution from old_instance to new_instance by mapping continuous line segments."""
    converted = solution.clone()
    converted.flight_costs.clear()
    converted.makespan_by_launch.clear()
    converted.objective = float("inf")
    
    for launch, flights in converted.flights_by_launch.items():
        for flight in flights:
            new_tasks: List[Task] = []
            if not flight.tasks:
                continue
                
            segments = []
            for task in flight.tasks:
                edge = old_instance.edge_by_id[task.edge_id]
                if not segments:
                    segments.append([edge.line_id, edge.start_index, edge.end_index, task.forward])
                else:
                    last = segments[-1]
                    if edge.line_id == last[0] and task.forward == last[3]:
                        if task.forward and last[2] == edge.start_index:
                            last[2] = edge.end_index
                            continue
                        elif not task.forward and last[1] == edge.end_index:
                            last[1] = edge.start_index
                            continue
                    segments.append([edge.line_id, edge.start_index, edge.end_index, task.forward])
            
            for line_id, s_idx, e_idx, forward in segments:
                pieces = []
                for new_edge in new_instance.required_edges:
                    if new_edge.line_id == line_id and new_edge.start_index >= s_idx and new_edge.end_index <= e_idx:
                        pieces.append(new_edge)
                pieces.sort(key=lambda e: e.start_index)
                
                if not forward:
                    pieces.reverse()
                    for p in pieces:
                        new_tasks.append(Task(p.edge_id, False))
                else:
                    for p in pieces:
                        new_tasks.append(Task(p.edge_id, True))
            
            flight.tasks = new_tasks
            
    return converted