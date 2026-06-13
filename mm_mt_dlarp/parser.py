from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from .models import Instance, OriginalLine, Vertex


class ParseError(ValueError):
    pass


def _parse_header_value(line: str, expected_prefix: str) -> str:
    if ":" not in line:
        raise ParseError(f"Malformed header line: {line!r}")
    prefix, value = line.split(":", 1)
    if prefix.strip().upper() != expected_prefix.upper():
        raise ParseError(f"Expected header {expected_prefix!r}, got {prefix.strip()!r}")
    return value.strip()


def parse_instance(path: str | Path) -> Instance:
    text = Path(path).read_text(encoding="utf-8")
    raw_lines = [line.rstrip() for line in text.splitlines()]
    lines = [line for line in raw_lines if line.strip() != ""]
    i = 0

    try:
        name = _parse_header_value(lines[i], "NOMBRE")
        i += 1
        original_vertex_count = int(_parse_header_value(lines[i], "VERTICES ORIGINALES"))
        i += 1
        total_vertex_count = int(_parse_header_value(lines[i], "VERTICES TOTALES"))
        i += 1
        depot_count = int(_parse_header_value(lines[i], "DEPOTS"))
        i += 1
    except IndexError as exc:
        raise ParseError("Unexpected end of file while reading headers") from exc

    depots: List[int] = []
    for _ in range(depot_count):
        if i >= len(lines):
            raise ParseError("Unexpected end of file while reading depot list")
        depots.append(int(lines[i].strip()))
        i += 1

    line_count = int(_parse_header_value(lines[i], "LINEAS ORIGINALES"))
    i += 1

    parsed_lines: List[OriginalLine] = []
    for line_id in range(line_count):
        if i >= len(lines):
            raise ParseError("Unexpected end of file while reading original lines")
        header = lines[i].strip()
        i += 1
        parts = header.split()
        if len(parts) != 5 or parts[0].upper() != "LINE":
            raise ParseError(f"Malformed LINE header: {header!r}")
        v1 = int(parts[1])
        v2 = int(parts[2])
        total_cost = float(parts[3])
        n_segments = int(parts[4])

        chain_vertices = [v1]
        segment_costs: List[float] = []
        for seg_idx in range(n_segments):
            if i >= len(lines):
                raise ParseError("Unexpected end of file while reading split vertices")
            seg_line = lines[i].strip()
            i += 1
            seg_parts = seg_line.split()
            if len(seg_parts) != 2:
                raise ParseError(f"Malformed segment line: {seg_line!r}")
            u = int(seg_parts[0])
            c = float(seg_parts[1])
            chain_vertices.append(u)
            segment_costs.append(c)
        if chain_vertices[-1] != v2:
            raise ParseError(
                f"Line {line_id} last split vertex {chain_vertices[-1]} does not match end vertex {v2}"
            )
        parsed_lines.append(
            OriginalLine(
                line_id=line_id,
                start_vertex=v1,
                end_vertex=v2,
                total_service_cost=total_cost,
                chain_vertices=tuple(chain_vertices),
                segment_costs=tuple(segment_costs),
            )
        )

    if i >= len(lines):
        raise ParseError("Missing COORDENADAS section")
    _ = _parse_header_value(lines[i], "COORDENADAS")
    i += 1

    vertices: Dict[int, Vertex] = {}
    while i < len(lines):
        parts = lines[i].split()
        i += 1
        if len(parts) != 3:
            raise ParseError(f"Malformed coordinate line: {lines[i - 1]!r}")
        vid = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        vertices[vid] = Vertex(vid, x, y)

    missing_vertices = set(depots)
    for line in parsed_lines:
        missing_vertices.update(line.chain_vertices)
    missing_vertices.difference_update(vertices)
    if missing_vertices:
        raise ParseError(f"Missing coordinates for vertices: {sorted(missing_vertices)}")

    return Instance(
        name=name,
        original_vertex_count=original_vertex_count,
        total_vertex_count=total_vertex_count,
        depot_vertices=tuple(depots),
        vertices=vertices,
        lines=tuple(parsed_lines),
    )
