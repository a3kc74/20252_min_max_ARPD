from __future__ import annotations

import csv
import math
import sys
import time
import zipfile
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.run_benchmark import build_instance, run_one_algorithm  # noqa: E402
from mm_mt_dlarp.algorithms import LNSSolver, VNDSolver, VNSSolver  # noqa: E402


CONFIGS: list[tuple[str, int, float]] = [
    ("MTLARP4_4_3_1.dat", 2, 847),
    ("MTLARP4_4_3_2.dat", 2, 1033),
    ("MTLARP4_4_4_1.dat", 2, 746),
    ("MTLARP4_4_4_1.dat", 3, 746),
    ("MTLARP4_4_4_2.dat", 2, 847),
    ("MTLARP4_4_4_2.dat", 3, 847),
    ("MTLARP4_4_5_1.dat", 2, 785),
    ("MTLARP4_4_5_1.dat", 3, 785),
    ("MTLARP4_4_5_1.dat", 4, 785),
    ("MTLARP4_4_5_2.dat", 2, 900),
    ("MTLARP4_4_5_2.dat", 3, 900),
    ("MTLARP4_4_5_2.dat", 4, 900),
    ("MTLARP4_4_6_1.dat", 2, 694),
    ("MTLARP4_4_6_1.dat", 3, 694),
    ("MTLARP4_4_6_1.dat", 4, 694),
    ("MTLARP4_4_6_1.dat", 5, 694),
    ("MTLARP4_4_6_2.dat", 2, 903),
    ("MTLARP4_4_6_2.dat", 3, 903),
    ("MTLARP4_4_6_2.dat", 4, 903),
    ("MTLARP4_4_6_2.dat", 5, 903),
]

ALGORITHMS = [
    ("vns", VNSSolver, {"k_max": 4, "max_iter": 100}),
    ("lns", LNSSolver, {"destroy_frac": 0.3, "max_iter": 500}),
    ("vnd", VNDSolver, {}),
]

FIELDS = [
    "Algorithm",
    "Instance",
    "P",
    "L",
    "Base",
    "Flight Limit",
    "Trucks Used",
    "Trucks Submitted",
    "Objective Type",
    "Objective",
    "Paper Makespan",
    "GHG Makespan",
    "Total GHG",
    "Time",
    "Flight Optimizer",
    "Seed",
    "Status",
    "Error",
]


def _format_value(value):
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        if math.isnan(value):
            return "nan"
        return round(value, 6)
    return value


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _excel_col(index: int) -> str:
    letters = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _sheet_xml(rows: Iterable[Iterable]) -> str:
    xml_rows = []
    for r_idx, row in enumerate(rows, start=1):
        cells = []
        for c_idx, value in enumerate(row, start=1):
            ref = f"{_excel_col(c_idx)}{r_idx}"
            if value is None:
                cells.append(f'<c r="{ref}"/>')
            elif isinstance(value, bool):
                cells.append(f'<c r="{ref}" t="b"><v>{1 if value else 0}</v></c>')
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>')
        xml_rows.append(f'<row r="{r_idx}">{"".join(cells)}</row>')

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
        'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        '<sheetData>'
        + "".join(xml_rows)
        + '</sheetData><autoFilter ref="A1:R1"/></worksheet>'
    )


def _write_xlsx(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = [FIELDS] + [[_format_value(row.get(field, "")) for field in FIELDS] for row in rows]

    files = {
        "[Content_Types].xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
            '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
            '</Types>'
        ),
        "_rels/.rels": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
            '</Relationships>'
        ),
        "docProps/core.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            '<dc:title>VNS LNS VND Benchmark</dc:title>'
            '<dc:creator>run_vns_lns_vnd_configs.py</dc:creator>'
            '</cp:coreProperties>'
        ),
        "docProps/app.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
            'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
            '<Application>Python</Application>'
            '</Properties>'
        ),
        "xl/workbook.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="results" sheetId="1" r:id="rId1"/></sheets>'
            '</workbook>'
        ),
        "xl/_rels/workbook.xml.rels": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            '</Relationships>'
        ),
        "xl/worksheets/sheet1.xml": _sheet_xml(table),
    }

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)


def main() -> None:
    output_dir = ROOT / "exp" / "results" / "vns_lns_vnd_configs"
    output_csv = output_dir / "results.csv"
    output_xlsx = output_dir / "results.xlsx"

    rows: list[dict] = []
    total = len(CONFIGS) * len(ALGORITHMS)
    done = 0

    for instance_name, p, l_value in CONFIGS:
        instance_path = ROOT / "data" / "instances" / instance_name
        instance, config = build_instance(instance_path, p, l_value, base_vertex=None)
        config.objective_type = "minmax_ghg"
        config.flight_optimizer = "bc"
        config.verbose = False

        for algorithm_name, solver_cls, kwargs in ALGORITHMS:
            done += 1
            print(f"[{done:02d}/{total}] {algorithm_name.upper():<3} {instance_name} P={p} L={l_value}", flush=True)
            try:
                bench_row = run_one_algorithm(
                    name=algorithm_name,
                    solver_cls=solver_cls,
                    instance=instance,
                    config_template=config,
                    seed=0,
                    verbose=False,
                    solver_kwargs=kwargs,
                )
                row = bench_row.as_csv_row()
                row.update(
                    {
                        "P": p,
                        "L": l_value,
                        "Seed": 0,
                        "Status": "ok",
                        "Error": "",
                    }
                )
                print(
                    f"       objective={row['Objective']} paper_makespan={row['Paper Makespan']} "
                    f"total_ghg={row['Total GHG']} time={row['Time']}s",
                    flush=True,
                )
            except Exception as exc:  # Keep the batch running and record failures in Excel.
                row = {
                    "Algorithm": algorithm_name,
                    "Instance": instance_name,
                    "P": p,
                    "L": l_value,
                    "Base": "",
                    "Flight Limit": l_value,
                    "Trucks Used": "",
                    "Trucks Submitted": p,
                    "Objective Type": config.objective_type,
                    "Objective": "",
                    "Paper Makespan": "",
                    "GHG Makespan": "",
                    "Total GHG": "",
                    "Time": "",
                    "Flight Optimizer": config.flight_optimizer,
                    "Seed": 0,
                    "Status": "error",
                    "Error": repr(exc),
                }
                print(f"       ERROR: {exc!r}", flush=True)

            rows.append(row)
            _write_csv(rows, output_csv)
            _write_xlsx(rows, output_xlsx)

    print(f"Wrote CSV : {output_csv}")
    print(f"Wrote XLSX: {output_xlsx}")


if __name__ == "__main__":
    started = time.perf_counter()
    main()
    print(f"Total wall time: {time.perf_counter() - started:.2f}s")