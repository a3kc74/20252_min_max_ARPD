"""
lcbimma_params.py — Single source of truth for LCB-IMMA parameter configuration.

Reads ``param_tunning_lcbimma.csv`` (same directory) and exposes:

  DEFAULTS    : dict
      {kwarg_key: value}  — from the "Default for others" column.
      Used as the fixed values for all parameters not currently being tuned.

  PARAM_SPECS : list of (no, short_name, kwarg_key, test_values)
      Ordered by parameter number; matches the CSV row order.
      ``short_name`` equals ``kwarg_key`` (used for directory/log labels).

Both constants are loaded once at module import.  Re-import or call
``load_params()`` directly if you need a fresh read (e.g. after editing the CSV).

CSV column layout (must not change):
  No, Parameter Name, Meaning (Vietnamese), Symbol (LaTeX),
  kwarg_key, Domain, Current Value, Test Values, Default for others

Test-value format: ``[v1;v2;v3]``  (semicolon-separated, brackets required).
"""
from __future__ import annotations

import csv
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate CSV relative to this file
# ---------------------------------------------------------------------------
_CSV_PATH = Path(__file__).resolve().parent / "param_tunning_lcbimma.csv"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_scalar(s: str):
    """Parse a string token to int (preferred) or float."""
    s = s.strip()
    # Try int first so integers keep their type (e.g. 40, not 40.0)
    try:
        return int(s)
    except ValueError:
        return float(s)


def _parse_test_values(s: str) -> list:
    """Parse ``'[0.5;1.0;2.0]'`` or ``'[5;10;20]'`` → Python list."""
    s = s.strip()
    if s.startswith("["):
        s = s[1:]
    if s.endswith("]"):
        s = s[:-1]
    return [_parse_scalar(tok) for tok in s.split(";")]


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_params(
    csv_path: Path = _CSV_PATH,
) -> tuple[dict, list[tuple[int, str, str, list]]]:
    """Read ``param_tunning_lcbimma.csv`` and return (DEFAULTS, PARAM_SPECS).

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file (default: same directory as this module).

    Returns
    -------
    defaults : dict
        ``{kwarg_key: default_value}`` from the "Default for others" column.
    param_specs : list of (no, short_name, kwarg_key, test_values)
        One entry per CSV row, ordered by ``No``.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"LCB-IMMA parameter CSV not found: {csv_path}\n"
            "Ensure 'param_tunning_lcbimma.csv' is in the same directory."
        )

    defaults: dict = {}
    param_specs: list[tuple[int, str, str, list]] = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Validate required columns
        required = {"No", "kwarg_key", "Test Values", "Default for others"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = required - set(reader.fieldnames or [])
            raise ValueError(
                f"CSV missing required columns: {missing}\n"
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            no          = int(row["No"].strip())
            kwarg       = row["kwarg_key"].strip()
            test_vals   = _parse_test_values(row["Test Values"])
            default_val = _parse_scalar(row["Default for others"])

            defaults[kwarg] = default_val
            # short_name == kwarg_key (used for output directory names / logs)
            param_specs.append((no, kwarg, kwarg, test_vals))

    return defaults, param_specs


# ---------------------------------------------------------------------------
# Module-level constants (loaded once at import)
# ---------------------------------------------------------------------------

DEFAULTS, PARAM_SPECS = load_params()

__all__ = ["DEFAULTS", "PARAM_SPECS", "load_params"]
