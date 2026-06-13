# -*- coding: utf-8 -*-
"""
kiemdinh.py — Statistical testing for LCB-IMMA vs VND vs VNS vs LNS.

Runs the full analysis for two metrics (GHG objective and running time) and
generates six LaTeX tables:

  GHG Objective
    tab:friedman          Friedman test
    tab:T10_wilcoxon      Pairwise Wilcoxon, raw p-values
    tab:exp2_wilcoxon     Pairwise Wilcoxon, Holm-corrected

  Running time
    tab:friedman_time     Friedman test
    tab:T10_wilcoxon_time Pairwise Wilcoxon, raw p-values
    tab:exp2_wilcoxon_time Pairwise Wilcoxon, Holm-corrected

Data source  : exp/summary_final.csv   (output of make_final_sum.py)
Output folder: exp/result/

Usage:
  python exp/kiemdinh.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ==============================================================
# Holm-Bonferroni correction (no external dependencies)
# ==============================================================

def holm_correction(p_values: list[float], alpha: float = 0.05) -> tuple:
    """
    Holm step-down multiple-comparison correction.

    Returns
    -------
    reject : bool ndarray — whether H0 is rejected at the corrected level
    p_adj  : float ndarray — Holm-adjusted p-values (same order as input)
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order        = np.argsort(p)
    p_sorted     = p[order]
    multipliers  = n - np.arange(n)
    p_adj_sorted = np.minimum(p_sorted * multipliers, 1.0)
    # Enforce monotonicity (step-down)
    for i in range(1, n):
        if p_adj_sorted[i] < p_adj_sorted[i - 1]:
            p_adj_sorted[i] = p_adj_sorted[i - 1]
    p_adj        = np.empty(n)
    p_adj[order] = p_adj_sorted
    return p_adj < alpha, p_adj


# ==============================================================
# 1.  Load and prepare data
# ==============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH  = SCRIPT_DIR / "summary_final.csv"
OUT_DIR    = SCRIPT_DIR / "result"

if not DATA_PATH.exists():
    print(f"[ERROR] summary_final.csv not found at {DATA_PATH}")
    print("Run make_final_sum.py first.")
    sys.exit(1)

df_all = pd.read_csv(DATA_PATH)
df_all["fitness"] = pd.to_numeric(df_all["fitness"], errors="coerce")
df_all["runtime_s"] = pd.to_numeric(df_all["runtime_s"], errors="coerce")
df_all["run_num"]   = pd.to_numeric(df_all["run_num"],   errors="coerce")

# Canonical algorithm names in the data (display name → internal key)
ALGO_MAP = {
    "LCB-IMMA": "LCB_IMMA",
    "IMMA":     "IMMA",
    "MA":       "MA",
    "GA":       "GA",
    "VND":      "VND",
    "VNS":      "VNS",
    "LNS":      "LNS",
}
df = df_all[df_all["algorithm"].isin(ALGO_MAP)].copy()
df["algorithm"] = df["algorithm"].map(ALGO_MAP)
df = df.sort_values(["instance", "run_num"]).reset_index(drop=True)

# ==============================================================
# 2.  Configuration
# ==============================================================

# Algorithms (ordered for display: proposed first, then ablation ladder, then baselines)
ALGOS = ["LCB_IMMA", "IMMA", "MA", "GA", "VND", "VNS", "LNS"]

# Pairwise comparisons of interest:
#   proposed vs baselines, and ablation ladder steps
PAIRS = [
    # Proposed vs single-solution baselines
    ("LCB_IMMA", "VND"),
    ("LCB_IMMA", "VNS"),
    ("LCB_IMMA", "LNS"),
    # Ablation ladder: each step isolates one component
    ("LCB_IMMA", "IMMA"),   # benefit of LinUCB bandit
    ("IMMA",     "MA"),     # benefit of island model
    ("MA",       "GA"),     # benefit of local search
]
ALPHA = 0.05
SEP   = "=" * 72

# Select representative instances (one per scale, or first N if data is partial)
# Update this list after running exp3 with the actual instance names you want to report.
_available_instances = sorted(df["instance"].unique())
INSTANCES: list[str] = _available_instances[:6] if len(_available_instances) >= 6 else _available_instances


def get_data(instance: str, algo: str, metric: str) -> np.ndarray:
    """Return metric values for (instance, algo), sorted by run_num."""
    mask = (df["instance"] == instance) & (df["algorithm"] == algo)
    return df.loc[mask].sort_values("run_num")[metric].to_numpy()


# ==============================================================
# 3.  LaTeX helpers
# ==============================================================

def _p_tex(p: float | None) -> str:
    if p is None:
        return "--"
    if p < 0.0001:
        return r"$<0.0001$"
    return "${:.4f}$".format(p)


def _w_tex(w: float | None) -> str:
    return "--" if w is None else "${:.1f}$".format(w)


def _chi2_tex(v: float | None) -> str:
    return "--" if v is None else "${:.2f}$".format(v)


# ==============================================================
# 4.  Core analysis for one metric
# ==============================================================

def analyse_metric(metric: str, metric_label: str) -> tuple:
    """Run Friedman + Wilcoxon (raw + Holm) for `metric` across INSTANCES."""

    # ── Friedman ──────────────────────────────────────────────────────────
    print()
    print(SEP)
    print(f"FRIEDMAN TEST -- {metric_label}")
    print(SEP)

    friedman_rows = []
    for g in INSTANCES:
        data = {a: get_data(g, a, metric) for a in ALGOS}
        if any(len(v) == 0 for v in data.values()):
            friedman_rows.append({"inst": g, "chi2": None, "df": len(ALGOS) - 1, "p": None})
            print(f"  {g:<32}  no data")
            continue
        is_tied = all(np.allclose(data[ALGOS[0]], data[a]) for a in ALGOS[1:])
        if is_tied:
            friedman_rows.append({"inst": g, "chi2": None, "df": len(ALGOS) - 1, "p": None})
            print(f"  {g:<32}  all ties")
        else:
            chi2, p = friedmanchisquare(*[data[a] for a in ALGOS])
            friedman_rows.append({"inst": g, "chi2": chi2, "df": len(ALGOS) - 1, "p": p})
            print(f"  {g:<32}  chi2={chi2:7.2f}   df={len(ALGOS)-1}   p={p:.4e}")

    # ── Pairwise Wilcoxon (raw) ───────────────────────────────────────────
    print()
    print(SEP)
    print(f"PAIRWISE WILCOXON (raw) -- {metric_label}")
    print(SEP)

    wilcox_rows = []
    for g in INSTANCES:
        data = {a: get_data(g, a, metric) for a in ALGOS}
        row = {"inst": g}
        for a1, a2 in PAIRS:
            d1 = data.get(a1, np.array([]))
            d2 = data.get(a2, np.array([]))
            key = f"{a1}_vs_{a2}"
            if len(d1) == 0 or len(d2) == 0 or len(d1) != len(d2):
                row[key] = {"W": None, "p": None, "tie": True}
                print(f"  {g:<32}  {a1} vs {a2:<12}  insufficient data")
                continue
            if np.allclose(d1 - d2, 0.0):
                row[key] = {"W": None, "p": None, "tie": True}
                print(f"  {g:<32}  {a1} vs {a2:<12}  ties")
                continue
            try:
                W, p = wilcoxon(d1, d2, alternative="two-sided", zero_method="wilcox")
                row[key] = {"W": W, "p": p, "tie": False}
                sig = ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < ALPHA else "ns")
                print(f"  {g:<32}  {a1} vs {a2:<12}  W={W:8.1f}   p={p:.4e}  {sig}")
            except ValueError as exc:
                row[key] = {"W": None, "p": None, "tie": True}
                print(f"  {g:<32}  {a1} vs {a2:<12}  ties ({exc})")
        wilcox_rows.append(row)

    # ── Holm correction ──────────────────────────────────────────────────
    print()
    print(SEP)
    print(f"PAIRWISE WILCOXON + HOLM -- {metric_label}")
    print(SEP)

    valid_tests = []
    for i, g in enumerate(INSTANCES):
        for a1, a2 in PAIRS:
            r = wilcox_rows[i].get(f"{a1}_vs_{a2}", {"tie": True})
            if not r.get("tie", True):
                valid_tests.append({
                    "inst": g, "a1": a1, "a2": a2,
                    "W": r["W"], "p_raw": r["p"],
                })

    if valid_tests:
        reject, p_adj_list = holm_correction([t["p_raw"] for t in valid_tests], alpha=ALPHA)
        for idx, t in enumerate(valid_tests):
            t["p_adj"]  = p_adj_list[idx]
            t["reject"] = bool(reject[idx])

    corrected = {(t["inst"], t["a1"], t["a2"]): t for t in valid_tests}
    for t in valid_tests:
        sig = "*" if t["reject"] else "ns"
        print(
            f"  {t['inst']:<32}  {t['a1']} vs {t['a2']:<12}  "
            f"W={t['W']:8.1f}  p_raw={t['p_raw']:.4e}  "
            f"p_adj(holm)={t['p_adj']:.4e}  {sig}"
        )

    return friedman_rows, wilcox_rows, corrected


# ==============================================================
# 5.  LaTeX builders
# ==============================================================

def build_friedman_tex(rows: list, label: str, metric_label: str, metric_unit: str) -> str:
    unit_note = f" ({metric_unit})" if metric_unit else ""
    df_val = len(ALGOS) - 1
    algos_str = ", ".join(ALGOS)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        (
            r"\caption{Friedman test (" + metric_label + unit_note + r") comparing "
            + algos_str + r". Each row: one representative instance, "
            + str(RUNS_REF) + r" runs per algorithm"
            + r" ($\alpha=0.05$, df\,=\," + str(df_val) + r").}"
        ),
        r"\label{" + label + r"}",
        r"\begin{tabular}{lrrl}",
        r"\toprule",
        r"Instance & $\chi^2$ & df & $p$-value \\",
        r"\midrule",
    ]
    for row in rows:
        g     = row["inst"].replace("_", r"\_")
        chi2s = _chi2_tex(row["chi2"])
        ps    = _p_tex(row["p"])
        note  = r"\quad\emph{(all ties)}" if row["p"] is None else ""
        lines.append(f"{g} & {chi2s} & {df_val} & {ps}{note} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def build_wilcoxon_tex(
    rows_raw: list,
    corrected_lookup: dict,
    label: str,
    metric_label: str,
    metric_unit: str,
    use_adj: bool,
) -> str:
    pair_headers = " & ".join(
        r"\multicolumn{2}{c}{" + f"{a1} vs {a2}" + "}"
        for a1, a2 in PAIRS
    )
    sub_col = r"$W$ & $p_{\mathrm{adj}}$" if use_adj else r"$W$ & $p$"
    sub_header = " & ".join([sub_col] * len(PAIRS))
    col_spec = "l" + " rr" * len(PAIRS)
    unit_note = f", {metric_unit}" if metric_unit else ""
    adj_note = (
        r" Asterisk ($^{*}$) marks pairs significant after Holm correction."
        if use_adj else ""
    )
    caption = (
        r"\caption{Pairwise Wilcoxon signed-rank test (" + metric_label + unit_note + r")."
        + r" " + str(RUNS_REF) + r" paired observations per instance."
        + r" $\alpha=0.05$." + adj_note + r"}"
    )
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        caption,
        r"\label{" + label + r"}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        f"Instance & {pair_headers} \\\\",
        f" & {sub_header} \\\\",
        r"\midrule",
    ]
    for i, g in enumerate(INSTANCES):
        row   = rows_raw[i]
        g_tex = g.replace("_", r"\_")
        cells = []
        for a1, a2 in PAIRS:
            key_raw = f"{a1}_vs_{a2}"
            raw     = row.get(key_raw, {"tie": True})
            key_cor = (g, a1, a2)
            if raw.get("tie", True):
                cells.append(r"\multicolumn{2}{c}{\emph{ties}}")
                continue
            W_tex = _w_tex(raw["W"])
            if use_adj:
                if key_cor in corrected_lookup:
                    t   = corrected_lookup[key_cor]
                    sig = "^{*}" if t["reject"] else ""
                    p   = t["p_adj"]
                    p_tex = (
                        f"$<0.0001{sig}$" if p < 0.0001
                        else "${:.4f}{sig}$".format(p, sig=sig)
                    )
                    cells.append(f"{W_tex} & {p_tex}")
                else:
                    cells.append(f"{W_tex} & --")
            else:
                cells.append(f"{W_tex} & {_p_tex(raw['p'])}")
        lines.append(f"{g_tex} & {' & '.join(cells)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    return "\n".join(lines)


# Number of runs per algorithm (for LaTeX captions)
RUNS_REF = int(df.groupby(["instance", "algorithm"])["run_num"].max().max()) if not df.empty else 20

# ==============================================================
# 6.  Run analysis for both metrics
# ==============================================================

METRICS = [
    # (column,      display_label,   unit,      label_suffix)
    ("fitness",    "GHG Objective", "",         ""),
    ("runtime_s",  "Running Time",  "seconds",  "_time"),
]

all_tex: dict[str, str] = {}

for col, label_str, unit, suffix in METRICS:
    fr_rows, wx_rows, corr = analyse_metric(col, label_str)

    lbl_fr  = f"tab:friedman{suffix}"
    lbl_raw = f"tab:T10_wilcoxon{suffix}"
    lbl_adj = f"tab:exp2_wilcoxon{suffix}"

    tex_fr  = build_friedman_tex(fr_rows,  lbl_fr,  label_str, unit)
    tex_raw = build_wilcoxon_tex(wx_rows, corr, lbl_raw, label_str, unit, use_adj=False)
    tex_adj = build_wilcoxon_tex(wx_rows, corr, lbl_adj, label_str, unit, use_adj=True)

    for title, tex in [(lbl_fr, tex_fr), (lbl_raw, tex_raw), (lbl_adj, tex_adj)]:
        print()
        print(SEP)
        print(f"LaTeX -- {title}")
        print(SEP)
        print(tex)

    all_tex[f"tab_friedman{suffix}.tex"]     = tex_fr
    all_tex[f"tab_T10_wilcoxon{suffix}.tex"] = tex_raw
    all_tex[f"tab_exp2_wilcoxon{suffix}.tex"] = tex_adj

# ==============================================================
# 7.  Save all .tex files
# ==============================================================

OUT_DIR.mkdir(parents=True, exist_ok=True)
print()
print(SEP)
print("Saved files")
print(SEP)
for fname, content in all_tex.items():
    fpath = OUT_DIR / fname
    fpath.write_text(content + "\n", encoding="utf-8")
    print(f"  {fpath}")

print()
print("Done.")
