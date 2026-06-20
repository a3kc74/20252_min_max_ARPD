# Implementation Notes

## 2026-06-20 — VNS Phase 1 improvement

### Files changed

- `mm_mt_dlarp/algorithms/vns.py`
- `docs/vns_context.md`
- `docs/implementation-notes.md`

### Motivation

The previous VNS implementation reused improvement-oriented neighborhoods for shaking. In particular, `destroy_and_repair()` only returned a candidate when it improved the input solution. That behavior is suitable for local search, but weak for VNS shaking because the search may need to move to feasible non-improving neighbors to escape local optima and plateaus.

Phase 1 strengthens the VNS diversification layer while preserving the existing feasibility/evaluation logic from `MatheuristicBase`.

### Implemented changes

#### Real perturbation shaking

Added explicit perturbation operators in `VNSSolver`:

- `_random_task_relocate()`
  - removes random tasks from the current solution;
  - reinserts each removed task into a random feasible position;
  - allows feasible non-improving perturbations.

- `_bottleneck_task_relocate()`
  - samples tasks from the current bottleneck launch;
  - reinserts them into feasible positions;
  - targets the part of the solution most likely to determine the active objective.

- `_destroy_repair_perturb()`
  - attempts the existing `destroy_and_repair()`;
  - falls back to random task relocation when no improvement is found;
  - keeps the existing repair behavior but guarantees shaking can still move.

These operators use the existing helpers:

- `all_possible_insertions()`
- `insert_task()`
- `is_feasible_solution()`
- `evaluate()`

#### Adaptive shaking strength

`_shake()` now scales perturbation strength using both:

- neighborhood index `k`;
- stagnation level `no_improve_iters`.

The selected shaking operator rotates by `k`:

- `k ≡ 1 mod 3`: random relocation;
- `k ≡ 2 mod 3`: bottleneck relocation;
- `k ≡ 0 mod 3`: destroy-and-repair perturbation with fallback.

#### Record-to-record acceptance

Added `_accepted_by_record_to_record()`.

Acceptance policy:

- strict improvements over the current incumbent are always accepted;
- otherwise, a candidate is accepted if it stays within an adaptive deviation of the global-best objective:

```text
candidate.objective <= best.objective * (1 + deviation)
```

Accepted non-improving candidates advance to the next neighborhood rather than resetting `k = 1`, avoiding potential cycling on plateau or bounded-worse candidates.

#### Adaptive deviation

Added VNS constructor parameters:

```python
initial_deviation = 0.01
min_deviation = 0.001
max_deviation = 0.05
deviation_decay = 0.90
deviation_growth = 1.10
```

The deviation:

- decreases after a global-best improvement;
- increases after a non-improving outer iteration;
- is reset after a stagnation restart.

#### Stagnation restart

Added `stagnation_patience` with default value:

```python
stagnation_patience = 30
```

When the solver reaches this many consecutive non-improving outer iterations, it restarts from the global best, applies a strong shake, then runs light VND.

### Verification

Syntax compilation was run successfully:

```bash
uv run python -m py_compile mm_mt_dlarp/algorithms/vns.py
```

## 2026-06-20 — Benchmark convergence logging

### Files changed

- `src/run_benchmark.py`
- `docs/implementation-notes.md`

### Motivation

Benchmark output previously reported only the final objective and runtime. For VNS analysis, the run should also expose when improvements were recorded in the solver convergence trace.

### Implemented changes

#### Console convergence log

Added `_print_convergence_log(name, solver)`.

After each solver finishes, `run_one_algorithm()` reads `solver.convergence_log` and prints records in this format:

```text
[vns] convergence:
  iteration=<iteration> time=<elapsed>s objective=<objective>
```

The helper is generic and works with any solver exposing `convergence_log` as tuples shaped like:

```python
(iteration, elapsed, solution_snapshot)
```

If the solver has no convergence log, it prints:

```text
[algorithm] convergence: <empty>
```

#### Result-table convergence iterations

Added `_format_convergence_iterations(solver)`.

This extracts the first element from each convergence-log tuple and stores the compact semicolon-separated trace in the benchmark result row, for example:

```text
0;1;100
```

The benchmark table now has an extra column:

```text
Convergence Iterations
```

This column is emitted consistently for Markdown, CSV, and JSON outputs through `BenchmarkRow.as_csv_row()` and the shared header lists.

### Follow-up considerations

- Run benchmark instances to compare solution quality and runtime before/after the VNS change.
- Tune `initial_deviation`, `max_deviation`, `stagnation_patience`, and `k_max` jointly.
- Consider logging acceptance type, deviation, and stagnation count for later experimental analysis.
- If convergence analysis needs exact objective/time pairs in the output file, add a separate `Convergence Trace` column or sidecar JSON file.

## 2026-06-20 — Benchmark convergence analysis

### Files changed

- `src/run_benchmark.py`
- `docs/implementation-notes.md`

### Motivation

The benchmark output already exposed the raw convergence iteration trace. For comparing VND, VNS, and especially LNS, the table should also summarize how much improvement happened during the run.

### Implemented changes

Added generic convergence analysis in `src/run_benchmark.py`:

- `_valid_convergence_entries(solver)`
  - normalizes solver convergence records shaped as `(iteration, elapsed, solution_snapshot)`.
- `_convergence_analysis(solver)`
  - computes table-friendly metrics from the first and final convergence records.

The benchmark output now includes these additional columns:

```text
Convergence Count
Convergence First Objective
Convergence Final Objective
Convergence Improvement
Convergence Improvement Percent
```

These metrics work for VND, VNS, and LNS because all three solvers expose `convergence_log`. For LNS, iteration indices may include candidate offsets because `LNSSolver` records improvements across multiple starting candidates.

### Follow-up considerations

- If detailed convergence curves are needed, persist `(iteration, elapsed, objective)` as a sidecar JSON file per solver.
- If LNS should distinguish global-best updates from per-candidate local improvements, add an event/type field to its convergence log entries.
