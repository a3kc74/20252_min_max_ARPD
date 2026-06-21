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

## 2026-06-21 — VNS Phase 4 adaptive VND and splitting

### Files changed

- `mm_mt_dlarp/algorithms/vns.py`
- `docs/vns_context.md`
- `docs/implementation-notes.md`

### Motivation

Phase 4 from `docs/vns_improve.md` recommends giving VNS a fairer refinement capability compared with the existing VND/LNS pipelines:

- adaptive ordering for local search;
- splitting phase for breakpoint refinement;
- full VND at the end of restarts or best updates.

The previous VNS already had adaptive shaking, elite restarts, and record-to-record acceptance, but local-search ordering was fixed and VNS did not run a splitting phase after the search.

### Implemented changes

#### Adaptive VND ordering

Added VND-local-search statistics:

```python
self.vnd_operator_stats = {
    name: {"uses": 0, "improvements": 0, "reward": 0.0}
}
```

Added:

- `_ordered_local_searches()`
  - orders local-search operators by average observed reward;
  - preserves base-order tie-breaking;
  - supports neighborhood-index rotation for exploration.
- `_adaptive_vnd()`
  - tries operators one at a time in the adaptive order;
  - updates per-operator usage, improvement count, and normalized reward;
  - resets the neighborhood scan after an improving move;
  - can call full VND with bottleneck optimization for final polishing.

Direct light-VND calls in the VNS flow now use `_adaptive_vnd()`.

#### Splitting phase inside VNS

Added `VNSSolver.splitting_phase()`.

The implementation mirrors the existing coarse-to-fine refinement pattern from `VNDSolver`:

1. add midpoint to every interval;
2. rebuild a refined discrete instance;
3. convert the incumbent solution to the refined instance;
4. improve with adaptive light VND;
5. detect used midpoints;
6. refine around used midpoints;
7. repeat while objective improves.

The method returns both the refined solution and the refined instance.

#### Final VND and final splitting

At the end of `solve()`:

1. VNS runs adaptive full VND with bottleneck optimization.
2. VNS runs `splitting_phase(best)`.
3. If splitting improves the objective:
   - `best` is updated;
   - `self.instance` is switched to the refined instance;
   - the distance cache is cleared;
   - the solution is added to elite memory;
   - convergence metadata records phase `"final_splitting"`.

#### VND adaptation metadata

Final convergence metadata now includes:

```python
"operator_stats": self.operator_stats,
"vnd_operator_stats": self.vnd_operator_stats,
```

This keeps shaking adaptation and local-search adaptation inspectable separately.

### Verification

Syntax compilation should be run after this change:

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

## 2026-06-21 — VNS Phase 3 operator adaptation

### Files changed

- `mm_mt_dlarp/algorithms/vns.py`
- `docs/vns_context.md`
- `docs/implementation-notes.md`

### Motivation

Phase 3 from `docs/vns_improve.md` recommends making VNS shaking adaptive. The previous implementation already had multiple shaking behaviors, but operator choice was mostly tied to neighborhood index rotation. That means the solver could continue spending time on weak operators even after enough evidence showed that another operator was more effective for a specific instance.

This change adds an adaptive operator-selection layer while preserving the existing Phase 1 perturbation operators, Phase 2 elite memory, and record-to-record acceptance logic.

### Implemented changes

#### Adaptive shaking operator set

`VNSSolver` now defines a named shaking operator set:

```python
self.shake_operators = (
    "random_task_relocate",
    "bottleneck_task_relocate",
    "destroy_repair_perturb",
)
```

Each operator is dispatched through `_apply_shake_operator()`.

#### UCB-style operator selection

Added:

- `_operator_score()`
- `_select_shake_operator()`
- `_adaptive_shake()`

Selection behavior:

1. warm up by rotating through all operators at least once;
2. then choose the operator with highest UCB-style score:

```text
average_reward + sqrt(2 * log(total_uses) / operator_uses)
```

This lets the solver exploit strong operators while still exploring less-used ones.

#### Operator statistics

`VNSSolver` now tracks per-operator counters in `self.operator_stats`:

```python
{
    "uses": 0,
    "accepted": 0,
    "best_improvements": 0,
    "current_improvements": 0,
    "feasible": 0,
    "reward": 0.0,
}
```

The statistics are reset at the beginning of every `solve()` call.

#### Reward and adaptive log

Added `_reward_operator()` and `self.adaptive_log`.

Reward combines several useful search signals:

- feasible candidate: `+0.05`;
- accepted move: `+0.25`;
- current-incumbent improvement: `+1.00`;
- objective-neutral candidates no longer receive a secondary metric reward;
- global-best improvement: `+3.00`.

`adaptive_log` records one event per adaptive shaking attempt with operator name, strength, reward, acceptance status, improvement status, deviation, stagnation level, and objective values before/after the attempt.

#### Convergence metadata

Global-best improvement records now include:

```python
{
    "operator": operator_name,
    "operator_reward": reward,
}
```

Restart and final convergence records include `operator_stats`, making operator behavior inspectable from solver logs.

### Verification

Syntax compilation was run successfully:

```bash
uv run python -m py_compile mm_mt_dlarp/algorithms/vns.py
```

## 2026-06-21 — Paper makespan as the only objective

### Files changed

- `mm_mt_dlarp/models.py`
- `mm_mt_dlarp/algorithms/base.py`
- `mm_mt_dlarp/algorithms/vns.py`
- `mm_mt_dlarp/algorithms/lcb_imma.py`
- `src/run_benchmark.py`
- `exp/vns_timing_experiment.py`
- `docs/algorithm_objectives_summary.md`
- `docs/current_objectives_and_search_algorithms.md`
- `docs/vns_context.md`
- generated benchmark/timing documents under `docs/`
- `tests/test_paper_makespan_only.py`

### Motivation

The repository now uses a single objective everywhere: `paper_makespan`. The previous selectable objective path and retired GHG-oriented output metrics were removed to avoid ambiguous comparisons between VND, VNS, and LNS.

### Implemented changes

- Removed the configurable objective selector from `SolverConfig`.
- Removed secondary objective fields from `Solution`; solutions now store `objective`, `paper_makespan`, per-launch route costs, and flight costs.
- Simplified `MatheuristicBase.evaluate()` so `solution.objective` is always equal to `solution.paper_makespan`.
- Restored the flight feasibility check to the paper range condition: `flight_cost(flight) <= L`.
- Converted legacy emission-named helper behavior used by LCB-IMMA into cost-based naming and routing.
- Removed benchmark CLI/output columns for retired objective selection and retired GHG metrics.
- Updated docs and generated comparison tables so they describe/report only the paper objective.

### Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests/test_paper_makespan_only.py`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m py_compile mm_mt_dlarp/models.py mm_mt_dlarp/algorithms/base.py mm_mt_dlarp/algorithms/vns.py mm_mt_dlarp/algorithms/lcb_imma.py src/run_benchmark.py exp/vns_timing_experiment.py`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python src/run_benchmark.py data/instances/MTLARP4_4_3_1.dat --num-trucks 2 --flight-limit 847 --algorithms vns,lns,vnd --output docs/compare_vns_lns_vnd_MTLARP4_4_3_1_P2_L847.md --quiet`

## 2026-06-21 — Grid search experiment for VNS/LNS/VND

### Files changed

- `exp/gridsearch_algorithms.py`
- `tests/test_gridsearch_algorithms.py`
- `docs/implementation-notes.md`

### Motivation

Added a reusable experiment runner to search parameter combinations for the three active heuristics (`vns`, `lns`, `vnd`) under the single `paper_makespan` objective. The runner is designed to be launched from CLI and to save full trial rows plus a compact best-parameter summary.

### Implemented changes

- Added `exp/gridsearch_algorithms.py`, which reuses `src.run_benchmark.run_one_algorithm()` so trial results share the same objective, timing, and convergence fields as regular benchmark runs.
- Added algorithm-specific grids:
  - VNS: `vns_k_max` × `vns_max_iter`, with default `vns_max_iter=50`.
  - LNS: `lns_destroy_frac` × `lns_max_iter`.
  - VND: `split_top_k` applied through `SolverConfig`.
- Added seed expansion so each parameter set can be evaluated across one or more seeds.
- Ranked best rows per algorithm by `(Objective, Time)`, minimizing objective first and runtime second.
- Added CSV, JSON, and Markdown summary outputs.

### Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests/test_gridsearch_algorithms.py`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall -q exp/gridsearch_algorithms.py tests/test_gridsearch_algorithms.py`

## 2026-06-21 — Multi-seed benchmark mean/std experiment

### Files changed

- `exp/multiseed_benchmark.py`
- `tests/test_multiseed_benchmark.py`
- `docs/implementation-notes.md`

### Motivation

Added an experiment runner for comparing `vns`, `lns`, and `vnd` over a fixed list of dataset configurations and three seeds. The runner reports raw per-seed benchmark rows and mean/std summaries by dataset configuration and algorithm.

### Implemented changes

- Added the 20 requested dataset configurations as the default experiment set.
- Added default seeds `0,1,2`.
- Fixed the requested algorithm parameters as defaults:
  - `lns_destroy_frac=0.4`
  - `lns_max_iter=200`
  - `vns_k_max=2`
  - `vns_max_iter=100`
- Reused `src.run_benchmark.run_one_algorithm()` for each trial so raw result fields match regular benchmark output.
- Added aggregation by `(Instance, Trucks Submitted, Flight Limit, Algorithm)` with mean and sample standard deviation for objective, paper makespan, and runtime.
- Added CSV, JSON, and Markdown outputs for both raw rows and summary rows.

### Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests/test_multiseed_benchmark.py`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall -q exp/multiseed_benchmark.py tests/test_multiseed_benchmark.py`
