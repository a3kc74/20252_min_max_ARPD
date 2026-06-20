# VNS Context and Phase 1 Implementation Notes

## Scope

This note documents the current `VNSSolver` implementation in `mm_mt_dlarp/algorithms/vns.py` after Phase 1 improvements. The solver targets the discrete MM-MT-dLARP representation used by the repository:

- a solution contains selected launch vertices;
- each launch has one or more UAV flights;
- each flight is an ordered list of required-edge service tasks;
- objective evaluation is delegated to `MatheuristicBase.evaluate()`, which sets the configured objective and supporting makespan/greenhouse-gas metrics.

## Objective handled by VNS

`VNSSolver` does not hard-code a separate objective. It optimizes the objective selected through `SolverConfig.objective_type` and computed by `MatheuristicBase.evaluate()`.

The current model tracks:

- `objective`: active scalar objective used by all search comparisons;
- `paper_makespan`: classical launch-route makespan style measure;
- `ghg_makespan`: maximum launch-side greenhouse-gas workload;
- `total_ghg`: total greenhouse-gas workload;
- `makespan_by_launch`: per-launch workload values used to identify bottlenecks.

VNS compares solutions by `solution.objective`, so it remains compatible with the objective mode selected by the experiment or solver configuration.

## High-level VNS flow

The solver is implemented as a VNS/VND hybrid:

1. Generate or receive an initial solution.
2. Apply a light VND pass with:
   - `intraroute_move`
   - `zero_to_l_exchange`
3. Repeat until `max_iter` or deadline:
   - shake the current incumbent with neighborhood index `k`;
   - improve the shaken solution using light VND;
   - accept strict global-best improvements immediately;
   - otherwise accept controlled record-to-record candidates when they stay close to the global best;
   - adapt deviation and stagnation counters;
   - restart around the global best after prolonged stagnation.
4. Apply full VND once at the end as a polishing step.
5. Return the best solution and the current discrete instance.

## Phase 1 changes implemented

### 1. Real perturbation shaking

The earlier VNS behavior depended heavily on improvement-style operators. In particular, `destroy_and_repair()` from `MatheuristicBase` returns `None` when it cannot find an improving solution, which is correct for local search but weak for shaking because VNS needs valid movement even when the move worsens the incumbent.

The new shaking layer adds explicit perturbation methods:

- `_random_task_relocate(solution, moves)`
  - randomly selects task locations across all launches/flights;
  - removes one task;
  - reinserts it into a random feasible insertion position;
  - repeats for the requested number of moves.

- `_bottleneck_task_relocate(solution, moves)`
  - samples tasks from the current bottleneck launch;
  - relocates them using the same feasible random reinsertion logic;
  - focuses diversification on the launch that currently determines the objective.

- `_destroy_repair_perturb(solution, moves)`
  - first attempts the existing `destroy_and_repair()`;
  - if it cannot improve, falls back to random task relocation;
  - this keeps the existing repair behavior while guaranteeing that shaking can still move on local optima and plateaus.

All reinsertion uses existing feasibility-aware helpers:

- `all_possible_insertions()`
- `insert_task()`
- `is_feasible_solution()`
- `evaluate()`

This avoids duplicating feasibility logic.

### 2. Adaptive shaking strength

`_shake(solution, k, no_improve_iters)` now scales perturbation strength by:

- neighborhood index `k`;
- stagnation level `no_improve_iters`.

The strength is:

```text
strength = max(1, k * stagnation_boost)
```

where `stagnation_boost` grows as the number of non-improving iterations increases.

The operator rotates by `k`:

- `k ≡ 1 mod 3`: random task relocation;
- `k ≡ 2 mod 3`: bottleneck task relocation;
- `k ≡ 0 mod 3`: destroy-and-repair perturbation with fallback.

This makes neighborhoods structurally different instead of repeatedly calling the same improving-only move.

### 3. Record-to-record acceptance

The solver now supports controlled non-improving acceptance through `_accepted_by_record_to_record()`.

Rules:

- always accept strict improvements over the current incumbent;
- otherwise accept a candidate if:

```text
candidate.objective <= best.objective * (1 + deviation)
```

This allows diversification while keeping the search anchored to the best known record.

Accepted non-improving candidates advance to the next neighborhood (`k += 1`) instead of resetting to `k = 1`. This prevents cycling forever on accepted plateau/worse candidates.

### 4. Adaptive deviation

The record-to-record deviation is adaptive:

- starts from `initial_deviation`;
- is bounded by `min_deviation` and `max_deviation`;
- decreases after global-best improvements using `deviation_decay`;
- increases after non-improving outer iterations using `deviation_growth`;
- resets after stagnation restarts.

Default parameters:

```python
initial_deviation = 0.01
min_deviation = 0.001
max_deviation = 0.05
deviation_decay = 0.90
deviation_growth = 1.10
```

### 5. Stagnation counter and restart

The solver tracks consecutive outer iterations without global-best improvement.

When `no_improve_iters >= stagnation_patience`, the solver:

1. restarts from the global best;
2. applies a strong shake;
3. runs light VND;
4. keeps the result only if it improves the global best, otherwise continues from the restarted current state;
5. resets stagnation and deviation.

Default:

```python
stagnation_patience = 30
```

## Key implementation entry points

Main methods added or changed in `VNSSolver`:

- `_task_locations()`
- `_remove_task_at()`
- `_random_feasible_reinsert()`
- `_random_task_relocate()`
- `_bottleneck_task_relocate()`
- `_destroy_repair_perturb()`
- `_shake()`
- `_accepted_by_record_to_record()`
- `_restart_from_best()`
- `solve()`

## Expected behavior after Phase 1

The VNS is now more robust in local optima because it can:

- perturb even when local search operators find no improvement;
- accept bounded non-improving moves;
- increase search radius during stagnation;
- restart around the best known solution with a stronger shake;
- still preserve feasibility through existing insertion/evaluation helpers;
- still finish with the full VND polish already available in `MatheuristicBase`.

## Verification performed

Syntax compilation was run with:

```bash
uv run python -m py_compile mm_mt_dlarp/algorithms/vns.py
```

The command executed successfully.