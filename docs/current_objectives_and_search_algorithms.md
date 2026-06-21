# Objective và các thuật toán VND / VNS / LNS hiện tại

Repo hiện chỉ dùng một objective duy nhất: `paper_makespan`.

## Objective

`paper_makespan` đo chi phí lớn nhất theo launch route:

```text
route_cost(d) = truck_cost(d) + sum flight_cost(f) for every flight f at launch d
paper_makespan = max_d route_cost(d)
```

`MatheuristicBase.evaluate()` cập nhật:

- `solution.makespan_by_launch`
- `solution.paper_makespan`
- `solution.objective`

Trong đó `solution.objective == solution.paper_makespan`.

## Search algorithms

- `VNDSolver`: xây pool nghiệm ban đầu, chạy VND, sau đó splitting phase.
- `VNSSolver`: shaking, adaptive VND, acceptance theo record-to-record, restart/elite memory, final VND và splitting.
- `LNSSolver`: destroy-repair theo nhiều candidate, VND refinement và splitting.
- `LCBIMMASolver`: island/metaheuristic baseline dùng cùng objective cố định.

## Benchmark

`src/run_benchmark.py` so sánh các thuật toán bằng cùng objective `paper_makespan`; không còn tham số CLI để chọn objective.
