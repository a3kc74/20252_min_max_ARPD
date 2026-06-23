# Tổng hợp objective và triển khai VND / LNS / VNS

Repo hiện thống nhất một objective duy nhất: `paper_makespan`.

## Objective duy nhất

`paper_makespan` là objective gốc của MM-MT-dLARP:

```text
minimize max_d [ truck_cost(d) + sum_k flight_cost(d, k) ]
```

Trong code:

- `Solution.objective` luôn bằng `Solution.paper_makespan`.
- `Solution.makespan_by_launch` luôn lưu route cost theo từng launch.
- `MatheuristicBase.evaluate()` không còn chọn objective bằng cấu hình.
- `SolverConfig` không còn trường chọn objective.

## Tác động lên thuật toán

Các solver VND, VNS, LNS và LCB-IMMA đều so sánh nghiệm bằng cùng một giá trị `Solution.objective`.
Vì objective đã cố định, benchmark không còn cột loại objective hay các metric objective phụ.

## Benchmark output

`src/run_benchmark.py` xuất các cột chính:

- Algorithm
- Instance
- Base
- Flight Limit
- Trucks Used
- Trucks Submitted
- Objective
- Paper Makespan
- Time
- Flight Optimizer
- Convergence metrics
