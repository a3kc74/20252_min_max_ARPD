# VNS Context and Paper-Makespan Objective Notes

`VNSSolver` hiện tối ưu objective cố định `paper_makespan`, được tính trong `MatheuristicBase.evaluate()`.

## Objective contract

- `Solution.objective` luôn bằng `Solution.paper_makespan`.
- `Solution.makespan_by_launch` luôn là route cost theo từng launch.
- VNS, LNS, VND dùng cùng tiêu chí so sánh nghiệm.

## VNS components

Các cải tiến VNS hiện có:

- real perturbation shaking;
- adaptive shaking strength;
- record-to-record acceptance quanh best record;
- stagnation restart và elite memory;
- adaptive VND ordering;
- final VND và splitting phase.

## Tie-breaking

Các tie-breaker trong VNS dùng `paper_makespan`/`objective`; không còn metric objective phụ.
