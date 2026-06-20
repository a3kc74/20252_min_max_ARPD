# Tóm tắt objective và các thuật toán VND / VNS / LNS đang triển khai

## 1. Bối cảnh bài toán trong repo

Repo đang triển khai một bộ giải matheuristic cho bài toán **MM-MT-dLARP** trên dữ liệu đường dạng polygonal chain. Theo cấu trúc code hiện tại, bài toán được mô hình hóa qua các lớp chính trong `mm_mt_dlarp/models.py`:

- `Instance`: instance gốc, gồm:
  - tập đỉnh,
  - depot/base,
  - các launch vertices,
  - các line gốc cần phục vụ.
- `OriginalLine`: một đường gốc dạng chuỗi đỉnh, có `segment_costs` và tổng service cost.
- `DiscreteInstance`: phiên bản rời rạc của instance gốc, trong đó mỗi line được chia thành các `RequiredEdge` theo các breakpoint đang chọn.
- `RequiredEdge`: đoạn bắt buộc phải được UAV phục vụ, tương ứng với một đoạn giữa hai breakpoint liên tiếp trên line gốc.
- `Task`: một required edge kèm hướng bay (`forward=True/False`).
- `Flight`: một chuyến bay UAV xuất phát từ một launch vertex, phục vụ một dãy `Task`, rồi quay về launch vertex.
- `Solution`: nghiệm gồm:
  - `selected_launches`,
  - `flights_by_launch`,
  - `objective`,
  - các metric phụ như `paper_makespan`, `ghg_makespan`, `total_ghg`.

Phần solver nền nằm trong `mm_mt_dlarp/algorithms/base.py`, lớp `MatheuristicBase`. Đây là nơi chứa logic chung cho:

- tính khoảng cách hình học,
- tính cost và emission,
- đánh giá objective,
- kiểm tra feasibility,
- khởi tạo nghiệm,
- tối ưu thứ tự task trong một flight,
- các neighborhood/local search,
- VND improvement dùng chung cho nhiều solver.

Các solver cụ thể hiện có:

- `VNDSolver` trong `mm_mt_dlarp/algorithms/vnd.py`
- `VNSSolver` trong `mm_mt_dlarp/algorithms/vns.py`
- `LNSSolver` trong `mm_mt_dlarp/algorithms/lns.py`

---

## 2. Objective hiện tại của bài toán

Objective được cấu hình qua `SolverConfig.objective_type` trong `mm_mt_dlarp/algorithms/base.py`.

Giá trị mặc định hiện tại là:

```python
objective_type: str = "minmax_ghg"
```

Hàm quyết định objective là `_objective_from_metrics(...)`, còn hàm đánh giá nghiệm là `evaluate(...)`.

Hiện tại code hỗ trợ 3 loại objective:

### 2.1. `minmax_ghg` — objective mặc định

Đây là objective mặc định đang được dùng.

Ý nghĩa:

> Tối thiểu hóa lượng phát thải lớn nhất trên một launch point / truck route.

Với mỗi launch vertex `d`, tổng emission của route tại launch đó gồm:

```text
truck_emission(d) + tổng flight_emission(f) với mọi flight f tại d
```

Trong code:

```python
truck_emission(launch) = emission_truck * truck_cost(launch)
truck_cost(launch) = 2 * distance(base_vertex, launch)
```

Và với mỗi UAV flight:

```python
flight_emission(flight) =
    2 * emission_drone_vt
    + emission_drone_cruise * flight_cost(flight)
```

Trong đó:

- `emission_truck`: hệ số phát thải của truck.
- `emission_drone_cruise`: hệ số phát thải khi drone bay cruise.
- `emission_drone_vt`: phát thải cố định cho takeoff/landing.
- `flight_cost(flight)`: tổng range/cost của flight, gồm:
  - từ launch đến task đầu,
  - service cost của từng required edge,
  - khoảng cách chuyển tiếp giữa các task,
  - từ task cuối quay lại launch.

Objective:

```text
minimize max_d GHG(d)
```

Trong code:

```python
if objective_type == "minmax_ghg":
    return max(ghg_by_launch.values()) if ghg_by_launch else float("inf")
```

Metric phụ được lưu:

```python
solution.ghg_makespan = max(ghg_by_launch.values())
solution.total_ghg = sum_d ghg_by_launch[d]
```

---

### 2.2. `paper_makespan`

Objective này tương ứng objective gốc của bài toán MM-MT-dLARP theo paper.

Ý nghĩa:

> Tối thiểu hóa route cost lớn nhất trên các launch point.

Với mỗi launch:

```text
paper_total(d) = truck_cost(d) + tổng flight_cost(f) tại d
```

Trong code:

```python
paper_total = self.truck_cost(launch)
paper_total += self.flight_cost(flight)
```

Objective:

```text
minimize max_d paper_total(d)
```

Trong code:

```python
if objective_type == "paper_makespan":
    return max(paper_by_launch.values()) if paper_by_launch else float("inf")
```

Metric phụ:

```python
solution.paper_makespan = max(paper_by_launch.values())
```

---

### 2.3. `total_ghg`

Objective này tối thiểu hóa tổng phát thải của toàn bộ nghiệm.

Ý nghĩa:

> Tối thiểu hóa tổng GHG emission của tất cả launch routes.

Objective:

```text
minimize sum_d GHG(d)
```

Trong code:

```python
if objective_type == "total_ghg":
    return total_ghg if paper_by_launch else float("inf")
```

---

## 3. Các ràng buộc feasibility chính

Hàm kiểm tra feasibility nằm trong `MatheuristicBase.is_feasible_solution(...)`.

Một nghiệm feasible khi:

### 3.1. Mỗi flight thỏa giới hạn range / emission

Trong `is_feasible_flight(...)`:

```python
2.0 * emission_drone_vt + flight_cost(flight) <= L
```

`L` là `flight_limit`.

Nếu không truyền `flight_limit`, code tự ước lượng bằng `estimate_default_flight_limit(...)`.

### 3.2. Mỗi required edge được phục vụ đúng một lần

Trong `is_feasible_solution(...)`, code gom tất cả `task.edge_id` đã dùng:

- nếu edge lặp lại → infeasible,
- cuối cùng tập used edge phải bằng toàn bộ `instance.required_edges`.

### 3.3. Số launch point được chọn không vượt quá số truck

```python
len(solution.selected_launches) <= config.num_trucks
```

---

## 4. Cách khởi tạo nghiệm

Khởi tạo nghiệm nằm trong `construct_initial_solution(...)` và `generate_initial_pool(...)`.

### 4.1. Chọn launch points

Code chọn tối đa `num_trucks` launch vertices.

Việc chọn launch có bias theo emission truck:

```python
p_d ∝ 1 / (emission_truck * truck_cost(d) + epsilon)
```

Tức launch gần base hơn hoặc có truck emission thấp hơn sẽ có xác suất được chọn cao hơn.

### 4.2. Gán required edges cho launch

Mỗi required edge được gán ngẫu nhiên có trọng số cho một launch đã chọn, với xác suất cao hơn cho launch gần edge hơn:

```python
weight = 1 / edge_distance_to_launch(edge, d)
```

### 4.3. Tạo giant tour

Với mỗi launch, các edge được sắp thành một giant tour bằng heuristic nearest-neighbor:

- từ vị trí hiện tại,
- chọn edge có endpoint gần nhất,
- quyết định hướng bay theo endpoint gần hơn.

### 4.4. Split giant tour thành nhiều flight

`split_giant_tour(...)` chia giant tour thành các flight sao cho mỗi flight không vượt `flight_limit`.

Nếu thêm task mới vẫn feasible thì thêm vào flight hiện tại; nếu không thì mở flight mới.

### 4.5. Initial pool

`generate_initial_pool(...)` lặp lại quá trình khởi tạo để tạo pool nghiệm feasible khác nhau, có giới hạn bởi:

- `max_construction_solutions`,
- `fast_construction_target`,
- `fast_construction_seconds`,
- `max_construction_seconds`,
- `max_construction_attempts`,
- `max_stall_attempts`.

---

## 5. VND đang được triển khai như thế nào

VND được triển khai chính trong:

- `MatheuristicBase.vnd_improvement(...)`
- dùng bởi `VNDSolver`, `LNSSolver`, `VNSSolver`

### 5.1. Danh sách neighborhood mặc định

Nếu không truyền `local_searches`, VND dùng 4 operator theo thứ tự:

```python
(
    "intraroute_move",
    "destroy_and_repair",
    "zero_to_l_exchange",
    "l1_l2_exchange",
)
```

Các operator được map trong `operator_map`:

```python
operator_map = {
    "intraroute_move": self.intraroute_move,
    "destroy_and_repair": self.destroy_and_repair,
    "zero_to_l_exchange": self.zero_to_l_exchange,
    "l1_l2_exchange": self.l1_l2_exchange,
}
```

### 5.2. Vòng lặp VND

VND hoạt động theo logic chuẩn:

1. Bắt đầu từ neighborhood `k = 0`.
2. Thử cải thiện bằng neighborhood hiện tại.
3. Nếu có cải thiện strict objective:
   - cập nhật nghiệm hiện tại,
   - reset `k = 0`.
4. Nếu không cải thiện:
   - chuyển sang neighborhood tiếp theo `k += 1`.
5. Dừng khi đi hết danh sách neighborhood hoặc hết time limit.

Trong code:

```python
while k < len(neighborhoods):
    improved = neighborhoods[k](current)
    if improved is not None and improved.objective + 1e-9 < current.objective:
        current = self.evaluate(improved)
        k = 0
    else:
        k += 1
```

### 5.3. `intraroute_move`

Mục tiêu:

> Cải thiện thứ tự task bên trong từng flight.

Cách làm:

- Với mỗi flight:
  - lấy từng task ra,
  - thử chèn lại vào mọi vị trí,
  - thử cả hai hướng `forward=True/False`,
  - chọn move làm giảm `flight_cost`,
  - chỉ nhận nếu flight vẫn feasible.

Operator này chỉ return solution nếu objective toàn cục giảm.

### 5.4. `destroy_and_repair`

Mục tiêu:

> Giảm tải launch point bottleneck bằng cách remove một số task rồi reinsert greedily.

Cách làm:

1. Tìm bottleneck launch:

```python
bottleneck = max(solution.makespan_by_launch)
```

Lưu ý: `makespan_by_launch` là paper cost nếu objective type là `paper_makespan`, còn nếu objective type khác thì là GHG by launch.

2. Random remove từ 1 đến `nmax_destroy` task ở bottleneck.

3. Với từng task bị remove:
   - thử mọi vị trí chèn khả thi trong mọi flight / launch,
   - thử cả hai hướng,
   - chọn insertion có objective tốt nhất.

4. Lặp tối đa `itmax_destroy` lần không cải thiện.

### 5.5. `zero_to_l_exchange`

Mục tiêu:

> Di chuyển một chuỗi `l` task từ bottleneck launch sang launch/flight khác mà không swap ngược lại.

Đặc điểm:

- `l` tăng từ 1 đến `lmax_exchange`.
- Nếu tìm được improvement:
  - nhận move,
  - reset `l = 1`.
- Nếu không:
  - tăng `l`.

Tên `zero_to_l_exchange` thể hiện dạng exchange 0-l: bên nhận không trả lại task nào.

### 5.6. `l1_l2_exchange`

Mục tiêu:

> Swap một chuỗi `l1` task từ bottleneck flight với một chuỗi `l2` task từ flight khác.

Đặc điểm:

- Duyệt `l1`, `l2` đến `lmax_exchange`.
- Chỉ nhận nếu nghiệm feasible và objective giảm.
- Nếu cải thiện:
  - reset `l1 = l2 = 1`.

### 5.7. Tối ưu bottleneck flight sau VND

Sau khi VND đạt local optimum, nếu `optimize_bottleneck=True`, code gọi:

```python
optimize_bottleneck_flights(current)
```

Operator này:

- tìm bottleneck launch,
- tối ưu từng flight ở launch đó bằng `exact_optimize_flight(...)`,
- nếu cải thiện objective thì gọi lại VND từ nghiệm mới.

`exact_optimize_flight(...)` có thể dùng:

- `dp_optimize_flight`,
- `branch_cut_optimize_flight`,
- hoặc `auto`.

Mặc định trong `SolverConfig`:

```python
flight_optimizer: str = "bc"
```

Tức tối ưu flight bằng Branch-and-Cut nếu có thể.

---

## 6. VNDSolver đang được triển khai như thế nào

`VNDSolver` nằm trong `mm_mt_dlarp/algorithms/vnd.py`.

Đây là solver đầy đủ dùng pipeline:

```text
generate initial pool
→ apply VND to each solution
→ select top candidates
→ splitting phase coarse-to-fine
→ return best solution
```

### 6.1. `solve()`

Luồng chính:

1. Set deadline nếu có `time_limit_seconds`.
2. Tạo initial pool bằng `generate_initial_pool()`.
3. Chạy `vnd_improvement(...)` cho từng nghiệm trong pool.
4. Sort pool theo objective.
5. Lấy top `split_top_k` nghiệm tốt nhất.
6. Với từng nghiệm top:
   - reset instance về discretization ban đầu,
   - chạy `splitting_phase(...)`,
   - cập nhật best nếu split candidate tốt hơn.
7. Evaluate và return final solution.

### 6.2. Splitting phase coarse-to-fine

`VNDSolver.splitting_phase(...)` triển khai cơ chế refine rời rạc hóa:

#### Bước 1

- Thêm midpoint vào mọi required edge:

```python
expanded_breakpoints = add_midpoints_to_all_intervals(...)
```

- Build refined instance.
- Convert solution từ instance cũ sang refined instance.
- Chạy VND trên refined solution với:

```python
optimize_bottleneck=False
```

- Nếu objective không cải thiện thì dừng.

#### Bước 2+

Lặp:

1. Detect midpoints thật sự được dùng:

```python
used = detect_used_midpoints(current_solution, current_instance)
```

2. Nếu không có midpoint nào được dùng thì dừng.

3. Refine quanh các midpoint được dùng:

```python
next_breakpoints = refine_breakpoints_from_used_midpoints(...)
```

4. Build instance mới.
5. Convert solution sang instance mới.
6. Chạy VND.
7. Nếu không cải thiện objective thì dừng, ngược lại cập nhật best và tiếp tục.

Ý nghĩa:

> Bắt đầu từ discretization thô, sau đó chỉ refine thêm ở những vùng có dấu hiệu hữu ích, tránh làm bài toán rời rạc quá lớn ngay từ đầu.

---

## 7. VNS đang được triển khai như thế nào

`VNSSolver` nằm trong `mm_mt_dlarp/algorithms/vns.py`.

VNS ở đây mở rộng `MatheuristicBase`, dùng **shaking + VND nhẹ** để thoát local optimum.

### 7.1. Tham số chính

Constructor:

```python
VNSSolver(instance, config, k_max=4, max_iter=100)
```

- `k_max`: neighborhood shaking lớn nhất.
- `max_iter`: số vòng lặp outer VNS.

### 7.2. Shaking

Hàm `_shake(solution, k)`:

```python
def _shake(self, solution, k):
    current = solution.clone()
    for _ in range(k):
        result = self.destroy_and_repair(current)
        if result is not None:
            current = result
    return current
```

Tức với mức shaking `k`, solver áp dụng `k` lần `destroy_and_repair`.

Lưu ý quan trọng:

- `destroy_and_repair(...)` trong base chỉ trả nghiệm nếu có cải thiện objective.
- Vì vậy shaking hiện tại không phải perturbation hoàn toàn tự do; nó vẫn thiên về improvement theo operator `destroy_and_repair`.

### 7.3. VND nhẹ trong vòng VNS

Trong phần khởi tạo và vòng lặp chính, VNS dùng `light_vnd`:

```python
light_vnd = ("intraroute_move", "zero_to_l_exchange")
```

Tức chỉ dùng 2 neighborhood:

- `intraroute_move`,
- `zero_to_l_exchange`.

Ngoài ra:

```python
optimize_bottleneck=False
```

để giảm chi phí tính toán trong vòng lặp VNS.

### 7.4. Luồng `solve()`

Nếu chưa có initial solution:

1. Tạo pool.
2. Với từng solution trong pool:
   - chạy light VND.
3. Chọn nghiệm tốt nhất làm `current`.

Nếu có initial solution:

- chạy light VND từ initial.

Sau đó:

```text
best = current
for iteration in max_iter:
    k = 1
    while k <= k_max:
        shaken = shake(current, k)
        improved = light_vnd(shaken)
        if improved better than current:
            current = improved
            update best nếu cần
            k = 1
        else:
            k += 1
```

Sau vòng lặp, solver chạy full VND một lần cuối:

```python
final = self.vnd_improvement(best)
```

Nếu final tốt hơn best thì cập nhật.

### 7.5. Khác biệt VNS so với VNDSolver

VNS hiện tại:

- không có splitting phase coarse-to-fine trong `solve()`;
- dùng shaking để thoát local optimum;
- trong search chính dùng VND nhẹ;
- chỉ chạy full VND ở cuối;
- return `best, self.instance`.

---

## 8. LNS đang được triển khai như thế nào

`LNSSolver` nằm trong `mm_mt_dlarp/algorithms/lns.py`.

LNS ở đây dùng large destroy + greedy repair, sau đó dùng local search để cải thiện nghiệm repaired.

### 8.1. Tham số chính

Constructor:

```python
LNSSolver(
    instance,
    config,
    destroy_frac=0.3,
    max_iter=200,
    accept=accept_improving,
)
```

- `destroy_frac`: tỷ lệ task bị remove ở mỗi bước destroy.
- `max_iter`: số vòng lặp LNS.
- `accept`: tiêu chí nhận nghiệm mới.

### 8.2. Acceptance criteria

Code có 2 acceptance function:

#### `accept_improving`

Chỉ nhận nghiệm cải thiện strict:

```python
candidate.objective + 1e-9 < current.objective
```

Đây là mặc định.

#### `accept_simulated_annealing`

Nhận nghiệm tốt hơn luôn, và có thể nhận nghiệm xấu hơn với xác suất:

```text
exp(-delta / temperature)
```

Tuy nhiên constructor mặc định vẫn là `accept_improving`.

### 8.3. Destroy operator

Hàm `_destroy(solution)`:

1. Clone solution.
2. Gom tất cả task trong mọi flight / launch.
3. Tính số task remove:

```python
n_remove = max(1, int(len(all_tasks) * destroy_frac))
```

4. Random sample `n_remove` task.
5. Remove task khỏi flight.
6. Evaluate candidate.
7. Return:

```python
candidate_without_tasks, removed_tasks
```

Đây là destroy lớn hơn so với `destroy_and_repair` của VND vì nó remove theo tỷ lệ toàn cục `destroy_frac`, không chỉ remove 1 đến `nmax_destroy` task ở bottleneck.

### 8.4. Repair operator

Hàm `_repair(solution, removed)`:

Với từng task bị remove:

1. Gọi `all_possible_insertions(...)` để sinh mọi cách chèn:
   - vào mọi launch,
   - mọi flight,
   - mọi vị trí,
   - cả hai hướng.
2. Gọi `insert_task(...)` để kiểm tra feasibility.
3. Chọn insertion có objective tốt nhất.
4. Nếu task nào không chèn được thì return `None`.

Đây là greedy best-insertion repair.

### 8.5. Local search trong mỗi iteration LNS

Trong `_run_lns_from(...)`, sau destroy và repair, code chạy VND nhẹ:

```python
iteration_searches = (
    "intraroute_move",
    # "zero_to_l_exchange",
)
```

Hiện tại chỉ bật:

- `intraroute_move`

`zero_to_l_exchange` đang bị comment.

Sau đó:

```python
repaired = self.vnd_improvement(
    repaired,
    local_searches=iteration_searches,
    optimize_bottleneck=False,
)
```

### 8.6. Vòng lặp LNS từ một candidate

Trong `_run_lns_from(...)`:

```text
current = start_solution
best = current

for iteration in 1..max_iter:
    destroyed, removed = destroy(current)
    repaired = repair(destroyed, removed)
    if repaired is None:
        continue
    repaired = light/local VND(repaired)
    if accept(repaired, current):
        current = repaired
        if current better than best:
            best = current
```

Với acceptance mặc định, LNS là dạng greedy hill-climbing trên large neighborhoods.

### 8.7. Luồng `solve()`

Nếu không có initial solution:

1. Tạo initial pool.
2. Chạy full VND cho từng nghiệm trong pool, nhưng:

```python
optimize_bottleneck=False
```

3. Sort pool theo objective.
4. Lấy top `split_top_k`.

Nếu có initial solution:

- chạy VND từ initial và đưa vào pool.

Sau đó:

1. Chạy `_run_lns_from(...)` cho từng nghiệm top.
2. Với mỗi candidate sau LNS:
   - chạy full VND một lần:

```python
final = self.vnd_improvement(candidate_best)
```

3. Gom các LNS candidates.
4. Lấy top candidates sau LNS.
5. Chạy splitting phase cho các candidate top.
6. Return best final solution.

### 8.8. Splitting phase trong LNS

`LNSSolver.splitting_phase(...)` gần giống `VNDSolver.splitting_phase(...)`:

- thêm midpoint cho mọi edge,
- build refined instance,
- convert solution,
- chạy VND improvement,
- nếu cải thiện thì tiếp tục refine theo used midpoints.

Điểm khác:

- refined solver được tạo là `LNSSolver`,
- nhưng trong splitting phase hiện tại vẫn gọi `vnd_improvement(...)`, không chạy full LNS loop trên refined instance.

---

## 9. So sánh ngắn gọn VND / VNS / LNS hiện tại

| Thuật toán | File | Ý tưởng chính | Local search dùng trong loop | Splitting phase |
|---|---|---|---|---|
| VND | `algorithms/vnd.py` + `base.py` | Duyệt tuần tự nhiều neighborhood, reset về neighborhood đầu khi cải thiện | Full VND mặc định: intraroute, destroy-repair, zero-to-l, l1-l2 | Có |
| VNS | `algorithms/vns.py` | Shaking bằng repeated destroy-repair rồi chạy VND nhẹ | `intraroute_move`, `zero_to_l_exchange` trong loop; full VND ở cuối | Không |
| LNS | `algorithms/lns.py` | Remove một tỷ lệ task toàn cục, repair bằng best insertion, accept theo criterion | Trong LNS iteration hiện chỉ bật `intraroute_move`; full VND sau mỗi top candidate | Có |

---

## 10. Các điểm triển khai đáng chú ý

### 10.1. Objective đã chuyển trọng tâm sang GHG

Mặc dù paper objective `paper_makespan` vẫn được hỗ trợ, cấu hình mặc định hiện tại là:

```python
objective_type = "minmax_ghg"
```

Do đó bottleneck và improvement chủ yếu đang tối ưu theo GHG makespan.

### 10.2. Có cả metric phụ để so sánh

Mỗi solution lưu đồng thời:

- `objective`,
- `paper_makespan`,
- `ghg_makespan`,
- `total_ghg`.

Điều này cho phép chạy một objective nhưng vẫn report các metric còn lại.

### 10.3. Branch-and-Cut / DP được dùng để tối ưu flight cục bộ

Trong base solver có hai cách tối ưu thứ tự task trong một flight:

- DP cho flight nhỏ,
- Branch-and-Cut qua module `grp`.

Mặc định `flight_optimizer = "bc"`.

VND có thể gọi bước này sau khi local search đạt local optimum, thông qua `optimize_bottleneck_flights(...)`.

### 10.4. Discretization refinement là một phần quan trọng của VND và LNS

Cả `VNDSolver` và `LNSSolver` đều không chỉ tìm kiếm trên một discrete instance cố định, mà còn refine breakpoint theo midpoint để cải thiện nghiệm.

Quy trình này giúp cân bằng giữa:

- bài toán ban đầu nhỏ, dễ search,
- và khả năng tạo nghiệm tốt hơn khi cần thêm breakpoint.

### 10.5. VNS hiện chưa tích hợp splitting phase

Khác với VND và LNS, `VNSSolver.solve()` hiện không gọi coarse-to-fine splitting phase.

Nếu muốn so sánh công bằng với VND/LNS trên refined discretization, có thể cần bổ sung splitting phase cho VNS hoặc chạy VNS trong một pipeline bên ngoài.

---

## 11. Tóm tắt pipeline tổng thể

Một pipeline điển hình với `VNDSolver` hoặc `LNSSolver` hiện tại:

```text
Raw Instance
→ Initial breakpoints
→ DiscreteInstance
→ Generate initial solution pool
→ Improve pool bằng VND
→ Chọn top candidates
→ Nếu LNS: chạy large destroy-repair search trên top candidates
→ Chạy full VND lại
→ Splitting phase coarse-to-fine
→ Evaluate final solution theo objective_type
→ Return Solution + DiscreteInstance tốt nhất
```

Với `VNSSolver`:

```text
DiscreteInstance
→ Generate initial solution pool
→ Light VND
→ VNS loop:
    shake bằng repeated destroy-and-repair
    light VND
    update current/best
→ Full VND cuối
→ Return best Solution
```

---

## 12. Kết luận

Code hiện tại đang triển khai một hệ matheuristic cho MM-MT-dLARP với trọng tâm objective mặc định là **minimize maximum GHG emission per launch route** (`minmax_ghg`). Bên cạnh đó vẫn giữ objective gốc của paper (`paper_makespan`) và tổng phát thải (`total_ghg`).

Ba hướng search chính được triển khai như sau:

- **VND** là lõi local search chung, dùng nhiều neighborhood theo thứ tự và reset khi có cải thiện.
- **VNS** thêm shaking để thoát local optimum, dùng VND nhẹ trong vòng lặp và full VND ở cuối.
- **LNS** remove một phần lớn task, repair bằng greedy best insertion, accept theo criterion, rồi refine bằng VND và splitting phase.

Hai solver `VNDSolver` và `LNSSolver` có thêm cơ chế **coarse-to-fine splitting phase**, giúp refine rời rạc hóa theo midpoint để cải thiện nghiệm mà không làm instance quá lớn ngay từ đầu.