# Survey Reactive / Adaptive VNS hiện đại và đề xuất cải thiện cho `VNSSolver`

## 1. Mục tiêu tài liệu

Tài liệu này survey các hướng **Reactive Variable Neighborhood Search (Reactive VNS)** và **Adaptive Variable Neighborhood Search (Adaptive VNS)** hiện đại, sau đó đề xuất các cải thiện có thể áp dụng trực tiếp lên khung `VNSSolver` hiện tại trong repo.

File VNS hiện tại:

```text
mm_mt_dlarp/algorithms/vns.py
```

VNS hiện tại đang có dạng:

```text
initial pool
→ light VND
→ while iteration < max_iter:
    k = 1
    while k <= k_max:
        shaken = _shake(current, k)
        improved = light_vnd(shaken)
        if improved better than current:
            current = improved
            update best
            k = 1
        else:
            k += 1
→ full VND cuối
```

Trong đó:

- `_shake(solution, k)` áp dụng `k` lần `destroy_and_repair`.
- Local search trong vòng lặp là `light_vnd = ("intraroute_move", "zero_to_l_exchange")`.
- Acceptance hiện tại là strict improvement so với `current`.
- `k` tăng tuyến tính nếu không cải thiện, reset về 1 nếu cải thiện.
- Chưa có cơ chế học / thích nghi dựa trên lịch sử hiệu quả của neighborhood.
- Chưa có splitting phase giống `VNDSolver` và `LNSSolver`.
- Chưa có diversification restart rõ ràng khi search bị stagnation.

---

## 2. Tổng quan VNS cổ điển

VNS cổ điển dựa trên ba ý tưởng:

1. Một local optimum với một neighborhood chưa chắc là local optimum với neighborhood khác.
2. Global optimum là local optimum theo mọi neighborhood.
3. Thay đổi neighborhood có hệ thống giúp thoát local optimum.

Khung VNS cơ bản:

```text
x = initial_solution
while not stopping_condition:
    k = 1
    while k <= k_max:
        x'  = shake(x, N_k)
        x'' = local_search(x')
        if f(x'') < f(x):
            x = x''
            k = 1
        else:
            k = k + 1
```

VNS hiện tại trong repo đang đi theo đúng cấu trúc này, nhưng phần `N_k` mới được cài đặt đơn giản bằng số lần gọi `destroy_and_repair`.

---

## 3. Reactive / Adaptive VNS là gì?

### 3.1. Reactive VNS

Reactive VNS là biến thể trong đó thuật toán **phản ứng với trạng thái search hiện tại** để điều chỉnh tham số hoặc hành vi.

Ví dụ:

- tăng cường shaking khi lâu không cải thiện,
- giảm shaking khi đang cải thiện tốt,
- thay đổi xác suất chọn neighborhood theo hiệu quả gần đây,
- restart khi stagnation quá lâu,
- thay đổi acceptance threshold theo tiến trình,
- thay đổi độ lớn destroy tùy theo diversity / improvement.

Từ khóa chính:

```text
feedback-driven search
online parameter control
reactive diversification
adaptive intensification
```

### 3.2. Adaptive VNS

Adaptive VNS thường nhấn mạnh việc **học online** để chọn neighborhood / shaking / local search operator tốt hơn.

Các cơ chế phổ biến:

- score-based operator selection,
- adaptive memory,
- multi-armed bandit,
- reinforcement learning nhẹ,
- adaptive neighborhood ordering,
- adaptive shaking amplitude,
- adaptive acceptance criterion.

Reactive và Adaptive thường chồng lấn nhau. Trong thực tế có thể xem:

- Reactive = điều chỉnh theo tín hiệu search.
- Adaptive = học chính sách chọn operator / tham số dựa trên hiệu quả lịch sử.

---

## 4. Các hướng Reactive / Adaptive VNS hiện đại

## 4.1. Adaptive neighborhood selection

### Ý tưởng

Thay vì dùng thứ tự neighborhood cố định:

```python
k = 1, 2, ..., k_max
```

thuật toán duy trì score cho từng neighborhood hoặc shaking operator.

Mỗi operator được chọn theo xác suất phụ thuộc vào score:

```text
P(operator_i) ∝ score_i
```

Score được cập nhật sau mỗi lần dùng operator.

### Reward thường dùng

Một operator được thưởng nếu:

- tạo nghiệm feasible,
- cải thiện `current`,
- cải thiện `best`,
- cải thiện nhanh trên mỗi đơn vị thời gian,
- tạo diversity tốt,
- giúp thoát stagnation.

Ví dụ reward:

```text
reward = w1 * relative_improvement
       + w2 * best_improvement_bonus
       + w3 * feasibility_bonus
       - w4 * runtime_penalty
```

Với minimization:

```text
relative_improvement = max(0, (old_obj - new_obj) / max(|old_obj|, eps))
```

### Cập nhật score

Một cách đơn giản là exponential moving average:

```text
score_i = (1 - alpha) * score_i + alpha * reward_i
```

Hoặc dạng ALNS thường dùng:

```text
weight_i = (1 - rho) * weight_i + rho * segment_score_i / uses_i
```

### Lợi ích cho repo

VNS hiện tại chỉ có một shaking kiểu `destroy_and_repair` lặp `k` lần. Adaptive selection sẽ giúp:

- không bị phụ thuộc vào `k_max` cố định,
- tự học mức perturbation phù hợp,
- ưu tiên operator tạo cải thiện cho objective `paper_makespan`,
- giảm thời gian lãng phí ở operator yếu.

### Đề xuất áp dụng

Tạo nhiều shaking operator:

```python
shaking_ops = [
    "base_destroy_repair_1",
    "base_destroy_repair_2",
    "random_relocate",
    "bottleneck_relocate",
    "cross_launch_reassign",
    "large_destroy_repair_10pct",
    "large_destroy_repair_20pct",
]
```

Sau đó chọn operator bằng roulette wheel hoặc softmax theo score.

---

## 4.2. Adaptive shaking amplitude

### Ý tưởng

Trong VNS cổ điển, `k` là chỉ số neighborhood. Với bài toán routing, `k` thường tương ứng với **độ mạnh perturbation**.

Thay vì:

```python
for k in 1..k_max:
    shake k lần
```

có thể dùng amplitude động:

```text
amplitude = function(stagnation, recent_success_rate, diversity)
```

Ví dụ:

- nếu cải thiện thường xuyên → amplitude nhỏ để intensification,
- nếu lâu không cải thiện → amplitude lớn để diversification,
- nếu vừa tìm best mới → reset amplitude nhỏ.

### Công thức đơn giản

```text
if improved:
    k = max(k_min, k - 1)
else:
    k = min(k_max, k + 1)

if no_improve_iters > patience:
    k = min(k_max, k + jump)
```

Hoặc:

```text
k = k_min + floor((k_max - k_min) * stagnation / stagnation_limit)
```

### Lợi ích cho repo

Hiện tại VNS reset `k = 1` khi cải thiện, tăng `k` khi không cải thiện. Tuy nhiên:

- `k` chỉ điều khiển số lần `destroy_and_repair`;
- không phân biệt stagnation ngắn / dài;
- không có jump hoặc restart;
- không có adaptive theo tỷ lệ thành công.

### Đề xuất áp dụng

Thêm biến:

```python
no_improve_iters
success_window
k_min
k_max
k_jump
```

Logic:

```python
if improved_best:
    k = k_min
    no_improve_iters = 0
elif improved_current:
    k = max(k_min, k - 1)
else:
    no_improve_iters += 1
    if no_improve_iters >= patience:
        k = min(k_max, k + k_jump)
    else:
        k = min(k_max, k + 1)
```

---

## 4.3. Reactive acceptance criterion

### Ý tưởng

VNS cổ điển chỉ nhận nghiệm tốt hơn:

```text
accept if f(candidate) < f(current)
```

Reactive VNS hiện đại thường cho phép nhận nghiệm xấu hơn có kiểm soát để thoát local optimum.

Các acceptance phổ biến:

1. Simulated Annealing acceptance.
2. Threshold accepting.
3. Record-to-record travel.
4. Late acceptance.
5. Great deluge.
6. Strategic oscillation.

### 4.3.1. Simulated Annealing acceptance

Nhận nghiệm xấu hơn với xác suất:

```text
P = exp(-(candidate_obj - current_obj) / T)
```

Nhiệt độ `T` giảm dần hoặc reactive theo stagnation.

Reactive temperature:

```text
if stagnation high:
    T = min(T_max, T * heat_factor)
else:
    T = max(T_min, T * cool_factor)
```

### 4.3.2. Threshold accepting

Nhận nếu candidate không quá xấu:

```text
candidate_obj <= current_obj + threshold
```

Threshold có thể là tỷ lệ của best objective:

```text
threshold = tau * best_obj
```

Reactive threshold:

```text
if no improvement:
    tau *= 1.05
if improvement:
    tau *= 0.90
```

### 4.3.3. Record-to-record travel

Nhận nếu candidate không quá xa nghiệm tốt nhất:

```text
candidate_obj <= best_obj * (1 + deviation)
```

Phù hợp với minimization và dễ kiểm soát.

### Lợi ích cho repo

Objective `paper_makespan` có thể có landscape gồ ghề do bottleneck launch. Strict improvement dễ làm VNS kẹt ở nghiệm mà mọi move nhỏ đều không giảm max route cost.

Cho phép nhận nghiệm hơi xấu hơn có thể:

- đổi bottleneck launch,
- tạo điều kiện cho local search sau đó cải thiện,
- tăng exploration.

### Đề xuất áp dụng

Ưu tiên implement `record_to_record` vì đơn giản và ổn định:

```python
def accept(candidate, current, best, deviation):
    if candidate.objective + eps < current.objective:
        return True
    return candidate.objective <= best.objective * (1.0 + deviation)
```

Deviation reactive:

```python
if best improved:
    deviation = max(dev_min, deviation * 0.8)
elif stagnation:
    deviation = min(dev_max, deviation * 1.2)
```

---

## 4.4. Multi-armed bandit cho chọn operator

### Ý tưởng

Xem mỗi shaking operator hoặc local search operator là một "arm". Mỗi lần chọn operator là một lần kéo arm. Thuật toán cần cân bằng:

- exploitation: dùng operator đang hiệu quả,
- exploration: thử operator ít dùng.

Các chiến lược phổ biến:

### 4.4.1. Epsilon-greedy

```text
with probability epsilon:
    chọn random operator
else:
    chọn operator có score cao nhất
```

`epsilon` có thể giảm dần hoặc reactive tăng khi stagnation.

### 4.4.2. UCB1

```text
UCB_i = mean_reward_i + c * sqrt(log(total_uses) / uses_i)
```

Chọn operator có UCB lớn nhất.

Ưu điểm:

- tự ưu tiên operator ít thử,
- không cần normalize probability,
- dễ implement.

### 4.4.3. Thompson sampling

Duy trì phân phối xác suất thành công cho từng operator. Phù hợp nếu reward là success/failure.

### Lợi ích cho repo

Khung VNS hiện tại có thể mở rộng:

- mỗi `k` là một arm,
- mỗi shaking operator là một arm,
- mỗi local search sequence là một arm.

Ví dụ arm:

```python
Arm(name="shake_dr_1", strength=1)
Arm(name="shake_dr_3", strength=3)
Arm(name="shake_lns_10", destroy_frac=0.10)
Arm(name="shake_lns_25", destroy_frac=0.25)
Arm(name="shake_bottleneck_reassign", ...)
```

Reward:

```text
10 nếu cải thiện best
3 nếu cải thiện current
1 nếu feasible và accepted
- runtime_penalty
```

---

## 4.5. Adaptive VND ordering inside VNS

### Ý tưởng

VNS hiện tại dùng VND nhẹ cố định:

```python
light_vnd = ("intraroute_move", "zero_to_l_exchange")
```

VND đầy đủ trong base có 4 operator:

```python
"intraroute_move"
"destroy_and_repair"
"zero_to_l_exchange"
"l1_l2_exchange"
```

Adaptive VND ordering sẽ thay đổi thứ tự operator theo hiệu quả gần đây.

### Các cách làm

#### Cách 1: Move-to-front

Nếu operator cải thiện, đưa operator đó lên đầu danh sách.

```text
[N1, N2, N3, N4]
N3 cải thiện
→ [N3, N1, N2, N4]
```

#### Cách 2: Score sorting

Sau mỗi segment, sort operator theo score giảm dần.

#### Cách 3: Probabilistic VND

Không duyệt cố định, mà chọn operator theo xác suất score-based cho đến khi đạt local optimum mềm.

### Lợi ích cho repo

Một số instance có thể hưởng lợi nhiều từ `zero_to_l_exchange`, một số instance lại chủ yếu cải thiện bằng `intraroute_move`.

Adaptive ordering giúp:

- giảm thời gian chạy operator kém,
- tăng intensification ở operator phù hợp,
- tự thích nghi với kích thước instance và objective.

### Đề xuất áp dụng

Bước dễ nhất:

- thêm tùy chọn `adaptive_light_vnd=True`,
- dùng danh sách operator có thể thay đổi thứ tự,
- khi operator cải thiện thì move-to-front.

---

## 4.6. Reactive diversification restart

### Ý tưởng

Khi search không cải thiện trong một số vòng, thay vì tiếp tục tăng `k`, thuật toán restart từ nghiệm khác.

Nguồn restart:

1. nghiệm tốt nhất đã lưu trong elite pool,
2. nghiệm random mới,
3. nghiệm construct lại bằng seed khác,
4. nghiệm recombine từ hai elite solutions,
5. nghiệm best nhưng perturb mạnh.

### Các trigger restart

```text
no_improve_iters >= patience
time_since_last_best >= time_patience
diversity below threshold
operator success rate too low
```

### Lợi ích cho repo

Hiện tại VNS không có restart. Nếu `_shake` không tạo cải thiện vì `destroy_and_repair` chỉ trả nghiệm cải thiện, search có thể xoay quanh cùng một vùng.

### Đề xuất áp dụng

Duy trì elite pool:

```python
elite_pool: list[Solution]
```

Khi stagnation:

```text
50%: restart từ elite random + strong shaking
30%: restart từ constructed random solution + light VND
20%: restart từ best + very strong shaking
```

Sau restart:

```python
current = restarted_solution
k = 1
```

---

## 4.7. Elite memory và path relinking

### Ý tưởng

Adaptive metaheuristics hiện đại thường dùng memory:

- lưu top solutions,
- tránh quay lại nghiệm quá giống,
- khai thác vùng giữa hai nghiệm tốt.

Path relinking:

```text
given solution A and B:
    gradually transform A toward B
    evaluate intermediate solutions
    keep best
```

Trong routing, transformation có thể là:

- chuyển task assignment theo launch của solution B,
- đổi hướng task theo B,
- đổi thứ tự task trong flight theo B,
- move một block task từ A sang giống B.

### Lợi ích cho repo

Bài toán có cấu trúc assignment:

```text
required edge → launch → flight → position → orientation
```

Vì vậy có thể đo similarity và thực hiện relinking tương đối rõ.

### Đề xuất áp dụng mức đơn giản

Không cần path relinking đầy đủ ngay. Có thể bắt đầu bằng elite memory:

- lưu top `elite_size` nghiệm khác nhau,
- diversity metric dựa trên mapping `edge_id -> launch`,
- restart từ elite ít giống best nhất nếu stagnation.

Diversity đơn giản:

```text
distance(A, B) =
    tỷ lệ required_edges có launch assignment khác nhau
```

---

## 4.8. Adaptive destroy-and-repair dùng ý tưởng ALNS

### Ý tưởng

LNS/ALNS hiện đại dùng nhiều destroy và repair operator, chọn adaptively theo score.

Có thể đưa ý tưởng này vào VNS bằng cách làm shaking mạnh hơn:

Destroy operators:

1. random removal,
2. worst removal,
3. bottleneck removal,
4. related removal,
5. route removal,
6. Shaw removal,
7. cost-contribution removal.

Repair operators:

1. greedy best insertion,
2. regret-2 insertion,
3. regret-3 insertion,
4. randomized greedy insertion,
5. bottleneck-aware insertion.

### Liên hệ với repo

Repo đã có:

- `destroy_and_repair` trong base,
- `_destroy` / `_repair` trong `LNSSolver`,
- `all_possible_insertions`,
- `insert_task`.

Có thể reuse nhiều logic từ `LNSSolver` để tạo shaking operators cho VNS.

### Đề xuất operator phù hợp với `paper_makespan`

#### 1. Bottleneck cost removal

Remove task từ launch đang có objective lớn nhất.

Score task theo đóng góp cost:

```text
contribution(task) = objective_before - objective_after_removing_task
```

Remove top tasks có contribution cao.

#### 2. Related removal theo line

Remove các task cùng original line hoặc gần nhau hình học.

Phù hợp với polygonal chain vì các required edge có quan hệ theo line.

#### 3. Cross-launch repair

Khi repair, ưu tiên chèn task vào launch không bottleneck nếu feasible.

Insertion cost nên không chỉ là objective absolute, mà gồm penalty bottleneck:

```text
score = new_objective
      + lambda_balance * std(ghg_by_launch)
      + lambda_total * paper_makespan
```

#### 4. Regret insertion

Thay vì chèn task có best insertion trước, tính regret:

```text
regret(task) = second_best_insertion_cost - best_insertion_cost
```

Chèn task có regret lớn trước để tránh bị kẹt.

---

## 4.9. Adaptive objective scalarization cho `paper_makespan`

### Vấn đề

Với objective `paper_makespan`, nhiều move có thể không giảm max objective ngay nhưng vẫn cải thiện cấu trúc nghiệm:

- giảm aggregate route cost,
- cân bằng tải giữa launch,
- giảm paper makespan,
- giảm số flight,
- giảm bottleneck tiềm năng.

Nếu chỉ accept khi `objective` giảm strict, search có thể bỏ qua các bước chuẩn bị hữu ích.

### Ý tưởng

Dùng auxiliary score trong acceptance hoặc tie-breaking:

```text
primary: paper_makespan
secondary: paper_makespan
tertiary: paper_makespan
quaternary: balance
```

Lexicographic hoặc weighted adaptive:

```text
search_score =
    objective
    + lambda_total * normalized_paper_makespan
    + lambda_balance * normalized_balance
```

Trong đó `lambda` tăng khi stagnation.

### Đề xuất áp dụng

Khi so sánh candidate trong VNS, dùng hai mức:

1. Nếu candidate objective tốt hơn → nhận.
2. Nếu objective bằng hoặc trong threshold nhỏ:
   - nhận nếu `paper_makespan` giảm,
   - hoặc `paper_makespan` giảm,
   - hoặc objective balance tốt hơn.

Điều này đặc biệt hữu ích cho plateau của min-max objective.

---

## 4.10. Strategic oscillation quanh feasibility / flight limit

### Ý tưởng

Một số metaheuristics hiện đại cho phép nghiệm tạm thời infeasible với penalty, sau đó kéo về feasible.

Với bài toán này, infeasibility chính là:

- flight vượt `flight_limit`,
- required edge thiếu / lặp,
- số launch vượt `num_trucks`.

Thay vì cấm hoàn toàn, có thể cho phép flight hơi vượt limit với penalty:

```text
penalized_obj = objective + penalty_weight * violation
```

Penalty reactive:

```text
if too many infeasible accepted:
    penalty_weight *= 1.2
if all candidates feasible for long:
    penalty_weight *= 0.9
```

### Lợi ích

Cho phép đi qua vùng infeasible để đổi cấu trúc flight / launch assignment, sau đó repair về feasible.

### Rủi ro

Cần sửa nhiều hàm hiện tại vì `insert_task` và feasibility đang loại nghiệm infeasible sớm.

### Đề xuất

Đây là hướng nâng cao, không nên làm đầu tiên. Chỉ nên triển khai sau khi các hướng adaptive shaking / acceptance / operator selection đã ổn.

---

## 4.11. Parallel / island VNS

### Ý tưởng

Chạy nhiều VNS process hoặc nhiều island với parameter khác nhau:

- island 1: intensification, k nhỏ,
- island 2: diversification, k lớn,
- island 3: SA acceptance,
- island 4: ALNS-style shaking.

Định kỳ trao đổi elite solutions.

### Lợi ích

Phù hợp nếu benchmark nhiều instance và có CPU nhiều core.

### Đề xuất

Có thể implement bên ngoài solver trước:

```text
run several VNSSolver variants with different seeds/configs
collect best solution
```

Sau đó nếu hiệu quả thì tích hợp island memory vào solver.

---

## 5. Đánh giá VNS hiện tại trong repo

## 5.1. Điểm mạnh

VNS hiện tại có các điểm tốt:

- đơn giản, dễ hiểu, dễ debug;
- reuse được `vnd_improvement`;
- có initial pool thay vì chỉ một nghiệm;
- có full VND cuối để intensification;
- cấu trúc gần chuẩn VNS cổ điển;
- ít tham số nên dễ chạy benchmark.

## 5.2. Điểm hạn chế

### Hạn chế 1: Shaking chưa thực sự là perturbation

Hiện tại:

```python
result = self.destroy_and_repair(current)
if result is not None:
    current = result
```

Nhưng `destroy_and_repair` trong base là local improvement operator, thường chỉ trả nghiệm nếu tốt hơn.

Do đó `_shake` có thể không tạo thay đổi nếu không tìm thấy improvement. Điều này trái với vai trò shaking là **làm xáo trộn nghiệm để thoát local optimum**, kể cả tạm thời xấu hơn.

### Hạn chế 2: Chỉ có một dạng shaking

`k` hiện chỉ là số lần lặp `destroy_and_repair`. Không có nhiều neighborhood khác nhau về bản chất.

### Hạn chế 3: Acceptance quá chặt

Chỉ nhận nếu:

```python
improved.objective + 1e-9 < current.objective
```

Điều này dễ kẹt với objective min-max.

### Hạn chế 4: Không có memory

Solver không nhớ:

- operator nào tốt,
- mức k nào tốt,
- elite solutions,
- diversity,
- stagnation pattern.

### Hạn chế 5: Không có restart

Nếu không cải thiện lâu, solver vẫn tiếp tục cùng logic cũ.

### Hạn chế 6: Không có splitting phase

VND và LNS có coarse-to-fine splitting, còn VNS thì chưa.

---

## 6. Các cải thiện đề xuất theo mức ưu tiên

## 6.1. Nhóm A — Dễ làm, tác động cao

### A1. Biến `_shake` thành perturbation thật

#### Hiện tại

```python
def _shake(self, solution: Solution, k: int) -> Solution:
    current = solution.clone()
    for _ in range(k):
        result = self.destroy_and_repair(current)
        if result is not None:
            current = result
    return current
```

#### Vấn đề

Nếu `destroy_and_repair` không cải thiện, `current` không đổi.

#### Đề xuất

Thêm shaking operator riêng, có thể tạo nghiệm xấu hơn nhưng vẫn feasible:

```python
def _random_relocate_shake(self, solution: Solution, moves: int) -> Solution:
    current = solution.clone()
    for _ in range(moves):
        task = sample_random_task(current)
        partial = remove_task(current, task)
        insertion = sample_feasible_insertion(partial, task)
        if insertion is not None:
            current = insertion
    return self.evaluate(current)
```

Nếu chưa muốn viết nhiều hàm mới, có thể reuse `_destroy/_repair` từ `LNSSolver` theo dạng random large destroy repair.

---

### A2. Thêm reactive acceptance

#### Đề xuất đơn giản

Thêm vào `VNSSolver.__init__`:

```python
acceptance: str = "record_to_record"
deviation_start: float = 0.01
deviation_min: float = 0.001
deviation_max: float = 0.05
```

Acceptance:

```python
def _accept(self, candidate, current, best, deviation):
    if candidate.objective + 1e-9 < current.objective:
        return True
    if self.acceptance == "record_to_record":
        return candidate.objective <= best.objective * (1.0 + deviation)
    return False
```

Update deviation:

```python
if improved_best:
    deviation = max(deviation_min, deviation * 0.8)
elif no_improve_iters >= patience:
    deviation = min(deviation_max, deviation * 1.2)
```

#### Lợi ích

- Cho phép thoát plateau.
- Dễ implement.
- Không phá cấu trúc solver.

---

### A3. Adaptive `k`

Thêm:

```python
stagnation_patience: int = 10
k_jump: int = 2
```

Logic:

```python
if candidate improves current:
    k = 1
elif accepted but not improving:
    k = max(1, k - 1)
else:
    k += 1

if no_improve_iters >= stagnation_patience:
    k = min(k_max, k + k_jump)
```

#### Lợi ích

- Shaking mạnh hơn khi lâu không cải thiện.
- Không cần thử tuần tự cứng nhắc.

---

### A4. Restart khi stagnation

Thêm:

```python
restart_patience: int = 30
elite_size: int = 5
```

Khi không cải thiện quá lâu:

```python
if no_improve_iters >= restart_patience:
    current = restart_from_elite_or_new_pool()
    k = 1
    no_improve_iters = 0
```

Restart đơn giản:

```text
best.clone() + strong shake
```

hoặc:

```text
random solution từ generate_initial_pool + light VND
```

---

## 6.2. Nhóm B — Trung bình, đáng làm sau nhóm A

### B1. Multi-operator shaking

Tạo nhiều shaking operator:

1. `repeat_destroy_repair(k)`
2. `random_task_relocate(k)`
3. `bottleneck_task_relocate(k)`
4. `lns_destroy_repair(frac=0.1)`
5. `lns_destroy_repair(frac=0.2)`
6. `cross_launch_reassign(k)`

Interface:

```python
@dataclass
class ShakeOperator:
    name: str
    strength: int
    weight: float = 1.0
    uses: int = 0
    reward: float = 0.0
```

Chọn operator:

```python
op = select_operator(shake_ops)
candidate = op.apply(current)
```

---

### B2. Score-based operator adaptation

Reward:

```python
reward = 0.0
if feasible:
    reward += 0.1
if accepted:
    reward += 1.0
if candidate improves current:
    reward += 3.0
if candidate improves best:
    reward += 10.0
reward += relative_improvement * 100.0
reward -= runtime_seconds * 0.01
```

Update:

```python
op.score = (1 - alpha) * op.score + alpha * reward
```

Selection:

```text
softmax(score / temperature)
```

hoặc roulette:

```text
P_i = score_i / sum(score)
```

---

### B3. Adaptive light VND

Thay:

```python
light_vnd = ("intraroute_move", "zero_to_l_exchange")
```

bằng danh sách có thể cấu hình:

```python
light_vnd_candidates = [
    ("intraroute_move",),
    ("intraroute_move", "zero_to_l_exchange"),
    ("intraroute_move", "destroy_and_repair"),
    ("zero_to_l_exchange", "intraroute_move"),
]
```

Chọn sequence theo bandit hoặc score.

---

### B4. Plateau-aware tie-breaking cho `paper_makespan`

Khi objective không đổi hoặc chênh rất nhỏ:

```python
if abs(candidate.objective - current.objective) <= eps_plateau:
    accept if candidate.paper_makespan < current.paper_makespan
```

Hoặc:

```python
secondary_score = (
    candidate.objective,
    candidate.paper_makespan,
    candidate.paper_makespan,
)
```

So sánh lexicographic.

---

### B5. Tích hợp splitting phase cho VNS

Có thể copy/adapt logic từ `VNDSolver.splitting_phase(...)`.

Pipeline mới:

```text
VNS search on coarse instance
→ choose top VNS/elite candidates
→ splitting phase coarse-to-fine
→ final VND
```

Đây là cải thiện quan trọng nếu muốn VNS cạnh tranh công bằng với VND/LNS hiện tại.

---

## 6.3. Nhóm C — Nâng cao / nghiên cứu sâu

### C1. ALNS-style VNS

Kết hợp VNS và ALNS:

- VNS quản lý shaking amplitude và local search.
- ALNS quản lý destroy/repair operator adaptation.

Pseudo:

```text
while not stop:
    destroy_op = select_destroy()
    repair_op = select_repair()
    x' = destroy_repair(x, destroy_op, repair_op, degree)
    x'' = adaptive_vnd(x')
    accepted = accept(x'', x, best)
    update_scores(destroy_op, repair_op, accepted, improvement)
    update_degree_reactively()
```

### C2. Learning-based operator selection

Có thể dùng contextual bandit với features:

- current objective,
- gap to best,
- stagnation,
- number of flights,
- bottleneck launch load,
- aggregate route cost,
- current `k`,
- instance size.

Action:

- chọn shaking operator,
- chọn local search sequence,
- chọn destroy fraction.

Reward:

- normalized improvement per second.

### C3. Path relinking giữa elite solutions

Dùng elite pool để tạo nghiệm mới bằng cách chuyển dần assignment từ solution A sang B.

### C4. Infeasible VNS với penalty adaptive

Cho phép temporary infeasible solution với penalty reactive.

### C5. Parallel island VNS

Chạy nhiều biến thể VNS song song, trao đổi elite solutions.

---

## 7. Thiết kế cụ thể cho `AdaptiveVNSSolver`

## 7.1. Mục tiêu thiết kế

Thêm solver mới thay vì sửa mạnh `VNSSolver` hiện tại:

```text
mm_mt_dlarp/algorithms/adaptive_vns.py
```

hoặc thêm option vào `VNSSolver`.

Khuyến nghị: tạo class mới để không phá benchmark hiện tại.

```python
class AdaptiveVNSSolver(VNSSolver):
    ...
```

## 7.2. Config đề xuất

```python
@dataclass
class AdaptiveVNSConfig:
    k_min: int = 1
    k_max: int = 8
    max_iter: int = 200

    acceptance: str = "record_to_record"
    deviation_start: float = 0.01
    deviation_min: float = 0.001
    deviation_max: float = 0.05

    adaptive_k: bool = True
    stagnation_patience: int = 10
    restart_patience: int = 40
    k_jump: int = 2

    elite_size: int = 5

    operator_alpha: float = 0.2
    operator_selection: str = "ucb"
    ucb_c: float = 1.0

    use_adaptive_vnd: bool = True
    use_splitting: bool = False
```

## 7.3. State cần lưu

```python
self.operator_stats = {
    name: {
        "uses": 0,
        "score": 1.0,
        "total_reward": 0.0,
        "successes": 0,
        "best_hits": 0,
        "runtime": 0.0,
    }
}

self.elite_pool = []
self.no_improve_iters = 0
self.deviation = deviation_start
```

## 7.4. Pseudo-code đề xuất

```text
initialize current by best of initial pool + light/adaptive VND
best = current
elite_pool = [best]

for iter in 1..max_iter:
    op = select_shake_operator(operator_stats, no_improve_iters)
    k = compute_adaptive_k(no_improve_iters, op)

    candidate = op.shake(current, k)
    candidate = adaptive_vnd(candidate)

    accepted = accept(candidate, current, best, deviation)

    reward = compute_reward(candidate, current, best, accepted, runtime)
    update_operator_stats(op, reward)

    if accepted:
        current = candidate

    if candidate improves best:
        best = candidate
        add_to_elite(best)
        no_improve_iters = 0
        deviation = cool(deviation)
        k = k_min
    else:
        no_improve_iters += 1
        if no_improve_iters >= stagnation_patience:
            deviation = heat(deviation)
            k = increase(k)

    if no_improve_iters >= restart_patience:
        current = restart(elite_pool, best)
        no_improve_iters = 0
        k = k_min

final = full_vnd(best)
optional splitting_phase(final)
return best
```

---

## 8. Các shaking operator nên thêm trước

## 8.1. `random_relocate`

Remove một task bất kỳ rồi chèn lại vào vị trí feasible ngẫu nhiên hoặc best trong sample.

```text
for move in 1..k:
    task = random task
    remove task
    sample up to S feasible insertions
    choose one randomly among top p%
```

Ưu điểm:

- perturbation thật,
- rẻ hơn full all insertion,
- giúp đổi cấu trúc nhỏ.

## 8.2. `bottleneck_relocate`

Tập trung vào launch đang là bottleneck theo `makespan_by_launch`.

```text
bottleneck = argmax ghg_by_launch
remove task from bottleneck
insert into non-bottleneck launch if feasible
```

Phù hợp trực tiếp với `paper_makespan`.

## 8.3. `lns_shake_small`

Reuse logic LNS với destroy fraction nhỏ:

```text
destroy_frac = 0.05 hoặc 0.10
```

Dùng repair greedy.

## 8.4. `lns_shake_medium`

```text
destroy_frac = 0.20 hoặc 0.30
```

Dùng khi stagnation cao.

## 8.5. `route_rebuild`

Chọn một launch hoặc một flight, remove toàn bộ task trong flight đó, rồi repair lại.

## 8.6. `orientation_flip_block`

Đảo hướng một số task hoặc block task, sau đó intraroute improvement.

Có thể hữu ích vì mỗi `Task` có hướng phục vụ edge.

---

## 9. Reward design cụ thể cho bài toán `paper_makespan`

Với minimization:

```python
old = current.objective
new = candidate.objective
best_old = best.objective
```

Reward đề xuất:

```text
reward = 0

if candidate feasible:
    reward += 0.1

if accepted:
    reward += 1.0

if new < old:
    reward += 3.0
    reward += 50 * (old - new) / max(abs(old), 1e-9)

if new < best_old:
    reward += 10.0
    reward += 100 * (best_old - new) / max(abs(best_old), 1e-9)

if new approximately equals old and candidate.paper_makespan < current.paper_makespan:
    reward += 0.5

reward -= 0.01 * runtime_seconds
```

Nên clamp reward:

```text
reward = min(max(reward, -1.0), 20.0)
```

để tránh operator được thưởng quá lớn do một lần may mắn.

---

## 10. Acceptance design đề xuất

## 10.1. Record-to-record mặc định

```python
def accept(candidate, current, best, deviation):
    if candidate.objective + eps < current.objective:
        return True

    threshold = best.objective * (1.0 + deviation)
    if candidate.objective <= threshold:
        return True

    return False
```

## 10.2. Plateau-aware acceptance

Thêm secondary metric:

```python
if abs(candidate.objective - current.objective) <= eps_plateau:
    if candidate.paper_makespan + eps < current.paper_makespan:
        return True
```

## 10.3. Reactive deviation

```python
if best_improved:
    deviation = max(deviation_min, deviation * 0.8)
elif no_improve_iters > stagnation_patience:
    deviation = min(deviation_max, deviation * 1.1)
else:
    deviation = max(deviation_min, deviation * 0.99)
```

---

## 11. Adaptive local search design

## 11.1. VND nhẹ hiện tại

```python
light_vnd = ("intraroute_move", "zero_to_l_exchange")
```

## 11.2. Đề xuất các profile local search

```python
profiles = {
    "cheap": ("intraroute_move",),
    "balanced": ("intraroute_move", "zero_to_l_exchange"),
    "repair": ("destroy_and_repair", "intraroute_move"),
    "full_no_exact": (
        "intraroute_move",
        "destroy_and_repair",
        "zero_to_l_exchange",
        "l1_l2_exchange",
    ),
}
```

Chọn profile theo phase:

```text
if frequent improvement:
    use cheap/balanced
if stagnation:
    use repair/full_no_exact
if close to final:
    use full VND
```

## 11.3. Budget-aware local search

Nếu time limit gần hết:

```text
remaining_time < 20% total
→ dùng cheap local search
→ dành cuối cho full VND trên best
```

---

## 12. Metrics nên log thêm

Để đánh giá Reactive/Adaptive VNS, nên log:

```text
iteration
time
current_objective
best_objective
candidate_objective
accepted
improved_current
improved_best
k
shake_operator
operator_reward
operator_score
deviation
no_improve_iters
restart_count
paper_makespan
paper_makespan
```

Có thể tạo:

```python
self.adaptive_log: list[dict]
```

Điều này giúp phân tích:

- operator nào hiệu quả,
- thời điểm restart,
- deviation có quá rộng không,
- acceptance có nhận quá nhiều nghiệm xấu không.

---

## 13. Benchmark protocol đề xuất

Để kiểm chứng cải thiện, nên so sánh:

1. `VNDSolver`
2. `VNSSolver` hiện tại
3. `LNSSolver`
4. `AdaptiveVNSSolver` mới

Trên cùng:

- random seed,
- time limit,
- initial discretization,
- objective,
- number of runs mỗi instance.

Metrics:

```text
best objective
mean objective
std objective
best paper_makespan
mean runtime
time-to-best
convergence curve
```

Vì metaheuristic có randomness, nên mỗi instance cần chạy nhiều lần:

```text
n_runs = 10 hoặc 30
```

Dùng kiểm định:

- Wilcoxon signed-rank test,
- Friedman test + post-hoc Nemenyi,
- Vargha-Delaney A12 effect size.

---

## 14. Lộ trình triển khai khuyến nghị

## Phase 1 — Sửa shaking và acceptance

Mục tiêu: cải thiện lớn với ít thay đổi.

- Tạo perturbation shaking thật:
  - `random_relocate_shake`
  - `bottleneck_relocate_shake`
- Thêm record-to-record acceptance.
- Thêm adaptive deviation.
- Thêm stagnation counter.

Kỳ vọng:

- VNS thoát local optimum tốt hơn.
- Convergence curve ít bị phẳng sớm.

---

## Phase 2 — Restart và elite memory

- Lưu `elite_pool`.
- Restart khi stagnation.
- Restart từ best + strong shake hoặc random initial solution.
- Diversity check đơn giản bằng assignment edge-to-launch.

Kỳ vọng:

- Tốt hơn trên instance lớn / khó.
- Giảm phụ thuộc vào seed.

---

## Phase 3 — Operator adaptation

- Tạo nhiều shaking operators.
- Thêm score / UCB selection.
- Log operator stats.
- Tuning reward.

Kỳ vọng:

- Solver tự thích nghi theo instance.
- Giảm thời gian cho operator kém.

---

## Phase 4 — Adaptive VND và splitting phase

- Adaptive ordering cho local search.
- Thêm splitting phase cho VNS.
- Full VND cuối mỗi restart hoặc mỗi best update.

Kỳ vọng:

- VNS cạnh tranh công bằng hơn với VND/LNS hiện tại.
- Tốt hơn khi nghiệm cần refined breakpoints.

---

## Phase 5 — ALNS-style VNS

- Nhiều destroy operators.
- Nhiều repair operators.
- Regret insertion.
- Related removal theo line / geometry.
- Adaptive destroy degree.

Kỳ vọng:

- Đây là hướng mạnh nhất về chất lượng nghiệm nhưng cần nhiều code và benchmark.

---

## 15. Thay đổi tối thiểu có thể implement ngay trong `vns.py`

Nếu muốn cải thiện nhanh mà chưa tạo solver mới, có thể sửa `VNSSolver` theo hướng sau.

### 15.1. Thêm tham số

```python
def __init__(
    self,
    instance,
    config,
    k_max: int = 8,
    max_iter: int = 200,
    acceptance: str = "record_to_record",
    deviation_start: float = 0.01,
    deviation_min: float = 0.001,
    deviation_max: float = 0.05,
    stagnation_patience: int = 10,
    restart_patience: int = 40,
):
```

### 15.2. Thêm `_accept`

```python
def _accept(self, candidate, current, best, deviation):
    if candidate.objective + 1e-9 < current.objective:
        return True
    if self.acceptance == "record_to_record":
        return candidate.objective <= best.objective * (1.0 + deviation)
    return False
```

### 15.3. Thêm `_update_deviation`

```python
def _update_deviation(self, deviation, improved_best, stagnated):
    if improved_best:
        return max(self.deviation_min, deviation * 0.8)
    if stagnated:
        return min(self.deviation_max, deviation * 1.1)
    return max(self.deviation_min, deviation * 0.99)
```

### 15.4. Sửa main loop

```text
for iteration:
    shaken = shake(current, k)
    improved = light_vnd(shaken)
    accepted = accept(improved, current, best, deviation)

    if accepted:
        current = improved

    if improved improves best:
        best = improved
        no_improve = 0
        k = 1
    else:
        no_improve += 1
        k = min(k_max, k + 1)

    if no_improve >= restart_patience:
        current = strong_shake(best)
        no_improve = 0
        k = 1
```

### 15.5. Thêm plateau tie-break

Khi objective gần bằng:

```python
if abs(candidate.objective - current.objective) <= 1e-9:
    if candidate.paper_makespan < current.paper_makespan:
        accept
```

---

## 16. Cảnh báo khi áp dụng

### 16.1. Không nên thêm quá nhiều cơ chế cùng lúc

Reactive/Adaptive VNS có nhiều tham số. Nếu thêm đồng thời:

- acceptance,
- restart,
- operator adaptation,
- adaptive VND,
- splitting,
- ALNS destroy/repair,

thì sẽ khó biết cải thiện đến từ đâu.

Nên làm incremental và benchmark từng phase.

### 16.2. Reward sai có thể làm solver tệ hơn

Nếu reward chỉ dựa trên best improvement, operator exploration có thể bị nghèo. Nên reward cả:

- accepted,
- current improvement,
- feasible perturbation,
- plateau secondary improvement.

### 16.3. Acceptance quá lỏng làm mất intensification

Nếu `deviation_max` quá lớn, solver có thể random walk. Với objective `paper_makespan`, nên bắt đầu nhỏ:

```text
deviation_start = 0.005 đến 0.02
deviation_max   = 0.03 đến 0.05
```

### 16.4. Shaking quá mạnh làm repair tốn thời gian

Large destroy nên giới hạn số task remove hoặc số insertion candidate.

### 16.5. Cần seed control

Adaptive stochastic solver cần seed ổn định để benchmark reproducible.

---

## 17. Mapping cải thiện với code hiện tại

| Hướng cải thiện | File chính | Mức khó | Tác động kỳ vọng |
|---|---|---:|---:|
| Record-to-record acceptance | `algorithms/vns.py` | Thấp | Cao |
| Adaptive `k` theo stagnation | `algorithms/vns.py` | Thấp | Trung bình-Cao |
| Restart từ best + strong shake | `algorithms/vns.py` | Thấp | Trung bình |
| Perturbation shaking thật | `algorithms/vns.py`, có thể reuse `lns.py` | Trung bình | Cao |
| Elite pool | `algorithms/vns.py` | Trung bình | Trung bình-Cao |
| Multi-operator shaking | `algorithms/vns.py` | Trung bình | Cao |
| UCB / bandit operator selection | `algorithms/vns.py` | Trung bình | Cao |
| Adaptive VND ordering | `algorithms/base.py` hoặc `vns.py` | Trung bình | Trung bình |
| Splitting phase cho VNS | `algorithms/vns.py`, reuse `vnd.py` | Trung bình | Cao |
| ALNS-style destroy/repair | `algorithms/vns.py`, `lns.py`, `base.py` | Cao | Rất cao |
| Path relinking | module mới | Cao | Trung bình-Cao |
| Infeasible VNS penalty | nhiều module | Cao | Không chắc, cần nghiên cứu |

---

## 18. Đề xuất phiên bản `ReactiveVNSSolver` thực dụng nhất

Nếu mục tiêu là cải thiện nhanh và có khả năng publish/report tốt, nên triển khai phiên bản sau:

### Thành phần

1. Real perturbation shaking:
   - random relocate,
   - bottleneck relocate,
   - small LNS destroy-repair.

2. Adaptive shaking amplitude:
   - tăng theo stagnation,
   - reset khi best improvement.

3. Record-to-record acceptance:
   - deviation reactive.

4. Elite restart:
   - restart khi stagnation.

5. Operator score:
   - reward đơn giản,
   - epsilon-greedy hoặc UCB.

6. Optional splitting phase:
   - dùng lại logic từ `VNDSolver`.

### Tên có thể dùng

```text
Reactive Adaptive VNS with Elite Memory and Record-to-Record Acceptance
```

Viết tắt:

```text
RA-VNS
```

Hoặc nếu có ALNS-style operators:

```text
ALNS-guided Reactive VNS
```

### Pseudo-code cuối

```text
Generate initial pool
Apply light VND
current = best initial
best = current
elite = {best}

Initialize shaking operator scores
deviation = deviation_start
k = k_min

for iter = 1..max_iter:
    op = select_operator(scores, epsilon/UCB)
    shaken = op.apply(current, strength=k)
    candidate = adaptive_light_vnd(shaken)

    accepted = record_to_record_accept(candidate, current, best, deviation)

    if accepted:
        current = candidate

    if candidate improves best:
        best = candidate
        update elite
        no_improve = 0
        k = k_min
        deviation = cool(deviation)
        reward op strongly
    else:
        no_improve += 1
        k = adaptively_increase(k, no_improve)
        deviation = heat_if_stagnated(deviation)
        reward op based on accepted/current improvement/secondary metrics

    if no_improve >= restart_patience:
        current = restart_from_elite_or_random(best, elite)
        no_improve = 0
        k = k_min

final = full_vnd(best)

if use_splitting:
    final = splitting_phase(final)

return final
```

---

## 19. Kết luận

VNS hiện tại trong repo là một baseline VNS cổ điển, dễ hiểu nhưng còn khá "tĩnh":

- shaking chưa thực sự phá nghiệm,
- `k` tăng/reset theo rule cố định,
- acceptance chỉ nhận cải thiện strict,
- không có operator learning,
- không có restart/elite memory,
- chưa có splitting phase.

Các hướng Reactive/Adaptive VNS hiện đại có thể cải thiện đáng kể bằng cách đưa feedback từ quá trình search vào quyết định:

- chọn shaking operator nào,
- shaking mạnh bao nhiêu,
- có nhận nghiệm xấu hơn nhẹ hay không,
- khi nào restart,
- local search nào nên ưu tiên,
- có cần mở rộng rời rạc hóa hay không.

Khuyến nghị triển khai theo thứ tự:

1. **real perturbation shaking**,
2. **record-to-record reactive acceptance**,
3. **adaptive `k` + stagnation restart**,
4. **elite memory**,
5. **multi-operator shaking với UCB/score adaptation**,
6. **splitting phase cho VNS**,
7. **ALNS-style destroy/repair nếu cần chất lượng cao hơn**.

Hướng khả thi và có tác động cao nhất cho code hiện tại là xây dựng một solver mới `ReactiveVNSSolver` hoặc `AdaptiveVNSSolver` kế thừa `VNSSolver`, giữ baseline cũ để so sánh, và benchmark từng phase trên cùng bộ instance.