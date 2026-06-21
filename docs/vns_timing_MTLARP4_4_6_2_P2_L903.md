# VNS Timing Experiment

| Metric | Value |
| --- | ---: |
| Instance                                    | MTLARP4_4_6_2.dat |
| Num trucks                                  | 2 |
| Flight limit                                | 903.000000 |
| Objective                                   | 1623.516226 |
| Paper makespan                              | 1623.516226 |
| Trucks used                                 | 2 |
| Total solve time (s)                        | 32.936153 |
| Adaptive shake total (s)                    | 20.848974 |
| Adaptive shake calls                        | 409 |
| Adaptive VND total (s)                      | 5.703792 |
| Adaptive VND calls                          | 411 |
| Splitting phase final time (s)              | 0.199546 |
| Splitting phase calls                       | 1 |
| Adaptive VND outside splitting estimate (s) | 5.504246 |

## Notes

- `Adaptive VND total` is cumulative over every `_adaptive_vnd(...)` call, including calls made inside `splitting_phase`.
- `Splitting phase final time` is inclusive wall-clock time for the final `splitting_phase(...)` call.
- `Adaptive VND outside splitting estimate` is computed as `max(0, adaptive_vnd_total - splitting_phase_time)`; it is an estimate because splitting time also contains conversion/refinement overhead.
