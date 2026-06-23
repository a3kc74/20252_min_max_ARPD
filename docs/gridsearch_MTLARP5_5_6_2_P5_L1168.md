# Grid Search Best Parameters

Ranked by objective first, then runtime.

| Algorithm | Seed | Parameters                            | Objective   | Paper Makespan | Time       | Trucks Used | Convergence Iterations                                                                                 |
| --------- | ---- | ------------------------------------- | ----------- | -------------- | ---------- | ----------- | ------------------------------------------------------------------------------------------------------ |
| vns       | 0    | vns_k_max=2;vns_max_iter=100           | 1611.435059 | 1611.435059    | 104.592691 | 5           | initial@0;restart_elite@30;final@50                                                                    |
| lns       | 0    | lns_destroy_frac=0.4;lns_max_iter=200 | 1515.346576 | 1515.346576    | 175.499431 | 5           | initial_pool@0;candidate2:lns@12;candidate2:lns@19;candidate2:candidate_vnd@200;candidate1:splitting@1 |
| vnd       | 0    | vnd_split_top_k=5                     | 1560.607088 | 1560.607088    | 113.037263 | 5           | 0;1                                                                                                    |

## All Trials

| Algorithm | Seed | Parameters                            | Objective   | Paper Makespan | Time       | Trucks Used | Convergence Iterations                                                                                 |
| --------- | ---- | ------------------------------------- | ----------- | -------------- | ---------- | ----------- | ------------------------------------------------------------------------------------------------------ |
| vns       | 0    | vns_k_max=2;vns_max_iter=50           | 1611.435059 | 1611.435059    | 104.592691 | 5           | initial@0;restart_elite@30;final@50                                                                    |
| vns       | 0    | vns_k_max=3;vns_max_iter=50           | 1611.435059 | 1611.435059    | 151.700858 | 5           | initial@0;restart_elite@30;final@50                                                                    |
| vns       | 0    | vns_k_max=4;vns_max_iter=50           | 1611.435059 | 1611.435059    | 180.478541 | 5           | initial@0;restart_elite@30;final@50                                                                    |
| lns       | 0    | lns_destroy_frac=0.2;lns_max_iter=100 | 1587.900891 | 1587.900891    | 146.273160 | 5           | initial_pool@0;candidate2:lns@20;candidate2:candidate_vnd@100                                          |
| lns       | 0    | lns_destroy_frac=0.2;lns_max_iter=200 | 1540.202891 | 1540.202891    | 159.238051 | 5           | initial_pool@0;candidate2:lns@47;candidate2:candidate_vnd@200;candidate1:splitting@1                   |
| lns       | 0    | lns_destroy_frac=0.3;lns_max_iter=100 | 1565.454514 | 1565.454514    | 153.922333 | 5           | initial_pool@0;candidate2:lns@72;candidate2:candidate_vnd@100;candidate10:splitting@10                 |
| lns       | 0    | lns_destroy_frac=0.3;lns_max_iter=200 | 1587.900891 | 1587.900891    | 149.702633 | 5           | initial_pool@0;candidate2:lns@7;candidate2:candidate_vnd@200                                           |
| lns       | 0    | lns_destroy_frac=0.4;lns_max_iter=100 | 1587.900891 | 1587.900891    | 148.855939 | 5           | initial_pool@0;candidate2:lns@37;candidate2:candidate_vnd@100                                          |
| lns       | 0    | lns_destroy_frac=0.4;lns_max_iter=200 | 1515.346576 | 1515.346576    | 175.499431 | 5           | initial_pool@0;candidate2:lns@12;candidate2:lns@19;candidate2:candidate_vnd@200;candidate1:splitting@1 |
| vnd       | 0    | vnd_split_top_k=5                     | 1560.607088 | 1560.607088    | 113.037263 | 5           | 0;1                                                                                                    |
| vnd       | 0    | vnd_split_top_k=10                    | 1560.607088 | 1560.607088    | 162.624590 | 5           | 0;1                                                                                                    |
| vnd       | 0    | vnd_split_top_k=15                    | 1560.607088 | 1560.607088    | 191.615323 | 5           | 0;1                                                                                                    |
