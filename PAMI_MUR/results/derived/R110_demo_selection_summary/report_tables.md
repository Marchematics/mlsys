<!-- generated from /root/icl_ess_threshold/PAMI_MUR/results/raw/R110_demo_selection_20260921_0408 at 2026-09-20T20:46:27Z -->

### Gate (R-MUR q=4 vs anchor-only, Static Utility, Cached Utility)

| benchmark | informative | gain | 95% CI (paired) | hier. 95% CI | d(Static) | CI | d(Cached) | CI | cls>0 | cls>Static | cls>Cached | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cifar10 | yes | 0.0643 | [0.0490, 0.0796] | [0.0345, 0.1032] | 0.0160 | [0.0075, 0.0246] | 0.0210 | [0.0140, 0.0280] | 1.00 | 1.00 | 1.00 | yes |
| cifar100 | yes | 0.5247 | [0.4858, 0.5636] | [0.4500, 0.6038] | -0.0522 | [-0.0758, -0.0285] | -0.0133 | [-0.0365, 0.0098] | 0.99 | 0.26 | 0.35 | no |
| dtd | yes | 0.3322 | [0.2721, 0.3923] | [0.2372, 0.4353] | -0.2658 | [-0.3388, -0.1929] | -0.3156 | [-0.3865, -0.2446] | 0.96 | 0.09 | 0.19 | no |
| eurosat | yes | 0.1294 | [0.1080, 0.1509] | [0.0763, 0.1870] | 0.0221 | [0.0090, 0.0351] | 0.0169 | [0.0057, 0.0282] | 1.00 | 0.80 | 0.80 | yes |
| svhn | yes | 0.8809 | [0.8328, 0.9291] | [0.7512, 1.0382] | 0.1279 | [0.0647, 0.1911] | 0.1121 | [0.0637, 0.1605] | 1.00 | 0.30 | 0.80 | yes |

### Predictor quality (frozen in-context classifier, test episodes)

| benchmark | demos | accuracy | CE |
|---|---|---|---|
| cifar10 | demos_0 | 0.9660 | 0.2048 |
| cifar10 | demos_16_pool | 0.9836 | 0.1006 |
| cifar10 | demos_2_relevance | 0.9777 | 0.1382 |
| cifar10 | demos_4_farthest | 0.9651 | 0.2172 |
| cifar10 | demos_4_random | 0.9758 | 0.1513 |
| cifar10 | demos_4_relevance | 0.9831 | 0.1073 |
| cifar100 | demos_0 | 0.8120 | 0.6761 |
| cifar100 | demos_16_pool | 0.9981 | 0.0098 |
| cifar100 | demos_2_relevance | 0.9886 | 0.0478 |
| cifar100 | demos_4_farthest | 0.7953 | 0.7742 |
| cifar100 | demos_4_random | 0.9702 | 0.1157 |
| cifar100 | demos_4_relevance | 0.9969 | 0.0160 |
| dtd | demos_0 | 0.7282 | 1.3487 |
| dtd | demos_16_pool | 0.9489 | 0.2203 |
| dtd | demos_2_relevance | 0.8729 | 0.5859 |
| dtd | demos_4_farthest | 0.7214 | 1.4021 |
| dtd | demos_4_random | 0.8490 | 0.7092 |
| dtd | demos_4_relevance | 0.9214 | 0.3379 |
| eurosat | demos_0 | 0.9457 | 0.2490 |
| eurosat | demos_16_pool | 0.9883 | 0.0575 |
| eurosat | demos_2_relevance | 0.9762 | 0.1117 |
| eurosat | demos_4_farthest | 0.9358 | 0.3109 |
| eurosat | demos_4_random | 0.9711 | 0.1390 |
| eurosat | demos_4_relevance | 0.9863 | 0.0697 |
| svhn | demos_0 | 0.6932 | 0.9537 |
| svhn | demos_16_pool | 0.9971 | 0.0126 |
| svhn | demos_2_relevance | 0.9789 | 0.0836 |
| svhn | demos_4_farthest | 0.5454 | 1.7046 |
| svhn | demos_4_random | 0.9290 | 0.2351 |
| svhn | demos_4_relevance | 0.9956 | 0.0208 |

### Headroom and baselines (mean query-batch CE reduction)

| benchmark | anchor-only CE | oracle_greedy | oracle_static | pool_all | relevance | random | H_state |
|---|---|---|---|---|---|---|---|
| cifar10 | 0.2048 | 0.0981 | 0.0980 | 0.1042 | 0.0975 | 0.0535 | 0.000 | yes |
| cifar100 | 0.6761 | 0.6606 | 0.6606 | 0.6664 | 0.6602 | 0.5605 | 0.000 | yes |
| dtd | 1.3487 | 1.0141 | 1.0140 | 1.1284 | 1.0108 | 0.6395 | 0.000 | yes |
| eurosat | 0.2490 | 0.1801 | 0.1800 | 0.1915 | 0.1793 | 0.1099 | 0.000 | yes |
| svhn | 0.9537 | 0.9354 | 0.9353 | 0.9411 | 0.9329 | 0.7186 | 0.000 | yes |

### All policies (mean gain per benchmark)

| benchmark | base_only | pool_all | random | relevance | mmr | facility_location | dpp | static_utility | cached_mur | oracle_static | oracle_greedy | ranked_response_q2 | ranked_response_q4 | ranked_response_q8 | ranked_response_full |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cifar10 | 0.0000 | 0.1042 | 0.0535 | 0.0975 | 0.0974 | 0.0976 | 0.0836 | 0.0482 | 0.0433 | 0.0980 | 0.0981 | 0.0396 | 0.0643 | 0.0716 | 0.0711 |
| cifar100 | 0.0000 | 0.6664 | 0.5605 | 0.6602 | 0.6600 | 0.6597 | 0.6511 | 0.5769 | 0.5381 | 0.6606 | 0.6606 | 0.4131 | 0.5247 | 0.5714 | 0.5660 |
| dtd | 0.0000 | 1.1284 | 0.6395 | 1.0108 | 1.0057 | 1.0045 | 0.9947 | 0.5981 | 0.6478 | 1.0140 | 1.0141 | 0.2242 | 0.3322 | 0.3943 | 0.4272 |
| eurosat | 0.0000 | 0.1915 | 0.1099 | 0.1793 | 0.1791 | 0.1795 | 0.1320 | 0.1074 | 0.1125 | 0.1800 | 0.1801 | 0.1008 | 0.1294 | 0.1452 | 0.1456 |
| svhn | 0.0000 | 0.9411 | 0.7186 | 0.9329 | 0.9324 | 0.9322 | 0.9250 | 0.7530 | 0.7688 | 0.9353 | 0.9354 | 0.7698 | 0.8809 | 0.8927 | 0.8861 |

### Compute (mean predictor rows per cell)

| benchmark | static_utility | cached_mur | oracle_static | oracle_greedy | ranked_response_q2 | ranked_response_q4 | ranked_response_q8 | ranked_response_full | evaluation_rows |
|---|---|---|---|---|---|---|---|---|---|
| cifar10 | 0 | 0 | 1360 | 4961 | 960 | 1600 | 2880 | 4966 | 1200 |
| cifar100 | 0 | 0 | 3400 | 12401 | 2400 | 4000 | 7200 | 12679 | 3000 |
| dtd | 0 | 0 | 1598 | 5828 | 1128 | 1880 | 3384 | 6104 | 1410 |
| eurosat | 0 | 0 | 1360 | 4960 | 960 | 1600 | 2880 | 4974 | 1200 |
| svhn | 0 | 0 | 1360 | 4960 | 960 | 1600 | 2880 | 4981 | 1200 |

### Per-seed gate (R-MUR q=4 gain)

| benchmark | seed | gain |
|---|---|---|
| cifar10 | 101 | 0.0720 |
| cifar10 | 202 | 0.0661 |
| cifar10 | 303 | 0.0548 |
| cifar100 | 101 | 0.5543 |
| cifar100 | 202 | 0.5167 |
| cifar100 | 303 | 0.5032 |
| dtd | 101 | 0.3473 |
| dtd | 202 | 0.4132 |
| dtd | 303 | 0.2362 |
| eurosat | 101 | 0.1158 |
| eurosat | 202 | 0.1394 |
| eurosat | 303 | 0.1331 |
| svhn | 101 | 0.9100 |
| svhn | 202 | 0.8976 |
| svhn | 303 | 0.8351 |
