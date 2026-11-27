# JCTC Pre-Submission Matched-Budget 2x2 Ablation Audit (Mg-MOF-74 / CO2, 298 K, 0.28 bar)

- Total archived runs: **12** (expected 12)
- Model: `mace-matpes-r2scan-0` (SHA-256 `8f147ecffa1d06a696e5648b13b81abda9c96fda9d0fb3faba3cd3fb27d9bba9`), D3 `False`, `N_max=128`, FP32+FP64.
- Shared 1B grid bit-identical: `True`.

## Group Medians & Convergence Pass Counts (0.28 bar)

| Group | Surrogate | REMC | Confirmed Pass | Final Pass | Uptake (mmol/g) | MCSE_eff | ESS_b(N) | ESS_t(N) | ESS_b(U) | ESS_b(OMS) | ESS_t(OMS) | Rhat(N) | ESS_b(N)/GPU-s | t_conv (s) | RT_prod |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 1B-only | OFF | 0/3 | 0/3 | 6.27 [6.22, 6.27] | 0.054 [0.053, 0.067] | 239.6 [238.7, 284.8] | 573.9 [509.7, 639.3] | 213.3 [194.5, 254.7] | 271.0 [266.1, 383.2] | 566.0 [563.2, 590.4] | 1.0156 [1.0147, 1.0195] | 0.00416 [0.00414, 0.00494] | censored (>budget) | 0 [0, 0] |
| B | 1B+2B | OFF | 0/3 | 0/3 | 6.27 [6.16, 6.30] | 0.070 [0.063, 0.080] | 231.9 [184.8, 254.0] | 522.6 [385.8, 576.9] | 200.9 [152.7, 206.6] | 275.5 [222.4, 290.4] | 625.0 [523.8, 646.5] | 1.0189 [1.0180, 1.0316] | 0.00402 [0.00321, 0.00441] | censored (>budget) | 0 [0, 0] |
| C | 1B-only | ON | 1/3 | 1/3 | 6.27 [6.24, 6.27] | 0.039 [0.034, 0.042] | 542.8 [412.2, 708.1] | 1471.4 [1373.8, 1960.4] | 477.8 [357.4, 595.1] | 642.9 [493.5, 699.6] | 1770.9 [1476.2, 2013.9] | 1.0069 [1.0037, 1.0117] | 0.00942 [0.00715, 0.01229] | 4800 [4800, 4800] (1/3) | 37 [29, 41] |
| D | 1B+2B | ON | 1/3 | 2/3 | 6.27 [6.24, 6.27] | 0.038 [0.034, 0.043] | 530.9 [496.4, 606.6] | 2022.5 [1284.7, 2264.1] | 451.2 [418.8, 496.5] | 603.5 [534.3, 621.7] | 2121.7 [1459.1, 2175.7] | 1.0052 [1.0042, 1.0072] | 0.00921 [0.00862, 0.01053] | 6600 [6600, 6600] (1/3) | 41 [34, 43] |

## 2x2 Factorial Ratios (Medians)

- **B_over_A_2B_effect_REMC_OFF**: eff_ratio = 0.968x, ess_ratio = 0.968x
- **D_over_C_2B_effect_REMC_ON**: eff_ratio = 0.978x, ess_ratio = 0.978x
- **C_over_A_REMC_effect_1B_only**: eff_ratio = 2.265x, ess_ratio = 2.265x
- **D_over_B_REMC_effect_1B_2B**: eff_ratio = 2.290x, ess_ratio = 2.289x
- **D_over_A_combined**: eff_ratio = 2.215x, ess_ratio = 2.215x

## Per-Run Details (0.28 bar)

| Run | Cyc (Prod) | Wall (s) | GPU-s | 0.28 Evals | Status | Gates | Failures / Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `rep_2026112601_A` | 46684 (45184) | 7201.8 | 57614 | 186735 | `fail` | 4/11 | N_rhat_max=1.0156>=1.01, N_ess_bulk=238.7<400, U_rhat_max=1.0169>=1.01, U_ess_bulk=213.3<400, OMS_rhat_max=1.0151>=1.01, OMS_ess_bulk=271.0<400, max_chain_drift=0.404>0.250 |
| `rep_2026112601_B` | 46404 (44904) | 7202.9 | 57623 | 185613 | `fail` | 2/11 | N_rhat_max=1.0189>=1.01, N_ess_bulk=184.8<400, N_ess_tail=385.8<400, U_rhat_max=1.0192>=1.01, U_ess_bulk=152.7<400, OMS_rhat_max=1.0172>=1.01, OMS_ess_bulk=222.4<400, MCSE=0.0628>0.0627_mmol_g, max_chain_drift=0.300>0.250 |
| `rep_2026112601_C` | 45986 (44486) | 7202.0 | 57616 | 183945 | `fail` | 7/11 | N_rhat_max=1.0117>=1.01, U_rhat_max=1.0144>=1.01, OMS_rhat_max=1.0116>=1.01, max_chain_drift=0.294>0.250 |
| `rep_2026112601_D` | 46134 (44634) | 7201.5 | 57612 | 184536 | `pass` | 11/11 | NONE (PASSED ALL GATES) |
| `rep_2026112602_A` | 47090 (45590) | 7202.0 | 57616 | 188360 | `fail` | 4/11 | N_rhat_max=1.0195>=1.01, N_ess_bulk=284.8<400, U_rhat_max=1.0214>=1.01, U_ess_bulk=254.7<400, OMS_ess_bulk=383.2<400, MCSE=0.0674>0.0627_mmol_g, max_chain_drift=0.262>0.250 |
| `rep_2026112602_B` | 45812 (44312) | 7201.7 | 57614 | 183250 | `fail` | 3/11 | N_rhat_max=1.0316>=1.01, N_ess_bulk=254.0<400, U_rhat_max=1.0371>=1.01, U_ess_bulk=206.6<400, OMS_rhat_max=1.0249>=1.01, OMS_ess_bulk=290.4<400, MCSE=0.0803>0.0616_mmol_g, max_chain_drift=0.486>0.250 |
| `rep_2026112602_C` | 46648 (45148) | 7201.8 | 57614 | 186593 | `pass` | 11/11 | NONE (PASSED ALL GATES) |
| `rep_2026112602_D` | 44112 (42612) | 7201.5 | 57612 | 176450 | `pass` | 11/11 | NONE (PASSED ALL GATES) |
| `rep_2026112603_A` | 47180 (45680) | 7201.8 | 57615 | 188722 | `fail` | 4/11 | N_rhat_max=1.0147>=1.01, N_ess_bulk=239.6<400, U_rhat_max=1.0172>=1.01, U_ess_bulk=194.5<400, OMS_rhat_max=1.0112>=1.01, OMS_ess_bulk=266.1<400, max_chain_drift=0.333>0.250 |
| `rep_2026112603_B` | 45480 (43980) | 7201.7 | 57614 | 181921 | `fail` | 4/11 | N_rhat_max=1.0180>=1.01, N_ess_bulk=231.9<400, U_rhat_max=1.0207>=1.01, U_ess_bulk=200.9<400, OMS_rhat_max=1.0138>=1.01, OMS_ess_bulk=275.5<400, MCSE=0.0704>0.0630_mmol_g |
| `rep_2026112603_C` | 45084 (43584) | 7201.9 | 57615 | 180338 | `fail` | 10/11 | U_ess_bulk=357.4<400 |
| `rep_2026112603_D` | 45132 (43632) | 7201.7 | 57613 | 180529 | `fail` | 10/11 | max_chain_drift=0.277>0.250 |
