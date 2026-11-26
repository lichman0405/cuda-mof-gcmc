# JCTC v2 Simulation & Reproducibility Audit Report

- **Source directory**: `/share/SL/cuda-mof-gcmc/build/jctc_v2_prod`
- **Traces archive**: `jctc_traces.npz` (SHA-256: `e507cb2435a5a826...`)
- **Runs / Points**: 23 runs, 76 pressure points (**PASS**: 41, **FAIL**: 34, **INSUFFICIENT**: 1, **UNKNOWN**: 0)
- **`model_sha256` completeness**: PASS
- **2B deployed hold-out completeness**: PASS
- **`max_mol` saturation safety**: PASS
- **Full-trace recomputation parity**: PASS

## 1. Per-Point Convergence, Thermodynamics, Efficiency & Safety

| Run | System | Proposer / Mode | Model | P (bar) | Uptake (mmol/g) | MCSE | Qst fluc (kJ/mol) | R-hat (N / U / OMS) | ESS_bulk(N) | ESS/GPU-s | t(MCSE=0.1) s (samp/amort/cold) | Ins/Del Sym | maxN/cap | Status |
|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---|---:|---|---|
| `exp1_mp0_nod3_0p28` | Mg-MOF-74 + CO2 | auto_mlip | `medium` | 0.28 | 0.646 ± 0.015 | 0.0149 | 26.4 ± 0.1 | 1.004 / 1.004 / 1.004 | 2010 | 2.646 | 17 / 60 / 60 | 1.20 | 12/128 (OK) | **PASS** |
| `jctc_exp1_auto_mlip_1b_only_r2scan` | Mg-MOF-74 + CO2 | auto_mlip (1B) | `mace-matpes-r2scan-0` | 0.012 | 0.179 ± 0.006 | 0.0057 | 34.8 ± 0.2 | 1.002 / 1.003 / 1.004 | 2457 | 8.384 | 1 / 14 / 41 | 0.98 | 6/128 (OK) | **PASS** |
| `jctc_exp1_auto_mlip_1b_only_r2scan` | Mg-MOF-74 + CO2 | auto_mlip (1B) | `mace-matpes-r2scan-0` | 0.28 | 6.300 ± 0.167 | 0.1667 | 39.5 ± 1.1 | 1.114 / 1.178 / 1.090 | 30 | 0.035 | 2412 / 2425 / 2452 | 0.62 | 36/128 (OK) | **FAIL** |
| `jctc_exp1_auto_mlip_1b_only_r2scan` | Mg-MOF-74 + CO2 | auto_mlip (1B) | `mace-matpes-r2scan-0` | 1 | 8.188 ± 0.062 | 0.0683 | 32.7 ± 1.3 | 1.066 / 1.178 / 1.112 | 46 | 0.048 | 446 / 460 / 487 | 0.88 | 43/128 (OK) | **FAIL** |
| `jctc_exp1_auto_mlip_r2scan_highload_0p28` | Mg-MOF-74 + CO2 | auto_mlip | `mace-matpes-r2scan-0` | 0.28 | 6.326 ± 0.112 | 0.1125 | 38.7 ± 0.8 | 1.057 / 1.082 / 1.113 | 76 | 0.095 | 1011 / 1054 / 1054 | 0.65 | 37/128 (OK) | **FAIL** |
| `jctc_exp1_auto_mlip_r2scan_no_remc` | Mg-MOF-74 + CO2 | auto_mlip | `mace-matpes-r2scan-0` | 0.012 | 0.178 ± 0.001 | 0.0043 | 34.9 ± 0.2 | 1.000 / 1.000 / 1.000 | 2369 | 2.699 | 2 / 16 / 45 | 1.07 | 7/128 (OK) | **PASS** |
| `jctc_exp1_auto_mlip_r2scan_no_remc` | Mg-MOF-74 + CO2 | auto_mlip | `mace-matpes-r2scan-0` | 0.28 | 6.346 ± 0.192 | 0.2130 | 39.9 ± 1.9 | 1.206 / 1.299 / 1.205 | 14 | 0.009 | 7343 / 7357 / 7386 | 1.04 | 38/128 (OK) | **FAIL** |
| `jctc_exp1_auto_mlip_r2scan_no_remc` | Mg-MOF-74 + CO2 | auto_mlip | `mace-matpes-r2scan-0` | 1 | 8.187 ± 0.029 | 0.0636 | 35.2 ± 2.2 | 1.061 / 1.177 / 1.147 | 57 | 0.058 | 401 / 416 / 445 | 1.14 | 43/128 (OK) | **FAIL** |
| `jctc_exp1_auto_mlip_r2scan_remc` | Mg-MOF-74 + CO2 | auto_mlip (REMC) | `mace-matpes-r2scan-0` | 0.012 | 0.179 ± 0.001 | 0.0014 | 34.7 ± 0.0 | 1.000 / 1.000 / 1.000 | 24444 | 3.432 | 1 / 7 / 45 | 0.94 | 8/128 (OK) | **PASS** |
| `jctc_exp1_auto_mlip_r2scan_remc` | Mg-MOF-74 + CO2 | auto_mlip (REMC) | `mace-matpes-r2scan-0` | 0.04 | 0.665 ± 0.003 | 0.0050 | 35.2 ± 0.0 | 1.000 / 1.000 / 1.000 | 7196 | 1.010 | 18 / 23 / 61 | 0.99 | 13/128 (OK) | **PASS** |
| `jctc_exp1_auto_mlip_r2scan_remc` | Mg-MOF-74 + CO2 | auto_mlip (REMC) | `mace-matpes-r2scan-0` | 0.08 | 1.528 ± 0.011 | 0.0155 | 36.2 ± 0.1 | 1.001 / 1.002 / 1.002 | 1971 | 0.277 | 171 / 176 / 214 | 0.91 | 22/128 (OK) | **PASS** |
| `jctc_exp1_auto_mlip_r2scan_remc` | Mg-MOF-74 + CO2 | auto_mlip (REMC) | `mace-matpes-r2scan-0` | 0.16 | 3.909 ± 0.075 | 0.0747 | 37.9 ± 0.1 | 1.015 / 1.015 / 1.013 | 412 | 0.058 | 3970 / 3975 / 4013 | 0.94 | 34/128 (OK) | **FAIL** |
| `jctc_exp1_auto_mlip_r2scan_remc` | Mg-MOF-74 + CO2 | auto_mlip (REMC) | `mace-matpes-r2scan-0` | 0.22 | 5.401 ± 0.061 | 0.0606 | 38.8 ± 0.1 | 1.017 / 1.019 / 1.015 | 412 | 0.058 | 2612 / 2618 / 2656 | 0.87 | 37/128 (OK) | **FAIL** |
| `jctc_exp1_auto_mlip_r2scan_remc` | Mg-MOF-74 + CO2 | auto_mlip (REMC) | `mace-matpes-r2scan-0` | 0.28 | 6.222 ± 0.041 | 0.0405 | 39.0 ± 0.2 | 1.014 / 1.018 / 1.014 | 446 | 0.063 | 1170 / 1175 / 1213 | 0.98 | 39/128 (OK) | **FAIL** |
| `jctc_exp1_auto_mlip_r2scan_remc` | Mg-MOF-74 + CO2 | auto_mlip (REMC) | `mace-matpes-r2scan-0` | 0.55 | 7.592 ± 0.028 | 0.0281 | 37.4 ± 0.2 | 1.014 / 1.021 / 1.007 | 583 | 0.082 | 561 / 566 / 604 | 1.12 | 41/128 (OK) | **FAIL** |
| `jctc_exp1_auto_mlip_r2scan_remc` | Mg-MOF-74 + CO2 | auto_mlip (REMC) | `mace-matpes-r2scan-0` | 1 | 8.187 ± 0.017 | 0.0174 | 33.2 ± 0.5 | 1.007 / 1.016 / 1.006 | 720 | 0.101 | 215 / 220 / 258 | 1.07 | 52/128 (OK) | **FAIL** |
| `jctc_exp1_direct_r2scan_0p28` | Mg-MOF-74 + CO2 | direct | `mace-matpes-r2scan-0` | 0.28 | 1.165 ± 0.210 | 0.2897 | 32.2 ± 0.7 | 1.777 / 1.857 / 1.736 | 6 | 0.004 | 12591 / 12591 / 12591 | 0.96 | 14/128 (OK) | **FAIL** |
| `jctc_exp1_grid_only_mp0d3` | Mg-MOF-74 + CO2 | auto_mlip (grid) | `medium+D3` | 0.012 | 15.103 ± 0.215 | 0.2582 | 48.8 ± 2.0 | 1.310 / 1.402 / 1.060 | 10 | 0.028 | 2424 / 2441 / 2474 | 1.00 | 79/160 (OK) | **FAIL** |
| `jctc_exp1_grid_only_mp0d3` | Mg-MOF-74 + CO2 | auto_mlip (grid) | `medium+D3` | 0.28 | 19.626 ± 0.096 | 0.1419 | 58.2 ± 9.8 | 1.297 / 1.500 / 1.232 | 10 | 0.022 | 949 / 966 / 999 | 1.02 | 91/160 (OK) | **FAIL** |
| `jctc_exp1_grid_only_mp0d3` | Mg-MOF-74 + CO2 | auto_mlip (grid) | `medium+D3` | 1 | 20.589 ± 0.099 | 0.1893 | 67.8 ± 8.6 | 1.580 / 1.705 / 1.325 | 7 | 0.014 | 1764 / 1781 / 1815 | 1.04 | 95/160 (OK) | **FAIL** |
| `jctc_exp1_grid_only_r2scan` | Mg-MOF-74 + CO2 | auto_mlip (grid) | `mace-matpes-r2scan-0` | 0.012 | 0.116 ± 0.002 | 0.0023 | 33.3 ± 0.1 | 1.001 / 1.001 / 1.001 | 5186 | 396.295 | 0 / 5 / 44 | 0.98 | 5/128 (OK) | **PASS** |
| `jctc_exp1_grid_only_r2scan` | Mg-MOF-74 + CO2 | auto_mlip (grid) | `mace-matpes-r2scan-0` | 0.04 | 0.386 ± 0.005 | 0.0067 | 33.4 ± 0.1 | 1.001 / 1.001 / 1.001 | 2039 | 90.509 | 0 / 6 / 44 | 0.99 | 8/128 (OK) | **PASS** |
| `jctc_exp1_grid_only_r2scan` | Mg-MOF-74 + CO2 | auto_mlip (grid) | `mace-matpes-r2scan-0` | 0.08 | 0.790 ± 0.011 | 0.0144 | 33.5 ± 0.2 | 1.004 / 1.005 / 1.004 | 892 | 27.466 | 1 / 6 / 44 | 1.01 | 14/128 (OK) | **PASS** |
| `jctc_exp1_grid_only_r2scan` | Mg-MOF-74 + CO2 | auto_mlip (grid) | `mace-matpes-r2scan-0` | 0.16 | 1.550 ± 0.013 | 0.0336 | 34.4 ± 0.2 | 1.011 / 1.012 / 1.012 | 334 | 6.863 | 6 / 11 / 49 | 1.00 | 17/128 (OK) | **FAIL** |
| `jctc_exp1_grid_only_r2scan` | Mg-MOF-74 + CO2 | auto_mlip (grid) | `mace-matpes-r2scan-0` | 0.22 | 2.081 ± 0.014 | 0.0409 | 34.4 ± 0.2 | 1.014 / 1.015 / 1.016 | 288 | 4.926 | 10 / 15 / 53 | 1.00 | 21/128 (OK) | **FAIL** |
| `jctc_exp1_grid_only_r2scan` | Mg-MOF-74 + CO2 | auto_mlip (grid) | `mace-matpes-r2scan-0` | 0.28 | 2.703 ± 0.029 | 0.0444 | 34.2 ± 0.3 | 1.008 / 1.011 / 1.011 | 261 | 3.720 | 14 / 19 / 57 | 1.01 | 23/128 (OK) | **FAIL** |
| `jctc_exp1_grid_only_r2scan` | Mg-MOF-74 + CO2 | auto_mlip (grid) | `mace-matpes-r2scan-0` | 0.55 | 4.179 ± 0.028 | 0.0590 | 33.5 ± 0.3 | 1.035 / 1.041 / 1.029 | 176 | 1.810 | 34 / 39 / 77 | 1.00 | 32/128 (OK) | **FAIL** |
| `jctc_exp1_grid_only_r2scan` | Mg-MOF-74 + CO2 | auto_mlip (grid) | `mace-matpes-r2scan-0` | 1 | 5.919 ± 0.059 | 0.0639 | 32.7 ± 0.4 | 1.027 / 1.045 / 1.042 | 151 | 1.168 | 53 / 58 / 96 | 1.00 | 38/128 (OK) | **FAIL** |
| `jctc_exp1_mercado_r2scan` | Mg-MOF-74 + CO2 | mercado | `mace-matpes-r2scan-0` | 0.28 | 6.044 ± 0.233 | 0.2947 | 38.8 ± 0.6 | 1.301 / 1.345 / 1.325 | 10 | 0.014 | 6421 / 6421 / 6421 | 0.55 | 35/128 (OK) | **FAIL** |
| `jctc_exp1_uff_r2scan` | Mg-MOF-74 + CO2 | uff | `mace-matpes-r2scan-0` | 0.28 | 3.157 ± 0.135 | 0.2515 | 35.0 ± 1.9 | 1.344 / 1.461 / 1.445 | 9 | 0.016 | 3581 / 3581 / 3581 | 0.39 (!) | 22/128 (OK) | **FAIL** |
| `jctc_finitesize_mgmof74_1x1x4_co2_r2scan_0p28` | Mg-MOF-74 + CO2 | auto_mlip | `mace-matpes-r2scan-0` | 0.28 | 6.028 ± 0.108 | 0.1079 | 37.6 ± 0.8 | 1.091 / 1.117 / 1.131 | 30 | 0.015 | 2223 / 2282 / 2282 | 0.82 | 66/256 (OK) | **FAIL** |
| `jctc_macedac_cubtc_co2` | Cu-BTC + CO2 | auto_mlip (REMC) | `build/models/mace-dac-1.model+D3` | 0.05 | 0.799 ± 0.001 | 0.0012 | 28.6 ± 0.1 | 1.000 / 1.000 / 1.000 | 10658 | 3.125 | 1 / 53 / 212 | 0.99 | 14/128 (OK) | **PASS** |
| `jctc_macedac_cubtc_co2` | Cu-BTC + CO2 | auto_mlip (REMC) | `build/models/mace-dac-1.model+D3` | 0.15 | 0.996 ± 0.002 | 0.0018 | 21.9 ± 0.1 | 1.000 / 1.000 / 1.000 | 7989 | 2.342 | 1 / 54 / 213 | 0.98 | 17/128 (OK) | **PASS** |
| `jctc_macedac_cubtc_co2` | Cu-BTC + CO2 | auto_mlip (REMC) | `build/models/mace-dac-1.model+D3` | 0.5 | 1.467 ± 0.005 | 0.0053 | 19.6 ± 0.1 | 1.001 / 1.002 / 1.000 | 4468 | 1.310 | 10 / 63 / 221 | 0.97 | 28/128 (OK) | **PASS** |
| `jctc_macedac_cubtc_co2` | Cu-BTC + CO2 | auto_mlip (REMC) | `build/models/mace-dac-1.model+D3` | 1 | 2.049 ± 0.010 | 0.0101 | 19.2 ± 0.1 | 1.002 / 1.003 / 1.001 | 2303 | 0.675 | 35 / 88 / 247 | 1.07 | 34/128 (OK) | **PASS** |
| `jctc_macedac_cubtc_h2o` | Cu-BTC + H2O | auto_mlip (REMC) | `build/models/mace-dac-1.model+D3` | 0.0005 | 0.001 ± 0.000 | 0.0001 | 21.5 ± 0.3 | 1.000 / 1.000 / 1.000 | 22939 | 64.192 | 0 / 71 / 213 | 0.72 | 2/256 (OK) | **FAIL** |
| `jctc_macedac_cubtc_h2o` | Cu-BTC + H2O | auto_mlip (REMC) | `build/models/mace-dac-1.model+D3` | 0.002 | 0.005 ± 0.000 | 0.0002 | 21.6 ± 0.1 | 1.000 / 1.000 / 1.000 | 28183 | 78.869 | 0 / 71 / 213 | 1.07 | 2/256 (OK) | **FAIL** |
| `jctc_macedac_cubtc_h2o` | Cu-BTC + H2O | auto_mlip (REMC) | `build/models/mace-dac-1.model+D3` | 0.008 | 0.022 ± 0.001 | 0.0007 | 21.8 ± 0.2 | 1.000 / 1.000 / 1.001 | 7553 | 21.137 | 0 / 71 / 213 | 0.87 | 3/256 (OK) | **INSUFFICIENT** |
| `jctc_macedac_irmof1_co2` | IRMOF-1 + CO2 | auto_mlip (REMC) | `build/models/mace-dac-1.model+D3` | 0.5 | 0.463 ± 0.001 | 0.0017 | 13.5 ± 0.0 | 1.000 / 1.000 / N/A | 25393 | 11.325 | 1 / 38 / 151 | 0.93 | 12/128 (OK) | **PASS** |
| `jctc_macedac_irmof1_co2` | IRMOF-1 + CO2 | auto_mlip (REMC) | `build/models/mace-dac-1.model+D3` | 1 | 0.910 ± 0.001 | 0.0033 | 13.6 ± 0.0 | 1.000 / 1.000 / N/A | 14102 | 6.289 | 2 / 40 / 153 | 0.95 | 18/128 (OK) | **PASS** |
| `jctc_macedac_irmof1_co2` | IRMOF-1 + CO2 | auto_mlip (REMC) | `build/models/mace-dac-1.model+D3` | 2.5 | 2.190 ± 0.006 | 0.0075 | 13.7 ± 0.0 | 1.001 / 1.001 / N/A | 6193 | 2.762 | 13 / 50 / 163 | 1.05 | 29/128 (OK) | **PASS** |
| `jctc_macedac_irmof1_co2` | IRMOF-1 + CO2 | auto_mlip (REMC) | `build/models/mace-dac-1.model+D3` | 5 | 4.032 ± 0.022 | 0.0220 | 13.9 ± 0.0 | 1.002 / 1.002 / N/A | 3045 | 1.358 | 109 / 146 / 259 | 0.94 | 44/128 (OK) | **PASS** |
| `jctc_macedac_mgmof74_co2` | Mg-MOF-74 + CO2 | auto_mlip (REMC) | `build/models/mace-dac-1.model+D3` | 0.0002 | 0.568 ± 0.003 | 0.0040 | 47.5 ± 0.1 | 1.001 / 1.001 / 1.001 | 7332 | 1.197 | 10 / 18 / 60 | 1.01 | 9/160 (OK) | **PASS** |
| `jctc_macedac_mgmof74_co2` | Mg-MOF-74 + CO2 | auto_mlip (REMC) | `build/models/mace-dac-1.model+D3` | 0.003 | 3.400 ± 0.020 | 0.0196 | 46.3 ± 0.1 | 1.003 / 1.003 / 1.003 | 1661 | 0.271 | 236 / 244 / 287 | 1.01 | 26/160 (OK) | **PASS** |
| `jctc_macedac_mgmof74_co2` | Mg-MOF-74 + CO2 | auto_mlip (REMC) | `build/models/mace-dac-1.model+D3` | 0.012 | 5.510 ± 0.012 | 0.0166 | 45.0 ± 0.2 | 1.005 / 1.006 / 1.005 | 1147 | 0.187 | 168 / 177 / 219 | 1.00 | 34/160 (OK) | **PASS** |
| `jctc_macedac_mgmof74_co2` | Mg-MOF-74 + CO2 | auto_mlip (REMC) | `build/models/mace-dac-1.model+D3` | 0.08 | 7.485 ± 0.019 | 0.0185 | 44.0 ± 0.4 | 1.006 / 1.006 / 1.005 | 1210 | 0.197 | 210 / 218 / 261 | 1.04 | 38/160 (OK) | **PASS** |
| `jctc_macedac_mgmof74_co2` | Mg-MOF-74 + CO2 | auto_mlip (REMC) | `build/models/mace-dac-1.model+D3` | 0.28 | 8.059 ± 0.006 | 0.0059 | 38.7 ± 0.5 | 1.001 / 1.005 / 1.003 | 3260 | 0.532 | 22 / 30 / 72 | 0.95 | 39/160 (OK) | **PASS** |
| `jctc_macedac_mgmof74_co2` | Mg-MOF-74 + CO2 | auto_mlip (REMC) | `build/models/mace-dac-1.model+D3` | 1 | 8.375 ± 0.003 | 0.0041 | 23.4 ± 0.5 | 1.001 / 1.003 / 1.004 | 3685 | 0.602 | 10 / 19 / 61 | 1.16 | 42/160 (OK) | **PASS** |
| `jctc_macedac_nimof74_co2` | Ni-MOF-74 + CO2 | auto_mlip (REMC) | `build/models/mace-dac-1.model+D3` | 0.012 | 0.055 ± 0.000 | 0.0005 | 22.3 ± 0.0 | 1.000 / 1.000 / 1.000 | 38723 | 18.397 | 0 / 13 / 51 | 0.97 | 5/128 (OK) | **PASS** |
| `jctc_macedac_nimof74_co2` | Ni-MOF-74 + CO2 | auto_mlip (REMC) | `build/models/mace-dac-1.model+D3` | 0.08 | 0.345 ± 0.002 | 0.0021 | 22.4 ± 0.1 | 1.000 / 1.000 / 1.000 | 21110 | 10.029 | 1 / 14 / 52 | 1.04 | 10/128 (OK) | **PASS** |
| `jctc_macedac_nimof74_co2` | Ni-MOF-74 + CO2 | auto_mlip (REMC) | `build/models/mace-dac-1.model+D3` | 0.28 | 1.069 ± 0.005 | 0.0053 | 22.4 ± 0.1 | 1.000 / 1.001 / 1.001 | 8776 | 4.169 | 6 / 19 / 57 | 1.19 | 15/128 (OK) | **PASS** |
| `jctc_macedac_nimof74_co2` | Ni-MOF-74 + CO2 | auto_mlip (REMC) | `build/models/mace-dac-1.model+D3` | 1 | 2.671 ± 0.010 | 0.0103 | 22.4 ± 0.1 | 1.002 / 1.003 / 1.002 | 3323 | 1.578 | 22 / 35 / 73 | 1.06 | 27/128 (OK) | **PASS** |
| `jctc_parity_direct_irmof1_co2_1bar` | IRMOF-1 + CO2 | direct | `mace-matpes-r2scan-0` | 1 | 0.863 ± 0.064 | 0.0802 | 16.2 ± 0.5 | 1.110 / 1.161 / N/A | 25 | 0.009 | 1737 / 1737 / 1737 | 0.99 | 15/128 (OK) | **FAIL** |
| `jctc_parity_nested_irmof1_co2_1bar` | IRMOF-1 + CO2 | auto_mlip | `mace-matpes-r2scan-0` | 1 | 0.874 ± 0.011 | 0.0112 | 16.3 ± 0.2 | 1.005 / 1.007 / N/A | 1271 | 0.480 | 33 / 179 / 179 | 0.91 | 19/128 (OK) | **PASS** |
| `jctc_remc_cubtc_co2_r2scan` | Cu-BTC + CO2 | auto_mlip (REMC) | `mace-matpes-r2scan-0` | 0.05 | 0.027 ± 0.000 | 0.0005 | 17.2 ± 0.1 | 1.000 / 1.000 / 1.000 | 25047 | 8.945 | 0 / 52 / 208 | 0.96 | 4/128 (OK) | **PASS** |
| `jctc_remc_cubtc_co2_r2scan` | Cu-BTC + CO2 | auto_mlip (REMC) | `mace-matpes-r2scan-0` | 0.15 | 0.082 ± 0.001 | 0.0008 | 17.3 ± 0.1 | 1.000 / 1.000 / 1.000 | 13418 | 4.792 | 0 / 52 / 208 | 0.95 | 8/128 (OK) | **PASS** |
| `jctc_remc_cubtc_co2_r2scan` | Cu-BTC + CO2 | auto_mlip (REMC) | `mace-matpes-r2scan-0` | 0.5 | 0.310 ± 0.005 | 0.0053 | 17.9 ± 0.1 | 1.004 / 1.004 / 1.001 | 1934 | 0.691 | 8 / 60 / 216 | 1.05 | 12/128 (OK) | **PASS** |
| `jctc_remc_cubtc_co2_r2scan` | Cu-BTC + CO2 | auto_mlip (REMC) | `mace-matpes-r2scan-0` | 1 | 0.680 ± 0.008 | 0.0110 | 18.4 ± 0.1 | 1.006 / 1.005 / 1.002 | 680 | 0.243 | 34 / 86 / 242 | 1.07 | 18/128 (OK) | **PASS** |
| `jctc_remc_irmof1_ch4_mp0_d3` | IRMOF-1 + CH4 | auto_mlip (REMC) | `medium+D3` | 0.5 | 2.528 ± 0.009 | 0.0089 | 21.3 ± 0.1 | 1.001 / 1.001 / 1.000 | 3928 | 0.922 | 34 / 70 / 181 | 0.82 | 28/128 (OK) | **PASS** |
| `jctc_remc_irmof1_ch4_mp0_d3` | IRMOF-1 + CH4 | auto_mlip (REMC) | `medium+D3` | 1 | 3.595 ± 0.012 | 0.0120 | 19.8 ± 0.2 | 1.002 / 1.001 / 1.001 | 2351 | 0.552 | 61 / 98 / 209 | 1.00 | 38/128 (OK) | **PASS** |
| `jctc_remc_irmof1_ch4_mp0_d3` | IRMOF-1 + CH4 | auto_mlip (REMC) | `medium+D3` | 2.5 | 5.497 ± 0.012 | 0.0218 | 17.5 ± 0.2 | 1.005 / 1.004 / 1.002 | 878 | 0.206 | 202 / 239 / 349 | 1.04 | 52/128 (OK) | **PASS** |
| `jctc_remc_irmof1_ch4_mp0_d3` | IRMOF-1 + CH4 | auto_mlip (REMC) | `medium+D3` | 5 | 7.536 ± 0.066 | 0.0680 | 16.6 ± 0.2 | 1.036 / 1.024 / 1.001 | 133 | 0.031 | 1968 / 2005 / 2115 | 0.70 | 63/128 (OK) | **FAIL** |
| `jctc_remc_irmof1_co2_r2scan` | IRMOF-1 + CO2 | auto_mlip (REMC) | `mace-matpes-r2scan-0` | 0.5 | 0.415 ± 0.004 | 0.0042 | 15.9 ± 0.0 | 1.001 / 1.001 / N/A | 9002 | 2.278 | 7 / 43 / 152 | 0.78 | 12/128 (OK) | **PASS** |
| `jctc_remc_irmof1_co2_r2scan` | IRMOF-1 + CO2 | auto_mlip (REMC) | `mace-matpes-r2scan-0` | 1 | 0.897 ± 0.009 | 0.0088 | 16.1 ± 0.1 | 1.001 / 1.002 / N/A | 2339 | 0.592 | 31 / 67 / 176 | 0.94 | 18/128 (OK) | **PASS** |
| `jctc_remc_irmof1_co2_r2scan` | IRMOF-1 + CO2 | auto_mlip (REMC) | `mace-matpes-r2scan-0` | 2.5 | 2.287 ± 0.029 | 0.0287 | 16.7 ± 0.2 | 1.009 / 1.012 / N/A | 667 | 0.169 | 326 / 363 / 472 | 1.00 | 34/128 (OK) | **FAIL** |
| `jctc_remc_irmof1_co2_r2scan` | IRMOF-1 + CO2 | auto_mlip (REMC) | `mace-matpes-r2scan-0` | 5 | 3.846 ± 0.036 | 0.0361 | 16.0 ± 0.2 | 1.013 / 1.009 / N/A | 366 | 0.093 | 516 / 552 / 661 | 0.87 | 41/128 (OK) | **FAIL** |
| `jctc_remc_mgmof74_co2_mp0_d3` | Mg-MOF-74 + CO2 | auto_mlip (REMC) | `medium+D3` | 0.0002 | 1.626 ± 0.025 | 0.0248 | 49.0 ± 0.1 | 1.004 / 1.004 / 1.004 | 905 | 0.182 | 306 / 314 / 356 | 1.10 | 20/160 (OK) | **PASS** |
| `jctc_remc_mgmof74_co2_mp0_d3` | Mg-MOF-74 + CO2 | auto_mlip (REMC) | `medium+D3` | 0.002 | 7.590 ± 0.047 | 0.0700 | 45.1 ± 1.0 | 1.024 / 1.028 / 1.039 | 163 | 0.033 | 2438 / 2446 / 2488 | 0.88 | 46/160 (OK) | **FAIL** |
| `jctc_remc_mgmof74_co2_mp0_d3` | Mg-MOF-74 + CO2 | auto_mlip (REMC) | `medium+D3` | 0.012 | 15.380 ± 0.206 | 0.2059 | 46.1 ± 1.8 | 1.121 / 1.133 / 1.074 | 23 | 0.005 | 21115 / 21123 / 21165 | 0.67 | 77/160 (OK) | **FAIL** |
| `jctc_remc_mgmof74_co2_mp0_d3` | Mg-MOF-74 + CO2 | auto_mlip (REMC) | `medium+D3` | 0.06 | 17.298 ± 0.106 | 0.1103 | 49.7 ± 2.3 | 1.190 / 1.250 / 1.078 | 16 | 0.003 | 6057 / 6065 / 6107 | 1.16 | 84/160 (OK) | **FAIL** |
| `jctc_remc_mgmof74_co2_mp0_d3` | Mg-MOF-74 + CO2 | auto_mlip (REMC) | `medium+D3` | 0.28 | 18.276 ± 0.088 | 0.1190 | 41.5 ± 2.7 | 1.222 / 1.193 / 1.045 | 13 | 0.003 | 7049 / 7057 / 7099 | 1.36 | 86/160 (OK) | **FAIL** |
| `jctc_remc_mgmof74_co2_mp0_d3` | Mg-MOF-74 + CO2 | auto_mlip (REMC) | `medium+D3` | 1 | 18.916 ± 0.019 | 0.0478 | 36.6 ± 6.9 | 1.073 / 1.045 / 1.123 | 42 | 0.008 | 1139 / 1147 / 1189 | 1.41 | 88/160 (OK) | **FAIL** |
| `jctc_remc_nimof74_co2_r2scan` | Ni-MOF-74 + CO2 | auto_mlip (REMC) | `mace-matpes-r2scan-0` | 0.012 | 0.007 ± 0.000 | 0.0004 | 24.9 ± 0.3 | 1.000 / 1.000 / 1.000 | 26072 | 19.367 | 0 / 11 / 44 | 0.83 | 2/128 (OK) | **FAIL** |
| `jctc_remc_nimof74_co2_r2scan` | Ni-MOF-74 + CO2 | auto_mlip (REMC) | `mace-matpes-r2scan-0` | 0.08 | 0.051 ± 0.001 | 0.0013 | 24.7 ± 0.1 | 1.000 / 1.000 / 1.000 | 23149 | 17.195 | 0 / 11 / 44 | 1.12 | 6/128 (OK) | **PASS** |
| `jctc_remc_nimof74_co2_r2scan` | Ni-MOF-74 + CO2 | auto_mlip (REMC) | `mace-matpes-r2scan-0` | 0.28 | 0.196 ± 0.002 | 0.0030 | 24.7 ± 0.2 | 1.000 / 1.000 / 1.000 | 5384 | 3.999 | 1 / 12 / 45 | 0.96 | 8/128 (OK) | **PASS** |
| `jctc_remc_nimof74_co2_r2scan` | Ni-MOF-74 + CO2 | auto_mlip (REMC) | `mace-matpes-r2scan-0` | 1 | 0.857 ± 0.019 | 0.0192 | 24.7 ± 0.5 | 1.004 / 1.005 / 1.005 | 579 | 0.430 | 50 / 61 / 94 | 0.85 | 15/128 (OK) | **PASS** |

## 2. Distillation Fidelity & Provenance Audit

| Cache ID | System | Model | `model_sha256` | 1B R² / MAE (kJ/mol) | 2B Deployed Hold-out MAE | 2B Classical Hold-out MAE | `n_2b_val` | 2B Audit | Wall (s) |
|---|---|---|---|---|---:|---:|---:|---|---:|
| `021dde18ddcebebc` | Mg-MOF-74 + CO2 | `mace-matpes-r2scan-0` | `8f147ecffa1d...` | 0.9869 / 2.42 | 0.75 | 1.23 | 38 | honest_deployed_holdout | 58.9 |
| `2193e0c1a0f139c5` | Cu-BTC + CO2 | `build/models/mace-dac-1.model+D3` | `6c31634523ce...` | 0.9908 / 1.36 | 0.08 | 0.08 | 38 | honest_deployed_holdout | 211.9 |
| `43e6790faa4527db` | Ni-MOF-74 + CO2 | `mace-matpes-r2scan-0` | `8f147ecffa1d...` | 0.9879 / 2.05 | 1.10 | 1.65 | 38 | honest_deployed_holdout | 44.0 |
| `5248a4184e9dc066` | IRMOF-1 + CO2 | `mace-matpes-r2scan-0` | `8f147ecffa1d...` | 0.9811 / 2.62 | 0.46 | 0.45 | 38 | honest_deployed_holdout | 145.2 |
| `561d9742bf86929e` | Mg-MOF-74 + CO2 | `medium` | `01bfe2210013...` | 0.9860 / 1.56 | 0.72 | 0.72 | 38 | honest_deployed_holdout | 43.1 |
| `605b8e72fe7065b9` | IRMOF-1 + CO2 | `build/models/mace-dac-1.model+D3` | `6c31634523ce...` | 0.9884 / 1.54 | 0.28 | 0.25 | 38 | honest_deployed_holdout | 150.1 |
| `7efad4a3493f82b5` | Mg-MOF-74 + CO2 | `mace-matpes-r2scan-0` | `8f147ecffa1d...` | 0.9522 / 2.54 | 1.54 | 2.57 | 38 | honest_deployed_holdout | 43.5 |
| `83dd22b8cff271ec` | Mg-MOF-74 + CO2 | `medium+D3` | `01bfe2210013...` | 0.9740 / 2.29 | 0.86 | 2.39 | 38 | honest_deployed_holdout | 50.3 |
| `952f747a9932f4c4` | Mg-MOF-74 + CO2 | `mace-matpes-r2scan-0` | `8f147ecffa1d...` | 0.9522 / 2.54 | N/A | N/A | 0 | 1b_only_ablation | 40.3 |
| `adb78c6205d779e6` | Mg-MOF-74 + CO2 | `build/models/mace-dac-1.model+D3` | `6c31634523ce...` | 0.9769 / 2.11 | 1.29 | 1.78 | 38 | honest_deployed_holdout | 50.6 |
| `c8f96a114ece6035` | Ni-MOF-74 + CO2 | `build/models/mace-dac-1.model+D3` | `6c31634523ce...` | 0.9713 / 2.34 | 0.53 | 0.64 | 38 | honest_deployed_holdout | 51.0 |
| `d830f8bc13d38ca7` | IRMOF-1 + CH4 | `medium+D3` | `01bfe2210013...` | 0.9942 / 0.89 | 0.04 | 0.05 | 38 | honest_deployed_holdout | 147.5 |
| `dce19a693744b613` | Cu-BTC + CO2 | `mace-matpes-r2scan-0` | `8f147ecffa1d...` | 0.9774 / 2.79 | 1.00 | 1.26 | 38 | honest_deployed_holdout | 208.0 |
| `eac7c40ffeb880a9` | Cu-BTC + H2O | `build/models/mace-dac-1.model+D3` | `6c31634523ce...` | 0.9236 / 2.85 | 0.17 | 0.33 | 38 | honest_deployed_holdout | 212.7 |

## 3. REMC Replica-Exchange Swap Acceptance Rates

| Run | Swap Every | Pressure Pair | Attempts | Accepted | Acceptance Rate |
|---|---:|---|---:|---:|---:|
| `jctc_exp1_auto_mlip_r2scan_remc` | 2 | 0.012 <-> 0.04 bar | 50000 | 14461 | 28.9% |
| `jctc_exp1_auto_mlip_r2scan_remc` | 2 | 0.04 <-> 0.08 bar | 49996 | 13243 | 26.5% |
| `jctc_exp1_auto_mlip_r2scan_remc` | 2 | 0.08 <-> 0.16 bar | 50000 | 2821 | 5.6% |
| `jctc_exp1_auto_mlip_r2scan_remc` | 2 | 0.16 <-> 0.22 bar | 49996 | 15504 | 31.0% |
| `jctc_exp1_auto_mlip_r2scan_remc` | 2 | 0.22 <-> 0.28 bar | 50000 | 25738 | 51.5% |
| `jctc_exp1_auto_mlip_r2scan_remc` | 2 | 0.28 <-> 0.55 bar | 49996 | 8009 | 16.0% |
| `jctc_exp1_auto_mlip_r2scan_remc` | 2 | 0.55 <-> 1 bar | 50000 | 19520 | 39.0% |
| `jctc_macedac_cubtc_co2` | 2 | 0.05 <-> 0.15 bar | 16000 | 5213 | 32.6% |
| `jctc_macedac_cubtc_co2` | 2 | 0.15 <-> 0.5 bar | 15996 | 1733 | 10.8% |
| `jctc_macedac_cubtc_co2` | 2 | 0.5 <-> 1 bar | 16000 | 2668 | 16.7% |
| `jctc_macedac_cubtc_h2o` | 2 | 0.0005 <-> 0.002 bar | 10000 | 9599 | 96.0% |
| `jctc_macedac_cubtc_h2o` | 2 | 0.002 <-> 0.008 bar | 9996 | 8597 | 86.0% |
| `jctc_macedac_irmof1_co2` | 2 | 0.5 <-> 1 bar | 16000 | 5356 | 33.5% |
| `jctc_macedac_irmof1_co2` | 2 | 1 <-> 2.5 bar | 15996 | 1003 | 6.3% |
| `jctc_macedac_irmof1_co2` | 2 | 2.5 <-> 5 bar | 16000 | 806 | 5.0% |
| `jctc_macedac_mgmof74_co2` | 2 | 0.0002 <-> 0.003 bar | 30000 | 0 | 0.0% |
| `jctc_macedac_mgmof74_co2` | 2 | 0.003 <-> 0.012 bar | 29996 | 359 | 1.2% |
| `jctc_macedac_mgmof74_co2` | 2 | 0.012 <-> 0.08 bar | 30000 | 158 | 0.5% |
| `jctc_macedac_mgmof74_co2` | 2 | 0.08 <-> 0.28 bar | 29996 | 6842 | 22.8% |
| `jctc_macedac_mgmof74_co2` | 2 | 0.28 <-> 1 bar | 30000 | 11690 | 39.0% |
| `jctc_macedac_nimof74_co2` | 2 | 0.012 <-> 0.08 bar | 16000 | 4334 | 27.1% |
| `jctc_macedac_nimof74_co2` | 2 | 0.08 <-> 0.28 bar | 15996 | 1939 | 12.1% |
| `jctc_macedac_nimof74_co2` | 2 | 0.28 <-> 1 bar | 16000 | 271 | 1.7% |
| `jctc_remc_cubtc_co2_r2scan` | 2 | 0.05 <-> 0.15 bar | 16000 | 10422 | 65.1% |
| `jctc_remc_cubtc_co2_r2scan` | 2 | 0.15 <-> 0.5 bar | 15996 | 4521 | 28.3% |
| `jctc_remc_cubtc_co2_r2scan` | 2 | 0.5 <-> 1 bar | 16000 | 4426 | 27.7% |
| `jctc_remc_irmof1_ch4_mp0_d3` | 2 | 0.5 <-> 1 bar | 16000 | 2109 | 13.2% |
| `jctc_remc_irmof1_ch4_mp0_d3` | 2 | 1 <-> 2.5 bar | 15996 | 352 | 2.2% |
| `jctc_remc_irmof1_ch4_mp0_d3` | 2 | 2.5 <-> 5 bar | 16000 | 693 | 4.3% |
| `jctc_remc_irmof1_co2_r2scan` | 2 | 0.5 <-> 1 bar | 40000 | 12785 | 32.0% |
| `jctc_remc_irmof1_co2_r2scan` | 2 | 1 <-> 2.5 bar | 39996 | 1918 | 4.8% |
| `jctc_remc_irmof1_co2_r2scan` | 2 | 2.5 <-> 5 bar | 40000 | 2998 | 7.5% |
| `jctc_remc_mgmof74_co2_mp0_d3` | 2 | 0.0002 <-> 0.002 bar | 16000 | 0 | 0.0% |
| `jctc_remc_mgmof74_co2_mp0_d3` | 2 | 0.002 <-> 0.012 bar | 15996 | 0 | 0.0% |
| `jctc_remc_mgmof74_co2_mp0_d3` | 2 | 0.012 <-> 0.06 bar | 16000 | 310 | 1.9% |
| `jctc_remc_mgmof74_co2_mp0_d3` | 2 | 0.06 <-> 0.28 bar | 15996 | 1207 | 7.5% |
| `jctc_remc_mgmof74_co2_mp0_d3` | 2 | 0.28 <-> 1 bar | 16000 | 2743 | 17.1% |
| `jctc_remc_nimof74_co2_r2scan` | 2 | 0.012 <-> 0.08 bar | 16000 | 12670 | 79.2% |
| `jctc_remc_nimof74_co2_r2scan` | 2 | 0.08 <-> 0.28 bar | 15996 | 8731 | 54.6% |
| `jctc_remc_nimof74_co2_r2scan` | 2 | 0.28 <-> 1 bar | 16000 | 2458 | 15.4% |

## 4. Flagged Issues & Warnings

- **Proposal asymmetry**: `jctc_exp1_uff_r2scan` @ 0.28 bar (ins_att=3721, del_att=9515, ratio=0.39)
