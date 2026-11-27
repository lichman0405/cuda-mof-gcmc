**English** | [简体中文](README.zh-CN.md)

# `cuda-mof-gcmc`

![CUDA C++](https://img.shields.io/badge/CUDA-C%2B%2B17-76B900?style=flat-square&logo=nvidia)
![Multi-GPU](https://img.shields.io/badge/Multi--GPU-sm__86%20%2F%20sm__120-2563EB?style=flat-square)
![RASPA3 Verified](https://img.shields.io/badge/RASPA3%203.0.29-%E2%89%A4%200.54%CF%83-0F766E?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-64748B?style=flat-square)

A GPU-accelerated CUDA/C++17 grand-canonical Monte Carlo (GCMC) engine for gas adsorption in rigid MOFs, running on Windows (RTX 5070, `sm_120`) and multi-GPU Linux nodes (8×RTX 3090, `sm_86`). It combines two verified components:

1. A classical and self-consistent polarizable (`Pol-SCF`) GCMC solver validated against `RASPA3 3.0.29`; see [classical validation](docs/CLASSICAL_VALIDATION.md).
2. An automated **Nested Monte Carlo + independent-ladder $\mu$-REMC** engine targeting full-system machine-learned interatomic potentials (`mace-matpes-r2scan-0`, `MACE-MP-0`, `MACE-DAC-1`, with or without D3(BJ)). For each `(framework, guest, MLIP)` tuple, a 1B radial-plus-electroneutral-Ewald tricubic grid and optional 2B in-pore residual LUT are distilled on demand directly from the target MLIP—without manual framework atom typing—and drive 128-trial Rosenbluth CBMC GPU bursts with equilibration-frozen marginal-drift $(\Delta\varepsilon, k)$ calibration and FP64 node-energy-accumulated batched MLIP outer corrections.

![Method overview](assets/fig1_architecture.png)

## Benchmark highlights

All headline numbers below come from archived raw production trajectories and deterministic verification scripts under `docs/results/`.

- **Unified continuous-coordinate periodic MIC hard-core domain:** Direct MLIP-GCMC and CUDA `auto_mlip` enforce the identical continuous-coordinate minimum-image hard-core predicate (`include/hard_core.hpp`: `1.20 Å` guest H–framework, `1.48 Å` guest heavy–framework, `0.50 Å` distinct guest real-atom pairs, virtual sites excluded, $N_{\max}=128$, zero grid-corner or surrogate-energy exclusion), verified to **`0 / 1,040` mismatches** against 27-image brute-force periodic search and **`58 / 58`** passing non-MLIP regression tests.
- **Matched-budget $2\times 2$ factorial benchmark (`Mg-MOF-74 + CO2`, `1×1×2`, 298 K, `0.28 bar`, `mace-matpes-r2scan-0` without D3, 8×RTX 3090):** Across 4 groups × 3 independent seeds (`7,200 s` wall-clock = `57,600` whole-ladder GPU-s/run across 8 pressures `0.012–1.0 bar`, 4 isolated ladders, bit-identical `v1.4` 1B grids, separate `39.7–43.7 s` distillation and `88.9–93.3 s` 8-GPU shift calibration), all 12 runs yield close fixed-budget uptake point estimates (`6.16–6.30 mmol/g`, grand median `6.27 mmol/g`), and **$\mu$-REMC is the dominant mixing driver**:
  - **REMC OFF (`0` swap attempts verified):** `0/3` pass for A (`1B-only`, median $\mathrm{ESS}_{\text{bulk}}(N) = 239.6$, $\hat{R}_N = 1.0156$) and `0/3` pass for B (`1B+2B`, median $\mathrm{ESS}_{\text{bulk}}(N) = 231.9$, $\hat{R}_N = 1.0189$); all 6 runs do not converge within the `7,200 s` budget.
  - **REMC ON (`29–43` production ladder round trips):** Increases the loading bulk ESS rate billed over total whole-ladder GPU time ($\mathrm{ESS}_{\text{bulk}}(N)/\text{GPU-s}$) by **`2.26×`** on 1B-only (`C/A`: `0.00942` vs. `0.00416`) and **`2.29×`** on 1B+2B (`D/B`: `0.00921` vs. `0.00402`), raising strict 11-gate pass counts from **`0/3` and `0/3`** (REMC OFF) to **`1/3` and `2/3`** at the budget endpoint (**`0/3, 0/3, 1/3, 1/3`** under two-consecutive-checkpoint confirmation: C at `4,800 s` / `5,100 s`, D at `6,600 s` / `6,900 s`).
  - **2B residual fitting vs. matched-budget ESS rate:** 2B LUT distillation reduces 80/20 deployed hold-out pair MAE from **`2.60` to `1.51 kJ/mol`** ($R^2 = 0.818$) and halves the calibrated shift magnitude (`-3.05` to `-1.42 kJ/mol`), while giving **`0.97×` (`B/A`) and `0.98×` (`D/C`)** matched-budget ESS-rate ratios and narrowing cross-seed diagnostic spread under REMC (`3/3` D seeds with $\hat{R}_{\max} \le 1.0089$ and bulk ESS $\ge 418.8$ across $N$, $U$, and OMS).
- **On-demand 1B+2B distillation & cross-MOF transferability:** Across 4 MOFs (Mg-MOF-74, Ni-MOF-74, Cu-BTC, IRMOF-1), 3 guests ($\text{CO}_2$, $\text{CH}_4$, $\text{H}_2\text{O}$), and 4 MLIP setups (`39.7–212.7 s` single-GPU build), 1B hold-out $R^2$ spans `0.9236–0.9942` (MAE `0.89–2.85 kJ/mol`), 2B deployed-LUT hold-out MAE spans `0.04–1.54 kJ/mol`, and domain-fine-tuned **`MACE-DAC-1+PBE-D3` $\text{CO}_2$ isotherms converge on 18/18 (100%)** state points across all four frameworks. Widom and equilibrium partitioning show D3(BJ) supplies `47.9%–53.0%` (`MACE-DAC-1`) and `45.8%–54.1%` (`MP0+D3`) of Mg-MOF-74 interaction energy.
- **Numerical precision, multi-GPU & classical baseline:** `FP32` forward with `FP64` `node_energy` summation matches full `FP64` to RMS `0.00044–0.00130 kJ/mol` with **`0%` accept/reject flips** across 64 outer test pairs and `5.93×–9.49×` speedup; 4×RTX 3090 replica sharding accelerates MC by **`3.26×`** (`2.19×` pressure queue end-to-end); classical Tricubic GCMC matches single-core `RASPA3 3.0.29` within `≤ 0.54σ` at `598,669 MC-steps/s`, and `Pol-SCF` matches `36×36` matrix inversion to `1.83×10⁻⁸ K`.

````carousel
![Matched-budget 2x2 ablation on Mg-MOF-74 + CO2 at 0.28 bar](assets/fig2_parity_efficiency.png)
<!-- slide -->
![Cross-MOF transferability and MACE-DAC-1 vs r2SCAN isotherms](assets/fig3_transferability_finetuned.png)
<!-- slide -->
![Target functional sensitivity, Widom, D3 partitioning, and exploratory 1x1x4 test](assets/fig4_physics_diagnostics.png)
<!-- slide -->
![Global Vehtari convergence audit, REMC swap acceptance, consistency, and GPU cost](assets/fig5_convergence_scaling.png)
````

## Code layout

1. **Framework and guest preprocessing** ([`scripts/qeq_charges.py`](scripts/qeq_charges.py), [`scripts/guests.py`](scripts/guests.py)): ASE expands any space group and preserves CIF charges; uncharged frameworks fall back to screened periodic Ewald QEq; rigid guests support up to 12 sites with Peng–Robinson fugacities.
2. **Surrogate distillation and GPU proposer** ([`scripts/mlip_distill.py`](scripts/mlip_distill.py), [`src/proposer_dll.cu`](src/proposer_dll.cu), [`include/proposer_api.h`](include/proposer_api.h), [`include/hard_core.hpp`](include/hard_core.hpp)): `float32` 1B tricubic grids and clipped 2B residual LUTs reside in device memory; one 128-thread CUDA block per simulation box evolves $k$-step on-device inner bursts (`128`-trial parallel Rosenbluth CBMC under unified continuous MIC hard-core, $N_{\max}=128$) with compact `~29 KB` end-state C ABI handoff and full FP64 energy recomputation.
3. **Nested MC, $\mu$-REMC, and convergence diagnostics** ([`scripts/nested_mc_engine.py`](scripts/nested_mc_engine.py), [`scripts/mcmc_diagnostics.py`](scripts/mcmc_diagnostics.py), [`scripts/multi_gpu.py`](scripts/multi_gpu.py)): batched MACE/D3 evaluation (`FP32` + `FP64` node sum), equilibration-frozen marginal-drift $(\Delta\varepsilon, k)$ calibration, isolated-ladder $\mu$-REMC, and rank-normalized split-$\hat{R}$ / bulk-tail ESS diagnostics.

## Quickstart

### Environment check and build

```powershell
# Windows: check prerequisites only
cmd /c fast_start.bat -CheckOnly
# Windows: install what is missing, build and run fast self-tests (-FullTest runs long parity checks)
cmd /c fast_start.bat
```

```bash
# Linux
./fast_start.sh --check-only
./fast_start.sh --gpu 0
```

### Nested MC & isolated-ladder $\mu$-REMC ([`scripts/run_mlip_gcmc.py`](scripts/run_mlip_gcmc.py))

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv-mlip/bin/python -u scripts/run_mlip_gcmc.py \
  --cif data/IRMOF-1.cif --supercell 1 1 1 --guest co2 --temp 298 \
  --pressures 0.5 1.0 2.5 5.0 --proposer auto_mlip --cbmc-k 128 \
  --model build/models/mace-dac-1.model --dispersion on --dispersion-xc pbe \
  --boxes 4 --k-steps 24 --eq-cycles 800 --prod-cycles 16000 \
  --distill-cache-dir build/distill_cache --gpus 0 1 2 3 \
  --parallel-mode remc --remc-swap on --remc-swap-every 2 \
  --out-json build/irmof1_co2_macedac_remc.json
```

Widom insertion and energy decomposition:

```bash
.venv-mlip/bin/python -u scripts/widom_mlip.py \
  --cif data/MgMOF-74.cif --supercell 1 1 2 --guest co2 --temp 298 \
  --model mace-matpes-r2scan-0 --dispersion off --n-samples 6000 \
  --out-json build/widom_mgmof74_r2scan.json
```

### Classical and `Pol-SCF` regression tests

```powershell
.\build\mof_gcmc_cli.exe --test-scf
.\build\mof_gcmc_cli.exe --config configs\mg_mof74_co2_298K_uff.json
.\.venv-mlip\Scripts\python.exe -m pytest -v -m "not mlip and not slow"
```

## Documentation

- **Classical & Polarizable Validation (`RASPA3 3.0.29`):** [Classical & polarizable validation](docs/CLASSICAL_VALIDATION.md) · [中文](docs/CLASSICAL_VALIDATION.zh-CN.md)
- **Precision & Hard-Core Benchmarks:** [`docs/results/precision_hc_audit.json`](docs/results/precision_hc_audit.json) · [`docs/results/phase2_phase3_summary.json`](docs/results/phase2_phase3_summary.json)

```text
cuda-mof-gcmc/
├── fast_start.bat / .sh  # environment check, dependency install, build and self-test
├── build.bat / .sh       # MSVC+CUDA / nvcc+g++ build scripts
├── configs/              # mof_gcmc_cli presets (UFF, Mercado, flue gas, Pol-SCF)
├── data/                 # CIFs, Mercado force-field files, experimental isotherms
├── docs/                 # classical validation docs and archived benchmark results
├── include/              # CUDA/C++17 headers, hard-core MIC check, proposer C ABI
├── src/                  # CUDA kernels and drivers; src/legacy/ holds older versions
├── scripts/              # run_mlip_gcmc, mlip_distill, widom_mlip, nested_mc_engine, multi_gpu, mcmc_diagnostics
├── tests/                # pytest suite
└── assets/               # benchmark figures (300 DPI)
```
