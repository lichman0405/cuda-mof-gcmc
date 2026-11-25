**English** | [简体中文](README.zh-CN.md)

# `cuda-mof-gcmc`

![CUDA C++](https://img.shields.io/badge/CUDA-C%2B%2B17-76B900?style=flat-square&logo=nvidia)
![Multi-GPU Verified](https://img.shields.io/badge/Multi--GPU-sm__86%20%2F%20sm__120-2563EB?style=flat-square)
![RASPA3 Verified](https://img.shields.io/badge/RASPA3%203.0.29-%E2%89%A4%200.54%CF%83-0F766E?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-64748B?style=flat-square)

A cross-platform GPU-native simulation engine for gas adsorption in **arbitrary rigid Metal–Organic Frameworks (MOFs)** (verified on **Windows RTX 5070** `sm_120` and **Linux RTX 3090** `sm_86`). It combines **(1) a `RASPA3`-validated CUDA classical and many-body self-consistent polarizable (`Pol-SCF`) GCMC solver** with **(2) a zero-manual-force-field `Auto-MLIP` (`1B+2B` online self-distillation + 128-thread block-parallel GPU Rosenbluth CBMC) Nested Monte Carlo engine** coupled directly via zero-I/O device pointers with batched `MACE + TorchDFTD3` and multi-GPU scaling.

---

## Key Verified Results (See [Technical Report](docs/TECHNICAL_REPORT.md))

![Auto-MLIP + GPU-CBMC Nested MC Architecture](assets/architecture_auto_mlip_cbmc.png)

- **Zero-Manual-Typing `1B+2B` Self-Distillation + Block-Parallel GPU-CBMC**: Distills environment-aware `1B` tricubic grids (15% holdout $R^2 = \mathbf{0.985\text{–}0.991}$, MAE `1.27–2.29 kJ/mol`) and in-pore `2B` pair LUTs (MAE `0.10–1.65 kJ/mol`) directly from MLIP samples in `52.8–209.5 s` with SHA-256 disk caching.
- **`Mg-MOF-74 + CO₂ @ 298 K` Method Benchmark & Experimental Parity**:
  ![Mg-MOF-74 Method Comparison](assets/fig1_mgmof74_method_comparison.png)
  - Relative to `Direct MLIP-GCMC` (`3.65%` insertion acc), `Auto-MLIP + CBMC` (`k=16`) raises **insertion acceptance by `15.8×` (`57.5%`)**, **$\text{ESJD}_N$ by `8.0×`** (`0.1314` vs. `0.0164`), and **`dN_moves/s` by `2.6×`**;
  - Bidirectional `empty` vs. `high_loading` initializations (`0.28 bar`) converge to **`6.322 ± 0.233`** and **`6.314 ± 0.171 mmol/g`** (**`0.0074 mmol/g` / `0.12%`** difference); at **`1.0 bar`**, `r2SCAN No-D3` yields **`8.148 ± 0.025 mmol/g`** (**`0.26%`** from NIST experimental `8.170 mmol/g`) with zero manual parameters.
- **Cross-MOF / Cross-Guest Generalization & MLIP Physics Audit**: Tested across **`Mg-MOF-74`, `Ni-MOF-74`, `Cu-BTC`, `IRMOF-1`** and **`CO₂` / `CH₄`** alongside **Widom** Henry-regime checks, diagnosing `MP0+D3` overbinding (`51%–57%` binding from D3 `-22.2` to `-24.0 kJ/mol`) and `r2SCAN` OOD artifacts on `IRMOF-1 + CH₄`.
- **`4× RTX 3090` Multi-GPU Scaling**: `replica` parallelization achieves **`3.26×` pure MC kernel speedup** (`51.10 s → 15.69 s`, **`81.4%`** efficiency; **`1.90×`** end-to-end with 4× MACE loads); `pressure` queue mode achieves **`2.19×`** end-to-end (`131.23 s → 59.79 s`).
- **Classical & `Pol-SCF` Baseline** (see [Classical Validation](docs/CLASSICAL_VALIDATION.md)): Matches `RASPA3 3.0.29` Tricubic within `≤ 0.54σ` at **`598,669 MC-steps/s`** (`153.2×`) and matches exact `36×36` matrix inversion in `Pol-SCF` to **`1.83e-08 K`**.

---

## Core Architecture Modules

1. **Universal MOF & Multi-Site Guest Preprocessing** ([`scripts/qeq_charges.py`](scripts/qeq_charges.py), [`scripts/guests.py`](scripts/guests.py)): Expands all 230 space groups via ASE (`spacegroup_kinds`), falls back to screened Ewald QEq (`eps_eff = 1.50`), and supports rigid guests up to `12` sites with Peng–Robinson EOS fugacities.
2. **Online `1B+2B` Self-Distillation & Zero-I/O CUDA Proposer** ([`scripts/mlip_distill.py`](scripts/mlip_distill.py), [`src/proposer_dll.cu`](src/proposer_dll.cu), [`include/proposer_api.h`](include/proposer_api.h)): Evaluates `O(h^4)` 3D tricubic grids and `2B` pore LUTs in GPU memory with 128-thread block-parallel Rosenbluth CBMC.
3. **Exact Nested MC, Strict Convergence Gates & Multi-GPU Orchestration** ([`scripts/nested_mc_engine.py`](scripts/nested_mc_engine.py), [`scripts/mcmc_diagnostics.py`](scripts/mcmc_diagnostics.py), [`scripts/multi_gpu.py`](scripts/multi_gpu.py)): Pairs `FastMaceD3Target` with median $\Delta\varepsilon$ calibration, split/folded rank-normalized $\hat{R}$, bulk/tail ESS, fluctuation $Q_{\text{st}}$, SHA-256 checkpointing ([`scripts/run_metadata.py`](scripts/run_metadata.py)), and `pressure`/`replica` multi-GPU scheduling.

---

## Quickstart (Windows & Linux)

### One-Click Setup & Self-Check (`fast_start.bat` / `fast_start.sh`)

```powershell
# Windows: check prerequisites / auto-build + self-test
cmd /c fast_start.bat -CheckOnly
cmd /c fast_start.bat
```

```bash
# Linux: check prerequisites / auto-setup .venv-mlip + build + self-test
./fast_start.sh --check-only
./fast_start.sh --gpu 0
```

### `Auto-MLIP + GPU-CBMC` Nested MC CLI ([`scripts/run_mlip_gcmc.py`](scripts/run_mlip_gcmc.py))

```bash
# Mg-MOF-74 + CO2 @ 298 K (r2SCAN No-D3, 128-trial GPU-CBMC, multi-GPU + checkpoint extend)
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=scripts .venv-mlip/bin/python -u scripts/run_mlip_gcmc.py \
  --cif data/MgMOF-74.cif --supercell 1 1 2 --guest co2 --temp 298 \
  --pressures 0.012 0.15 0.28 1.0 --proposer auto_mlip --cbmc-k 128 \
  --model mace-matpes-r2scan-0 --dispersion off \
  --boxes 4 --k-steps 16 --pilot-cycles 40 --eq-cycles 1200 --prod-cycles 4800 \
  --distill-cache-dir build/distill_cache --gpus 0 1 2 3 --parallel-mode auto \
  --checkpoint-dir build/ckpt_mg74 --checkpoint-every 200 --resume-mode extend \
  --out-json build/mgmof74_co2_r2scan.json

# Widom Henry constant & MACE vs. D3 energy decomposition
PYTHONPATH=scripts .venv-mlip/bin/python -u scripts/widom_mlip.py \
  --cif data/MgMOF-74.cif --supercell 1 1 2 --guest co2 --temp 298 \
  --model mace-matpes-r2scan-0 --dispersion off --n-samples 6000 \
  --out-json build/widom_mgmof74_r2scan.json
```

### Classical & `Pol-SCF` Verification

```powershell
.\build\mof_gcmc_cli.exe --test-scf
.\build\mof_gcmc_cli.exe --config configs\mg_mof74_co2_298K_uff.json
.\.venv-mlip\Scripts\python.exe -m pytest -v -m "not mlip"
```

---

## Documentation & Layout

- [**MLIP Nested MC Technical Report (EN)**](docs/TECHNICAL_REPORT.md) | [**中文技术报告**](docs/TECHNICAL_REPORT.zh-CN.md)
- [**Classical & Polarizable Validation (EN)**](docs/CLASSICAL_VALIDATION.md) | [**中文**](docs/CLASSICAL_VALIDATION.zh-CN.md)
- [**Architecture & Validation Plan (ZH)**](PLAN.zh-CN.md) | [**Verified Benchmark JSON Archive**](docs/results/phase2_phase3_summary.json)

```text
cuda-mof-gcmc/
├── fast_start.bat / .sh  # One-click check, dependency install, build & self-test
├── build.bat / .sh       # MSVC+CUDA / nvcc+g++ build wrappers
├── configs/              # JSON presets for mof_gcmc_cli (UFF, Mercado, Flue Gas, Pol-SCF)
├── data/                 # MOF CIFs, Mercado FF & NIST ISODB experimental isotherm CSVs
├── docs/                 # Technical reports, classical validation docs & results/phase2_phase3_summary.json
├── include/              # CUDA/C++17 headers & zero-I/O proposer C ABI (proposer_api.h)
├── src/                  # GPU GCMC & 128-thread block-parallel GPU-CBMC kernels + src/legacy/
├── scripts/              # run_mlip_gcmc, mlip_distill, widom_mlip, nested_mc_engine, multi_gpu, plot_phase4_figures
├── tests/                # Pytest regression & unit tests (incl. test_phase1_distill_and_cbmc.py)
└── assets/               # Architecture diagram & publication validation figures
```
