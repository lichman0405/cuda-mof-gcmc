**English** | [简体中文](README.zh-CN.md)

# `cuda-mof-gcmc`

![CUDA C++](https://img.shields.io/badge/CUDA-C%2B%2B17-76B900?style=flat-square&logo=nvidia)
![Multi-GPU](https://img.shields.io/badge/Multi--GPU-sm__86%20%2F%20sm__120-2563EB?style=flat-square)
![RASPA3 Verified](https://img.shields.io/badge/RASPA3%203.0.29-%E2%89%A4%200.54%CF%83-0F766E?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-64748B?style=flat-square)

A CUDA/C++17 GCMC engine for gas adsorption in rigid MOFs, running on Windows (RTX 5070, `sm_120`) and multi-GPU Linux (RTX 3090, `sm_86`). It has two parts:

1. A classical and fully self-consistent polarizable (`Pol-SCF`) GCMC solver, benchmarked against `RASPA3`; see the [classical validation](docs/CLASSICAL_VALIDATION.md).
2. A nested Monte Carlo engine whose target is a machine-learned potential (MACE, optionally with D3). The reference potential is distilled online from the target MLIP, so no hand-built force field is needed; inner moves are 128-trial Rosenbluth configurational-bias moves on the GPU, and the outer step applies a batched MLIP evaluation with an exact Metropolis correction. Methods and results are in the [technical report](docs/TECHNICAL_REPORT.md).

![Method overview](assets/fig1_architecture.png)

## Main results

All numbers come from the [technical report](docs/TECHNICAL_REPORT.md) and [`docs/results/phase2_phase3_summary.json`](docs/results/phase2_phase3_summary.json).

- **Surrogate construction.** Each (framework, guest, MLIP) combination is fitted to 1,200 single-guest poses and 192 in-pore dimers in 56–210 s on one GPU. Across the seven caches used in production, the 1B hold-out R² is 0.975–0.990 and the MAE 1.27–2.44 kJ/mol.
- **Sampling efficiency (Mg-MOF-74 + CO₂, 0.28 bar, same r2SCAN target).** Direct MLIP-GCMC reaches a box-averaged uptake of about 1.8 mmol/g after 300 s; a UFF proposer gives 2.9% outer acceptance. Our sampler converges to 6.32 ± 0.23 and 6.31 ± 0.17 mmol/g from empty and preloaded starts, with an efficiency comparable to the system-specific Mercado-2016 force field.
- **Comparison with experiment.** Agreement is set mainly by the target MLIP. r2SCAN is close to experiment for Mg-MOF-74 at 1 bar (8.15 vs. 7.02–8.16 mmol/g) and IRMOF-1 at 5 bar (3.74 vs. 4.70), but underestimates Mg-MOF-74 at 0.012 bar by 8–14× and Ni-MOF-74 and Cu-BTC by about an order of magnitude. MP0+D3 overestimates Mg-MOF-74 by about 2.4×, with 51%–57% of its binding energy from D3.
- **Convergence.** Six of 16 production points pass all criteria ($\hat{R}<1.01$, ESS ≥ 400, etc.). Failures, mostly in the steep part of isotherms and at high loading, are listed individually in the report.
- **Multi-GPU.** On four RTX 3090s, replica sharding speeds up the MC phase by 3.26× and a pressure queue speeds up end-to-end runs by 2.19×.
- **Classical and `Pol-SCF` baseline.** Mg-MOF-74 matches single-core `RASPA3 3.0.29` tricubic results within ≤ 0.54σ; the classical kernel runs at 598,669 MC-steps/s; `Pol-SCF` matches exact 36×36 matrix inversion to 1.83×10⁻⁸ K.

![Experiment 1: Mg-MOF-74 + CO₂](assets/fig2_exp1_mgmof74.png)

## Code layout

1. **Framework and guest preprocessing** ([`scripts/qeq_charges.py`](scripts/qeq_charges.py), [`scripts/guests.py`](scripts/guests.py)): ASE expands any space group and keeps CIF charges; uncharged frameworks fall back to screened periodic Ewald QEq; rigid guests have up to 12 sites, with Peng–Robinson fugacities.
2. **Surrogate distillation and GPU proposer** ([`scripts/mlip_distill.py`](scripts/mlip_distill.py), [`src/proposer_dll.cu`](src/proposer_dll.cu), [`include/proposer_api.h`](include/proposer_api.h)): the 1B grids and 2B table stay in device memory, one thread block per box, with zero-copy exchange to Python through a C ABI.
3. **Nested MC, convergence diagnostics and multi-GPU scheduling** ([`scripts/nested_mc_engine.py`](scripts/nested_mc_engine.py), [`scripts/mcmc_diagnostics.py`](scripts/mcmc_diagnostics.py), [`scripts/multi_gpu.py`](scripts/multi_gpu.py), [`scripts/run_metadata.py`](scripts/run_metadata.py)): batched MACE/D3 evaluation, Δε calibration, rank-normalized $\hat{R}$ and bulk/tail ESS, checkpointing with SHA-256 metadata checks, and `pressure` / `replica` multi-GPU modes.

## Quickstart

### Environment check and build

```powershell
# Windows: check prerequisites only
cmd /c fast_start.bat -CheckOnly
# Windows: install what is missing, build and self-test
cmd /c fast_start.bat
```

```bash
# Linux
./fast_start.sh --check-only
./fast_start.sh --gpu 0
```

### Nested MC ([`scripts/run_mlip_gcmc.py`](scripts/run_mlip_gcmc.py))

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=scripts .venv-mlip/bin/python -u scripts/run_mlip_gcmc.py \
  --cif data/MgMOF-74.cif --supercell 1 1 2 --guest co2 --temp 298 \
  --pressures 0.012 0.15 0.28 1.0 --proposer auto_mlip --cbmc-k 128 \
  --model mace-matpes-r2scan-0 --dispersion off \
  --boxes 4 --k-steps 16 --pilot-cycles 40 --eq-cycles 1200 --prod-cycles 4800 \
  --distill-cache-dir build/distill_cache --gpus 0 1 2 3 --parallel-mode auto \
  --checkpoint-dir build/ckpt_mg74 --checkpoint-every 200 --resume-mode extend \
  --out-json build/mgmof74_co2_r2scan.json
```

Widom insertion and energy decomposition:

```bash
PYTHONPATH=scripts .venv-mlip/bin/python -u scripts/widom_mlip.py \
  --cif data/MgMOF-74.cif --supercell 1 1 2 --guest co2 --temp 298 \
  --model mace-matpes-r2scan-0 --dispersion off --n-samples 6000 \
  --out-json build/widom_mgmof74_r2scan.json
```

Regenerate the report figures:

```bash
.venv-mlip/bin/python scripts/plot_phase4_figures.py
```

### Classical and `Pol-SCF` regression tests

```powershell
.\build\mof_gcmc_cli.exe --test-scf
.\build\mof_gcmc_cli.exe --config configs\mg_mof74_co2_298K_uff.json
.\.venv-mlip\Scripts\python.exe -m pytest -v -m "not mlip"
```

## Documentation

- [Technical report (English)](docs/TECHNICAL_REPORT.md) · [技术报告（中文）](docs/TECHNICAL_REPORT.zh-CN.md)
- [Classical and polarizable validation](docs/CLASSICAL_VALIDATION.md) · [中文](docs/CLASSICAL_VALIDATION.zh-CN.md)
- [Design plan and stage gates (Chinese)](PLAN.zh-CN.md)
- [Result summary JSON](docs/results/phase2_phase3_summary.json)

```text
cuda-mof-gcmc/
├── fast_start.bat / .sh  # environment check, dependency install, build and self-test
├── build.bat / .sh       # MSVC+CUDA / nvcc+g++ build scripts
├── configs/              # mof_gcmc_cli presets (UFF, Mercado, flue gas, Pol-SCF)
├── data/                 # CIFs, Mercado force-field files, experimental isotherms
├── docs/                 # technical reports, classical validation, results/
├── include/              # CUDA/C++17 headers and the proposer C ABI
├── src/                  # CUDA kernels and drivers; src/legacy/ holds older versions
├── scripts/              # run_mlip_gcmc, mlip_distill, widom_mlip, nested_mc_engine, multi_gpu, plot_phase4_figures, ...
├── tests/                # pytest suite
└── assets/               # figures used in the reports and docs
```
