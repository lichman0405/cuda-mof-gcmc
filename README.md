**English** | [简体中文](README.zh-CN.md)

# `cuda-mof-gcmc`

![CUDA C++](https://img.shields.io/badge/CUDA-C%2B%2B17-76B900?style=flat-square&logo=nvidia)
![Multi-GPU Verified](https://img.shields.io/badge/Multi--GPU-sm__86%20%2F%20sm__120-2563EB?style=flat-square)
![RASPA3 Verified](https://img.shields.io/badge/RASPA3%203.0.29-%E2%89%A4%200.54%CF%83-0F766E?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-64748B?style=flat-square)

A cross-platform GPU-native simulation engine for gas adsorption in **arbitrary rigid Metal–Organic Frameworks (MOFs)** (verified on **Windows RTX 5070** `sm_120` and **Linux RTX 3090** `sm_86`). It combines **(1) a verified CUDA classical and many-body self-consistent polarizable (`Pol-SCF`) GCMC solver** with **(2) a zero-I/O CUDA-to-PyTorch Nested Monte Carlo (Nested MC) engine** ([`mof_gcmc_proposer`](include/proposer_api.h)) coupled in memory with batched Machine-Learning Interatomic Potentials (`MACE + TorchDFTD3`) and multi-GPU orchestration (`--gpus`).

> **Active Architecture Refactor**: The new **On-the-Fly 1B+2B Self-Distilled MLIP + Block-Parallel GPU-CBMC Nested MC** design and validation gates are specified in [**`PLAN.zh-CN.md`**](PLAN.zh-CN.md). Legacy unconverged MLIP reports and plots have been removed and will be rewritten once all strict $\hat{R} / \text{ESS} / \text{MCSE}$ gates pass.

---

## Core Architecture Modules

1. **Universal MOF & Multi-Site Guest Preprocessing** ([`scripts/qeq_charges.py`](scripts/qeq_charges.py), [`scripts/guests.py`](scripts/guests.py)): ASE expands any of the `230` space groups while preserving asymmetric-unit charges and labels (`spacegroup_kinds`), with a screened periodic Ewald QEq fallback (`eps_eff = 1.50`) for uncharged frameworks; rigid guests up to `12` sites (including massless `Z = 0` dummy charge sites) use unit-quaternion 3D rotations and Peng–Robinson EOS fugacities.
2. **Zero-I/O CUDA Proposer Library** ([`src/proposer_dll.cu`](src/proposer_dll.cu), [`include/cell.hpp`](include/cell.hpp), [`include/grid3d.hpp`](include/grid3d.hpp)): Evaluates $O(h^4)$ 3D tricubic framework grids and shared-memory guest–guest interactions directly on GPU, advancing inner reversible GCMC chains via `ctypes` pointers with in-memory state rollback.
3. **Batched MLIP Scoring, Convergence Diagnostics & Multi-GPU Scheduling** ([`scripts/nested_mc_engine.py`](scripts/nested_mc_engine.py), [`scripts/mcmc_diagnostics.py`](scripts/mcmc_diagnostics.py), [`scripts/multi_gpu.py`](scripts/multi_gpu.py)): `FastMaceD3Target` batches `no_grad` MACE + cached framework-edge GPU DFT-D3 evaluation, paired with rank-normalized split/folded $\hat{R}$, bulk/tail ESS, independent fluctuation $Q_{\text{st}}$, SHA-256 provenance checkpointing ([`scripts/run_metadata.py`](scripts/run_metadata.py)), and `pressure` / `replica` multi-GPU scaling.

---

## How This Differs from `RASPA3` and `gRASPA`

| Aspect | CPU `RASPA3` / GPU `gRASPA` | `cuda-mof-gcmc` |
| :--- | :--- | :--- |
| **Primary Role** | Full-featured classical GCMC ecosystem (broad force fields, flexible CBMC molecules, high-throughput screening) | **GPU-native many-body polarization (`Pol-SCF`)** + **zero-I/O Nested MLIP-GCMC proposer architecture** |
| **Polarization** | Standard fixed point charges (LJ + Ewald) | 3D tricubic framework E-fields (`Pol-3Site`) + block-collective FP64 Applequist–Thole **many-body `Pol-SCF`** |
| **MLIP Sampling** | No built-in zero-I/O CUDA proposer library for PyTorch MLIP reweighting | Native C ABI ([`proposer_api.h`](include/proposer_api.h)) coupled in memory with `FastMaceD3Target`, chemical-potential shift calibration, $\hat{R}/\text{ESS}$ gates, and bit-exact resume |
| **Scope** | For standard classical force-field screening, use `RASPA3` / `gRASPA` | Focused on arbitrary rigid MOFs + rigid small molecules (`CO₂`, `CH₄`, `N₂`, `H₂`, `Ar`, `H₂O`, `NH₃`) |

---

## Verified Classical & `Pol-SCF` Benchmarks

![Step 4 Mg-MOF-74 Tricubic Validation](assets/step4_validated_mercado_raspa3.png)

- **`RASPA3 3.0.29` Parity (`Mg-MOF-74`, `298 K`, UFF+REPEAT)**: Matches single-core `RASPA3` Tricubic Grid within **`+2.32% / +1.39% / -0.41%`** at `0.15 / 1.0 / 5.0 bar` ($\Delta Q_{\text{st}} \le 0.54\ \text{kJ/mol}$), with full FP64 recompute keeping max drift at **`≤ 1.02e-08 K`** over `64M` steps.
- **Classical GPU Throughput**: **`598,669 MC-steps/s`** (`153.2×` vs. single-core CPU `RASPA3` Tricubic Grid `3,908/s`; `492.6×` vs. Exact Ewald `1,215/s`); builds `256×256×72` 3D grids in `3.03 s` (`38.9×`).
- **`Pol-SCF` Exact Parity**: Matches Python exact `36×36` matrix inversion in `--test-scf` to **`1.83e-08 K`** and **`1.13e-11 e·Å`**.
- **Physical Scope**: Supports **rigid MOF frameworks and rigid multi-site guests** (`≤ 12` sites, `≤ 6` LJ types; no flexible CBMC chains or framework breathing).

---

## Quickstart (Windows & Linux)

### One-Click Setup & Self-Check (`fast_start.bat` / `fast_start.sh`)

- **Windows** — Run [`fast_start.bat`](fast_start.bat) (wraps [`scripts/fast_start_win.ps1`](scripts/fast_start_win.ps1)) from the repo root to detect GPU compute capability (`sm_XX`), CUDA Toolkit, MSVC Build Tools, Python 3.10+, CUDA PyTorch, and dependencies, compile [`build.bat`](build.bat), and run self-checks:

```powershell
# Inspect environment only (no installs or builds)
cmd /c fast_start.bat -CheckOnly

# Auto-install missing prerequisites, build core DLL/EXE, and run Pol-SCF + non-MLIP pytest
cmd /c fast_start.bat
```

- **Linux** — Run [`fast_start.sh`](fast_start.sh) from the repo root (`--gpu <id>` pins `CUDA_VISIBLE_DEVICES`):

```bash
# Inspect environment without making changes
./fast_start.sh --check-only

# Create .venv-mlip, compile core binaries via ./build.sh, and verify
./fast_start.sh --gpu 0
```

### Manual Build & Classical Verification

```powershell
# Windows: build core targets (mof_gcmc_cli.exe + mof_gcmc_proposer.dll)
cmd /c build.bat

# Linux: build core targets (mof_gcmc_cli + libmof_gcmc_proposer.so)
./build.sh
```

```powershell
# Classical / Polarizable CLI presets & verification (replace `.\build\mof_gcmc_cli.exe` with `./build/mof_gcmc_cli` on Linux)
.\build\mof_gcmc_cli.exe --test-scf
.\build\mof_gcmc_cli.exe --config configs\mg_mof74_co2_298K_uff.json
.\build\mof_gcmc_cli.exe --config configs\mg_mof74_fluegas_298K_pol3site.json
.\.venv-mlip\Scripts\python.exe -u scripts\verify_cli.py
.\.venv-mlip\Scripts\python.exe -u scripts\verify_scf_pol.py
.\.venv-mlip\Scripts\python.exe -m pytest -v -m "not mlip"
```

### Nested MLIP-GCMC CLI Entry (`run_mlip_gcmc.py`)

```powershell
.\.venv-mlip\Scripts\python.exe -u scripts\run_mlip_gcmc.py `
  --cif data\IRMOF-1.cif --supercell 1 1 1 --temp 298 --pressures 0.5 1.0 5.0 `
  --guest ch4 --proposer uff --charge-mode auto `
  --model medium --dispersion on --xc pbe `
  --boxes 4 --k-steps 25 --eq-cycles 120 --prod-cycles 320 `
  --out-json build\irmof1_ch4.json --out-csv build\irmof1_ch4.csv
```

- Multi-GPU scaling is enabled via `--gpus 0 1 2 3 --parallel-mode auto`, with strict metadata validation and bit-exact checkpointing via `--checkpoint-dir` and `--resume-mode {none,skip,extend,fresh}`.

---

## Documentation & Repository Layout

- [**Architecture & Validation Plan (Chinese)**](PLAN.zh-CN.md) — On-the-fly 1B+2B self-distilled MLIP + block-parallel GPU-CBMC Nested MC specification and convergence gates.
- [**Classical & Polarizable Validation (Steps 1–5)**](docs/CLASSICAL_VALIDATION.md) — Crystal geometry, tricubic MAE, `RASPA3` parity tables, co-adsorption selectivity, and `Pol-SCF` checks.

```text
cuda-mof-gcmc/
├── PLAN.zh-CN.md   # Active architecture & validation plan (1B+2B Self-Distilled MLIP + GPU-CBMC Nested MC)
├── fast_start.bat  # Windows one-click setup, auto-build & self-check launcher
├── fast_start.sh   # Linux one-click setup (uv/venv), auto-build & self-check launcher
├── build.bat       # Windows MSVC + CUDA compiler wrapper
├── build.sh        # Linux nvcc + g++ compiler wrapper
├── pyproject.toml  # Package metadata & pytest markers (mlip, slow)
├── configs/        # JSON presets for mof_gcmc_cli (UFF, Mercado 2016, 313K PSA, Flue Gas, Pol-SCF)
├── data/           # Crystal CIFs (Mg-MOF-74, Ni-MOF-74, Cu-BTC, IRMOF-1), Mercado FF & NIST ISODB tables
├── docs/           # Classical & Pol-SCF validation docs (CLASSICAL_VALIDATION*.md)
├── include/        # C++17 / CUDA headers: triclinic MIC geometry, 103-elem UFF + PR, 3D grids & proposer C ABI
├── src/            # Active CUDA kernels & drivers (mof_gcmc_cli, mof_gcmc_proposer.dll / .so) + src/legacy/
├── scripts/        # Core engine (run_mlip_gcmc, nested_mc_engine, multi_gpu, mcmc_diagnostics, qeq_charges, guests)
├── tests/          # Automated pytest suite & null-bias verification tests
└── assets/         # Classical validation figures (Steps 1–5)
```
