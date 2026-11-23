**English** | [简体中文](README.zh-CN.md)

# `cuda-mof-gcmc`

![CUDA C++](https://img.shields.io/badge/CUDA-C%2B%2B17-76B900?style=flat-square&logo=nvidia)
![Target GPU](https://img.shields.io/badge/RTX%205070-sm__120-2563EB?style=flat-square)
![RASPA3 Verified](https://img.shields.io/badge/RASPA3%203.0.29-%E2%89%A4%200.54%CF%83-0F766E?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-64748B?style=flat-square)

GPU-accelerated batched Grand-Canonical Monte Carlo (GCMC) simulation engine in CUDA/C++17 for rigid Metal–Organic Frameworks (MOFs) on **NVIDIA RTX 5070** (`sm_120`). Combines a native $O(h^4)$ 3D tricubic classical/polarizable CUDA kernel (`598,669 MC-steps/s`) with a chemical-potential-shifted zero-I/O GPU proposal DLL for Nested Machine-Learning Interatomic Potential (MLIP) GCMC (`MACE + TorchDFTD3`).

![Step 4 Mg-MOF-74 Tricubic Validation](assets/step4_validated_mercado_raspa3.png)

---

## Key Features & Verified Benchmarks

- **Native `RASPA3 (3.0.29)` Parity (`≤ 0.54σ`)**: On `Mg-MOF-74` (`298 K`, UFF+REPEAT), the `0.10 Å` `O(h⁴)` tricubic GPU solver matches single-core RASPA3 Tricubic Grid within **`+2.32% / +1.39% / -0.41%`** at `0.15 / 1.0 / 5.0 bar` ($\Delta Q_{\text{st}} \le 0.54\ \text{kJ/mol}$) while maintaining zero-drift `double` state (`≤ 1.02e-08 K` max drift across `64M` steps).
- **High-Throughput GPU Kernels**: Sustains **`598,669 MC-steps/s`** (`153.2×` vs. single-core RASPA3 Tricubic Grid `3,908/s`; `492.6×` vs. Exact Ewald `1,215/s`), and builds `256×256×72` 3D grids in `3.03 s` (`38.9×` vs. CPU RASPA3).
- **Anisotropic & Many-Body SCF Polarization**: Supports framework-only `Pol-COM` / `Pol-3Site` and block-collective shared-memory Applequist-Thole self-consistent dipole solver `Pol-SCF` (agrees with exact `36×36` Python matrix inversion to **`1.83e-08 K`** and **`1.13e-11 e·Å`**).
- **Unified JSON/CLI Solver ([`src/main_cli.cu`](src/main_cli.cu))**: Single binary supporting generic CIF expansion (`19` MOF elements, optional `_atom_site_charge`), `UFF` / `Mercado 2016` DFT-FF, binary CO₂/N₂ & CO₂/CH₄ Identity Swap, and Peng–Robinson EOS fugacities.
- **Zero-I/O Nested MLIP-GCMC Engine ([`scripts/run_mlip_gcmc.py`](scripts/run_mlip_gcmc.py))**:
  - CUDA proposer DLL ([`proposer_set_chem_pot_shift`](include/proposer_api.h)) matches kernel energy to **`0.000e+00 K`** and passes the `Pol-3Site -> Mercado` exact null-bias proof (`z = 0.27σ`).
  - Batched `FastMaceD3Target` achieves **`4.21×`** speedup over ASE (`157.7 → 37.5 ms/box`, parity `≤ 0.0082 kJ/mol/CO2`), with verified `1×1×2` vs. `1×1×4` supercell invariance (`z = 0.04σ` classical, `z = 0.63σ` MLIP).

> **Scope & Limitations**: The classical/polarizable solver assumes **rigid frameworks and small rigid molecules** (`CO₂`, `N₂`, `CH₄`; no CBMC flexible chains or breathing). `Pol-SCF` solves $O(N^2)$ FP64 iterations per trial move and pairs with non-polarizable TraPPE/UFF baselines without dedicated polarizable parameters. The MLIP Nested MC pipeline is an **active research engine**: physical accuracy depends strongly on the chosen universal MLIP, dispersion treatment, DFT reference calibration, and grand-canonical chain length (see [MLIP Notes & Open Issues](docs/MLIP_NESTED_MC_NOTES.md)).

---

## Quickstart

### Prerequisites
- Windows 10/11 + NVIDIA CUDA Toolkit (`sm_120`, RTX 5070) + MSVC (`Visual Studio 2022` Build Tools)
- Optional (for MLIP Nested MC & verification plots): Python 3.11+ with PyTorch CUDA, `mace-torch`, `torch-dftd`, `ase`, `numpy`, `matplotlib`

### Build & Run

```powershell
# Compile all 5 targets (mof_gcmc, mof_gcmc_mix, mof_gcmc_v2, mof_gcmc_cli, mof_gcmc_proposer.dll)
cmd /c build.bat all

# Classical / Polarizable unified CLI presets
.\build\mof_gcmc_cli.exe --test-scf
.\build\mof_gcmc_cli.exe --config configs\mg_mof74_co2_298K_uff.json
.\build\mof_gcmc_cli.exe --config configs\mg_mof74_fluegas_298K_pol3site.json
.\build\mof_gcmc_cli.exe --config configs\mg_mof74_co2_298K_scf.json
uv run --with numpy python scripts\verify_cli.py

# Hybrid GPU-GCMC + MLIP Nested MC CLI (research / WIP)
.venv-mlip\Scripts\python.exe scripts\verify_nested_mc.py
.venv-mlip\Scripts\python.exe -u scripts\run_mlip_gcmc.py `
  --cif data\MgMOF-74.cif --pressures 0.01 0.05 0.15 `
  --proposer mercado --model mace-matpes-r2scan-0 --dispersion on --xc r2scan
```

---

## Documentation

- [**Classical & Polarizable Validation (Steps 1–5)**](docs/CLASSICAL_VALIDATION.md) — Crystal geometry checks, tricubic grid MAE, RASPA3 exact/grid parity tables, mixture selectivity, and Level-2 `Pol-SCF` benchmarks.
- [**MLIP Nested MC: Verified Mechanics & WIP Notes**](docs/MLIP_NESTED_MC_NOTES.md) — Reweighting zero-bias proofs, chemical-potential shifting, batched MACE speedup, foundation-model short-range diagnostics, preliminary results, and open scientific issues.

---

## Repository Layout

```text
cuda-mof-gcmc/
├── configs/   # JSON presets for mof_gcmc_cli.exe (UFF, Mercado 2016, PSA 313K, Flue Gas, Pol-SCF)
├── data/      # CIF crystal structures (Mg-MOF-74, Ni-MOF-74, Cu-BTC) & Mercado 2016 parameters
├── docs/      # Detailed validation benchmarks, derivations & WIP research notes (EN / ZH)
├── include/   # C++17 / CUDA headers: cell MIC, force fields, 3D grids, CIF loader & proposer C ABI
├── src/       # CUDA kernels & C++ drivers (mof_gcmc, mix, v2, cli, mof_gcmc_proposer.dll)
├── scripts/   # Python verification suites, RASPA3 benchmark wrappers & MLIP Nested MC engine
└── assets/    # Verified verification & benchmark plots (300 DPI)
```
