**English** | [简体中文](README.zh-CN.md)

# `cuda-mof-gcmc`

![CUDA C++](https://img.shields.io/badge/CUDA-C%2B%2B17-76B900?style=flat-square&logo=nvidia)
![Target GPU](https://img.shields.io/badge/RTX%205070-sm__120-2563EB?style=flat-square)
![RASPA3 Verified](https://img.shields.io/badge/RASPA3%203.0.29-%E2%89%A4%200.54%CF%83-0F766E?style=flat-square)
![Pytest](https://img.shields.io/badge/pytest-14%20passed-059669?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-64748B?style=flat-square)

A GPU-native simulation engine for gas adsorption in **arbitrary rigid Metal–Organic Frameworks (MOFs)** on **NVIDIA RTX 5070** (`sm_120`). It combines **(1) a verified CUDA classical and many-body self-consistent polarizable (`Pol-SCF`) GCMC solver** with **(2) a zero-I/O CUDA-to-PyTorch Nested Monte Carlo (Nested MC) engine** that uses a general 3D quaternion GPU proposer (`103`-element UFF + periodic QEq, or `Mercado 2016` DFT-FF) as an inner Markov chain for Machine-Learning Interatomic Potentials (`MACE + TorchDFTD3`).

---

## System Architecture

![cuda-mof-gcmc System Architecture](assets/architecture.png)

1. **Universal MOF & Multi-Site Guest Preprocessing** ([`scripts/qeq_charges.py`](scripts/qeq_charges.py), [`scripts/guests.py`](scripts/guests.py)): ASE expands any of the `230` space groups while preserving asymmetric-unit charges/labels (`spacegroup_kinds`), with a screened periodic Ewald QEq fallback (`eps_eff = 1.50`) for uncharged frameworks; rigid guests up to `12` sites (including massless `Z = 0` dummy charge sites) use unit-quaternion 3D rotations and generic Peng–Robinson EOS fugacities.
2. **Inner GPU Classical Burst (`K`-step proposer)** ([`src/proposer_dll.cu`](src/proposer_dll.cu), [`include/cell.hpp`](include/cell.hpp), [`include/forcefield.hpp`](include/forcefield.hpp)): Runs $K$ reversible rigid-body GCMC steps across `N_boxes` replicas in `~1.8 ms/burst`, screening hard-core overlaps before touching PyTorch.
3. **In-Memory MLIP Scoring & Exact Unbiased Reweighting** ([`scripts/nested_mc_engine.py`](scripts/nested_mc_engine.py)): Only the $K$-step endpoint coordinates pass via `ctypes` pointers to batched `FastMaceD3Target` (chunked `no_grad` MACE + cached framework-edge GPU DFT-D3(BJ) + VRAM fragmentation guard). Pre-calibrated 3-stage bias $\Delta\varepsilon$ balances insertions and deletions, while the outer Metropolis criterion cancels $\Delta\varepsilon\,\Delta N$ and rolls back rejected bursts—preserving the **exact target MLIP grand-canonical ensemble**.

---

## Head-to-Head Sampling Speedup vs. Direct MLIP-GCMC

![Head-to-Head GPU Sampling Speedup](assets/benchmark_ess_speedup.png)

Evaluated on the **same NVIDIA RTX 5070** starting from identical equilibrated MLIP configurations against two unbiased `Direct MLIP-GCMC` baselines ([`scripts/direct_mlip_gcmc.py`](scripts/direct_mlip_gcmc.py))—a standard **Serial ASE** baseline and a stronger **hard-sphere-screened (`r_hs = 1.20 Å`) Batched GPU** baseline sharing `FastMaceD3Target` ([`scripts/benchmark_speedup.py`](scripts/benchmark_speedup.py)):

| System (298 K) | Method | `Uptake (mmol/g)` | `Outer / Move Acc` | `ESS_N / GPU-s` | `SJD_N / GPU-s` | Speedup vs. Serial ASE (`SJD_N` / `ESS_N`) | Speedup vs. Batched GPU (`SJD_N` / `ESS_N`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Mg-MOF-74 + CO₂`** (`0.05 bar`, `r2SCAN+D3`) | `Direct Serial ASE` (`r_hs=1.20 Å`) | `10.14` | `20.0%` | `0.197` | `0.080` | `1.0×` / `1.0×` | — |
| | `Direct Batched GPU` (`4 boxes, r_hs=1.20 Å`) | `9.79 ± 0.27` | `17.5%` | `0.337` | `0.314` | `3.9×` / `1.7×` | `1.0×` / `1.0×` |
| | `Nested GPU (Mercado K=25)` | `9.94 ± 0.40` | `51.6%` | **`0.725`** | `1.735` | `21.6×` / **`3.7×`** | `5.5×` / **`2.1×`** |
| | `Nested GPU (Mercado K=50)` | `9.98 ± 0.32` | `33.3%` | `0.681` | **`2.534`** | **`31.5×`** / `3.5×` | **`8.1×`** / `2.0×` |
| **`IRMOF-1 + CH₄`** (`5.0 bar`, `MACE-MP0+D3`) | `Direct Serial ASE` (`r_hs=1.20 Å`) | `9.04` | `47.0%` | `0.077` | `0.903` | `1.0×` / `1.0×` | — |
| | `Direct Batched GPU` (`4 boxes, r_hs=1.20 Å`) | `8.19 ± 0.15` | `45.6%` | `0.301` | `3.176` | `3.5×` / `3.9×` | `1.0×` / `1.0×` |
| | `Nested GPU (Generic UFF+QEq K=25)` | `8.43 ± 0.32` | `28.1%` | `0.300` | **`7.063`** | **`7.8×`** / `3.9×` | **`2.2×`** / `1.0×` |
| | `Nested GPU (Generic UFF+QEq K=50)` | `8.70 ± 0.08` | `17.0%` | **`0.322`** | `6.833` | `7.6×` / **`4.2×`** | **`2.2×`** / `1.1×` |

> **Physical Diagnostics**: (1) On `Mg-MOF-74`, uncharged UFF starves outer acceptance (`6.2%`), while screened periodic QEq lifts general `UFF` outer acceptance to `38.8%` (`5.45 ± 0.30 mmol/g`); accessing the `~2.25 Å` open-`Mg²⁺` chemisorption well (`9.2–10.0 mmol/g`) still favors softer core potentials such as `Mercado 2016`. (2) PyTorch CUDA caching-allocator cleanup (`empty_cache()` on `memory_reserved` bloat) reduced full `Mg-MOF-74` `0.05 bar` production wall time from `355.8 s` to **`90.3 s` (`3.9×`)** on 12 GB VRAM.

---

## How This Differs from `RASPA3` and `gRASPA`

| Aspect | CPU `RASPA3` / GPU `gRASPA` | `cuda-mof-gcmc` |
| :--- | :--- | :--- |
| **Primary Role** | Full-featured classical GCMC ecosystem (broad force fields, flexible CBMC molecules, high-throughput screening) | **GPU-native many-body polarization (`Pol-SCF`)** + **zero-I/O Nested MLIP-GCMC proposer architecture** |
| **Polarization** | Standard fixed point charges (LJ + Ewald) | 3D tricubic framework E-fields (`Pol-3Site`) + block-collective FP64 Applequist–Thole **many-body `Pol-SCF`** |
| **MLIP Sampling** | No built-in zero-I/O CUDA proposer DLL for PyTorch MLIP reweighting | Native C ABI ([`proposer_api.h`](include/proposer_api.h)) coupled in memory with `FastMaceD3Target`, 3-stage $\Delta\mu_{\text{ref}}$, Pooled-Jackknife $Q_{\text{st}}$, and `ESS` diagnostics |
| **Benchmark Scope** | Benchmarked directly against CPU **`RASPA3 3.0.29`** (not same-GPU vs. `gRASPA`); for standard classical screening, use `gRASPA` | Tuned on single RTX 5070 for arbitrary rigid MOFs + rigid small guests (`CO₂`, `CH₄`, `N₂`, `H₂`, `Ar`, `H₂O`, `NH₃`) |

---

## Modules & Maturity

| Module | What It Does | Status |
| :--- | :--- | :---: |
| **Tricubic Classical CUDA GCMC** ([`src/main_cli.cu`](src/main_cli.cu)) | `0.10 Å` $O(h^4)$ 3D tricubic grids, `UFF` (`103` elements) / `Mercado 2016`, arbitrary triclinic MIC, `Identity Swap`, PR EOS | **Verified** (vs. `RASPA3 3.0.29` `≤ 0.54σ`) |
| **Polarizable & `Pol-SCF` Solvers** | Framework `Pol-COM` / `Pol-3Site` + block-collective FP64 many-body Thole `Pol-SCF` | **Verified** (`1.83e-08 K` vs. `36×36` exact inversion) |
| **Universal Nested MLIP-GCMC Engine** ([`src/proposer_dll.cu`](src/proposer_dll.cu), [`scripts/nested_mc_engine.py`](scripts/nested_mc_engine.py)) | 3D quaternion GPU proposer + `qeq_charges.py` + `guests.py` + batched `FastMaceD3Target` + VRAM guard | **Verified** (`pytest` `14 passed`; `Pol3->Merc` `z = 1.15`, `0.000 K` quat parity) |
| **Direct Baselines & Multi-MLIP Roadmap** ([`scripts/benchmark_speedup.py`](scripts/benchmark_speedup.py)) | Same-GPU `ESS / GPU-s` & `SJD_N / GPU-s` benchmarks vs. Direct MLIP-GCMC (`Mg-MOF-74`, `IRMOF-1`) | **Active** (multi-MLIP plugins & Linux 8-GPU queued next) |

---

## Verified Classical & Parity Benchmarks

![Step 4 Mg-MOF-74 Tricubic Validation](assets/step4_validated_mercado_raspa3.png)

- **`RASPA3 3.0.29` Parity (`Mg-MOF-74`, `298 K`, UFF+REPEAT)**: Matches single-core RASPA3 Tricubic Grid within **`+2.32% / +1.39% / -0.41%`** at `0.15 / 1.0 / 5.0 bar` ($\Delta Q_{\text{st}} \le 0.54\ \text{kJ/mol}$), with full FP64 recompute keeping max drift at **`≤ 1.02e-08 K`** over `64M` steps.
- **Classical GPU Throughput**: **`598,669 MC-steps/s`** (`153.2×` vs. single-core CPU `RASPA3` Tricubic Grid `3,908/s`; `492.6×` vs. Exact Ewald `1,215/s`); builds `256×256×72` 3D grids in `3.03 s` (`38.9×`).
- **`Pol-SCF` Exact Parity**: Matches Python exact `36×36` matrix inversion in `--test-scf` to **`1.83e-08 K`** and **`1.13e-11 e·Å`**.
- **Universal Proposer & QEq Parity**: `Pol-3Site -> Mercado` reproduces direct GPU GCMC within **`z = 1.15`** (`|ΔQst| = 0.43 kJ/mol`, `0.000e+00 K` energy error); `SPC/E H₂O` quaternion `eval_energy_quat` matches to `0.000e+00 K`; periodic QEq matches `REPEAT` / `Cu-BTC` DFT charges with `MAE = 0.153 e` / `0.110 e`.

> **Scope & Limitations**: Supports **rigid MOF frameworks and rigid multi-site guests** (`≤ 12` sites, `≤ 6` LJ types; no flexible CBMC chains or framework breathing). Physical MLIP accuracy depends on the underlying foundation model and dispersion treatment—for instance, `mace-matpes-r2scan-0` severely overbinds `CH₄` in `IRMOF-1` (`-30.6` to `-36.4 kJ/mol`), whereas `MACE-MP-0 medium + D3(pbe)` yields `-12.03 kJ/mol`. See [MLIP Nested MC Notes](docs/MLIP_NESTED_MC_NOTES.md).

---

## Quickstart (Windows & Linux Single-GPU)

### One-Click Setup & Self-Check (`fast_start.bat` / `fast_start.sh`)

- **Windows** — Run [`fast_start.bat`](fast_start.bat) (wraps [`scripts/fast_start_win.ps1`](scripts/fast_start_win.ps1)) from the repo root. It detects your NVIDIA GPU compute capability (`sm_XX`), CUDA Toolkit (`nvcc`), MSVC Build Tools (VS 2022 or VS 2019), Python 3.10+, CUDA PyTorch, and required packages—installing only what is missing (`winget` / `pip`), compiling [`build.bat`](build.bat), and running quick self-checks:

```powershell
# Inspect environment only (no installs or builds)
cmd /c fast_start.bat -CheckOnly

# Auto-install any missing prerequisites, build core DLL/EXE, and run Pol-SCF + pytest
cmd /c fast_start.bat

# Optional flags: force clean CUDA rebuild or include live MACE smoke test
cmd /c fast_start.bat -Rebuild -SmokeMlip
```

- **Linux** — Run [`fast_start.sh`](fast_start.sh) from the repo root (`uv` is used automatically when available, falling back to `python3 -m venv`; `--gpu <id>` pins `CUDA_VISIBLE_DEVICES` on shared multi-GPU servers):

```bash
# Inspect GPU, CUDA, uv/Python, packages, and binaries without making changes
./fast_start.sh --check-only

# Create .venv-mlip (cu124/cu128 auto-selected from driver), compile core binaries via ./build.sh, and verify
./fast_start.sh --gpu 0 --smoke-mlip
```

### Manual Prerequisites & Build
- **Windows**: Windows 10/11 + NVIDIA Driver + CUDA Toolkit (`CUDA_ARCH` auto-detected via `nvidia-smi`, or override `set CUDA_ARCH=sm_89`) + MSVC (VS **2022** or **2019** Build Tools; [`build.bat`](build.bat) probes both)
- **Linux**: Ubuntu/Linux + NVIDIA Driver + CUDA Toolkit (`nvcc` in `PATH`, `/usr/local/cuda`, or NVHPC; [`build.sh`](build.sh) auto-detects `CUDA_ARCH`, or override `CUDA_ARCH=sm_86 ./build.sh`) + `g++`
- **Python / MLIP**: Python 3.10+ (install CUDA PyTorch first, then `pip install -r requirements.txt`; see [requirements.txt](requirements.txt) and [pyproject.toml](pyproject.toml))

```powershell
# Windows: build core targets (mof_gcmc_cli.exe + mof_gcmc_proposer.dll); `build.bat all` builds legacy binaries too
cmd /c build.bat

# Linux: build core targets (mof_gcmc_cli + libmof_gcmc_proposer.so); `./build.sh all` builds legacy binaries too
./build.sh
```

```powershell
# Classical / Polarizable CLI presets & verification (replace `.\build\mof_gcmc_cli.exe` with `./build/mof_gcmc_cli` on Linux)
.\build\mof_gcmc_cli.exe --test-scf
.\build\mof_gcmc_cli.exe --config configs\mg_mof74_co2_298K_uff.json
.\build\mof_gcmc_cli.exe --config configs\mg_mof74_fluegas_298K_pol3site.json
.\.venv-mlip\Scripts\python.exe -u scripts\verify_cli.py
.\.venv-mlip\Scripts\python.exe -m pytest -v
```

### Single-GPU Usage Guide (`run_mlip_gcmc.py`)

```powershell
# General rigid MOF CIF + built-in guest (auto spacegroup expansion + periodic QEq charges)
.\.venv-mlip\Scripts\python.exe -u scripts\run_mlip_gcmc.py `
  --cif path\to\your_mof.cif --supercell 1 1 1 --temp 298 --pressures 0.1 1.0 5.0 `
  --guest ch4 --proposer uff --charge-mode auto `
  --model medium --dispersion on --xc pbe `
  --boxes 4 --k-steps 25 --pilot-cycles 60 --eq-cycles 120 --prod-cycles 320 `
  --out-json build\my_mof_ch4.json --out-csv build\my_mof_ch4.csv

# Add --resume to the exact command above after an interruption to skip completed pressures
```

| Parameter | How to Choose on a Single GPU |
| :--- | :--- |
| `--supercell na nb nc` | Ensure supercell perpendicular width is `>= 12 Å` (`2 × MACE r_max`): e.g. `1 1 1` for `IRMOF-1` (`25.8 Å`), `1 1 2` for `Mg-MOF-74` (`c = 6.9 Å`). |
| `--proposer` | Use **`uff`** for any general MOF/guest; use `mercado` **only** for `Mg-MOF-74 + CO₂`. |
| `--charge-mode` | Default `auto` keeps non-zero CIF charges (or `REPEAT` for `Mg-MOF-74`) and computes screened periodic QEq otherwise; `qeq` forces QEq, `zero` disables charges. |
| `--guest` / `--guest-json` | Presets: `co2`, `ch4`, `n2`, `h2`, `ar`, `h2o`, `nh3`; pass `--guest-json file.json` for custom rigid molecules (up to `12` sites). |
| `--mace-chunk` | Auto-selected by VRAM; set `--mace-chunk 1` on `<= 12 GB` GPUs when supercells exceed `~400` framework atoms (e.g. `IRMOF-1 + CH₄`). |
| `--model` / `--xc` | Recommended baseline: `medium --dispersion on --xc pbe`; for `Mg-MOF-74 + CO₂`, `mace-matpes-r2scan-0 --dispersion on --xc r2scan` works well (note it overbinds `CH₄`). |

<details>
<summary>Custom <code>--guest-json</code> format &amp; reading output CSV columns</summary>

Example custom 3-site rigid molecule JSON (`Z = 0` designates massless electrostatic sites excluded from MLIP evaluation):

```json
{
  "name": "custom_co",
  "formula": "CO",
  "is_isotropic": false,
  "Tc_K": 132.92,
  "Pc_bar": 34.99,
  "omega": 0.048,
  "sites": [
    {"symbol": "C", "Z": 6, "pos": [-0.646, 0.0, 0.0], "q": -0.24, "eps_K": 36.4, "sigma_A": 3.55},
    {"symbol": "O", "Z": 8, "pos": [ 0.482, 0.0, 0.0], "q": -0.27, "eps_K": 51.0, "sigma_A": 2.95},
    {"symbol": "X", "Z": 0, "pos": [ 0.000, 0.0, 0.0], "q":  0.51, "eps_K":  0.0, "sigma_A": 0.00}
  ]
}
```

Key columns in `--out-csv`: `uptake_mmol_g` ± `uptake_sem_mmol_g`, Pooled-Jackknife `Qst_kJ_mol` ± `Qst_sem_kJ_mol`, production swaps `prod_ins_acc` / `prod_del_acc` (aim for `>= 5` each), `tau_int_N`, `ess_N_per_s`, and `converged` (`False` prints specific diagnostic warnings—usually resolved by increasing `--eq-cycles` / `--prod-cycles`).

</details>

```powershell
# Head-to-head ESS / GPU-s & SJD_N / GPU-s speedup benchmark vs. Direct MLIP-GCMC
.\.venv-mlip\Scripts\python.exe -u scripts\benchmark_speedup.py
```

### Multi-GPU Execution & 4× RTX 3090 Scaling (`--gpus`)

Pass `--gpus 0 1 2 3` (via [`scripts/multi_gpu.py`](scripts/multi_gpu.py)) to scale across multiple GPUs in two modes (`--parallel-mode auto` selects `pressure` when `>= 2` pressures remain, or `replica` for single high-box-count runs):

```bash
# 1) Multi-pressure isotherm queue (1 worker per GPU; ideal for full isotherms)
.venv-mlip/bin/python -u scripts/run_mlip_gcmc.py \
  --cif data/IRMOF-1.cif --supercell 1 1 1 --temp 298 --pressures 0.5 1.0 2.5 5.0 \
  --guest ch4 --proposer uff --charge-mode auto --model medium --dispersion on --xc pbe \
  --boxes 4 --k-steps 25 --eq-cycles 120 --prod-cycles 320 \
  --gpus 0 1 2 3 --parallel-mode pressure --out-json build/irmof1_4gpu.json

# 2) Single-pressure replica sharding (splits --boxes evenly across GPUs with exact Jackknife/ESS merge)
.venv-mlip/bin/python -u scripts/run_mlip_gcmc.py \
  --cif data/IRMOF-1.cif --supercell 1 1 1 --temp 298 --pressures 5.0 \
  --guest ch4 --proposer uff --charge-mode auto --model medium --dispersion on --xc pbe \
  --boxes 16 --k-steps 25 --eq-cycles 40 --prod-cycles 80 \
  --gpus 0 1 2 3 --parallel-mode replica --out-json build/irmof1_replica_4gpu.json
```

Measured apples-to-apples (`--worker-threads 4` on both single- and multi-GPU runs to avoid OpenMP oversubscription on dual-socket EPYC) on **`IRMOF-1 + CH₄`** (`MACE-MP0 medium + D3(pbe)`, `298 K`, `40 eq + 80 prod` cycles, **4× NVIDIA RTX 3090**):

| Workload | `1× RTX 3090` Wall (`MC Loop`) | `4× RTX 3090` Wall (`MC Loop`) | End-to-End Speedup (`MC Loop`) | `ESS_N / wall-s` (`1 GPU -> 4 GPUs`) |
| :--- | :---: | :---: | :---: | :---: |
| **`pressure` mode** (`0.5, 1.0, 2.5, 5.0 bar`, `4 boxes/pt`) | `270.55 s` (`78.81 s`) | **`78.66 s`** (`21.55 s` max) | **`3.44×`** (`3.66×`) | `0.34 -> 1.59` (`4.7×`) |
| **`replica` mode** (`5.0 bar`, `16 boxes`) | `125.52 s` (`69.24 s`) | **`79.76 s`** (`22.33 s`) | **`1.57×`** (**`3.10×`**) | `0.74 -> 1.07` (`1.4×`) |

> **Why `pressure` scales higher than short `replica` runs**: Each `run_pressure_point` call builds the `240³` (`13.8M`-point) tricubic grid (`~34.2 s` on `IRMOF-1`) + warms up MACE (`~4.7 s`) once per GPU. In `pressure` mode all `4` GPUs overlap grid builds across pressures (`3.44×` already on `120` cycles), whereas in short `120`-cycle `replica` runs that fixed `~55 s` per-point grid/calibration setup stays unsharded (larger `--eq-cycles` / `--prod-cycles` amortizes it toward `~2.3–3.0×`).

---

## Documentation & Repository Layout

- [**Technical Report (Preprint Draft)**](docs/TECHNICAL_REPORT.md) — Unified methods, `RASPA3` / `Pol-SCF` / Nested MC verification, foundation-model diagnostics & 4-MOF NIST ISODB benchmarks.
- [**Classical & Polarizable Validation (Steps 1–5)**](docs/CLASSICAL_VALIDATION.md) — Crystal geometry, tricubic MAE, RASPA3 parity tables, co-adsorption selectivity & `Pol-SCF` checks.
- [**MLIP Nested MC: Verified Mechanics & WIP Notes**](docs/MLIP_NESTED_MC_NOTES.md) — Reweighting null-bias proofs, $\Delta\mu_{\text{ref}}$ calibration, batched MACE speedup, foundation-model diagnostics & open scientific issues.

```text
cuda-mof-gcmc/
├── fast_start.bat  # Windows one-click setup, auto-build & self-check launcher
├── fast_start.sh   # Linux one-click setup (uv/venv), auto-build & self-check launcher
├── build.bat       # Windows MSVC + CUDA compiler wrapper (VS2022/2019 & GPU sm_XX)
├── build.sh        # Linux nvcc + g++ compiler wrapper (auto-detects CUDA & GPU sm_XX)
├── pyproject.toml  # Package metadata & pytest markers (mlip, slow)
├── configs/        # JSON presets for mof_gcmc_cli.exe (UFF, Mercado 2016, 313K PSA, Flue Gas, Pol-SCF)
├── data/           # Crystal CIFs (Mg-MOF-74, Ni-MOF-74, Cu-BTC, IRMOF-1), Mercado FF & NIST ISODB tables
├── docs/           # Detailed validation benchmarks, equations & WIP research notes (EN / ZH)
├── include/        # C++17 / CUDA headers: triclinic MIC geometry, 103-elem UFF + PR, 3D grids & proposer C ABI
├── src/            # Active CUDA kernels & drivers (mof_gcmc_cli, mof_gcmc_proposer.dll) + src/legacy/ (v1–v3)
├── scripts/        # Core engine (run_mlip_gcmc, nested_mc_engine, multi_gpu, qeq_charges, guests) + experiments/ & legacy/
├── tests/          # Automated pytest suite (18 tests: 103-elem, QEq, quaternion guests, multi-GPU merge, Direct parity)
└── assets/         # Architecture diagrams & benchmark/verification plots (300 DPI)
```
