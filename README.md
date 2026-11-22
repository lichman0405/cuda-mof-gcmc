**English** | [简体中文](README.zh-CN.md)

# `cuda-mof-gcmc`: GPU-Accelerated Batched GCMC for Open-Metal-Site MOFs

A CUDA simulation engine for gas adsorption in Metal–Organic Frameworks (MOFs) on **NVIDIA GeForce RTX 5070** (`sm_120`), built specifically around open-metal-site (OMS) adsorption physics such as **$\text{CO}_2$ capture in Mg-MOF-74 ($\text{Mg}_2(\text{dobdc})$)**.

![Step 1 Mg-MOF-74 CO2 298K Results](assets/step1_mof74_co2_298K.png)

---

## Roadmap (Three-Step Plan)

- **[Done] Step 1 — GPU Rigid 3D Framework Grid + Batched GCMC Baseline (UFF + REPEAT + TraPPE)**
  - Precomputes 6 periodic 3D unit-cell grids (`U_lj_C`, `U_lj_O`, `Phi_elec`, `Ex`, `Ey`, `Ez`) via exact 12-6 LJ + 3D Ewald summation on GPU, runs 32 independent pressure/replica boxes concurrently (`128` threads/box, shared-memory molecules, exact hexagonal MIC, Marsaglia/Rodrigues rotation, DSF electrostatics, Peng–Robinson fugacity), and verifies detailed balance + replica convergence.
- **[Done] Step 2 (Branch B1) — Explicit $\text{Mg}^{2+}$ Electric-Field Polarization + Thole Coupling**
  - Activates the precomputed `(Ex, Ey, Ez)` 3D Ewald electric-field grids with anisotropic $\text{CO}_2$ polarizability ($\alpha_\parallel = 4.05\ \text{\AA}^3,\ \alpha_\perp = 1.95\ \text{\AA}^3,\ \bar{\alpha} = 2.65\ \text{\AA}^3$), comparing COM polarization (`Pol-COM`) and 3-site distributed polarization (`Pol-3Site`, $w_C = 0.40, w_O = 0.30$) with Thole-damped induced-dipole–induced-dipole guest interactions ($a = 2.1304$).
- **[Done] Step 3 — Competitive Multi-Component Co-Adsorption ($\text{CO}_2/\text{N}_2$ & $\text{CO}_2/\text{CH}_4$)**
  - Extends the GPU 3D-grid + anisotropic polarization engine (`build\mof_gcmc_mix.exe`) with TraPPE $\text{N}_2$ (`3`-site) / $\text{CH}_4$ (`1`-site UA), binary Peng–Robinson EOS mixture fugacities, Thole-damped cross-species induced dipoles, and symmetric **Identity Swap (`A <-> B`)** moves verified via both an analytical ideal-gas Poisson test and a real-pore `Swap-ON` vs. `Swap-OFF` check.
- **[Done] Step 4 (Publication-Grade Credibility Upgrades) — `0.10 Å` Zero-Drift Engine, Official Mercado 2016 DFT-FF & Native RASPA3 3.0.29 Back-to-Back Verification**
  - Upgrades numerical precision (`256x256x72` `0.10 Å` grids, `double` shared-memory coordinates, periodic full-box recomputation every `50k` steps, `4 × 1.0M` replicas = `64M` steps/sweep in `build\mof_gcmc_v2.exe`), integrates the official **Mercado et al. (*J. Phys. Chem. C* 2016, 120, 12590)** DFT-derived Buckingham/LJ + EPM2 force field, and verifies the UFF baseline head-to-head against native **RASPA3 (`3.0.29`)** exact Ewald simulations on Windows (`623×` GPU speedup).

---

## Step 1 Physical & Numerical Verification (RTX 5070)

### 1. Crystal Structure & Symmetry ([`data/MgMOF-74.cif`](data/MgMOF-74.cif))
- **Unit Cell (`R -3`, hexagonal setting)**: $a = b = 26.136\ \text{\AA},\ c = 6.942\ \text{\AA},\ \alpha = \beta = 90^\circ,\ \gamma = 120^\circ$
- **Expanded Unit Cell (`1x1x1`)**: `162` atoms (`18 Mg`, `54 O`, `72 C`, `18 H`), REPEAT charge sum = `0.000000 e`, molar mass = `2184.372 g/mol`
- **Minimum Bond Lengths**: `Mg–O = 1.948 Å`, `C–O = 1.337 Å`, `C–C = 1.420 Å`, `C–H = 1.123 Å`
- **GCMC Supercell (`1x1x4`)**: $V = 16,426.82\ \text{\AA}^3$, mass = `8737.488 g/mol`, minimum perpendicular width = `22.634 Å > 2 * 11.0 Å`
- **Exact Hexagonal MIC**: `0 / 10,000` mismatches against `3x3x3` brute-force search (verified in both C++ and Python)

### 2. GPU 3D Grid Precomputation (`128 x 128 x 36`)
- **Build Time**: **`356.79 ms`** for all 6 grids (`U_lj_C`, `U_lj_O`, `Phi_elec`, `Ex`, `Ey`, `Ez`, Ewald $\alpha = 0.30\ \text{\AA}^{-1}, r_{\text{cut}} = 11.0\ \text{\AA}$)
- **Independent Verification** (vs. direct Ewald $\alpha_{\text{ref}} = 0.25\ \text{\AA}^{-1}$ across `1,000` pore points):
  - `MAE(U_lj_C) = 3.491 K`, `MAE(U_lj_O) = 10.482 K`, `MAE(Phi) = 4.4038 K/e`

### 3. Batched GCMC Performance & 298 K $\text{CO}_2$ Isotherm
- **Configuration**: `32` boxes (`16` pressures `0.005–20.0 bar` × `2` independent replicas), `100k` equilibration + `200k` production steps/box (**`9.6M` total MC steps**)
- **Runtime**: **`13.906 s`** (`0.69 M MC-steps/sec`)
- **Key Isotherm Results (UFF + REPEAT + TraPPE baseline)**:
  - `0.150 bar` ($\phi = 0.9992$): **`1.1724 ± 0.0301 mmol/g`**, $Q_{\text{st}} = \mathbf{26.07\ \text{kJ/mol}}$ (`acc_ins = acc_del = 0.09`)
  - `1.000 bar` ($\phi = 0.9945$): **`6.7019 ± 0.0769 mmol/g`**, $Q_{\text{st}} = \mathbf{27.49\ \text{kJ/mol}}$ (`acc_ins = acc_del = 0.05`)
  - `20.000 bar` ($\phi = 0.8934$): **`14.7214 ± 0.1619 mmol/g`**, $Q_{\text{st}} = \mathbf{36.46\ \text{kJ/mol}}$
- **Physical Diagnostic**: Just as established in the MOF literature, generic non-polarizable UFF+REPEAT underpredicts the 298 K, `0.15 bar` uptake (`1.17` vs experimental `~3.6–4.6 mmol/g`) and zero-loading heat (`~25.6` vs `~40–47 kJ/mol`), directly motivating **Step 2 (Branch B1)** polarization.

---

## Step 2 (Branch B1): Anisotropic $\text{Mg}^{2+}$ Electric-Field Polarization (RTX 5070)

![Step 2 Mg-MOF-74 Polarization Comparison](assets/step2_mof74_pol_comparison.png)

### 1. Electric-Field Verification
- **Analytical Ewald Gradient Check**: Smooth full-shell $-\nabla \Phi_{\text{FD}}$ vs. analytical `E_direct` across `100` pore points yields **`MAE = 2.21e-06 K/(e·Å)`** (`h = 1e-4 Å`).
- **Grid Interpolation Accuracy**: Trilinear `(Ex, Ey, Ez)` on `128x128x36` vs. independent `alpha_ref = 0.25 Å^-1` across `1,000` pore points yields **`MAE(|E|) = 8.909 K/(e·Å)`**.

### 2. Side-by-Side Model Comparison (`9.6M` MC steps per model on RTX 5070)

| Model | GPU Time | `0.15 bar` Uptake (mmol/g) | `1.00 bar` Uptake (mmol/g) | `20.0 bar` Uptake (mmol/g) | Low-$P$ $Q_{\text{st}}$ (kJ/mol) | Mean $U_{\text{pol}}/N$ (kJ/mol) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`NonPol` (Step 1)** | `14.210 s` | `1.1724 ± 0.0301` | `6.7019 ± 0.0769` | `14.7214 ± 0.1619` | `25.5–26.1` | `0.00` |
| **`Pol-COM`** | `16.627 s` | `1.5406 ± 0.0125` | **`7.6421 ± 0.0038`** | `14.9957 ± 0.15` | `26.5–27.6` | `-0.67 → -0.45` |
| **`Pol-3Site` (Step 2 B1)** | `18.014 s` | **`2.3890 ± 0.0392`** (`+104%`) | `9.1446 ± 0.1418` | `15.3591 ± 0.43` | **`29.2–31.4`** | **`-2.50 → -1.58`** |

- **Key Physical Insight**: Distributing polarizability onto the 3 molecular sites (`Pol-3Site`) captures the steep $r^{-4}$ electric-field gradient right at the coordinating $\text{O}_{\text{CO}_2}\text{–Mg}^{2+}$ contact—boosting polarization energy **3.7×** (`-2.50` vs. `-0.67 kJ/mol` for `Pol-COM`), doubling `0.15 bar` uptake (`1.17 → 2.39 mmol/g`), and demonstrating OMS saturation as `-U_pol/N` drops from `2.50` (low loading) to `1.58 kJ/mol` (`20 bar`). Detailed balance holds across every pressure (`max|acc_ins - acc_del| <= 0.0007`). The remaining ~10 kJ/mol gap vs. experimental `40–47 kJ/mol` reflects short-range orbital charge transfer/exchange beyond generic UFF `sigma_Mg` (bridged by DFT-derived OMS parameters or MLIP Nested MC).

---

## Step 3: Competitive Multi-Component Co-Adsorption ($\text{CO}_2/\text{N}_2$ & $\text{CO}_2/\text{CH}_4$ @ 298 K)

![Step 3 Mg-MOF-74 Binary Co-Adsorption](assets/step3_mof74_binary_coadsorption.png)

### 1. Multi-Species Grids & Identity-Swap Detailed Balance
- **8 Precomputed Grids (`128x128x36`)**: Built in **`410.76 ms`** (`U_lj[C_CO2, O_CO2, N_N2, CH4]`, `Phi_elec`, `Ex`, `Ey`, `Ez`); `MAE` vs. direct reference ($\alpha_{\text{ref}} = 0.25\ \text{\AA}^{-1}$, `1,000` pore points) = `0.844 / 2.493 / 2.974 / 14.415 K` (LJ), `3.9038 K/e` (`Phi`), `7.535 K/(e·Å)` (`|E|`).
- **Test A (Ideal-Gas Identity Swap)**: `<N0> = 11.9827` (exact `12.0`), `<N1> = 39.8378` (exact `40.0`).
- **Test B (`1.0 bar` Flue Gas `Pol-3Site` Swap Consistency)**: `Swap-ON` (`CO2 = 2.5498`, `N2 = 0.2774 mmol/g`) vs. `Swap-OFF` (`CO2 = 2.4995`, `N2 = 0.2772 mmol/g`).

### 2. Competitive Selectivity Boost on RTX 5070 (`24` boxes × `350k` steps per sweep)

| Mixture (`298 K`) | Pressure | `NonPol` Uptake (mmol/g) | `NonPol` $S$ | `Pol-3Site` Uptake (mmol/g) | `Pol-3Site` $S$ | Selectivity Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Flue Gas $\text{CO}_2/\text{N}_2$ (`15:85`)** | `1.0 bar` | `1.159 / 0.254` | `25.8` | **`2.751 / 0.276`** | **`56.4`** | **`2.19×`** |
| **Flue Gas $\text{CO}_2/\text{N}_2$ (`15:85`)** | `2.0 bar` | `2.290 / 0.471` | `27.5` | **`4.648 / 0.450`** | **`58.6`** | **`2.13×`** |
| **Biogas $\text{CO}_2/\text{CH}_4$ (`50:50`)** | `0.5 bar` | `1.967 / 0.196` | `10.0` | **`3.995 / 0.201`** | **`19.9`** | **`1.99×`** |
| **Biogas $\text{CO}_2/\text{CH}_4$ (`50:50`)** | `1.0 bar` | `3.765 / 0.363` | `10.4` | **`6.296 / 0.340`** | **`18.5`** | **`1.78×`** |

---

## Step 4: Publication-Grade Credibility Upgrades (`0.10 Å`, Mercado 2016 & RASPA3)

![Step 4 Validation: 0.10 A Grid, RASPA3 & Mercado 2016 DFT-FF](assets/step4_validated_mercado_raspa3.png)

### 1. Upgrade 1 — Numerical Precision & Sampling (`build\mof_gcmc_v2.exe`)
- **Grid Refinement (`128x128x36` `0.20 Å` → `256x256x72` `0.10 Å`)**: Build time = `3.13 s` (6 grids, `113 MB` VRAM); radial `Mg–O` MAE (`2.3–3.8 Å`) drops **15.2×** (`190.46 K → 12.53 K`); pore MAE vs. direct Ewald ($\alpha_{\text{ref}} = 0.25\ \text{\AA}^{-1}$, `1,000` pts) drops to `U_lj_C = 0.793 K`, `U_lj_O = 2.387 K`, `Phi = 3.3867 K/e`.
- **Zero-Drift Arithmetic**: Guest coordinates and accumulating energies are held in full `double` shared memory with exact periodic full-box energy recomputation every `50,000` steps — maximum energy drift across all `128` GPU boxes (`64M` steps/FF) is **`6.345e-09 K`** (UFF) and **`8.498e-09 K`** (Mercado 2016), with detailed balance `max|acc_ins - acc_del| <= 0.000171`.

### 2. Upgrade 2 — Official Mercado 2016 DFT-Derived Buckingham/LJ + EPM2 Force Field
- Integrated directly from [`data/mercado2016/`](data/mercado2016/) (`rociomer/DFT-derived-force-field`, *J. Phys. Chem. C* 2016, 120, 12590; `Mg_O_CO2` Buckingham potential with exact turnover clamp + shifted `r_cut = 11.0 Å` + Mercado partial charges + EPM2 $\text{CO}_2$).
- **Low-Pressure Open-Metal Physics (`0.15 bar`, 298 K)**: Zero/low-loading $Q_{\text{st}}$ jumps from `26.86 kJ/mol` (UFF) to **`36.04 kJ/mol`**, ideal-crystal uptake rises from `1.34` to **`6.77 mmol/g`**, and the radial $\text{Mg}^{2+}\text{–O}_{\text{CO}_2}$ coordination peak shifts **`-0.25 Å`** inward (`2.775 Å → 2.525 Å`), clearly illustrating both improved open-metal attraction and why idealized rigid 100%-accessible crystal models typically exceed defect-limited experimental uptake (`3.6–4.6 mmol/g @ 0.15 bar`, `7.5–8.6 mmol/g @ 1.0 bar`; cf. ~80% accessible sites dotted curve).

### 3. Upgrade 3 — Back-to-Back Verification vs. Native `RASPA3 3.0.29` (Exact Ewald, Same CIF & `r_cut = 11.0 Å`)

| Pressure | Step 1 GPU `0.20 Å` | **GPU `v2` `0.10 Å` (`4 × 1M`)** | **Native `RASPA3 3.0.29` (Exact Ewald)** | GPU vs. RASPA3 Diff | $Q_{\text{st}}$ Diff |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **`0.15 bar`** | `1.1724 mmol/g` | **`1.3404 ± 0.0152 mmol/g`** | `1.4087 ± 0.0328 mmol/g` | **`-4.85%`** | `-0.12 kJ/mol` |
| **`1.00 bar`** | `6.7019 mmol/g` | **`7.3513 ± 0.0262 mmol/g`** | `7.3384 ± 0.1329 mmol/g` | **`+0.18%`** | `-0.48 kJ/mol` |
| **`5.00 bar`** | `12.6165 mmol/g` | **`13.0857 ± 0.0583 mmol/g`** | `13.3048 ± 0.2118 mmol/g` | **`-1.65%`** | `+0.48 kJ/mol` |

- **Speedup**: GPU `v2` achieves **`757,119 MC-steps/s`** (`0.76 M steps/s`, `64M` steps in `84.53 s`) vs. single-trajectory CPU `RASPA3` exact Ewald at **`1,215 MC-steps/s`** (`1,106–1,360 steps/s`) — a **`623.0×`** wall-clock throughput speedup while staying within `<= 4.85%` (`0.18%` at `1.0 bar`) and `< 0.5 kJ/mol` $Q_{\text{st}}$ agreement.

---

## Build & Run

```powershell
# Compile all targets (mof_gcmc.exe, mof_gcmc_mix.exe, mof_gcmc_v2.exe) for RTX 5070 (sm_120)
cmd /c build.bat all

# Step 1 + Step 2 single-component runs & plots
.\build\mof_gcmc.exe
uv run --with matplotlib --with numpy python scripts\verify_physics.py
uv run --with matplotlib --with numpy python scripts\verify_step2.py

# Step 3 binary competitive adsorption (CO2/N2 & CO2/CH4) + PR check & plot
.\build\mof_gcmc_mix.exe
uv run --with matplotlib --with numpy python scripts\verify_step3.py

# Step 4 high-precision 0.10 A UFF + Mercado 2016 + native RASPA3 benchmark & validation plot
cmd /c build.bat v2
.\build\mof_gcmc_v2.exe
uv run python scripts\run_raspa3_bench.py
uv run --with matplotlib --with numpy python scripts\verify_upgrades.py
```

---

## Repository Layout

```text
cuda-mof-gcmc/
├── data/
│   ├── MgMOF-74.cif              # Official RASPA2 R-3 Mg-MOF-74 crystal structure
│   └── mercado2016/              # Official Mercado et al. (JPCC 2016) DFT-FF & EPM2 definitions
├── include/
│   ├── cell.hpp                  # Hexagonal/triclinic cell transform & exact 5-candidate MIC
│   ├── forcefield.hpp            # UFF, REPEAT, Mercado 2016, TraPPE/EPM2 & single/binary PR EOS
│   ├── grid3d.hpp                # Periodic 3D grid view (U_lj, Phi, Ex, Ey, Ez) & trilinear lookup
│   ├── cif_loader.hpp            # CIF symmetry expander & geometry checker interface
│   ├── gcmc_gpu.hpp              # Step 1/2 grid builder & single-component batched GCMC API
│   ├── mix_gcmc.hpp              # Step 3 multi-species 8-grid builder & binary GCMC API
│   └── v2_gcmc.hpp               # Step 4 high-precision 0.10 A UFF & Mercado 2016 solver API
├── src/
│   ├── cif_loader.cpp            # Symmetry expansion -> 162 atoms & bond/charge validation
│   ├── grid_builder.cu           # Step 1/2 GPU 3D LJ + Ewald potential & E-field precomputation
│   ├── gcmc_kernel.cu            # Step 1/2 batched GCMC (128 threads/box, DSF + Thole coupling)
│   ├── main.cu                   # Step 1/2 driver (NonPol vs Pol-COM vs Pol-3Site)
│   ├── mix_grid_builder.cu       # Step 3 GPU 8-grid precomputation (C_CO2, O_CO2, N_N2, CH4 + E-field)
│   ├── mix_gcmc_kernel.cu        # Step 3 binary GCMC (5 moves incl. Identity Swap + Thole coupling)
│   ├── main_mix.cu               # Step 3 driver (Ideal-gas & swap checks + CO2/N2 & CO2/CH4 sweeps)
│   ├── grid_builder_v2.cu        # Step 4 256x256x72 Buckingham/LJ + Ewald grid builder
│   ├── gcmc_kernel_v2.cu         # Step 4 double-SMEM + 50k periodic recompute GCMC + Mg-O RDF
│   └── main_v2.cu                # Step 4 driver (UFF 0.10 A vs Mercado 2016 0.10 A, 64M steps each)
├── scripts/
│   ├── verify_physics.py         # Independent MIC/PR checks & Step 1 300 DPI plot
│   ├── verify_step2.py           # Step 2 polarization verification & 300 DPI comparison plot
│   ├── verify_step3.py           # Binary PR EOS verification & Step 3 300 DPI comparison plot
│   ├── run_raspa3_bench.py       # Native Windows RASPA3 3.0.29 parallel exact-Ewald runner
│   └── verify_upgrades.py        # Step 4 automated assertions & 300 DPI validation figure
├── assets/
│   ├── step1_mof74_co2_298K.png          # Step 1: 298 K isotherm, Qst & pore density
│   ├── step2_mof74_pol_comparison.png    # Step 2: NonPol vs Pol-COM vs Pol-3Site comparison
│   ├── step3_mof74_binary_coadsorption.png # Step 3: CO2/N2 & CO2/CH4 selectivity & competitive map
│   └── step4_validated_mercado_raspa3.png  # Step 4: 0.10 A vs RASPA3 vs Mercado 2016 validation
├── build.bat                     # VS2019 + nvcc (sm_120) build script (all / single / mix / v2)
├── README.md                     # English documentation
└── README.zh-CN.md               # Chinese documentation
```
