**English** | [简体中文](README.zh-CN.md)

# `cuda-mof-gcmc`: GPU-Accelerated Batched GCMC for Open-Metal-Site MOFs

A CUDA simulation engine for gas adsorption in Metal–Organic Frameworks (MOFs) on **NVIDIA GeForce RTX 5070** (`sm_120`), built specifically around open-metal-site (OMS) adsorption physics such as **$\text{CO}_2$ capture in Mg-MOF-74 ($\text{Mg}_2(\text{dobdc})$)**.

![Step 4 Mg-MOF-74 Tricubic Validation](assets/step4_validated_mercado_raspa3.png)

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

## Step 4: Publication-Grade Credibility Upgrades (`0.10 Å` Tricubic, Mercado 2016 & RASPA3)

![Step 4 Validation: 0.10 A Tricubic Grid, RASPA3 & Mercado 2016 DFT-FF](assets/step4_validated_mercado_raspa3.png)

### 1. Upgrade 1 — `O(h⁴)` 3D Tricubic Interpolation & Zero-Drift Sampling (`build\mof_gcmc_v2.exe`)
- **Grid & Interpolation Refinement (`128×128×36` Trilinear → `256×256×72` Trilinear → `256×256×72` 4-pt Lagrange Tricubic)**:
  - Grid build time = **`3.03 s`** (UFF, 6 grids = `113 MB` VRAM) — **`38.9×`** faster than single-core CPU RASPA3's `117.98 s` (`151×151×46`).
  - Near-field `Mg–O` radial MAE (`2.3–3.8 Å`) drops **`290.3×`**: **`190.46 K`** (`128`-Trilin) → **`12.53 K`** (`256`-Trilin) → **`0.656 K`** (`256`-Tricubic); Mercado 2016 drops `4.99 K → 0.092 K`.
  - Pore MAE (`1,000` pts vs. direct Ewald $\alpha_{\text{ref}} = 0.25\ \text{\AA}^{-1}$): `U_lj_C` drops `0.793 K → 0.061 K`, `U_lj_O` drops `2.387 K → 0.148 K` (Mercado: `C = 0.055 K, O = 0.035 K`).
- **Zero-Drift Double Precision**: Guest coordinates and energies stay in `double` shared memory with full fresh `u_fw + u_gg` periodic recomputation every `50,000` steps — maximum drift across all `128` boxes (`64M` steps/FF) is **`5.879e-09 K`** (UFF) and **`1.024e-08 K`** (Mercado 2016), with detailed balance `max|acc_ins - acc_del| <= 0.000253`.

### 2. Upgrade 2 — Official Mercado 2016 DFT-Derived Buckingham/LJ + EPM2 Force Field
- Integrated directly from [`data/mercado2016/`](data/mercado2016/) (`rociomer/DFT-derived-force-field`, *J. Phys. Chem. C* 2016, 120, 12590; `Mg_O_CO2` Buckingham potential with exact turnover clamp + shifted `r_cut = 11.0 Å` + Mercado partial charges + EPM2 $\text{CO}_2$).
- **Low-Pressure Open-Metal Physics (`0.15 bar`, 298 K)**: Low-loading $Q_{\text{st}}$ rises from `26.94 kJ/mol` (UFF) to **`35.65 kJ/mol`**, ideal-crystal uptake rises from `1.43` to **`7.15 mmol/g`** (`11.11 mmol/g` at `1.0 bar`), and the radial $\text{Mg}^{2+}\text{–O}_{\text{CO}_2}$ peak contracts **`-0.25 Å`** inward (`2.775 Å → 2.525 Å`), illustrating both sharper metal coordination and why idealized rigid 100%-accessible crystals overshoot defect-limited experiments (`3.6–4.6 mmol/g @ 0.15 bar`, `7.5–8.6 mmol/g @ 1.0 bar`; see `×0.80` effective-site curve).

### 3. Upgrade 3 — Dual Head-to-Head Verification vs. Native `RASPA3 3.0.29` (Exact Ewald & Tricubic Grid)

| Pressure | Step 1 `0.20 Å` Trilin | `v2` `0.10 Å` Trilin | **`v2` `0.10 Å` Tricubic** | **`RASPA3` Exact Ewald** | **`RASPA3` Tricubic Grid** | `v2` vs. Grid ($z$) | $\Delta Q_{\text{st}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0.15 bar`** | `1.1724` | `1.3404` | **`1.4276 ± 0.0123`** | `1.4087 ± 0.0328` (`+1.34%`) | `1.3952 ± 0.0719` | **`+2.32%`** (`0.44σ`) | `+0.14 kJ/mol` |
| **`1.00 bar`** | `6.7019` | `7.3513` | **`7.6349 ± 0.1576`** | `7.3384 ± 0.1329` (`1.44σ`) | `7.5303 ± 0.1156` | **`+1.39%`** (`0.54σ`) | `-0.38 kJ/mol` |
| **`5.00 bar`** | `12.6165` | `13.0857` | **`13.1862 ± 0.0221`** | `13.3048 ± 0.2118` (`-0.89%`) | `13.2402 ± 0.1871` | **`-0.41%`** (`0.29σ`) | `+0.54 kJ/mol` |

- **Transparent Dual Speedup**: GPU `v2` Tricubic achieves **`598,669 MC-steps/s`** (`64M` steps in `106.90 s`), delivering **`153.2×`** throughput over single-core `RASPA3 3.0.29` running the **same tricubic grid algorithm** (`3,908 MC-steps/s`; `~10–20×` normalized against 8–16 CPU threads), and **`492.6×`** over single-core `RASPA3` exact Ewald without grids (`1,215 MC-steps/s`).

---

## Step 5: Unified JSON/CLI High-Precision Engine (`build\mof_gcmc_cli.exe`)

Converges the `O(h⁴)` 3D tricubic interpolation, Mercado 2016 DFT-FF / UFF support, anisotropic polarization (`Pol-COM` / `Pol-3Site` + Thole coupling + **Level 2 full self-consistent many-body `Pol-SCF`**), binary Identity Swap, and generic CIF parsing (`19` common MOF elements, optional `_atom_site_charge`, P1 fallback) into a single runtime-configurable CLI binary driven by JSON presets ([`configs/`](configs/)):

```powershell
.\build\mof_gcmc_cli.exe --test-scf
.\build\mof_gcmc_cli.exe --config configs\mg_mof74_fluegas_298K_pol3site.json
.\build\mof_gcmc_cli.exe --config configs\mg_mof74_co2_298K_scf.json
.\build\mof_gcmc_cli.exe --config configs\mg_mof74_co2_298K_mercado.json --temp 313 --out build\psa_313K.csv
```

| JSON Preset (`0.10 Å` Tricubic, 4 Replicas) | Condition | Verified Output Highlights | Max Drift ($T$-full recompute) |
| :--- | :---: | :--- | :---: |
| [`mg_mof74_co2_298K_uff.json`](configs/mg_mof74_co2_298K_uff.json) | `298 K`, UFF, $\text{CO}_2$ | `1.3987 / 7.3723 / 13.2996 mmol/g` (`+0.25% / -2.10% / +0.45%` vs. RASPA3 Grid) | `4.66e-09 K` |
| [`mg_mof74_co2_298K_mercado.json`](configs/mg_mof74_co2_298K_mercado.json) | `298 K`, Mercado 2016, $\text{CO}_2$ | `7.0587 / 11.1077 / 14.9680 mmol/g` ($Q_{\text{st}} = 35.63\ \text{kJ/mol}$ @ `0.15 bar`) | `4.37e-09 K` |
| [`mg_mof74_co2_313K_psa.json`](configs/mg_mof74_co2_313K_psa.json) | `313 K`, Mercado 2016, $\text{CO}_2$ | `5.0424 / 9.7915 / 13.3012 mmol/g` (clean `-2.02 / -1.32 / -1.67` thermal swing) | `4.13e-09 K` |
| [`mg_mof74_fluegas_298K_pol3site.json`](configs/mg_mof74_fluegas_298K_pol3site.json) | `298 K`, `Pol-3Site`, $\text{CO}_2/\text{N}_2$ (`15:85`) | `S(CO2/N2) = 60.39` (`0.5 bar`), **`64.58`** (`1.0 bar`), **`67.38`** (`2.0 bar`); `double` SMEM dipoles | `2.04e-09 K` |
| [`mg_mof74_co2_298K_scf.json`](configs/mg_mof74_co2_298K_scf.json) | `298 K`, `Pol-SCF`, $\text{CO}_2$ | `2.1322 / 8.3536 / 12.8982 mmol/g` (`+52.4%` @ `0.15 bar` vs. NonPol); mean `16.63` SCF iters | `7.25e-07 K` |
| [`mg_mof74_fluegas_298K_scf.json`](configs/mg_mof74_fluegas_298K_scf.json) | `298 K`, `Pol-SCF`, $\text{CO}_2/\text{N}_2$ (`15:85`) | `S(CO2/N2) = 37.28` (`1.0 bar`), **`43.24`** (`2.0 bar`) with mutual depolarization; `14.78` iters | `2.69e-07 K` |

> **Level 2 Many-Body `Pol-SCF` (`polarization: "scf"`)**: Block-collective shared-memory Applequist-Thole SCF solver (`SCF_TOL = 1e-10 e·Å`, $O(N)$ incremental permanent field $\mathbf{E}_0$ update, warm-started induced dipoles, pre-SCF repulsion filter, and `s_scf_conv` convergence guard) solving $\boldsymbol{\mu}_i = K_E^{-1}\boldsymbol{\alpha}_i(\mathbf{E}_i^{\text{fw}} + \sum_{j\neq i}\mathbf{E}_{j\to i}^{\text{DSF,perm}} + \sum_{j\neq i}\mathbf{T}_{ij}^{\text{Thole}}\boldsymbol{\mu}_j)$ across every trial move (`1.83e-08 K` & `1.13e-11 e·Å` match against exact `36×36` Python matrix inversion in `--test-scf`).

![Level 2 Pol-SCF Visualization](assets/step5_level2_scf_polarization.png)

---

## Build & Run

```powershell
# Compile all 4 targets (mof_gcmc, mof_gcmc_mix, mof_gcmc_v2, mof_gcmc_cli) for RTX 5070 (sm_120)
cmd /c build.bat all

# Step 1 + Step 2 single-component runs & plots
.\build\mof_gcmc.exe
uv run --with matplotlib --with numpy python scripts\verify_physics.py
uv run --with matplotlib --with numpy python scripts\verify_step2.py

# Step 3 binary competitive adsorption (CO2/N2 & CO2/CH4) + PR check & plot
.\build\mof_gcmc_mix.exe
uv run --with matplotlib --with numpy python scripts\verify_step3.py

# Step 4 Tricubic 0.10 A UFF + Mercado 2016 + native RASPA3 (Exact & Tricubic Grid) & plot
cmd /c build.bat v2
.\build\mof_gcmc_v2.exe
uv run python scripts\run_raspa3_bench.py
uv run python scripts\run_raspa3_grid_bench.py
uv run --with matplotlib --with numpy python scripts\verify_upgrades.py

# Step 5 Unified JSON/CLI high-precision engine & 4-preset + Level-2 Pol-SCF automated verification & plot
cmd /c build.bat cli
uv run --with numpy python scripts\verify_cli.py
uv run --with numpy python scripts\verify_scf_pol.py --check-only
uv run --with matplotlib --with numpy python scripts\plot_scf_pol.py
```

---

## Repository Layout

```text
cuda-mof-gcmc/
├── configs/                      # JSON presets for unified CLI engine (UFF, Mercado, PSA 313K, Flue Gas)
├── data/
│   ├── MgMOF-74.cif              # Official RASPA2 R-3 Mg-MOF-74 crystal structure
│   └── mercado2016/              # Official Mercado et al. (JPCC 2016) DFT-FF & EPM2 definitions
├── include/
│   ├── cell.hpp                  # Hexagonal/triclinic cell transform & exact 5-candidate MIC
│   ├── forcefield.hpp            # UFF (19 MOF elements), REPEAT, Mercado 2016, TraPPE/EPM2 & PR EOS
│   ├── grid3d.hpp                # Periodic 3D grids, trilinear & O(h^4) tricubic lookup
│   ├── cif_loader.hpp            # Generic + Mg-MOF-74 CIF symmetry expander & geometry checker
│   ├── simple_json.hpp           # Zero-dependency C++17 JSON parser
│   ├── gcmc_gpu.hpp              # Step 1/2 grid builder & single-component batched GCMC API
│   ├── mix_gcmc.hpp              # Step 3 multi-species 8-grid builder & binary GCMC API
│   ├── v2_gcmc.hpp               # Step 4 high-precision 0.10 A Tricubic UFF & Mercado 2016 API
│   └── unified_gcmc.hpp          # Step 5 unified Tricubic + Pol + Swap GPU solver API
├── src/
│   ├── cif_loader.cpp            # Generic & Mg-MOF-74 CIF symmetry expansion & charge/mass check
│   ├── grid_builder.cu           # Step 1/2 GPU 3D LJ + Ewald potential & E-field precomputation
│   ├── gcmc_kernel.cu            # Step 1/2 batched GCMC (128 threads/box, DSF + Thole coupling)
│   ├── main.cu                   # Step 1/2 driver (NonPol vs Pol-COM vs Pol-3Site)
│   ├── mix_grid_builder.cu       # Step 3 GPU 8-grid precomputation (C_CO2, O_CO2, N_N2, CH4 + E-field)
│   ├── mix_gcmc_kernel.cu        # Step 3 binary GCMC (5 moves incl. Identity Swap + Thole coupling)
│   ├── main_mix.cu               # Step 3 driver (Ideal-gas & swap checks + CO2/N2 & CO2/CH4 sweeps)
│   ├── grid_builder_v2.cu        # Step 4 256x256x72 Buckingham/LJ + Ewald grid builder & stats
│   ├── gcmc_kernel_v2.cu         # Step 4 double-SMEM Tricubic + 50k periodic recompute + Mg-O RDF
│   ├── main_v2.cu                # Step 4 driver (UFF vs Mercado 2016 0.10 A Tricubic, 64M steps each)
│   ├── unified_grid_builder.cu   # Step 5 unified 8-grid GPU builder (UFF & Mercado 2016)
│   ├── unified_gcmc_kernel.cu    # Step 5 unified full-double Tricubic + Pol-3Site + Identity Swap kernel
│   └── main_cli.cu               # Step 5 JSON/CLI entrypoint
├── scripts/
│   ├── verify_physics.py         # Independent MIC/PR checks & Step 1 300 DPI plot
│   ├── verify_step2.py           # Step 2 polarization verification & 300 DPI comparison plot
│   ├── verify_step3.py           # Binary PR EOS verification & Step 3 300 DPI comparison plot
│   ├── run_raspa3_bench.py       # Native Windows RASPA3 3.0.29 parallel exact-Ewald runner
│   ├── run_raspa3_grid_bench.py  # Native Windows RASPA3 3.0.29 parallel tricubic-grid runner
│   ├── verify_upgrades.py        # Step 4 automated assertions & 300 DPI validation figure
│   ├── verify_cli.py             # Step 5 end-to-end 4-preset CLI verification suite
│   ├── verify_scf_pol.py         # Level 2 exact 36x36 matrix inversion & Pol-SCF GCMC test
│   └── plot_scf_pol.py           # Level 2 Pol-SCF 300 DPI 3-panel visualization script
├── assets/
│   ├── step1_mof74_co2_298K.png          # Step 1: 298 K isotherm, Qst & pore density
│   ├── step2_mof74_pol_comparison.png    # Step 2: NonPol vs Pol-COM vs Pol-3Site comparison
│   ├── step3_mof74_binary_coadsorption.png # Step 3: CO2/N2 & CO2/CH4 selectivity & competitive map
│   ├── step4_validated_mercado_raspa3.png  # Step 4: 0.10 A vs RASPA3 vs Mercado 2016 validation
│   └── step5_level2_scf_polarization.png   # Step 5: Level 2 Pol-SCF isotherm, screening & 36x36 check
├── build.bat                     # VS2019 + nvcc (sm_120) build script (all / single / mix / v2 / cli)
├── LICENSE                       # MIT License
├── README.md                     # English documentation
└── README.zh-CN.md               # Chinese documentation
```
