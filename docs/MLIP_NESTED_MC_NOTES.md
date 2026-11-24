**English** | [简体中文](MLIP_NESTED_MC_NOTES.zh-CN.md) | [Back to README](../README.md)

# Hybrid GPU-GCMC + MLIP Nested Monte Carlo: Verified Mechanics & Stage-1 Validation Notes

> **Status: Active Research Engine (Stage-1 Validation Complete; DFT Reference & Fine-Tuning Pending)**. The CUDA proposal engine, chemical-potential reweighting math, batched MLIP evaluation, `float32` vs. `float64` parity, supercell finite-size checks, NIST ISODB experimental tables, equilibrated `HKUST-1` and `IRMOF-1` baselines, and `1,200`-cycle differential $Q_{\text{st}}^{\text{diff}}$ runs below are complete. Periodic DFT single-point ground-truth decomposition and optional MLIP fine-tuning remain open (see [Remaining Open Work](#4-remaining-open-work-stage-2--stage-3)).

---

## 1. Verified Engine Mechanics & Numerical Proofs

### Phase 1: Latency & Correlation Probe ([`scripts/experiments/phase1_mlip_probe.py`](../scripts/experiments/phase1_mlip_probe.py))
![Phase 1 MLIP Nested MC Probe](../assets/phase1_mlip_nested_mc_probe.png)

- On `Mg-MOF-74` (`1×1×2`, `324` framework atoms, $c = 13.884$ Å $> 2 \times 6.0$ Å MACE cutoff) on RTX 5070 (`PyTorch cu128` + `mace-torch 0.3.17` + `torch-dftd 0.5.3`):
  - Single-point `MACE-MP-0 (Medium) + D3` ASE latency: `107.72 ms/eval` (`1 CO₂`) and `129.89 ms/eval` (`16 CO₂`; `206.03 ms/eval` on `1×1×4`).
  - Across `60` non-overlapping pore poses vs. `MACE-MP-0 + D3`, **`Mercado 2016 DFT-FF`** achieves the highest inner-proposer correlation ($R^2 = 0.911$, $\sigma(\Delta\Delta E) = 3.42\ \text{kJ/mol}$, theoretical 1-molecule Nested MC acceptance `84.4%`) over `UFF + REPEAT` ($R^2 = 0.719$) and `UFF + Pol-3Site` ($R^2 = 0.755$).

### Phase 2: Exact Reweighting & Zero-Bias Null Proof ([`src/proposer_dll.cu`](../src/proposer_dll.cu), [`scripts/experiments/verify_nested_mc.py`](../scripts/experiments/verify_nested_mc.py))
![Phase 2 Nested MC Verification](../assets/phase2_nested_mc_verification.png)

- Each outer cycle executes $K$ strictly reversible CUDA GCMC moves (`DELTA_TRANS = 1.0 Å`, `MAX_ANGLE = 0.55 rad`, full double-precision end-of-burst energy recomputation) and accepts/rejects $\mathbf{x}'$ under the target Hamiltonian via Metropolis reweighting on $\Delta U_{\text{tgt}} - \Delta U_{\text{ref}}$.
- **`proposer_eval_energy` Parity**: Matches kernel `U_ref_K` to **`0.000e+00 K`**.
- **Test 1 (Zero-Bias Null Test, `Pol-3Site -> Mercado 2016`, `298 K, 0.15 bar`, `1×1×2`, 4 boxes)**:
  - Nested MC (`K = 15`, `2,000 eq + 10,000 prod` cycles) yields **`5.9668 ± 0.2072 mmol/g`**, $Q_{\text{st}}^{\text{diff}} = 35.05 \pm 1.27\ \text{kJ/mol}$ (`56.2%` outer acceptance), reproducing Direct `Mercado 2016` GPU GCMC (`200k` steps: **`6.0230 ± 0.0406 mmol/g`**, $Q_{\text{st}}^{\text{diff}} = 36.28 \pm 0.25\ \text{kJ/mol}$) within **`z = 0.27σ`**.
- **Test 2 (Unshifted Live `MACE-MP-0 + D3` Deadlock Diagnostic, `0.15 bar`)**:
  - Without chemical-potential biasing, `MACE+D3` binds stronger than `Mercado 2016`, yielding **`0` accepted deletions** over `450` outer cycles (`N` climbs monotonically from `30` to `41` -> `8.6223 mmol/g`), proving why inner chemical-potential biasing is required.

### Phase 2.5: Three-Stage $\Delta\mu_{\text{ref}}$ Calibration, `4.21×` Batched MACE, FP32 Parity & Supercell Check
![Phase 2.5 Shifted & Batched Nested MC](../assets/phase25_shifted_batched_mace.png)

- **Exact Grand-Canonical Bias (`proposer_set_chem_pot_shift`)** ([`scripts/experiments/verify_phase25_shifted_nested.py`](../scripts/experiments/verify_phase25_shifted_nested.py), [`scripts/run_mlip_gcmc.py`](../scripts/run_mlip_gcmc.py)):
  - Adjusts inner fugacity $f_{\text{ref}} = f_{\text{tgt}} \exp(-\Delta\varepsilon / k_B T)$ via three-stage pre-production calibration (two classical stages capped at `+2.0 kJ/mol` plus a `60–80`-cycle target-ensemble `PilotStage3` refinement) and fixes $\Delta\varepsilon$ during equilibration/production while cancelling it identically in $\exp[-\beta(\Delta U_{\text{tgt}} - \Delta U_{\text{ref}} - \Delta\varepsilon\,\Delta N)]$.
  - **Shifted Null Test (`Pol-3Site -> Mercado`, `Δε = -2.57 kJ/mol`, `2.82× f_ref`)**: `5.8723 ± 0.0526` vs. `6.0230 ± 0.0406 mmol/g` (`z = 2.27σ`, $Q_{\text{st}}^{\text{diff}} = 35.55 \pm 0.63$ vs. `36.28 ± 0.25 kJ/mol`) with symmetric sampling (`Ins = 1,288 / 4,274`, `Del = 1,262 / 3,511`, `57.5%` acceptance).
- **`FastMaceD3Target` (`4.21×` Speedup) & FP32 vs. FP64 Parity** ([`scripts/experiments/verify_fp32_parity.py`](../scripts/experiments/verify_fp32_parity.py)):
  - Micro-batched `no_grad` MACE (`compute_force=False`, `mace_chunk=2` for `<550` atoms, `1` for `>=550` atoms) + cached static-framework GPU edge lists for `TorchDFTD3Calculator`: cuts 4-box inference from `157.7 ms/box` (ASE) to **`37.5 ms/box`** (`149.9 ms/batch`), matching ASE within `0.0025–0.0082 kJ/mol/CO2`.
  - Across `15` `Mg-MOF-74 1×1×2` snapshots (`N = 1, 13–16, 33–37 CO₂`), live FP32 `FastMaceD3Target` matches full FP64 `MACE + TorchDFTD3`:
    - `MACE-MatPES-r2SCAN-0 + D3(r2scan)`: max total `|ΔU| = 0.2058 kJ/mol` (`RMS = 0.0951`), max per-molecule `|ΔU/N| = 0.0940 kJ/mol/CO2`, max pairwise transition error `0.3500 kJ/mol` (`0.141 kB*T @ 298 K`).
    - `MACE-MP-0 (Medium) + D3(pbe)`: max total `|ΔU| = 0.4650 kJ/mol`, max per-molecule `0.3130 kJ/mol/CO2`, max pairwise transition error `0.4712 kJ/mol` (`0.190 kB*T @ 298 K`).
- **Supercell Finite-Size Convergence (`1×1×2` vs. `1×1×4`, `298 K, 0.01 bar`, [`scripts/experiments/paper_exp4_finite_size.py`](../scripts/experiments/paper_exp4_finite_size.py))**:
  - Direct `Mercado`: `0.8018 ± 0.0115` (`324` atoms, $r_{\text{cut}} = 6.66$ Å) vs. `0.8030 ± 0.0337 mmol/g` (`648` atoms, $r_{\text{cut}} = 10.86$ Å), **`z = 0.04σ`**.
  - Live `MACE-MP-0 + D3`: `9.3398 ± 0.3610` vs. `9.0615 ± 0.2539 mmol/g`, **`z = 0.63σ`** (`Molar enthalpy -<U>/<N>+RT = 44.99` vs. `45.01 kJ/mol`).

---

## 2. Mechanistic Audit of Universal MACE Models

![MLIP Mechanism & Overbinding Audit](../assets/paper_fig3_mlip_mechanism_audit.png)

Tested across 4 MACE generations in [`scripts/experiments/phase3_model_probe.py`](../scripts/experiments/phase3_model_probe.py) and [`scripts/experiments/paper_exp1_model_mechanism.py`](../scripts/experiments/paper_exp1_model_mechanism.py):

- **Far-Field Zero-Offset Proof (`Mg-MOF-74` pore center**, $r_{\min} = 7.31$ Å $> 6.0$ Å MACE cutoff):
  - Raw MLIP without D3: `MP-0` (`+0.01`), `MPA-0` (`+0.00`), `MP-0b3` (`-0.21`), `MatPES-r2SCAN-0` (`-0.03 kJ/mol`); with D3 dispersion tail: `-2.38` to `-2.56 kJ/mol`. This confirms zero far-field baseline offset; differences arise from short-range interaction physics and D3 parameterizations.
- **Short-Range Collapse in `MPA-0` / `MP-0b3` (`Mg²⁺–O=C=O` axial scan & `N ≈ 25` pore ensemble)**:

| Model | Unobstructed `Mg²⁺` Radial Min (`r_min`) | Loaded Mean `-U/N` (`kJ/mol`) | Diagnostic Finding |
| :--- | :---: | :---: | :--- |
| **`MACE-MP-0 + D3(PBE)`** | `-48.74 kJ/mol` (`2.20 Å`) | `41.63` | Plausible single-site well; PBE-D3 overbinds dense pore states |
| **`MACE-MPA-0 + D3(PBE)`** | **`-93.98 kJ/mol`** (`2.35 Å`) | **`82.84`** | Severe short-range collapse (`-75.26` No-D3) |
| **`MACE-MP-0b3 + D3(PBE)`** | **`-105.17 kJ/mol`** (`2.20 Å`) | **`92.15`** | Severe short-range collapse (`-85.48` No-D3) |
| **`MACE-MatPES-r2SCAN-0 (No D3)`** | `-35.39 kJ/mol` (`2.35 Å`) | `25.47` | Underbinds without dispersion tail |
| **`MACE-MatPES-r2SCAN-0 + D3(r2scan)`** | **`-46.45 kJ/mol`** (`2.20 Å`) | **`40.13`** | Closest static match to OMS well, yet overpredicts isotherms (see Sec. 3–4) |

---

## 3. Curated NIST ISODB Reference & Four-MOF Benchmarks (`298 K` & `313 K`)

![Stage-1 Four-MOF Benchmark vs. NIST ISODB](../assets/stage1_four_mof_nist_isodb.png)

All experimental comparisons below use unmodified `298 K CO₂` gravimetric tables downloaded via [`scripts/experiments/fetch_nist_isodb.py`](../scripts/experiments/fetch_nist_isodb.py) into [`data/exp_isotherms/manifest.json`](../data/exp_isotherms/manifest.json):
- `Mg-MOF-74`: DOI `10.1021/acs.jpcc.7b08239` (`n=16`, `1e-6–9.995 bar`) & `10.1039/c5dt03137k` (`n=24`)
- `Ni-MOF-74`: DOI `10.1021/acs.jpcc.5b03700` (`n=18`, `0.0101–1.01 bar`) & `10.1080/15422119.2014.884507` (`n=9`)
- `HKUST-1`: DOI `10.1002/aic.15837` (`n=79`, `0.0137–1.05 bar`)
- `IRMOF-1`: DOI `10.1021/acs.jctc.5b00093` (`n=30`, `2.074–42.18 bar`) & `10.1038/srep12045` (`n=11`)

> **Thermodynamic Distinction**: Fluctuation heat $Q_{\text{st}}^{\text{diff}} = RT - \text{Cov}(U, N)/\text{Var}(N)$ measures the **differential** isosteric heat of adding an $(N+1)$-th molecule, whereas $-\langle U/N \rangle + RT$ measures the **integral** mean molar enthalpy over all $N$ molecules. Above 1:1 OMS saturation (`~8.24 mmol/g` in `Mg-MOF-74`), secondary pore sites lower differential $Q_{\text{st}}^{\text{diff}}$ while integral enthalpy stays higher—these are distinct physical quantities.

### A. Non-OMS Positive Control: `IRMOF-1 (MOF-5)` ([`data/IRMOF-1.cif`](../data/IRMOF-1.cif), `424` atoms, `298 K`)
- Classical zero-charge `UFF` yields `0.3554 / 0.7447 / 1.5457 / 4.5386 mmol/g` (`0.5 / 1.0 / 2.0 / 5.0 bar`, `Qst = 12.62–14.38 kJ/mol`), closely matching NIST ISODB `10.1021/acs.jctc.5b00093` (`2.074 bar -> 1.838`, `5.240 bar -> 4.917 mmol/g`).
- In contrast, **`MACE-MatPES-r2SCAN-0 + D3(r2scan)`** (`UFF` proposer, `360 prod` cycles) predicts **`1.2886 ± 0.1831 / 2.1874 ± 0.1978 / 5.2688 ± 0.5154 / 12.2190 ± 0.5134 mmol/g`** (integral molar enthalpy `~18.4–19.0 kJ/mol`, **2.5–2.9× higher than experiment**). Because `IRMOF-1` has **no open metal sites**, this isolates a systematic `+4–5 kJ/mol/CO2` background overbinding in the `r2SCAN-0 + D3(r2scan)` setup itself.

### B. Cross-Topology Equilibrated `HKUST-1` ([`data/Cu-BTC.cif`](../data/Cu-BTC.cif), `PilotStage3`, `480 prod` cycles, `298 K`)
- With `PilotStage3` target-ensemble bias refinement (`-0.28 / +0.90 / +1.34 kJ/mol`), `HKUST-1` (`UFF` proposer -> `r2SCAN-0 + D3(r2scan)`) equilibrates monotonically with balanced outer exchange (overturning the earlier unconverged `200`-cycle `1.27 mmol/g` plateau artifact):
  - `0.15 bar`: **`1.2199 ± 0.1482 mmol/g`** (`Ins=69/197, Del=56/1271`, $Q_{\text{st}}^{\text{diff}} = 25.49 \pm 3.67$, integral `25.51 ± 0.49 kJ/mol`) vs. NIST ISODB `0.147 bar`: `0.880 mmol/g`.
  - `0.50 bar`: **`2.3279 ± 0.1678 mmol/g`** (`Ins=79/220, Del=60/1263`, $Q_{\text{st}}^{\text{diff}} = 21.46 \pm 3.61$, integral `23.91 kJ/mol`) vs. NIST ISODB `0.505 bar`: `2.670 mmol/g`.
  - `1.00 bar`: **`4.0984 ± 0.0657 mmol/g`** (`Ins=76/182, Del=52/1303`, $Q_{\text{st}}^{\text{diff}} = 24.34 \pm 0.62$, integral `23.53 kJ/mol`) vs. NIST ISODB `0.998 bar`: `4.530 mmol/g`.
- Remarkably, static axial probing along the `Cu²⁺` paddlewheel still yields a weak `-10.07 kJ/mol` minimum (`r2SCAN+D3`) / `-7.72 kJ/mol` (`MP-0+D3`), indicating partial error cancellation between weak `Cu²⁺` axial binding and strong framework/cage dispersion (`~24 kJ/mol` integral heat).

### C. `Mg-MOF-74` (`1,200 prod` cycles) & `Ni-MOF-74` (`298 K` & `313 K`)

| MOF & Condition | Uptake (mmol/g) | Differential $Q_{\text{st}}^{\text{diff}}$ (Fluc) | Integral $-\langle U/N \rangle + RT$ | `Prod Ins / Del` Acc | NIST ISODB (`298 K`) Comparison |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`Mg-MOF-74`, `298 K, 0.01 bar` (`1200 prod`)** | `6.3407 ± 0.4486` | `50.28 ± 1.81 kJ/mol` | `48.07 ± 0.25 kJ/mol` | `133/590`, `112/435` | `0.0101 bar`: `2.102 mmol/g` (MLIP `~3.0×` high) |
| **`Mg-MOF-74`, `298 K, 0.05 bar` (`1200 prod`)** | `9.4714 ± 0.2595` | `36.31 ± 1.97 kJ/mol` | `45.77 ± 0.27 kJ/mol` | `182/573`, `171/823` | `0.0315 bar`: `4.410`, `0.0992 bar`: `5.867` |
| **`Mg-MOF-74`, `298 K, 0.15 bar` (`480 prod`)** | `11.4721 ± 0.1519` | `31.97 ± 4.23 kJ/mol` | `43.17 ± 0.32 kJ/mol` | `71/199`, `72/302` | `0.099–0.311 bar`: `5.87–7.04` (MLIP `~1.8×` high) |
| **`Mg-MOF-74`, `313 K, 0.01 bar` (`480 prod`)** | `3.9595 ± 0.6894` | `42.59 ± 6.03 kJ/mol` | `46.81 ± 0.44 kJ/mol` | `46/338`, `39/135` | Thermal desorption `-2.38 mmol/g` vs. `298 K` |
| **`Ni-MOF-74`, `298 K, 0.05 bar` (`480 prod`)** | `1.4671 ± 0.1705` | `35.57 ± 1.30 kJ/mol` | `36.48 ± 0.66 kJ/mol` | `75/182`, `55/757` | `0.0512 bar`: `2.091 mmol/g` |
| **`Ni-MOF-74`, `298 K, 0.15 bar` (`480 prod`)** | `4.2921 ± 0.3457` | `35.23 ± 2.74 kJ/mol` | `36.98 ± 0.50 kJ/mol` | `79/170`, `59/876` | `0.1512 bar`: `4.456 mmol/g` |
| **`Ni-MOF-74`, `313 K, 0.05 bar` (`480 prod`)** | `1.1449 ± 0.2693` | `38.55 ± 1.71 kJ/mol` | `37.14 ± 0.39 kJ/mol` | `74/270`, `60/641` | Thermal desorption `-0.32 mmol/g` vs. `298 K` |

Notice how `1,200` outer MCMC cycles on `Mg-MOF-74` at `0.05 bar` cut the fluctuation $Q_{\text{st}}^{\text{diff}}$ SEM by **5.4×** (from `±10.56` down to **`±1.97 kJ/mol`**) and cleanly resolve differential heat dropping from `50.28` (`0.01 bar`) to **`36.31 kJ/mol`** (`0.05 bar`, above 1:1 OMS saturation `~8.24 mmol/g`), while integral enthalpy remains `45.77 kJ/mol`.

![Stage-1 FP32 Parity, Differential Qst Convergence & Thermal Desorption](../assets/stage1_qst_fp32_diagnostics.png)

---

## 4. Remaining Open Work (Stage 2 / Stage 3)

1. **Periodic DFT Single-Point Ground Truth & D3 Parameter Audit (Priority Next)**:
   - Even with accurate NIST ISODB references (`Mg-MOF-74`: `2.10–7.04 mmol/g`; `IRMOF-1`: `1.84–4.92 mmol/g`), `MACE-MatPES-r2SCAN-0 + D3(r2scan)` systematically overpredicts uptake on both `Mg-MOF-74` (`~1.8–3.0×`) and non-OMS `IRMOF-1` (`~2.5–2.9×`). Representative snapshots across `Mg-MOF-74`, `Ni-MOF-74`, `HKUST-1`, and `IRMOF-1` must be evaluated with true periodic DFT (`r2SCAN`, `r2SCAN+D3`, `PBE+D3`, plus spin-polarized `Cu²⁺` checks) and audited against `torch-dftd`'s `r2scan` damping parameters to separate MLIP error, XC bias, and dispersion double-counting.
2. **Optional Small-Sample DFT Fine-Tuning / Δ-Learning**:
   - Once periodic DFT reference labels are computed, lightweight MACE fine-tuning or a Δ-learning residual head can be plugged directly into [`scripts/run_mlip_gcmc.py`](../scripts/run_mlip_gcmc.py) to test physical isotherm recovery.
3. **Physical Model Scope Boundaries (Structural Assumptions)**:
   - Framework and adsorbate geometries are strictly rigid (omitting framework breathing and CO₂ bending), and classical `Pol-SCF` adds many-body Thole induction directly onto unpolarized TraPPE/UFF parameters (induction double-counting relative to polarizable-fit force fields).
