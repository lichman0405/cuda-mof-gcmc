**English** | [简体中文](TECHNICAL_REPORT.zh-CN.md) | [Back to README](../README.md) | [Classical & polarizable validation](CLASSICAL_VALIDATION.md)

# Nested Monte Carlo with Machine-Learned Potentials Using an Online-Distilled Surrogate Proposer: a GPU Implementation for Gas Adsorption in Rigid MOFs

*Preprint draft, November 2026. Every number below can be regenerated from [`docs/results/phase2_phase3_summary.json`](results/phase2_phase3_summary.json) with [`scripts/plot_phase4_figures.py`](../scripts/plot_phase4_figures.py).*

## Abstract

Running grand-canonical Monte Carlo (GCMC) directly on a machine-learned interatomic potential (MLIP) is expensive for two reasons: every insertion or deletion requires a neural-network evaluation, and in strongly binding regions such as open metal sites almost every random insertion is rejected. The usual workaround is nested Monte Carlo, in which a classical force field generates proposals that are then reweighted with the MLIP, but the classical force field has to be refitted for every framework and guest. We describe an alternative that needs no hand-built force field. For each (framework, guest, target MLIP) combination, a surrogate consisting of a one-body (1B) 3D grid and a two-body (2B) residual table is fitted online to MLIP energies of 1,200 single-guest poses and 192 in-pore dimers, which takes 56–210 s. The surrogate then drives a configurational-bias MC on the GPU, with 128 Rosenbluth trials per CUDA thread block, and every 16 inner moves the candidate configurations are scored with a batched MLIP call and an exact Metropolis correction.

We tested the method on four MOFs (Mg-MOF-74, Ni-MOF-74, Cu-BTC, IRMOF-1) and two guests (CO₂, CH₄). (1) For Mg-MOF-74/CO₂ at 0.28 bar and a fixed target (`mace-matpes-r2scan-0` without D3), direct MLIP-GCMC reached a box-averaged uptake of only 1.76 mmol/g after 300 s, and a generic UFF proposer gave 2.9% outer acceptance. Our sampler converged to 6.32 ± 0.23 and 6.31 ± 0.17 mmol/g from empty and preloaded starts, with an efficiency comparable to the system-specific Mercado-2016 force field. (2) Across the seven surrogates used in production, the 1B hold-out R² ranges from 0.975 to 0.990. (3) Agreement with experiment is set mainly by the target MLIP. r2SCAN is close to experiment for Mg-MOF-74 at 1 bar (8.15 vs. 7.02–8.16 mmol/g) and IRMOF-1 at 5 bar (3.74 vs. 4.70 mmol/g), but underestimates Ni-MOF-74 and Cu-BTC by about an order of magnitude; MP0+D3 overestimates Mg-MOF-74 by a factor of 2.4, with 51%–57% of its binding energy coming from the D3 term. (4) Six of the 16 production points pass all convergence criteria ($\hat{R}<1.01$, bulk ESS ≥ 400, etc.); every failure is listed. (5) On four RTX 3090 GPUs, replica sharding speeds up the MC phase by 3.26×, and a pressure queue speeds up end-to-end runs by 2.19×.

---

## 1 Introduction

Foundation MLIPs such as MACE-MP-0 [1,2] and MACE models trained on the MatPES r2SCAN data [3] make it possible, in principle, to simulate adsorption in MOFs without fitting a force field for each framework. The difficulty is cost. GCMC needs very many insertion and deletion attempts, a MACE evaluation is several orders of magnitude more expensive than a grid lookup, and in materials with open metal sites such as Mg-MOF-74 the acceptance of random insertions is low. Together these make direct MLIP-GCMC slow to converge.

Nested Markov chain Monte Carlo [6,7] is the standard way around this. A cheap reference potential drives a short inner chain, and the end point is accepted or rejected with a Metropolis test on the target potential. As long as the inner chain is reversible with respect to the reference distribution, the outer chain samples the target distribution exactly, and the quality of the reference only affects efficiency. In practice the hard part is finding a good reference. Classical force fields are either poor at open metal sites (UFF [12]) or valid only for the system they were fitted to (for example the periodic-DFT force field of Mercado et al. for Mg-MOF-74 [10]).

Here the reference potential is distilled directly from the target MLIP, without any manual atom typing. Because the reference and target are aligned by construction, the outer acceptance is governed by the residual many-body difference between them, not by an unrelated classical model. Three sets of experiments ask whether this proposer is actually more efficient than direct MLIP-GCMC and classical proposers (Experiment 1), whether it transfers to other frameworks and guests without code changes (Experiment 2), and whether the method is physically and statistically consistent and scales across GPUs (Experiment 3).

## 2 Methods

![Figure 1](../assets/fig1_architecture.png)

**Figure 1** Method overview. **(a)** Surrogate construction, performed once per (framework, guest, target MLIP). On the right is the Mg-MOF-74/r2SCAN cache used in production: the 1B grid for the CO₂ oxygen channel, minimized along *c* (dark spots are binding sites next to the Mg²⁺ open metal sites), and the 2B centre-of-mass residual table. **(b)** One outer cycle of the nested chain. The inner chain makes 16 moves on the shifted surrogate; the outer step scores the candidate with one batched MLIP call and accepts it or rolls back according to the formula shown. **(c)** The two multi-GPU layouts with measured timings.

### 2.1 Systems, targets and reference data

All frameworks are rigid. Mg-MOF-74 and Ni-MOF-74 use a 1×1×2 supercell (324 atoms for Mg), while Cu-BTC and IRMOF-1 use the conventional cell (624 and 424 atoms). CO₂ is a rigid three-site model and CH₄ a rigid five-site all-atom model. The temperature is 298 K throughout. Three target potentials are used: `mace-matpes-r2scan-0` without dispersion correction (r2SCAN below), MACE-MP-0 medium (MP0), and MP0 with D3(BJ, PBE) dispersion [4,5] (MP0+D3).

Experimental isotherms are those stored in [`data/exp_isotherms/`](../data/exp_isotherms/) (exported from NIST/ISODB; DOIs in refs. 15–22). For comparison they are interpolated linearly in ln *P*, without extrapolation; where no experimental point covers a pressure, no comparison is given.

### 2.2 Online surrogate construction

See Figure 1a. Framework atoms are grouped into 6–8 environment classes by element and neighbour composition. Guest atoms are grouped into symmetry channels by element and distance from the centre of mass (two channels for both CO₂ and CH₄). The surrogate has two parts.

- **One-body term (1B).** For each (guest channel, framework class) pair we use a set of radial basis functions (13 Gaussians placed non-uniformly between 1.75 and 6.15 Å, plus an $r^{-12}$ repulsive core). Electrostatics are represented by charge-neutral Ewald null-space modes acting as effective framework charges. Training data are MLIP interaction energies of 1,200 single-guest poses stratified by their minimum distance to the framework (poses above 1.5×10⁴ K are discarded, leaving 1,002–1,171). The fit is a bound-constrained weighted least-squares problem with Tikhonov regularization that varies by radial range; this prevents cancellation between channels for guests such as CH₄, whose central atom is shielded by its hydrogens. The result is rendered per channel onto a unit-cell grid (240×240×64 for Mg-MOF-74) and interpolated tricubically on the GPU.
- **Two-body term (2B).** 192 in-pore dimers are assembled from low-energy single-guest poses, and $U_2=U_{AB}-U_A-U_B$ is computed with the MLIP. We fit scaling factors for the classical LJ and Coulomb terms plus a 128-point residual table in the centre-of-mass distance (set to zero inside 2.75 Å).

Fifteen percent of the 1B poses are held out to report R² and MAE. Building the surrogate takes 56–210 s on one RTX 3090. The result is cached under the SHA-256 of its inputs and reused for all later pressures of the same combination.

### 2.3 Configurational-bias inner moves on the GPU

Each simulation box is assigned to one CUDA thread block (Figure 1b). In every outer cycle the block performs *k* = 16 inner moves on the surrogate: translation (±1.0 Å), rotation (up to 0.55 rad), insertion and deletion, each chosen with probability 0.25. For an insertion, each of 128 threads generates a random centre-of-mass position and orientation, and one is selected by its Rosenbluth weight [8,14]. For a deletion, thread 0 keeps the molecule being removed and the other 127 threads generate trial configurations, which preserves detailed balance. Framework–guest energies come from the 1B grids; guest–guest energies come from the scaled classical terms plus the 2B table.

### 2.4 Exact outer correction and chemical-potential offset

The inner chain samples a shifted reference distribution with energy $U_{\rm ref}(x)+\Delta\varepsilon N$. The outer step evaluates $U_{\rm MLIP}(x')$ for the candidates of all boxes in one batch and accepts with probability

$$\alpha=\min\left\{1,\ \exp\!\left(-\beta\left[\Delta U_{\rm MLIP}-\Delta U_{\rm ref}-\Delta\varepsilon\,\Delta N\right]\right)\right\},$$

where Δ denotes candidate minus current. A rejected candidate is restored in device memory. This rule makes the outer chain sample the grand-canonical distribution of the target MLIP exactly, for any value of Δε. The role of Δε is to align the mean per-molecule energy of the surrogate with that of the target, so that the inner chain sits at roughly the right loading and the acceptance stays high. Δε is set once per pressure: first as the median over boxes of $(U_{\rm MLIP}-U_{\rm ref})/N$ on the starting configurations (the median, rather than the mean, is used when there are at least three boxes, so a single outlier box cannot dominate), then refined with a 40-cycle pilot run.

### 2.5 Convergence criteria and uncertainties

Each pressure is run with four independent boxes (four chains). After production we compute rank-normalized split/folded $\hat{R}$ and bulk/tail ESS [9] for the number of molecules *N*, the total energy *U*, and the open-metal-site occupancy. A point passes only if all of the following hold: $\hat{R}<1.01$; bulk and tail ESS ≥ 400 (100 per chain); MCSE of the uptake ≤ max(0.05 mmol/g, 1%); half-to-half drift of every chain within tolerance; and a production outer acceptance that is not too low. The reported uptake uncertainty is the standard error over boxes. Two estimates of $Q_{\rm st}$ are given: the fluctuation formula and the mean molar binding energy. For points that fail the criteria, the standard error is computed from an ESS that is too small, so the true uncertainty is probably larger.

### 2.6 Computational details

All MLIP evaluations use float32 and GPU batching. Production runs were performed on one node with eight RTX 3090 GPUs (`sm_86`), of which only GPUs 0–3 were used. Equilibration was 200–1,200 outer cycles and production 1,200–16,500 outer cycles per pressure (see Tables 1–3). For one high-pressure point (IRMOF-1, 5 bar) Δε was set by hand; the reason is given in Section 3.3.

---

## 3 Results

### 3.1 Experiment 1: Mg-MOF-74 + CO₂, proposer efficiency and choice of target

**Purpose.** The Mg²⁺ open metal sites of Mg-MOF-74 bind CO₂ strongly, and the isotherm rises steeply between 0.01 and 0.3 bar. This makes it one of the hardest cases for GCMC sampling. We first compare five sampling strategies on the same target, to measure what the surrogate proposer actually gains, and then change the target while keeping the sampler fixed, to see how sensitive the result is to the MLIP.

**Setup.** The target is r2SCAN without D3, at 0.012, 0.28 and 1.0 bar. At 0.28 bar we compare: direct MLIP-GCMC with no reference potential (300 s); nested MC with a UFF proposer; nested MC with the Mercado-2016 force field as proposer; and nested MC with our surrogate, started from an empty box and from a preloaded box (1,200 equilibration and 4,800 production cycles each). We also run GCMC on the surrogate alone (no outer correction) and nested MC with MP0 and MP0+D3 as targets.

![Figure 2](../assets/fig2_exp1_mgmof74.png)

**Figure 2** Mg-MOF-74 + CO₂ at 298 K. **(a)** Isotherms on log–log axes. Open symbols are two experimental data sets; solid lines are nested MC and the dashed line is the surrogate alone; error bars are ±1 standard error. **(b)** Box-averaged uptake at 0.28 bar versus wall-clock time on one GPU, for five samplers with the same target. Curves for the surrogate proposer are shifted right by the 56 s distillation time (grey band). The horizontal band is the range of the two experimental data sets at 0.28 bar. Time is obtained assuming a constant cost per cycle. **(c)** Difference between nested MC and surrogate-only uptake; error bars combine the two standard errors in quadrature. **(d)** Uptake at 0.28 bar for three targets with the same sampler; labels give the fluctuation $Q_{\rm st}$.

**Analysis of the figure.**

Figure 2b answers the efficiency question most directly. Direct MLIP-GCMC (black) made 16,269 MLIP calls in 300 s and its box-averaged uptake crept from zero to about 1.8 mmol/g, far below the experimental range. The UFF proposer (purple) describes the Mg²⁺ sites very differently from r2SCAN: Δε calibrated to 3.48 kJ/mol, the outer acceptance was only 2.9%, and the chain stalled near 1.4 mmol/g after 110 s. The Mercado-2016 force field, fitted to DFT specifically for Mg-MOF-74/CO₂, is a much better proposer. It reached the experimental range after about 120 s, with 16.4% outer acceptance and Δε = 5.68 kJ/mol. That run, however, had only 2,200 production cycles and its uptake was still rising (chain drift 1.75 mmol/g), so its time average of 4.87 mmol/g is not an equilibrium value. With the surrogate proposer, after the 56 s distillation, the chain started from an empty box (blue) entered the experimental range within about 100 s of MC time. The chain started from a preloaded box (teal) first fell from about 7 mmol/g and then recovered. Both chains reached the same plateau after about 300 s, with production averages of 6.32 ± 0.23 and 6.31 ± 0.17 mmol/g.

In terms of relaxation speed, our sampler is roughly on par with the Mercado force field (accepted changes in *N* per second: 2.51 vs. 2.89; outer acceptance: 30.8% vs. 16.4%). The difference is that the Mercado force field requires a prior DFT fit for this particular system and cannot be reused for another framework, whereas our reference is obtained automatically from the target MLIP, and the same pipeline is applied unchanged to three other frameworks in Experiment 2.

Figures 2a and 2c show when the outer MLIP correction matters. At 0.012 and 1.0 bar, nested MC and the surrogate alone agree (differences −0.01 and −0.03 mmol/g, within error). Only at 0.28 bar, the steepest part of the isotherm, does nested MC exceed the surrogate, by 2.33 ± 0.30 mmol/g. This difference has two sources: the fitting error of the surrogate (hold-out MAE 2.44 kJ/mol for 1B and 1.92 kJ/mol for 2B for this cache) and three-body and higher interactions that the surrogate cannot represent. Our data cannot separate the two. What they do show is that a 1B+2B surrogate alone underestimates the steep region by about 37%, so the outer correction is necessary. With MP0+D3 the differences are within ±0.3 mmol/g at all three pressures, so for that target the surrogate is already close.

Figure 2d shows that, with the sampler unchanged, the uptake at 0.28 bar goes from 0.66 mmol/g (MP0 without dispersion) to 6.32 mmol/g (r2SCAN) to 18.54 mmol/g (MP0+D3), a range of almost 30×. Without dispersion MP0 clearly underbinds. With D3 it overbinds badly, reaching 15.06 mmol/g already at 0.012 bar, close to pore saturation. Experiment 3 decomposes this energetically.

**Results.** Table 1 collects all Experiment 1 numbers.

**Table 1** Mg-MOF-74 + CO₂ at 298 K. *q* is the uptake (±1 standard error); $Q_{\rm st}$ is given as fluctuation / mean molar estimate; "pass" means all criteria of Section 2.5.

| Target / sampler | *P* (bar) | Equil. + prod. cycles | *q* (mmol/g) | $Q_{\rm st}$ (kJ/mol) | Δε (kJ/mol) | Outer acc. | $\hat{R}$ (N) | Bulk ESS | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| r2SCAN, direct MLIP-GCMC | 0.28 | 300 s | 1.07 ± 0.15 (final ≈ 1.76) | 33.4 / – | – | 3.6% per insertion | 1.995 | 5 | fail |
| r2SCAN, UFF proposer | 0.28 | 250 + 1,200 | 1.36 ± 0.12 | 30.5 / 32.4 | 3.48 | 2.9% | 1.490 | 8 | fail |
| r2SCAN, Mercado-2016 proposer | 0.28 | 250 + 2,200 | 4.87 ± 0.16 | 37.8 / 36.4 | 5.68 | 16.4% | 1.386 | 9 | fail |
| r2SCAN, surrogate proposer, empty start | 0.012 | 250 + 5,200 | 0.179 ± 0.004 | 34.6 / 34.5 | −0.51 | 72.8% | 1.0021 | 1,213 | pass |
| r2SCAN, surrogate proposer, empty start | 0.28 | 1,200 + 4,800 | 6.32 ± 0.23 | 40.5 / 37.1 | 0.58 | 30.8% | 1.209 | 14 | fail |
| r2SCAN, surrogate proposer, preloaded start | 0.28 | 1,200 + 4,800 | 6.31 ± 0.17 | 39.5 / 37.0 | 0.39 | 28.7% | 1.167 | 19 | fail |
| r2SCAN, surrogate proposer | 1.0 | 250 + 15,000 | 8.15 ± 0.03 | 33.1 / 37.1 | 0.03 | 35.1% | 1.0145 | 200 | fail |
| r2SCAN, surrogate only | 0.012 / 0.28 / 1.0 | 250 + 2,200 | 0.193 / 3.99 / 8.18 | 35.5 / 36.1 / 32.5 | – | – | 1.005 / 1.116 / 1.243 | 1,012 / 32 / 13 | pass / fail / fail |
| MP0, surrogate proposer | 0.28 | 200 + 4,800 | 0.659 ± 0.010 | 26.1 / 26.5 | 1.05 | 66.7% | 1.0033 | 911 | pass |
| MP0+D3, surrogate proposer | 0.012 | 200 + 5,000 | 15.06 ± 0.12 | 41.1 / 45.9 | 3.77 | 38.2% | 1.169 | 16 | fail |
| MP0+D3, surrogate proposer | 0.28 | 200 + 1,800 | 18.54 ± 0.14 | 41.5 / 44.9 | 3.23 | 72.7% | 1.590 | 7 | fail |
| MP0+D3, surrogate proposer | 1.0 | 200 + 1,800 | 19.36 ± 0.15 | 43.5 / 44.1 | 2.83 | 75.2% | 1.938 | 6 | fail |
| Experiment (two data sets, interpolated) | 0.012 / 0.28 / 1.0 | – | 1.37–2.45 / 4.74–6.94 / 7.02–8.16 | – | – | – | – | – | – |

Compared with experiment, r2SCAN lies between the two data sets at 0.28 bar and at the upper edge of the experimental range at 1.0 bar, but underestimates the uptake at 0.012 bar by a factor of 8–14. In other words, the r2SCAN isotherm is much flatter than experiment at low pressure and steeper at intermediate pressure. The agreement at 1 bar comes partly from this compensation in shape and should not be read as evidence that the model describes Mg²⁺–CO₂ binding accurately.

On convergence: the two chains at 0.28 bar have almost identical means (difference 0.008 mmol/g), so there is no sign of a dependence on the starting state or of trapping in a metastable state. But both have $\hat{R}$ of 1.17–1.21 and bulk ESS of only 14–19. This pressure sits at the steepest point of the isotherm, where *N* fluctuates strongly (variance about 25) and the autocorrelation time reaches several hundred cycles, so 4,800 production cycles are not enough to meet the criteria. We report it as a failure and consider its true uncertainty larger than the tabulated standard error.

---

### 3.2 Experiment 2: transfer to other frameworks and guests

**Purpose.** To test whether the same pipeline, without code changes or any force-field parameters, can be applied to frameworks with different topology, metal and pore size and to a different guest; and to compare simulation with experiment in a way that separates sampling problems from problems with the target.

**Setup.** Ni-MOF-74 (isostructural with Mg-MOF-74, Ni²⁺ instead of Mg²⁺), Cu-BTC (cubic cages with copper paddle-wheel sites) and IRMOF-1 (large pores, no open metal sites) adsorb CO₂ with r2SCAN as the target. IRMOF-1 + CH₄ uses MP0+D3, for reasons given below. All runs use the parameters of Experiment 1 (*k* = 16, *K* = 128, four boxes).

![Figure 3](../assets/fig3_exp2_transferability.png)

**Figure 3** Other frameworks and guests at 298 K. **(a)–(c)** CO₂ isotherms in Ni-MOF-74, Cu-BTC and IRMOF-1 with r2SCAN (no D3); open symbols are experiments. **(d)** CH₄ isotherm in IRMOF-1 with MP0+D3. Inset: single-molecule CH₄ interaction energies from three MACE models at four in-pore poses (labelled by the minimum distance to a framework atom). **(e)** Simulated versus interpolated experimental CO₂ uptake on log–log axes; the solid line is *y* = *x*, dashed and dotted lines mark factors of 2 and 10; horizontal bars span the two experimental data sets; filled symbols pass all convergence criteria. **(f)** Hold-out errors (15%) of the seven surrogate caches used in production, labelled with the 1B R² and the build time.

**Analysis of the figure.**

Figure 3f underpins transferability. The seven caches cover four frameworks, two guests and three targets; the 1B hold-out R² is 0.975–0.990 and the MAE 1.27–2.44 kJ/mol. The 2B tables have MAEs of 0.10–1.92 kJ/mol. Their R² values are generally poor (−0.68 to 0.76) because the in-pore dimer interaction is only a few kJ/mol and has little variance, so R² is not informative there; the MAE is the more meaningful measure. Build times of 56–210 s are dominated by grid rendering, which is slowest for the 240³ grids of Cu-BTC and IRMOF-1.

In Figures 3a–c the sampling works on all three frameworks. Cu-BTC at 0.15 and 1.0 bar, Ni-MOF-74 at 0.28 bar and IRMOF-1 at 1.0 bar pass every criterion, and Ni-MOF-74 at 1.0 bar fails only because the energy $\hat{R}$ is 1.0118. The gap to experiment, however, is large. Ni-MOF-74 at 1 bar gives 0.83 mmol/g against 7.62 mmol/g measured, and Cu-BTC at 1 bar gives 0.69 against 3.97–4.54 mmol/g. The corresponding heats of adsorption are only 25 kJ/mol (Ni) and 18 kJ/mol (Cu). Since these points are converged, the discrepancy cannot be blamed on sampling; it reflects how r2SCAN describes CO₂ binding at Ni²⁺ and Cu²⁺ sites and the absence of a dispersion correction in this target. IRMOF-1 has no open metal sites and adsorption is dominated by weaker physisorption; there r2SCAN gives 3.74 mmol/g at 5 bar, about 20% below experiment (4.70 mmol/g).

Figure 3e puts these comparisons on one plot. Mg-MOF-74 at 0.28 and 1 bar and IRMOF-1 at 5 bar fall inside the factor-of-two band. Most of the Cu-BTC and Ni-MOF-74 points, and Mg-MOF-74 at 0.012 bar, lie near or beyond the factor-of-ten line. The errors all have the same sign (underestimation) and are largest at open transition-metal sites.

Figure 3d records a diagnosis for CH₄. Our first IRMOF-1 + CH₄ run used r2SCAN, and the boxes filled to their 128-molecule cap at 1 bar, with Δε calibrating to +21 kJ/mol. We first suspected the surrogate fit and corrected a cross-channel collinearity problem specific to CH₄ (commit `df9bff6`), but the behaviour did not change. We then evaluated four in-pore single-molecule poses directly with ASE's `MACECalculator`. r2SCAN gave −26.5 to −35.5 kJ/mol (our batched evaluator agrees within 0.7 kJ/mol), and among random poses the deepest value was −60.0 kJ/mol, where the surrogate grid gives −58.6 kJ/mol. For comparison, MP0 without dispersion gives −2.0 to +2.5 kJ/mol at the same poses and MP0+D3 gives −1.7 to −10.1 kJ/mol. A model with no dispersion correction that binds non-polar CH₄ three to four times more strongly than MP0+D3 is hard to reconcile with physics, and we attribute it to a failure of the r2SCAN model to extrapolate to these configurations. The surrogate faithfully reproduces that behaviour; the problem lies in the target, not in the distillation. The production CH₄ runs therefore use MP0+D3: 3.53 ± 0.02 mmol/g at 1 bar (failing only on an energy bulk ESS of 394.5) and 7.36 ± 0.08 mmol/g at 5 bar (not converged). The repository contains no experimental IRMOF-1 + CH₄ isotherm, so we make no comparison.

**Results.** Table 2 lists every pressure point of Experiment 2, and Table 3 the surrogates used.

**Table 2** Other frameworks and guests at 298 K. Experimental values are interpolated in ln *P*; "–" means no experimental coverage.

| System / target | *P* (bar) | Equil. + prod. cycles | *q* (mmol/g) | Experiment (mmol/g) | $Q_{\rm st}$ (kJ/mol) | Outer acc. | $\hat{R}$ (N) | Bulk / tail ESS | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Ni-MOF-74 / r2SCAN | 0.012 | 250 + 2,200 | 0.009 ± 0.002 | 0.50 | 24.4 / 24.5 | 96.0% | 1.0073 | 649 / 649 | dilute, insufficient samples |
| Ni-MOF-74 / r2SCAN | 0.28 | 250 + 5,600 | 0.200 ± 0.004 | 4.68–5.50 | 25.2 / 25.1 | 65.4% | 1.0038 | 1,118 / 814 | pass |
| Ni-MOF-74 / r2SCAN | 1.0 | 250 + 16,500 | 0.833 ± 0.014 | 7.62 | 25.5 / 24.9 | 41.3% | 1.0083 | 620 / 994 | fail (U $\hat{R}$ = 1.0118) |
| Cu-BTC / r2SCAN | 0.15 | 250 + 4,600 | 0.082 ± 0.002 | 0.30–0.90 | 17.6 / 17.4 | 58.2% | 1.0021 | 2,138 / 986 | pass |
| Cu-BTC / r2SCAN | 1.0 | 250 + 13,000 | 0.689 ± 0.015 | 3.97–4.54 | 18.1 / 18.0 | 18.2% | 1.0069 | 436 / 751 | pass |
| IRMOF-1 / r2SCAN | 1.0 | 200 + 9,600 | 0.824 ± 0.018 | – | 15.8 / 15.7 | 29.0% | 1.0041 | 645 / 647 | pass |
| IRMOF-1 / r2SCAN | 5.0 | 600 + 6,000 | 3.74 ± 0.08 | 4.70 | 16.2 / 16.5 | 4.7% | 1.103 | 30 / 112 | fail (Δε set to 2.5 by hand) |
| IRMOF-1 + CH₄ / MP0+D3 | 1.0 | 250 + 3,200 | 3.53 ± 0.02 | – | 20.1 / 22.7 | 39.8% | 1.0079 | 440 / 557 | fail (U ESS = 394.5) |
| IRMOF-1 + CH₄ / MP0+D3 | 5.0 | 250 + 3,200 | 7.36 ± 0.08 | – | 17.1 / 20.2 | 29.1% | 1.062 | 58 / 139 | fail |

**Table 3** Surrogate caches used in production (15% hold-out).

| Cache | System / target | Env. classes | 1B poses kept | 1B R² | 1B MAE / near-wall MAE (kJ/mol) | 2B MAE (kJ/mol) | Build time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `5e543d05` | Mg-MOF-74, CO₂ / r2SCAN | 8 | 1,026 | 0.975 | 2.44 / 3.85 | 1.92 | 56 |
| `e523efe1` | Mg-MOF-74, CO₂ / MP0+D3 | 8 | 1,044 | 0.985 | 2.00 / 2.62 | 1.17 | 63 |
| `269c6d37` | Mg-MOF-74, CO₂ / MP0 | 8 | 1,019 | 0.986 | 1.57 / 3.13 | 0.78 | 127 |
| `c829b9bb` | Ni-MOF-74, CO₂ / r2SCAN | 8 | 1,002 | 0.987 | 2.16 / 3.01 | 1.65 | 57 |
| `9be1cad1` | Cu-BTC, CO₂ / r2SCAN | 6 | 1,037 | 0.987 | 2.27 / 3.79 | 0.81 | 210 |
| `0ca4d69d` | IRMOF-1, CO₂ / r2SCAN | 7 | 1,057 | 0.990 | 2.13 / 3.13 | 0.56 | 144 |
| `5134add5` | IRMOF-1, CH₄ / MP0+D3 | 7 | 1,171 | 0.987 | 1.27 / 2.05 | 0.10 | 151 |
| `7cfa1abf` (diagnostic only) | IRMOF-1, CH₄ / r2SCAN | 7 | 1,180 | 0.912 | 3.16 / 4.00 | 1.14 | 147 |

---

### 3.3 Experiment 3: physical consistency, parameter choices, multi-GPU efficiency and a convergence audit

**Purpose.** Experiments 1 and 2 give uptakes. This set asks four narrower questions: why MP0+D3 overbinds so strongly; whether the choices *k* = 16 and a 1×1×2 supercell are justified; how much multi-GPU execution actually helps; and how the convergence criteria and the method's internal consistency fare across all production points.

![Figure 4](../assets/fig4_exp3_physics_scaling.png)

**Figure 4** **(a)** Widom insertion in Mg-MOF-74 + CO₂ (6,000 insertions): zero-coverage heat of adsorption $Q_{\rm st}(0)$ and Henry constant $K_H$ (mmol g⁻¹ bar⁻¹). **(b)** Mean interaction energy per CO₂ molecule on equilibrium configurations at each pressure, with MP0+D3 split into its MACE and D3 parts; at 0.012 bar r2SCAN has a single non-empty configuration (*N* = 1), so that value carries little weight. **(c)** Scan of the inner burst length *k* (0.28 bar, short runs): outer acceptance and the expected squared jump in *N* per cycle, ESJD$_N$. **(d)** Wall time on one versus four RTX 3090 GPUs (60 + 200 cycles per pressure, Δε fixed at 0.5 kJ/mol). **(e)** Bulk ESS and $\hat{R}-1$ for the 16 production points; the green region satisfies the gates, and filled symbols pass all criteria. **(f)** Internal consistency checks; error bars are ±2 standard errors.

**Analysis of the figure.**

*Where the binding comes from (Figures 4a, b).* Widom insertion does not depend on a GCMC chain and is therefore an independent check. For r2SCAN, $Q_{\rm st}(0)$ is 30.1 ± 1.1 kJ/mol and $K_H$ is 7.3 ± 2.6; for MP0+D3 they are 48.0 ± 1.8 kJ/mol and 8,450 ± 3,670, three orders of magnitude higher. This is consistent with the 0.18 versus 15.1 mmol/g that the two targets give at 0.012 bar. Splitting the MP0+D3 interaction energy on equilibrium configurations, the total binding at the three pressures is 43.4, 42.9 and 41.9 kJ/mol, of which D3 contributes 22.2, 23.7 and 24.0 kJ/mol (51%–57%). The MACE part alone is only 17.9–21.1 kJ/mol, weaker than r2SCAN's 33.9–35.4 kJ/mol. MP0 itself therefore underbinds Mg²⁺–CO₂, D3 overcompensates, and the net result is about 8 kJ/mol stronger than r2SCAN. For r2SCAN the mean binding increases from 22.7 kJ/mol at 0.012 bar to 35.4 kJ/mol at 0.28 bar, reflecting CO₂–CO₂ cooperativity once the metal sites are occupied, although the 0.012 bar value comes from a single molecule and should not be over-interpreted.

*Parameter choices (Figure 4c and finite size).* Increasing *k* from 5 to 25 lowers the outer acceptance monotonically from 68.2% to 25.1%, while ESJD$_N$ peaks near *k* = 15 (0.269). *k* = 16 balances the two and keeps the outer acceptance at 30%–40%. The effective sample rate for *N* is slightly higher at *k* = 25 (1.03 s⁻¹ vs. 0.73 s⁻¹), but the scan runs are short (ESS 17–30), the difference is not significant, and we did not act on it. For finite size, equilibrium configurations of the 1×1×2 supercell were replicated periodically to 1×1×4 and their MLIP interaction energies recomputed. The largest absolute differences per molecule at the three pressures are 8.6×10⁻⁴, 3.3×10⁻⁵ and 3.4×10⁻⁵ kJ/mol, so the 1×1×2 cell (minimum perpendicular width 13.88 Å) introduces no detectable periodic-image error.

*Multi-GPU efficiency (Figure 4d).* The pressure queue distributes four pressures over four GPUs and reduces end-to-end time from 131 s to 60 s (2.19×). Replica sharding splits the eight boxes of one pressure two per GPU; the MC time drops from 51.1 s to 15.7 s (3.26×, 82% parallel efficiency), but end-to-end only 1.90×, because each worker loads its own copy of the MACE model (about 14–18 s). These test runs are only 260 cycles long, so the load time is a large fraction; for longer production runs the end-to-end speed-up approaches the MC-only value. One-GPU and four-GPU results agree (replica mode: 4.39 ± 0.24 vs. 4.40 ± 0.19 mmol/g), but these short runs are not converged and serve only to measure throughput.

*Convergence audit (Figure 4e).* Six of the 16 production points pass every criterion. They are the low-to-moderate loadings where *N* fluctuates little: Mg-MOF-74 at 0.012 bar, MP0 at 0.28 bar, Ni-MOF-74 at 0.28 bar, both Cu-BTC pressures and IRMOF-1/CO₂ at 1 bar. Three further points (Ni-MOF-74 at 1 bar, IRMOF-1/CH₄ at 1 bar, Mg-MOF-74/r2SCAN at 1 bar) nearly meet the *N* criteria and fail on the energy $\hat{R}$, the energy ESS, or an *N* ESS of 200, respectively. The remaining failures fall into three groups: the steep region of Mg-MOF-74 at 0.28 bar; MP0+D3 states close to pore saturation, where insertions and deletions are rarely accepted ($\hat{R}$ of 1.6–1.9); and IRMOF-1 at 5 bar, a high loading with only 4.7% outer acceptance.

*Internal consistency (Figure 4f).* Three comparisons test the theoretical property that the result does not depend on Δε. First, the empty-start and preloaded chains at 0.28 bar in Mg-MOF-74 used Δε = 0.58 and 0.39 kJ/mol and gave 6.32 and 6.31 mmol/g. Second, IRMOF-1 at 5 bar gave 3.85 ± 0.21 mmol/g with Δε = 0 (recorded in the run log; the JSON was later overwritten) and 3.74 ± 0.08 mmol/g with Δε = 2.5 kJ/mol, consistent within error. Third, the same point with the automatically calibrated Δε = 10.53 kJ/mol (the second calibration stage was pulled up by a single high-loading box) had 1.9% outer acceptance and after 2,000 cycles was still stuck at 1.23 mmol/g. Δε thus does not change the target distribution, but a poor value can leave the chain effectively frozen. Switching to the median reduced the influence of single outlier boxes, but Δε calibration at high loading is still not robust, which is why we set it by hand for this point.

**Results.**

**Table 4** Widom insertion and energy decomposition (Mg-MOF-74 + CO₂).

| Target | $K_H$ (mmol g⁻¹ bar⁻¹) | $Q_{\rm st}(0)$ (kJ/mol) | $\mu_{\rm ex}^0$ (kJ/mol) | −⟨U⟩/N at 0.012 / 0.28 / 1.0 bar (kJ/mol) | of which D3 (kJ/mol) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| r2SCAN | 7.31 ± 2.61 | 30.05 ± 1.13 | −12.57 | 22.7 (N = 1) / 35.4 / 33.9 | – |
| MP0+D3 | 8,450 ± 3,666 | 47.98 ± 1.84 | −30.05 | 43.4 / 42.9 / 41.9 | 22.2 / 23.7 / 24.0 |

**Table 5** One versus four RTX 3090 GPUs (Mg-MOF-74 + CO₂, r2SCAN, 60 + 200 cycles).

| Mode | Workload | 1 GPU (s) | 4 GPUs (s) | Speed-up |
| :--- | :--- | :---: | :---: | :---: |
| Pressure queue, end-to-end | 4 pressures × 4 boxes | 131.2 | 59.8 | 2.19 |
| Replica sharding, end-to-end | 1 pressure × 8 boxes | 64.9 | 34.1 | 1.90 |
| Replica sharding, MC only | same | 51.1 | 15.7 | 3.26 |

---

## 4 Discussion

**What the method solves.** It automates the step of finding a good reference potential for nested MC. Experiment 1 shows that a reference distilled from the target MLIP proposes about as efficiently as a DFT force field fitted specifically for Mg-MOF-74, and far better than a generic force field or direct MLIP sampling. Experiment 2 shows that the same pipeline transfers to other frameworks and guests without parameter changes, with surrogate accuracy of R² ≈ 0.98–0.99. The outer Metropolis correction ensures the results correspond to the target MLIP's distribution, and the checks in Figure 4f support this as far as they can be tested.

**What it does not yet solve.** First, statistical convergence remains slow in the steep part of isotherms and at high loading. Mg-MOF-74 at 0.28 bar still has $\hat{R}$ = 1.2 after 6,000 cycles, and IRMOF-1 at 5 bar has outer acceptance below 5%. In such states an inner burst of *k* = 16 and four chains are probably not enough; longer runs, more chains or enhanced sampling such as parallel tempering will be needed. Second, automatic Δε calibration is unstable at high loading and was overridden by hand for one point. Third, the surrogate still differs from the target by about 2 mmol/g in the steep region (Figure 2c); a more accurate 1B fit or an explicit three-body term may raise the acceptance there.

**The gap to experiment comes mainly from the target.** For converged points, deviations from experiment approach an order of magnitude, all in the same direction (underestimation), and are largest at Ni²⁺ and Cu²⁺ open metal sites; switching to MP0+D3 turns this into a 2.4× overestimate for Mg-MOF-74. These deviations are far outside the sampling uncertainty. Because the method makes no approximation to the physics of the target, it can serve as a tool for assessing the adsorption thermodynamics of foundation MLIPs: differences obtained with the same sampler are differences between models. Predicting experimental isotherms directly will require MLIPs that have been validated or fine-tuned for MOF–gas interactions.

## 5 Conclusions

We have implemented a GPU nested GCMC method for MLIPs that uses an online-distilled surrogate as its proposer. Building the surrogate requires no hand-built force-field parameters and takes about one to three and a half minutes. Combining 128 inner Rosenbluth trials per thread block with batched outer MLIP corrections, the method matches the proposal efficiency of a system-specific force field on Mg-MOF-74 and transfers directly to other frameworks and guests. Of the 16 production points tested, six pass all convergence criteria, and the reasons for each failure are given. Comparison with experiment indicates that binding errors of current foundation MLIPs at open metal sites are the main factor limiting the accuracy of predicted uptakes.

## Data and code availability

Source code, experimental isotherms, the result summary and plotting scripts are in this repository: [`scripts/run_mlip_gcmc.py`](../scripts/run_mlip_gcmc.py), [`scripts/mlip_distill.py`](../scripts/mlip_distill.py), [`src/proposer_dll.cu`](../src/proposer_dll.cu), [`docs/results/phase2_phase3_summary.json`](results/phase2_phase3_summary.json), [`docs/results/arch_inputs_mgmof74_r2scan.npz`](results/arch_inputs_mgmof74_r2scan.npz) and [`scripts/plot_phase4_figures.py`](../scripts/plot_phase4_figures.py). The *k* scan and the IRMOF-1 5 bar result with Δε = 0 exist only in run logs; this is noted in the summary file.

## References

1. I. Batatia, D. P. Kovács, G. N. C. Simm, C. Ortner, G. Csányi. MACE: Higher order equivariant message passing neural networks for fast and accurate force fields. *NeurIPS* 2022.
2. I. Batatia et al. A foundation model for atomistic materials chemistry. arXiv:2401.00096 (2023).
3. A. D. Kaplan et al. A foundational potential energy surface dataset for materials (MatPES). arXiv:2503.04070 (2025).
4. S. Grimme, J. Antony, S. Ehrlich, H. Krieg. *J. Chem. Phys.* **132**, 154104 (2010).
5. S. Grimme, S. Ehrlich, L. Goerigk. *J. Comput. Chem.* **32**, 1456 (2011).
6. R. Iftimie, D. Salahub, D. Wei, J. Schofield. *J. Chem. Phys.* **113**, 4852 (2000).
7. L. D. Gelb. *J. Chem. Phys.* **118**, 7747 (2003).
8. D. Frenkel, B. Smit. *Understanding Molecular Simulation*, 2nd ed. Academic Press (2002).
9. A. Vehtari, A. Gelman, D. Simpson, B. Carpenter, P.-C. Bürkner. *Bayesian Anal.* **16**, 667 (2021).
10. R. Mercado et al. *J. Phys. Chem. C* **120**, 12590 (2016).
11. D. Dubbeldam, S. Calero, D. E. Ellis, R. Q. Snurr. *Mol. Simul.* **42**, 81 (2016).
12. A. K. Rappé, C. J. Casewit, K. S. Colwell, W. A. Goddard III, W. M. Skiff. *J. Am. Chem. Soc.* **114**, 10024 (1992).
13. A. K. Rappé, W. A. Goddard III. *J. Phys. Chem.* **95**, 3358 (1991).
14. M. N. Rosenbluth, A. W. Rosenbluth. *J. Chem. Phys.* **23**, 356 (1955).
15. Mg-MOF-74 CO₂ isotherm, doi:10.1021/acs.jpcc.7b08239.
16. Mg-MOF-74 CO₂ isotherm, doi:10.1039/c5dt03137k.
17. Ni-MOF-74 CO₂ isotherm, doi:10.1021/acs.jpcc.5b03700.
18. Ni-DOBDC CO₂ isotherm, doi:10.1080/15422119.2014.884507.
19. Cu-BTC CO₂ isotherm, doi:10.1002/aic.15837.
20. Cu-BTC CO₂ isotherm, doi:10.1007/s10450-013-9527-2.
21. IRMOF-1 CO₂ isotherm, doi:10.1021/acs.jctc.5b00093.
22. IRMOF-1 CO₂ isotherm, doi:10.1038/srep12045.
