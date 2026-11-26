"""Pure-CPU verification suite for Stage-3 robust shift calibration, equilibration-only adaptation, and pressure-REMC."""

import math
from pathlib import Path
import numpy as np
import pytest

from mcmc_diagnostics import compute_vehtari_mcmc_stats
from multi_gpu import split_pressures_across_gpus
from nested_mc_engine import (
    K_TO_KJMOL,
    _compute_model_sha256,
    attempt_remc_swaps,
    run_nested_mc,
    run_remc_isotherm,
)
from run_mlip_gcmc import _evaluate_shift_stage_from_trace, parse_args


def exact_grand_canonical_moments(
    u_tgt_kJ_fn,
    pressure_bar: float,
    T_K: float = 298.0,
    max_mol: int = 30,
    z_per_bar: float = 1.0,
) -> dict:
    """Exact analytical grand-canonical partition function, <N>, Var(N), and Qst for discrete N in [0, max_mol]."""
    ns = np.arange(max_mol + 1, dtype=np.float64)
    z = float(pressure_bar) * float(z_per_bar)
    u_kJ = np.array([float(u_tgt_kJ_fn(int(n))) for n in range(max_mol + 1)], dtype=np.float64)
    u_K = u_kJ / K_TO_KJMOL
    log_fact = np.array([math.lgamma(n + 1) for n in range(max_mol + 1)], dtype=np.float64)
    log_w = ns * math.log(max(z, 1e-300)) - log_fact - (u_K / float(T_K))
    log_w -= float(np.max(log_w))
    w = np.exp(log_w)
    p_n = w / float(np.sum(w))
    mean_n = float(np.sum(ns * p_n))
    var_n = float(np.sum(((ns - mean_n) ** 2) * p_n))
    mean_u_K = float(np.sum(u_K * p_n))
    cov_u_n_K = float(np.sum((u_K - mean_u_K) * (ns - mean_n) * p_n))
    qst_kJ = float((T_K - cov_u_n_K / var_n) * K_TO_KJMOL) if var_n > 1e-12 else None
    return {
        "P_N": p_n,
        "mean_N": mean_n,
        "var_N": var_n,
        "mean_U_kJ": mean_u_K * K_TO_KJMOL,
        "Qst_kJ_mol": qst_kJ,
    }


class ToyProposer:
    """Exact pure-CPU 1D/lattice GCMC proposer implementing the CudaGcmcProposer quaternion API."""

    def __init__(
        self,
        u_ref_kJ_fn,
        pressure_bar: float,
        T_K: float = 298.0,
        n_boxes: int = 4,
        max_mol: int = 30,
        z_per_bar: float = 1.0,
        fw_mass_gmol: float = 1000.0,
        seed: int = 202601,
        init_N=0,
        p_ins_del: float = 1.0,
    ):
        self.u_ref_kJ_fn = u_ref_kJ_fn
        self.pressure_bar = float(pressure_bar)
        self.T_K = float(T_K)
        self.n_boxes = int(n_boxes)
        self.max_mol = int(max_mol)
        self.z_per_bar = float(z_per_bar)
        self.z_tgt = self.pressure_bar * self.z_per_bar
        self.f_Pa = self.pressure_bar * 1.0e5
        self.fw_mass_gmol = float(fw_mass_gmol)
        self.p_ins_del = float(p_ins_del)
        self.delta_eps_K = 0.0

        self._rngs = [np.random.default_rng(int(seed) + 104729 * (b + 1)) for b in range(self.n_boxes)]
        if isinstance(init_N, (list, tuple, np.ndarray)):
            self._acc_N = np.array([int(x) for x in init_N], dtype=np.int32)
        else:
            self._acc_N = np.full(self.n_boxes, int(init_N), dtype=np.int32)
        self._acc_jitter = np.zeros(self.n_boxes, dtype=np.float64)
        self._tri_N = self._acc_N.copy()
        self._tri_jitter = self._acc_jitter.copy()

        # Instrumentation for strict freeze verification
        self.cycle_idx = 0
        self.shift_calls = []  # list of (cycle_idx, new_de_kJ)
        self.k_history = []    # k_steps used on each cycle

    def _u_ref_K(self, n: int) -> float:
        return float(self.u_ref_kJ_fn(int(n))) / K_TO_KJMOL

    def _make_state(self, n: int, jitter: float, dU_ref_K: float = 0.0, inner_acc: int = 0) -> dict:
        n_int = int(n)
        cx = np.arange(n_int, dtype=np.float64) + float(jitter)
        zeros = np.zeros(n_int, dtype=np.float64)
        ones = np.ones(n_int, dtype=np.float64)
        return {
            "N": n_int,
            "U_ref_K": self._u_ref_K(n_int),
            "dU_ref_K": float(dU_ref_K),
            "inner_acc": int(inner_acc),
            "cx": cx,
            "cy": zeros.copy(),
            "cz": zeros.copy(),
            "qw": ones,
            "qx": zeros.copy(),
            "qy": zeros.copy(),
            "qz": zeros.copy(),
        }

    def set_chem_pot_shift(self, delta_eps_kJ_mol: float):
        val = float(delta_eps_kJ_mol)
        if not math.isfinite(val):
            raise ValueError(f"Non-finite delta_eps_kJ_mol={val}")
        self.delta_eps_K = val / K_TO_KJMOL
        self.shift_calls.append((int(self.cycle_idx), val))

    def get_accepted_quat(self):
        return [self._make_state(int(self._acc_N[b]), float(self._acc_jitter[b])) for b in range(self.n_boxes)]

    def set_accepted_quat(self, states):
        if len(states) != self.n_boxes:
            raise ValueError(f"Expected {self.n_boxes} states, got {len(states)}")
        for b, st in enumerate(states):
            n = int(st["N"])
            self._acc_N[b] = n
            self._acc_jitter[b] = float(st["cx"][0]) if n > 0 and len(st["cx"]) > 0 else 0.0

    def propose_k_quat(self, k_steps: int):
        k_n = int(k_steps)
        self.k_history.append(k_n)
        z_ref = self.z_tgt * math.exp(max(-50.0, min(50.0, -self.delta_eps_K / self.T_K)))
        trials = []
        for b in range(self.n_boxes):
            rng = self._rngs[b]
            n = int(self._acc_N[b])
            u_cur = self._u_ref_K(n)
            u_start = u_cur
            jit = float(self._acc_jitter[b])
            acc_cnt = 0
            for _ in range(k_n):
                r_move = rng.random()
                if r_move < 0.5 * self.p_ins_del:
                    # Insertion N -> N+1
                    if n < self.max_mol:
                        u_new = self._u_ref_K(n + 1)
                        log_acc = math.log(max(z_ref, 1e-300) / float(n + 1)) - (u_new - u_cur) / self.T_K
                        if log_acc >= 0.0 or (log_acc > -50.0 and rng.random() < math.exp(log_acc)):
                            n += 1
                            u_cur = u_new
                            acc_cnt += 1
                elif r_move < self.p_ins_del:
                    # Deletion N -> N-1
                    if n > 0:
                        u_new = self._u_ref_K(n - 1)
                        log_acc = math.log(float(n) / max(z_ref, 1e-300)) - (u_new - u_cur) / self.T_K
                        if log_acc >= 0.0 or (log_acc > -50.0 and rng.random() < math.exp(log_acc)):
                            n -= 1
                            u_cur = u_new
                            acc_cnt += 1
                else:
                    if n > 0:
                        jit += float(rng.uniform(-0.05, 0.05))
                        acc_cnt += 1
            self._tri_N[b] = n
            self._tri_jitter[b] = jit
            trials.append(self._make_state(n, jit, dU_ref_K=u_cur - u_start, inner_acc=acc_cnt))
        return trials

    def commit(self, mask):
        for b in range(self.n_boxes):
            if int(mask[b]) == 1:
                self._acc_N[b] = self._tri_N[b]
                self._acc_jitter[b] = self._tri_jitter[b]
        self.cycle_idx += 1

    def get_rng_states(self):
        return np.array([int(r.bit_generator.state["state"]["state"]) & ((1 << 64) - 1) for r in self._rngs], dtype=np.uint64)

    def set_rng_states(self, _states):
        return None

    def snapshot_rng_full(self):
        return [r.bit_generator.state for r in self._rngs]

    def restore_rng_full(self, states):
        for r, st in zip(self._rngs, states):
            r.bit_generator.state = st


def test_stage3_robust_shift_estimator_resists_outlier_and_captures_marginal():
    """Test 1: Stage-3 robust shift estimator resists a +500 kJ/mol outlier box and captures marginal slope."""
    rng = np.random.default_rng(202611)
    T, B = 25, 4

    # 1A: 3 healthy boxes with (U_tgt - U_ref) = -3.0 * N, 1 outlier box stuck at N=1 with +500 kJ/mol!
    n_hist = rng.integers(8, 15, size=(T, B)).astype(np.float64)
    n_hist[:, 3] = 1.0  # outlier box stuck at N=1
    ur_hist = -20.0 * n_hist
    ut_hist = ur_hist - 3.0 * n_hist + rng.normal(0.0, 0.15, size=(T, B))
    ut_hist[:, 3] = ur_hist[:, 3] + 500.0  # massive +500 kJ/mol clash/outlier

    res_pilot = {
        "prod_N_history": n_hist.tolist(),
        "prod_U_history": ut_hist.tolist(),
        "prod_U_ref_history": ur_hist.tolist(),
    }
    de_est, info = _evaluate_shift_stage_from_trace(res_pilot, -24.0, 24.0, fallback_shift=-2.5, max_step_kJ_mol=3.5)
    assert info["raw_unfiltered_mean_kJ_mol"] > 100.0  # naive mean would explode!
    assert abs(de_est - (-3.0)) < 0.35, f"Outlier contaminated Stage-3 estimate: got {de_est:.3f}"
    assert not info["hit_trust_region"]

    # 1B: Outlier box ALSO has dN != 0 transitions with +500 kJ/mol jumps!
    n_hist_b = n_hist.copy()
    n_hist_b[:, 3] = np.where(np.arange(T) % 2 == 0, 1.0, 2.0)
    ur_b = -20.0 * n_hist_b
    ut_b = ur_b - 3.0 * n_hist_b + rng.normal(0.0, 0.10, size=(T, B))
    ut_b[:, 3] = ur_b[:, 3] + 500.0 * n_hist_b[:, 3]
    de_est_b, _ = _evaluate_shift_stage_from_trace(
        {"prod_N_history": n_hist_b.tolist(), "prod_U_history": ut_b.tolist(), "prod_U_ref_history": ur_b.tolist()},
        -24.0, 24.0, fallback_shift=-2.5, max_step_kJ_mol=3.5,
    )
    assert abs(de_est_b - (-3.0)) < 0.35, f"Marginal outlier contaminated Stage-3 estimate: got {de_est_b:.3f}"

    # 1C: Quadratic c2*N^2 mismatch -> marginal slope d(U_tgt-U_ref)/dN = -1.0 - 0.30*N ≈ -4.0 at N≈10
    n_q = rng.integers(8, 13, size=(30, 4)).astype(np.float64)
    ur_q = -20.0 * n_q
    ut_q = ur_q - 1.0 * n_q - 0.15 * (n_q ** 2)
    de_q, info_q = _evaluate_shift_stage_from_trace(
        {"prod_N_history": n_q.tolist(), "prod_U_history": ut_q.tolist(), "prod_U_ref_history": ur_q.tolist()},
        -24.0, 24.0, fallback_shift=-3.0, max_step_kJ_mol=3.5,
    )
    assert info_q["marginal_est_kJ_mol"] is not None
    assert abs(de_q - (-4.0)) < 0.40, f"Expected marginal shift ~ -4.0 kJ/mol, got {de_q:.3f}"

    # 1D: Trust-region clamping relative to fallback_shift (max_step_kJ_mol=3.5)
    ut_extreme = ur_hist + 40.0 * n_hist
    de_tr, info_tr = _evaluate_shift_stage_from_trace(
        {"prod_N_history": n_hist.tolist(), "prod_U_history": ut_extreme.tolist(), "prod_U_ref_history": ur_hist.tolist()},
        -24.0, 24.0, fallback_shift=-2.0, max_step_kJ_mol=3.5,
    )
    assert math.isclose(de_tr, 1.5, abs_tol=1e-9), f"Trust region clamp failed: {de_tr}"
    assert info_tr["hit_trust_region"] is True


def test_adaptive_shift_balances_proposals_freezes_and_stays_unbiased():
    """Test 2: Equilibration-only adaptive delta_eps balances ins/del proposals, freezes at eq//2, and stays unbiased."""
    T_K = 298.0
    P_bar = 1.0
    z_per_bar = 3.5e-4
    max_mol = 30

    # Strong quadratic mismatch: U_tgt(N) = -26*N - 0.18*N^2 + 0.008*N^3 vs U_ref(N) = -20*N
    def u_tgt_kJ(n: int) -> float:
        return -26.0 * n - 0.18 * (n ** 2) + 0.008 * (n ** 3)

    def u_ref_kJ(n: int) -> float:
        return -20.0 * n

    def batch_tgt(states):
        return [(u_tgt_kJ(int(s["N"])) / K_TO_KJMOL, False) for s in states]

    exact = exact_grand_canonical_moments(u_tgt_kJ, P_bar, T_K=T_K, max_mol=max_mol, z_per_bar=z_per_bar)
    assert 6.0 < exact["mean_N"] < 24.0

    # Baseline WITHOUT shift adaptation starting at delta_eps = 0.0 -> severely asymmetric proposals!
    prop_base = ToyProposer(
        u_ref_kJ, pressure_bar=P_bar, T_K=T_K, n_boxes=6, max_mol=max_mol,
        z_per_bar=z_per_bar, seed=202621, init_N=int(round(exact["mean_N"])),
    )
    res_base = run_nested_mc(
        prop_base, None, k_steps=6, eq_cycles=100, prod_cycles=400, seed=202622,
        batch_target_fn=batch_tgt, adapt_shift_frac=0.0, adapt_k_frac=0.0,
    )
    base_ratio = res_base["prod_ins_att"] / max(1, res_base["prod_del_att"])
    assert base_ratio < 0.25, f"Expected severely asymmetric baseline ins/del ratio (<0.25), got {base_ratio:.3f}"

    # WITH equilibration-only adaptive shift (adapt_shift_frac=0.5, strictly frozen after eq_cycles//2)
    eq_cycles = 400
    prod_cycles = 1600
    prop_ad = ToyProposer(
        u_ref_kJ, pressure_bar=P_bar, T_K=T_K, n_boxes=6, max_mol=max_mol,
        z_per_bar=z_per_bar, seed=202623, init_N=int(round(exact["mean_N"])),
    )
    res_ad = run_nested_mc(
        prop_ad, None, k_steps=6, eq_cycles=eq_cycles, prod_cycles=prod_cycles, seed=202624,
        batch_target_fn=batch_tgt, adapt_shift_frac=0.5, adapt_k_frac=0.0, adapt_window=20,
    )

    # 1. Strictly frozen at eq_cycles // 2
    assert res_ad["adapt_shift_end_cycle"] == eq_cycles // 2
    assert len(res_ad["adaptation_history"]) > 0
    assert max(ev["cycle"] for ev in res_ad["adaptation_history"]) <= eq_cycles // 2
    assert all(cyc <= eq_cycles // 2 for cyc, _ in prop_ad.shift_calls), (
        f"Shift mutated after freeze! shift_calls={prop_ad.shift_calls}"
    )

    # 2. Proposals balanced from <0.25 to ~1:1
    ad_ratio = res_ad["prod_ins_att"] / max(1, res_ad["prod_del_att"])
    assert 0.70 <= ad_ratio <= 1.45, f"Expected balanced ins/del ratio near 1.0, got {ad_ratio:.3f}"
    assert res_ad["delta_eps_kJ_mol"] < -4.0

    # 3. Unbiased production <N> vs exact analytical <N> (< 2.5 sigma)
    n_prod = np.asarray(res_ad["prod_N_history"], dtype=np.float64)
    mean_mc = float(np.mean(n_prod))
    stats_n = compute_vehtari_mcmc_stats(n_prod)
    box_means = np.mean(n_prod, axis=0)
    sem_boxes = float(np.std(box_means, ddof=1) / math.sqrt(len(box_means)))
    sigma = max(sem_boxes, float(stats_n["mcse"] or 0.0), 1e-6)
    z_err = abs(mean_mc - exact["mean_N"]) / sigma
    assert z_err < 2.5, (
        f"Adaptive-shift production biased: mc={mean_mc:.3f} vs exact={exact['mean_N']:.3f} "
        f"(sigma={sigma:.3f}, z={z_err:.2f})"
    )


def test_adaptive_k_adjusts_freezes_and_stays_unbiased():
    """Test 3: Equilibration-only adaptive k scales down on low acceptance, scales up on high acceptance, freezes at eq//2, stays unbiased."""
    T_K = 298.0
    P_bar = 1.0
    z_per_bar = 3.5e-4
    max_mol = 30

    # Steep mismatch vs flat reference: large k=24 overshoots the target basin (acc < 0.15), smaller k recovers acceptance
    def u_tgt_kJ(n: int) -> float:
        return -25.5 * n - 0.18 * (n ** 2) + 0.0085 * (n ** 3)

    def u_ref_kJ(n: int) -> float:
        return -21.0 * n

    def batch_tgt(states):
        return [(u_tgt_kJ(int(s["N"])) / K_TO_KJMOL, False) for s in states]

    exact = exact_grand_canonical_moments(u_tgt_kJ, P_bar, T_K=T_K, max_mol=max_mol, z_per_bar=z_per_bar)

    # Confirm fixed k=24 has low outer acceptance (< 0.20)
    prop_fixed = ToyProposer(
        u_ref_kJ, pressure_bar=P_bar, T_K=T_K, n_boxes=6, max_mol=max_mol,
        z_per_bar=z_per_bar, seed=202630, init_N=int(round(exact["mean_N"])),
    )
    res_fixed = run_nested_mc(
        prop_fixed, None, k_steps=24, eq_cycles=80, prod_cycles=200, seed=202635,
        batch_target_fn=batch_tgt, adapt_shift_frac=0.0, adapt_k_frac=0.0,
    )
    assert res_fixed["prod_outer_acc_rate"] < 0.20, (
        f"Expected low baseline acceptance at fixed k=24, got {res_fixed['prod_outer_acc_rate']:.3f}"
    )

    eq_cycles = 360
    prod_cycles = 1600
    prop = ToyProposer(
        u_ref_kJ, pressure_bar=P_bar, T_K=T_K, n_boxes=6, max_mol=max_mol,
        z_per_bar=z_per_bar, seed=202631, init_N=int(round(exact["mean_N"])),
    )
    res = run_nested_mc(
        prop, None, k_steps=24, eq_cycles=eq_cycles, prod_cycles=prod_cycles, seed=202632,
        batch_target_fn=batch_tgt, adapt_shift_frac=0.0, adapt_k_frac=0.5,
        k_min=2, k_max=24, target_outer_acc=0.35, adapt_window=20,
    )

    # 1. k scaled down from 24 and improved outer acceptance
    assert res["k_steps_initial"] == 24
    assert 2 <= res["k_steps"] < 24, f"Expected k_steps to scale down from 24, got {res['k_steps']}"
    assert res["prod_outer_acc_rate"] > res_fixed["prod_outer_acc_rate"]

    # 2. Strictly frozen for second half of eq + all of production
    frozen_slice = prop.k_history[eq_cycles // 2:]
    assert len(frozen_slice) == (eq_cycles - eq_cycles // 2) + prod_cycles
    assert len(set(frozen_slice)) == 1 and frozen_slice[0] == res["k_steps"]

    # 3. Unbiased production <N> vs exact analytical <N> (< 2.5 sigma)
    n_prod = np.asarray(res["prod_N_history"], dtype=np.float64)
    mean_mc = float(np.mean(n_prod))
    stats_n = compute_vehtari_mcmc_stats(n_prod)
    box_means = np.mean(n_prod, axis=0)
    sem_boxes = float(np.std(box_means, ddof=1) / math.sqrt(len(box_means)))
    sigma = max(sem_boxes, float(stats_n["mcse"] or 0.0), 1e-6)
    z_err = abs(mean_mc - exact["mean_N"]) / sigma
    assert z_err < 2.5, (
        f"Adaptive-k production biased: mc={mean_mc:.3f} vs exact={exact['mean_N']:.3f} "
        f"(sigma={sigma:.3f}, z={z_err:.2f})"
    )

    # 4. Also verify k scales UP when starting at k=2 with near-exact reference (acc > 0.50)
    prop_up = ToyProposer(
        u_ref_kJ, pressure_bar=P_bar, T_K=T_K, n_boxes=4, max_mol=max_mol,
        z_per_bar=z_per_bar, seed=202633, init_N=8,
    )
    res_up = run_nested_mc(
        prop_up, None, k_steps=2, eq_cycles=160, prod_cycles=100, seed=202634,
        batch_target_fn=lambda sts: [(u_ref_kJ(int(s["N"])) / K_TO_KJMOL, False) for s in sts],
        adapt_shift_frac=0.0, adapt_k_frac=0.5, k_min=2, k_max=24, adapt_window=20,
    )
    assert res_up["k_steps"] > 2, f"Expected k_steps to scale up from 2 on exact surrogate, got {res_up['k_steps']}"
    assert len(set(prop_up.k_history[80:])) == 1


def test_remc_isotherm_bimodal_exact_moments_and_ess_gain():
    """Test 4: REMC across 4 pressures on a bimodal steep-isotherm toy model matches exact <N>, Var(N), Qst & improves ESS."""
    T_K = 298.0
    max_mol = 28
    z_per_bar = 1.8e-4

    # Bimodal S-shaped isotherm with attractive c2*N^2 cooperativity + Gaussian free-energy barrier at N=11
    def u_tgt_kJ(n: int) -> float:
        barrier = 14.5 * math.exp(-0.5 * ((n - 11.0) / 2.2) ** 2)
        return -25.8 * n - 0.30 * (n ** 2) + 0.011 * (n ** 3) + barrier

    # Linear surrogate (strong quadratic + barrier mismatch!)
    def u_ref_kJ(n: int) -> float:
        return -25.8 * n

    def batch_tgt(states):
        return [(u_tgt_kJ(int(s["N"])) / K_TO_KJMOL, False) for s in states]

    # Choose 4 pressures spanning below, across, and above the bimodal inflection
    pressures = [0.40, 0.70, 0.95, 1.50]
    exacts = [
        exact_grand_canonical_moments(u_tgt_kJ, p, T_K=T_K, max_mol=max_mol, z_per_bar=z_per_bar)
        for p in pressures
    ]
    # Confirm steep S-shaped bimodal jump across the 4 pressures
    assert exacts[0]["mean_N"] < 5.0
    assert exacts[3]["mean_N"] > 14.0
    # Find the steepest bimodal inflection pressure (highest Var(N))
    inflect_idx = int(np.argmax([ex["var_N"] for ex in exacts]))
    assert exacts[inflect_idx]["var_N"] > 15.0

    # Confirm ladder independence of attempt_remc_swaps directly
    rngs_a = [np.random.default_rng(100 + b) for b in range(4)]
    rngs_b = [np.random.default_rng(100 + b) for b in range(4)]
    n_mat_a = [[2, 3, 4, 5], [10, 8, 7, 9], [15, 16, 14, 17], [20, 21, 19, 22]]
    n_mat_b = [[22, 3, 4, 5], [0, 8, 7, 9], [22, 16, 14, 17], [0, 21, 19, 22]]  # drastically perturb ONLY ladder 0
    fugs_test = [p * 1.0e5 for p in pressures]
    pairs_a, _, _ = attempt_remc_swaps(n_mat_a, fugs_test, parity=0, ladder_rngs=rngs_a)
    pairs_b, _, _ = attempt_remc_swaps(n_mat_b, fugs_test, parity=0, ladder_rngs=rngs_b)
    assert [p for p in pairs_a if p[2] > 0] == [p for p in pairs_b if p[2] > 0], (
        "Ladder 0 perturbed swap outcomes of independent ladders 1..3!"
    )

    n_boxes = 8
    eq_cycles = 400
    prod_cycles = 2000
    k_steps = 4
    proposers = [
        ToyProposer(
            u_ref_kJ, pressure_bar=p, T_K=T_K, n_boxes=n_boxes, max_mol=max_mol,
            z_per_bar=z_per_bar, seed=202640 + 37 * m,
            init_N=[2 if b % 2 == 0 else 18 for b in range(n_boxes)],
        )
        for m, p in enumerate(pressures)
    ]

    remc_res = run_remc_isotherm(
        proposers=proposers,
        pressures_bar=pressures,
        batch_target_fn=batch_tgt,
        k_steps=k_steps,
        eq_cycles=eq_cycles,
        prod_cycles=prod_cycles,
        swap_every=2,
        seed=202649,
        adapt_shift_frac=0.5,
        adapt_k_frac=0.0,
        adapt_window=20,
    )
    assert len(remc_res) == 4

    for m, (res, ex) in enumerate(zip(remc_res, exacts)):
        assert res["parallel_mode"] == "remc"
        assert math.isclose(res["P_bar"], pressures[m], rel_tol=1e-9)
        assert len(res["prod_N_history"]) == prod_cycles
        assert len(res["prod_U_history"]) == prod_cycles
        assert res["max_N_observed"] <= max_mol
        assert res["max_mol"] == max_mol

        # Positive swap rate on every adjacent pair (m, m+1)
        if m < len(pressures) - 1:
            assert res["remc_swap_att_up"] > 0
            assert res["remc_swap_rate_up"] > 0.05, (
                f"Low REMC swap rate at pair ({m},{m+1}): {res['remc_swap_rate_up']:.3f}"
            )
        else:
            assert res["remc_swap_rate_up"] is None

        n_prod = np.asarray(res["prod_N_history"], dtype=np.float64)
        mean_mc = float(np.mean(n_prod))
        var_mc = float(np.var(n_prod))
        stats_n = compute_vehtari_mcmc_stats(n_prod)

        # 1. Exact <N> within < 2.5 sigma
        box_means = np.mean(n_prod, axis=0)
        sem_n = float(np.std(box_means, ddof=1) / math.sqrt(n_boxes))
        sigma_n = max(sem_n, float(stats_n["mcse"] or 0.0), 1e-6)
        z_n = abs(mean_mc - ex["mean_N"]) / sigma_n
        assert z_n < 2.5, (
            f"P={pressures[m]} <N> mismatch: mc={mean_mc:.3f} vs exact={ex['mean_N']:.3f} "
            f"(sigma={sigma_n:.3f}, z={z_n:.2f})"
        )

        # 2. Exact Var(N) within < 2.5 sigma (leave-one-chain-out jackknife SEM for Var(N))
        loo_vars = [
            float(np.var(np.delete(n_prod, b, axis=1)))
            for b in range(n_boxes)
        ]
        var_sem = math.sqrt(((n_boxes - 1.0) / n_boxes) * float(np.sum((np.array(loo_vars) - np.mean(loo_vars)) ** 2)))
        sigma_v = max(var_sem, 0.05 * ex["var_N"], 1e-6)
        z_v = abs(var_mc - ex["var_N"]) / sigma_v
        assert z_v < 2.5, (
            f"P={pressures[m]} Var(N) mismatch: mc={var_mc:.3f} vs exact={ex['var_N']:.3f} "
            f"(sigma={sigma_v:.3f}, z={z_v:.2f})"
        )

        # 3. Exact fluctuation Qst within < 2.5 sigma
        assert res["Qst_kJ_mol"] is not None and res["Qst_sem_kJ_mol"] is not None
        sigma_q = max(float(res["Qst_sem_kJ_mol"]), 1e-6)
        z_q = abs(float(res["Qst_kJ_mol"]) - ex["Qst_kJ_mol"]) / sigma_q
        assert z_q < 2.5, (
            f"P={pressures[m]} Qst mismatch: mc={res['Qst_kJ_mol']:.3f} vs exact={ex['Qst_kJ_mol']:.3f} "
            f"(sem={sigma_q:.3f}, z={z_q:.2f})"
        )

    # 4. Non-REMC comparison at the bimodal inflection pressure (identical cycle budget & initial split states)
    prop_nore = ToyProposer(
        u_ref_kJ, pressure_bar=pressures[inflect_idx], T_K=T_K, n_boxes=n_boxes, max_mol=max_mol,
        z_per_bar=z_per_bar, seed=202640 + 37 * inflect_idx,
        init_N=[2 if b % 2 == 0 else 18 for b in range(n_boxes)],
    )
    res_nore = run_nested_mc(
        prop_nore, None, k_steps=k_steps, eq_cycles=eq_cycles, prod_cycles=prod_cycles,
        seed=202649 + inflect_idx * 1009, batch_target_fn=batch_tgt,
        adapt_shift_frac=0.5, adapt_k_frac=0.0, adapt_window=20,
    )
    ess_remc = float(remc_res[inflect_idx]["diagnostics"]["N_stats"]["ess_bulk"])
    ess_nore = float(compute_vehtari_mcmc_stats(np.asarray(res_nore["prod_N_history"], dtype=np.float64))["ess_bulk"])
    assert ess_remc > 1.5 * ess_nore, (
        f"Expected REMC to measurably improve bulk ESS at bimodal inflection P={pressures[inflect_idx]}: "
        f"REMC ess_bulk={ess_remc:.1f} vs non-REMC ess_bulk={ess_nore:.1f}"
    )


def test_cli_args_remc_helpers_and_model_sha256_fallback(tmp_path):
    """Test 5: CLI flags, pressure GPU sharding, and model_sha256 file + state_dict fallback."""
    args = parse_args([
        "--cif", "data/MgMOF-74.cif",
        "--pressures", "0.01", "0.1", "1.0",
        "--out-json", "build/out.json",
        "--distill-1b-only",
        "--parallel-mode", "remc",
        "--remc-swap-every", "4",
        "--adapt-shift", "on",
        "--adapt-k", "off",
        "--k-min", "3",
        "--k-max", "20",
    ])
    assert args.distill_1b_only is True
    assert args.parallel_mode == "remc"
    assert args.remc_swap_every == 4
    assert args.adapt_shift == "on"
    assert args.adapt_k == "off"
    assert args.k_min == 3
    assert args.k_max == 20

    # Default values check
    args_def = parse_args(["--cif", "data/MgMOF-74.cif", "--pressures", "0.1", "--out-json", "build/out.json"])
    assert args_def.distill_1b_only is False
    assert args_def.adapt_shift == "on"
    assert args_def.adapt_k == "on"
    assert args_def.k_min == 2
    assert args_def.k_max == 24
    assert args_def.remc_swap_every == 2

    # Pressure partitioning across GPUs
    assert split_pressures_across_gpus(4, [0, 1, 2, 3]) == [(0, [0]), (1, [1]), (2, [2]), (3, [3])]
    assert split_pressures_across_gpus(6, [0, 1, 2, 3]) == [(0, [0, 1]), (1, [2, 3]), (2, [4]), (3, [5])]
    assert split_pressures_across_gpus(2, [0, 1, 2, 3]) == [(0, [0]), (1, [1])]

    # model_sha256 file path vs state_dict fallback
    dummy_file = tmp_path / "model.pt"
    dummy_file.write_bytes(b"fake-mace-weights-v1")
    h_file, src_file = _compute_model_sha256(str(dummy_file), None)
    assert src_file == "file" and isinstance(h_file, str) and len(h_file) == 64

    class FakeModel:
        def state_dict(self):
            return {
                "b": np.array([1.0, 2.0], dtype=np.float32),
                "a": np.array([[3.0, 4.0]], dtype=np.float32),
            }

    h_sd1, src_sd1 = _compute_model_sha256(None, FakeModel())
    h_sd2, src_sd2 = _compute_model_sha256("/nonexistent/path.pt", FakeModel())
    assert src_sd1 == "state_dict" and src_sd2 == "state_dict"
    assert h_sd1 == h_sd2 and len(h_sd1) == 64


def test_nested_mc_checkpoint_resume_exact_and_empty_box_guard():
    """Test 6: Bit-exact checkpoint resume after adaptation freeze and empty-box zero-drift guard."""
    import json

    T_K = 298.0
    P_bar = 1.0
    z_per_bar = 3.5e-4
    max_mol = 30

    def u_tgt_kJ(n: int) -> float:
        return -26.0 * n - 0.16 * (n ** 2) + 0.008 * (n ** 3)

    def u_ref_kJ(n: int) -> float:
        return -21.0 * n

    def batch_tgt(states):
        return [(u_tgt_kJ(int(s["N"])) / K_TO_KJMOL, False) for s in states]

    # 1. Bit-exact resume after equilibration adaptation
    prop_cont = ToyProposer(u_ref_kJ, P_bar, T_K=T_K, n_boxes=4, max_mol=max_mol, z_per_bar=z_per_bar, seed=202660, init_N=10)
    res_cont = run_nested_mc(
        prop_cont, None, k_steps=12, eq_cycles=80, prod_cycles=120, seed=202661,
        batch_target_fn=batch_tgt, adapt_shift_frac=0.5, adapt_k_frac=0.5, k_min=2, k_max=24, adapt_window=20,
    )

    saved = {}
    prop_p1 = ToyProposer(u_ref_kJ, P_bar, T_K=T_K, n_boxes=4, max_mol=max_mol, z_per_bar=z_per_bar, seed=202660, init_N=10)

    def _ckpt_cb(st):
        saved["mc"] = json.loads(json.dumps(st))
        saved["rng"] = prop_p1.snapshot_rng_full()

    _ = run_nested_mc(
        prop_p1, None, k_steps=12, eq_cycles=80, prod_cycles=60, seed=202661,
        batch_target_fn=batch_tgt, adapt_shift_frac=0.5, adapt_k_frac=0.5, k_min=2, k_max=24, adapt_window=20,
        checkpoint_fn=_ckpt_cb, checkpoint_every=140,
    )

    prop_p2 = ToyProposer(u_ref_kJ, P_bar, T_K=T_K, n_boxes=4, max_mol=max_mol, z_per_bar=z_per_bar, seed=999999, init_N=0)
    prop_p2.set_chem_pot_shift(saved["mc"]["delta_eps_K"] * K_TO_KJMOL)
    prop_p2.restore_rng_full(saved["rng"])
    res_resumed = run_nested_mc(
        prop_p2, None, k_steps=saved["mc"]["k_steps"], eq_cycles=80, prod_cycles=120, seed=202661,
        batch_target_fn=batch_tgt, adapt_shift_frac=0.5, adapt_k_frac=0.5, k_min=2, k_max=24, adapt_window=20,
        resume_state=saved["mc"],
    )

    assert res_resumed["prod_N_history"] == res_cont["prod_N_history"]
    assert np.allclose(res_resumed["prod_U_history"], res_cont["prod_U_history"], atol=1e-12)
    assert res_resumed["k_steps"] == res_cont["k_steps"]
    assert math.isclose(res_resumed["delta_eps_kJ_mol"], res_cont["delta_eps_kJ_mol"], abs_tol=1e-12)
    assert math.isclose(res_resumed["uptake_mmol_g"], res_cont["uptake_mmol_g"], abs_tol=1e-12)
    assert res_resumed["acceptance_breakdown"] == res_cont["acceptance_breakdown"]

    # 2. Empty-box guard: all boxes stuck at N=0 with ins_att==0 must NOT spuriously push delta_eps upward
    prop_empty = ToyProposer(
        lambda n: +50.0 * n, pressure_bar=0.01, T_K=T_K, n_boxes=4, max_mol=max_mol,
        z_per_bar=1.0e-8, seed=202662, init_N=0,
    )
    prop_empty.set_chem_pot_shift(2.0)
    res_empty = run_nested_mc(
        prop_empty, None, k_steps=6, eq_cycles=80, prod_cycles=40, seed=202663,
        batch_target_fn=lambda sts: [(+50.0 * int(s["N"]) / K_TO_KJMOL, False) for s in sts],
        adapt_shift_frac=0.5, adapt_k_frac=0.5, adapt_window=20,
    )
    assert math.isclose(res_empty["delta_eps_kJ_mol"], 2.0, abs_tol=1e-12)
    assert res_empty["max_N_observed"] == 0
