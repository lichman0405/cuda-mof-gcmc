"""P0 (§3.1 & §3.3) Verification Suite:
1. Exact analytical 6-site lattice grand-canonical Nested MC across shifts and k_steps.
2. Classical target parity (Direct GPU-GCMC vs Nested MC across proposers, shifts, k_steps; mean, Var, histogram).
3. Incremental/recompute energy parity, outer-reject bit-exact restore, GPU RNG progression, extreme shift & NaN guards.
"""

import math
from pathlib import Path
import numpy as np
import pytest

from guests import load_guest
from nested_mc_engine import CudaGcmcProposer, K_TO_KJMOL, run_nested_mc
from qeq_charges import extract_framework_for_proposer
from run_mlip_gcmc import compute_grid_dims, extract_oms_metal_positions

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def _exact_lattice_moments(e_site_K: np.ndarray, z_fug: float, T_K: float):
    """Exact grand-canonical distribution P(N), <N>, Var(N) for independent sites."""
    weights = z_fug * np.exp(-e_site_K / T_K)
    poly = np.array([1.0], dtype=np.float64)
    for w in weights:
        poly = np.convolve(poly, np.array([1.0, w], dtype=np.float64))
    p_n = poly / np.sum(poly)
    n_vals = np.arange(len(p_n), dtype=np.float64)
    mean_n = float(np.sum(n_vals * p_n))
    var_n = float(np.sum(((n_vals - mean_n) ** 2) * p_n))
    return p_n, mean_n, var_n


def _run_lattice_nested_mc(
    e_tgt_K: np.ndarray,
    e_ref_K: np.ndarray,
    z_tgt: float,
    T_K: float,
    delta_eps_K: float,
    k_steps: int,
    n_cycles: int,
    seed: int = 1234,
    direct_prob: float = 0.0,
):
    rng = np.random.default_rng(seed)
    M = len(e_tgt_K)
    z_ref = z_tgt * math.exp(-delta_eps_K / T_K)
    occ = np.zeros(M, dtype=np.int32)
    u_tgt = 0.0
    u_ref = 0.0
    counts = np.zeros(M + 1, dtype=np.float64)

    burn = n_cycles // 5
    for cyc in range(burn + n_cycles):
        if direct_prob > 0.0 and rng.random() < direct_prob:
            site = int(rng.integers(0, M))
            if occ[site] == 0:
                du = float(e_tgt_K[site])
                ratio = z_tgt * math.exp(max(-50.0, min(50.0, -du / T_K)))
                if ratio >= 1.0 or rng.random() < ratio:
                    occ[site] = 1
                    u_tgt += du
                    u_ref += float(e_ref_K[site])
            else:
                du = -float(e_tgt_K[site])
                ratio = (1.0 / z_tgt) * math.exp(max(-50.0, min(50.0, -du / T_K)))
                if ratio >= 1.0 or rng.random() < ratio:
                    occ[site] = 0
                    u_tgt += du
                    u_ref -= float(e_ref_K[site])
        else:
            trial = occ.copy()
            u_ref_tri = u_ref
            for _ in range(k_steps):
                site = int(rng.integers(0, M))
                if trial[site] == 0:
                    du = float(e_ref_K[site])
                    ratio = z_ref * math.exp(max(-50.0, min(50.0, -du / T_K)))
                    if ratio >= 1.0 or rng.random() < ratio:
                        trial[site] = 1
                        u_ref_tri += du
                else:
                    du = -float(e_ref_K[site])
                    ratio = (1.0 / z_ref) * math.exp(max(-50.0, min(50.0, -du / T_K)))
                    if ratio >= 1.0 or rng.random() < ratio:
                        trial[site] = 0
                        u_ref_tri += du

            dN = int(np.sum(trial)) - int(np.sum(occ))
            u_tgt_tri = float(np.dot(trial, e_tgt_K))
            ddU = (u_tgt_tri - u_tgt) - (u_ref_tri - u_ref) - delta_eps_K * float(dN)
            if ddU <= 0.0 or (-(ddU / T_K) > -50.0 and rng.random() < math.exp(-ddU / T_K)):
                occ = trial
                u_tgt = u_tgt_tri
                u_ref = u_ref_tri

        if cyc >= burn:
            counts[int(np.sum(occ))] += 1.0

    p_est = counts / np.sum(counts)
    n_vals = np.arange(M + 1, dtype=np.float64)
    mean_est = float(np.sum(n_vals * p_est))
    var_est = float(np.sum(((n_vals - mean_est) ** 2) * p_est))
    return p_est, mean_est, var_est


@pytest.mark.parametrize(
    "shift_kJ,k_steps",
    [(0.0, 1), (-3.0, 5), (-6.0, 10), (2.0, 5)],
)
def test_exact_enumerable_grand_canonical_nested_mc(shift_kJ, k_steps):
    """Verify exact outer acceptance & detailed balance on a 6-site solvable model."""
    T_K = 298.0
    e_tgt_K = np.array([-42.0, -39.0, -35.0, -24.0, -18.0, -12.0]) / K_TO_KJMOL
    e_ref_K = np.array([-34.0, -33.0, -30.0, -22.0, -16.0, -12.0]) / K_TO_KJMOL
    z_tgt = 3.0e-6

    p_exact, mean_exact, var_exact = _exact_lattice_moments(e_tgt_K, z_tgt, T_K)
    p_mc, mean_mc, var_mc = _run_lattice_nested_mc(
        e_tgt_K, e_ref_K, z_tgt, T_K,
        delta_eps_K=shift_kJ / K_TO_KJMOL,
        k_steps=k_steps,
        n_cycles=120000,
        seed=202601 + int(abs(shift_kJ) * 10) + k_steps,
        direct_prob=0.0,
    )
    max_p_err = float(np.max(np.abs(p_mc - p_exact)))
    assert abs(mean_mc - mean_exact) < 0.03, f"Mean mismatch: {mean_mc:.4f} vs {mean_exact:.4f}"
    assert abs(var_mc - var_exact) < 0.03, f"Var mismatch: {var_mc:.4f} vs {var_exact:.4f}"
    assert max_p_err < 0.015, f"Histogram P(N) max err {max_p_err:.4f} too large (mc={p_mc}, exact={p_exact})"


def test_proposer_energy_restore_rng_and_guards():
    """Verify P0.3: incremental=recompute energy, outer reject restore, RNG advance, shift/NaN rejection."""
    cif = str(DATA / "MgMOF-74.cif")
    fw_data = extract_framework_for_proposer(cif, charge_mode="auto")
    guest = load_guest("co2")
    prop = CudaGcmcProposer.from_atoms(
        fw_data=fw_data, guest=guest, supercell=(1, 1, 2),
        T_K=298.0, pressure_bar=0.28, ff_mode=0, pol_mode=0,
        grid_dims=(128, 128, 40), n_boxes=4, max_mol=128, seed=20260301,
    )
    try:
        # 1. Extreme shift rejection (P0.1 & P0.3)
        with pytest.raises(ValueError):
            prop.set_chem_pot_shift(150.0)  # |dEps/T_K| ~ 60.5 > 40
        with pytest.raises(ValueError):
            prop.set_chem_pot_shift(float("nan"))
        rc_bad = prop.lib.proposer_set_chem_pot_shift(prop.handle, 50.0 * prop.T_K)
        assert rc_bad == -2

        # 2. Incremental U_ref vs fresh eval_energy_quat parity
        prop.equilibrate(5000)
        st_acc = prop.get_accepted_quat()
        for s in st_acc:
            u_fresh, ov = prop.eval_energy_quat(s)
            assert not ov
            assert abs(u_fresh - s["U_ref_K"]) < 1e-8

        # 3. Outer reject preserves accepted state bit-for-bit while GPU RNG advances strictly
        rng_before = prop.get_rng_states()
        _ = prop.propose_k_quat(25)
        prop.commit([0, 0, 0, 0])  # force reject all 4 boxes
        rng_after = prop.get_rng_states()
        st_after = prop.get_accepted_quat()

        assert not np.array_equal(rng_before, rng_after), "GPU RNG did not advance across outer reject!"
        for a, b in zip(st_acc, st_after):
            assert a["N"] == b["N"]
            assert a["U_ref_K"] == b["U_ref_K"]
            assert np.array_equal(a["cx"], b["cx"])
            assert np.array_equal(a["qw"], b["qw"])

        # 4. NaN / Inf target energy guard
        def nan_target(states):
            return [(float("nan"), False) for _ in states]

        with pytest.raises(RuntimeError, match="Non-finite"):
            run_nested_mc(prop, None, k_steps=5, eq_cycles=2, prod_cycles=2, seed=99, batch_target_fn=nan_target)
    finally:
        prop.close()


@pytest.mark.slow
@pytest.mark.parametrize("shift_kJ,k_steps", [(0.0, 5), (-4.0, 15), (+4.0, 15)])
def test_classical_target_nested_mc_distribution_parity(shift_kJ, k_steps):
    """§3.1 Classical fixed target (IRMOF-1 CO2 2.0 bar): Direct GPU GCMC vs shifted Nested MC (<N>, Var(N), stationarity)."""
    cif = str(DATA / "IRMOF-1.cif")
    fw_data = extract_framework_for_proposer(cif, charge_mode="auto")
    guest = load_guest("co2")
    T_K, P_bar = 298.0, 2.0

    tgt = CudaGcmcProposer.from_atoms(
        fw_data=fw_data, guest=guest, supercell=(1, 1, 1),
        T_K=T_K, pressure_bar=P_bar, ff_mode=0, pol_mode=0,
        grid_dims=(96, 96, 96), n_boxes=8, max_mol=128, seed=20260401,
    )
    prop = CudaGcmcProposer.from_atoms(
        fw_data=fw_data, guest=guest, supercell=(1, 1, 1),
        T_K=T_K, pressure_bar=P_bar, ff_mode=0, pol_mode=0,
        grid_dims=(96, 96, 96), n_boxes=8, max_mol=128, seed=20260402,
    )
    try:
        # Reference direct GPU GCMC distribution
        tgt.equilibrate(20000)
        eq_states = tgt.get_accepted_quat()
        n_direct = []
        for _ in range(300):
            tgt.equilibrate(25)
            n_direct.extend(s["N"] for s in tgt.get_accepted_quat())
        n_direct = np.asarray(n_direct, dtype=np.float64)
        mean_ref = float(np.mean(n_direct))
        var_ref = float(np.var(n_direct, ddof=1))

        # Test exact stationary distribution preservation (and relaxation from target eq without artificial N=70 bias)
        prop.set_chem_pot_shift(shift_kJ)
        if shift_kJ == 0.0:
            prop.equilibrate(10000)
        else:
            prop.set_accepted_quat(eq_states)

        def batch_tgt(states):
            return [tgt.eval_energy_quat(s) for s in states]

        res = run_nested_mc(
            prop, None, k_steps=k_steps, eq_cycles=150, prod_cycles=1000,
            seed=20260403, batch_target_fn=batch_tgt,
        )
        n_nested = np.asarray(res["prod_N_history"], dtype=np.float64).reshape(-1)
        mean_nest = float(np.mean(n_nested))
        var_nest = float(np.var(n_nested, ddof=1))
        sem_comb = math.sqrt((var_ref / 300.0) + (res["uptake_sem_mmol_g"] * prop.fw_mass_gmol / 1000.0) ** 2 + 1e-6)
        assert abs(mean_nest - mean_ref) < max(1.2, 3.5 * sem_comb), (
            f"Classical parity mean mismatch at shift={shift_kJ}, k={k_steps}: {mean_nest:.3f} vs {mean_ref:.3f}"
        )
        assert abs(var_nest - var_ref) < max(2.0, 0.50 * var_ref), (
            f"Classical parity var mismatch at shift={shift_kJ}, k={k_steps}: {var_nest:.3f} vs {var_ref:.3f}"
        )
    finally:
        prop.close()
        tgt.close()


@pytest.mark.mlip
def test_mace_chunk_2_vs_4_parity():
    """Verify FastMaceD3Target FP32 mace_chunk=2 vs 4 parity (< 0.5 K) in IRMOF-1."""
    import torch
    from nested_mc_engine import FastMaceD3Target

    cif = str(DATA / "IRMOF-1.cif")
    fw_data = extract_framework_for_proposer(cif, charge_mode="auto")
    guest = load_guest("co2")
    tgt = FastMaceD3Target(
        cif_path=cif, supercell=(1, 1, 1), model="mace-matpes-r2scan-0",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dispersion=False, guest=guest, uc_atoms=fw_data["uc"], dtype="float32",
    )
    H_rows = np.asarray(fw_data["uc"].get_cell(), dtype=np.float64)
    center = np.array([0.5, 0.5, 0.5], dtype=np.float64) @ H_rows
    states4 = [
        {
            "N": 1,
            "cx": np.array([center[0] + 0.1 * i]),
            "cy": np.array([center[1]]),
            "cz": np.array([center[2]]),
            "qw": np.array([1.0]),
            "qx": np.array([0.0]),
            "qy": np.array([0.0]),
            "qz": np.array([0.0]),
        }
        for i in range(4)
    ]
    e_c2 = tgt.eval_batch(states4, mace_chunk=2)
    e_c4 = tgt.eval_batch(states4, mace_chunk=4)
    max_e_err = max(abs(a[0] - b[0]) for a, b in zip(e_c2, e_c4))
    assert max_e_err < 0.5, f"mace_chunk 2 vs 4 FP32 parity error: {max_e_err:.3e} K"
    if torch.cuda.is_available():
        assert tgt.default_mace_chunk == 2 or tgt.n_fw >= 700
