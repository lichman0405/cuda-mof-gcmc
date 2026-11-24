"""Unit tests for C++/CUDA proposer, 103-element triclinic MIC, PR EOS parity, Jackknife Qst, and ESS."""

from pathlib import Path
import numpy as np
import pytest
from ase import Atoms

from direct_mlip_gcmc import compute_pure_fugacity_pr_py
from guests import load_guest
from nested_mc_engine import (
    CudaGcmcProposer,
    compute_ess_stats,
    compute_pooled_jackknife_qst,
)
from qeq_charges import extract_framework_for_proposer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def test_exotic_103_element_triclinic_mof_proposer():
    cell = np.array([
        [14.2, 0.0, 0.0],
        [1.1, 13.8, 0.0],
        [0.9, -1.2, 15.1],
    ], dtype=np.float64)
    syms = ["Si", "P", "Sc", "Ti", "Ga", "Se", "Ru", "Ag", "La", "Eu", "Pt", "Lr"]
    frac = np.array([
        [0.05, 0.08, 0.10],
        [0.18, 0.85, 0.14],
        [0.88, 0.12, 0.22],
        [0.12, 0.16, 0.86],
        [0.90, 0.88, 0.84],
        [0.08, 0.45, 0.09],
        [0.45, 0.08, 0.91],
        [0.92, 0.48, 0.11],
        [0.11, 0.91, 0.48],
        [0.86, 0.14, 0.92],
        [0.15, 0.84, 0.89],
        [0.84, 0.86, 0.16],
    ], dtype=np.float64)
    uc = Atoms(syms, scaled_positions=frac, cell=cell, pbc=True)
    charges = np.array([0.4, -0.4, 0.6, -0.6, 0.3, -0.3, 0.5, -0.5, 0.2, -0.2, 0.1, -0.1], dtype=np.float64)
    fw_data = {
        "uc": uc,
        "H_unit_cols": np.ascontiguousarray(cell.T, dtype=np.float64),
        "frac": np.ascontiguousarray(frac, dtype=np.float64),
        "Z": np.ascontiguousarray(uc.get_atomic_numbers(), dtype=np.int32),
        "charges": charges,
        "type_ids": np.full(len(uc), -1, dtype=np.int32),
        "charge_source": "test",
    }
    guest = load_guest("co2")
    prop = CudaGcmcProposer.from_atoms(
        fw_data=fw_data, guest=guest, supercell=(1, 1, 1),
        T_K=298.0, pressure_bar=0.5, ff_mode=0, pol_mode=0,
        grid_dims=(96, 96, 96), n_boxes=2, max_mol=32, seed=20260103,
    )
    try:
        H_rows = prop.H_cols.T
        assert np.max(np.abs(H_rows - cell)) < 1e-12
        c_frac, c_Z = prop.get_unit_atoms()
        assert np.array_equal(c_Z, fw_data["Z"])
        assert np.max(np.abs(c_frac - frac)) < 1e-7
        prop.equilibrate(4000)
        st = prop.get_accepted_quat()
        assert sum(s["N"] for s in st) > 0
        for s in st:
            u_eval, ov = prop.eval_energy_quat(s)
            assert not ov
            assert abs(u_eval - s["U_ref_K"]) < 1e-9
    finally:
        prop.close()


@pytest.mark.parametrize(
    "cif_name,guest_name,pressure_bar",
    [
        ("IRMOF-1.cif", "co2", 1.0),
        ("IRMOF-1.cif", "ch4", 5.0),
        ("IRMOF-1.cif", "n2", 5.0),
        ("IRMOF-1.cif", "ar", 5.0),
        ("MgMOF-74.cif", "h2o", 0.05),
    ],
)
def test_multi_guest_energy_and_state_roundtrip(cif_name, guest_name, pressure_bar):
    fw_data = extract_framework_for_proposer(str(DATA / cif_name), charge_mode="auto")
    guest = load_guest(guest_name)
    prop = CudaGcmcProposer.from_atoms(
        fw_data=fw_data, guest=guest, supercell=(1, 1, 1),
        T_K=298.0, pressure_bar=pressure_bar, ff_mode=0, pol_mode=0,
        grid_dims=(96, 96, 96), n_boxes=2, max_mol=64, seed=20260404,
    )
    try:
        # Verify Python PR EOS matches C++ PR EOS fugacity
        f_py = compute_pure_fugacity_pr_py(298.0, pressure_bar, guest.Tc_K, guest.Pc_bar, guest.omega)
        rel_err = abs(f_py - prop.f_Pa) / max(1.0, prop.f_Pa)
        assert rel_err < 1e-9

        prop.equilibrate(5000)
        st = prop.get_accepted_quat()
        assert sum(s["N"] for s in st) > 0, f"Expected N > 0 for {cif_name} + {guest_name}"
        for s in st:
            u_eval, ov = prop.eval_energy_quat(s)
            assert not ov
            assert abs(u_eval - s["U_ref_K"]) < 1e-9

        prop.set_accepted_quat(st)
        st_rt = prop.get_accepted_quat()
        for a, b in zip(st, st_rt):
            assert abs(a["U_ref_K"] - b["U_ref_K"]) < 1e-9
    finally:
        prop.close()


def test_pooled_jackknife_qst_and_ess_metrics():
    rng = np.random.default_rng(7)
    n_cyc, n_b = 400, 4
    T_K = 298.0
    # Independent Gaussian white-noise trace -> tau_int near 1, high jump rate
    n_wn = rng.normal(20.0, 3.0, size=(n_cyc, n_b))
    u_wn = -4000.0 * n_wn + rng.normal(0.0, 50.0, size=(n_cyc, n_b))
    stats_wn = compute_ess_stats(n_wn.tolist(), u_wn.tolist(), elapsed_s=10.0)
    assert 1.0 <= stats_wn["tau_int_N"] < 2.0
    assert stats_wn["ess_N"] > 800.0
    assert stats_wn["esjd_N"] > 10.0
    assert stats_wn["sjd_N_per_s"] > 1000.0
    assert stats_wn["dN_moves_per_s"] > 100.0

    # Correlated AR(1) rho=0.9 -> larger tau_int, smaller ESS & smaller ESJD
    n_ar = np.zeros_like(n_wn)
    for b in range(n_b):
        x = 20.0
        for t in range(n_cyc):
            x = 20.0 + 0.9 * (x - 20.0) + rng.normal(0.0, 1.0)
            n_ar[t, b] = x
    stats_ar = compute_ess_stats(n_ar.tolist(), (-4000.0 * n_ar).tolist(), elapsed_s=10.0)
    assert stats_ar["tau_int_N"] > 5.0
    assert stats_ar["ess_N"] < stats_wn["ess_N"] * 0.3

    # Pooled + Jackknife Qst recovers -cov(U, N)/var(N) + T
    moments = []
    for b in range(n_b):
        nb = n_wn[:, b]
        ub = u_wn[:, b]
        moments.append({
            "n_samples": n_cyc,
            "sum_N": float(np.sum(nb)),
            "sum_N2": float(np.sum(nb * nb)),
            "sum_U": float(np.sum(ub)),
            "sum_UN": float(np.sum(ub * nb)),
        })
    jack = compute_pooled_jackknife_qst(moments, T_K)
    expected_qst = (T_K + 4000.0) * 8.31446261815324e-3  # ~35.74 kJ/mol
    assert abs(jack["Qst_kJ_mol"] - expected_qst) < 0.2
    assert 0.0 < jack["Qst_sem_kJ_mol"] < 0.5
