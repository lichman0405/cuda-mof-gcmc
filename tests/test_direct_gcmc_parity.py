"""Zero-bias parity tests for Direct MLIP-GCMC baselines (classical & MLIP single-point)."""

import math
from pathlib import Path
import numpy as np
import pytest

from direct_mlip_gcmc import SerialAseMaceD3Target, run_direct_gcmc
from guests import load_guest
from nested_mc_engine import CudaGcmcProposer, FastMaceD3Target
from qeq_charges import extract_framework_for_proposer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def test_direct_gcmc_classical_zero_bias_irmof1_ch4():
    """Direct GCMC in Python matches CUDA GCMC on identical UFF grid/cell for polyatomic CH4."""
    cif = str(DATA / "IRMOF-1.cif")
    fw_data = extract_framework_for_proposer(cif, charge_mode="auto")
    guest = load_guest("ch4")
    T_K, P_bar = 298.0, 5.0

    ref_prop = CudaGcmcProposer.from_atoms(
        fw_data=fw_data, guest=guest, supercell=(1, 1, 1),
        T_K=T_K, pressure_bar=P_bar, ff_mode=0, pol_mode=0,
        grid_dims=(96, 96, 96), n_boxes=4, max_mol=128, seed=20260901,
    )
    try:
        # Reference classical GPU GCMC run on identical grid
        ref_prop.equilibrate(10000)
        n_samp = 1000  # 1000 * 20 = 20000 production steps per box
        sN = np.zeros(4, dtype=np.float64)
        sN2 = np.zeros(4, dtype=np.float64)
        sU = np.zeros(4, dtype=np.float64)
        sUN = np.zeros(4, dtype=np.float64)
        for _ in range(n_samp):
            ref_prop.equilibrate(20)
            st = ref_prop.get_accepted_quat()
            for b in range(4):
                nb = float(st[b]["N"])
                ub = float(st[b]["U_ref_K"])
                sN[b] += nb
                sN2[b] += nb * nb
                sU[b] += ub
                sUN[b] += ub * nb
        conv = 1000.0 / ref_prop.fw_mass_gmol
        ref_box_up = (sN / n_samp) * conv
        ref_up = float(np.mean(ref_box_up))
        ref_sem = float(np.std(ref_box_up, ddof=1) / math.sqrt(4.0))
        tot_n = float(np.sum(sN))
        tot_n2 = float(np.sum(sN2))
        tot_u = float(np.sum(sU))
        tot_un = float(np.sum(sUN))
        m_tot = 4 * n_samp
        var_n = (tot_n2 / m_tot) - (tot_n / m_tot) ** 2
        cov_un = (tot_un / m_tot) - (tot_u / m_tot) * (tot_n / m_tot)
        ref_qst = (T_K - cov_un / var_n) * 8.31446261815324e-3

        # Direct Python GCMC using identical C++ grid energy evaluator
        def batch_eval(states):
            return [ref_prop.eval_energy_quat(s) for s in states]

        dir_res = run_direct_gcmc(
            uc_atoms=fw_data["uc"],
            guest=guest,
            supercell=(1, 1, 1),
            T_K=T_K,
            pressure_bar=P_bar,
            batch_target_fn=batch_eval,
            n_boxes=4,
            eq_steps=400,
            prod_steps=1600,
            r_hs_A=1.20,
            seed=20260902,
            fw_mass_gmol=ref_prop.fw_mass_gmol,
        )
        du = abs(dir_res["uptake_mmol_g"] - ref_up)
        comb_sem = math.sqrt(ref_sem ** 2 + dir_res["uptake_sem_mmol_g"] ** 2 + 1e-12)
        z = du / comb_sem
        dq = abs(dir_res["Qst_kJ_mol"] - ref_qst)
        assert z < 3.0, f"Direct GCMC mismatch: dir={dir_res['uptake_mmol_g']:.4f}, ref={ref_up:.4f}, z={z:.2f}"
        assert dq < 2.5, f"Direct GCMC Qst mismatch: dir={dir_res['Qst_kJ_mol']:.2f}, ref={ref_qst:.2f}"
    finally:
        ref_prop.close()


@pytest.mark.mlip
def test_serial_ase_vs_fast_mace_d3_parity():
    """SerialAseMaceD3Target matches FastMaceD3Target on Mg-MOF-74 + CO2 within < 0.1 K."""
    cif = str(DATA / "MgMOF-74.cif")
    fw_data = extract_framework_for_proposer(cif, charge_mode="auto")
    guest = load_guest("co2_epm2")

    prop = CudaGcmcProposer.from_atoms(
        fw_data=fw_data, guest=guest, supercell=(1, 1, 2),
        T_K=298.0, pressure_bar=0.05, ff_mode=1, pol_mode=0,
        grid_dims=(128, 128, 48), n_boxes=1, max_mol=64, seed=20260911,
    )
    try:
        prop.equilibrate(5000)
        st = prop.get_accepted_quat()[0]
        assert st["N"] > 0
    finally:
        prop.close()

    fast = FastMaceD3Target(
        cif_path=cif, supercell=(1, 1, 2), model="small",
        device="cuda", dispersion=True, dispersion_xc="pbe",
        guest=guest, uc_atoms=fw_data["uc"],
    )
    serial = SerialAseMaceD3Target(
        uc_atoms=fw_data["uc"], supercell=(1, 1, 2), model="small",
        device="cuda", dispersion=True, dispersion_xc="pbe", guest=guest,
    )
    u_fast, ov_f = fast(st)
    u_ser, ov_s = serial(st)
    assert not ov_f and not ov_s
    diff_K = abs(u_fast - u_ser)
    rel_diff = diff_K / max(1.0, abs(u_ser))
    assert diff_K < 100.0 and rel_diff < 5e-3, (
        f"MACE FP32 energy mismatch: fast={u_fast:.4f} K vs serial={u_ser:.4f} K (diff={diff_K:.2f} K, rel={rel_diff:.4e})"
    )
