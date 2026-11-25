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
    """Verify FP64 exact parity with SerialASE, FP32(node64) within |err(log_alpha)|<=0.01, and D3 switch identity."""
    cif = str(DATA / "MgMOF-74.cif")
    fw_data = extract_framework_for_proposer(cif, charge_mode="auto")
    guest = load_guest("co2_epm2")
    T_K = 298.0

    prop = CudaGcmcProposer.from_atoms(
        fw_data=fw_data, guest=guest, supercell=(1, 1, 2),
        T_K=T_K, pressure_bar=0.05, ff_mode=1, pol_mode=0,
        grid_dims=(128, 128, 48), n_boxes=1, max_mol=64, seed=20260911,
    )
    try:
        prop.equilibrate(5000)
        st_multi = prop.get_accepted_quat()[0]
        assert st_multi["N"] > 0
    finally:
        prop.close()

    st_empty = {
        "N": 0, "cx": np.zeros(0), "cy": np.zeros(0), "cz": np.zeros(0),
        "qw": np.ones(0), "qx": np.zeros(0), "qy": np.zeros(0), "qz": np.zeros(0),
    }
    st_single = {
        "N": 1,
        "cx": np.array([st_multi["cx"][0]], dtype=np.float64),
        "cy": np.array([st_multi["cy"][0]], dtype=np.float64),
        "cz": np.array([st_multi["cz"][0]], dtype=np.float64),
        "qw": np.array([st_multi["qw"][0]], dtype=np.float64),
        "qx": np.array([st_multi["qx"][0]], dtype=np.float64),
        "qy": np.array([st_multi["qy"][0]], dtype=np.float64),
        "qz": np.array([st_multi["qz"][0]], dtype=np.float64),
    }
    st_del = {
        "N": st_multi["N"] - 1,
        "cx": st_multi["cx"][1:].copy(), "cy": st_multi["cy"][1:].copy(), "cz": st_multi["cz"][1:].copy(),
        "qw": st_multi["qw"][1:].copy(), "qx": st_multi["qx"][1:].copy(),
        "qy": st_multi["qy"][1:].copy(), "qz": st_multi["qz"][1:].copy(),
    }
    test_states = [st_empty, st_single, st_multi, st_del]

    comps64 = {}
    for disp in (False, True):
        fast64 = FastMaceD3Target(
            cif_path=cif, supercell=(1, 1, 2), model="mace-matpes-r2scan-0",
            device="cuda", dispersion=disp, dispersion_xc="r2scan",
            guest=guest, uc_atoms=fw_data["uc"], dtype="float64",
        )
        serial64 = SerialAseMaceD3Target(
            uc_atoms=fw_data["uc"], supercell=(1, 1, 2), model="mace-matpes-r2scan-0",
            device="cuda", dispersion=disp, dispersion_xc="r2scan", guest=guest, dtype="float64",
        )
        fast32 = FastMaceD3Target(
            cif_path=cif, supercell=(1, 1, 2), model="mace-matpes-r2scan-0",
            device="cuda", dispersion=disp, dispersion_xc="r2scan",
            guest=guest, uc_atoms=fw_data["uc"], dtype="float32",
        )
        u64_list = [fast64(s)[0] for s in test_states]
        u32_list = [u for u, _ in fast32.eval_batch(test_states, mace_chunk=2)]
        for idx, s in enumerate(test_states):
            u_ser, ov_s = serial64(s)
            assert not ov_s
            assert abs(u64_list[idx] - u_ser) < 1e-3, (
                f"FP64 parity mismatch (disp={disp}, N={s['N']}): fast={u64_list[idx]:.6f} vs ser={u_ser:.6f}"
            )
            err_log_alpha = abs(u32_list[idx] - u64_list[idx]) / T_K
            assert err_log_alpha <= 0.01, (
                f"FP32(node64) state log_alpha error exceeds 0.01 (disp={disp}, N={s['N']}): {err_log_alpha:.5f}"
            )
        # Move pairs: empty->single and multi->del
        for ia, ib in ((0, 1), (2, 3)):
            du32 = u32_list[ib] - u32_list[ia]
            du64 = u64_list[ib] - u64_list[ia]
            move_err = abs(du32 - du64) / T_K
            assert move_err <= 0.01, f"Move pair log_alpha error > 0.01 (disp={disp}): {move_err:.5f}"
        comps64[disp] = [fast64.eval_components(s) for s in test_states]

    # Fixed-config D3 switch identity in FP64
    for idx, s in enumerate(test_states):
        c_off = comps64[False][idx]
        c_on = comps64[True][idx]
        err_sw = abs((c_on["u_int_K"] - c_off["u_int_K"]) - c_on["u_d3_int_K"])
        assert err_sw < 1e-4, f"FP64 D3 switch identity mismatch at N={s['N']}: {err_sw:.4e} K"
