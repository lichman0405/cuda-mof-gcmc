"""Phase 1 Mathematical & Bit-Exact Verification Suite for Auto-MLIP + 128-Thread GPU-CBMC."""

import json
import math
from pathlib import Path
import numpy as np
import pytest

from guests import load_guest
from mlip_distill import (
    _unscaled_gg_pair_energy_K,
    assign_framework_environments,
    distill_mlip_surrogate,
    load_distill_cache,
    make_distilled_guest_spec,
)
from nested_mc_engine import CudaGcmcProposer, K_TO_KJMOL, run_nested_mc
from qeq_charges import extract_framework_for_proposer
from run_mlip_gcmc import compute_grid_dims

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def _make_synthetic_oms_distill_res(fw_data: dict, guest, grid_dims=(64, 64, 24), well_depth_K: float = -4200.0):
    """Build deterministic physical 3D unit-cell external grids + 2B LUT for classical verification."""
    nx, ny, nz = [int(d) for d in grid_dims]
    uc = fw_data["uc"]
    H = np.asarray(uc.get_cell(), dtype=np.float64)
    inv_H = np.linalg.inv(H)
    fw_frac = np.mod(np.asarray(uc.get_scaled_positions(), dtype=np.float64), 1.0)
    Z = np.asarray(uc.get_atomic_numbers(), dtype=np.int32)

    sa = np.arange(nx, dtype=np.float64) / float(nx)
    sb = np.arange(ny, dtype=np.float64) / float(ny)
    sc = np.arange(nz, dtype=np.float64) / float(nz)
    kz, ky, kx = np.meshgrid(sc, sb, sa, indexing="ij")
    f_pts = np.stack([kx.ravel(), ky.ravel(), kz.ravel()], axis=1)

    df = f_pts[:, None, :] - fw_frac[None, :, :]
    df -= np.round(df)
    dr = df @ H
    d_mat = np.sqrt(np.sum(dr * dr, axis=-1))

    is_mg = (Z == 12)
    d_nonm = np.min(d_mat[:, ~is_mg], axis=1) if np.any(is_mg) else np.min(d_mat, axis=1)
    d_mg = np.min(d_mat[:, is_mg], axis=1) if np.any(is_mg) else np.full(len(f_pts), 99.0)

    # Smooth potential well around non-metal wall (3.3 A) and OMS Mg2+ (2.35 A)
    v_o = (
        600.0 * np.exp(-((np.maximum(d_nonm, 1.5) - 2.6) / 0.25) ** 2)
        - 1600.0 * np.exp(-((d_nonm - 3.35) / 0.55) ** 2)
        + 0.45 * well_depth_K * np.exp(-((d_mg - 2.35) / 0.30) ** 2)
        + 25.0 * np.power(2.20 / np.maximum(d_mg, 1.4), 12.0) * (d_mg < 3.0)
        + 25.0 * np.power(2.35 / np.maximum(d_nonm, 1.4), 12.0) * (d_nonm < 3.2)
    )
    v_c = (
        -1200.0 * np.exp(-((d_nonm - 3.45) / 0.55) ** 2)
        + 25.0 * np.power(2.20 / np.maximum(d_nonm, 1.4), 12.0) * (d_nonm < 3.2)
    )
    v_o = np.where((d_nonm < 1.8) | (d_mg < 1.8), 1.0e6, np.clip(v_o, -2.0e5, 1.0e5))
    v_c = np.where((d_nonm < 1.8) | (d_mg < 1.8), 1.0e6, np.clip(v_c, -2.0e5, 1.0e5))

    n_chan = max(1, max(s.lj_type for s in guest.sites) + 1)
    grids = np.zeros((n_chan, nz, ny, nx), dtype=np.float32)
    for c in range(n_chan):
        grids[c] = (v_c if c == 0 else v_o).astype(np.float32).reshape(nz, ny, nx)

    r_lut = np.linspace(2.0, 10.0, 128, dtype=np.float64)
    lut_K = (-120.0 * np.exp(-((r_lut - 3.8) / 0.9) ** 2) * (r_lut < 9.5)).astype(np.float32)
    lut_K[-1] = 0.0
    return {
        "grids_u_lj": np.ascontiguousarray(grids, dtype=np.float32),
        "gg_com_lut_K": lut_K,
        "gg_q_scale": 0.92,
        "gg_lj_scale": 1.08,
        "gg_lut_rmin_A": 2.0,
        "gg_lut_rmax_A": 10.0,
    }


def test_env_classification_and_guest_channels():
    for cif_name, min_envs in [("MgMOF-74.cif", 6), ("Cu-BTC.cif", 4), ("IRMOF-1.cif", 4)]:
        fw = extract_framework_for_proposer(str(DATA / cif_name), charge_mode="auto")
        env = assign_framework_environments(fw["uc"], charges=fw["charges"])
        assert len(env["unique_sigs"]) >= min_envs
        assert len(env["env_ids"]) == len(fw["uc"])
        assert np.sum(env["env_counts"]) == len(fw["uc"])
        if cif_name == "MgMOF-74.cif":
            assert any(sig[0] == 12 and len(sig[1]) == 5 for sig in env["unique_sigs"])

    for g_name, exp_chan in [("co2", 2), ("h2o", 2), ("ch4", 2), ("n2", 1), ("ar", 1)]:
        dg = make_distilled_guest_spec(load_guest(g_name))
        real_chans = {s.lj_type for s in dg.sites if s.Z > 0}
        assert len(real_chans) == exp_chan
        assert all(s.eps_K > 0.0 and s.sigma_A > 0.0 for s in dg.sites if s.Z > 0)


def test_external_grid_and_gg_lut_parity_and_reject_restore():
    """Gate 1: External 3D grid + 2B LUT energy parity (<1e-8 K) and outer-reject bit-exact restore."""
    cif = str(DATA / "MgMOF-74.cif")
    fw_data = extract_framework_for_proposer(cif, charge_mode="auto")
    guest = make_distilled_guest_spec(load_guest("co2"))
    distill_res = _make_synthetic_oms_distill_res(fw_data, guest, grid_dims=(64, 64, 24))

    prop = CudaGcmcProposer.from_distilled(
        fw_data=fw_data, guest=guest, distill_res=distill_res,
        supercell=(1, 1, 2), T_K=298.0, pressure_bar=0.28,
        n_boxes=4, max_mol=64, seed=20261101, cbmc_k=128, p_ins_del=0.60,
    )
    try:
        prop.equilibrate(400)
        st_acc = prop.get_accepted_quat()
        assert sum(s["N"] for s in st_acc) > 0

        # 1. Incremental U_ref vs fresh eval_energy_quat (<1e-8 K)
        for s in st_acc:
            u_eval, ov = prop.eval_energy_quat(s)
            assert not ov
            assert abs(u_eval - s["U_ref_K"]) < 1e-8

        # 2. Outer reject preserves state bit-for-bit while RNG strictly advances
        rng0 = prop.get_rng_states()
        _ = prop.propose_k_quat(15)
        prop.commit([0, 0, 0, 0])
        rng1 = prop.get_rng_states()
        st_after = prop.get_accepted_quat()
        assert not np.array_equal(rng0, rng1)
        for a, b in zip(st_acc, st_after):
            assert a["N"] == b["N"]
            assert a["U_ref_K"] == b["U_ref_K"]
            assert np.array_equal(a["cx"], b["cx"])
            assert np.array_equal(a["qw"], b["qw"])

        # 3. Verify Python 2-body formula matches CUDA pair energy for a dimer in open space
        H = np.asarray(prop.H_cols.T, dtype=np.float64)
        inv_H = np.linalg.inv(H)
        sa = {"N": 1, "cx": np.array([1.0]), "cy": np.array([1.0]), "cz": np.array([1.0]),
              "qw": np.array([1.0]), "qx": np.array([0.0]), "qy": np.array([0.0]), "qz": np.array([0.0])}
        sb = {"N": 1, "cx": np.array([4.8]), "cy": np.array([1.2]), "cz": np.array([1.0]),
              "qw": np.array([0.8]), "qx": np.array([0.6]), "qy": np.array([0.0]), "qz": np.array([0.0])}
        sab = {"N": 2, "cx": np.array([1.0, 4.8]), "cy": np.array([1.0, 1.2]), "cz": np.array([1.0, 1.0]),
               "qw": np.array([1.0, 0.8]), "qx": np.array([0.0, 0.6]), "qy": np.array([0.0, 0.0]), "qz": np.array([0.0, 0.0])}
        ua, _ = prop.eval_energy_quat(sa)
        ub, _ = prop.eval_energy_quat(sb)
        uab, _ = prop.eval_energy_quat(sab)
        u2_cuda = uab - ua - ub
        ulj0, uq0, rc_com, _ = _unscaled_gg_pair_energy_K(
            sa, sb, H, inv_H, guest, r_cut_gg=prop.r_cut_gg, dsf_alpha=0.20
        )
        lut_arr = distill_res["gg_com_lut_K"]
        u_pos = (rc_com - 2.0) * (len(lut_arr) - 1) / (10.0 - 2.0)
        i0 = int(math.floor(u_pos))
        f = u_pos - i0
        lut_val = float(lut_arr[i0]) * (1.0 - f) + float(lut_arr[i0 + 1]) * f
        u2_py = distill_res["gg_lj_scale"] * ulj0 + (distill_res["gg_q_scale"] ** 2) * uq0 + lut_val
        assert abs(u2_cuda - u2_py) < 1e-6, f"CUDA vs Python 2B LUT mismatch: {u2_cuda} vs {u2_py}"
    finally:
        prop.close()


def test_gpu_cbmc_128_vs_1_zero_bias_from_empty_and_high_loading():
    """Gate 2: 128-thread GPU-CBMC must sample the exact same grand-canonical distribution as cbmc_k=1."""
    cif = str(DATA / "MgMOF-74.cif")
    fw_data = extract_framework_for_proposer(cif, charge_mode="auto")
    guest = load_guest("co2_epm2")
    grid_dims = compute_grid_dims(fw_data["uc"], 0.14)
    T_K, P_bar = 298.0, 0.02

    p_ref = CudaGcmcProposer.from_atoms(
        fw_data=fw_data, guest=guest, supercell=(1, 1, 2),
        T_K=T_K, pressure_bar=P_bar, ff_mode=1, pol_mode=0,
        grid_dims=grid_dims, n_boxes=16, max_mol=128, seed=20262001, cbmc_k=1, p_ins_del=0.50,
    )
    p_cbmc_empty = CudaGcmcProposer.from_atoms(
        fw_data=fw_data, guest=guest, supercell=(1, 1, 2),
        T_K=T_K, pressure_bar=P_bar, ff_mode=1, pol_mode=0,
        grid_dims=grid_dims, n_boxes=16, max_mol=128, seed=20262002, cbmc_k=128, p_ins_del=0.50,
    )
    p_cbmc_high = CudaGcmcProposer.from_atoms(
        fw_data=fw_data, guest=guest, supercell=(1, 1, 2),
        T_K=T_K, pressure_bar=P_bar, ff_mode=1, pol_mode=0,
        grid_dims=grid_dims, n_boxes=16, max_mol=128, seed=20262003, cbmc_k=128, p_ins_del=0.50,
    )
    try:
        p_cbmc_high.set_chem_pot_shift(-12.0)
        p_cbmc_high.equilibrate(1500)
        hl_N = [s["N"] for s in p_cbmc_high.get_accepted_quat()]
        assert min(hl_N) >= 35
        p_cbmc_high.set_chem_pot_shift(0.0)

        p_ref.equilibrate(35000)
        p_cbmc_empty.equilibrate(6000)
        p_cbmc_high.equilibrate(6000)

        def _sample(prop_obj, stride, n_draws):
            ns = [[] for _ in range(16)]
            for _ in range(n_draws):
                prop_obj.equilibrate(stride)
                for b, s in enumerate(prop_obj.get_accepted_quat()):
                    ns[b].append(float(s["N"]))
            bmeans = np.array([np.mean(x) for x in ns])
            return float(np.mean(bmeans)), float(np.std(bmeans, ddof=1) / math.sqrt(16)), float(np.var(np.asarray(ns), ddof=1))

        m_ref, se_ref, v_ref = _sample(p_ref, 150, 120)
        m_emp, se_emp, v_emp = _sample(p_cbmc_empty, 35, 120)
        m_hi, se_hi, v_hi = _sample(p_cbmc_high, 35, 120)

        for tag, m_val, se_val, v_val in [("empty", m_emp, se_emp, v_emp), ("high", m_hi, se_hi, v_hi)]:
            se_c = math.sqrt(se_ref ** 2 + se_val ** 2 + 1e-6)
            assert abs(m_val - m_ref) < max(1.0, 3.0 * se_c), (
                f"GPU-CBMC ({tag}) mean {m_val:.3f}+-{se_val:.3f} vs k=1 ref {m_ref:.3f}+-{se_ref:.3f}"
            )
            assert abs(v_val - v_ref) < max(2.5, 0.45 * v_ref), (
                f"GPU-CBMC ({tag}) var {v_val:.2f} vs k=1 ref {v_ref:.2f}"
            )
    finally:
        p_ref.close()
        p_cbmc_empty.close()
        p_cbmc_high.close()


@pytest.fixture(scope="module")
def mercado_distill_bundle(tmp_path_factory):
    """Run real distill_mlip_surrogate + cache round-trip once against classical Mercado oracle."""
    cif = str(DATA / "MgMOF-74.cif")
    fw_data = extract_framework_for_proposer(cif, charge_mode="auto")
    guest_tgt = load_guest("co2_epm2")
    guest_dist = make_distilled_guest_spec(guest_tgt)
    grid_dims = compute_grid_dims(fw_data["uc"], 0.14)
    tgt = CudaGcmcProposer.from_atoms(
        fw_data=fw_data, guest=guest_tgt, supercell=(1, 1, 2),
        T_K=298.0, pressure_bar=0.02, ff_mode=1, pol_mode=0,
        grid_dims=grid_dims, n_boxes=1, max_mol=128, seed=20263001, cbmc_k=128, p_ins_del=0.50,
    )

    class _MockTargetOracle:
        def __init__(self, prop_obj, fw_uc):
            import torch
            self._prop = prop_obj
            self.uc = fw_uc
            self.fw = fw_uc.repeat((1, 1, 2))
            self.r_max = 6.0
            self.dispersion = False
            self.d3_xc = ""
            self.dtype = "float32"
            self.model_source = "mock_mercado"
            self.device = torch.device("cuda:0") if torch.cuda.is_available() else None
            self.guest_local_pos = guest_dist.real_local_positions()
            self.eval_calls = 0

        def eval_batch(self, states, mace_chunk=None):
            self.eval_calls += len(states)
            return [self._prop.eval_energy_quat(s) for s in states]

    oracle = _MockTargetOracle(tgt, fw_data["uc"])
    cache_dir = tmp_path_factory.mktemp("distill_cache_gate3")
    distill_res = distill_mlip_surrogate(
        cif_path=cif,
        fw_data=fw_data,
        guest=guest_dist,
        supercell=(1, 1, 2),
        target=oracle,
        grid_dims=(48, 48, 20),
        n_1b=320,
        n_2b=80,
        distill_seed=20263055,
        cache_dir=str(cache_dir),
        model_name="mock_mercado",
        dispersion=False,
        dispersion_xc="",
        dtype="float32",
    )
    assert distill_res["summary"]["r2_1b"] > 0.85
    reloaded = load_distill_cache(distill_res["cache_npz_path"])
    assert np.allclose(reloaded["grids_u_lj"], distill_res["grids_u_lj"], atol=1e-6)
    try:
        yield cif, fw_data, guest_tgt, guest_dist, grid_dims, reloaded
    finally:
        tgt.close()


@pytest.mark.parametrize("shift_kJ", [-3.0, 0.0, +3.0])
def test_cross_potential_nested_mc_zero_bias_with_cbmc_128(mercado_distill_bundle, shift_kJ):
    """Gate 3: Distilled external grid + 2B LUT with cbmc_k=128 samples exact Mercado target with zero bias."""
    _, fw_data, guest_tgt, guest_dist, grid_dims, distill_res = mercado_distill_bundle
    T_K, P_bar = 298.0, 0.02

    tgt = CudaGcmcProposer.from_atoms(
        fw_data=fw_data, guest=guest_tgt, supercell=(1, 1, 2),
        T_K=T_K, pressure_bar=P_bar, ff_mode=1, pol_mode=0,
        grid_dims=grid_dims, n_boxes=8, max_mol=128, seed=20263001, cbmc_k=128, p_ins_del=0.50,
    )
    prop = CudaGcmcProposer.from_distilled(
        fw_data=fw_data, guest=guest_dist, distill_res=distill_res,
        supercell=(1, 1, 2), T_K=T_K, pressure_bar=P_bar,
        n_boxes=8, max_mol=128, seed=20263002 + int(shift_kJ * 10), cbmc_k=128, p_ins_del=0.50,
    )
    try:
        tgt.equilibrate(5000)
        box_n = []
        for _ in range(150):
            tgt.equilibrate(25)
            box_n.append([float(s["N"]) for s in tgt.get_accepted_quat()])
        arr_ref = np.asarray(box_n, dtype=np.float64)
        mean_ref = float(np.mean(arr_ref))
        sem_ref = float(np.std(np.mean(arr_ref, axis=0), ddof=1) / math.sqrt(8))

        prop.set_chem_pot_shift(shift_kJ)
        prop.equilibrate(4000)
        res = run_nested_mc(
            prop, None, k_steps=12, eq_cycles=150, prod_cycles=350,
            seed=20263100 + int(shift_kJ * 10),
            batch_target_fn=lambda sts: [tgt.eval_energy_quat(s) for s in sts],
        )
        mean_nest = float(np.mean(res["prod_N_history"]))
        sem_nest = float(res["uptake_sem_mmol_g"] * prop.fw_mass_gmol / 1000.0)
        se_c = math.sqrt(sem_ref ** 2 + sem_nest ** 2 + 1e-6)
        assert res["prod_ins_acc"] >= 25
        assert res["prod_del_acc"] >= 25
        assert abs(mean_nest - mean_ref) < max(1.5, 3.2 * se_c), (
            f"Nested (cbmc_k=128, shift={shift_kJ}) mean {mean_nest:.3f}+-{sem_nest:.3f} vs {mean_ref:.3f}+-{sem_ref:.3f}"
        )
    finally:
        prop.close()
        tgt.close()


def test_cbmc_128_external_grid_bit_exact_checkpoint_resume():
    """Gate 4: Bit-exact checkpoint save and extend resume under cbmc_k=128 + external grids + 2B LUT."""
    cif = str(DATA / "MgMOF-74.cif")
    fw_data = extract_framework_for_proposer(cif, charge_mode="auto")
    guest = make_distilled_guest_spec(load_guest("co2"))
    distill_res = _make_synthetic_oms_distill_res(fw_data, guest, grid_dims=(64, 64, 24))
    T_K, P_bar = 298.0, 0.28

    prop_tgt = CudaGcmcProposer.from_distilled(
        fw_data=fw_data, guest=guest, distill_res=distill_res,
        supercell=(1, 1, 2), T_K=T_K, pressure_bar=P_bar,
        n_boxes=1, max_mol=128, seed=20264001, cbmc_k=128, p_ins_del=0.60,
    )
    prop_cont = CudaGcmcProposer.from_distilled(
        fw_data=fw_data, guest=guest, distill_res=distill_res,
        supercell=(1, 1, 2), T_K=T_K, pressure_bar=P_bar,
        n_boxes=4, max_mol=128, seed=20264002, cbmc_k=128, p_ins_del=0.60,
    )
    try:
        prop_cont.set_chem_pot_shift(-1.5)
        res_cont = run_nested_mc(
            prop_cont, None, k_steps=10, eq_cycles=40, prod_cycles=80,
            seed=20264099, batch_target_fn=lambda sts: [prop_tgt.eval_energy_quat(s) for s in sts],
        )
    finally:
        prop_cont.close()

    ckpt_store = {}
    prop_p1 = CudaGcmcProposer.from_distilled(
        fw_data=fw_data, guest=guest, distill_res=distill_res,
        supercell=(1, 1, 2), T_K=T_K, pressure_bar=P_bar,
        n_boxes=4, max_mol=128, seed=20264002, cbmc_k=128, p_ins_del=0.60,
    )
    try:
        prop_p1.set_chem_pot_shift(-1.5)
        _ = run_nested_mc(
            prop_p1, None, k_steps=10, eq_cycles=40, prod_cycles=40,
            seed=20264099, batch_target_fn=lambda sts: [prop_tgt.eval_energy_quat(s) for s in sts],
            checkpoint_fn=lambda c: ckpt_store.__setitem__("c", json.loads(json.dumps(c))),
            checkpoint_every=80,
        )
    finally:
        prop_p1.close()

    prop_p2 = CudaGcmcProposer.from_distilled(
        fw_data=fw_data, guest=guest, distill_res=distill_res,
        supercell=(1, 1, 2), T_K=T_K, pressure_bar=P_bar,
        n_boxes=4, max_mol=128, seed=99999999, cbmc_k=128, p_ins_del=0.60,
    )
    try:
        prop_p2.set_chem_pot_shift(-1.5)
        res_resumed = run_nested_mc(
            prop_p2, None, k_steps=10, eq_cycles=40, prod_cycles=80,
            seed=20264099, batch_target_fn=lambda sts: [prop_tgt.eval_energy_quat(s) for s in sts],
            resume_state=ckpt_store["c"],
        )
    finally:
        prop_p2.close()
        prop_tgt.close()

    assert res_resumed["prod_N_history"] == res_cont["prod_N_history"]
    assert np.allclose(res_resumed["prod_U_history"], res_cont["prod_U_history"], atol=1e-12)
    assert np.allclose(res_resumed["prod_U_ref_history"], res_cont["prod_U_ref_history"], atol=1e-12)
    assert res_resumed["acceptance_breakdown"] == res_cont["acceptance_breakdown"]
    assert res_resumed["final_cuda_rng_states"] == res_cont["final_cuda_rng_states"]
    for b in range(4):
        assert np.array_equal(res_resumed["final_states"][b]["cx"], res_cont["final_states"][b]["cx"])
