"""P1 (§4.1-4.3): Stan/Vehtari diagnostics, non-fallback Qst, bit-exact resume, metadata checks, init separation, and shard merging."""

import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from guests import load_guest
from mcmc_diagnostics import compute_independent_qst, evaluate_v2_convergence
from multi_gpu import merge_replica_shard_results
from nested_mc_engine import CudaGcmcProposer, K_TO_KJMOL, run_nested_mc
from qeq_charges import extract_framework_for_proposer
from run_metadata import collect_run_metadata, validate_resume_metadata
from run_mlip_gcmc import run_pressure_point

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


class ClassicalTargetWrapper:
    """Fast GPU classical energy target matching Target interface for P1 end-to-end tests."""

    def __init__(self, cif_path: str, fw_data: dict, guest, supercell=(1, 1, 2), ff_mode: int = 0, pol_mode: int = 0):
        self.prop = CudaGcmcProposer.from_atoms(
            fw_data=fw_data,
            guest=guest,
            supercell=tuple(supercell),
            T_K=298.0,
            pressure_bar=0.28,
            ff_mode=ff_mode,
            pol_mode=pol_mode,
            grid_dims=(128, 128, 48),
            n_boxes=1,
            max_mol=128,
            seed=20269991,
        )
        from ase.io import read as ase_read
        self.uc = ase_read(str(cif_path))
        self.fw = self.uc.repeat(tuple(supercell))
        self.guest_local_pos = guest.real_local_positions()
        self.n_evals = 0
        self.device = None

    def eval_batch(self, states, mace_chunk=None):
        self.n_evals += len(states)
        return [self.prop.eval_energy_quat(s) for s in states]

    def __call__(self, state):
        self.n_evals += 1
        return self.prop.eval_energy_quat(state)

    def close(self):
        self.prop.close()


def test_vehtari_diagnostics_synthetic_cases():
    rng = np.random.default_rng(12345)
    n_cyc, n_c = 1000, 4
    conv = 0.2

    # 1. All constant chains -> non-estimable fail, ess_bulk = 0
    const_n = np.full((n_cyc, n_c), 5.0)
    const_u = np.full((n_cyc, n_c), -100.0)
    d_const = evaluate_v2_convergence(const_n.tolist(), const_u.tolist(), conv, 298.0, 0, 0, 0, 0, 0.0, 0.0)
    assert d_const["status"] == "fail"
    assert not d_const["N_stats"]["estimable"]
    assert d_const["N_stats"]["ess_bulk"] == 0.0
    assert d_const["N_stats"]["rhat_max"] is None

    # 2. Distinct constant plateaus ([10, 10, 20, 20]) -> fail
    plat_n = np.column_stack([np.full(n_cyc, 10.0), np.full(n_cyc, 10.0), np.full(n_cyc, 20.0), np.full(n_cyc, 20.0)])
    d_plat = evaluate_v2_convergence(plat_n.tolist(), (-20.0 * plat_n).tolist(), conv, 298.0, 0, 0, 0, 0, 0.0, float(np.std([2, 2, 4, 4], ddof=1) / 2.0))
    assert d_plat["status"] == "fail"
    assert d_plat["N_stats"]["distinct_constant_plateaus"] is True
    assert d_plat["N_stats"]["ess_bulk"] == 0.0

    # 3. Opposite-sign linear drift across replicas (pooled mean flat, individual chains drifting!)
    t = np.linspace(-1.0, 1.0, n_cyc)
    drift_n = np.column_stack([
        20.0 + 4.0 * t + rng.normal(0.0, 0.5, n_cyc),
        20.0 - 4.0 * t + rng.normal(0.0, 0.5, n_cyc),
        20.0 + 4.0 * t + rng.normal(0.0, 0.5, n_cyc),
        20.0 - 4.0 * t + rng.normal(0.0, 0.5, n_cyc),
    ])
    d_drift = evaluate_v2_convergence(drift_n.tolist(), (-20.0 * drift_n).tolist(), conv, 298.0, 100, 100, 500, 500, 0.5, 0.02)
    assert d_drift["status"] == "fail"
    assert d_drift["N_stats"]["rhat_max"] >= 1.01
    assert d_drift["pooled_abs_drift_mmol_g"] < 0.05
    assert d_drift["max_abs_chain_drift_mmol_g"] > 0.70

    # 4. Distinct-mean stationary chains (N(10,1) vs N(20,1)) -> rank-normalized Rhat ~ 1.73 >> 1.01, fail
    split_means = np.column_stack([
        10.0 + rng.normal(0, 1, n_cyc),
        10.0 + rng.normal(0, 1, n_cyc),
        20.0 + rng.normal(0, 1, n_cyc),
        20.0 + rng.normal(0, 1, n_cyc),
    ])
    d_split = evaluate_v2_convergence(split_means.tolist(), (-20.0 * split_means).tolist(), conv, 298.0, 100, 100, 500, 500, 0.5, 1.5)
    assert d_split["status"] == "fail"
    assert d_split["N_stats"]["rhat_max"] > 1.5

    # 5. AR(1) rho = 0.9, 4 chains x 2000 -> theoretical ESS = 4*2000*(1-0.9)/(1+0.9) ~ 421
    ar_n = np.zeros((2000, 4), dtype=np.float64)
    rho = 0.9
    sig_eps = math.sqrt(1.0 - rho * rho)
    for k in range(1, 2000):
        ar_n[k] = rho * ar_n[k - 1] + rng.normal(0.0, sig_eps, 4)
    ar_u = -25.0 * (ar_n + 20.0) + rng.normal(0.0, 2.0, (2000, 4))
    ar_n_pos = ar_n + 20.0
    d_ar = evaluate_v2_convergence(ar_n_pos.tolist(), ar_u.tolist(), conv, 298.0, 200, 200, 1000, 1000, 0.4, 0.01)
    assert 200.0 <= d_ar["N_stats"]["ess_bulk"] <= 700.0
    assert d_ar["N_stats"]["rhat_max"] < 1.03

    # 6. i.i.d. normal (4 chains x 1000) -> Rhat < 1.01, ess_bulk > 3000, ess_tail > 2500, pass
    iid_n = 25.0 + rng.normal(0.0, 1.0, (1000, 4))
    iid_u = -30.0 * iid_n + rng.normal(0.0, 0.5, (1000, 4))
    d_iid = evaluate_v2_convergence(iid_n.tolist(), iid_u.tolist(), conv, 298.0, 200, 200, 1000, 1000, 0.5, 0.01)
    assert d_iid["status"] == "pass"
    assert d_iid["N_stats"]["rhat_max"] < 1.01
    assert d_iid["N_stats"]["ess_bulk"] > 3000.0
    assert d_iid["N_stats"]["ess_tail"] > 2500.0

    # 7. Dilute sparse limit (98% empty + 1 constant empty chain) -> insufficient, NEVER pass
    dil_n = np.zeros((400, 4), dtype=np.float64)
    dil_n[10:15, 0] = 1.0
    dil_n[100:103, 1] = 1.0
    dil_n[200:202, 2] = 1.0
    # chain 3 identically 0.0
    dil_u = -28.0 * dil_n
    d_dil = evaluate_v2_convergence(dil_n.tolist(), dil_u.tolist(), conv, 298.0, 7, 7, 100, 100, 0.96, 0.002)
    assert d_dil["dilute_limit"] is True
    assert d_dil["status"] == "insufficient"
    assert not d_dil["converged"]


def test_qst_independent_no_silent_fallback():
    rng = np.random.default_rng(777)

    # 1. Zero var(N) -> MUST return None and status="insufficient" (no silent molar fallback)
    n0 = np.full((100, 4), 5.0)
    u0 = np.full((100, 4), -120.0)
    q0 = compute_independent_qst(n0.tolist(), u0.tolist(), T_K=298.0, prod_ins_acc=0, prod_del_acc=0)
    assert q0["qst_fluctuation_kJ_mol"] is None
    assert q0["qst_status"] == "insufficient"
    assert any("varN" in r for r in q0["qst_reasons"])

    # 2. Correlated (N, U) with known true slope dU/dN = -32.0 kJ/mol -> Qst = RT + 32.0
    n1 = 20.0 + rng.normal(0.0, 2.5, (400, 4))
    u1 = -32.0 * n1 + rng.normal(0.0, 1.5, (400, 4))
    q1 = compute_independent_qst(n1.tolist(), u1.tolist(), T_K=298.0, prod_ins_acc=150, prod_del_acc=150)
    expected_qst = 298.0 * K_TO_KJMOL + 32.0
    assert q1["qst_status"] == "pass"
    assert abs(q1["qst_fluctuation_kJ_mol"] - expected_qst) < 0.5

    # 3. Insufficient particle exchanges -> qst_status == "insufficient"
    q2 = compute_independent_qst(n1.tolist(), u1.tolist(), T_K=298.0, prod_ins_acc=2, prod_del_acc=2)
    assert q2["qst_status"] == "insufficient"


def test_true_checkpoint_resume_bit_exact(tmp_path):
    cif = str(DATA / "IRMOF-1.cif")
    fw_data = extract_framework_for_proposer(cif, charge_mode="auto")
    guest = load_guest("ch4")
    T_K, P_bar = 298.0, 5.0

    # Continuous run: eq=40, prod=80
    prop_cont = CudaGcmcProposer.from_atoms(
        fw_data=fw_data, guest=guest, supercell=(1, 1, 1),
        T_K=T_K, pressure_bar=P_bar, ff_mode=0, pol_mode=0,
        grid_dims=(96, 96, 96), n_boxes=4, max_mol=128, seed=20265001,
    )
    prop_tgt = CudaGcmcProposer.from_atoms(
        fw_data=fw_data, guest=guest, supercell=(1, 1, 1),
        T_K=T_K, pressure_bar=P_bar, ff_mode=0, pol_mode=0,
        grid_dims=(96, 96, 96), n_boxes=1, max_mol=128, seed=20265002,
    )
    try:
        def eval_tgt(states):
            return [prop_tgt.eval_energy_quat(s) for s in states]

        prop_cont.set_chem_pot_shift(-2.0)
        prop_cont.equilibrate(1000)
        res_cont = run_nested_mc(
            prop_cont, None, k_steps=10, eq_cycles=40, prod_cycles=80,
            seed=20265099, batch_target_fn=eval_tgt,
        )
    finally:
        prop_cont.close()

    # Interrupted run: eq=40, prod=40 -> save ckpt -> destroy -> new proposer -> resume to prod=80
    ckpt_store = {}
    def _save_ckpt(c):
        ckpt_store["latest"] = json.loads(json.dumps(c))

    prop_p1 = CudaGcmcProposer.from_atoms(
        fw_data=fw_data, guest=guest, supercell=(1, 1, 1),
        T_K=T_K, pressure_bar=P_bar, ff_mode=0, pol_mode=0,
        grid_dims=(96, 96, 96), n_boxes=4, max_mol=128, seed=20265001,
    )
    try:
        prop_p1.set_chem_pot_shift(-2.0)
        prop_p1.equilibrate(1000)
        _ = run_nested_mc(
            prop_p1, None, k_steps=10, eq_cycles=40, prod_cycles=40,
            seed=20265099, batch_target_fn=lambda sts: [prop_tgt.eval_energy_quat(s) for s in sts],
            checkpoint_fn=_save_ckpt, checkpoint_every=80,
        )
    finally:
        prop_p1.close()

    # Recreate proposer with DIFFERENT init seed to prove checkpoint overwrites RNG cleanly
    prop_p2 = CudaGcmcProposer.from_atoms(
        fw_data=fw_data, guest=guest, supercell=(1, 1, 1),
        T_K=T_K, pressure_bar=P_bar, ff_mode=0, pol_mode=0,
        grid_dims=(96, 96, 96), n_boxes=4, max_mol=128, seed=99999999,
    )
    try:
        prop_p2.set_chem_pot_shift(-2.0)
        res_resumed = run_nested_mc(
            prop_p2, None, k_steps=10, eq_cycles=40, prod_cycles=80,
            seed=20265099, batch_target_fn=lambda sts: [prop_tgt.eval_energy_quat(s) for s in sts],
            resume_state=ckpt_store["latest"],
        )
    finally:
        prop_p2.close()
        prop_tgt.close()

    assert res_resumed["prod_N_history"] == res_cont["prod_N_history"]
    assert np.allclose(res_resumed["prod_U_history"], res_cont["prod_U_history"], atol=1e-12)
    assert np.allclose(res_resumed["prod_U_ref_history"], res_cont["prod_U_ref_history"], atol=1e-12)
    assert res_resumed["prod_accepted_history"] == res_cont["prod_accepted_history"]
    assert res_resumed["prod_dN_history"] == res_cont["prod_dN_history"]
    assert res_resumed["acceptance_breakdown"] == res_cont["acceptance_breakdown"]
    assert math.isclose(res_resumed["uptake_mmol_g"], res_cont["uptake_mmol_g"], rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(res_resumed["Qst_kJ_mol"], res_cont["Qst_kJ_mol"], rel_tol=0.0, abs_tol=1e-12)
    assert res_resumed["final_cuda_rng_states"] == res_cont["final_cuda_rng_states"]
    for b in range(4):
        assert res_resumed["final_states"][b]["N"] == res_cont["final_states"][b]["N"]
        assert np.allclose(res_resumed["final_states"][b]["cx"], res_cont["final_states"][b]["cx"], atol=1e-12)


def test_multi_gpu_merge_recomputes_v2_diagnostics():
    rng = np.random.default_rng(314)
    n_cyc = 300
    conv = 0.2

    def make_shard(mean_val: float):
        n_tr = np.clip(np.round(mean_val + rng.normal(0.0, 1.5, (n_cyc, 2))), 0, 60).astype(int)
        u_tr = -30.0 * n_tr + rng.normal(0.0, 1.0, (n_cyc, 2))
        n_hist = n_tr.tolist()
        moments = []
        box_ups = []
        for b in range(2):
            nb = n_tr[:, b].astype(np.float64)
            ub_K = u_tr[:, b] / K_TO_KJMOL
            box_ups.append(float(np.mean(nb) * conv))
            moments.append({
                "n_samples": n_cyc,
                "sum_N": float(np.sum(nb)),
                "sum_N2": float(np.sum(nb * nb)),
                "sum_U": float(np.sum(ub_K)),
                "sum_UN": float(np.sum(ub_K * nb)),
            })
        return {
            "n_boxes": 2,
            "k_steps": 10,
            "eq_cycles": 0,
            "prod_cycles": n_cyc,
            "delta_eps_kJ_mol": 0.0,
            "elapsed_s": 1.0,
            "uptake_mmol_g": float(np.mean(box_ups)),
            "uptake_sem_mmol_g": float(np.std(box_ups, ddof=1) / math.sqrt(2.0)),
            "box_uptakes_mmol_g": box_ups,
            "box_qst_kJ_mol": [32.0, 32.0],
            "box_moments": moments,
            "outer_acc_rate": 0.5,
            "prod_outer_acc_rate": 0.5,
            "prod_ins_att": 100,
            "prod_ins_acc": 40,
            "prod_del_att": 100,
            "prod_del_acc": 40,
            "acceptance_breakdown": {
                "total_trials": 600, "rejected": 300, "acc_empty_to_empty": 0,
                "acc_identity_nonempty": 0, "acc_same_N_changed": 150, "acc_dN_changed": 150,
            },
            "mean_inner_acc_per_burst": 5.0,
            "effective_mc_steps_per_target_eval": 5.0,
            "box_N_history": n_hist,
            "prod_N_history": n_hist,
            "prod_U_history": u_tr.tolist(),
            "prod_U_ref_history": u_tr.tolist(),
            "prod_accepted_history": [[1, 1] for _ in range(n_cyc)],
            "prod_dN_history": [[0, 0] for _ in range(n_cyc)],
            "prod_proposed_dN_history": [[0, 0] for _ in range(n_cyc)],
            "final_states": [{"N": 10, "U_ref_K": -1000.0, "cx": [], "cy": [], "cz": []}] * 2,
            "P_bar": 0.28,
            "fugacity_Pa": 28000.0,
            "fw_mass_gmol": 1000.0 / conv,
            "mlip_evals": 300,
            "total_inner_steps": 6000,
        }

    # Each shard looks internally well mixed, but means are far apart (10 vs 22)
    merged = merge_replica_shard_results([make_shard(10.0), make_shard(22.0)], T_K=298.0, eq_cycles=0, wall_elapsed_s=1.0, conv_mmol_g=conv)
    assert merged["diagnostics"]["status"] == "fail"
    assert not merged["diagnostics"]["converged"]
    assert merged["diagnostics"]["N_stats"]["rhat_max"] > 1.5


def test_resume_metadata_conflict_rejected(tmp_path):
    cif1 = DATA / "MgMOF-74.cif"
    cif2 = tmp_path / "MgMOF-74.cif"
    cif2.write_text(cif1.read_text(encoding="utf-8") + "\n# modified\n", encoding="utf-8")

    fw1 = extract_framework_for_proposer(str(cif1), charge_mode="auto")
    guest = load_guest("co2_epm2")
    args = SimpleNamespace(
        cif=str(cif1), supercell=[1, 1, 2], temp=298.0, charge_mode="auto",
        proposer="mercado", model="mace-matpes-r2scan-0", dispersion="off", xc="r2scan",
        dtype="float32", mace_chunk=None, grid_spacing=0.11, k_steps=25, eq_cycles=2000,
        prod_cycles=8000, pilot_cycles=60, shift_mode="auto", shift_override=None,
        init_mode="empty", init_loading=8.0, boxes=4, max_mol=128, seed=2026, device="cuda",
    )
    meta1 = collect_run_metadata(args, fw1, guest, (128, 128, 48))
    validate_resume_metadata(meta1, meta1, allow_extend_prod=False)

    # Filename identical (`MgMOF-74.cif`) but content hash differs -> must reject!
    args_cif2 = SimpleNamespace(**{**vars(args), "cif": str(cif2)})
    meta_cif2 = collect_run_metadata(args_cif2, fw1, guest, (128, 128, 48))
    with pytest.raises(ValueError, match="cif_sha256"):
        validate_resume_metadata(meta1, meta_cif2)

    for field, val in [
        ("dtype", "float64"),
        ("proposer", "uff"),
        ("shift_override", 4.0),
        ("init_mode", "high_loading"),
        ("framework_charges_sha256", "deadbeef" * 8),
    ]:
        bad = dict(meta1)
        bad[field] = val
        with pytest.raises(ValueError):
            validate_resume_metadata(meta1, bad)

    # Extending prod_cycles allowed only when allow_extend_prod=True
    ext = dict(meta1)
    ext["prod_cycles"] = 16000
    with pytest.raises(ValueError, match="prod_cycles"):
        validate_resume_metadata(meta1, ext, allow_extend_prod=False)
    validate_resume_metadata(meta1, ext, allow_extend_prod=True)


def test_init_mode_empty_vs_high_loading(tmp_path):
    cif = str(DATA / "MgMOF-74.cif")
    fw_data = extract_framework_for_proposer(cif, charge_mode="auto")
    guest = load_guest("co2_epm2")
    tgt = ClassicalTargetWrapper(cif, fw_data, guest, supercell=(1, 1, 2), ff_mode=1, pol_mode=0)
    try:
        res_empty = run_pressure_point(
            fw_data=fw_data, guest=guest, supercell=(1, 1, 2), T_K=298.0, P_bar=0.28,
            ff_mode=1, pol_mode=0, grid_dims=(128, 128, 48), n_boxes=4, max_mol=128,
            k_steps=5, eq_cycles=2, prod_cycles=4, seed=20266001, shift_mode="auto",
            target=tgt, pilot_cycles=0, init_mode="empty",
        )
        assert res_empty["init_state_summary"]["init_mode"] == "empty"
        assert res_empty["init_state_summary"]["init_N_per_box"] == [0, 0, 0, 0]

        res_hl = run_pressure_point(
            fw_data=fw_data, guest=guest, supercell=(1, 1, 2), T_K=298.0, P_bar=0.28,
            ff_mode=1, pol_mode=0, grid_dims=(128, 128, 48), n_boxes=4, max_mol=128,
            k_steps=5, eq_cycles=2, prod_cycles=4, seed=20266002, shift_mode="auto",
            target=tgt, pilot_cycles=0, init_mode="high_loading", init_loading=6.0,
            checkpoint_dir=str(tmp_path / "ckpt"), checkpoint_every=3,
        )
        assert res_hl["init_state_summary"]["init_mode"] == "high_loading"
        assert all(n > 20 for n in res_hl["init_state_summary"]["init_N_per_box"])
        assert res_hl["init_state_summary"]["init_min_gg_A"] >= 2.6
        assert all(o >= 8 for o in res_hl["init_state_summary"]["init_oms_occ_per_box"])
        assert all(u < -20.0 for u in res_hl["init_state_summary"]["init_U_tgt_per_mol_kJ"])
        assert len(res_hl["prod_oms_occ_history"]) == 4
        assert (tmp_path / "ckpt" / "ckpt_P0.28.json").exists()
    finally:
        tgt.close()
