"""Unit tests for multi-GPU ID parsing, replica sharding, and exact statistical merging."""

import math
import numpy as np
import pytest

from multi_gpu import (
    merge_replica_shard_results,
    parse_gpu_ids,
    resolve_worker_cuda_visible_devices,
    split_boxes_across_gpus,
)
from nested_mc_engine import (
    K_TO_KJMOL,
    compute_ess_stats,
    compute_pooled_jackknife_qst,
)
from run_mlip_gcmc import evaluate_convergence_diagnostics


def test_parse_gpu_ids_formats():
    assert parse_gpu_ids(["0", "1", "2", "3"]) == [0, 1, 2, 3]
    assert parse_gpu_ids(["0,1,2,3"]) == [0, 1, 2, 3]
    assert parse_gpu_ids(["2,3", "2", "0,1"]) == [2, 3, 0, 1]
    assert parse_gpu_ids(None) == []
    with pytest.raises(ValueError):
        parse_gpu_ids(["-1"])


def test_resolve_worker_cuda_visible_devices():
    assert resolve_worker_cuda_visible_devices(2, None) == "2"
    assert resolve_worker_cuda_visible_devices(2, "") == "2"
    assert resolve_worker_cuda_visible_devices(2, "0,1,2,3") == "2"
    assert resolve_worker_cuda_visible_devices(1, "4,5,6,7") == "5"
    assert resolve_worker_cuda_visible_devices(5, "4,5,6,7") == "5"


def test_split_boxes_across_gpus():
    assert split_boxes_across_gpus(16, [0, 1, 2, 3]) == [(0, 4), (1, 4), (2, 4), (3, 4)]
    assert split_boxes_across_gpus(10, [0, 1, 2, 3]) == [(0, 3), (1, 3), (2, 2), (3, 2)]
    assert split_boxes_across_gpus(2, [0, 1, 2, 3]) == [(0, 1), (1, 1)]


def _build_synthetic_result(N_full: np.ndarray, U_K_full: np.ndarray, eq_cycles: int, T_K: float, fw_mass_gmol: float, wall_s: float) -> dict:
    """Compute exact unsharded reference statistics from full (tot_cycles, n_boxes) arrays."""
    tot_cycles, n_b = N_full.shape
    prod_cycles = tot_cycles - eq_cycles
    conv_mmol_g = 1000.0 / fw_mass_gmol
    prod_N = N_full[eq_cycles:]
    prod_U_K = U_K_full[eq_cycles:]
    prod_U_kJ = prod_U_K * K_TO_KJMOL

    box_moments = []
    box_uptakes = []
    box_qst = []
    for b in range(n_b):
        sN = float(np.sum(prod_N[:, b]))
        sN2 = float(np.sum(prod_N[:, b] ** 2))
        sU = float(np.sum(prod_U_K[:, b]))
        sUN = float(np.sum(prod_U_K[:, b] * prod_N[:, b]))
        box_moments.append({
            "n_samples": prod_cycles,
            "sum_N": sN, "sum_N2": sN2, "sum_U": sU, "sum_UN": sUN,
        })
        mN = sN / prod_cycles
        mN2 = sN2 / prod_cycles
        mU = sU / prod_cycles
        mUN = sUN / prod_cycles
        box_uptakes.append(mN * conv_mmol_g)
        varN = mN2 - mN * mN
        box_qst.append((T_K - (mUN - mU * mN) / varN) * K_TO_KJMOL)

    jack = compute_pooled_jackknife_qst(box_moments, T_K)
    ess = compute_ess_stats(prod_N.tolist(), prod_U_kJ.tolist(), wall_s)
    prod_hist = prod_N * conv_mmol_g
    half = prod_cycles // 2
    h1 = float(np.mean(prod_hist[:half]))
    h2 = float(np.mean(prod_hist[half:]))

    rng = np.random.default_rng(1234 + n_b)
    return {
        "n_boxes": n_b,
        "k_steps": 25,
        "eq_cycles": eq_cycles,
        "prod_cycles": prod_cycles,
        "delta_eps_kJ_mol": -4.2,
        "elapsed_s": wall_s,
        "uptake_mmol_g": float(np.mean(box_uptakes)),
        "uptake_sem_mmol_g": float(np.std(box_uptakes, ddof=1) / math.sqrt(n_b)) if n_b > 1 else 0.0,
        "box_uptakes_mmol_g": box_uptakes,
        "Qst_estimator": "pooled_jackknife",
        "Qst_kJ_mol": float(jack["Qst_kJ_mol"]),
        "Qst_sem_kJ_mol": float(jack["Qst_sem_kJ_mol"]),
        "Qst_per_box_mean_kJ_mol": float(np.mean(box_qst)),
        "Qst_per_box_sem_kJ_mol": float(np.std(box_qst, ddof=1) / math.sqrt(n_b)) if n_b > 1 else 0.0,
        "box_qst_kJ_mol": box_qst,
        "box_moments": box_moments,
        "mean_molar_qst_kJ_mol": float(np.mean([(T_K - (m["sum_U"] / m["sum_N"])) * K_TO_KJMOL for m in box_moments])),
        "tau_int_N": ess["tau_int_N"],
        "ess_N": ess["ess_N"],
        "ess_N_per_s": ess["ess_N_per_s"],
        "tau_int_U": ess["tau_int_U"],
        "ess_U": ess["ess_U"],
        "ess_U_per_s": ess["ess_U_per_s"],
        "esjd_N": ess["esjd_N"],
        "sjd_N_per_s": ess["sjd_N_per_s"],
        "dN_moves_per_s": ess["dN_moves_per_s"],
        "outer_acc_rate": float(0.32),
        "prod_outer_acc_rate": float(0.31),
        "prod_ins_att": int(10 * n_b),
        "prod_ins_acc": int(4 * n_b),
        "prod_del_att": int(9 * n_b),
        "prod_del_acc": int(4 * n_b),
        "mean_inner_acc_per_burst": float(14.5),
        "effective_mc_steps_per_target_eval": float(8.0),
        "box_N_history": N_full.tolist(),
        "prod_N_history": prod_N.tolist(),
        "prod_U_history": prod_U_kJ.tolist(),
        "final_states": [{"N": int(N_full[-1, b])} for b in range(n_b)],
        "P_bar": 5.0,
        "T_K": T_K,
        "fugacity_Pa": 495000.0,
        "fw_mass_gmol": fw_mass_gmol,
        "first_half_uptake_mmol_g": h1,
        "second_half_uptake_mmol_g": h2,
        "stationarity_drift_mmol_g": abs(h2 - h1),
        "mlip_evals": int((eq_cycles + prod_cycles + 1) * n_b + rng.integers(0, 1)),
        "total_inner_steps": int(25 * (eq_cycles + prod_cycles) * n_b),
    }


def test_merge_replica_shards_exact_equivalence():
    rng = np.random.default_rng(20261124)
    eq_cycles, prod_cycles = 20, 60
    tot_cycles = eq_cycles + prod_cycles
    n_boxes = 8
    T_K = 298.0
    fw_mass_gmol = 6155.0
    wall_s = 12.5

    N_full = rng.integers(30, 65, size=(tot_cycles, n_boxes), dtype=np.int64)
    U_K_full = -2100.0 * N_full + rng.normal(0.0, 450.0, size=(tot_cycles, n_boxes))

    ref = _build_synthetic_result(N_full, U_K_full, eq_cycles, T_K, fw_mass_gmol, wall_s)

    # Split unevenly (3 + 2 + 2 + 1 boxes across 4 GPUs) to verify non-uniform sharding
    slices = [(0, 3), (3, 5), (5, 7), (7, 8)]
    shards = [
        _build_synthetic_result(N_full[:, a:b], U_K_full[:, a:b], eq_cycles, T_K, fw_mass_gmol, wall_s=3.2 + 0.1 * i)
        for i, (a, b) in enumerate(slices)
    ]

    merged = merge_replica_shard_results(
        shards=shards,
        T_K=T_K,
        eq_cycles=eq_cycles,
        wall_elapsed_s=wall_s,
        conv_mmol_g=1000.0 / fw_mass_gmol,
        diagnostics_fn=evaluate_convergence_diagnostics,
    )

    for key in (
        "uptake_mmol_g",
        "uptake_sem_mmol_g",
        "Qst_kJ_mol",
        "Qst_sem_kJ_mol",
        "Qst_per_box_mean_kJ_mol",
        "Qst_per_box_sem_kJ_mol",
        "mean_molar_qst_kJ_mol",
        "tau_int_N",
        "ess_N",
        "ess_N_per_s",
        "tau_int_U",
        "ess_U",
        "ess_U_per_s",
        "esjd_N",
        "sjd_N_per_s",
        "dN_moves_per_s",
        "first_half_uptake_mmol_g",
        "second_half_uptake_mmol_g",
        "stationarity_drift_mmol_g",
        "outer_acc_rate",
        "prod_outer_acc_rate",
    ):
        assert abs(merged[key] - ref[key]) < 1e-12, f"Mismatch in {key}: {merged[key]} vs {ref[key]}"

    assert merged["n_boxes"] == n_boxes
    assert merged["prod_ins_acc"] == ref["prod_ins_acc"]
    assert merged["prod_del_acc"] == ref["prod_del_acc"]
    assert merged["total_inner_steps"] == ref["total_inner_steps"]
    assert merged["box_N_history"] == ref["box_N_history"]
    assert len(merged["final_states"]) == n_boxes
