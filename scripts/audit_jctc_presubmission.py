#!/usr/bin/env python3
"""Deterministic post-hoc convergence audit and trace packager for JCTC pre-submission ablation.

Recomputes every §2.3 metric and convergence gate from raw archived traces (.npz) and run JSONs,
determines empirical time-to-convergence (two consecutive passing 5-min checkpoints, no extrapolation),
evaluates 2x2 factorial contrasts (2B fitting & REMC), and emits:
  - build/jctc_presubmission/matched_budget_summary.json
  - build/jctc_presubmission/matched_budget_traces.npz
  - build/jctc_presubmission/matched_budget_audit.md
"""

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from mcmc_diagnostics import compute_vehtari_mcmc_stats, evaluate_v2_convergence

GROUPS_ORDER = ["A", "B", "C", "D"]
PRIMARY_P_BAR = 0.28
N_MAX = 128


def _med_range(vals: List[Optional[float]]) -> Dict[str, Any]:
    clean = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not clean:
        return {"n": 0, "median": None, "min": None, "max": None, "values": []}
    return {
        "n": len(clean),
        "median": float(np.median(clean)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
        "values": clean,
    }


def _strict_gate_breakdown(diag: dict, oms_mean: float) -> Dict[str, Any]:
    n_st = diag.get("N_stats") or {}
    u_st = diag.get("U_stats") or {}
    o_st = diag.get("OMS_stats") or {}
    mcse_v = diag.get("mcse_mmol_g")
    mcse_t = diag.get("mcse_target_mmol_g")
    drift_v = diag.get("max_abs_chain_drift_mmol_g")
    drift_lim = max(0.25, 3.0 * float(mcse_v)) if mcse_v is not None else 0.25

    gates = {
        "N_rhat_lt_1.01": bool(n_st.get("estimable") and n_st.get("rhat_max") is not None and n_st["rhat_max"] < 1.01),
        "U_rhat_lt_1.01": bool(u_st.get("estimable") and u_st.get("rhat_max") is not None and u_st["rhat_max"] < 1.01),
        "OMS_rhat_lt_1.01": (
            True
            if oms_mean < 0.25
            else bool(o_st.get("estimable") and o_st.get("rhat_max") is not None and o_st["rhat_max"] < 1.01)
        ),
        "N_ess_bulk_ge_400": bool(n_st.get("estimable") and (n_st.get("ess_bulk") or 0.0) >= 400.0),
        "N_ess_tail_ge_400": bool(n_st.get("estimable") and (n_st.get("ess_tail") or 0.0) >= 400.0),
        "U_ess_bulk_ge_400": bool(u_st.get("estimable") and (u_st.get("ess_bulk") or 0.0) >= 400.0),
        "U_ess_tail_ge_400": bool(u_st.get("estimable") and (u_st.get("ess_tail") or 0.0) >= 400.0),
        "OMS_ess_bulk_ge_400": (
            True if oms_mean < 0.25 else bool(o_st.get("estimable") and (o_st.get("ess_bulk") or 0.0) >= 400.0)
        ),
        "mcse_within_target": bool(mcse_v is not None and mcse_t is not None and mcse_v <= mcse_t),
        "drift_within_limit": bool(drift_v is not None and drift_v <= drift_lim),
        "outer_acc_ge_5pct": bool((diag.get("prod_outer_acc_rate") or 0.0) >= 0.05),
    }
    return {
        "gates": gates,
        "all_gates_passed": bool(all(gates.values()) and diag.get("converged", False)),
        "drift_limit_mmol_g": float(drift_lim),
    }


def _recompute_prefix_from_traces(
    n_arr: np.ndarray,
    u_arr: np.ndarray,
    oms_arr: Optional[np.ndarray],
    acc_arr: np.ndarray,
    prop_dN_arr: np.ndarray,
    dN_arr: np.ndarray,
    conv_mmol_g: float,
    T_K: float,
    n_prefix: int,
) -> Dict[str, Any]:
    """Deterministically evaluate §2.3 convergence diagnostics on raw prefix [:n_prefix]."""
    n_sub = n_arr[:n_prefix]
    u_sub = u_arr[:n_prefix]
    oms_sub = oms_arr[:n_prefix] if oms_arr is not None and len(oms_arr) >= n_prefix else None
    acc_sub = acc_arr[:n_prefix]
    pdn_sub = prop_dN_arr[:n_prefix]
    dn_sub = dN_arr[:n_prefix]

    n_cyc, n_b = n_sub.shape
    box_up = np.mean(n_sub, axis=0) * conv_mmol_g
    sem_up = float(np.std(box_up, ddof=1) / math.sqrt(n_b)) if n_b > 1 else 0.0
    ins_att = int(np.sum(pdn_sub > 0))
    ins_acc = int(np.sum(dn_sub > 0))
    del_att = int(np.sum(pdn_sub < 0))
    del_acc = int(np.sum(dn_sub < 0))
    outer_acc = float(np.mean(acc_sub)) if acc_sub.size else 0.0

    diag = evaluate_v2_convergence(
        prod_N_history=n_sub.tolist(),
        prod_U_history_kJ=u_sub.tolist(),
        conv_mmol_g=conv_mmol_g,
        T_K=T_K,
        prod_ins_acc=ins_acc,
        prod_del_acc=del_acc,
        prod_ins_att=ins_att,
        prod_del_att=del_att,
        prod_outer_acc_rate=outer_acc,
        uptake_sem_mmol_g=sem_up,
        oms_occ_history=oms_sub.tolist() if oms_sub is not None else None,
    )
    oms_mean = float(np.mean(oms_sub)) if oms_sub is not None and oms_sub.size else 0.0
    gate_info = _strict_gate_breakdown(diag, oms_mean)
    return {"diag": diag, "oms_mean": oms_mean, **gate_info}


def _histogram_int(arr: np.ndarray) -> Dict[str, Any]:
    if arr.size == 0:
        return {"values": [], "counts": [], "probs": []}
    vals, cnts = np.unique(arr.astype(np.int64), return_counts=True)
    tot = float(np.sum(cnts))
    return {
        "values": [int(v) for v in vals],
        "counts": [int(c) for c in cnts],
        "probs": [float(c) / tot for c in cnts],
    }


def _branch_overlap(arr_2d: np.ndarray, start_frac: float = 0.5) -> Dict[str, Any]:
    """Compare empty-start ladders (0, 1) vs high-loading ladders (2, 3) over the second half."""
    n_cyc, n_b = arr_2d.shape
    if n_b < 4 or n_cyc < 4:
        return {"comparable": False}
    i0 = int(round(start_frac * n_cyc))
    sub = arr_2d[i0:]
    emp = sub[:, [0, 1]].reshape(-1)
    hl = sub[:, [2, 3]].reshape(-1)
    emp_min, emp_max = float(np.min(emp)), float(np.max(emp))
    hl_min, hl_max = float(np.min(hl)), float(np.max(hl))
    emp_q = np.quantile(emp, [0.1, 0.5, 0.9]).tolist()
    hl_q = np.quantile(hl, [0.1, 0.5, 0.9]).tolist()
    range_overlap = (min(emp_max, hl_max) >= max(emp_min, hl_min))
    q10_q90_overlap = (min(emp_q[2], hl_q[2]) >= max(emp_q[0], hl_q[0]))
    return {
        "comparable": True,
        "empty_half2_mean": float(np.mean(emp)),
        "high_half2_mean": float(np.mean(hl)),
        "empty_half2_min_max": [emp_min, emp_max],
        "high_half2_min_max": [hl_min, hl_max],
        "empty_half2_q10_q50_q90": [float(x) for x in emp_q],
        "high_half2_q10_q50_q90": [float(x) for x in hl_q],
        "range_overlaps": bool(range_overlap),
        "q10_q90_overlaps": bool(q10_q90_overlap),
    }


def audit_single_run(run_dir: Path) -> Dict[str, Any]:
    run_res = json.loads((run_dir / "run_result.json").read_text(encoding="utf-8"))
    npz = np.load(run_dir / "run_traces.npz")
    ckpt_file = run_dir / "checkpoints.json"
    ckpts_raw = json.loads(ckpt_file.read_text(encoding="utf-8")).get("checkpoints", []) if ckpt_file.exists() else []

    rsum = run_res.get("remc_run_summary") or {}
    pts = sorted(run_res["points"], key=lambda p: float(p["P_bar"]))
    pressures = [float(p["P_bar"]) for p in pts]
    n_gpus = int(rsum.get("n_gpus", pts[0].get("remc_n_gpus", 8)))
    sampling_wall_s = float(rsum.get("sampling_wall_s", pts[0].get("remc_isotherm_wall_s", 0.0)))
    total_gpu_s = sampling_wall_s * float(n_gpus)

    # Locate 0.28 bar point and recompute from raw NPZ
    p028_pt = min(pts, key=lambda pt: abs(float(pt["P_bar"]) - PRIMARY_P_BAR))
    assert abs(float(p028_pt["P_bar"]) - PRIMARY_P_BAR) < 1e-6, f"Missing 0.28 bar in {run_dir}"
    p_tag = f"{float(p028_pt['P_bar']):g}".replace(".", "p")

    n_arr = np.asarray(npz[f"p_{p_tag}_N"], dtype=np.float64)
    u_arr = np.asarray(npz[f"p_{p_tag}_U_tgt_kJ"], dtype=np.float64)
    oms_arr = np.asarray(npz[f"p_{p_tag}_oms_occ"], dtype=np.int32)
    acc_arr = np.asarray(npz[f"p_{p_tag}_accepted"], dtype=np.int8)
    dN_arr = np.asarray(npz[f"p_{p_tag}_dN"], dtype=np.int32)
    prop_dN_arr = np.asarray(npz[f"p_{p_tag}_proposed_dN"], dtype=np.int32)
    box_n_all = np.asarray(npz[f"p_{p_tag}_box_N_all"], dtype=np.int32)

    conv_mmol_g = 1000.0 / float(p028_pt["fw_mass_gmol"])
    T_K = float(p028_pt["T_K"])
    n_prod = int(n_arr.shape[0])
    recomp = _recompute_prefix_from_traces(
        n_arr, u_arr, oms_arr, acc_arr, prop_dN_arr, dN_arr, conv_mmol_g, T_K, n_prod,
    )
    diag = recomp["diag"]
    n_st = diag.get("N_stats") or {}
    u_st = diag.get("U_stats") or {}
    o_st = diag.get("OMS_stats") or {}

    # Effective state-changing acceptance rate
    ab = p028_pt.get("acceptance_breakdown") or {}
    eff_acc_rate = float(ab.get("effective_move_acc_rate", p028_pt.get("prod_outer_acc_rate", 0.0)))
    ess_bulk_N = float(n_st.get("ess_bulk", 0.0) or 0.0)
    total_target_evals_ladder = int(sum(int(pt.get("target_evals", 0)) for pt in pts))
    target_evals_028 = int(p028_pt.get("target_evals", 0))

    # Recompute/verify checkpoints from raw traces and compute empirical time-to-convergence
    recomputed_ckpts = []
    for ck in ckpts_raw:
        k_done = int(ck.get("primary_prod_cycles_done", ck.get("prod_cycles_done", 0)))
        row = {
            "wall_s": float(ck["wall_s"]),
            "total_ladder_gpu_s": float(ck["wall_s"]) * float(n_gpus),
            "completed_cycles": int(ck.get("completed_cycles", 0)),
            "prod_cycles_done": int(k_done),
            "total_target_evals": int(ck.get("total_target_evals", 0)),
            "is_final_endpoint": bool(ck.get("is_final_endpoint", False)),
            "converged": False,
            "status": "insufficient",
            "reasons": [],
            "N_ess_bulk": 0.0,
            "N_rhat_max": None,
            "mcse_mmol_g": None,
            "OMS_ess_bulk": None,
        }
        if 8 <= k_done <= n_prod:
            rc_k = _recompute_prefix_from_traces(
                n_arr, u_arr, oms_arr, acc_arr, prop_dN_arr, dN_arr, conv_mmol_g, T_K, k_done,
            )
            dk = rc_k["diag"]
            row.update({
                "converged": bool(rc_k["all_gates_passed"]),
                "status": dk["status"],
                "reasons": list(dk.get("reasons", [])),
                "N_ess_bulk": float((dk.get("N_stats") or {}).get("ess_bulk", 0.0) or 0.0),
                "N_rhat_max": (dk.get("N_stats") or {}).get("rhat_max"),
                "U_ess_bulk": (dk.get("U_stats") or {}).get("ess_bulk"),
                "mcse_mmol_g": dk.get("mcse_mmol_g"),
                "OMS_ess_bulk": (dk.get("OMS_stats") or {}).get("ess_bulk"),
                "ess_bulk_N_per_total_gpu_s": float((dk.get("N_stats") or {}).get("ess_bulk", 0.0) or 0.0) / max(1e-9, row["total_ladder_gpu_s"]),
            })
        recomputed_ckpts.append(row)

    # First passing checkpoint confirmed by the next consecutive checkpoint (no extrapolation!)
    t_conv_s = None
    t_confirm_s = None
    t_conv_gpu_s = None
    right_censored = True
    unconfirmed_terminal = False
    for i in range(len(recomputed_ckpts) - 1):
        if recomputed_ckpts[i]["converged"] and recomputed_ckpts[i + 1]["converged"]:
            t_conv_s = recomputed_ckpts[i]["wall_s"]
            t_confirm_s = recomputed_ckpts[i + 1]["wall_s"]
            t_conv_gpu_s = recomputed_ckpts[i]["total_ladder_gpu_s"]
            right_censored = False
            break
    if right_censored and recomputed_ckpts and recomputed_ckpts[-1]["converged"]:
        unconfirmed_terminal = True

    # Adjacent swaps (enforce 0 in OFF groups!)
    swap_pairs = []
    for m in range(len(pts) - 1):
        pt_m = pts[m]
        att_p = int(pt_m.get("remc_swap_att_up", 0))
        acc_p = int(pt_m.get("remc_swap_acc_up", 0))
        att_t = int(pt_m.get("remc_total_swap_att_up", 0))
        acc_t = int(pt_m.get("remc_total_swap_acc_up", 0))
        swap_pairs.append({
            "pair": f"{pts[m]['P_bar']:g}<->{pts[m+1]['P_bar']:g}",
            "prod_att": att_p,
            "prod_acc": acc_p,
            "prod_rate": (float(acc_p) / att_p) if att_p > 0 else 0.0,
            "tot_att": att_t,
            "tot_acc": acc_t,
            "tot_rate": (float(acc_t) / att_t) if att_t > 0 else 0.0,
        })
    if not run_res["remc_enable_swaps"]:
        for sp in swap_pairs:
            if sp["tot_att"] != 0 or sp["prod_att"] != 0:
                raise AssertionError(f"{run_dir}: REMC OFF group recorded non-zero swaps: {sp}")

    # Re-evaluate all 8 pressures for full coverage audit
    isotherm_all = []
    for pt in pts:
        pt_tag = f"{float(pt['P_bar']):g}".replace(".", "p")
        pn = np.asarray(npz[f"p_{pt_tag}_N"], dtype=np.float64)
        pu = np.asarray(npz[f"p_{pt_tag}_U_tgt_kJ"], dtype=np.float64)
        po = np.asarray(npz[f"p_{pt_tag}_oms_occ"], dtype=np.int32)
        pa = np.asarray(npz[f"p_{pt_tag}_accepted"], dtype=np.int8)
        ppdn = np.asarray(npz[f"p_{pt_tag}_proposed_dN"], dtype=np.int32)
        pdn = np.asarray(npz[f"p_{pt_tag}_dN"], dtype=np.int32)
        pr = _recompute_prefix_from_traces(pn, pu, po, pa, ppdn, pdn, conv_mmol_g, T_K, int(pn.shape[0]))
        pd = pr["diag"]
        p_n_st = pd.get("N_stats") or {}
        isotherm_all.append({
            "P_bar": float(pt["P_bar"]),
            "uptake_mmol_g": float(np.mean(pn) * conv_mmol_g),
            "uptake_sem_mmol_g": float(np.std(np.mean(pn, axis=0) * conv_mmol_g, ddof=1) / math.sqrt(pn.shape[1])),
            "oms_mean": float(np.mean(po)) if po.size else 0.0,
            "N_rhat_max": p_n_st.get("rhat_max"),
            "N_ess_bulk": p_n_st.get("ess_bulk"),
            "N_ess_tail": p_n_st.get("ess_tail"),
            "mcse_mmol_g": pd.get("mcse_mmol_g"),
            "qst_kJ_mol": pd.get("qst_fluctuation_kJ_mol"),
            "qst_sem_kJ_mol": pd.get("qst_fluctuation_sem_kJ_mol"),
            "qst_status": pd.get("qst_status"),
            "converged": bool(pr["all_gates_passed"]),
            "status": pd["status"],
            "reasons": list(pd.get("reasons", [])),
        })

    per_chain_N_mean = np.mean(n_arr, axis=0).tolist()
    per_chain_up_mean = (np.mean(n_arr, axis=0) * conv_mmol_g).tolist()
    per_chain_oms_mean = np.mean(oms_arr, axis=0).tolist() if oms_arr.size else [0.0] * 4
    max_N_obs = int(np.max(box_n_all)) if box_n_all.size else int(np.max(n_arr))

    return {
        "run_id": run_res["run_id"],
        "group": run_res["group"],
        "group_label": run_res["group_label"],
        "only_1b": bool(run_res["only_1b"]),
        "remc_enable_swaps": bool(run_res["remc_enable_swaps"]),
        "replicate_seed": int(run_res["replicate_seed"]),
        "git_commit": (run_res.get("git") or {}).get("commit"),
        "model_sha256": run_res.get("model_sha256"),
        "distill_sha256": run_res.get("distill_sha256"),
        "cache_npz_sha256": run_res.get("cache_npz_sha256"),
        "init_states_sha256": run_res.get("init_states_sha256"),
        "shifts_kJ_mol": run_res.get("shifts_kJ_mol"),
        "gpu_health_before": run_res.get("gpu_health_before"),
        "timing": {
            "distill_wall_s": float(run_res.get("distill_wall_s", 0.0)),
            "calib_wall_s": float(run_res.get("calib_wall_s", 0.0)),
            "prep_wall_s": float(rsum.get("prep_wall_s", 0.0)),
            "sampling_wall_s": sampling_wall_s,
            "write_wall_s": float(run_res.get("write_wall_s", 0.0)),
            "finalize_wall_s": float(rsum.get("finalize_wall_s", 0.0)),
            "diag_overhead_s": float(rsum.get("diag_overhead_s", 0.0)),
            "total_wall_s": float(run_res.get("total_wall_s", 0.0)),
            "n_gpus": n_gpus,
            "total_ladder_gpu_s": total_gpu_s,
            "amortized_028_gpu_s": total_gpu_s / float(len(pts)),
            "stop_reason": rsum.get("stop_reason", p028_pt.get("remc_stop_reason")),
            "completed_cycles": int(rsum.get("completed_cycles", p028_pt.get("remc_completed_cycles", 0))),
            "prod_cycles_done": n_prod,
        },
        "eval_counts": {
            "total_target_evals_ladder": total_target_evals_ladder,
            "target_evals_028bar": target_evals_028,
        },
        "primary_028bar": {
            "P_bar": float(p028_pt["P_bar"]),
            "init_N_per_box": p028_pt.get("init_state_summary", {}).get("init_N_per_box"),
            "mean_N": float(np.mean(n_arr)),
            "uptake_mmol_g": float(np.mean(n_arr) * conv_mmol_g),
            "uptake_sem_mmol_g": float(np.std(np.mean(n_arr, axis=0) * conv_mmol_g, ddof=1) / math.sqrt(n_arr.shape[1])),
            "oms_mean": float(np.mean(oms_arr)) if oms_arr.size else 0.0,
            "N_stats": n_st,
            "U_stats": u_st,
            "OMS_stats": o_st,
            "mcse_eff_mmol_g": diag.get("mcse_mmol_g"),
            "mcse_target_mmol_g": diag.get("mcse_target_mmol_g"),
            "max_abs_chain_drift_mmol_g": diag.get("max_abs_chain_drift_mmol_g"),
            "per_chain_signed_drift_mmol_g": diag.get("per_chain_signed_drift_mmol_g"),
            "drift_limit_mmol_g": recomp["drift_limit_mmol_g"],
            "prod_ins_acc": int(np.sum(dN_arr > 0)),
            "prod_ins_att": int(np.sum(prop_dN_arr > 0)),
            "prod_del_acc": int(np.sum(dN_arr < 0)),
            "prod_del_att": int(np.sum(prop_dN_arr < 0)),
            "prod_outer_acc_rate": float(diag.get("prod_outer_acc_rate", 0.0)),
            "effective_move_acc_rate": eff_acc_rate,
            "qst_fluctuation_kJ_mol": diag.get("qst_fluctuation_kJ_mol"),
            "qst_fluctuation_sem_kJ_mol": diag.get("qst_fluctuation_sem_kJ_mol"),
            "qst_status": diag.get("qst_status"),
            "status": diag["status"],
            "converged": bool(recomp["all_gates_passed"]),
            "gates": recomp["gates"],
            "reasons": list(diag.get("reasons", [])),
            "per_chain_N_mean": per_chain_N_mean,
            "per_chain_uptake_mmol_g": per_chain_up_mean,
            "per_chain_oms_mean": per_chain_oms_mean,
            "N_branch_overlap": _branch_overlap(n_arr),
            "OMS_branch_overlap": _branch_overlap(oms_arr) if oms_arr.size else {"comparable": False},
            "N_hist": _histogram_int(n_arr),
            "OMS_hist": _histogram_int(oms_arr),
            "max_N_observed": max_N_obs,
            "max_N_frac_of_Nmax": float(max_N_obs) / float(N_MAX),
            "ess_bulk_N_per_total_gpu_s": (ess_bulk_N / total_gpu_s) if total_gpu_s > 0.0 else 0.0,
            "ess_bulk_N_per_amortized_028_gpu_s": (ess_bulk_N / (total_gpu_s / len(pts))) if total_gpu_s > 0.0 else 0.0,
            "ess_bulk_N_per_10k_target_evals_028": (ess_bulk_N / (target_evals_028 / 1e4)) if target_evals_028 > 0 else 0.0,
        },
        "empirical_convergence": {
            "right_censored": bool(right_censored),
            "unconfirmed_terminal_pass": bool(unconfirmed_terminal),
            "time_to_convergence_s": t_conv_s,
            "confirmation_time_s": t_confirm_s,
            "time_to_convergence_gpu_s": t_conv_gpu_s,
            "checkpoints": recomputed_ckpts,
        },
        "ladder_swaps": {
            "remc_enable_swaps": bool(run_res["remc_enable_swaps"]),
            "pairs": swap_pairs,
            "total_prod_att": int(sum(sp["prod_att"] for sp in swap_pairs)),
            "total_prod_acc": int(sum(sp["prod_acc"] for sp in swap_pairs)),
            "round_trips_total": rsum.get("round_trips_total", []),
            "round_trips_prod": rsum.get("round_trips_prod", []),
            "sum_round_trips_prod": int(sum(rsum.get("round_trips_prod", []) or [])),
            "up_traversals_prod": rsum.get("up_traversals_prod", []),
            "down_traversals_prod": rsum.get("down_traversals_prod", []),
        },
        "isotherm_all_pressures": isotherm_all,
        "_raw": {
            "N_028": n_arr.astype(np.int32),
            "U_tgt_kJ_028": u_arr,
            "oms_occ_028": oms_arr.astype(np.int32),
            "box_N_all_028": box_n_all.astype(np.int32),
        },
    }


def _safe_ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or not math.isfinite(num) or not math.isfinite(den) or abs(den) < 1e-15:
        return None
    return float(num) / float(den)


def main():
    p = argparse.ArgumentParser(description="Deterministic audit for JCTC pre-submission matched-budget ablation")
    p.add_argument("--root", default=str(ROOT / "build" / "jctc_presubmission"))
    p.add_argument("--out-dir", default="", help="Optional output directory (defaults to --root)")
    args = p.parse_args()

    out_root = Path(args.root)
    write_root = Path(args.out_dir) if args.out_dir else out_root
    write_root.mkdir(parents=True, exist_ok=True)
    proto = json.loads((out_root / "protocol.json").read_text(encoding="utf-8"))
    runs_dir = out_root / "runs"

    audited_runs = []
    traces_npz = {}
    for rep_seed in proto["sampling"]["replicate_seeds"]:
        for grp in GROUPS_ORDER:
            rdir = runs_dir / f"rep_{rep_seed}_{grp}"
            if not (rdir / "run_result.json").exists():
                print(f"[Audit WARN] Missing run: {rdir}")
                continue
            r = audit_single_run(rdir)
            raw = r.pop("_raw")
            rid = r["run_id"]
            traces_npz[f"{rid}_N_028"] = raw["N_028"]
            traces_npz[f"{rid}_U_tgt_kJ_028"] = raw["U_tgt_kJ_028"]
            traces_npz[f"{rid}_oms_occ_028"] = raw["oms_occ_028"]
            traces_npz[f"{rid}_box_N_all_028"] = raw["box_N_all_028"]
            ck = r["empirical_convergence"]["checkpoints"]
            traces_npz[f"{rid}_ckpt_wall_s"] = np.array([c["wall_s"] for c in ck], dtype=np.float64)
            traces_npz[f"{rid}_ckpt_gpu_s"] = np.array([c["total_ladder_gpu_s"] for c in ck], dtype=np.float64)
            traces_npz[f"{rid}_ckpt_ess_bulk_N"] = np.array([c["N_ess_bulk"] for c in ck], dtype=np.float64)
            traces_npz[f"{rid}_ckpt_converged"] = np.array([1 if c["converged"] else 0 for c in ck], dtype=np.int8)
            traces_npz[f"{rid}_iso_P_bar"] = np.array([x["P_bar"] for x in r["isotherm_all_pressures"]], dtype=np.float64)
            traces_npz[f"{rid}_iso_uptake"] = np.array([x["uptake_mmol_g"] for x in r["isotherm_all_pressures"]], dtype=np.float64)
            traces_npz[f"{rid}_swap_rates"] = np.array([x["prod_rate"] for x in r["ladder_swaps"]["pairs"]], dtype=np.float64)
            audited_runs.append(r)

    group_summaries = {}
    for grp in GROUPS_ORDER:
        g_runs = [r for r in audited_runs if r["group"] == grp]
        n_rep = len(g_runs)
        n_pass_final = sum(1 for r in g_runs if r["primary_028bar"]["converged"])
        n_pass_conf = sum(1 for r in g_runs if not r["empirical_convergence"]["right_censored"])
        group_summaries[grp] = {
            "group": grp,
            "label": proto["groups"][grp]["label"],
            "only_1b": proto["groups"][grp]["only_1b"],
            "remc_enable_swaps": proto["groups"][grp]["remc_enable_swaps"],
            "n_replicates": n_rep,
            "n_passed_final": n_pass_final,
            "n_passed_confirmed_empirical": n_pass_conf,
            "ess_bulk_N_per_total_gpu_s": _med_range([r["primary_028bar"]["ess_bulk_N_per_total_gpu_s"] for r in g_runs]),
            "ess_bulk_N": _med_range([r["primary_028bar"]["N_stats"].get("ess_bulk") for r in g_runs]),
            "ess_tail_N": _med_range([r["primary_028bar"]["N_stats"].get("ess_tail") for r in g_runs]),
            "ess_bulk_U": _med_range([r["primary_028bar"]["U_stats"].get("ess_bulk") for r in g_runs]),
            "ess_tail_U": _med_range([r["primary_028bar"]["U_stats"].get("ess_tail") for r in g_runs]),
            "ess_bulk_OMS": _med_range([(r["primary_028bar"].get("OMS_stats") or {}).get("ess_bulk") for r in g_runs]),
            "ess_tail_OMS": _med_range([(r["primary_028bar"].get("OMS_stats") or {}).get("ess_tail") for r in g_runs]),
            "rhat_N": _med_range([r["primary_028bar"]["N_stats"].get("rhat_max") for r in g_runs]),
            "rhat_U": _med_range([r["primary_028bar"]["U_stats"].get("rhat_max") for r in g_runs]),
            "rhat_OMS": _med_range([(r["primary_028bar"].get("OMS_stats") or {}).get("rhat_max") for r in g_runs]),
            "mcse_eff_mmol_g": _med_range([r["primary_028bar"]["mcse_eff_mmol_g"] for r in g_runs]),
            "max_abs_chain_drift_mmol_g": _med_range([r["primary_028bar"]["max_abs_chain_drift_mmol_g"] for r in g_runs]),
            "uptake_mmol_g": _med_range([r["primary_028bar"]["uptake_mmol_g"] for r in g_runs]),
            "oms_mean": _med_range([r["primary_028bar"]["oms_mean"] for r in g_runs]),
            "qst_kJ_mol": _med_range([r["primary_028bar"]["qst_fluctuation_kJ_mol"] for r in g_runs]),
            "prod_outer_acc_rate": _med_range([r["primary_028bar"]["prod_outer_acc_rate"] for r in g_runs]),
            "effective_move_acc_rate": _med_range([r["primary_028bar"]["effective_move_acc_rate"] for r in g_runs]),
            "round_trips_prod_sum": _med_range([r["ladder_swaps"]["sum_round_trips_prod"] for r in g_runs]),
            "sampling_wall_s": _med_range([r["timing"]["sampling_wall_s"] for r in g_runs]),
            "total_ladder_gpu_s": _med_range([r["timing"]["total_ladder_gpu_s"] for r in g_runs]),
            "time_to_convergence_s": _med_range([r["empirical_convergence"]["time_to_convergence_s"] for r in g_runs]),
            "time_to_convergence_gpu_s": _med_range([r["empirical_convergence"]["time_to_convergence_gpu_s"] for r in g_runs]),
        }

    eff_med = {g: group_summaries[g]["ess_bulk_N_per_total_gpu_s"]["median"] for g in GROUPS_ORDER}
    ess_med = {g: group_summaries[g]["ess_bulk_N"]["median"] for g in GROUPS_ORDER}
    contrasts = {
        "B_over_A_2B_effect_REMC_OFF": {
            "eff_ratio": _safe_ratio(eff_med["B"], eff_med["A"]),
            "ess_ratio": _safe_ratio(ess_med["B"], ess_med["A"]),
        },
        "D_over_C_2B_effect_REMC_ON": {
            "eff_ratio": _safe_ratio(eff_med["D"], eff_med["C"]),
            "ess_ratio": _safe_ratio(ess_med["D"], ess_med["C"]),
        },
        "C_over_A_REMC_effect_1B_only": {
            "eff_ratio": _safe_ratio(eff_med["C"], eff_med["A"]),
            "ess_ratio": _safe_ratio(ess_med["C"], ess_med["A"]),
        },
        "D_over_B_REMC_effect_1B_2B": {
            "eff_ratio": _safe_ratio(eff_med["D"], eff_med["B"]),
            "ess_ratio": _safe_ratio(ess_med["D"], ess_med["B"]),
        },
        "D_over_A_combined": {
            "eff_ratio": _safe_ratio(eff_med["D"], eff_med["A"]),
            "ess_ratio": _safe_ratio(ess_med["D"], ess_med["A"]),
        },
    }

    summary = {
        "protocol_version": proto["protocol_version"],
        "system": proto["system"],
        "surrogates": proto["surrogates"],
        "runs": audited_runs,
        "group_summaries": group_summaries,
        "factorial_contrasts": contrasts,
    }

    sum_json = write_root / "matched_budget_summary.json"
    npz_out = write_root / "matched_budget_traces.npz"
    md_out = write_root / "matched_budget_audit.md"
    sum_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(npz_out, **traces_npz)

    lines = [
        "# JCTC Pre-Submission Matched-Budget 2x2 Ablation Audit (Mg-MOF-74 / CO2, 298 K, 0.28 bar)",
        "",
        f"- Total archived runs: **{len(audited_runs)}** (expected 12)",
        f"- Model: `{proto['system']['model']}` (SHA-256 `{proto['system']['verified_model_sha256']}`), D3 `{proto['system']['dispersion']}`, `N_max={N_MAX}`, FP32+FP64.",
        f"- Shared 1B grid bit-identical: `{proto['surrogates']['shared_1b_grid_bit_exact']}`.",
        "",
        "## Group Medians & Convergence Pass Counts (0.28 bar)",
        "",
        "| Group | Surrogate | REMC | Confirmed Pass | Final Pass | Uptake (mmol/g) | MCSE_eff | ESS_b(N) | ESS_t(N) | ESS_b(U) | ESS_b(OMS) | ESS_t(OMS) | Rhat(N) | ESS_b(N)/GPU-s | t_conv (s) | RT_prod |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    def _fmt(d, f=".2f"):
        return "N/A" if d["median"] is None else f"{d['median']:{f}} [{d['min']:{f}}, {d['max']:{f}}]"

    for grp in GROUPS_ORDER:
        gs = group_summaries[grp]
        tc = gs["time_to_convergence_s"]
        tc_str = "censored (>budget)" if tc["n"] == 0 else f"{tc['median']:.0f} [{tc['min']:.0f}, {tc['max']:.0f}] ({tc['n']}/{gs['n_replicates']})"
        lines.append(
            f"| {grp} | {'1B-only' if gs['only_1b'] else '1B+2B'} | {'ON' if gs['remc_enable_swaps'] else 'OFF'} | "
            f"{gs['n_passed_confirmed_empirical']}/{gs['n_replicates']} | {gs['n_passed_final']}/{gs['n_replicates']} | "
            f"{_fmt(gs['uptake_mmol_g'])} | {_fmt(gs['mcse_eff_mmol_g'], '.3f')} | {_fmt(gs['ess_bulk_N'], '.1f')} | "
            f"{_fmt(gs['ess_tail_N'], '.1f')} | {_fmt(gs['ess_bulk_U'], '.1f')} | {_fmt(gs['ess_bulk_OMS'], '.1f')} | "
            f"{_fmt(gs['ess_tail_OMS'], '.1f')} | {_fmt(gs['rhat_N'], '.4f')} | {_fmt(gs['ess_bulk_N_per_total_gpu_s'], '.5f')} | "
            f"{tc_str} | {_fmt(gs['round_trips_prod_sum'], '.0f')} |"
        )

    lines.extend([
        "",
        "## 2x2 Factorial Ratios (Medians)",
        "",
        *[
            f"- **{k}**: eff_ratio = {v['eff_ratio']:.3f}x, ess_ratio = {v['ess_ratio']:.3f}x"
            if v["eff_ratio"] is not None
            else f"- **{k}**: N/A"
            for k, v in contrasts.items()
        ],
        "",
        "## Per-Run Details (0.28 bar)",
        "",
        "| Run | Cyc (Prod) | Wall (s) | GPU-s | 0.28 Evals | Status | Gates | Failures / Reasons |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for r in audited_runs:
        p0 = r["primary_028bar"]
        tm = r["timing"]
        lines.append(
            f"| `{r['run_id']}` | {tm['completed_cycles']} ({tm['prod_cycles_done']}) | {tm['sampling_wall_s']:.1f} | "
            f"{tm['total_ladder_gpu_s']:.0f} | {r['eval_counts']['target_evals_028bar']} | `{p0['status']}` | "
            f"{sum(p0['gates'].values())}/{len(p0['gates'])} | {', '.join(p0['reasons']) or 'NONE (PASSED ALL GATES)'} |"
        )

    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Audit] Wrote:\n  {sum_json}\n  {npz_out}\n  {md_out}")


if __name__ == "__main__":
    main()
