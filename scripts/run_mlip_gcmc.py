"""Unified Hybrid GPU-GCMC + MLIP Nested Monte Carlo CLI Engine.

Example:
  .venv-mlip\\Scripts\\python.exe -u scripts\\run_mlip_gcmc.py ^
      --cif data\\MgMOF-74.cif --supercell 1 1 2 --temp 298 ^
      --pressures 0.002 0.01 0.05 0.15 --proposer mercado ^
      --model mace-matpes-r2scan-0 --dispersion off ^
      --boxes 4 --k-steps 25 --eq-cycles 120 --prod-cycles 320 ^
      --out-json build\\phase3_mgmof74_r2scan.json
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import time
import numpy as np

from multi_gpu import (
    parse_gpu_ids,
    resolve_worker_cuda_visible_devices,
    run_distill_on_gpu,
    run_pressures_parallel,
    run_replicas_parallel,
)
from mcmc_diagnostics import evaluate_v2_convergence
from mlip_distill import (
    compute_distill_cache_key,
    distill_mlip_surrogate,
    load_distill_cache,
    make_distilled_guest_spec,
)
from nested_mc_engine import (
    CudaGcmcProposer,
    FastMaceD3Target,
    K_TO_KJMOL,
    count_oms_occupied,
    run_nested_mc,
    verify_proposer_target_consistency,
)
from guests import load_guest
from qeq_charges import extract_framework_for_proposer
from run_metadata import collect_run_metadata, validate_resume_metadata

PROPOSER_MAP = {
    "auto_mlip": (2, 0),  # On-the-fly 1B+2B self-distilled MLIP grid + 2B LUT
    "uff": (0, 0),        # Generic UFF + REPEAT (0 charge supported) + TraPPE CO2
    "pol3site": (0, 2),   # UFF + 3-site framework E-field polarization + Thole
    "mercado": (1, 0),    # Mercado 2016 DFT-FF + EPM2 CO2
}


def parse_args():
    p = argparse.ArgumentParser(description="Hybrid GPU-GCMC + MLIP Nested Monte Carlo Engine")
    p.add_argument("--cif", required=True, help="Path to MOF CIF file")
    p.add_argument("--supercell", nargs=3, type=int, default=[1, 1, 2], help="Supercell replicas na nb nc")
    p.add_argument("--temp", type=float, default=298.0, help="Temperature in Kelvin")
    p.add_argument("--pressures", nargs="+", type=float, required=True, help="Pressures in bar")
    p.add_argument("--guest", default="co2", help="Rigid guest preset (co2, ch4, n2, h2, ar, h2o, nh3)")
    p.add_argument("--guest-json", default="", help="Optional custom rigid guest JSON path")
    p.add_argument("--proposer", choices=["auto_mlip", "uff", "pol3site", "mercado"], default="auto_mlip", help="Inner GPU proposer")
    p.add_argument("--charge-mode", choices=["auto", "cif", "qeq", "zero"], default="auto", help="Inner framework charge mode")
    p.add_argument("--model", default="medium", help="MACE foundation model name or checkpoint path")
    p.add_argument("--dispersion", choices=["on", "off"], default="on", help="Enable TorchDFTD3 D3(BJ)")
    p.add_argument("--xc", default="pbe", help="D3(BJ) functional parameter set (e.g. pbe, r2scan)")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32", help="MLIP PyTorch floating-point precision")
    p.add_argument("--device", default="cuda", help="PyTorch MLIP device")
    p.add_argument("--mace-chunk", type=int, default=None, help="Micro-batch size for MACE forward (auto by VRAM if omitted)")
    p.add_argument("--grid-dims", nargs=3, type=int, default=None, help="3D tricubic grid dims (auto from --grid-spacing if omitted)")
    p.add_argument("--grid-spacing", type=float, default=0.11, help="Target unit-cell grid spacing (Angstrom) for auto grid dims")
    p.add_argument("--boxes", type=int, default=4, help="Parallel simulation replicas")
    p.add_argument("--max-mol", type=int, default=128, help="Max guest molecules per supercell")
    p.add_argument("--k-steps", type=int, default=25, help="Inner GPU GCMC steps per outer Nested MC cycle")
    p.add_argument("--cbmc-k", type=int, default=128, help="Block-parallel GPU CBMC trial poses per inner insert/delete (1..128)")
    p.add_argument("--p-ins-del", type=float, default=0.50, help="Total insertion+deletion move fraction inside GPU proposer")
    p.add_argument("--distill-samples-1b", type=int, default=1200, help="Single-molecule MACE evaluations for auto_mlip 1B distillation")
    p.add_argument("--distill-samples-2b", type=int, default=192, help="Pore dimer MACE evaluations for auto_mlip 2B distillation")
    p.add_argument("--distill-cache-dir", default="build/distill_cache", help="Disk cache directory for distilled 1B+2B grids")
    p.add_argument("--grid-only", action="store_true", help="Run pure GPU surrogate grid GCMC without outer MLIP reweighting (ablation mode)")
    p.add_argument("--eq-cycles", type=int, default=120, help="Outer equilibration cycles")
    p.add_argument("--prod-cycles", type=int, default=320, help="Outer production cycles")
    p.add_argument("--pilot-cycles", type=int, default=40, help="Target-ensemble pilot burn-in cycles for Stage-3 shift calibration (0 disables)")
    p.add_argument("--shift-override", type=float, default=None, help="Optional fixed chem-pot shift (kJ/mol) overriding auto calibration")
    p.add_argument("--init-mode", choices=["auto", "empty", "high_loading"], default="auto", help="Initial state prepared right before target equilibration (detached from shift calibration)")
    p.add_argument("--init-loading", type=float, default=8.0, help="Target loading (mmol/g) when --init-mode high_loading")
    p.add_argument("--seed", type=int, default=20261400, help="Base PRNG seed")
    p.add_argument("--shift-mode", choices=["auto", "none"], default="auto", help="Multi-stage self-consistent chem-pot shift")
    p.add_argument("--gpus", nargs="+", default=None, help="GPU IDs for multi-GPU execution (e.g. --gpus 0 1 2 3 or 0,1,2,3)")
    p.add_argument("--parallel-mode", choices=["auto", "pressure", "replica"], default="auto", help="Multi-GPU parallel strategy")
    p.add_argument("--worker-threads", type=int, default=4, help="CPU threads per multi-GPU worker process")
    p.add_argument("--resume", action="store_true", help="Alias for --resume-mode skip when --resume-mode is not specified")
    p.add_argument("--resume-mode", choices=["none", "skip", "extend", "fresh"], default="none", help="Task/checkpoint resume policy")
    p.add_argument("--checkpoint-dir", default="", help="Optional directory for bit-exact cycle checkpoints")
    p.add_argument("--checkpoint-every", type=int, default=0, help="Save bit-exact checkpoint every N cycles (0 saves only at end)")
    p.add_argument("--snapshot-stride", type=int, default=0, help="Save production configuration snapshots every N prod cycles")
    p.add_argument("--out-json", required=True, help="Output JSON path")
    p.add_argument("--out-csv", default="", help="Optional summary CSV path")
    return p.parse_args()


def compute_grid_dims(uc_atoms, grid_spacing: float = 0.11):
    lengths = np.linalg.norm(np.array(uc_atoms.get_cell(), dtype=np.float64), axis=1)
    dims = []
    for L in lengths:
        n = int(math.ceil(float(L) / float(grid_spacing)))
        n = ((n + 7) // 8) * 8
        dims.append(max(32, min(256, n)))
    return tuple(dims)


def parse_cuda_device_id(device_str: str) -> int:
    s = str(device_str).strip().lower()
    if s.startswith("cuda:"):
        return int(s.split(":", 1)[1])
    return 0


def evaluate_convergence_diagnostics(res: dict) -> dict:
    uptake = float(res["uptake_mmol_g"])
    sem = float(res["uptake_sem_mmol_g"])
    drift = float(res["stationarity_drift_mmol_g"])
    ins_acc = int(res["prod_ins_acc"])
    del_acc = int(res["prod_del_acc"])
    outer_acc = float(res["prod_outer_acc_rate"])

    ins_acc_ok = (ins_acc >= 5) if uptake >= 1e-4 else True
    del_acc_ok = (del_acc >= 5) if uptake >= 1e-4 else True
    drift_limit = max(0.25, 3.0 * sem)
    drift_ok = bool(drift <= drift_limit)
    outer_acc_ok = bool(outer_acc >= 0.05)

    warnings = []
    if not ins_acc_ok:
        warnings.append(f"Low production insertions accepted ({ins_acc} < 5 at uptake {uptake:.3f} mmol/g)")
    if not del_acc_ok:
        warnings.append(f"Low production deletions accepted ({del_acc} < 5 at uptake {uptake:.3f} mmol/g)")
    if not drift_ok:
        warnings.append(f"High half-split drift ({drift:.3f} > max(0.25, 3*SEM)={drift_limit:.3f} mmol/g)")
    if not outer_acc_ok:
        warnings.append(f"Low production outer acceptance rate ({outer_acc*100:.1f}% < 5.0%)")

    return {
        "converged": bool(ins_acc_ok and del_acc_ok and drift_ok and outer_acc_ok),
        "ins_acc_ok": ins_acc_ok,
        "del_acc_ok": del_acc_ok,
        "drift_ok": drift_ok,
        "outer_acc_ok": outer_acc_ok,
        "warnings": warnings,
    }


def save_outputs(out_json: Path, out_csv: Path | None, payload: dict):
    payload["points"] = sorted(payload["points"], key=lambda pt: float(pt["P_bar"]))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_json = out_json.with_suffix(out_json.suffix + ".tmp")
    tmp_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_json, out_json)

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        tmp_csv = out_csv.with_suffix(out_csv.suffix + ".tmp")
        with tmp_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "P_bar", "uptake_mmol_g", "uptake_sem_mmol_g", "Qst_kJ_mol", "Qst_sem_kJ_mol",
                "qst_status", "delta_eps_kJ_mol", "outer_acc_rate", "prod_outer_acc_rate",
                "prod_ins_acc", "prod_ins_att", "prod_del_acc", "prod_del_att",
                "stationarity_drift_mmol_g", "max_abs_chain_drift_mmol_g",
                "tau_int_N", "ess_N", "ess_N_per_s", "rhat_N", "ess_bulk_N", "ess_tail_N",
                "mcse_mmol_g", "elapsed_s", "diag_status", "converged",
            ])
            for pt in payload["points"]:
                diag = pt.get("diagnostics")
                if isinstance(diag, dict) and "status" in diag:
                    diag_status = diag["status"]
                    conv = diag.get("converged", "unknown")
                    rhat_N = (diag.get("N_stats") or {}).get("rhat_max", "")
                    ess_b = (diag.get("N_stats") or {}).get("ess_bulk", "")
                    ess_t = (diag.get("N_stats") or {}).get("ess_tail", "")
                    mcse_v = diag.get("mcse_mmol_g", "")
                    max_cdrift = diag.get("max_abs_chain_drift_mmol_g", "")
                else:
                    diag_status = "unknown"
                    conv = "unknown"
                    rhat_N = ""
                    ess_b = ""
                    ess_t = ""
                    mcse_v = ""
                    max_cdrift = ""
                q_val = "" if pt.get("Qst_kJ_mol") is None else pt["Qst_kJ_mol"]
                q_sem = "" if pt.get("Qst_sem_kJ_mol") is None else pt["Qst_sem_kJ_mol"]
                w.writerow([
                    pt["P_bar"], pt["uptake_mmol_g"], pt["uptake_sem_mmol_g"],
                    q_val, q_sem, pt.get("qst_status", "unknown"),
                    pt["delta_eps_kJ_mol"], pt["outer_acc_rate"], pt.get("prod_outer_acc_rate", ""),
                    pt["prod_ins_acc"], pt["prod_ins_att"],
                    pt["prod_del_acc"], pt["prod_del_att"],
                    pt["stationarity_drift_mmol_g"], max_cdrift,
                    pt.get("tau_int_N", ""), pt.get("ess_N", ""), pt.get("ess_N_per_s", ""),
                    rhat_N if rhat_N is not None else "", ess_b, ess_t,
                    mcse_v if mcse_v is not None else "", pt["elapsed_s"], diag_status, conv,
                ])
        os.replace(tmp_csv, out_csv)


OMS_ATOMIC_NUMBERS = {12, 13, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30}


def extract_oms_metal_positions(fw_atoms) -> np.ndarray:
    """Return (n_oms, 3) Cartesian positions of open-metal atoms in `fw_atoms`."""
    z_arr = np.asarray(fw_atoms.get_atomic_numbers(), dtype=np.int32)
    pos = np.asarray(fw_atoms.get_positions(), dtype=np.float64)
    mask = np.isin(z_arr, list(OMS_ATOMIC_NUMBERS))
    return pos[mask]


def _rand_unit_quat(rng: np.random.Generator):
    u1, u2, u3 = rng.random(3)
    u2 *= 2.0 * math.pi
    u3 *= 2.0 * math.pi
    s1 = math.sqrt(max(0.0, 1.0 - u1))
    s2 = math.sqrt(max(0.0, u1))
    return (
        s2 * math.cos(u3),
        s1 * math.sin(u2),
        s1 * math.cos(u2),
        s2 * math.sin(u3),
    )


def build_legal_dispersed_high_loading_states(
    prop: CudaGcmcProposer,
    target,
    target_N: int,
    seed: int = 2026777,
    min_gg_A: float = 3.2,
    max_per_mol_energy_K: float = 1500.0,
    fw_pos: np.ndarray | None = None,
    oms_metal_pos: np.ndarray | None = None,
    guest_local_pos: np.ndarray | None = None,
    n_candidates_per_mol: int = 4,
):
    """Build spatially dispersed, non-overlapping high-loading states checked on proposer and target."""
    rng = np.random.default_rng(seed)
    H_rows = np.asarray(prop.H_cols.T, dtype=np.float64)
    inv_H = np.linalg.inv(H_rows)
    n_b = int(prop.n_boxes)
    target_N = int(max(0, min(int(prop.max_mol), target_N)))
    n_cands_req = max(1, int(n_candidates_per_mol))

    if target_N == 0:
        empty = [
            {
                "N": 0, "cx": np.zeros(0), "cy": np.zeros(0), "cz": np.zeros(0),
                "qw": np.ones(0), "qx": np.zeros(0), "qy": np.zeros(0), "qz": np.zeros(0),
            }
            for _ in range(n_b)
        ]
        return empty, {
            "target_N_per_box": 0,
            "init_N_per_box": [0] * n_b,
            "init_min_gg_A": None,
            "init_U_tgt_K": [0.0] * n_b,
            "init_U_tgt_per_mol_kJ": [0.0] * n_b,
            "init_oms_occ_per_box": [0] * n_b if oms_metal_pos is not None else None,
        }

    has_oms = oms_metal_pos is not None and len(oms_metal_pos) > 0
    oms_arr = np.asarray(oms_metal_pos, dtype=np.float64) if has_oms else None

    def _sample_pos_quat_batch(n_samp: int):
        if has_oms:
            n_half = n_samp // 2
            # Half around random OMS shells (r in 2.7..3.9 A), half uniform in supercell
            m_idx = rng.integers(0, len(oms_arr), size=n_half)
            u_dir = rng.normal(size=(n_half, 3))
            u_dir /= np.maximum(1e-12, np.linalg.norm(u_dir, axis=1, keepdims=True))
            radii = rng.uniform(2.7, 3.9, size=(n_half, 1))
            pos_oms = oms_arr[m_idx] + radii * u_dir
            pos_oms = ((pos_oms @ inv_H) % 1.0) @ H_rows
            pos_uni = rng.random((n_samp - n_half, 3)) @ H_rows
            pos = np.vstack([pos_oms, pos_uni])
        else:
            pos = rng.random((n_samp, 3)) @ H_rows
        r1 = rng.random(n_samp)
        r2 = rng.random(n_samp) * (2.0 * math.pi)
        r3 = rng.random(n_samp) * (2.0 * math.pi)
        s1 = np.sqrt(np.maximum(0.0, 1.0 - r1))
        s2 = np.sqrt(np.maximum(0.0, r1))
        quat = np.stack(
            [s2 * np.cos(r3), s1 * np.sin(r2), s1 * np.cos(r2), s2 * np.sin(r3)], axis=1
        )
        return pos, quat

    boxes = [
        {
            "cx": [], "cy": [], "cz": [],
            "qw": [], "qx": [], "qy": [], "qz": [],
            "u_ref": 0.0, "active": True,
        }
        for _ in range(n_b)
    ]

    for m in range(target_N):
        batch_cands = []
        batch_meta = []
        for b in range(n_b):
            bd = boxes[b]
            if not bd["active"]:
                continue
            prev_c = np.stack([bd["cx"], bd["cy"], bd["cz"]], axis=1) if bd["cx"] else None
            box_cands = []
            box_u_refs = []
            spacings = (float(min_gg_A), max(2.6, float(min_gg_A) - 0.4))
            for spacing_try in spacings:
                pos_p, q_p = _sample_pos_quat_batch(512)
                if prev_c is not None:
                    df = (pos_p @ inv_H)[:, None, :] - (prev_c @ inv_H)[None, :, :]
                    df -= np.round(df)
                    dr = df @ H_rows
                    d2_min = np.min(np.sum(dr * dr, axis=-1), axis=1)
                    keep = d2_min >= (spacing_try * spacing_try)
                    pos_p = pos_p[keep]
                    q_p = q_p[keep]
                if len(pos_p) == 0:
                    continue
                scored = []
                for idx_p in range(min(64, len(pos_p))):
                    c_try = pos_p[idx_p]
                    q_try = q_p[idx_p]
                    cand = {
                        "N": m + 1,
                        "cx": np.asarray(bd["cx"] + [float(c_try[0])], dtype=np.float64),
                        "cy": np.asarray(bd["cy"] + [float(c_try[1])], dtype=np.float64),
                        "cz": np.asarray(bd["cz"] + [float(c_try[2])], dtype=np.float64),
                        "qw": np.asarray(bd["qw"] + [float(q_try[0])], dtype=np.float64),
                        "qx": np.asarray(bd["qx"] + [float(q_try[1])], dtype=np.float64),
                        "qy": np.asarray(bd["qy"] + [float(q_try[2])], dtype=np.float64),
                        "qz": np.asarray(bd["qz"] + [float(q_try[3])], dtype=np.float64),
                    }
                    u_ref_new, ov_ref = prop.eval_energy_quat(cand)
                    if ov_ref or not math.isfinite(float(u_ref_new)):
                        continue
                    if (float(u_ref_new) - bd["u_ref"]) > 5000.0:
                        continue
                    scored.append((float(u_ref_new), cand))
                scored.sort(key=lambda x: x[0])
                for ur, cand in scored[:n_cands_req]:
                    box_cands.append(cand)
                    box_u_refs.append(ur)
                if box_cands:
                    break
            if not box_cands:
                bd["active"] = False
            else:
                for cand, ur in zip(box_cands, box_u_refs):
                    batch_cands.append(cand)
                    batch_meta.append((b, ur))

        if not batch_cands:
            break

        if target is not None:
            if hasattr(target, "eval_batch"):
                evals = target.eval_batch(batch_cands)
            else:
                evals = [target(c) for c in batch_cands]
        else:
            evals = [(0.0, False)] * len(batch_cands)

        best_per_box = {}
        for cand, (b, ur), (ut, ov_t) in zip(batch_cands, batch_meta, evals):
            if ov_t or not math.isfinite(float(ut)):
                continue
            prev_best = best_per_box.get(b)
            if prev_best is None or float(ut) < prev_best[0]:
                best_per_box[b] = (float(ut), cand, ur)

        for b in range(n_b):
            bd = boxes[b]
            if not bd["active"]:
                continue
            chosen = best_per_box.get(b)
            if chosen is None:
                bd["active"] = False
                continue
            _, cand_best, ur_best = chosen
            bd["cx"].append(float(cand_best["cx"][-1]))
            bd["cy"].append(float(cand_best["cy"][-1]))
            bd["cz"].append(float(cand_best["cz"][-1]))
            bd["qw"].append(float(cand_best["qw"][-1]))
            bd["qx"].append(float(cand_best["qx"][-1]))
            bd["qy"].append(float(cand_best["qy"][-1]))
            bd["qz"].append(float(cand_best["qz"][-1]))
            bd["u_ref"] = ur_best

    states = []
    min_gg_list = []
    for b in range(n_b):
        bd = boxes[b]
        if not bd["cx"]:
            raise RuntimeError(f"Failed to place any molecule for high_loading init in box {b}")
        c_arr = np.stack([bd["cx"], bd["cy"], bd["cz"]], axis=1)
        if len(bd["cx"]) >= 2:
            df = (c_arr[:, None, :] - c_arr[None, :, :]) @ inv_H
            df -= np.round(df)
            dr = df @ H_rows
            dist = np.linalg.norm(dr, axis=-1)
            np.fill_diagonal(dist, np.inf)
            min_gg_list.append(float(np.min(dist)))
        states.append({
            "N": len(bd["cx"]),
            "cx": np.asarray(bd["cx"], dtype=np.float64),
            "cy": np.asarray(bd["cy"], dtype=np.float64),
            "cz": np.asarray(bd["cz"], dtype=np.float64),
            "qw": np.asarray(bd["qw"], dtype=np.float64),
            "qx": np.asarray(bd["qx"], dtype=np.float64),
            "qy": np.asarray(bd["qy"], dtype=np.float64),
            "qz": np.asarray(bd["qz"], dtype=np.float64),
        })

    if target is not None:
        evals = target.eval_batch(states) if hasattr(target, "eval_batch") else [target(s) for s in states]
        for b, (ut, ov) in enumerate(evals):
            if ov or not math.isfinite(float(ut)):
                raise RuntimeError(f"High-loading initial state in box {b} failed target check (ov={ov}, u={ut})")
            per_mol = float(ut) / max(1, int(states[b]["N"]))
            if per_mol > max_per_mol_energy_K:
                raise RuntimeError(
                    f"High-loading initial state in box {b} exceeds per-molecule energy limit: {per_mol:.1f} K > {max_per_mol_energy_K} K"
                )
        u_tgt_list = [float(ut) for ut, _ in evals]
    else:
        u_tgt_list = [0.0] * n_b

    u_per_mol_kJ = [
        float(u_tgt_list[b]) * K_TO_KJMOL / max(1, int(states[b]["N"]))
        for b in range(n_b)
    ]
    oms_occ_list = None
    if oms_metal_pos is not None and len(oms_metal_pos) > 0:
        oms_occ_list = [
            count_oms_occupied(
                states[b], oms_metal_pos, H_rows,
                guest_local_pos=guest_local_pos, inv_cell_rows=inv_H,
            )
            for b in range(n_b)
        ]

    return states, {
        "target_N_per_box": target_N,
        "init_N_per_box": [int(s["N"]) for s in states],
        "init_min_gg_A": min(min_gg_list) if min_gg_list else None,
        "init_U_tgt_K": u_tgt_list,
        "init_U_tgt_per_mol_kJ": u_per_mol_kJ,
        "init_oms_occ_per_box": oms_occ_list,
    }


def _evaluate_shift_stage(states, target_evals, low_cap: float, high_cap: float, fallback_shift: float):
    raw_per_box = []
    box_N = []
    box_u_tgt_kJ = []
    box_u_ref_kJ = []
    for st, (u_t, ov_t) in zip(states, target_evals):
        n = int(st["N"])
        box_N.append(n)
        u_t_kJ = float(u_t) * K_TO_KJMOL if not ov_t else None
        u_r_kJ = float(st["U_ref_K"]) * K_TO_KJMOL
        box_u_tgt_kJ.append(u_t_kJ)
        box_u_ref_kJ.append(u_r_kJ)
        if n > 0 and not ov_t:
            raw_per_box.append(float((float(u_t) - float(st["U_ref_K"])) * K_TO_KJMOL / n))
        else:
            raw_per_box.append(None)
    valid = [v for v in raw_per_box if v is not None]
    if valid:
        raw_mean = float(np.median(valid)) if len(valid) >= 3 else float(np.mean(valid))
        clamped = float(max(low_cap, min(high_cap, raw_mean)))
        hit_upper = bool(raw_mean > high_cap)
        hit_lower = bool(raw_mean < low_cap)
    else:
        raw_mean = None
        clamped = float(fallback_shift)
        hit_upper = False
        hit_lower = False
    info = {
        "n_active_boxes": int(len(valid)),
        "box_N": box_N,
        "raw_per_box_kJ_mol": raw_per_box,
        "raw_mean_kJ_mol": raw_mean,
        "clamped_kJ_mol": clamped,
        "low_cap_kJ_mol": float(low_cap),
        "high_cap_kJ_mol": float(high_cap),
        "hit_upper_cap": hit_upper,
        "hit_lower_cap": hit_lower,
        "box_U_tgt_kJ": box_u_tgt_kJ,
        "box_U_ref_kJ": box_u_ref_kJ,
    }
    return clamped, info


def _evaluate_shift_stage_from_trace(res_pilot: dict, low_cap: float, high_cap: float, fallback_shift: float):
    n_arr = np.asarray(res_pilot["prod_N_history"], dtype=np.float64)
    ut_arr = np.asarray(res_pilot["prod_U_history"], dtype=np.float64)
    ur_arr = np.asarray(res_pilot["prod_U_ref_history"], dtype=np.float64)
    mask = n_arr > 0.0
    per_box_means = []
    for b in range(n_arr.shape[1]):
        mb = mask[:, b]
        if np.any(mb):
            per_box_means.append(float(np.mean((ut_arr[mb, b] - ur_arr[mb, b]) / n_arr[mb, b])))
        else:
            per_box_means.append(None)
    if np.any(mask):
        all_diffs = (ut_arr[mask] - ur_arr[mask]) / n_arr[mask]
        raw_mean = float(np.mean(all_diffs))
        clamped = float(max(low_cap, min(high_cap, raw_mean)))
        hit_upper = bool(raw_mean > high_cap)
        hit_lower = bool(raw_mean < low_cap)
    else:
        raw_mean = None
        clamped = float(fallback_shift)
        hit_upper = False
        hit_lower = False
    info = {
        "n_nonzero_samples": int(np.sum(mask)),
        "n_total_samples": int(mask.size),
        "raw_per_box_kJ_mol": per_box_means,
        "raw_mean_kJ_mol": raw_mean,
        "clamped_kJ_mol": clamped,
        "low_cap_kJ_mol": float(low_cap),
        "high_cap_kJ_mol": float(high_cap),
        "hit_upper_cap": hit_upper,
        "hit_lower_cap": hit_lower,
        "end_box_N": [int(x) for x in n_arr[-1].tolist()] if len(n_arr) else [],
        "end_box_U_tgt_kJ": [float(x) for x in ut_arr[-1].tolist()] if len(ut_arr) else [],
        "end_box_U_ref_kJ": [float(x) for x in ur_arr[-1].tolist()] if len(ur_arr) else [],
    }
    return clamped, info


def _write_ckpt_json(ckpt_path: Path, payload: dict):
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, ckpt_path)


def _cif_matches(old_cif: str, new_cif: str) -> bool:
    p_old, p_new = Path(old_cif), Path(new_cif)
    try:
        if p_old.resolve() == p_new.resolve():
            return True
    except OSError:
        pass
    return p_old.name == p_new.name


def run_pressure_point(
    fw_data: dict,
    guest,
    supercell,
    T_K: float,
    P_bar: float,
    ff_mode: int,
    pol_mode: int,
    grid_dims,
    n_boxes: int,
    max_mol: int,
    k_steps: int,
    eq_cycles: int,
    prod_cycles: int,
    seed: int,
    shift_mode: str,
    target: FastMaceD3Target,
    pilot_cycles: int = 60,
    shift_override=None,
    verify_consistency: bool = False,
    device_id: int = 0,
    init_mode: str = "auto",
    init_loading: float = 8.0,
    checkpoint_dir: str = "",
    checkpoint_every: int = 0,
    checkpoint_tag: str = "",
    resume_mode: str = "none",
    snapshot_stride: int = 0,
    run_metadata: dict | None = None,
    grid_only: bool = False,
    cbmc_k: int = 128,
    p_ins_del: float = 0.50,
    distill_res: dict | None = None,
):
    t_prop0 = time.perf_counter()
    if int(ff_mode) == 2:
        if distill_res is None:
            raise ValueError("ff_mode=2 (auto_mlip) requires distill_res")
        prop = CudaGcmcProposer.from_distilled(
            fw_data=fw_data,
            guest=guest,
            distill_res=distill_res,
            supercell=tuple(supercell),
            T_K=T_K,
            pressure_bar=P_bar,
            n_boxes=n_boxes,
            max_mol=max_mol,
            seed=seed,
            device_id=device_id,
            cbmc_k=int(cbmc_k),
            p_ins_del=float(p_ins_del),
        )
    else:
        prop = CudaGcmcProposer.from_atoms(
            fw_data=fw_data,
            guest=guest,
            supercell=tuple(supercell),
            T_K=T_K,
            pressure_bar=P_bar,
            ff_mode=ff_mode,
            pol_mode=pol_mode,
            grid_dims=tuple(grid_dims),
            n_boxes=n_boxes,
            max_mol=max_mol,
            seed=seed,
            device_id=device_id,
            cbmc_k=int(cbmc_k),
            p_ins_del=float(p_ins_del),
        )
    prop_grid_wall_s = float(time.perf_counter() - t_prop0)
    if verify_consistency and target is not None:
        cons = verify_proposer_target_consistency(prop, target)
        print(
            f"  [Consistency OK] unit={cons['n_unit_atoms']} super={cons['n_super_atoms']} "
            f"max_cell_err={cons['max_cell_err_A']:.2e} A max_pos_err={cons['max_pos_err_A']:.2e} A "
            f"min_perp={cons['min_perp_width_A']:.2f} A"
        )
    conv_mmol_g = 1000.0 / prop.fw_mass_gmol

    if target is not None and getattr(target, "fw", None) is not None:
        fw_super = target.fw
    else:
        fw_super = fw_data["uc"].repeat(tuple(supercell))
    fw_pos = np.asarray(fw_super.get_positions(), dtype=np.float64)
    oms_metal_pos = extract_oms_metal_positions(fw_super)
    oms_cell_rows = np.asarray(fw_super.get_cell(), dtype=np.float64) if len(oms_metal_pos) > 0 else None
    guest_local_pos = guest.real_local_positions() if hasattr(guest, "real_local_positions") else getattr(target, "guest_local_pos", None)

    ckpt_path = None
    if checkpoint_dir:
        tag_suf = f"_{checkpoint_tag}" if checkpoint_tag else ""
        ckpt_path = Path(checkpoint_dir) / f"ckpt_P{float(P_bar):.6g}{tag_suf}.json"

    loaded_ckpt = None
    if resume_mode == "extend" and ckpt_path is not None and ckpt_path.exists():
        loaded_ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
        if run_metadata is not None and "metadata" in loaded_ckpt:
            validate_resume_metadata(loaded_ckpt["metadata"], run_metadata, allow_extend_prod=True)
        de_final = float(loaded_ckpt["delta_eps_kJ_mol"])
        prop.set_chem_pot_shift(de_final)
        shift_calib = loaded_ckpt.get("shift_calibration", {"mode": "resumed"})
        init_state_summary = loaded_ckpt.get("init_state_summary", {"init_mode": init_mode})
        shift_calib_wall_s = 0.0
        print(
            f"  [P={P_bar:.4g} bar RESUME-EXTEND] Loaded checkpoint at "
            f"{loaded_ckpt['mc_state']['completed_cycles']} cycles (delta_eps={de_final:.2f} kJ/mol)"
        )
    else:
        t_calib0 = time.perf_counter()
        eq_fast = 15000 if int(cbmc_k) >= 32 else 30000
        eq_mid = 12000 if int(cbmc_k) >= 32 else 20000
        eq_boost = 8000 if int(cbmc_k) >= 32 else 15000
        shift_calib = {"mode": "override" if shift_override is not None else str(shift_mode)}
        if grid_only:
            de_final = 0.0
            prop.set_chem_pot_shift(0.0)
            if init_mode == "auto":
                prop.equilibrate(eq_fast)
            shift_calib = {"mode": "grid_only", "final_kJ_mol": 0.0}
        elif shift_override is not None:
            de_final = float(shift_override)
            prop.set_chem_pot_shift(de_final)
            if init_mode == "auto":
                prop.equilibrate(eq_fast)
            fug_mult = math.exp(-de_final / (K_TO_KJMOL * T_K))
            shift_calib["override_kJ_mol"] = de_final
            print(f"  [P={P_bar:.4g} bar] Override Shift={de_final:.2f} kJ/mol ({fug_mult:.2f}x inner f_ref)")
        elif shift_mode == "auto":
            prop.equilibrate(eq_mid)
            cal1_st = prop.get_accepted_quat()
            boosts_tried = []
            mean_n1 = float(np.mean([s["N"] for s in cal1_st]))
            if mean_n1 < 4.0:
                for boost in (-4.0, -8.0, -12.0, -16.0):
                    boosts_tried.append(boost)
                    prop.set_chem_pot_shift(boost)
                    prop.equilibrate(eq_boost)
                    cal1_st = prop.get_accepted_quat()
                    if np.mean([s["N"] for s in cal1_st]) >= 4.0:
                        break
            elif mean_n1 > 0.85 * float(max_mol):
                for boost in (4.0, 8.0, 12.0):
                    boosts_tried.append(boost)
                    prop.set_chem_pot_shift(boost)
                    prop.equilibrate(eq_boost)
                    cal1_st = prop.get_accepted_quat()
                    if np.mean([s["N"] for s in cal1_st]) <= 0.85 * float(max_mol):
                        break
            cal1_u = target.eval_batch(cal1_st)
            de1, s1_info = _evaluate_shift_stage(cal1_st, cal1_u, -24.0, 24.0, 0.0)
            s1_info["boosts_tried_kJ_mol"] = boosts_tried
            shift_calib["stage1"] = s1_info
            prop.set_chem_pot_shift(de1)
            prop.equilibrate(eq_fast)

            cal2_st = prop.get_accepted_quat()
            cal2_u = target.eval_batch(cal2_st)
            de2, s2_info = _evaluate_shift_stage(cal2_st, cal2_u, -24.0, 24.0, de1)
            shift_calib["stage2"] = s2_info
            prop.set_chem_pot_shift(de2)
            prop.equilibrate(eq_mid)

            if pilot_cycles > 0:
                if pilot_cycles >= 20:
                    p1 = pilot_cycles // 2
                    p2 = pilot_cycles - p1
                    p1_eq = max(1, p1 // 2)
                    p1_pr = max(1, p1 - p1_eq)
                    r3a = run_nested_mc(
                        prop, None, k_steps=k_steps, eq_cycles=p1_eq, prod_cycles=p1_pr,
                        seed=seed + 5, log_every=p1 + 10, batch_target_fn=target.eval_batch,
                        oms_metal_pos=oms_metal_pos, oms_cell_rows=oms_cell_rows,
                        guest_local_pos=guest_local_pos,
                    )
                    de3a, s3a_info = _evaluate_shift_stage_from_trace(r3a, -24.0, 24.0, de2)
                    shift_calib["stage3a"] = s3a_info
                    prop.set_chem_pot_shift(de3a)
                    if abs(de3a - de2) > 1.0:
                        prop.equilibrate(10000)
                    p2_eq = max(1, p2 // 2)
                    p2_pr = max(1, p2 - p2_eq)
                    r3b = run_nested_mc(
                        prop, None, k_steps=k_steps, eq_cycles=p2_eq, prod_cycles=p2_pr,
                        seed=seed + 9, log_every=p2 + 10, batch_target_fn=target.eval_batch,
                        oms_metal_pos=oms_metal_pos, oms_cell_rows=oms_cell_rows,
                        guest_local_pos=guest_local_pos,
                    )
                    de_final, s3b_info = _evaluate_shift_stage_from_trace(r3b, -24.0, 24.0, de3a)
                    shift_calib["stage3b"] = s3b_info
                    prop.set_chem_pot_shift(de_final)
                    if init_mode == "auto" and abs(de_final - de3a) > 1.0:
                        prop.equilibrate(10000)
                else:
                    p_eq = max(1, pilot_cycles // 2)
                    p_pr = max(1, pilot_cycles - p_eq)
                    r3 = run_nested_mc(
                        prop, None, k_steps=k_steps, eq_cycles=p_eq, prod_cycles=p_pr,
                        seed=seed + 5, log_every=pilot_cycles + 10, batch_target_fn=target.eval_batch,
                        oms_metal_pos=oms_metal_pos, oms_cell_rows=oms_cell_rows,
                        guest_local_pos=guest_local_pos,
                    )
                    de_final, s3_info = _evaluate_shift_stage_from_trace(r3, -24.0, 24.0, de2)
                    shift_calib["stage3"] = s3_info
                    prop.set_chem_pot_shift(de_final)
                    if init_mode == "auto":
                        prop.equilibrate(10000)

                fug_mult = math.exp(-de_final / (K_TO_KJMOL * T_K))
                print(
                    f"  [P={P_bar:.4g} bar] Shift Stage1={de1:.2f} -> Stage2={de2:.2f} -> "
                    f"PilotStage3({pilot_cycles}cyc)={de_final:.2f} kJ/mol ({fug_mult:.2f}x inner f_ref)"
                )
            else:
                de_final = de2
                fug_mult = math.exp(-de_final / (K_TO_KJMOL * T_K))
                print(
                    f"  [P={P_bar:.4g} bar] Shift Stage1={de1:.2f} -> Stage2={de_final:.2f} kJ/mol ({fug_mult:.2f}x inner f_ref)"
                )
        else:
            de_final = 0.0
            if init_mode == "auto":
                prop.equilibrate(30000)

        shift_calib["final_kJ_mol"] = de_final
        shift_calib_wall_s = float(time.perf_counter() - t_calib0)

        # Explicitly prepare production initial state AFTER shift calibration (without classical re-equilibration on empty/high_loading)
        if init_mode == "empty":
            empty_states = [
                {
                    "N": 0, "cx": np.zeros(0), "cy": np.zeros(0), "cz": np.zeros(0),
                    "qw": np.ones(0), "qx": np.zeros(0), "qy": np.zeros(0), "qz": np.zeros(0),
                }
                for _ in range(n_boxes)
            ]
            prop.set_accepted_quat(empty_states)
            init_state_summary = {
                "init_mode": "empty",
                "init_N_per_box": [0] * n_boxes,
                "init_min_gg_A": None,
                "init_U_tgt_K": [0.0] * n_boxes,
            }
        elif init_mode == "high_loading":
            target_N = int(round(float(init_loading) / conv_mmol_g))
            hl_states, init_state_summary = build_legal_dispersed_high_loading_states(
                prop, target, target_N=target_N, seed=seed + 777,
                fw_pos=fw_pos, oms_metal_pos=oms_metal_pos, guest_local_pos=guest_local_pos,
            )
            init_state_summary["init_mode"] = "high_loading"
            init_state_summary["requested_loading_mmol_g"] = float(init_loading)
            prop.set_accepted_quat(hl_states)
        else:
            init_st_now = prop.get_accepted_quat()
            init_state_summary = {
                "init_mode": "auto",
                "init_N_per_box": [int(s["N"]) for s in init_st_now],
                "init_min_gg_A": None,
            }

    warm_st = prop.get_accepted_quat()
    init_state_summary["init_U_ref_K"] = [float(s["U_ref_K"]) for s in warm_st]
    print(
        f"  [P={P_bar:.4g} bar] Init({init_mode}) N={[s['N'] for s in warm_st]} "
        f"({[round(s['N']*conv_mmol_g, 2) for s in warm_st]} mmol/g)"
    )

    def _ckpt_cb(mc_state: dict):
        if ckpt_path is None:
            return
        _write_ckpt_json(ckpt_path, {
            "P_bar": float(P_bar),
            "T_K": float(T_K),
            "delta_eps_kJ_mol": float(de_final),
            "shift_calibration": shift_calib,
            "init_state_summary": init_state_summary,
            "metadata": run_metadata or {},
            "mc_state": mc_state,
        })

    if target is not None:
        target.n_evals = 0
    res = run_nested_mc(
        prop, None,
        k_steps=k_steps, eq_cycles=eq_cycles, prod_cycles=prod_cycles,
        seed=seed + 17, log_every=max(50, (eq_cycles + prod_cycles) // 5),
        batch_target_fn=None if grid_only else target.eval_batch,
        oms_metal_pos=oms_metal_pos if len(oms_metal_pos) > 0 else None,
        oms_cell_rows=oms_cell_rows,
        guest_local_pos=guest_local_pos,
        snapshot_stride=snapshot_stride,
        resume_state=loaded_ckpt["mc_state"] if loaded_ckpt is not None else None,
        checkpoint_fn=_ckpt_cb if ckpt_path is not None else None,
        checkpoint_every=checkpoint_every,
        grid_only=bool(grid_only),
    )

    hist = np.array(res["box_N_history"], dtype=np.float64) * conv_mmol_g
    prod_hist = hist[eq_cycles:]
    if prod_hist.shape[0] >= 2:
        half = prod_hist.shape[0] // 2
        mean_h1 = float(np.mean(prod_hist[:half]))
        mean_h2 = float(np.mean(prod_hist[half:]))
        drift = abs(mean_h2 - mean_h1)
    elif prod_hist.shape[0] == 1:
        mean_h1 = float(np.mean(prod_hist))
        mean_h2 = mean_h1
        drift = 0.0
    else:
        mean_h1 = 0.0
        mean_h2 = 0.0
        drift = 0.0

    res["P_bar"] = float(P_bar)
    res["T_K"] = float(T_K)
    res["fugacity_Pa"] = float(prop.f_Pa)
    res["fw_mass_gmol"] = float(prop.fw_mass_gmol)
    res["first_half_uptake_mmol_g"] = mean_h1
    res["second_half_uptake_mmol_g"] = mean_h2
    res["stationarity_drift_mmol_g"] = drift
    res["mlip_evals"] = 0 if (grid_only or target is None) else int(target.n_evals)
    res["total_inner_steps"] = int(n_boxes * k_steps * (eq_cycles + prod_cycles))
    res["cbmc_k"] = int(cbmc_k)
    res["p_ins_del"] = float(p_ins_del)
    res["prop_grid_wall_s"] = prop_grid_wall_s
    res["shift_calib_wall_s"] = shift_calib_wall_s
    res["shift_calibration"] = shift_calib
    res["init_state_summary"] = init_state_summary
    if run_metadata is not None:
        res["metadata"] = run_metadata

    v2_diag = evaluate_v2_convergence(
        prod_N_history=res["prod_N_history"],
        prod_U_history_kJ=res["prod_U_history"],
        conv_mmol_g=conv_mmol_g,
        T_K=T_K,
        prod_ins_acc=res["prod_ins_acc"],
        prod_del_acc=res["prod_del_acc"],
        prod_ins_att=res["prod_ins_att"],
        prod_del_att=res["prod_del_att"],
        prod_outer_acc_rate=res["prod_outer_acc_rate"],
        uptake_sem_mmol_g=res["uptake_sem_mmol_g"],
        acceptance_breakdown=res.get("acceptance_breakdown"),
        oms_occ_history=res.get("prod_oms_occ_history") or None,
    )
    res["diagnostics"] = v2_diag
    res["legacy_diagnostics"] = evaluate_convergence_diagnostics(res)

    q_str = (
        f"{res['Qst_kJ_mol']:.2f} +- {res['Qst_sem_kJ_mol']:.2f}"
        if res["Qst_kJ_mol"] is not None and res["Qst_sem_kJ_mol"] is not None
        else f"N/A ({res['qst_status']})"
    )
    rhat_n = (v2_diag.get("N_stats") or {}).get("rhat_max")
    r_str = f"{rhat_n:.4f}" if rhat_n is not None else "N/A"
    eb_n = (v2_diag.get("N_stats") or {}).get("ess_bulk", 0.0)
    et_n = (v2_diag.get("N_stats") or {}).get("ess_tail", 0.0)
    print(
        f"  [P={P_bar:.4g} bar DONE] Uptake={res['uptake_mmol_g']:.4f} +- {res['uptake_sem_mmol_g']:.4f} mmol/g | "
        f"Fluc Qst={q_str} | Molar Qst={res['mean_molar_qst_kJ_mol']:.2f} +- {res['mean_molar_qst_sem_kJ_mol']:.2f} kJ/mol | "
        f"Ins={res['prod_ins_acc']}/{res['prod_ins_att']} Del={res['prod_del_acc']}/{res['prod_del_att']} | "
        f"Acc={res['outer_acc_rate']*100:.1f}% | Rhat={r_str} ESS(b/t)={eb_n:.0f}/{et_n:.0f} | "
        f"diag={v2_diag['status']} | {res['elapsed_s']:.1f}s"
    )
    if v2_diag["status"] != "pass":
        for msg in v2_diag.get("reasons", []):
            print(f"  [DIAG {v2_diag['status'].upper()}] P={P_bar:.4g} bar: {msg}")
    prop.close()
    if getattr(target, "device", None) is not None and target.device.type == "cuda":
        target.torch.cuda.empty_cache()
    return res


def main():
    t_start_all = time.perf_counter()
    args = parse_args()
    ff_mode, pol_mode = PROPOSER_MAP[args.proposer]
    use_disp = args.dispersion == "on"
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv) if args.out_csv else None

    resume_mode = str(args.resume_mode)
    if args.resume and resume_mode == "none":
        resume_mode = "skip"

    gpu_ids = parse_gpu_ids(args.gpus) if args.gpus else []
    if len(gpu_ids) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = resolve_worker_cuda_visible_devices(
            gpu_ids[0], os.environ.get("CUDA_VISIBLE_DEVICES")
        )
        args.device = "cuda:0"
        device_id = 0
    else:
        device_id = parse_cuda_device_id(args.device)

    guest = load_guest(name=args.guest, guest_json=args.guest_json, proposer=args.proposer)
    guest_label = Path(args.guest_json).stem if args.guest_json else args.guest.strip().lower()
    if ff_mode == 1 and guest.formula != "CO2":
        raise ValueError(f"--proposer mercado only supports CO2; use --proposer auto_mlip or uff for '{guest.name}'.")
    if ff_mode == 2:
        guest = make_distilled_guest_spec(guest)

    t_init0 = time.perf_counter()
    fw_data = extract_framework_for_proposer(args.cif, charge_mode=args.charge_mode)
    grid_dims = tuple(args.grid_dims) if args.grid_dims is not None else compute_grid_dims(fw_data["uc"], args.grid_spacing)
    init_wall_s = float(time.perf_counter() - t_init0)

    eff_mode = "single"
    if len(gpu_ids) > 1:
        if args.parallel_mode == "auto":
            eff_mode = "pressure" if len(args.pressures) >= 2 else ("replica" if args.boxes >= 2 else "pressure")
        else:
            eff_mode = args.parallel_mode

    distill_cache_path = None
    distill_sha256 = None
    distill_res = None
    if ff_mode == 2:
        model_file_guess = str(args.model) if Path(str(args.model)).is_file() else None
        distill_sha256 = compute_distill_cache_key(
            cif_path=args.cif,
            supercell=tuple(args.supercell),
            guest=guest,
            grid_dims=grid_dims,
            model_name=args.model,
            model_file_path=model_file_guess,
            dispersion=use_disp,
            dispersion_xc=args.xc,
            dtype=args.dtype,
            n_1b=int(args.distill_samples_1b),
            n_2b=int(args.distill_samples_2b),
            distill_seed=int(args.seed) + 50001,
        )
        cand_npz = Path(args.distill_cache_dir) / f"distill_{distill_sha256[:16]}.npz"
        cand_json = cand_npz.with_suffix(".json")
        if cand_npz.exists() and cand_json.exists():
            distill_res = load_distill_cache(str(cand_npz))
            distill_cache_path = str(cand_npz)

    run_metadata = collect_run_metadata(
        args=args, fw_data=fw_data, guest=guest, grid_dims=grid_dims,
        prop_info={"distill_sha256": distill_sha256},
    )

    print(
        f"=== Hybrid GPU-GCMC + MLIP Engine ===\n"
        f"  CIF={args.cif} supercell={args.supercell} T={args.temp} K Pressures={args.pressures}\n"
        f"  Guest={guest.name} ({guest.formula}, {len(guest.sites)} sites) | Grid={grid_dims} | Charges={fw_data['charge_source']}\n"
        f"  Proposer={args.proposer} (ff={ff_mode}, pol={pol_mode}, cbmc_k={args.cbmc_k}, grid_only={bool(args.grid_only)}, init={args.init_mode}) | "
        f"MLIP={args.model} (dtype={args.dtype}, dispersion={use_disp}, xc={args.xc})\n"
        f"  GPUs={gpu_ids if gpu_ids else [device_id]} | ParallelMode={eff_mode} | ResumeMode={resume_mode}"
    )

    payload = {
        "cif": str(args.cif),
        "supercell": list(args.supercell),
        "T_K": float(args.temp),
        "guest": guest_label,
        "charge_mode": args.charge_mode,
        "charge_source": fw_data["charge_source"],
        "proposer": args.proposer,
        "cbmc_k": int(args.cbmc_k),
        "p_ins_del": float(args.p_ins_del),
        "grid_only": bool(args.grid_only),
        "init_mode": args.init_mode,
        "model": args.model,
        "dtype": args.dtype,
        "dispersion": use_disp,
        "dispersion_xc": args.xc,
        "gpus": gpu_ids if gpu_ids else [device_id],
        "parallel_mode": eff_mode,
        "init_wall_s": init_wall_s,
        "metadata": run_metadata,
        "points": [],
    }
    if distill_res is not None:
        payload["distillation"] = distill_res["summary"]

    if resume_mode in ("skip", "extend") and out_json.exists():
        existing = json.loads(out_json.read_text(encoding="utf-8"))
        if "metadata" in existing:
            validate_resume_metadata(
                existing["metadata"], run_metadata, allow_extend_prod=(resume_mode == "extend")
            )
        else:
            mismatches = []
            if not _cif_matches(str(existing.get("cif", "")), str(args.cif)):
                mismatches.append(f"cif: {existing.get('cif')} != {args.cif}")
            if list(existing.get("supercell", [])) != list(args.supercell):
                mismatches.append(f"supercell: {existing.get('supercell')} != {list(args.supercell)}")
            if not math.isclose(float(existing.get("T_K", -1.0)), float(args.temp), rel_tol=1e-6, abs_tol=1e-6):
                mismatches.append(f"T_K: {existing.get('T_K')} != {args.temp}")
            if existing.get("guest", guest_label) != guest_label:
                mismatches.append(f"guest: {existing.get('guest')} != {guest_label}")
            if existing.get("proposer") != args.proposer:
                mismatches.append(f"proposer: {existing.get('proposer')} != {args.proposer}")
            if existing.get("model") != args.model:
                mismatches.append(f"model: {existing.get('model')} != {args.model}")
            if bool(existing.get("dispersion")) != bool(use_disp):
                mismatches.append(f"dispersion: {existing.get('dispersion')} != {use_disp}")
            if existing.get("dispersion_xc") != args.xc:
                mismatches.append(f"dispersion_xc: {existing.get('dispersion_xc')} != {args.xc}")
            if mismatches:
                raise ValueError(
                    f"Cannot resume '{out_json}' due to configuration mismatch: " + "; ".join(mismatches)
                )
        payload["points"] = list(existing.get("points", []))
        print(f"  [RESUME:{resume_mode}] Loaded {len(payload['points'])} existing pressure point(s) from {out_json}")

    pending = []
    for i, p_bar in enumerate(args.pressures):
        match_pts = [
            pt for pt in payload["points"]
            if math.isclose(float(pt["P_bar"]), float(p_bar), rel_tol=1e-6, abs_tol=1e-9)
        ]
        if resume_mode == "skip" and match_pts:
            print(f"  [RESUME] Skipping completed P={p_bar} bar")
            continue
        if resume_mode == "extend" and match_pts and int(match_pts[0].get("prod_cycles", 0)) >= int(args.prod_cycles):
            print(f"  [RESUME] Skipping already >= {args.prod_cycles} prod_cycles at P={p_bar} bar")
            continue
        pending.append((i, p_bar))

    if len(gpu_ids) > 1 and args.parallel_mode == "auto":
        eff_mode = "pressure" if len(pending) >= 2 else ("replica" if args.boxes >= 2 else "pressure")
        payload["parallel_mode"] = eff_mode

    def _upsert_point(pt: dict):
        p_val = float(pt["P_bar"])
        payload["points"] = [
            old for old in payload["points"]
            if not math.isclose(float(old["P_bar"]), p_val, rel_tol=1e-6, abs_tol=1e-9)
        ]
        payload["points"].append(pt)
        payload["total_wall_s"] = float(time.perf_counter() - t_start_all)
        save_outputs(out_json, out_csv, payload)
        print(f"  [Saved output] {out_json} ({len(payload['points'])} points)")

    if pending:
        if len(gpu_ids) > 1:
            if ff_mode == 2 and distill_res is None:
                distill_res = run_distill_on_gpu(
                    gpu_ids[0],
                    {
                        "cif_path": str(args.cif),
                        "fw_data": fw_data,
                        "guest": guest,
                        "supercell": tuple(args.supercell),
                        "grid_dims": grid_dims,
                        "n_1b": int(args.distill_samples_1b),
                        "n_2b": int(args.distill_samples_2b),
                        "distill_seed": int(args.seed) + 50001,
                        "cache_dir": args.distill_cache_dir,
                        "model": args.model,
                        "dtype": args.dtype,
                        "dispersion": use_disp,
                        "dispersion_xc": args.xc,
                        "mace_chunk": args.mace_chunk,
                        "worker_threads": int(args.worker_threads),
                    },
                )
                distill_cache_path = distill_res["cache_npz_path"]
                distill_sha256 = distill_res["distill_sha256"]
                payload["distillation"] = distill_res["summary"]
                run_metadata["distill_sha256"] = distill_sha256
                payload["metadata"] = run_metadata

            run_cfg = {
                "cif_path": str(args.cif),
                "fw_data": fw_data,
                "guest": guest,
                "supercell": tuple(args.supercell),
                "T_K": float(args.temp),
                "ff_mode": ff_mode,
                "pol_mode": pol_mode,
                "grid_dims": grid_dims,
                "n_boxes": int(args.boxes),
                "max_mol": int(args.max_mol),
                "k_steps": int(args.k_steps),
                "cbmc_k": int(args.cbmc_k),
                "p_ins_del": float(args.p_ins_del),
                "grid_only": bool(args.grid_only),
                "distill_cache_path": distill_cache_path,
                "eq_cycles": int(args.eq_cycles),
                "prod_cycles": int(args.prod_cycles),
                "base_seed": int(args.seed),
                "shift_mode": args.shift_mode,
                "pilot_cycles": int(args.pilot_cycles),
                "shift_override": args.shift_override,
                "init_mode": args.init_mode,
                "init_loading": float(args.init_loading),
                "checkpoint_dir": args.checkpoint_dir,
                "checkpoint_every": int(args.checkpoint_every),
                "resume_mode": resume_mode,
                "snapshot_stride": int(args.snapshot_stride),
                "run_metadata": run_metadata,
                "model": args.model,
                "dtype": args.dtype,
                "dispersion": use_disp,
                "dispersion_xc": args.xc,
                "mace_chunk": args.mace_chunk,
                "worker_threads": int(args.worker_threads),
            }

            if eff_mode == "pressure":
                run_pressures_parallel(pending, gpu_ids, run_cfg, on_point_done=_upsert_point)
            else:
                run_replicas_parallel(pending, gpu_ids, run_cfg, on_point_done=_upsert_point)
        else:
            need_target = (not bool(args.grid_only)) or (ff_mode == 2 and distill_res is None)
            target = None
            model_load_wall_s = 0.0
            if need_target:
                t_str = str(max(1, int(args.worker_threads)))
                for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                    os.environ.setdefault(var, t_str)
                t_load0 = time.perf_counter()
                target = FastMaceD3Target(
                    cif_path=args.cif,
                    supercell=tuple(args.supercell),
                    model=args.model,
                    device=args.device,
                    dispersion=use_disp,
                    dispersion_xc=args.xc,
                    guest=guest,
                    mace_chunk=args.mace_chunk,
                    uc_atoms=fw_data["uc"],
                    dtype=args.dtype,
                )
                target.torch.set_num_threads(max(1, int(args.worker_threads)))
                model_load_wall_s = float(time.perf_counter() - t_load0)
            payload["model_load_wall_s"] = model_load_wall_s

            if ff_mode == 2 and distill_res is None:
                distill_res = distill_mlip_surrogate(
                    cif_path=args.cif,
                    fw_data=fw_data,
                    guest=guest,
                    supercell=tuple(args.supercell),
                    target=target,
                    grid_dims=grid_dims,
                    n_1b=int(args.distill_samples_1b),
                    n_2b=int(args.distill_samples_2b),
                    distill_seed=int(args.seed) + 50001,
                    cache_dir=args.distill_cache_dir,
                    model_name=args.model,
                    dispersion=use_disp,
                    dispersion_xc=args.xc,
                    dtype=args.dtype,
                )
                payload["distillation"] = distill_res["summary"]
                distill_sha256 = distill_res["distill_sha256"]

            run_metadata = collect_run_metadata(
                args=args, fw_data=fw_data, guest=guest, grid_dims=grid_dims,
                target=target, prop_info={"distill_sha256": distill_sha256},
            )
            payload["metadata"] = run_metadata

            consistency_checked = False
            for i, p_bar in pending:
                print(f"\n--- Pressure Point {i+1}/{len(args.pressures)}: {p_bar} bar ---")
                t_pt = time.perf_counter()
                pt = run_pressure_point(
                    fw_data=fw_data,
                    guest=guest,
                    supercell=args.supercell,
                    T_K=args.temp,
                    P_bar=p_bar,
                    ff_mode=ff_mode,
                    pol_mode=pol_mode,
                    grid_dims=grid_dims,
                    n_boxes=args.boxes,
                    max_mol=args.max_mol,
                    k_steps=args.k_steps,
                    eq_cycles=args.eq_cycles,
                    prod_cycles=args.prod_cycles,
                    seed=args.seed + i * 100,
                    shift_mode=args.shift_mode,
                    target=None if args.grid_only else target,
                    pilot_cycles=args.pilot_cycles,
                    shift_override=args.shift_override,
                    verify_consistency=(not consistency_checked and target is not None),
                    device_id=device_id,
                    init_mode=args.init_mode,
                    init_loading=args.init_loading,
                    checkpoint_dir=args.checkpoint_dir,
                    checkpoint_every=args.checkpoint_every,
                    resume_mode=resume_mode,
                    snapshot_stride=args.snapshot_stride,
                    run_metadata=run_metadata,
                    grid_only=bool(args.grid_only),
                    cbmc_k=int(args.cbmc_k),
                    p_ins_del=float(args.p_ins_del),
                    distill_res=distill_res,
                )
                pt["point_wall_s"] = float(time.perf_counter() - t_pt)
                pt["model_load_wall_s"] = model_load_wall_s
                consistency_checked = True
                _upsert_point(pt)
    else:
        payload["total_wall_s"] = float(time.perf_counter() - t_start_all)
        save_outputs(out_json, out_csv, payload)
        print("  [RESUME] All requested pressures already completed.")

    payload["total_wall_s"] = float(time.perf_counter() - t_start_all)
    save_outputs(out_json, out_csv, payload)
    print(f"\nTotal Wall Time: {payload['total_wall_s']:.2f} s")
    print(f"Final JSON: {out_json}")
    if out_csv is not None:
        print(f"Final CSV:  {out_csv}")


if __name__ == "__main__":
    main()
