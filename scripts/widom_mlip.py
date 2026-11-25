"""Phase 3 Mechanistic & Zero-Loading Physics Verifier for MOF MLIP Adsorption.

Provides:
1. Independent zero-loading Widom test-particle insertion on FastMaceD3Target:
   - Henry constant K_H (mmol/(g*bar) and mol/(kg*Pa))
   - Excess chemical potential mu_ex (kJ/mol)
   - Zero-coverage isosteric heat of adsorption Q_st(0) (kJ/mol) with bootstrap SEM
2. Quantitative MACE intrinsic vs TorchDFTD3 dispersion decomposition on saved snapshots.
3. Supercell finite-size check (e.g. 1x1x2 vs 1x1x4) on tiled guest configurations.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from direct_mlip_gcmc import KB_SI, compute_pure_fugacity_pr_py, has_hard_sphere_overlap, rand_uniform_quat
from guests import load_guest
from nested_mc_engine import FastMaceD3Target, K_TO_KJMOL, quat_rotate_batch
from qeq_charges import extract_framework_for_proposer


def _min_dist_mask_batch(
    pts_flat: np.ndarray,
    n_real: int,
    fw_pos: np.ndarray,
    cell_rows: np.ndarray,
    inv_cell_rows: np.ndarray,
    r_hs_A: float,
    shifts_27: Optional[np.ndarray],
) -> np.ndarray:
    """Fast vectorized hard-sphere screen returning bool array (n_mol,) True = legal (no clash)."""
    n_mol = len(pts_flat) // n_real
    if r_hs_A <= 0.0:
        return np.ones(n_mol, dtype=bool)
    r2_cut = float(r_hs_A) * float(r_hs_A)
    df = (pts_flat[:, None, :] - fw_pos[None, :, :]) @ inv_cell_rows
    df -= np.round(df)
    if shifts_27 is None:
        dr = df @ cell_rows
        r2_min = np.min(np.sum(dr * dr, axis=-1), axis=1)
    else:
        dr_all = (df[:, :, None, :] + shifts_27[None, None, :, :]) @ cell_rows
        r2_min = np.min(np.sum(dr_all * dr_all, axis=-1), axis=(1, 2))
    mol_min_r2 = np.min(r2_min.reshape(n_mol, n_real), axis=1)
    return mol_min_r2 >= r2_cut


def run_widom_insertion(
    target: FastMaceD3Target,
    guest,
    supercell: Tuple[int, int, int],
    T_K: float,
    n_samples: int = 2000,
    r_hs_A: float = 1.25,
    batch_size: int = 64,
    seed: int = 20263000,
    n_bootstrap: int = 400,
    ref_pressure_bar: float = 0.012,
) -> dict:
    """Sample uniform poses in V x SO(3) and compute exact zero-loading K_H and Q_st(0)."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(int(seed))
    fw = target.fw
    fw_pos = np.asarray(fw.get_positions(), dtype=np.float64)
    cell_rows = np.asarray(fw.get_cell(), dtype=np.float64)
    inv_cell_rows = np.linalg.inv(cell_rows)
    vol_super_A3 = abs(float(np.linalg.det(cell_rows)))
    fw_mass_gmol = float(np.sum(fw.get_masses()))
    perp_widths = 1.0 / np.linalg.norm(inv_cell_rows, axis=0)
    shifts_27 = None
    if r_hs_A > 0.0 and float(np.min(perp_widths)) < 4.0 * float(r_hs_A):
        g1 = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
        shifts_27 = np.stack(np.meshgrid(g1, g1, g1, indexing="ij"), axis=-1).reshape(-1, 3)

    local_pos = guest.real_local_positions()
    n_real = local_pos.shape[0]
    n_tot = int(n_samples)

    frac = rng.random((n_tot, 3))
    centers = frac @ cell_rows
    if guest.is_isotropic:
        quats = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), (n_tot, 1))
    else:
        quats = np.array([rand_uniform_quat(rng) for _ in range(n_tot)], dtype=np.float64)

    offs = quat_rotate_batch(quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3], local_pos)
    pts_flat = (centers[:, None, :] + offs).reshape(n_tot * n_real, 3)

    # Chunk hard-sphere pre-screening so memory stays bounded on large cells
    legal = np.zeros(n_tot, dtype=bool)
    screen_chunk = 512
    for s in range(0, n_tot, screen_chunk):
        e = min(n_tot, s + screen_chunk)
        sub_pts = pts_flat[s * n_real : e * n_real]
        legal[s:e] = _min_dist_mask_batch(sub_pts, n_real, fw_pos, cell_rows, inv_cell_rows, r_hs_A, shifts_27)

    legal_idx = np.nonzero(legal)[0]
    energies_K = np.full(n_tot, np.inf, dtype=np.float64)
    states_legal = [
        {
            "N": 1,
            "cx": np.array([float(centers[idx, 0])]),
            "cy": np.array([float(centers[idx, 1])]),
            "cz": np.array([float(centers[idx, 2])]),
            "qw": np.array([float(quats[idx, 0])]),
            "qx": np.array([float(quats[idx, 1])]),
            "qy": np.array([float(quats[idx, 2])]),
            "qz": np.array([float(quats[idx, 3])]),
        }
        for idx in legal_idx
    ]
    bsize = max(1, int(batch_size))
    for s in range(0, len(states_legal), bsize):
        sub = states_legal[s : s + bsize]
        ev = target.eval_batch(sub)
        for k, (u_k, ov) in enumerate(ev):
            if not ov and math.isfinite(float(u_k)):
                energies_K[legal_idx[s + k]] = float(u_k)

    def _stats_from_u(u_arr: np.ndarray) -> Tuple[float, float, float]:
        finite = np.isfinite(u_arr) & (u_arr < 5.0e5)
        if not np.any(finite):
            return 0.0, float("nan"), float("inf")
        u_f = u_arr[finite]
        logw = -u_f / float(T_K)
        # Guard extreme artificial outliers while preserving valid pocket Boltzmann factors
        mlw = float(np.max(logw))
        w_scaled = np.exp(logw - mlw)
        mean_w = (float(np.sum(w_scaled)) / float(len(u_arr))) * math.exp(mlw)
        # <U * exp(-beta U)> / <exp(-beta U)> computed in shifted log space
        avg_u_K = float(np.sum(u_f * w_scaled) / max(1e-300, float(np.sum(w_scaled))))
        qst_0 = (float(T_K) - avg_u_K) * K_TO_KJMOL
        mu_ex = -float(T_K) * K_TO_KJMOL * (mlw + math.log(max(1e-300, float(np.sum(w_scaled)) / float(len(u_arr)))))
        return mean_w, qst_0, mu_ex

    mean_ros, qst_0_kJ, mu_ex_kJ = _stats_from_u(energies_K)
    prefactor_mmol_g_bar = (1000.0 / fw_mass_gmol) * (1.0e5 * vol_super_A3 * 1.0e-30) / (KB_SI * float(T_K))
    kh_mmol_g_bar = prefactor_mmol_g_bar * mean_ros
    kh_mol_kg_Pa = kh_mmol_g_bar * 1.0e-5

    boot_kh, boot_qst = [], []
    rng_b = np.random.default_rng(int(seed) + 991)
    for _ in range(max(50, int(n_bootstrap))):
        idx_b = rng_b.integers(0, n_tot, size=n_tot)
        mw_b, q_b, _ = _stats_from_u(energies_K[idx_b])
        boot_kh.append(prefactor_mmol_g_bar * mw_b)
        if math.isfinite(q_b):
            boot_qst.append(q_b)

    kh_sem_mmol_g_bar = float(np.std(boot_kh, ddof=1)) if len(boot_kh) > 1 else 0.0
    qst_sem_kJ = float(np.std(boot_qst, ddof=1)) if len(boot_qst) > 1 else 0.0

    f_ref_Pa = compute_pure_fugacity_pr_py(
        float(T_K), float(ref_pressure_bar), float(guest.Tc_K), float(guest.Pc_bar), float(guest.omega)
    )
    henry_uptake_ref = kh_mol_kg_Pa * f_ref_Pa  # 1 mol/kg == 1 mmol/g

    finite_legal = energies_K[np.isfinite(energies_K)]
    wall_s = float(time.perf_counter() - t0)
    return {
        "T_K": float(T_K),
        "supercell": [int(x) for x in supercell],
        "n_samples": n_tot,
        "n_legal_prescreen": int(len(legal_idx)),
        "legal_fraction": float(len(legal_idx) / max(1, n_tot)),
        "r_hs_A": float(r_hs_A),
        "fw_mass_gmol": fw_mass_gmol,
        "volume_super_A3": vol_super_A3,
        "rosenbluth_mean": float(mean_ros),
        "K_H_mmol_g_bar": float(kh_mmol_g_bar),
        "K_H_sem_mmol_g_bar": kh_sem_mmol_g_bar,
        "K_H_mol_kg_Pa": float(kh_mol_kg_Pa),
        "K_H_sem_mol_kg_Pa": kh_sem_mmol_g_bar * 1.0e-5,
        "Qst_0_kJ_mol": float(qst_0_kJ),
        "Qst_0_sem_kJ_mol": qst_sem_kJ,
        "mu_ex_0_kJ_mol": float(mu_ex_kJ),
        "min_u_int_kJ_mol": float(np.min(finite_legal) * K_TO_KJMOL) if len(finite_legal) else None,
        "ref_pressure_bar": float(ref_pressure_bar),
        "ref_fugacity_Pa": float(f_ref_Pa),
        "henry_uptake_at_ref_P_mmol_g": float(henry_uptake_ref),
        "henry_uptake_sem_mmol_g": float(kh_sem_mmol_g_bar * 1.0e-5 * f_ref_Pa),
        "wall_s": wall_s,
    }


def _extract_states_from_run_json(json_path: str) -> List[Tuple[float, List[dict]]]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    out = []
    for pt in data.get("points", []):
        p_bar = float(pt.get("P_bar", 0.0))
        states = list(pt.get("final_states", []))
        if states:
            out.append((p_bar, states))
    return out


def decompose_mace_vs_d3(target: FastMaceD3Target, states_by_p: List[Tuple[float, List[dict]]]) -> List[dict]:
    """Break down total interaction energy into intrinsic MACE and TorchDFTD3 components."""
    rows = []
    for p_bar, states in states_by_p:
        per_box = []
        for b, st in enumerate(states):
            n = int(st.get("N", 0))
            if n <= 0:
                continue
            comp = target.eval_components(st)
            u_tot_kJ = float(comp["u_int_K"]) * K_TO_KJMOL
            u_mace_kJ = float(comp["u_mace_int_K"]) * K_TO_KJMOL
            u_d3_kJ = float(comp["u_d3_int_K"]) * K_TO_KJMOL
            per_box.append({
                "box": b,
                "N": n,
                "u_tot_per_mol_kJ_mol": u_tot_kJ / n,
                "u_mace_per_mol_kJ_mol": u_mace_kJ / n,
                "u_d3_per_mol_kJ_mol": u_d3_kJ / n,
            })
        if not per_box:
            continue
        rows.append({
            "P_bar": p_bar,
            "n_boxes_nonempty": len(per_box),
            "mean_u_tot_per_mol_kJ_mol": float(np.mean([r["u_tot_per_mol_kJ_mol"] for r in per_box])),
            "mean_u_mace_per_mol_kJ_mol": float(np.mean([r["u_mace_per_mol_kJ_mol"] for r in per_box])),
            "mean_u_d3_per_mol_kJ_mol": float(np.mean([r["u_d3_per_mol_kJ_mol"] for r in per_box])),
            "per_box": per_box,
        })
    return rows


def check_supercell_finite_size(
    cif_path: str,
    guest,
    uc_atoms,
    base_supercell: Tuple[int, int, int],
    large_supercell: Tuple[int, int, int],
    target_base: FastMaceD3Target,
    states_by_p: List[Tuple[float, List[dict]]],
    model: str,
    dispersion: bool,
    d3_xc: str,
    dtype: str,
    device: str,
    mace_chunk: Optional[int],
) -> List[dict]:
    """Compare intensive adsorption energy U/N between base and tiled larger supercell."""
    reps = tuple(int(l) // int(b) for b, l in zip(base_supercell, large_supercell))
    if any(r * b != l or r < 1 for r, b, l in zip(reps, base_supercell, large_supercell)):
        raise ValueError(f"large_supercell {large_supercell} must be an integer multiple of base {base_supercell}")

    H_base = np.asarray(uc_atoms.repeat(tuple(base_supercell)).get_cell(), dtype=np.float64)
    shifts = []
    for ia in range(reps[0]):
        for ib in range(reps[1]):
            for ic in range(reps[2]):
                shifts.append(np.array([ia, ib, ic], dtype=np.float64) @ H_base)
    shifts = np.asarray(shifts, dtype=np.float64)

    target_large = FastMaceD3Target(
        cif_path=cif_path,
        supercell=tuple(large_supercell),
        model=model,
        device=device,
        dispersion=bool(dispersion),
        dispersion_xc=d3_xc,
        guest=guest,
        mace_chunk=mace_chunk,
        uc_atoms=uc_atoms,
        dtype=dtype,
    )

    out = []
    for p_bar, states in states_by_p:
        diffs_kJ = []
        base_vals, large_vals = [], []
        for st in states[:2]:
            n = int(st.get("N", 0))
            if n <= 0:
                continue
            u_base_K, _ = target_base(st)
            u_base_mol_kJ = float(u_base_K) * K_TO_KJMOL / n

            cx, cy, cz, qw, qx, qy, qz = [], [], [], [], [], [], []
            for sh in shifts:
                cx.append(np.asarray(st["cx"], dtype=np.float64) + sh[0])
                cy.append(np.asarray(st["cy"], dtype=np.float64) + sh[1])
                cz.append(np.asarray(st["cz"], dtype=np.float64) + sh[2])
                qw.append(np.asarray(st["qw"], dtype=np.float64))
                qx.append(np.asarray(st["qx"], dtype=np.float64))
                qy.append(np.asarray(st["qy"], dtype=np.float64))
                qz.append(np.asarray(st["qz"], dtype=np.float64))
            st_large = {
                "N": n * len(shifts),
                "cx": np.concatenate(cx),
                "cy": np.concatenate(cy),
                "cz": np.concatenate(cz),
                "qw": np.concatenate(qw),
                "qx": np.concatenate(qx),
                "qy": np.concatenate(qy),
                "qz": np.concatenate(qz),
            }
            u_large_K, _ = target_large(st_large)
            u_large_mol_kJ = float(u_large_K) * K_TO_KJMOL / st_large["N"]
            base_vals.append(u_base_mol_kJ)
            large_vals.append(u_large_mol_kJ)
            diffs_kJ.append(u_large_mol_kJ - u_base_mol_kJ)
        if diffs_kJ:
            out.append({
                "P_bar": p_bar,
                "base_supercell": list(base_supercell),
                "large_supercell": list(large_supercell),
                "mean_base_kJ_mol": float(np.mean(base_vals)),
                "mean_large_kJ_mol": float(np.mean(large_vals)),
                "mean_diff_kJ_mol": float(np.mean(diffs_kJ)),
                "max_abs_diff_kJ_mol": float(np.max(np.abs(diffs_kJ))),
            })
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Widom Henry/Qst0 + D3 Decomposition + Supercell Finite-Size Verifier")
    p.add_argument("--cif", required=True)
    p.add_argument("--supercell", nargs=3, type=int, default=[1, 1, 2])
    p.add_argument("--guest", default="co2")
    p.add_argument("--temp", type=float, default=298.0)
    p.add_argument("--model", default="mace-matpes-r2scan-0")
    p.add_argument("--dispersion", choices=["on", "off"], default="on")
    p.add_argument("--xc", default="pbe")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--device", default="cuda")
    p.add_argument("--mace-chunk", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--r-hs", type=float, default=1.25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--ref-p-bar", type=float, default=0.012)
    p.add_argument("--seed", type=int, default=20263000)
    p.add_argument("--decompose-json", default="", help="Optional run_mlip_gcmc JSON for MACE vs D3 snapshot breakdown")
    p.add_argument("--finite-size-supercell", nargs=3, type=int, default=None, help="Larger supercell e.g. 1 1 4 for U/N check")
    p.add_argument("--out-json", default="build/widom_report.json")
    return p.parse_args()


def main():
    args = parse_args()
    use_disp = args.dispersion == "on"
    guest = load_guest(args.guest)
    fw_data = extract_framework_for_proposer(args.cif, charge_mode="auto")
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

    report: Dict[str, object] = {
        "cif": str(args.cif),
        "supercell": [int(x) for x in args.supercell],
        "guest": guest.name,
        "T_K": float(args.temp),
        "model": args.model,
        "dispersion": use_disp,
        "dispersion_xc": args.xc,
        "dtype": args.dtype,
    }

    if args.n_samples > 0:
        wid = run_widom_insertion(
            target=target,
            guest=guest,
            supercell=tuple(args.supercell),
            T_K=args.temp,
            n_samples=args.n_samples,
            r_hs_A=args.r_hs,
            batch_size=args.batch_size,
            seed=args.seed,
            ref_pressure_bar=args.ref_p_bar,
        )
        report["widom"] = wid
        print(
            f"[Widom DONE] K_H = {wid['K_H_mmol_g_bar']:.4f} +- {wid['K_H_sem_mmol_g_bar']:.4f} mmol/(g*bar) "
            f"({wid['K_H_mol_kg_Pa']:.3e} mol/(kg*Pa)) | Qst(0) = {wid['Qst_0_kJ_mol']:.2f} +- {wid['Qst_0_sem_kJ_mol']:.2f} kJ/mol | "
            f"q_Henry({args.ref_p_bar} bar) = {wid['henry_uptake_at_ref_P_mmol_g']:.4f} mmol/g | {wid['wall_s']:.1f}s"
        )

    if args.decompose_json:
        states_by_p = _extract_states_from_run_json(args.decompose_json)
        decomp = decompose_mace_vs_d3(target, states_by_p)
        report["mace_d3_decomposition"] = decomp
        if args.finite_size_supercell is not None:
            report["finite_size_check"] = check_supercell_finite_size(
                cif_path=args.cif,
                guest=guest,
                uc_atoms=fw_data["uc"],
                base_supercell=tuple(args.supercell),
                large_supercell=tuple(args.finite_size_supercell),
                target_base=target,
                states_by_p=states_by_p,
                model=args.model,
                dispersion=use_disp,
                d3_xc=args.xc,
                dtype=args.dtype,
                device=args.device,
                mace_chunk=args.mace_chunk,
            )

    out_p = Path(args.out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[Saved Widom/Physics report] {out_p}")


if __name__ == "__main__":
    main()
